import os, time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from .config import VAPORConfig

def psi_mutual_independence_loss(Psi_list, alpha=1.0, beta=1.0):
    """Compute mutual independence loss for Psi matrices."""
    n = len(Psi_list)
    loss_size = 0.0
    for psi in Psi_list:
        loss_size += torch.norm(psi, p='fro')**2
    loss_size = loss_size / n
    
    loss_inter = 0.0
    num_pairs = n * (n - 1) / 2
    for i in range(n):
        for j in range(i+1, n):
            inter = Psi_list[i].T @ Psi_list[j]
            loss_inter += torch.norm(inter, p='fro')**2
    loss_inter = loss_inter / num_pairs
    return alpha * loss_size + beta * loss_inter

class MinimalFlowSupervision:
    """Minimal flow supervision for optional start/terminal markers."""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def start_activity_loss(self, v_start: torch.Tensor, target_speed: float = 0.15) -> torch.Tensor:
        """Start points should have moderate speed."""
        if v_start.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        
        start_speeds = torch.norm(v_start, dim=1)
        speed_loss = F.mse_loss(start_speeds, 
                               torch.full_like(start_speeds, target_speed))
        return speed_loss
    
    def terminal_stability_loss(self, v_term: torch.Tensor) -> torch.Tensor:
        """Terminal points should have low velocity."""
        if v_term.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        
        stability_loss = torch.norm(v_term, dim=1).mean()
        return stability_loss
    
    def weak_influence_loss(self, z: torch.Tensor, v: torch.Tensor,
                           is_start: torch.BoolTensor, is_term: torch.BoolTensor,
                           influence_radius: float = None) -> torch.Tensor:
        """Weak repulsion-attraction loss for directional guidance."""
        if not (is_start.any() or is_term.any()):
            return torch.tensor(0.0, device=self.device)
        
        if influence_radius is None:
            all_dists = torch.cdist(z, z)
            influence_radius = torch.median(all_dists[all_dists > 0]).item()
        
        loss = 0.0
        count = 0
        
        for i in range(z.size(0)):
            if is_start[i] or is_term[i]:
                continue
            
            current_pos = z[i]
            current_vel = v[i]
            
            if torch.norm(current_vel) <= 1e-6:
                continue
            
            vel_direction = F.normalize(current_vel, dim=0)
            
            # Start repulsion
            if is_start.any():
                start_positions = z[is_start]
                dist_to_starts = torch.norm(start_positions - current_pos.unsqueeze(0), dim=1)
                min_start_dist, closest_start_idx = torch.min(dist_to_starts, dim=0)
                
                if min_start_dist < influence_radius:
                    closest_start = start_positions[closest_start_idx]
                    repulsion_direction = F.normalize(current_pos - closest_start, dim=0)
                    similarity = torch.dot(repulsion_direction, vel_direction)
                    weight = 1.0 - (min_start_dist / influence_radius)
                    loss += F.relu(0.1 - similarity) * weight * 0.5
                    count += 1
            
            # Terminal attraction
            if is_term.any():
                term_positions = z[is_term]
                dist_to_terms = torch.norm(term_positions - current_pos.unsqueeze(0), dim=1)
                min_term_dist, closest_term_idx = torch.min(dist_to_terms, dim=0)
                
                if min_term_dist < influence_radius:
                    closest_term = term_positions[closest_term_idx]
                    attraction_direction = F.normalize(closest_term - current_pos, dim=0)
                    similarity = torch.dot(attraction_direction, vel_direction)
                    weight = 1.0 - (min_term_dist / influence_radius)
                    loss += F.relu(0.1 - similarity) * weight * 0.5
                    count += 1
        
        return loss / count if count > 0 else torch.tensor(0.0, device=self.device)
    
    def trajectory_progression_loss(self, z_traj: torch.Tensor, 
                                  is_start: torch.BoolTensor, is_term: torch.BoolTensor) -> torch.Tensor:
        """Start trajectories should show progression."""
        if not is_start.any() or z_traj.size(0) < 2:
            return torch.tensor(0.0, device=self.device)
        
        z_initial = z_traj[0]
        z_final = z_traj[-1]
        
        start_initial = z_initial[is_start]
        start_final = z_final[is_start]
        
        if start_initial.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        
        loss = 0.0
        
        # Start points should have displacement
        displacements = torch.norm(start_final - start_initial, dim=1)
        min_displacement = 0.05
        stagnation_loss = F.relu(min_displacement - displacements).mean()
        loss += stagnation_loss
        
        # Progress toward terminals if available
        if is_term.any():
            term_positions = z_initial[is_term]
            
            progress_loss = 0.0
            for i in range(start_initial.size(0)):
                initial_dists = torch.norm(start_initial[i].unsqueeze(0) - term_positions, dim=1)
                final_dists = torch.norm(start_final[i].unsqueeze(0) - term_positions, dim=1)
                
                best_progress = torch.max(initial_dists - final_dists)
                progress_loss += F.relu(-best_progress)
            
            loss += progress_loss / start_initial.size(0) * 0.5
        
        return loss
    
    def compute_supervision_loss(self, model, z0: torch.Tensor, z_traj: torch.Tensor,
                               is_start: torch.BoolTensor = None, is_term: torch.BoolTensor = None,
                               weights: dict = None) -> tuple:
        """Compute minimal supervision loss."""
        if weights is None:
            weights = {
                'start_activity': 0.1,
                'terminal_stability': 0.1,
                'weak_influence': 0.3,
                'trajectory_progress': 0.5,
            }
        
        v0 = model.compute_velocities(z0)
        loss_components = {}
        total_loss = torch.tensor(0.0, device=self.device)
        
        has_start = is_start is not None and is_start.any()
        has_term = is_term is not None and is_term.any()
        
        if not (has_start or has_term):
            # No supervision signals
            loss_components = {
                'start_activity': torch.tensor(0.0, device=self.device),
                'terminal_stability': torch.tensor(0.0, device=self.device),
                'weak_influence': torch.tensor(0.0, device=self.device),
                'trajectory_progress': torch.tensor(0.0, device=self.device)
            }
            return total_loss, loss_components
        
        # Start activity loss
        if has_start:
            start_loss = self.start_activity_loss(v0[is_start])
            loss_components['start_activity'] = start_loss
            total_loss += weights.get('start_activity', 1.0) * start_loss
        else:
            loss_components['start_activity'] = torch.tensor(0.0, device=self.device)
        
        # Terminal stability loss
        if has_term:
            term_loss = self.terminal_stability_loss(v0[is_term])
            loss_components['terminal_stability'] = term_loss
            total_loss += weights.get('terminal_stability', 1.0) * term_loss
        else:
            loss_components['terminal_stability'] = torch.tensor(0.0, device=self.device)
        
        # Weak influence loss
        if has_start or has_term:
            start_mask = is_start if has_start else torch.zeros(z0.size(0), dtype=torch.bool, device=self.device)
            term_mask = is_term if has_term else torch.zeros(z0.size(0), dtype=torch.bool, device=self.device)
            
            influence_loss = self.weak_influence_loss(z0, v0, start_mask, term_mask)
            loss_components['weak_influence'] = influence_loss
            total_loss += weights.get('weak_influence', 0.5) * influence_loss
        else:
            loss_components['weak_influence'] = torch.tensor(0.0, device=self.device)
        
        # Trajectory progression loss
        if has_start:
            start_mask = is_start if has_start else torch.zeros(z0.size(0), dtype=torch.bool, device=self.device)
            term_mask = is_term if has_term else torch.zeros(z0.size(0), dtype=torch.bool, device=self.device)
            
            progress_loss = self.trajectory_progression_loss(z_traj, start_mask, term_mask)
            loss_components['trajectory_progress'] = progress_loss
            total_loss += weights.get('trajectory_progress', 0.8) * progress_loss
        else:
            loss_components['trajectory_progress'] = torch.tensor(0.0, device=self.device)
        
        return total_loss, loss_components

@torch.no_grad()
def _evaluate_vae_on_loader(model, loader, device="cuda"):
    model.eval()
    total_mse_mean = 0.0 
    total_kld_mean = 0.0
    total_n = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch.get("x")
        else:
            x = batch
        if x is None:
            continue

        x = x.to(device)
        recon, z0, mu, logvar = model.encode(x)

        mse_mean = F.mse_loss(recon, x)  # mean over elements
        kld_mean = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        bs = x.size(0)
        total_mse_mean += mse_mean.item() * bs
        total_kld_mean += kld_mean.item() * bs
        total_n += bs

    if total_n == 0:
        return np.nan, np.nan
    return total_mse_mean / total_n, total_kld_mean / total_n


# ============== Optional ==============
def _save_epoch_csv_and_plots(history, save_dir: Path, exp_name: str = "run"):
    """
    history: dict with lists
      - 'epoch', 'time', 'train_mse','train_kld','train_traj','train_prior','train_psi',
        'test_mse','test_kld'
    """
    save_dir = Path(save_dir)
    (save_dir / "csv").mkdir(parents=True, exist_ok=True)
    (save_dir / "plots").mkdir(parents=True, exist_ok=True)

    # --- CSV ---
    df = pd.DataFrame({
        "epoch": history["epoch"],
        "time_per_epoch": history["time"],
        "train_mse": history["train_mse"],
        "train_kld": history["train_kld"],
        "train_traj": history["train_traj"],
        "train_prior": history["train_prior"],
        "train_psi": history["train_psi"],
        "test_mse": history["test_mse"],
        "test_kld": history["test_kld"],
    })
    csv_path = save_dir / "csv" / f"{exp_name}_metrics.csv"
    df.to_csv(csv_path, index=False)

    # --- Plot VAE (train + test) ---
    epochs = history["epoch"]
    plt.figure(figsize=(9,5))
    if history["train_mse"]:
        plt.plot(epochs, history["train_mse"], label="Train Recon", linewidth=2)
    if history["train_kld"]:
        plt.plot(epochs, history["train_kld"], label="Train KL", linewidth=2)
    if any(v is not None for v in history["test_mse"]):
        plt.plot(epochs, history["test_mse"], label="Test Recon", linewidth=2, linestyle="--")
    if any(v is not None for v in history["test_kld"]):
        plt.plot(epochs, history["test_kld"], label="Test KL", linewidth=2, linestyle="--")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("VAE Losses (Train & Test)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / "plots" / f"{exp_name}_vae_losses.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Plot Transport（train only） ---
    plt.figure(figsize=(9,5))
    if history["train_traj"]:
        plt.plot(epochs, history["train_traj"], label="Trajectory", linewidth=2)
    if history["train_prior"]:
        plt.plot(epochs, history["train_prior"], label="Prior", linewidth=2)
    if history["train_psi"]:
        plt.plot(epochs, history["train_psi"], label="Psi", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Transport Losses (Train only)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / "plots" / f"{exp_name}_transport_losses.png", dpi=300, bbox_inches="tight")
    plt.close()

def train_model(
    model: 'VAPOR',
    dataset: 'AnnDataDataset',
    config: Optional[Union[VAPORConfig, Dict[str, Any]]] = None,
    split_train_test: bool = True,
    test_size: float = 0.2,
    eval_each_epoch: bool = True,          
    save_dir: Optional[Union[str, Path]] = None,  
    exp_name: str = "run",
    **kwargs
) -> 'VAPOR':
    
    # ---------- config ----------
    if config is None:
        config = VAPORConfig()
    elif isinstance(config, dict):
        config = VAPORConfig.from_dict(config)
    elif not isinstance(config, VAPORConfig):
        raise ValueError("config must be VAPORConfig object, dict, or None")

    if kwargs:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}' ignored")

    print(f"Training on device: {config.device}")
    print(f"Training for {config.epochs} epochs with batch size {config.batch_size}")

    model.to(config.device)

    # ---------- optimizer ----------
    vae_params = list(model.vae.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("vae.")]
    opt_vae = torch.optim.Adam(vae_params, lr=config.vae_lr_factor * config.lr)
    opt_to  = torch.optim.Adam(other_params, lr=config.lr)

    # ---------- DataLoader ----------
    if split_train_test:
        n = len(dataset)
        n_test = max(1, int(round(n * test_size)))
        n_train = n - n_test
        g = torch.Generator().manual_seed(42)
        train_subset, test_subset = torch.utils.data.random_split(dataset, [n_train, n_test], generator=g)
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        test_loader  = DataLoader(test_subset,  batch_size=config.batch_size, shuffle=False)
        print(f"Data split: train={n_train}, test={n_test} (test_size={test_size})")
    else:
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        test_loader  = None

    # ---------- params ----------
    history = {
        "epoch": [],
        "time": [],
        "train_mse": [],
        "train_kld": [],
        "train_traj": [],
        "train_prior": [],
        "train_psi": [],
        "test_mse": [],
        "test_kld": [],
    }

    print("\nStarting training...")
    print("-" * 80)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        model.train()

        epoch_metrics = dict(mse=0.0, kld=0.0, traj=0.0, prior=0.0, psi=0.0, vae=0.0, to=0.0)
        batch_count = 0

        horizons = torch.randperm(config.t_max, device=config.device).add_(1).tolist()

        for batch_idx, (x, t_data, is_root, is_term) in enumerate(train_loader):
            batch_count += 1
            x, t_data = x.to(config.device), t_data.to(config.device)
            is_root, is_term = is_root.to(config.device), is_term.to(config.device)

            # ---- VAE step ----
            t0 = time.time()
            recon, z0, mu, logvar = model.encode(x)
            recon_loss = F.mse_loss(recon, x)  # mean
            kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss   = recon_loss + config.beta * kl_loss

            opt_vae.zero_grad()
            vae_loss.backward()
            opt_vae.step()
            t1 = time.time()

            epoch_metrics['vae'] += (t1 - t0)
            epoch_metrics['mse'] += recon_loss.item()
            epoch_metrics['kld'] += kl_loss.item()

            # ---- Transport step ----
            t2 = time.time()
            t_rand = horizons[batch_idx % config.t_max]
            t_span = torch.linspace(0, t_rand, steps=t_rand + 1, device=config.device)

            z_traj = model.integrate(z0, t_span)
            z0_detached = z0.detach()

            eps = torch.median(torch.cdist(z0_detached, z0_detached).topk(30, 1, False).values[:, -1]).item()
            traj_loss, paths, adj_idx, adj_mask = model.directed_graph_tcl_loss(z0_detached, z_traj, eps)

            v0 = model.compute_velocities(z0_detached)
            prior_loss = model.flag_direction_loss_graph(z0_detached, v0, is_root, is_term, adj_idx, adj_mask)
            psi_loss   = psi_mutual_independence_loss(model.transport_op.Psi, alpha=config.eta_a, beta=1.0-config.eta_a)

            to_loss = (config.alpha * traj_loss + config.gamma * prior_loss + config.eta * psi_loss)

            opt_to.zero_grad()
            to_loss.backward()
            opt_to.step()
            t3 = time.time()

            epoch_metrics['to']    += (t3 - t2)
            epoch_metrics['traj']  += traj_loss.item()
            epoch_metrics['prior'] += prior_loss.item()
            epoch_metrics['psi']   += psi_loss.item()

        if batch_count > 0:
            for k in ['mse', 'kld', 'traj', 'prior', 'psi']:
                epoch_metrics[k] /= batch_count

        epoch_time = time.time() - epoch_start
        history["epoch"].append(epoch)
        history["time"].append(epoch_time)
        history["train_mse"].append(epoch_metrics['mse'])
        history["train_kld"].append(epoch_metrics['kld'])
        history["train_traj"].append(epoch_metrics['traj'])
        history["train_prior"].append(epoch_metrics['prior'])
        history["train_psi"].append(epoch_metrics['psi'])

        if epoch % config.print_freq == 0:
            print(f"Epoch {epoch:3d}/{config.epochs} | "
                  f"Time: {epoch_time:5.2f}s | "
                  f"Recon: {epoch_metrics['mse']:.4f} | "
                  f"KL: {epoch_metrics['kld']:.4f} | "
                  f"Traj: {epoch_metrics['traj']:.4f} | "
                  f"Prior: {epoch_metrics['prior']:.4f} | "
                  f"Psi: {epoch_metrics['psi']:.4f}")

        if epoch % max(1, config.epochs // 10) == 0:
            model.transport_op.sort_and_prune_psi()
            norms = [psi.pow(2).mean().sqrt().item() for psi in model.transport_op.Psi]
            print(f"Psi norms: {[f'{n:.4f}' for n in norms]}")

        if split_train_test and eval_each_epoch:
            test_mse, test_kld = _evaluate_vae_on_loader(model, test_loader, device=config.device)
            history["test_mse"].append(float(test_mse) if not np.isnan(test_mse) else None)
            history["test_kld"].append(float(test_kld) if not np.isnan(test_kld) else None)
        else:
            history["test_mse"].append(None)
            history["test_kld"].append(None)

        if save_dir is not None:
            _save_epoch_csv_and_plots(history, Path(save_dir), exp_name=exp_name)

    print("-" * 80)
    print("Training completed!")

    if split_train_test:
        if save_dir is None:
            _save_epoch_csv_and_plots(history, Path("."), exp_name=exp_name)
    else:
        if config.plot_losses:
            _plot_training_metrics({
                'mse_losses': history['train_mse'],
                'kld_losses': history['train_kld'],
                'traj_losses': history['train_traj'],
                'prior_losses': history['train_prior'],
                'psi_losses': history['train_psi'],
            })

    return model

def _plot_training_metrics(metrics: Dict[str, List[float]], has_supervision: bool = False):
    """Plot training loss curves in separate clean axes."""
    epochs = list(range(1, len(metrics['mse_losses']) + 1))
    
    # Adjust subplot layout based on supervision
    if has_supervision:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # VAE losses
    axes[0].plot(epochs, metrics['mse_losses'], label='Reconstruction')
    axes[0].plot(epochs, metrics['kld_losses'], label='KL Divergence')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('VAE Losses')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Transport losses
    axes[1].plot(epochs, metrics['traj_losses'], label='Trajectory')
    axes[1].plot(epochs, metrics['prior_losses'], label='Prior')
    axes[1].plot(epochs, metrics['psi_losses'], label='Psi')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Transport Losses')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Total losses
    total_vae = [mse + kld for mse, kld in zip(metrics['mse_losses'], metrics['kld_losses'])]
    total_transport = [traj + prior + psi for traj, prior, psi in 
                      zip(metrics['traj_losses'], metrics['prior_losses'], metrics['psi_losses'])]
    axes[2].plot(epochs, total_vae, label='Total VAE')
    axes[2].plot(epochs, total_transport, label='Total Transport')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Total Losses')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Supervision losses (only if supervision is used)
    if has_supervision and 'supervision_losses' in metrics:
        axes[3].plot(epochs, metrics['supervision_losses'], label='Supervision', color='purple')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Loss')
        axes[3].set_title('Supervision Loss')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()