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

def psi_structure_loss(
    Psi_list,
    w_scale: float = 0.2,
    w_orth: float = 1.0,
    spec_cap: float = None,
    eps: float = 1e-8
):
    
    if isinstance(Psi_list, (list, tuple)):
        Psi = torch.stack(list(Psi_list), dim=0)
    else:
        Psi = torch.stack(list(Psi_list), dim=0)
    M, d, _ = Psi.shape

    fro2 = (Psi ** 2).sum(dim=(1, 2))                      # (M,)
    scale_loss = ((fro2 / d - 1.0) ** 2).mean()

    fro = fro2.sqrt().clamp_min(eps).view(M, 1, 1)
    Psi_hat = Psi / fro                                     # (M,d,d)
    # G_ij = <Psi_i, Psi_j> = tr(Psi_i^T Psi_j)
    G = torch.einsum('mij,nij->mn', Psi_hat, Psi_hat)       # (M,M)
    orth_loss = (G - torch.eye(M, device=Psi.device)).pow(2).sum() / (M*M - M + eps)

    spec_loss = Psi.new_tensor(0.0)
    if spec_cap is not None:
        iters = 5
        v = torch.randn(M, d, 1, device=Psi.device)
        v = v / (v.norm(dim=1, keepdim=True) + eps)
        for _ in range(iters):
            v = Psi.transpose(1, 2) @ (Psi @ v)
            v = v / (v.norm(dim=1, keepdim=True) + eps)
        s_max = (Psi @ v).norm(dim=1).squeeze(-1)           # (M,)
        spec_violation = (s_max - spec_cap).clamp_min(0.0)
        spec_loss = (spec_violation ** 2).mean()

    total = (w_scale * scale_loss +
             w_orth  * orth_loss  +
             spec_loss)

    return total, {
        'scale': scale_loss.detach(),
        'orth':  orth_loss.detach(),
        'spec':  spec_loss.detach(),
        'fro2_mean': fro2.mean().detach(),
    }

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
    verbose: bool = True,
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

    vae_scale = model.vae.latent_dim / model.vae.input_dim
    opt_vae = torch.optim.Adam(vae_params, lr= vae_scale * config.lr)
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

        for batch_idx, batch in enumerate(train_loader):
            batch_count += 1
            
            if len(batch) == 5:
                x, t_data, is_root, is_term, coords = batch
            else:
                x, t_data, is_root, is_term = batch
                coords = None

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

            B = z0_detached.size(0)
            k_nn = min(config.eps_z_k, B - 1)
            d = torch.cdist(z0_detached, z0_detached)
            eps_z = torch.median(d.topk(k_nn, dim=1, largest=False).values[:, -1]).item()

            traj_loss, paths, adj_idx, adj_mask = model.directed_graph_tcl_loss(
                z0_detached, z_traj, eps_z,
                min_samples=config.min_samples,
                k=config.graph_k,
                threshold=0.0,  # cos_threshold (config?)
                coords=coords,
                eps_xy=config.eps_xy,  # None => latent-only
                coords_mean=getattr(dataset, "spatial_mean", None),
                coords_std=getattr(dataset, "spatial_std", None),
                zscore_mode=config.zscore_mode,
            )
            # eps = torch.median(torch.cdist(z0_detached, z0_detached).topk(30, 1, False).values[:, -1]).item()
            # traj_loss, paths, adj_idx, adj_mask = model.directed_graph_tcl_loss(z0_detached, z_traj, eps)

            v0 = model.compute_velocities(z0_detached)
            prior_loss = model.flag_direction_loss_graph(z0_detached, v0,
                                                        is_root, is_term,
                                                        adj_idx, adj_mask)
            # psi_loss   = psi_mutual_independence_loss(model.transport_op.Psi, alpha=config.eta_a, beta=1.0-config.eta_a)
            psi_loss, _ = psi_structure_loss(
                model.transport_op.Psi,
                w_scale = 0.2, #config.psi_w_scale,     # 0.1~1.0
                w_orth  = config.eta, #config.psi_w_orth,      # 0.1~1.0
                spec_cap= 0.0 #config.psi_spec_cap
                )

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

        if verbose:
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
            if verbose:
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
    _plot_training_metrics({
                'mse_losses': history['train_mse'],
                'kld_losses': history['train_kld'],
                'traj_losses': history['train_traj'],
                'prior_losses': history['train_prior'],
                'psi_losses': history['train_psi'],
            })

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
    
    plt.tight_layout()
    plt.show()