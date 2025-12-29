import os, time
import math
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
# from .dataset import GroupedBatchSampler
from .utils import get_base_dataset, resolve_device

@torch.no_grad()
def set_optimizer_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)

@torch.no_grad()
def get_optimizer_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])

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

class _WithIndex(torch.utils.data.Dataset):
    """Wrap a Dataset or Subset so __getitem__ returns (..., idx_global)."""
    def __init__(self, base):
        self.base = base
        # if base is a Subset, it has .indices; store for global mapping
        self._subset_indices = getattr(base, "indices", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        # map to original global index if possible
        if self._subset_indices is not None:
            idx_global = int(self._subset_indices[i])
        else:
            idx_global = int(i)
        if isinstance(item, tuple):
            return (*item, idx_global)
        return (item, idx_global)

@torch.no_grad()
def _encode_all_z(model: 'VAPOR',
                dataset, 
                device, 
                batch_size: int = 1024, 
                use_mu: bool = False):
    """Encode the whole dataset to latent z (or mu) in the ORIGINAL dataset index space.
    Returns z_all: (N,D) on CPU float32.
    """
    model.eval()
    loader = DataLoader(_WithIndex(dataset), batch_size=batch_size, shuffle=False)
    # infer N
    N = len(dataset)
    z_list = []
    idx_list = []
    for batch in loader:
        x = batch[0].to(device)
        idx_global = torch.as_tensor(batch[-1], device=device, dtype=torch.long)
        recon, z, mu, logvar = model.encode(x)
        z_use = mu if use_mu else z
        z_list.append(z_use.detach().float().cpu())
        idx_list.append(idx_global.detach().cpu())
    z_cat = torch.cat(z_list, dim=0)
    idx_cat = torch.cat(idx_list, dim=0)
    # scatter into full array in correct order
    D = z_cat.size(1)
    z_all = torch.empty((N, D), dtype=torch.float32)
    z_all[idx_cat] = z_cat
    return z_all

@torch.no_grad()
def _build_global_knn_graph_sklearn(z_all_cpu: torch.Tensor, K: int = 50, metric: str = "euclidean", n_jobs: int = -1):
    """Build global kNN graph on CPU using sklearn. Returns (nbr_idx_cpu, nbr_dist_cpu) as torch tensors."""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    z_np = z_all_cpu.numpy()
    nn = NearestNeighbors(n_neighbors=K+1, metric=metric, n_jobs=n_jobs, algorithm="auto")
    nn.fit(z_np)
    dist, idx = nn.kneighbors(z_np, return_distance=True)
    nbr_idx = torch.from_numpy(idx[:, 1:K+1].astype(np.int64))          # (N,K)
    nbr_dist = torch.from_numpy(dist[:, 1:K+1].astype(np.float32))      # (N,K)
    return nbr_idx, nbr_dist

@torch.no_grad()
def _sample_from_q_top_p(q: torch.Tensor, nbrs: torch.Tensor, top_p: float = 0.9):
    """Nucleus sampling from q over neighbor indices.
    q: (B,K) prob, nbrs: (B,K) indices (same index space as z_all).
    returns nxt: (B,) indices.
    """
    B, K = q.shape
    q_sorted, idx_sorted = torch.sort(q, dim=1, descending=True)
    cdf = torch.cumsum(q_sorted, dim=1)
    keep = cdf <= top_p
    keep[:, 0] = True
    q_nuc = q_sorted * keep.float()
    q_nuc = q_nuc / q_nuc.sum(dim=1, keepdim=True).clamp_min(1e-12)
    sampled_pos = torch.multinomial(q_nuc, num_samples=1).squeeze(1)
    sampled_k = idx_sorted[torch.arange(B, device=q.device), sampled_pos]
    nxt = nbrs[torch.arange(B, device=q.device), sampled_k]
    return nxt

@torch.no_grad()
def _build_directed_soft_targets_avgv_global(
    z_all: torch.Tensor,
    v_all: torch.Tensor,
    nbr_idx_global: torch.Tensor,
    T: int,
    idx0: torch.Tensor,
    cos_threshold: float = 0.0,
    tau_q: float = 0.25,
    top_p: float = 0.8,
):
    """Soft target mu + stochastic rollout on global kNN graph."""
    B = idx0.numel()
    D = z_all.size(1)
    mu_targets = torch.zeros((B, T, D), device=z_all.device)

    paths = torch.zeros((B, T), dtype=torch.long, device=z_all.device)
    curr = idx0.clone()
    paths[:, 0] = curr

    for t in range(1, T):
        nbrs = nbr_idx_global[curr]           # (B,K)
        z_n  = z_all[nbrs]                    # (B,K,D)
        z_c  = z_all[curr].unsqueeze(1)       # (B,1,D)
        diffs = z_n - z_c                     # (B,K,D)

        # --- base: velocity projection term ---
        v_dir = F.normalize(v_all[curr], dim=1, eps=1e-6).unsqueeze(1)
        cosines = F.cosine_similarity(diffs, v_dir, dim=-1)            # (B,K) in [-1,1]
        cos_norm = (cosines + 1.0) / 2.0                               # -> [0,1]

        if cos_threshold > 0.0:
            cos_norm = cos_norm.masked_fill(cos_norm < cos_threshold, 0.0)

        # stretch each row to [0,1] (same as your original)
        c_min = cos_norm.min(dim=1, keepdim=True).values
        c_max = cos_norm.max(dim=1, keepdim=True).values
        cos_stretched = (cos_norm - c_min) / (c_max - c_min + 1e-18)   # (B,K)

        # --- distance kernel ---
        d2 = (diffs * diffs).sum(dim=-1)                               # (B,K)
        d  = torch.sqrt(d2 + 1e-18)

        sigma = d.median(dim=1, keepdim=True).values.clamp_min(1e-12)  # (B,1)
        gauss_K = torch.exp(-d2 / (2.0 * sigma * sigma + 1e-18))       # (B,K)
        g_max = gauss_K.max(dim=1, keepdim=True).values
        gauss_norm = gauss_K / (g_max + 1e-18)                         # (B,K) in (0,1]

        # --- final base score (bounded, stable) ---
        score = cos_stretched * gauss_norm                              # (B,K) in [0,1]

        q = torch.softmax(score / max(tau_q, 1e-6), dim=1)
        nxt = _sample_from_q_top_p(q, nbrs, top_p=top_p)
        paths[:, t] = nxt
        target = z_all[nxt]                             # (B,D)
        mu_targets[:, t] = target
        curr = nxt

    return mu_targets, paths

import math, time
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import cycle

import math, time
from pathlib import Path
from typing import Optional, Union, Dict, Any
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_model(
    model: 'VAPOR',
    dataset: 'AnnDataDataset',
    config: Optional[Union['VAPORConfig', Dict[str, Any]]] = None,
    split_train_test: bool = True,
    test_size: float = 0.2,
    save_dir: Optional[Union[str, Path]] = 'vapor/out/',
    exp_name: str = "run",
    verbose: bool = True,
    graph_k: Optional[int] = None,
    graph_update_every_steps: int = 50,
    graph_build_batch_size: int = 2048,
    graph_use_mu: bool = False,
    soft_tau_q: float = 0.3,
    soft_top_p: float = 0.7,
    cos_threshold: float = 0.0,
    # optional knobs
    prior_weight: float = 1.0,          # 0.0 to disable prior
    traj_weight: float = 1.0,           # trajectory loss weight
    viz_every_steps: int = 0,           # 0 disables
    log_every: Optional[int] = None,    # default falls back to config.log_every or 50
    save_every_steps: Optional[int] = None,  # default falls back to config.save_every_steps or 500
    **kwargs
) -> 'VAPOR':

    # ---------- config ----------
    if config is None:
        config = VAPORConfig()
    elif isinstance(config, dict):
        config = VAPORConfig(**config)

    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
        else:
            print(f"Warning: Unknown parameter '{k}' ignored")

    # ---------- device ----------
    device = resolve_device(config)
    config.device = device
    model.to(device)

    # ---------- split ----------
    if split_train_test:
        n = len(dataset)
        n_test = max(1, int(round(n * test_size)))
        n_train = n - n_test
        g = torch.Generator().manual_seed(42)
        train_subset, test_subset = torch.utils.data.random_split(dataset, [n_train, n_test], generator=g)
        train_base, test_base = train_subset, test_subset
        if verbose:
            print(f"Train / Test split: train={n_train}, test={n_test} (test_size={test_size})")
    else:
        train_base, test_base = dataset, None

    train_dataset = _WithIndex(train_base)
    test_dataset = _WithIndex(test_base) if test_base is not None else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=getattr(config, "num_workers", 0),
        pin_memory=getattr(config, "pin_memory", False),
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=getattr(config, "num_workers", 0),
            pin_memory=getattr(config, "pin_memory", False),
        )

    # ---------- optimizer split ----------
    vae_params = [p for n, p in model.named_parameters() if n.startswith("vae.")]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("vae.")]
    assert len({id(p) for p in vae_params} & {id(p) for p in other_params}) == 0

    vae_scale = float(model.vae.latent_dim) / float(model.vae.input_dim)
    base_lr_vae = vae_scale * float(config.lr)
    base_lr_to  = float(config.lr)

    opt_vae = torch.optim.AdamW(vae_params, lr=base_lr_vae)
    opt_to  = torch.optim.AdamW(other_params, lr=base_lr_to)

    # ---------- step schedule ----------
    total_steps = int(config.total_steps)
    s1 = int(0.10 * total_steps)   # VAE-only
    s2 = int(0.20 * total_steps)   # ramp
    if verbose:
        print(f"Training (step-based) | batch={config.batch_size} | total_steps={total_steps}")

    # ---------- logging controls ----------
    log_every = int(log_every or getattr(config, "log_every", 50) or 50)
    save_every_steps = int(save_every_steps or getattr(config, "save_every_steps", 500) or 500)

    # ---------- global graph cache ----------
    K = int(graph_k or getattr(config, "graph_k", 50) or 50)
    soft_tau_q = float(soft_tau_q)
    soft_top_p = float(soft_top_p)

    z_all_cpu = None
    z_all_dev = None
    v_all = None
    nbr_idx_global = None

    @torch.no_grad()
    def _refresh_global_graph(step: int):
        nonlocal z_all_cpu, z_all_dev, v_all, nbr_idx_global
        # if verbose:
            # print(f"[Graph] Refreshing global kNN graph at step {step} (K={K}) ...")
        z_all_cpu = _encode_all_z(
            model, dataset, device=device,
            batch_size=graph_build_batch_size,
            use_mu=graph_use_mu,
        )  # CPU float32 (N,D)
        z_all_dev = z_all_cpu.to(device, non_blocking=True)  # cache on device ONCE per refresh
        nbr_idx_cpu, _ = _build_global_knn_graph_sklearn(z_all_cpu, K=K)
        nbr_idx_global = nbr_idx_cpu.to(device, non_blocking=True)
        v_all = model.compute_velocities(z_all_dev).detach()
        # if verbose:
            # print(f"[Graph] Done. z_all={tuple(z_all_cpu.shape)}, nbr_idx={tuple(nbr_idx_global.shape)}")

    _refresh_global_graph(step=0)

    # ---------- dataset flags ----------
    has_spatial = bool(getattr(dataset, "has_spatial", False))
    has_batch = bool(getattr(dataset, "has_batch", False))

    # infinite loader (reshuffle happens each epoch because loader is recreated each pass internally)
    loader_iter = cycle(train_loader)

    # ---------- step history (for CSV) ----------
    step_rows = []
    if save_dir is not None:
        save_dir = Path(save_dir)
        (save_dir / "csv").mkdir(parents=True, exist_ok=True)

    # ---------- window logging ----------
    win = dict(mse=0.0, kld=0.0, traj=0.0, prior=0.0, n=0)

    history = {
    "step": [],
    "recon": [],
    "kld": [],
    "traj": [],
    "prior": [],
    "total": [],
    "phase": [],}

    t_start = time.time()
    model.train()

    for global_step in range(total_steps):
        batch = next(loader_iter)

        # phase
        if global_step < s1:
            phase = "vae_only"
        elif global_step < s2:
            phase = "ramp_transport"
        else:
            phase = "full"
        is_warmup = (phase == "vae_only")

        # refresh global graph
        if graph_update_every_steps > 0 and (global_step % graph_update_every_steps == 0) and global_step != 0 and (not is_warmup):
            _refresh_global_graph(step=global_step)

        # unpack + idx_global
        idx_global = torch.as_tensor(batch[-1], device=device, dtype=torch.long)

        if has_spatial and has_batch:
            x, t_data, is_root, is_term, coords, batch_id = batch[:-1]
        elif has_spatial and (not has_batch):
            x, t_data, is_root, is_term, coords = batch[:-1]
            batch_id = None
        elif (not has_spatial) and has_batch:
            x, t_data, is_root, is_term, batch_id = batch[:-1]
            coords = None
        else:
            x, t_data, is_root, is_term = batch[:-1]
            coords = None
            batch_id = None

        x = x.to(device, non_blocking=True)
        t_data = t_data.to(device, non_blocking=True)
        is_root = torch.as_tensor(is_root, device=device, dtype=torch.bool)
        is_term = torch.as_tensor(is_term, device=device, dtype=torch.bool)

        # -------- forward VAE --------
        recon, z0, mu0, logvar0 = model.encode(x)
        recon_loss = F.mse_loss(recon, x)
        kl_loss = (-0.5 * (1 + logvar0 - mu0.pow(2) - logvar0.exp())).mean()
        vae_loss = recon_loss + float(config.beta) * kl_loss

        traj_loss = torch.zeros((), device=device)
        prior_loss = torch.zeros((), device=device)

        # -------- transport losses --------
        if not is_warmup:
            t_rand = int(torch.randint(1, int(config.t_max) + 1, (1,), device=device).item())
            t_span = torch.linspace(0, t_rand, t_rand + 1, device=device)
            z_traj = model.integrate(z0, t_span)

            mu_targets, _, = _build_directed_soft_targets_avgv_global(
                z_all=z_all_dev,
                v_all=v_all,
                nbr_idx_global=nbr_idx_global,
                T=z_traj.size(0),
                idx0=idx_global,
                tau_q=soft_tau_q,
                top_p=soft_top_p,
                cos_threshold=cos_threshold,
            )

            if z_traj.size(0) > 1:
                traj_loss = torch.stack(
                    [F.mse_loss(z_traj[t], mu_targets[:, t]) for t in range(1, z_traj.size(0))]
                ).mean()

            if prior_weight != 0.0:
                B = z0.size(0)
                if B >= 2:
                    z0_for_prior = z0  # or z0.detach()
                    v0 = model.compute_velocities(z0_for_prior)
                    k_eff = min(K, B - 1)
                    dists = torch.cdist(z0_for_prior, z0_for_prior)
                    knn_last = dists.topk(k_eff, dim=1, largest=False).values[:, -1]
                    eps_batch = torch.median(knn_last).item()
                    adj_idx_b, adj_mask_b = model.build_radius_graph(
                        z0_for_prior,
                        eps_batch,
                        getattr(config, "min_samples", 5),
                        getattr(config, "graph_k", 20),
                    )
                    prior_loss = model.flag_direction_loss_graph(
                        z0_for_prior, v0, is_root, is_term, adj_idx_b, adj_mask_b
                    )

        # ramp schedule
        if phase == "ramp_transport":
            r = (global_step - s1) / max(s2 - s1, 1)
            r = float(max(0.0, min(1.0, r)))
            w_traj = traj_weight * r
            w_prior = prior_weight * r
            lr_to = base_lr_to * r
        elif phase == "full":
            w_traj = traj_weight
            w_prior = prior_weight
            lr_to = base_lr_to
        else:
            w_traj = 0.0
            w_prior = 0.0
            lr_to = 0.0

        loss_transport = w_traj * traj_loss + w_prior * prior_loss
        loss = vae_loss + loss_transport

        # -------- backward + step --------
        opt_vae.zero_grad(set_to_none=True)
        opt_to.zero_grad(set_to_none=True)

        for pg in opt_vae.param_groups:
            pg["lr"] = base_lr_vae
        for pg in opt_to.param_groups:
            pg["lr"] = lr_to

        loss.backward()

        try:
            torch.nn.utils.clip_grad_norm_(model.transport_op.parameters(), max_norm=float(config.grad_clip))
        except Exception:
            pass

        opt_vae.step()
        if not is_warmup:
            opt_to.step()

        # -------- record history (STEP-BASED) --------
        history["step"].append(global_step)
        history["recon"].append(float(recon_loss.item()))
        history["kld"].append(float(kl_loss.item()))
        history["traj"].append(float(traj_loss.item()) if not is_warmup else 0.0)
        history["prior"].append(float(prior_loss.item()) if not is_warmup else 0.0)
        history["total"].append(float(loss.item()))
        history["phase"].append(phase)    

        # -------- window logging --------
        win["mse"] += float(recon_loss.item())
        win["kld"] += float(kl_loss.item())
        win["traj"] += float(traj_loss.item()) if not is_warmup else 0.0
        win["prior"] += float(prior_loss.item()) if (not is_warmup and prior_weight != 0.0) else 0.0
        win["n"] += 1

        if verbose and (global_step % log_every == 0) and win["n"] > 0:
            print(
                f"Step {global_step:06d}/{total_steps} | phase={phase:<13} | "
                f"Recon {win['mse']/win['n']:.4f} | KL {win['kld']/win['n']:.4f} | "
                f"Traj {win['traj']/win['n']:.4f} | Prior {win['prior']/win['n']:.4f} | "
                f"lr_to {lr_to:.2e}"
            )
            win = dict(mse=0.0, kld=0.0, traj=0.0, prior=0.0, n=0)

        # -------- save step csv --------
        if (save_dir is not None) and (global_step % save_every_steps == 0):
            row = dict(
                step=global_step,
                phase=phase,
                recon=float(recon_loss.item()),
                kl=float(kl_loss.item()),
                traj=float(traj_loss.item()) if not is_warmup else 0.0,
                prior=float(prior_loss.item()) if not is_warmup else 0.0,
                lr_vae=float(base_lr_vae),
                lr_to=float(lr_to),
            )
            step_rows.append(row)
            df = pd.DataFrame(step_rows)
            df.to_csv(save_dir / "csv" / f"{exp_name}_step_metrics.csv", index=False)

        # -------- optional viz hook --------
        if (viz_every_steps > 0) and (global_step % viz_every_steps == 0) and (save_dir is not None) and (not is_warmup):
            pass

    if verbose:
        print("-" * 80)
        print(f"Training completed. total_time={time.time()-t_start:.2f}s")

    plot_losses(history, save_dir=save_dir, show=True)

    return model

def ema(x, alpha=0.98):
    """Exponential moving average."""
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i-1] + (1 - alpha) * x[i]
    return y

def plot_losses(history, save_dir=None, show=True):
    import matplotlib.pyplot as plt
    from pathlib import Path

    plt.figure(figsize=(6,4))
    plt.plot(history["recon"], label="Recon")
    plt.plot(history["kld"], label="KLD")
    plt.plot(history["traj"], label="Traj")
    plt.plot(history["prior"], label="Prior")
    plt.plot(history["total"], label="Total")
    plt.legend()
    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "loss.png", dpi=200)

    if show:
        plt.show()
    else:
        plt.close()