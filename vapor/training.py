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
        nbrs = nbr_idx_global[curr]         # (B,K)
        z_n  = z_all[nbrs]                  # (B,K,D)
        z_c  = z_all[curr].unsqueeze(1)     # (B,1,D)
        diffs = z_n - z_c

        v_nbrs = v_all[nbrs]
        v_avg = v_nbrs.mean(dim=1)
        v_dir = F.normalize(v_avg, dim=1, eps=1e-6).unsqueeze(1)

        cosines = F.cosine_similarity(diffs, v_dir, dim=-1)
        cos_norm = (cosines + 1) / 2
        cos_norm = cos_norm.masked_fill(cos_norm < cos_threshold, 0.0)
        c_min = cos_norm.min(dim=1, keepdim=True).values
        c_max = cos_norm.max(dim=1, keepdim=True).values
        cos_stretched = (cos_norm - c_min) / (c_max - c_min + 1e-18)

        d2 = (diffs * diffs).sum(dim=-1)
        d = torch.sqrt(d2 + 1e-18)
        sigma = d.median(dim=1, keepdim=True).values
        gauss = torch.exp(-d2 / (2 * sigma * sigma + 1e-18))
        gauss_norm = gauss / (gauss.max(dim=1, keepdim=True).values + 1e-18)

        score = cos_stretched * gauss_norm
        q = torch.softmax(score / max(tau_q, 1e-6), dim=1)

        mu = torch.einsum("bk,bkd->bd", q, z_n)
        mu_targets[:, t] = mu

        nxt = _sample_from_q_top_p(q, nbrs, top_p=top_p)
        paths[:, t] = nxt
        curr = nxt

    return mu_targets, paths

def train_model(
    model: 'VAPOR',
    dataset: 'AnnDataDataset',
    config: Optional[Union[VAPORConfig, Dict[str, Any]]] = None,
    split_train_test: bool = True,
    test_size: float = 0.2,
    eval_each_epoch: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    exp_name: str = "run_fullgraph",
    verbose: bool = True,
    graph_k: Optional[int] = None,
    graph_update_every: int = 5,
    graph_build_batch_size: int = 2048,
    graph_use_mu: bool = False,
    soft_tau_q: float = 0.3,
    soft_top_p: float = 0.7,
    cos_threshold: float = 0.0,
    **kwargs
) -> 'VAPOR':
    """Train VAPOR using a full-data (global) kNN graph, with both hard and soft targets.
    - Keeps your existing optimizer split (VAE vs transport op), horizons sampling, total_steps logic.
    - Adds: global graph cache updated every N epochs; logs hard-vs-soft diagnostics; optional visualization.
    """

    # ---------- config ----------
    if config is None:
        config = VAPORConfig()
    elif isinstance(config, dict):
        config = VAPORConfig(**config)

    if kwargs:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}' ignored")

    # ---------- device ----------
    device = resolve_device(config)
    config.device = device
    model.to(config.device)

    # ---------- DataLoader / split ----------
    use_grouped = config.by_batch and (getattr(dataset, "batch_ids", None) is not None)

    if split_train_test:
        n = len(dataset)
        n_test = max(1, int(round(n * test_size)))
        n_train = n - n_test
        g = torch.Generator().manual_seed(42)
        train_subset, test_subset = torch.utils.data.random_split(dataset, [n_train, n_test], generator=g)
        print(f"Train / Test split: train={n_train}, test={n_test} (test_size={test_size})")
        train_base = train_subset
        test_base = test_subset
    else:
        train_base = dataset
        test_base = None

    train_dataset = _WithIndex(train_base)
    test_dataset = _WithIndex(test_base) if test_base is not None else None

    if use_grouped and hasattr(dataset, "batch_ids") and dataset.batch_ids is not None:
        # keep your existing grouped sampler logic if present (currently same loader as else)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False) if test_dataset is not None else None
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False) if test_dataset is not None else None

    # ---------- optimizers split ----------
    vae_params = [p for n, p in model.named_parameters() if n.startswith("vae.")]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("vae.")]
    vae_ids = {id(p) for p in vae_params}
    other_ids = {id(p) for p in other_params}
    print("overlap params:", len(vae_ids & other_ids))
    assert len(vae_ids & other_ids) == 0

    vae_scale = model.vae.latent_dim / model.vae.input_dim
    base_lr_vae = vae_scale * config.lr
    base_lr_to  = config.lr

    opt_vae = torch.optim.AdamW(vae_params, lr=base_lr_vae)
    opt_to  = torch.optim.AdamW(other_params, lr=base_lr_to)

    # ---------- steps / epochs ----------
    N_train = len(train_dataset)
    steps_per_epoch = math.ceil(N_train / config.batch_size)
    total_steps = int(config.total_steps)
    s1 = int(0.10 * total_steps)   # VAE-only
    s2 = int(0.20 * total_steps)   # ramp transport
    epochs = max(1, math.ceil(total_steps / steps_per_epoch))
    print(f"Training (full-graph) for {epochs} epochs | batch={config.batch_size} | steps/epoch={steps_per_epoch}")
    global_step = 0

    # ---------- history ----------
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

    # ---------- global graph cache ----------
    K = int(graph_k or getattr(config, "graph_k", 50) or 50)
    soft_tau_q = float(soft_tau_q)
    soft_top_p = float(soft_top_p)

    z_all_cpu = None
    z_all_dev = None
    v_all = None
    nbr_idx_global = None
    emb_all_2d = None

    def _refresh_global_graph(epoch: int):
        nonlocal z_all_cpu, z_all_dev, v_all, nbr_idx_global, emb_all_2d
        print(f"[Graph] Refreshing global kNN graph at epoch {epoch} (K={K}) ...")
        z_all_cpu = _encode_all_z(model, dataset, device=config.device,
                                 batch_size=graph_build_batch_size, use_mu=graph_use_mu)
        z_all_dev = z_all_cpu.to(config.device)
        nbr_idx_cpu, _ = _build_global_knn_graph_sklearn(z_all_cpu, K=K)
        nbr_idx_global = nbr_idx_cpu.to(config.device)
        v_all = model.compute_velocities(z_all_dev).detach()
        try:
            from sklearn.decomposition import PCA
            emb_all_2d = PCA(n_components=2, random_state=0).fit_transform(z_all_cpu.numpy()).astype(np.float32)
        except Exception as e:
            emb_all_2d = None
            print(f"[Graph] PCA embedding failed: {repr(e)}")
        print(f"[Graph] Done. z_all shape={tuple(z_all_cpu.shape)}, nbr_idx shape={tuple(nbr_idx_global.shape)}")

    _refresh_global_graph(epoch=0)

    # ---------- update-based balancing controller (AdamW-aware) ----------
    @torch.no_grad()
    def _snap_params(params):
        return [p.detach().clone() for p in params if p.requires_grad]

    @torch.no_grad()
    def _median_rel_update(params, params_old, eps: float = 1e-12):
        vals = []
        j = 0
        for p in params:
            if not p.requires_grad:
                continue
            dp = p.detach() - params_old[j]
            denom = p.detach().abs().mean().item() + eps
            vals.append(dp.abs().mean().item() / denom)
            j += 1
        return float(np.median(vals)) if vals else 0.0

    # multipliers that will be adapted
    vae_mult = 1.0
    to_mult  = 1.0

    # soft bounds
    vae_mult_min, vae_mult_max = 0.5, 2.0
    to_mult_min,  to_mult_max  = 0.3, 3.0

    adapt_every = 25
    ema_beta = 0.9
    u_ratio_ema = 1.0

    # --- auto-calibration of target_u_ratio ---
    target_u_ratio = None          # will be set automatically
    calib_steps = 100              # number of FULL steps for calibration
    calib_buf = []                 # store u_ratio samples during calibration

    # --- controller knobs ---
    deadband = 0.15                # ±15% around target: no action
    max_step_to  = 0.05            # to_mult per update (log-space)
    max_step_vae = 0.03            # vae_mult per update (log-space)

    # --- base_lr_to auto-rescale (for robustness across datasets) ---
    base_to_scale = 1.0            # multiplies base_lr_to effectively
    rescale_patience = 8           # how many adapt events stuck at bounds before rescaling
    stuck_count = 0
    rescale_step = 0.2             # change base_to_scale by exp(±0.2) ≈ x1.22
    base_to_scale_min, base_to_scale_max = 0.1, 10.0

    eps = 1e-12

    # ---------- training ----------
    print("\nStarting training...\n" + "-" * 80)
    has_spatial = getattr(dataset, "has_spatial", False)
    has_batch = getattr(dataset, "has_batch", False)

    for epoch in range(1, epochs + 1):
        if global_step >= total_steps:
            break

        epoch_start = time.time()
        model.train()

        epoch_metrics = dict(mse=0.0, kld=0.0, traj=0.0, prior=0.0, psi=0.0, loss_hard=0.0, loss_soft=0.0)
        batch_count = 0

        horizons = torch.randperm(config.t_max, device=config.device).add_(1).tolist()

        if graph_update_every > 0 and (epoch % graph_update_every) == 0:
            _refresh_global_graph(epoch=epoch)

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= total_steps:
                break

            # -------- phase (based on global_step) --------
            if global_step < s1:
                phase = "vae_only"
            elif global_step < s2:
                phase = "ramp_transport"
            else:
                phase = "full"

            is_warmup = (phase == "vae_only")

            # unpack + idx_global (last item)
            idx_global = torch.as_tensor(batch[-1], device=config.device, dtype=torch.long)

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

            x = x.to(config.device)
            t_data = t_data.to(config.device)
            is_root = torch.as_tensor(is_root, device=config.device, dtype=torch.bool)
            is_term = torch.as_tensor(is_term, device=config.device, dtype=torch.bool)
            if coords is not None:
                coords = coords.to(config.device)

            # -------- forward (VAE) --------
            recon, z0, mu0, logvar0 = model.encode(x)
            recon_loss = F.mse_loss(recon, x)
            kl_loss = (-0.5 * (1 + logvar0 - mu0.pow(2) - logvar0.exp())).mean()
            vae_loss = recon_loss + config.beta * kl_loss

            traj_loss_soft = torch.tensor(0.0, device=config.device)
            prior_loss = torch.tensor(0.0, device=config.device)
            loss_transport = torch.tensor(0.0, device=config.device)

            # -------- build transport losses if not warmup --------
            if not is_warmup:
                t_rand = horizons[batch_idx % config.t_max]
                t_span = torch.linspace(0, t_rand, t_rand + 1, device=config.device)
                z_traj = model.integrate(z0, t_span)

                mu_targets, paths_soft = _build_directed_soft_targets_avgv_global(
                    z_all=z_all_dev,
                    v_all=v_all,
                    nbr_idx_global=nbr_idx_global,
                    T=z_traj.size(0),
                    idx0=idx_global,
                    cos_threshold=cos_threshold,
                    tau_q=soft_tau_q,
                    top_p=soft_top_p,
                )

                if z_traj.size(0) > 1:
                    traj_loss_soft = torch.stack(
                        [F.mse_loss(z_traj[t], mu_targets[:, t]) for t in range(1, z_traj.size(0))]
                    ).mean()

                # prior graph (batch-local)
                B = z0.size(0)
                if B >= 2:
                    z0_for_prior = z0  # keep gradients; change to z0.detach() if desired
                    v0 = model.compute_velocities(z0_for_prior)
                    k_eff = min(K, B - 1)
                    dists = torch.cdist(z0_for_prior, z0_for_prior)
                    eps_batch = torch.median(dists.topk(k_eff, dim=1, largest=False).values[:, -1]).item()

                    adj_idx_b, adj_mask_b = model.build_radius_graph(
                        z0_for_prior,
                        eps_batch,
                        getattr(config, "min_samples", 5),
                        getattr(config, "graph_k", 20),
                    )
                    prior_loss = model.flag_direction_loss_graph(
                        z0_for_prior, v0, is_root, is_term, adj_idx_b, adj_mask_b
                    )

                # schedule multipliers
                if phase == "ramp_transport":
                    r = (global_step - s1) / max(s2 - s1, 1)
                    r = float(max(0.0, min(1.0, r)))
                    sched_to = r
                    w_traj = r
                    w_prior = r
                else:
                    sched_to = 1.0
                    w_traj = 1.0
                    w_prior = 1.0

                loss_transport = w_traj * traj_loss_soft + w_prior * prior_loss
            else:
                sched_to = 0.0

            # -------- set LRs for this step (ramp fixed to_mult=1) --------
            set_optimizer_lr(opt_vae, base_lr_vae * vae_mult)
            to_mult_eff = to_mult if phase == "full" else 1.0
            set_optimizer_lr(opt_to,  base_lr_to  * sched_to * to_mult_eff)

            if verbose and (global_step % adapt_every == 0):
                print(f"[LR] lr_vae={get_optimizer_lr(opt_vae):.2e} lr_to={get_optimizer_lr(opt_to):.2e} (sched_to={sched_to:.3f})")

            # -------- backward --------
            opt_vae.zero_grad(set_to_none=True)
            opt_to.zero_grad(set_to_none=True)

            loss = vae_loss + loss_transport
            loss.backward()

            # clip transport gradients (if exists)
            try:
                torch.nn.utils.clip_grad_norm_(model.transport_op.parameters(), max_norm=config.grad_clip)
            except Exception:
                pass

            # -------- update-based adaptation snapshots (only full) --------
            do_adapt = (phase == "full") and (global_step % adapt_every == 0)

            if do_adapt:
                vae_old = _snap_params(vae_params)
                to_old  = _snap_params(other_params)

            opt_vae.step()
            if not is_warmup:
                opt_to.step()

            if do_adapt:
                u_vae = _median_rel_update(vae_params, vae_old)
                u_to  = _median_rel_update(other_params, to_old)
                u_ratio = u_vae / (u_to + eps)    # <1 means TO updating more than VAE

                # EMA of observed ratio
                u_ratio_ema = ema_beta * u_ratio_ema + (1 - ema_beta) * u_ratio

                # -------- 1) auto-calibration --------
                # collect samples for first calib_steps full steps
                if target_u_ratio is None:
                    calib_buf.append(u_ratio)
                    if len(calib_buf) >= calib_steps:
                        # robust target: median (less sensitive than mean)
                        target_u_ratio = float(np.median(calib_buf))
                        # safety clamp: don't let target be crazy
                        target_u_ratio = float(np.clip(target_u_ratio, 0.05, 20.0))
                        if verbose:
                            print(f"[CALIB] target_u_ratio set to {target_u_ratio:.3f} from {len(calib_buf)} samples")
                else:
                    # optional: very slow drift of target (keeps robust if dynamics change)
                    drift = 0.01
                    target_u_ratio = (1 - drift) * target_u_ratio + drift * u_ratio

                # if still calibrating, don't adapt multipliers yet (avoid chasing noise)
                if target_u_ratio is None:
                    if verbose:
                        print(f"[UPD] step={global_step} u_vae={u_vae:.3e} u_to={u_to:.3e} u_ratio={u_ratio:.3f} (calibrating)")
                else:
                    # -------- 2) robust deadband control around target --------
                    # normalized error: >0 means VAE updates too much relative to TO
                    err = math.log((u_ratio_ema + eps) / (target_u_ratio + eps))

                    # deadband: ignore small fluctuations
                    if abs(err) < deadband:
                        step_to = 0.0
                        step_vae = 0.0
                    else:
                        # symmetric control (small steps, clamped)
                        step_to  = max(-max_step_to,  min(max_step_to,  +0.5 * err))
                        step_vae = max(-max_step_vae, min(max_step_vae, -0.5 * err))

                    # apply
                    to_mult  *= math.exp(step_to)
                    vae_mult *= math.exp(step_vae)

                    # clamp
                    to_mult  = float(np.clip(to_mult,  to_mult_min,  to_mult_max))
                    vae_mult = float(np.clip(vae_mult, vae_mult_min, vae_mult_max))

                    # -------- 3) bound-stuck detector -> rescale base_lr_to --------
                    stuck = (to_mult <= to_mult_min + 1e-9) or (to_mult >= to_mult_max - 1e-9) \
                        or (vae_mult <= vae_mult_min + 1e-9) or (vae_mult >= vae_mult_max - 1e-9)

                    if stuck and abs(err) >= deadband:
                        stuck_count += 1
                    else:
                        stuck_count = max(0, stuck_count - 1)

                    if stuck_count >= rescale_patience:
                        # If u_ratio_ema < target => TO too strong => shrink base_to_scale
                        # If u_ratio_ema > target => TO too weak => grow base_to_scale
                        direction = -1.0 if (u_ratio_ema < target_u_ratio) else +1.0
                        base_to_scale *= math.exp(direction * rescale_step)
                        base_to_scale = float(np.clip(base_to_scale, base_to_scale_min, base_to_scale_max))
                        stuck_count = 0

            # -------- metrics accumulate --------
            epoch_metrics["mse"] += float(recon_loss.item())
            epoch_metrics["kld"] += float(kl_loss.item())
            epoch_metrics["traj"] += float(traj_loss_soft.item()) if not is_warmup else 0.0
            epoch_metrics["prior"] += float(prior_loss.item()) if not is_warmup else 0.0
            epoch_metrics["psi"] += 0.0
            epoch_metrics["loss_soft"] += float(traj_loss_soft.item()) if not is_warmup else 0.0

            batch_count += 1
            global_step += 1

        # epoch averages
        if batch_count > 0:
            for k in epoch_metrics:
                epoch_metrics[k] /= batch_count

        history["epoch"].append(epoch)
        history["time"].append(time.time() - epoch_start)
        history["train_mse"].append(epoch_metrics["mse"])
        history["train_kld"].append(epoch_metrics["kld"])
        history["train_traj"].append(epoch_metrics["traj"])
        history["train_prior"].append(epoch_metrics["prior"])
        history["train_psi"].append(epoch_metrics["psi"])

        if verbose:
            print(
                f"Epoch {epoch}/{epochs} | time {history['time'][-1]:.2f}s | "
                f"Recon {epoch_metrics['mse']:.4f} | KL {epoch_metrics['kld']:.4f} | "
                f"TrajSoft {epoch_metrics['traj']:.4f} | Prior {epoch_metrics['prior']:.4f} | phase={phase}"
            )

        if save_dir is not None:
            try:
                _save_epoch_csv_and_plots(history, Path(save_dir), exp_name=exp_name)
            except Exception:
                pass

    print("-" * 80)
    print("Training completed (full-graph)!")

    _plot_training_metrics({
        "mse_losses": history["train_mse"],
        "kld_losses": history["train_kld"],
        "traj_losses": history["train_traj"],
        "prior_losses": history["train_prior"],
        "psi_losses": history["train_psi"],
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