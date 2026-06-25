import math
import time
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .config import VAPORConfig
from .utils import resolve_device

@torch.no_grad()
def set_optimizer_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)

@torch.no_grad()
def get_optimizer_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])

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
                mu_only: bool = False):
    was_training = model.training
    model.eval()

    loader = DataLoader(_WithIndex(dataset), batch_size=batch_size, shuffle=False)
    mu_list = []
    lv_list = [] if not mu_only else None
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        mu, logvar = model.vae.encode(x)
        mu_list.append(mu)
        if not mu_only:
            lv_list.append(logvar)
    mu_all = torch.cat(mu_list, dim=0)

    if was_training:
        model.train()

    if mu_only:
        return mu_all.cpu(), None

    lv_all = torch.cat(lv_list, dim=0)
    std_all = (0.5 * lv_all).exp()
    return mu_all.cpu(), std_all.cpu()

def _faiss_available(gpu: bool = False) -> bool:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import faiss
        if gpu:
            return faiss.get_num_gpus() > 0
        return True
    except ImportError:
        return False

@torch.no_grad()
def _build_global_knn_graph(z_all_cpu: torch.Tensor, K: int = 50, device=None):
    """Build kNN graph. Uses FAISS-GPU if available, else FAISS-CPU, else sklearn."""
    z_np = np.ascontiguousarray(z_all_cpu.numpy(), dtype=np.float32)
    N, D = z_np.shape
    K1 = K + 1  # include self

    use_gpu = (device is not None and str(device).startswith("cuda"))

    # Only use FAISS on GPU; sklearn is competitive on CPU
    if use_gpu and _faiss_available(gpu=True):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import faiss
        index = faiss.IndexFlatL2(D)
        if use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_cpu_to_all_gpus(index)
        index.add(z_np)
        dist_sq, idx = index.search(z_np, K1)
        nbr_idx = torch.from_numpy(idx[:, 1:K1].astype(np.int64))
        nbr_dist = torch.from_numpy(np.sqrt(np.maximum(dist_sq[:, 1:K1], 0)).astype(np.float32))
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=K1, metric="euclidean", n_jobs=-1, algorithm="auto")
        nn.fit(z_np)
        dist, idx = nn.kneighbors(z_np, return_distance=True)
        nbr_idx = torch.from_numpy(idx[:, 1:K1].astype(np.int64))
        nbr_dist = torch.from_numpy(dist[:, 1:K1].astype(np.float32))

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
def _compute_geo_dist_from_roots(
    nbr_idx_cpu: torch.Tensor,           # (N, K) int64
    nbr_dist_cpu: torch.Tensor,          # (N, K) float32
    root_idx: Optional[List[int]],
) -> Optional[torch.Tensor]:
    """Geodesic distance from any-of-root on a symmetrized kNN graph.

    Returns a (N,) float32 tensor. Cells in components without a root
    get NaN (the soft-target builder treats NaN as a neutral signal,
    so those cells fall back to v-only ranking). Returns None when
    no roots are supplied so callers can skip the geo branch entirely.
    """
    if not root_idx:
        return None
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    N, K = nbr_idx_cpu.shape
    rows = np.repeat(np.arange(N), K)
    cols = nbr_idx_cpu.numpy().ravel()
    data = nbr_dist_cpu.numpy().ravel()
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    # kNN is asymmetric; geodesic distance is on the undirected graph.
    A = A.maximum(A.T)

    d_mat = dijkstra(A, directed=False, indices=list(root_idx),
                     return_predecessors=False)
    d = np.min(d_mat, axis=0)             # min distance to ANY root
    d = np.where(np.isinf(d), np.nan, d)  # unreachable -> NaN
    return torch.from_numpy(d.astype(np.float32))

@torch.no_grad()
def _build_directed_soft_targets_avgv_global(
    z_all,
    v_all,
    nbr_idx_global,
    T,
    idx0,
    tau_q: float = 0.25,
    top_p: float = 0.8,
    geo_dist: Optional[torch.Tensor] = None,
    geo_mode: str = "forward_mask",          # 'forward_mask' | 'geo_v_blend' | 'off'
    geo_mask_floor: float = 0.05,
    w_v: float = 1.0,                        # only consulted when geo_mode='geo_v_blend'
    w_inertia: float = 0.0,                  # velocity-inertia weight (local)
    ref_dir: Optional[torch.Tensor] = None,  # (N, D) per-cell reference dir
    w_ref: float = 0.0,                      # weight on ref_dir alignment
):
    """Soft target rollout on a global kNN graph."""
    assert geo_mode in ("forward_mask", "geo_v_blend", "off"), \
        f"geo_mode must be 'forward_mask', 'geo_v_blend', or 'off'; got {geo_mode!r}"

    B = idx0.numel()
    D = z_all.size(1)
    mu_targets = torch.zeros((B, T, D), device=z_all.device)
    paths = torch.zeros((B, T), dtype=torch.long, device=z_all.device)
    curr = idx0.clone()
    paths[:, 0] = curr
    prev = None  # NEW: tracks z_all-index of the previous hop, for inertia

    for t in range(1, T):
        nbrs = nbr_idx_global[curr]
        z_n = z_all[nbrs]
        z_c = z_all[curr].unsqueeze(1)
        diffs = z_n - z_c

        # ── model-v cosine score (stretched to [0, 1]) ──
        v_dir = F.normalize(v_all[curr], dim=1, eps=1e-6).unsqueeze(1)
        cosines = F.cosine_similarity(diffs, v_dir, dim=-1)
        cos_norm = (cosines + 1.0) / 2.0
        c_min = cos_norm.min(dim=1, keepdim=True).values
        c_max = cos_norm.max(dim=1, keepdim=True).values
        cos_stretched = (cos_norm - c_min) / (c_max - c_min + 1e-18)

        # ── directional anchor ──
        if geo_dist is None or geo_mode == "off":
            score = cos_stretched

        elif geo_mode == "geo_v_blend":
            geo_self = geo_dist[curr].unsqueeze(1)
            geo_nbrs = geo_dist[nbrs]
            delta_geo = geo_nbrs - geo_self
            nan_mask = torch.isnan(delta_geo)
            delta_geo = torch.where(nan_mask, torch.zeros_like(delta_geo),
                                    delta_geo)
            d_min = delta_geo.min(dim=1, keepdim=True).values
            d_max = delta_geo.max(dim=1, keepdim=True).values
            geo_score = (delta_geo - d_min) / (d_max - d_min + 1e-18)
            score = (1.0 - w_v) * geo_score + w_v * cos_stretched

        elif geo_mode == "forward_mask":
            geo_self = geo_dist[curr].unsqueeze(1)
            geo_nbrs = geo_dist[nbrs]
            delta_geo = geo_nbrs - geo_self
            nan_mask = torch.isnan(delta_geo)
            delta_geo = torch.where(nan_mask, torch.zeros_like(delta_geo),
                                    delta_geo)
            fwd_mask = torch.where(
                delta_geo > 0,
                torch.ones_like(delta_geo),
                torch.full_like(delta_geo, geo_mask_floor),
            )
            fwd_mask = torch.where(nan_mask, torch.full_like(fwd_mask, 0.5),
                                   fwd_mask)
            score = cos_stretched * fwd_mask

        # velocity inertia: favor neighbors consistent with previous hop direction
        if w_inertia > 0.0 and prev is not None:
            prev_step = z_all[curr] - z_all[prev]
            prev_dir = F.normalize(prev_step, dim=1, eps=1e-6).unsqueeze(1)
            inertia_cos = F.cosine_similarity(diffs, prev_dir, dim=-1)
            score = score + w_inertia * (inertia_cos + 1.0) / 2.0

        # reference-direction anchor (optional, used in simulations)
        if ref_dir is not None and w_ref > 0.0:
            ref_curr = ref_dir[curr]                                # (B, D)
            ref_norm = ref_curr.norm(dim=1, keepdim=True)
            has_ref = (ref_norm.squeeze(-1) > 1e-6).float()         # (B,)
            ref_unit = ref_curr / ref_norm.clamp_min(1e-9)          # (B, D)
            ref_cos = F.cosine_similarity(
                diffs, ref_unit.unsqueeze(1), dim=-1)               # (B, K)
            ref_score = (ref_cos + 1.0) / 2.0
            score = score + w_ref * has_ref.unsqueeze(1) * ref_score

        q = torch.softmax(score / max(tau_q, 1e-6), dim=1)
        nxt = _sample_from_q_top_p(q, nbrs, top_p=top_p)
        paths[:, t] = nxt
        mu_targets[:, t] = z_all[nxt]
        prev = curr      # NEW
        curr = nxt

    return mu_targets, paths


import math as _math

def _make_sigmoid_schedule(sharpness=10):
    """Pre-compute constants for the sigmoid schedule."""
    y0 = 1 / (1 + _math.exp(sharpness * 0.5))
    y1 = 1 / (1 + _math.exp(-sharpness * 0.5))
    inv_range = 1.0 / (y1 - y0)
    def _sigmoid(start, end, step, total_steps):
        x = step / total_steps
        y = 1 / (1 + _math.exp(-sharpness * (x - 0.5)))
        t = (y - y0) * inv_range
        return start + (end - start) * t
    return _sigmoid

sigmoid = _make_sigmoid_schedule(sharpness=10)

def train_model(
    model: 'VAPOR',
    dataset: 'AnnDataDataset',
    config: Optional[Union['VAPORConfig', Dict[str, Any]]] = None,
    split_train_test: bool = True,
    test_size: float = 0.2,
    save_dir: Optional[Union[str, Path]] = 'vapor/out/',
    exp_name: str = "run",
    verbose: bool = True,
    graph_update_every_steps: int = 50,
    graph_build_batch_size: int = 2048,
    graph_use_mu: Optional[bool] = None,
    soft_tau_q: float = 0.3,
    soft_top_p: float = 0.7,
    # optional knobs
    prior_weight: float = 1.0,          # 0.0 to disable prior
    # viz_every_steps: int = 0,           # 0 disables
    log_every: Optional[int] = None,    # default falls back to config.log_every or 50
    save_every_steps: Optional[int] = None,
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

    # ---------- resolve defaults from config ----------
    if graph_use_mu is None:
        graph_use_mu = bool(config.graph_use_mu)

    # ---------- seed (ONE place) ----------
    seed = config.seed
    deterministic = bool(config.deterministic)
    seed_split = bool(config.seed_split)
    
    if seed is not None:
        from .utils import set_seed
        set_seed(int(seed), deterministic=deterministic)

    # ---------- device ----------
    device = resolve_device(config)
    config.device = device
    model.to(device)

    # ---------- mixed precision ----------
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.float16

    # ---------- split ----------
    if split_train_test:
        n = len(dataset)
        n_test = max(1, int(round(n * test_size)))
        n_train = n - n_test
        split_seed = 42 if seed is None else seed
        g = torch.Generator().manual_seed(split_seed)
        train_subset, test_subset = torch.utils.data.random_split(dataset, [n_train, n_test], generator=g)
        train_base, test_base = train_subset, test_subset
    else:
        train_base, test_base = dataset, None

    train_dataset = _WithIndex(train_base)
    test_dataset = _WithIndex(test_base) if test_base is not None else None

    # ---------- DataLoader generator ----------
    loader_seed = int(seed) if seed is not None else 0

    g_train = torch.Generator()
    g_train.manual_seed(loader_seed)
    g_test = torch.Generator()
    g_test.manual_seed(loader_seed + 1) 

    def _worker_init_fn(worker_id):
        import random as _rnd
        base = (loader_seed + worker_id) % (2**31)
        np.random.seed(base)
        _rnd.seed(base)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=getattr(config, "num_workers", 0),
        pin_memory=getattr(config, "pin_memory", False),
        generator=g_train,
        worker_init_fn=_worker_init_fn,
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
            generator=g_test,
            worker_init_fn=_worker_init_fn,
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

    # ---------- logging controls ----------
    log_every = int(log_every or getattr(config, "log_every", 50) or 50)
    save_every_steps = int(save_every_steps or getattr(config, "save_every_steps", 500) or 500)
    ckpt_every_steps = int(getattr(config, "ckpt_every_steps", 10000) or 10000)
    
    # ---------- global graph cache ----------
    K = int(config.graph_k)
    soft_tau_q = float(soft_tau_q)
    soft_top_p = float(soft_top_p)

    z_all_cpu = None
    z_all_dev = None
    v_all = None
    nbr_idx_global = None
    geo_dist_global = None

    from .utils import get_base_dataset
    base_ds = get_base_dataset(dataset)
    root_idx_list = (sorted(base_ds.root_indices)
                     if getattr(base_ds, "root_indices", None) else [])

    geo_refresh_every = max(500, graph_update_every_steps * 10)
    _last_geo_step = -geo_refresh_every  # force first computation

    @torch.no_grad()
    def _refresh_global_graph(step: int):
        nonlocal z_all_cpu, z_all_dev, v_all, nbr_idx_global, geo_dist_global
        nonlocal _last_geo_step
        mu_all, std_all = _encode_all_z(model, dataset, device=device,
                                        batch_size=graph_build_batch_size,
                                        mu_only=graph_use_mu)
        if graph_use_mu:
            z_all_cpu = mu_all
        else:
            z_all_cpu = mu_all + std_all * torch.randn_like(mu_all)
        z_all_dev = z_all_cpu.to(device)
        nbr_idx_cpu, nbr_dist_cpu = _build_global_knn_graph(z_all_cpu, K=K, device=device)
        nbr_idx_global = nbr_idx_cpu.to(device)
        if step - _last_geo_step >= geo_refresh_every:
            geo_t = _compute_geo_dist_from_roots(nbr_idx_cpu, nbr_dist_cpu,
                                                 root_idx_list)
            geo_dist_global = geo_t.to(device) if geo_t is not None else None
            _last_geo_step = step
        v_all = model.compute_velocities(z_all_dev).detach()

    _graph_initialized = False

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

    # ---------- dataset flags ----------
    def _infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    loader_iter = _infinite_loader(train_loader)

    # ---------- step history (for CSV) ----------
    step_rows = []
    test_rows = []
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        (save_dir / "csv").mkdir(parents=True, exist_ok=True)

    history = {
        "step": [], "recon": [], "kld": [],
        "traj": [], "prior": [], "total": [], "phase": [],
        "test_step": [],
        "test_recon": [],
        "test_kld": [],
    }

    t_start = time.time()
    model.train()

    pbar = None
    if verbose and tqdm is not None:
        pbar = tqdm(total=total_steps, desc="Training", unit="step")

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

        # refresh global graph (lazy init — skip during VAE-only warmup)
        if not is_warmup:
            if not _graph_initialized:
                if verbose:
                    print("Building initial graph...", flush=True)
                _refresh_global_graph(step=global_step)
                _graph_initialized = True
            elif graph_update_every_steps > 0 and (global_step % graph_update_every_steps == 0):
                _refresh_global_graph(step=global_step)

        # unpack + idx_global
        idx_global = torch.as_tensor(batch[-1], device=device, dtype=torch.long)

        x, is_root, is_term = batch[:-1]
        x = x.to(device, non_blocking=True)
        is_root = torch.as_tensor(is_root, device=device, dtype=torch.bool)
        is_term = torch.as_tensor(is_term, device=device, dtype=torch.bool)

        # -------- forward (with AMP) --------
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            recon, z0, mu0, logvar0 = model.encode(x)
            recon_loss = F.mse_loss(recon, x)
            kl_loss = (-0.5 * (1 + logvar0 - mu0.pow(2) - logvar0.exp())).mean()
            vae_loss = recon_loss + float(config.beta) * kl_loss

            traj_loss = torch.tensor(0.0, device=config.device)
            prior_loss = torch.tensor(0.0, device=config.device)
            loss_transport = torch.tensor(0.0, device=config.device)

            # -------- transport losses --------
            if not is_warmup:
                t_rand = int(torch.randint(1, int(config.t_max) + 1, (1,), device=device).item())
                t_span = torch.linspace(0, t_rand, t_rand + 1, device=device)
                z_traj = model.integrate(z0.float(), t_span, dt=config.dt)

                mu_targets, paths = _build_directed_soft_targets_avgv_global(
                    z_all=z_all_dev,
                    v_all=v_all,
                    nbr_idx_global=nbr_idx_global,
                    T=z_traj.size(0),
                    idx0=idx_global,
                    tau_q=soft_tau_q,
                    top_p=soft_top_p,
                        geo_dist=geo_dist_global,
                    geo_mode='geo_v_blend',
                    geo_mask_floor=0.05,
                    w_v=sigmoid(0, 1, global_step, total_steps),
                    w_inertia=2.0
                )

                if z_traj.size(0) > 1:
                    traj_loss = F.mse_loss(z_traj[1:], mu_targets[:, 1:].permute(1, 0, 2))

                if prior_weight != 0.0:
                    if is_root.any() or is_term.any():
                        z_at_curr = z_all_dev[idx_global]
                        v0 = model.compute_velocities(z_at_curr)
                        prior_loss = model.flag_direction_loss_graph_global(
                            z0=z_at_curr,
                            v0=v0,
                            is_start=is_root,
                            is_term=is_term,
                            z_all=z_all_dev,
                            nbr_idx_global=nbr_idx_global,
                            idx_global=idx_global,
                            nbr_mask_global=None,
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

                loss_transport = w_traj * traj_loss + w_prior * prior_loss
            else:
                sched_to = 0.0

            # -------- total loss --------
            loss = vae_loss + loss_transport
            lambda_orth = float(getattr(config, "lambda_orth", 0.0))
            orth_mode   = str(getattr(config, "orth_mode", "stiefel"))
            if lambda_orth > 0:
                orth = model.transport_op.orthogonality_loss(mode=orth_mode)
                loss = loss + lambda_orth * orth

        # -------- backward (with AMP) --------
        set_optimizer_lr(opt_vae, base_lr_vae * vae_mult)
        set_optimizer_lr(opt_to,  base_lr_to * base_to_scale * sched_to)

        opt_vae.zero_grad(set_to_none=True)
        opt_to.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()

        grad_clip = getattr(config, "grad_clip", None)
        if grad_clip is not None:
            scaler.unscale_(opt_vae)
            scaler.unscale_(opt_to)
            torch.nn.utils.clip_grad_norm_(model.transport_op.parameters(), max_norm=float(grad_clip))

        # -------- update-based adaptation snapshots (only full) --------
        do_adapt = (phase == "full") and (global_step % adapt_every == 0)

        if do_adapt:
            vae_old = _snap_params(vae_params)
            to_old  = _snap_params(other_params)

        scaler.step(opt_vae)
        if not is_warmup:
            scaler.step(opt_to)
        scaler.update()

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
                    if verbose and (global_step % log_every == 0):
                        print(f"[CALIB] target_u_ratio set to {target_u_ratio:.3f} from {len(calib_buf)} samples")
            else:
                # optional: very slow drift of target (keeps robust if dynamics change)
                drift = 0.01
                target_u_ratio = (1 - drift) * target_u_ratio + drift * u_ratio

            if target_u_ratio is not None:
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

        # -------- record history (STEP-BASED) --------
        history["step"].append(global_step)
        history["recon"].append(float(recon_loss.item()))
        history["kld"].append(float(kl_loss.item()))
        history["traj"].append(float(traj_loss.item()) if not is_warmup else 0.0)
        history["prior"].append(float(prior_loss.item()) if not is_warmup else 0.0)
        history["total"].append(float(loss.item()))
        history["phase"].append(phase)    

        # -------- progress --------
        if pbar is not None:
            pbar.update(1)
            if global_step % log_every == 0:
                postfix = {
                    "phase": phase,
                    "recon": f"{recon_loss.item():.4f}",
                    "kl": f"{kl_loss.item():.4f}",
                }
                if not is_warmup:
                    postfix["traj"] = f"{traj_loss.item():.4f}"
                pbar.set_postfix(postfix)
        elif verbose and (global_step % log_every == 0):
            traj_str = f" | traj={traj_loss.item():.4f}" if not is_warmup else ""
            print(f"[{global_step}/{total_steps}] {phase} | recon={recon_loss.item():.4f} | kl={kl_loss.item():.4f}{traj_str}", flush=True)

        step_rows.append(dict(
            step=global_step,
            phase=phase,
            recon=float(recon_loss.item()),
            kl=float(kl_loss.item()),
            traj=float(traj_loss.item()) if not is_warmup else 0.0,
            prior=float(prior_loss.item()) if not is_warmup else 0.0,
            total=float(loss.item()),
            lr_vae=float(get_optimizer_lr(opt_vae)),
            lr_to=float(get_optimizer_lr(opt_to)),
        ))

        if (save_dir is not None) and (global_step % save_every_steps == 0) and global_step > 0:
            pd.DataFrame(step_rows).to_csv(
                save_dir / "csv" / f"{exp_name}_step_metrics.csv", index=False
            )
            
        # -------- test loss -------- 
        if (test_loader is not None
            and global_step % save_every_steps == 0
            and global_step > 0):
            t_recon, t_kl = _eval_vae(model, test_loader, device)
            test_rows.append(dict(step=global_step, recon=t_recon, kl=t_kl))
            history["test_step"].append(global_step)
            history["test_recon"].append(t_recon)
            history["test_kld"].append(t_kl) 
            
            if save_dir is not None:
                pd.DataFrame(test_rows).to_csv(
                    save_dir / "csv" / f"{exp_name}_test_metrics.csv", index=False
                )

        # -------- periodic checkpoint --------
        if (ckpt_every_steps is not None
            and ckpt_every_steps > 0
            and global_step > 0
            and global_step % ckpt_every_steps == 0
            and save_dir is not None):
            
            from .utils import save_checkpoint
            ckpt_path = save_dir / "checkpoints" / f"{exp_name}_step{global_step:07d}.pt"
            save_checkpoint(
                model, config, ckpt_path,
                extra={
                    "step": global_step,
                    "phase": phase,
                    "exp_name": exp_name,
                },
            )
    if pbar is not None:
        pbar.close()
        
    step_rows.append(dict(
        step=global_step,
        phase=phase,
        recon=float(recon_loss.item()),
        kl=float(kl_loss.item()),
        traj=float(traj_loss.item()) if not is_warmup else 0.0,
        prior=float(prior_loss.item()) if not is_warmup else 0.0,
        total=float(loss.item()),
        lr_vae=float(get_optimizer_lr(opt_vae)),
        lr_to=float(get_optimizer_lr(opt_to)),
        ))

    if save_dir is not None:
        pd.DataFrame(step_rows).to_csv(
            save_dir / "csv" / f"{exp_name}_step_metrics.csv", index=False
        )

    plot_losses(history, save_dir=save_dir, show=True, s1=s1, s2=s2)

    return model

def ema(x, alpha=0.98):
    """Exponential moving average."""
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i-1] + (1 - alpha) * x[i]
    return y

@torch.no_grad()
def _eval_vae(model, loader, device):
    model.eval()
    tot_recon, tot_kl, n = 0.0, 0.0, 0
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        recon, z, mu, logvar = model.encode(x)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        b = x.size(0)
        tot_recon += recon_loss.item() * b
        tot_kl    += kl_loss.item()    * b
        n += b
    model.train()
    return tot_recon / max(n, 1), tot_kl / max(n, 1)

def plot_losses(history, save_dir=None, show=True, s1=None, s2=None):
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    steps = history["step"]

    # ===== Top: VAE losses =====
    ax = axes[0]
    ax.plot(steps, history["recon"], label="Recon (train)", color="C0", linewidth=1.5)
    ax.plot(steps, history["kld"],   label="KL (train)",    color="C1", linewidth=1.5)

    if history.get("test_step"):
        ax.plot(history["test_step"], history["test_recon"],
                label="Recon (test)", color="C0",
                linestyle="--", marker="o", markersize=3, linewidth=1.2)
        ax.plot(history["test_step"], history["test_kld"],
                label="KL (test)",    color="C1",
                linestyle="--", marker="o", markersize=3, linewidth=1.2)

    ax.set_ylabel("VAE loss")
    ax.set_title("VAE losses")
    ax.legend(loc="best", fontsize=9)

    # ===== Bottom: Transport losses =====
    ax = axes[1]
    ax.plot(steps, history["traj"],  label="Traj",  color="C2", linewidth=1.5)
    ax.plot(steps, history["prior"], label="Prior", color="C3", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Transport loss")
    ax.set_title("Transport losses")
    ax.legend(loc="best", fontsize=9)

    # ===== Phase boundaries on both subplots =====
    for ax in axes:
        if s1 is not None:
            ax.axvline(s1, color="gray", linestyle=":", alpha=0.6, linewidth=1)
        if s2 is not None:
            ax.axvline(s2, color="gray", linestyle=":", alpha=0.6, linewidth=1)

    # ===== Phase labels (top subplot only, to avoid clutter) =====
    if s1 is not None and s2 is not None:
        ax = axes[0]
        ymin, ymax = ax.get_ylim()
        y_text = ymax - 0.05 * (ymax - ymin)
        ax.text(s1 / 2,         y_text, "vae_only",
                ha="center", va="top", fontsize=8, color="gray", style="italic")
        ax.text((s1 + s2) / 2,  y_text, "ramp",
                ha="center", va="top", fontsize=8, color="gray", style="italic")
        ax.text((s2 + steps[-1]) / 2, y_text, "full",
                ha="center", va="top", fontsize=8, color="gray", style="italic")

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "loss.png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()