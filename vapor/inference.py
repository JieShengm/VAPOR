import numpy as np
import torch

def row_zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    row_means = X.mean(axis=1, keepdims=True)
    row_stds = X.std(axis=1, keepdims=True, ddof=0)
    row_stds = np.where(row_stds == 0, 1.0, row_stds)
    return (X - row_means) / row_stds

@torch.no_grad()
def extract_latents_and_dynamics(
    model,
    adata,
    scale: bool = True,
    device: str = "cpu",
):
    """
    Returns a dict:
      recon, z, decomposed_dynamics, Uhat, g_t, Psi_list
    """
    import anndata as ad
    import numpy as np
        
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    if scale:
        X = row_zscore(X)

    model = model.eval().to(device)
    x_t = torch.tensor(X, dtype=torch.float32, device=device)

    _, z, *_ = model.vae(x_t)

    # dynamics decomposition
    Uhat = model.transport_op.unit_directions(z)              
    g_t = model.transport_op.get_mixture_weights(Uhat)        


    adata_VAPOR = ad.AnnData(np.array(z))
    adata_VAPOR.obs = adata.obs
    
    g_t_seq = g_t.numpy()
    for i in range(g_t_seq.shape[1]):
         adata_VAPOR.obs[f'pw_{i+1}']  = g_t_seq[:,i]

    for i in range(len(model.transport_op.Psi)):
        adata_VAPOR.layers[f'v_psi{i+1}'] = np.array(Uhat[:,i,:])
        
    t = torch.zeros(1).to(device)
    dz = model.transport_op(t, z)
    adata_VAPOR.layers[f'v_VAPOR'] = np.array(dz)
    
    adata_VAPOR.obsm['X_VAPOR'] = np.array(z)
    adata_VAPOR.layers['vapor']=adata_VAPOR.obsm['X_VAPOR']
    return adata_VAPOR

import math, time
from typing import Dict, List, Optional, Tuple, Iterable
import torch
import torch.nn.functional as F

try:
    from torch.autograd.functional import jvp as _torch_jvp
    _HAS_JVP = True
except Exception:
    _HAS_JVP = False
    
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _weighted_aggregate(acc_sum: torch.Tensor, acc_w: float, acc_w2: float,
                        tang: torch.Tensor, w: torch.Tensor):
    """Streaming aggregate: acc_sum += (w[:,None]*tang).sum(0), etc."""
    acc_sum += (tang * w[:, None]).sum(0)
    acc_w += float(w.sum().item())
    acc_w2 += float((w*w).sum().item())
    return acc_sum, acc_w, acc_w2

def _compute_gate_thresholds_minmax(
    model, z_all: torch.Tensor, q: float,
    batch_size: int = 2048, progress: bool = False, pbar=None,
    robust: bool = True, q_low: float = 0.01, q_high: float = 0.99, eps: float = 1e-8,
):
    model.eval()
    to = model.transport_op
    vals = []
    with torch.no_grad():
        for start in range(0, z_all.size(0), batch_size):
            z = z_all[start:start+batch_size].to(next(to.parameters()).device, non_blocking=True)
            Uhat = to.unit_directions(z)          # (B,M,d)
            g = to.get_mixture_weights(Uhat)      # (B,M)
            vals.append(g.detach().cpu())
            if progress and pbar is not None:
                pbar.update(1)
    G = torch.cat(vals, dim=0)                    # (N,M) on CPU
    if robust:
        g_min = torch.quantile(G, q_low,  dim=0)
        g_max = torch.quantile(G, q_high, dim=0)
    else:
        g_min = G.min(dim=0).values
        g_max = G.max(dim=0).values
    rng = (g_max - g_min).clamp_min(eps)
    Gn = ((G - g_min) / rng).clamp_(0, 1)
    thr_norm = torch.quantile(Gn, q, dim=0)       # (M,)
    thr_raw = g_min + thr_norm * rng              # back to original scale
    return thr_raw, dict(g_min=g_min, g_max=g_max, rng=rng, thr_norm=thr_norm)


@torch.no_grad()
def directional_gene_scores_jvp_progress(
    model,
    z_all: torch.Tensor,
    # gene_names: Optional[List[str]] = None,
    modes: Optional[Iterable[int]] = None,
    batch_size: int = 256,
    alpha: float = 1.0,
    tau_quantile: Optional[float] = 0.6,
    speed_normalize: bool = True,
    density_weights: Optional[torch.Tensor] = None,
    use_autocast: bool = False,
    autocast_dtype: torch.dtype = torch.float16,
    clip_gate_pct: Optional[float] = None,
    show_progress: bool = True,
):
    device = next(model.vae.parameters()).device
    model.eval()
    vae = model.vae
    to = model.transport_op
    mode_ids = list(range(to.n_dynamics)) if modes is None else list(modes)

    # Pre-compute loop sizes for progress
    n_batches_main = math.ceil(z_all.size(0) / batch_size)
    n_batches_speed = math.ceil(z_all.size(0) / (batch_size*4))
    total_iters = 0
    # speed pass counts per batch per mode
    total_iters += n_batches_speed * len(mode_ids)
    # gate-quantile pass counts per batch (no per-mode loop)
    if tau_quantile is not None:
        total_iters += math.ceil(z_all.size(0) / max(1024, batch_size*4))
    # main pass counts per batch per mode
    total_iters += n_batches_main * len(mode_ids)

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total_iters, desc="Directional gene scoring", leave=True)

    # ---------- Phase A: per-mode median speed ||v_m|| ----------
    speed_medians = {}
    with torch.no_grad():
        for start in range(0, z_all.size(0), batch_size*4):
            z = z_all[start:start+batch_size*4].to(device, non_blocking=True)
            for m in mode_ids:
                v = z @ to.Psi[m]
                vv = torch.linalg.vector_norm(v, dim=1)
                if m not in speed_medians:
                    speed_medians[m] = []
                speed_medians[m].append(vv.detach().cpu())
                if pbar is not None: pbar.update(1)
        for m in mode_ids:
            speed_medians[m] = torch.cat(speed_medians[m]).median().item() + 1e-8

    # ---------- Phase B: gate thresholds (quantiles) ----------
    # thr = None
    # if tau_quantile is not None:
    #     thr = _compute_gate_quantiles(model, z_all, tau_quantile,
    #                                   batch_size=max(1024, batch_size*4),
    #                                   progress=True if pbar is not None else False,
    #                                   pbar=pbar)
    thr_raw = None
    # thr_stats = None
    if tau_quantile is not None:
        thr_raw, _ = _compute_gate_thresholds_minmax(
            model, z_all, tau_quantile,
            batch_size=max(1024, batch_size*4),
            progress=True if pbar is not None else False,
            pbar=pbar,
            robust=True, q_low=0.01, q_high=0.99,
        )

    # ---------- Storage ----------
    G = vae.decoder[-1].out_features  # assumes final layer is Linear to genes
    scores = {m: torch.zeros(G, device=device) for m in mode_ids}
    sum_w = {m: 0.0 for m in mode_ids}
    sum_w2 = {m: 0.0 for m in mode_ids}
    used_cells = {m: 0 for m in mode_ids}
    pos_mass = {m: 0.0 for m in mode_ids}
    neg_mass = {m: 0.0 for m in mode_ids}

    autocast_cm = torch.autocast(device_type=(device.type if device.type != "mps" else "cpu"),
                                 dtype=autocast_dtype, enabled=use_autocast)
    # ---------- Phase C: main JVP loop ----------
    for start in range(0, z_all.size(0), batch_size):
        z = z_all[start:start+batch_size].to(device, non_blocking=True)
        z.requires_grad_(True)
        with torch.set_grad_enabled(True), autocast_cm:
            u = to.unit_directions(z)
            gates = to.get_mixture_weights(u)  # (B,M)
            if clip_gate_pct is not None:
                g_hi = torch.quantile(gates, clip_gate_pct, dim=0, keepdim=True)
                gates = torch.minimum(gates, g_hi)

            for m in mode_ids:
                c = gates[:, m]
                # keep = (c >= thr[m]) if thr is not None else torch.ones_like(c, dtype=torch.bool)
                keep = (c >= thr_raw[m]) if thr_raw is not None else torch.ones_like(c, dtype=torch.bool)
                if keep.sum() == 0:
                    if pbar is not None: pbar.update(1)
                    continue

                z_k = z[keep]
                c_k = c[keep]
                v = z_k @ to.Psi[m]
                sp = torch.linalg.vector_norm(v, dim=1)
                if speed_normalize:
                    v = v / (sp.unsqueeze(1) + 1e-8)
                v = v * c_k.unsqueeze(1)

                # base weights
                w = (c_k.clamp_min(0)**alpha) * (sp / speed_medians[m]).clamp_min(1e-6)
                if density_weights is not None:
                    w = w * density_weights[start:start+batch_size][keep].to(device)

                # JVP (fallback to finite diff if unavailable)
                if _HAS_JVP:
                    def f(z_in): return vae.decode(z_in)
                    _, tang = _torch_jvp(f, (z_k,), (v,), create_graph=False, strict=True)
                else:
                    eps = 1e-2
                    x0 = vae.decode(z_k)
                    x1 = vae.decode(z_k + eps * v)
                    tang = (x1 - x0) / eps

                # accumulate
                scores[m], sum_w[m], sum_w2[m] = _weighted_aggregate(scores[m], sum_w[m], sum_w2[m], tang, w)
                used_cells[m] += int(w.numel())
                pos_mass[m] += float((w * (tang.mean(dim=1) > 0).float()).sum().item())
                neg_mass[m] += float((w * (tang.mean(dim=1) < 0).float()).sum().item())

                if pbar is not None: pbar.update(1)

        z.grad = None
        del z

    if pbar is not None: pbar.close()

    # ---------- Finalize ----------
    out_scores: Dict[int, torch.Tensor] = {}
    # info: Dict[int, dict] = {}
    for m in mode_ids:
        if sum_w[m] <= 0:
            out_scores[m] = scores[m].detach()
    #         info[m] = dict(sum_w=0.0, n_eff=0.0, used_cells=used_cells[m],
    #                        pos_frac=float('nan'),
    #                        gate_threshold=(thr[m].item() if thr is not None else None),
    #                        median_speed=speed_medians[m])
    #         continue
        s = scores[m] / sum_w[m]
        scale = s.abs().median().item() + 1e-8
        s = s / scale
        out_scores[m] = s.detach()

        # n_eff = (sum_w[m]**2) / (sum_w2[m] + 1e-12)
        # pos_frac = pos_mass[m] / (pos_mass[m] + neg_mass[m] + 1e-12)
        # info[m] = dict(sum_w=sum_w[m], n_eff=n_eff, used_cells=used_cells[m],
        #                pos_frac=pos_frac,
        #                gate_threshold=(thr[m].item() if thr is not None else None),
        #                median_speed=speed_medians[m])        
        # info[m] = dict(
        # sum_w=sum_w[m], n_eff=n_eff, used_cells=used_cells[m],
        # pos_frac=pos_frac,
        # gate_threshold=(thr_raw[m].item() if thr_raw is not None else None),
        # median_speed=speed_medians[m],)

    return out_scores#, info


@torch.no_grad()
def summarize_gene_lists(scores: Dict[int, torch.Tensor], gene_names: List[str], top: int = 200):
    res = {}
    gnames = list(gene_names)
    for m, s in scores.items():
        s = s.to('cpu')
        idx_up = torch.argsort(-s)[:top]
        idx_dn = torch.argsort(s)[:top]
        res[m] = dict(
            up_genes=[gnames[i] for i in idx_up.tolist()],
            down_genes=[gnames[i] for i in idx_dn.tolist()],
            scores=s
        )
    return res