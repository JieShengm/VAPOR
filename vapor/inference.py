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


    adata_vapor = ad.AnnData(np.array(z))
    adata_vapor.obs = adata.obs
    
    g_t_seq = g_t.numpy()
    for i in range(g_t_seq.shape[1]):
         adata_vapor.obs[f'pw_{i+1}']  = g_t_seq[:,i]

    for i in range(len(model.transport_op.Psi)):
        adata_vapor.layers[f'v_psi{i+1}'] = np.array(Uhat[:,i,:])
        
    t = torch.zeros(1).to(device)
    dz = model.transport_op(t, z)
    adata_vapor.layers[f'v_VAPOR'] = np.array(dz)
    
    adata_vapor.obsm['X_VAPOR'] = np.array(z)
    adata_vapor.layers['vapor']=adata_vapor.obsm['X_VAPOR']

    adata_vapor.uns["gene_names"] = adata.var_names.tolist()
    adata_vapor.uns["n_dynamics"] = model.transport_op.n_dynamics
    if 'X_umap' in adata.obsm and adata.obsm['X_umap'] is not None:
        adata_VAPOR.obsm['X_gex_umap'] = adata.obsm['X_umap']
    
    return adata_vapor

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

# @torch.no_grad()
def directional_gene_scores_jvp_progress(
    model,
    adata_vapor,
    layer_key: Optional[str] = None,     # if None: use adata_vapor.X ; else adata_vapor.layers[latent_key]
    store_key: str = "vapor_directional_gene_scores",
    # z_all: torch.Tensor,
    # gene_names: Optional[List[str]] = None,
    psi_indices: Optional[Iterable[int]] = None,
    batch_size: int = 256,
    alpha: float = 1.0,
    tau_quantile: Optional[float] = 0.6,
    speed_normalize: bool = True,
    density_weights: Optional[torch.Tensor] = None,
    use_autocast: bool = False,
    autocast_dtype: torch.dtype = torch.float16,
    clip_gate_pct: Optional[float] = None,
    show_progress: bool = True,
    device: Optional[str] = None,
):
    # -------- resolve device --------
    if device is None:
        device_t = next(model.vae.parameters()).device
    else:
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        device_t = torch.device(device)

    model = model.to(device_t).eval()

    # -------- fetch latent matrix --------
    if layer_key is None:
        Z = adata_vapor.X
    else:
        Z = adata_vapor.layers[layer_key]

    # ensure dense numpy
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    Z = np.asarray(Z)

    # keep z_all on CPU; the core function moves batches to GPU/CPU device
    z_all = torch.from_numpy(Z).float().cpu()

    # device = next(model.vae.parameters()).device
    # model.eval()
    vae = model.vae
    to = model.transport_op
    # mode_ids = list(range(to.n_dynamics)) if modes is None else list(modes)
    if psi_indices is None:
        psi_ids_1b = list(range(1, to.n_dynamics + 1))
    else:
        psi_ids_1b = list(psi_indices)
        if len(psi_ids_1b) == 0:
            raise ValueError("psi_indices cannot be empty.")
    # dedup + cast
    psi_ids_1b = sorted(set(int(i) for i in psi_ids_1b))
    # range check (1-based)
    for i in psi_ids_1b:
        if i < 1 or i > to.n_dynamics:
            raise ValueError(
                f"Invalid psi index {i}; valid range is [1, {to.n_dynamics}]"
            )
    # convert ONCE to 0-based
    psi_ids_0b = [i - 1 for i in psi_ids_1b]

    # Pre-compute loop sizes for progress
    n_batches_main = math.ceil(z_all.size(0) / batch_size)
    n_batches_speed = math.ceil(z_all.size(0) / (batch_size*4))
    total_iters = 0
    # speed pass counts per batch per mode
    total_iters += n_batches_speed * len(psi_ids_0b)
    # gate-quantile pass counts per batch (no per-mode loop)
    if tau_quantile is not None:
        total_iters += math.ceil(z_all.size(0) / max(1024, batch_size*4))
    # main pass counts per batch per mode
    total_iters += n_batches_main * len(psi_ids_0b)

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=total_iters, desc="Directional gene scoring", leave=True)

    # ---------- Phase A: per-mode median speed ||v_m|| ----------
    speed_medians = {}
    with torch.no_grad():
        for start in range(0, z_all.size(0), batch_size*4):
            z = z_all[start:start+batch_size*4].to(device_t, non_blocking=True)
            for m in psi_ids_0b:
                v = z @ to.Psi[m]
                vv = torch.linalg.vector_norm(v, dim=1)
                if m not in speed_medians:
                    speed_medians[m] = []
                speed_medians[m].append(vv.detach().cpu())
                if pbar is not None: pbar.update(1)
        for m in psi_ids_0b:
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
    scores = {m: torch.zeros(G, device=device_t) for m in psi_ids_0b}
    sum_w = {m: 0.0 for m in psi_ids_0b}
    sum_w2 = {m: 0.0 for m in psi_ids_0b}
    used_cells = {m: 0 for m in psi_ids_0b}
    pos_mass = {m: 0.0 for m in psi_ids_0b}
    neg_mass = {m: 0.0 for m in psi_ids_0b}

    autocast_cm = torch.autocast(device_type=(device_t.type if device_t.type != "mps" else "cpu"),
                                 dtype=autocast_dtype, enabled=use_autocast)
    # ---------- Phase C: main JVP loop ----------
    for start in range(0, z_all.size(0), batch_size):
        z = z_all[start:start+batch_size].to(device_t, non_blocking=True)
        z.requires_grad_(True)
        with torch.set_grad_enabled(True), autocast_cm:
            u = to.unit_directions(z)
            gates = to.get_mixture_weights(u)  # (B,M)
            if clip_gate_pct is not None:
                g_hi = torch.quantile(gates, clip_gate_pct, dim=0, keepdim=True)
                gates = torch.minimum(gates, g_hi)

            for m in psi_ids_0b:
                c = gates[:, m]
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
                    w = w * density_weights[start:start+batch_size][keep].to(device_t)

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
    for m in psi_ids_0b:
        if sum_w[m] <= 0:
            out_scores[m] = scores[m].detach()

        s = scores[m] / sum_w[m]
        scale = s.abs().median().item() + 1e-8
        s = s / scale
        out_scores[m] = s.detach()

    gene_names = adata_vapor.uns.get("gene_names", None)
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(G)]
    
    adata_vapor.uns.setdefault(store_key, {})
    for m, s in out_scores.items():
        adata_vapor.uns[store_key][f"psi{m+1}"] = {
            "scores": s.detach().cpu().numpy(),
            # "gene_names": gene_names,
        }
    return adata_vapor

import numpy as np
import pandas as pd

def _select_top_genes(scores, gene_names, top_n=250, select="pos"):
    """
    scores: (n_genes,) numpy
    select: "pos" | "neg" | "both"
    """
    scores = np.asarray(scores, dtype=float)
    gene_names = np.asarray(gene_names)

    out = {}

    if select in ("pos", "both"):
        idx = np.argsort(-scores)
        idx = idx[scores[idx] > 0]
        out["pos"] = gene_names[idx[:top_n]].tolist()

    if select in ("neg", "both"):
        idx = np.argsort(scores)
        idx = idx[scores[idx] < 0]
        out["neg"] = gene_names[idx[:top_n]].tolist()

    return out  # {"pos": [...], "neg": [...]}

def _run_enrichment_internal(gene_list, gene_sets, organism="Human", cutoff=0.05):

    # safety: ensure python list[str]
    gene_list = [str(g) for g in gene_list if g is not None and str(g) != "nan"]
    gene_sets = list(gene_sets) if isinstance(gene_sets, (list, tuple)) else [gene_sets]

    import gseapy as gp

    # step A: verify library exists (according to gseapy)
    libs = gp.get_library_name(organism=organism)

    # step B: try one by one (so we know which one fails)
    enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism=organism,
            outdir=None,
            no_plot=True,
            cutoff=0.05
        )
        
    return enr.results

def run_enrichment(
    adata_VAPOR,
    psi_indices=None,                 # 1-based; None => all
    gene_names=None,                  # default from adata_VAPOR.uns
    store_key="vapor_directional_gene_scores",
    top_n=250,
    select="pos",                     # "pos" | "neg" | "both"
    organism="Mouse",
    gene_sets=(
        "GO_Biological_Process_2025",
        "GO_Cellular_Component_2025",
    ),
):
    """
    Returns:
        dict[str -> DataFrame], key like:
        - psi1_pos
        - psi2_neg
    """

    # ---------- gene names ----------
    if gene_names is None:
        gene_names = adata_VAPOR.uns.get("gene_names", None)
    if gene_names is None:
        raise ValueError("gene_names not provided and adata_VAPOR.uns['gene_names'] missing.")

    # ---------- fetch gene scores ----------
    scores_store = adata_VAPOR.uns.get(store_key, None)
    if scores_store is None:
        raise ValueError(f"adata_VAPOR.uns['{store_key}'] not found.")

    # ---------- psi indices (1-based) ----------
    # infer number of dynamics
    psi_keys = sorted([k for k in scores_store if k.startswith("psi")])
    n_dyn = len(psi_keys)

    if psi_indices is None:
        psi_ids_1b = list(range(1, n_dyn + 1))
    else:
        psi_ids_1b = sorted(set(int(i) for i in psi_indices))
        for i in psi_ids_1b:
            if i < 1 or i > n_dyn:
                raise ValueError(f"Invalid psi index {i}; valid range [1, {n_dyn}]")

    results = {}

    # ---------- main loop ----------
    for psi1 in psi_ids_1b:
        psi_key = f"psi{psi1}"
        item = scores_store.get(psi_key, None)

        if item is None:
            print(f"[WARN] Missing scores for {psi_key}, skipping.")
            continue

        # support either raw array or {"scores": ...}
        scores = item["scores"] if isinstance(item, dict) else item

        picks = _select_top_genes(
            scores,
            gene_names,
            top_n=top_n,
            select=select,
        )

        for direction, genes in picks.items():
            if len(genes) < 10:
                print(f"[WARN] {psi_key}_{direction}: too few genes ({len(genes)}), skip.")
                continue

            key = f"{psi_key}_{direction}"
            print(f"Running enrichment for {key} ({len(genes)} genes)")


            try:
                result = _run_enrichment_internal(
                    gene_list=genes,
                    gene_sets=gene_sets,
                    organism=organism,
                )
                if result is None or len(result) == 0:
                    continue
            
                key = f"psi{psi1}_{direction}"   
            
                df = result.copy()
                df["psi"] = int(psi1)
                df["direction"] = str(direction)
            
                results[key] = df
            
            except Exception as e:
                print(f"Enrichment failed for psi{psi1}_{direction}: {e}")

    return results


import pandas as pd
from collections import defaultdict

def _pick_score(df):
    for col in ['Adjusted P-value','Adjusted P-value (FDR)','FDR p-value','FDR']:
        if col in df.columns:
            return -np.log10(df[col].astype(float).clip(lower=1e-300))
    for col in ['P-value','P value','P']:
        if col in df.columns:
            return -np.log10(df[col].astype(float).clip(lower=1e-300))
    if 'Combined Score' in df.columns:
        return df['Combined Score'].astype(float)
    raise ValueError('No known score column in enrichment results')
    

def build_heatmap_mats(enrichment_results, top_n=None):
    all_suffixes = set()
    rows_by_suffix = defaultdict(list)

    for key in enrichment_results:
        parts = key.split("_")
        if len(parts) < 2:
            continue
        suffix = "_".join(parts[2:]) if parts[0] == "Dynamic" else "_".join(parts[1:])
        if suffix:
            all_suffixes.add(suffix)

    for key, df in enrichment_results.items():
        if df is None or len(df) == 0:
            continue

        suffix = None
        for sfx in all_suffixes:
            if key.endswith(f"_{sfx}"):
                suffix = sfx
                dyn = key[:-(len(sfx) + 1)]
                break
        if suffix is None:
            continue  # skip unknown suffix

        s = _pick_score(df)
        tmp = pd.DataFrame({
            'Term': df['Term'].str.replace(r'\s*\(GO:\d+\)$','', regex=True),
            'score': s
        })

        if top_n is not None:
            tmp = tmp.sort_values('score', ascending=False).head(top_n)

        tmp = tmp.groupby('Term', as_index=False)['score'].max()
        tmp['Dynamic'] = dyn
        rows_by_suffix[suffix].append(tmp)

    mat_by_suffix = {}
    all_terms, all_dyns = set(), set()

    for suffix, rows in rows_by_suffix.items():
        df = pd.concat(rows)
        mat = df.pivot_table(index='Term', columns='Dynamic', values='score', fill_value=0.0)
        mat_by_suffix[suffix] = mat
        all_terms.update(mat.index)
        all_dyns.update(mat.columns)

    all_terms = sorted(all_terms)
    all_dyns = sorted(all_dyns)
    for suffix in mat_by_suffix:
        mat_by_suffix[suffix] = mat_by_suffix[suffix].reindex(index=all_terms, columns=all_dyns, fill_value=0.0)

    return mat_by_suffix

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_heatmap(mat, title, cmap='rocket_r',
                 zscore_rows=False,
                 cluster=False,             
                 row_cluster=True, col_cluster=True,
                 metric='correlation', method='average',
                 vmin=None, vmax=None, center=None,
                 figsize=(8, 10), save=None):
    M = mat.copy()
    M = M.apply(pd.to_numeric, errors='coerce')

    M = M.replace([np.inf, -np.inf], np.nan)
    row_keep = M.notna().any(axis=1) & (M.var(axis=1, skipna=True) > 1e-12)
    col_keep = M.notna().any(axis=0) & (M.var(axis=0, skipna=True) > 1e-12)
    M = M.loc[row_keep, col_keep]

    if M.shape[0] < 2 or M.shape[1] < 2:
        print(f"[plot_heatmap_flexible] too small after cleaning: {M.shape}")
        return

    if zscore_rows:
        M = (M.sub(M.mean(axis=1), axis=0)
               .div(M.std(axis=1).replace(0, np.nan), axis=0)).fillna(0.0)

    if not cluster:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows=1, ncols=15, figure=fig)
        ax  = fig.add_subplot(gs[:, :14])
        cax = fig.add_subplot(gs[:, 14])

        sns.heatmap(M, ax=ax, cbar=True, cbar_ax=cax,
                    cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                    linewidths=0)
        cax.set_ylabel(title)
        ax.set_xlabel('Dynamic'); ax.set_ylabel('Term')
        ax.set_title(title, pad=12)
        ax.tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Saved: {save}")
        plt.show()
        return

    try:
        g = sns.clustermap(M, cmap=cmap,
                           metric=metric, method=method,
                           row_cluster=row_cluster, col_cluster=col_cluster,
                           figsize=figsize,
                           vmin=vmin, vmax=vmax, center=center,
                           cbar_pos=(0.02, 0.8, 0.02, 0.15),
                           cbar_kws={'label': title})
        g.fig.suptitle(title, y=1.02)
        g.tick_params(axis='x', labelrotation=90)
    except ValueError as e:
        print(f"[plot_heatmap_flexible] correlation failed ({e}); falling back to euclidean.")
        g = sns.clustermap(M, cmap=cmap,
                           metric='euclidean', method=method,
                           row_cluster=row_cluster, col_cluster=col_cluster,
                           figsize=figsize,
                           vmin=vmin, vmax=vmax, center=center,
                           cbar_pos=(0.02, 0.8, 0.02, 0.15),
                           cbar_kws={'label': title})
        g.fig.suptitle(title, y=1.02)

    if save:
        g.fig.savefig(save, dpi=300, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.tight_layout()
    plt.show()