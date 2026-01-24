import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List

from .config import VAPORConfig

from torch.utils.data import Subset

def get_base_dataset(ds):
    return ds.dataset if isinstance(ds, Subset) else ds

def resolve_device(config):
    requested = getattr(config, "device", None)

    if requested is None:
        return torch.device("cpu")

    requested = requested.lower()

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        else:
            print(
                "[WARN] CUDA requested but not available. "
                "Falling back to CPU."
            )
            return torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device specifier: {requested}")

def init_vae_weights(m):
    """Initialize VAE weights."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def initialize_model(
    input_dim: int,
    config: Optional[VAPORConfig] = None,
    **kwargs
) -> 'VAPOR':
    from .models import VAE, TransportOperator, VAPOR
    
    if config is None:
        config = VAPORConfig()
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown parameter '{key}' ignored")
    
    print(f"Initializing model:")
    print(f"  Input dim: {input_dim}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Encoder dims: {config.encoder_dims}")
    print(f"  Decoder dims: {config.decoder_dims}")
    print(f"  N dynamics: {config.n_dynamics}")
    
    vae = VAE(
        input_dim=input_dim,
        latent_dim=config.latent_dim,
        encoder_dims=config.encoder_dims,
        decoder_dims=config.decoder_dims
    )
    
    print(config.tau)
    transport_op = TransportOperator(
        latent_dim=config.latent_dim,
        n_dynamics=config.n_dynamics,
        tau=config.tau
    )
    
    model = VAPOR(vae=vae, transport_op=transport_op)
    return model

import torch
from pathlib import Path

def save_checkpoint(model, config, path: str, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": vars(config) if config is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)

def load_checkpoint(model, path: str, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt

import copy
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

@dataclass
class LRFindResult:
    lrs: List[float]
    losses: List[float]
    best_lr: float
    lr_at_min_loss: float
    lr_at_diverge: float
    diverged: bool
    meta: Dict[str, Any]


@torch.no_grad()
def _unpack_batch(batch):
    """
    Your dataset may return 4/5/6-tuple; we only need x (and maybe time).
    Expected: (x, t, is_root, is_term, [coords], [batch_id])
    """
    if isinstance(batch, (list, tuple)):
        x = batch[0]
    else:
        x = batch
    return x

def plot_lr_finder(res, title="LR Finder", save=None):
    lrs = np.array(res.lrs, dtype=float)
    losses = np.array(res.losses, dtype=float)

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)

    ax.plot(lrs, losses)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Smoothed loss")
    ax.set_title(title)

    lr_min = float(res.lr_at_min_loss)
    lr_rec = float(res.best_lr)

    ax.axvline(lr_min, linestyle="--", linewidth=1.5)
    ax.axvline(lr_rec, linestyle=":", linewidth=2.0)

    ymin, ymax = ax.get_ylim()
    ax.text(lr_min, ymin + 0.05*(ymax-ymin), f"min_loss_lr={lr_min:.2e}",
            rotation=90, va="bottom")
    ax.text(lr_rec, ymin + 0.05*(ymax-ymin), f"recommended={lr_rec:.2e}",
            rotation=90, va="bottom")

    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Saved: {save}")

    return fig

def lr_finder_vapor(
    model,
    dataset,
    beta: float = 0.02,
    batch_size: int = 256,
    num_steps: int = 300,
    lr_min: float = 1e-6,
    lr_max: float = 1e-3,
    device: Optional[str] = None,
    mode: str = "vae",  # "vae" (recommended) or "total" (TODO)
    smooth_beta: float = 0.98,  # EMA smoothing for loss curve
    diverge_factor: float = 5.0,  # consider diverged if loss > diverge_factor * best_smoothed_loss
    grad_clip: Optional[float] = 1.0,
    seed: int = 42,
    save_plot : str = "lr_finder.png",
    verbose: bool = True,
) -> LRFindResult:
    """
    Simple LR finder:
      - Run num_steps mini-batches
      - LR increases log-linearly from lr_min to lr_max
      - Track (smoothed) loss; stop when diverges
      - Recommend base_lr = lr_at_diverge / 5 ~ /10 

    Returns LRFindResult containing curves + suggested lr.
    """

    assert num_steps >= 10, "num_steps too small (>=10)"
    assert lr_min > 0 and lr_max > lr_min

    # ---- device ----
    if device is None:
        device = next(model.parameters()).device
        if isinstance(device, torch.device):
            device = device.type
    if isinstance(device, str):
        if device.startswith("cuda") and not torch.cuda.is_available():
            if verbose:
                print("[LR Finder] CUDA requested but not available; using CPU.")
            device = "cpu"
        device_t = torch.device(device)
    else:
        device_t = device

    # ---- deterministic-ish ----
    torch.manual_seed(seed)
    if device_t.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ---- snapshot model weights ----
    model_state = copy.deepcopy(model.state_dict())
    model = model.to(device_t).train()

    # ---- optimizer (simple: one Adam over all params) ----
    # Use a single optimizer for LR finding; can map result to opt_vae/opt_to later.
    opt = torch.optim.AdamW(model.parameters(), lr=lr_min)

    # ---- loader ----
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(loader)

    # ---- lr schedule (log space) ----
    log_min = math.log10(lr_min)
    log_max = math.log10(lr_max)

    lrs: List[float] = []
    losses: List[float] = []

    best_smoothed = float("inf")
    lr_at_min_loss = lr_min
    lr_at_diverge = lr_max
    diverged = False

    smoothed = None

    for step in range(num_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x = _unpack_batch(batch).to(device_t)

        # set lr for this step
        frac = step / max(1, num_steps - 1)
        lr = 10 ** (log_min + frac * (log_max - log_min))
        for pg in opt.param_groups:
            pg["lr"] = lr

        # ---- forward + loss ----
        # VAE-only: recon + beta * KL
        recon, z, mu, logvar = model.vae(x)
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        # TODO: "total" mode, implement it here
        # if mode == "total":
        #     ... include TO losses ...
        # else:  # "vae"
        #     ...

        # ---- backward ----
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()

        # ---- smooth loss ----
        loss_val = float(loss.detach().cpu().item())
        if smoothed is None:
            smoothed = loss_val
        else:
            smoothed = smooth_beta * smoothed + (1 - smooth_beta) * loss_val

        lrs.append(lr)
        losses.append(smoothed)

        # ---- track best ----
        if smoothed < best_smoothed:
            best_smoothed = smoothed
            lr_at_min_loss = lr

        # ---- diverge check ----
        if smoothed > diverge_factor * best_smoothed:
            diverged = True
            lr_at_diverge = lr
            if verbose:
                print(f"[LR Finder] Diverged at step {step}/{num_steps}, lr={lr:.2e}, smoothed_loss={smoothed:.4g}")
            break

        if verbose and (step % max(1, num_steps // 10) == 0):
            print(f"[LR Finder] step {step:4d} | lr={lr:.2e} | smoothed_loss={smoothed:.4g}")

    # ---- recommend ----
    best_lr = lr_at_diverge / 20.0 if diverged else lr_at_min_loss / 20.0

    # ---- restore model ----
    model.load_state_dict(model_state)

    meta = dict(
        mode=mode,
        beta=beta,
        batch_size=batch_size,
        num_steps=num_steps,
        lr_min=lr_min,
        lr_max=lr_max,
        smooth_beta=smooth_beta,
        diverge_factor=diverge_factor,
        grad_clip=grad_clip,
        device=str(device_t),
    )

    if verbose:
        print("\n[LR Finder] Done.")
        print(f"  lr_at_min_loss = {lr_at_min_loss:.2e}")
        print(f"  lr_at_diverge  = {lr_at_diverge:.2e} (diverged={diverged})")
        print(f"  suggested base_lr ≈ {best_lr:.2e}  (rule: diverge/20)")

    res = LRFindResult(
        lrs=lrs,
        losses=losses,
        best_lr=best_lr,
        lr_at_min_loss=lr_at_min_loss,
        lr_at_diverge=lr_at_diverge,
        diverged=diverged,
        meta=meta,
    )
    plot_lr_finder(res, title="LR Finder (VAE loss)", save=save_plot)
    return res
    