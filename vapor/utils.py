from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Subset

from .config import VAPORConfig

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
    
    if config.seed is not None:
        set_seed(seed=int(config.seed), deterministic=config.deterministic)
    
    vae = VAE(
        input_dim=input_dim,
        latent_dim=config.latent_dim,
        encoder_dims=config.encoder_dims,
        decoder_dims=config.decoder_dims
    )
    
    transport_op = TransportOperator(
        latent_dim=config.latent_dim,
        n_dynamics=config.n_dynamics,
        gate_temperature=config.gate_temperature,
        gate_mode=config.gate_mode,
        use_bias=config.use_bias
    )
    
    model = VAPOR(vae=vae, transport_op=transport_op)
    return model


def save_checkpoint(model, config, path: str, extra: dict | None = None):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": vars(config) if config is not None else None,
        "input_dim": model.vae.input_dim,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model=None, map_location="cpu"):
    """Load a VAPOR checkpoint.

    If `model` is None, reconstructs the model from the saved config.
    Returns (model, ckpt_dict).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is None:
        config = VAPORConfig.from_dict(ckpt["config"])
        input_dim = ckpt.get("input_dim")
        if input_dim is None:
            raise ValueError(
                "Checkpoint missing 'input_dim'. Pass a pre-built model instead."
            )
        model = initialize_model(input_dim, config=config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt

import os
import random
import numpy as np

def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for full reproducibility across all randomness sources.
    
    Args:
        seed: int, the seed value
        deterministic: if True, also enable deterministic CUDA operations
                      (slightly slower but fully reproducible)
    
    Notes:
        - Call this BEFORE creating model, optimizer, dataloader, etc.
        - For DataLoader with num_workers > 0, also pass `generator` and 
          `worker_init_fn` (see seed_dataloader below)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    
    # Environment variable for hash seed (affects dict ordering, etc.)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # Force CUDA operations to be deterministic
        # NOTE: This may slow down training slightly
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # PyTorch >= 1.8 deterministic algorithms
        # Note: This may raise errors if non-deterministic ops are used.
        # Comment out if you hit issues.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass  # older PyTorch versions 
        
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True