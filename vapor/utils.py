import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List

from .config import VAPORConfig

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
    
    transport_op = TransportOperator(
        latent_dim=config.latent_dim,
        n_dynamics=config.n_dynamics
    )
    
    model = VAPOR(vae=vae, transport_op=transport_op)
    return model