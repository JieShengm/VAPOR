import argparse
import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class VAPORConfig:
    """Simple VAPOR configuration."""
    
    # Data
    adata_file: str = "./data/pasca_development_hvg5k_scaled.h5ad"
    save_path: str = "./out/vapor.pth"
    time_label: Optional[str] = None
    root_indices: Optional[List] = None
    terminal_indices: Optional[List] = None
    scale: bool = True
    
    # Model
    latent_dim: int = 64
    n_dynamics: int = 10
    encoder_dims: List[int] = None
    decoder_dims: List[int] = None
    
    # Training
    epochs: int = 350
    batch_size: int = 512
    lr: float = 1e-4
    vae_lr_factor: float = 0.05
    device: Optional[str] = None
    
    # Loss weights
    beta: float = 0.1         # KL weight
    alpha: float = 1.0        # Trajectory weight  
    gamma: float = 1.0        # Prior weight
    eta: float = 1.0          # Psi weight
    eta_a: float = 0.5          # Psi weight
    supervision_weight: float = 0.3
    
    # Training options
    t_max: int = 3
    prune: bool = False
    grad_clip: float = 1.0
    supervision_strength: str = 'light'
    print_freq: int = 1
    plot_losses: bool = True
    
    def __post_init__(self):
        # Set defaults
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.encoder_dims is None:
            self.encoder_dims = [2048, 512, 128]
        
        if self.decoder_dims is None:
            self.decoder_dims = [128, 512, 2048]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        # Only keep valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)
    
    def update(self, **kwargs):
        """Update config with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}' ignored")


def parse_args():
    """Parse command line arguments and return VAPORConfig."""
    parser = argparse.ArgumentParser(description='VAPOR training')
    
    # Data
    parser.add_argument('--adata_file', type=str, default="./data/pasca_development_hvg5k_scaled.h5ad")
    parser.add_argument('--save_path', type=str, default="./out/vapor.pth")
    parser.add_argument('--time_label', type=str, default=None)
    
    # Model  
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--n_dynamics', type=int, default=10)
    parser.add_argument('--encoder_dims', type=str, default="2048,512,128")
    parser.add_argument('--decoder_dims', type=str, default="128,512,2048")
    
    # Training
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--vae_lr_factor', type=float, default=0.05)
    parser.add_argument('--device', type=str, default=None)
    
    # Loss weights
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--eta_a', type=float, default=0.5)
    parser.add_argument('--supervision_weight', type=float, default=0.3)
    
    # Options
    parser.add_argument('--t_max', type=int, default=3)
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--supervision_strength', type=str, default='light')
    
    args = parser.parse_args()
    
    # Convert string lists to int lists
    if hasattr(args, 'encoder_dims') and isinstance(args.encoder_dims, str):
        args.encoder_dims = [int(x) for x in args.encoder_dims.split(',')]
    if hasattr(args, 'decoder_dims') and isinstance(args.decoder_dims, str):  
        args.decoder_dims = [int(x) for x in args.decoder_dims.split(',')]
    
    # Create config from args
    return VAPORConfig(**vars(args))


# Convenience functions
def get_default_config():
    """Get default configuration."""
    return VAPORConfig()

def create_config(**kwargs):
    """Create config with custom parameters."""
    config = VAPORConfig()
    config.update(**kwargs)
    return config
