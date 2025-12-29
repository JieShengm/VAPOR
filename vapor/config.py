from __future__ import annotations

from dataclasses import dataclass, field, fields, MISSING
from typing import Optional, List, Dict, Any, get_origin, get_args, Union
import argparse


# =========================
# Config
# =========================

@dataclass
class VAPORConfig:
    """VAPOR configuration (single source of truth)."""

    # Data
    # adata_file: str = "./data/pasca_development_hvg5k_scaled.h5ad"
    # save_path: str = "./out/vapor.pth"
    time_label: Optional[str] = None
    root_indices: Optional[List[int]] = None
    terminal_indices: Optional[List[int]] = None
    scale: bool = True

    # Graph / path (TCL)
    eps_z_k: int = 30          # estimate eps_z using the k-th nearest neighbor
    graph_k: int = 50          # maximum number of neighbors kept per node (top-k)
    min_samples: int = 5       # minimum number of neighbors required; otherwise treated as noise

    # Spatial (optional)
    spatial_key: Optional[str] = None   # key in adata.obsm (e.g. "spatial"); None => non-spatial mode
    batch_key: Optional[str] = None     # key in adata.obs for grouping spatial coords (e.g. "batch"); None => all data together
    by_batch:  bool = True          # whether to group by batch_key during training
    eps_xy: Optional[float] = None      # spatial radius in z-scored coord space; None => adaptive per-node radius
    zscore_mode: str = "batch"         # "global" | "batch"
    expand_factor: float = 1.25         # adaptive spatial radius multiplier
    eps_cap: Optional[float] = 3.0      # cap adaptive eps_xy_i (z-scored units)

    # Model
    latent_dim: int = 64
    n_dynamics: int = 10
    encoder_dims: List[int] = field(default_factory=lambda: [2048, 512, 128])
    decoder_dims: List[int] = field(default_factory=lambda: [128, 512, 2048])

    # Training
    total_steps: int = 1750
    batch_size: int = 512
    lr: float = 3e-4
    device: Optional[str] = "cuda"

    # Loss weights
    beta: float = 0.01
    alpha: float = 1.0
    gamma: float = 1.0
    eta: float = 1.0
    tau: float = 0.5

    # Training options
    t_max: int = 5
    prune: bool = False
    grad_clip: float = 0.2
    print_freq: int = 1
    plot_losses: bool = True

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown config field: {k}")
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VAPORConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# =========================
# Argparse auto-builder
# =========================

def _is_optional(t) -> bool:
    origin = get_origin(t)
    if origin is Union:
        return type(None) in get_args(t)
    return False

def _strip_optional(t):
    if not _is_optional(t):
        return t
    return next(x for x in get_args(t) if x is not type(None))

def _is_list(t) -> bool:
    origin = get_origin(t)
    return origin in (list, List)

def _list_elem_type(t):
    return get_args(t)[0] if get_args(t) else str

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("VAPOR training")
    cfg = VAPORConfig()

    for f in fields(VAPORConfig):
        name = f.name
        ann = f.type
        default = getattr(cfg, name)

        # Determine base type (strip Optional)
        base = _strip_optional(ann)

        arg = f"--{name}"

        # bool -> --flag / --no-flag (py3.9+)
        if base is bool:
            parser.add_argument(
                arg,
                action=argparse.BooleanOptionalAction,
                default=default,
                help=f"(bool) default={default}",
            )
            continue

        # List[T] -> --x a b c (nargs="+")
        if _is_list(base):
            elem_t = _list_elem_type(base)
            parser.add_argument(
                arg,
                type=elem_t,
                nargs="+",
                default=default,
                help=f"(list[{getattr(elem_t, '__name__', str(elem_t))}]) default={default}",
            )
            continue

        # Optional[List[T]] where default None:
        # allow either omitted (None) or provided values.
        if _is_optional(ann) and _is_list(base):
            elem_t = _list_elem_type(base)
            parser.add_argument(
                arg,
                type=elem_t,
                nargs="+",
                default=default,
                help=f"(optional list) default={default}",
            )
            continue

        # Everything else: str/int/float
        # If Optional[str] default None, argparse will pass None if omitted.
        parser.add_argument(
            arg,
            type=base if base in (int, float, str) else str,
            default=default,
            help=f"default={default}",
        )

    return parser


def config_from_cli(argv=None) -> VAPORConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    return VAPORConfig(**vars(args))


def default_config() -> VAPORConfig:
    return VAPORConfig()

def create_config(**kwargs):
    """Create config with custom parameters."""
    config = VAPORConfig()
    config.update(**kwargs)
    return config
