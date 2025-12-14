#!/usr/bin/env python3
"""
VAPOR parameter grid search with checkpointing.
Uses flexible root/terminal selection via:
  --root_where COLUMN=VALUE (repeatable, AND)
  --terminal_where COLUMN=VALUE (repeatable, AND)

Example:
  python grid_search.py \
    --adata_file ./data/pasca_development_hvg5k_scaled.h5ad \
    --param_grid_json grids/grid1.json \
    --root_where celltype=Early\\ RG --root_where Age=pcw16 --root_n 200 \
    --terminal_where celltype=Glutamatergic --terminal_where Age=pcw24 --terminal_n 200 \
    --seed 42 \
    --total_epochs 1000 --ckpt_every 250 \
    --batch_size 512
"""

import os
import json
import itertools
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import torch
import scanpy as sc

import vapor

# Optional: only needed if you want to save indices/selection meta for reproducibility
try:
    from vapor.dataset import select_obs_indices
except Exception:
    select_obs_indices = None


# -------------------------
# Grid
# -------------------------

def load_param_grid(path: Path) -> Dict[str, List[Any]]:
    with open(path, "r") as f:
        grid = json.load(f)
    if not isinstance(grid, dict) or not grid:
        raise ValueError("param_grid_json must be a non-empty dict of lists.")
    for k, v in grid.items():
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"Grid entry '{k}' must be a non-empty list.")
    return grid


def generate_experiment_configs(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    experiments: List[Dict[str, Any]] = []
    for i, combo in enumerate(itertools.product(*values), start=1):
        cfg = dict(zip(keys, combo))

        # stable name
        parts = []
        for k in keys:
            v = cfg[k]
            if isinstance(v, float) and k in ("lr"):
                parts.append(f"{k}{v:.0e}")
            else:
                parts.append(f"{k}{v}")
        cfg["exp_name"] = "_".join(parts)
        cfg["exp_id"] = i
        experiments.append(cfg)
    return experiments


# -------------------------
# Checkpoint schedule
# -------------------------

def compute_checkpoint_epochs(
    total_epochs: int,
    ckpt_every: Optional[int],
    ckpt_epochs: Optional[List[int]],
) -> List[int]:
    if ckpt_epochs and len(ckpt_epochs) > 0:
        xs = sorted(set(int(x) for x in ckpt_epochs))
        xs = [x for x in xs if 1 <= x <= total_epochs]
        if total_epochs not in xs:
            xs.append(total_epochs)
        return xs

    if not ckpt_every or ckpt_every <= 0:
        return [total_epochs]

    xs = list(range(ckpt_every, total_epochs + 1, ckpt_every))
    if total_epochs not in xs:
        xs.append(total_epochs)
    if 5 < total_epochs and 5 not in xs:
        xs = [5] + xs
    return sorted(set(xs))


# -------------------------
# Train segments
# -------------------------

def save_checkpoint_with_params(model, config, epoch: int, save_dir: Path, exp_name: str) -> Path:
    ckpt_name = f"{exp_name}_epoch{epoch}.pt"
    ckpt_path = save_dir / ckpt_name
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": vars(config),
            "exp_name": exp_name,
            "timestamp": datetime.now().isoformat(),
        },
        ckpt_path,
    )
    print(f"  ✓ saved checkpoint: {ckpt_name}")
    return ckpt_path


def train_in_segments(
    model,
    dataset,
    config,
    checkpoint_epochs: List[int],
    exp_name: str,
    save_dir: Path,
):
    current = 0
    for target in checkpoint_epochs:
        n = target - current
        if n <= 0:
            continue

        print(f"\n  Training epochs {current+1}..{target} ({n} epochs)")
        seg_cfg = vapor.create_config(**vars(config))
        seg_cfg.epochs = n
        seg_cfg.plot_losses = False

        result = vapor.train_model(model, dataset, seg_cfg)
        if isinstance(result, tuple):
            model, _metrics = result
        else:
            model = result

        save_checkpoint_with_params(model, config, target, save_dir, exp_name)
        current = target

    return model


# -------------------------
# Experiment runner
# -------------------------

def is_completed(exp_dir: Path, exp_name: str) -> bool:
    p = exp_dir / f"{exp_name}_summary.json"
    if not p.exists():
        return False
    try:
        with open(p, "r") as f:
            s = json.load(f)
        return s.get("status") == "completed"
    except Exception:
        return False


def run_single_experiment(
    exp_config: Dict[str, Any],
    dataset,
    n_genes: int,
    results_dir: Path,
    base_train_overrides: Dict[str, Any],
    checkpoint_epochs: List[int],
    device: str,
    skip_completed: bool,
) -> bool:
    exp_name = exp_config["exp_name"]
    exp_id = exp_config["exp_id"]

    exp_dir = results_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if skip_completed and is_completed(exp_dir, exp_name):
        print(f"\n[SKIP] {exp_id}: {exp_name} (completed)")
        return True

    print(f"\n{'='*90}")
    print(f"Experiment {exp_id}: {exp_name}")
    print("Grid params:", {k: v for k, v in exp_config.items() if k not in ("exp_name", "exp_id")})
    print(f"Checkpoint epochs: {checkpoint_epochs}")
    print(f"{'='*90}")

    try:
        # merge base overrides + grid params
        cfg_kwargs = dict(base_train_overrides)
        cfg_kwargs.update({k: v for k, v in exp_config.items() if k not in ("exp_name", "exp_id")})
        cfg_kwargs["device"] = device
        cfg_kwargs.setdefault("epochs", checkpoint_epochs[-1])

        config = vapor.create_config(**cfg_kwargs)
        config.plot_losses = False

        with open(exp_dir / f"{exp_name}_config.json", "w") as f:
            json.dump(vars(config), f, indent=2, default=str)

        model = vapor.initialize_model(n_genes)

        t0 = time.time()
        _ = train_in_segments(model, dataset, config, checkpoint_epochs, exp_name, exp_dir)
        t1 = time.time()

        with open(exp_dir / f"{exp_name}_summary.json", "w") as f:
            json.dump(
                {
                    "exp_name": exp_name,
                    "exp_id": exp_id,
                    "grid_params": {k: v for k, v in exp_config.items() if k not in ("exp_name", "exp_id")},
                    "base_train_overrides": base_train_overrides,
                    "checkpoint_epochs": checkpoint_epochs,
                    "training_time_hours": (t1 - t0) / 3600,
                    "status": "completed",
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
                default=str,
            )

        print(f"\n  Completed: {exp_name} ({(t1-t0)/3600:.2f} hours)")
        return True

    except Exception as e:
        print(f"\n  Failed: {exp_name}\n  Error: {e}")
        with open(exp_dir / f"{exp_name}_error.json", "w") as f:
            json.dump(
                {"exp_name": exp_name, "error": str(e), "timestamp": datetime.now().isoformat()},
                f,
                indent=2,
            )
        return False


# -------------------------
# CLI
# -------------------------

def build_parser():
    import argparse

    p = argparse.ArgumentParser("VAPOR grid search")

    # Data / output
    p.add_argument("--adata_file", type=str, required=True)
    p.add_argument("--results_dir", type=str, default=None)

    # Grid
    p.add_argument("--param_grid_json", type=str, required=True)
    p.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=True)

    # root/terminal where (AND)
    p.add_argument("--root_where", action="append", default=[], help="Root filter COLUMN=VALUE (repeat for AND)")
    p.add_argument("--terminal_where", action="append", default=[], help="Terminal filter COLUMN=VALUE (repeat for AND)")
    p.add_argument("--root_n", type=int, default=200)
    p.add_argument("--terminal_n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)

    # dataset
    p.add_argument("--scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--time_label", type=str, default=None)

    # training shared overrides
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", type=str, default=None)

    # epochs / checkpointing
    p.add_argument("--total_epochs", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--ckpt_epochs", type=int, nargs="*", default=None)

    # optional fixed params
    p.add_argument("--latent_dim", type=int, default=None)
    p.add_argument("--n_dynamics", type=int, default=None)

    return p


def main():
    args = build_parser().parse_args()

    # results dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) if args.results_dir else Path(f"./grid_search_{ts}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {results_dir}")

    # load adata
    if not os.path.exists(args.adata_file):
        raise FileNotFoundError(args.adata_file)

    print("Loading adata...")
    adata = sc.read_h5ad(args.adata_file)
    print(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # (Recommended) save selection rules + indices for reproducibility
    if select_obs_indices is not None:
        indices_dir = results_dir / "indices"
        indices_dir.mkdir(exist_ok=True)

        root_idx, root_parsed, root_matched = select_obs_indices(
            adata, args.root_where, n=args.root_n, seed=args.seed, return_names=True
        )
        term_idx, term_parsed, term_matched = select_obs_indices(
            adata, args.terminal_where, n=args.terminal_n, seed=args.seed, return_names=True
        )
        root_idx.to_series().to_csv(indices_dir / "root_indices.txt", index=False, header=False)
        term_idx.to_series().to_csv(indices_dir / "terminal_indices.txt", index=False, header=False)
        with open(indices_dir / "selection_rules.json", "w") as f:
            json.dump(
                {
                    "adata_file": args.adata_file,
                    "root": {"where": root_parsed, "n": args.root_n, "seed": args.seed, "matched": root_matched},
                    "terminal": {"where": term_parsed, "n": args.terminal_n, "seed": args.seed, "matched": term_matched},
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
                default=str,
            )

    # build dataset (selection happens inside dataset_from_adata)
    dataset = vapor.dataset_from_adata(
        adata,
        time_label=args.time_label,
        root_where=args.root_where,
        terminal_where=args.terminal_where,
        root_n=args.root_n,
        terminal_n=args.terminal_n,
        seed=args.seed,
        scale=bool(args.scale),
    )

    # grid
    param_grid = load_param_grid(Path(args.param_grid_json))
    experiments = generate_experiment_configs(param_grid)
    with open(results_dir / "experiment_plan.json", "w") as f:
        json.dump(experiments, f, indent=2, default=str)
    print(f"Generated {len(experiments)} experiments")

    # checkpoints
    checkpoint_epochs = compute_checkpoint_epochs(
        total_epochs=int(args.total_epochs),
        ckpt_every=int(args.ckpt_every) if args.ckpt_every is not None else None,
        ckpt_epochs=list(args.ckpt_epochs) if args.ckpt_epochs else None,
    )

    # base overrides
    base_train_overrides: Dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "epochs": int(args.total_epochs),
    }
    if args.latent_dim is not None:
        base_train_overrides["latent_dim"] = int(args.latent_dim)
    if args.n_dynamics is not None:
        base_train_overrides["n_dynamics"] = int(args.n_dynamics)

    # run
    completed = 0
    failed = 0

    for i, exp_cfg in enumerate(experiments, start=1):
        print(f"\nProgress: {i}/{len(experiments)}")
        ok = run_single_experiment(
            exp_config=exp_cfg,
            dataset=dataset,
            n_genes=adata.shape[1],
            results_dir=results_dir,
            base_train_overrides=base_train_overrides,
            checkpoint_epochs=checkpoint_epochs,
            device=device,
            skip_completed=bool(args.skip_completed),
        )
        if ok:
            completed += 1
        else:
            failed += 1

        with open(results_dir / "progress.json", "w") as f:
            json.dump(
                {
                    "total_experiments": len(experiments),
                    "completed": completed,
                    "failed": failed,
                    "current_experiment": i,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(experiments)}  Completed: {completed}  Failed: {failed}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()