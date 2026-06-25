# vapor/dataset.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Tuple, Union, Any


# =========================
# Selection helpers (root/terminal)
# =========================

def select_obs_indices(
    adata: Any,
    where: Dict[str, str],
    n: Optional[int] = None,
    seed: int = 0,
    return_names: bool = True,
) -> Tuple[Union[pd.Index, List[int]], int]:
    """Select cells from adata.obs by AND-ing column conditions.

    Args:
      adata: AnnData object
      where: dict of {column: value} conditions (AND-ed)
      n: sample size; if None, return all matched
      seed: random seed for sampling
      return_names: if True, return obs_names; else return integer positions

    Returns:
      selected: pd.Index of obs_names (if return_names) else List[int] positions
      matched_count: number matched before sampling
    """
    obs = adata.obs
    mask = pd.Series(True, index=obs.index)
    for col, val in where.items():
        if col not in obs.columns:
            raise KeyError(f"Column '{col}' not found in adata.obs.")
        mask &= (obs[col].astype(str) == str(val))

    matched = int(mask.sum())
    if matched == 0:
        raise ValueError(f"No cells match conditions: {where}")

    matched_names = obs.index[mask]

    if n is not None:
        n = min(int(n), matched)
        matched_names = obs.loc[matched_names].sample(n=n, random_state=seed).index

    if return_names:
        return pd.Index(matched_names), matched

    positions = [adata.obs_names.get_loc(name) for name in matched_names]
    return positions, matched


def dataset_from_adata(
    adata,
    *,
    root_indices=None,
    terminal_indices=None,
    scale=True,
):
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    # row-wise z-score scaling (optional)
    if scale:
        row_means = X.mean(axis=1, keepdims=True)
        row_stds = X.std(axis=1, keepdims=True, ddof=0)
        row_stds[row_stds == 0] = 1.0
        X = (X - row_means) / row_stds
        print("Data scaled per row.")

    dataset = AnnDataDataset(
        X,
        obs_names=adata.obs_names,
        root_indices=root_indices,
        terminal_indices=terminal_indices,
    )
    return dataset

class AnnDataDataset(Dataset):
    def __init__(
        self,
        X,
        obs_names=None,
        root_indices=None,
        terminal_indices=None,
    ):
        self.data = torch.tensor(X, dtype=torch.float32)

        # root / terminal indices: can be int indices or obs_names strings
        self.root_indices = self._normalize_indices(root_indices, obs_names, "root_indices")
        self.terminal_indices = self._normalize_indices(terminal_indices, obs_names, "terminal_indices")
        self.output_fields = ["x", "is_root", "is_terminal"]

    @staticmethod
    def _normalize_indices(indices, obs_names, field_name: str):
        if indices is None:
            return set()
        if len(indices) == 0:
            return set()

        # strings => map from obs_names
        if isinstance(indices[0], str):
            if obs_names is None:
                raise ValueError(f"obs_names must be provided when {field_name} are cell names.")
            # obs_names is pandas Index; get_loc works
            return set([obs_names.get_loc(name) for name in indices])
        # ints
        return set(indices)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        is_root = idx in self.root_indices
        is_terminal = idx in self.terminal_indices
        
        out = [x, is_root, is_terminal]
        return tuple(out)
