# vapor/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union, Any
from sklearn.preprocessing import MinMaxScaler


# =========================
# Selection helpers (root/terminal)
# =========================

WhereClause = Optional[List[str]]  # e.g. ["celltype=Early RG", "Age=pcw16"]

def _parse_where(where: WhereClause) -> Dict[str, str]:
    """
    Parse ["col=val", "col2=val2"] -> {"col":"val","col2":"val2"}.
    Multiple clauses are AND-ed.
    """
    where = where or []
    out: Dict[str, str] = {}
    for item in where:
        if "=" not in item:
            raise ValueError(f"Invalid where clause '{item}'. Use COLUMN=VALUE.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or v == "":
            raise ValueError(f"Invalid where clause '{item}'. Use COLUMN=VALUE with non-empty parts.")
        # last one wins if repeated column
        out[k] = v
    return out


def select_obs_indices(
    adata: Any,
    where: WhereClause,
    n: Optional[int] = None,
    seed: int = 0,
    return_names: bool = True,
) -> Tuple[Union[pd.Index, List[int]], Dict[str, str], int]:
    """
    Select cells from adata.obs by AND-ing conditions like ["col=val", "col2=val2"].

    Args:
      adata: AnnData-like object with .obs (DataFrame) and .obs_names (Index)
      where: list[str] of COLUMN=VALUE, AND semantics
      n: sample size; if None, return all matched
      seed: random seed for sampling
      return_names: if True, return obs_names (pd.Index); else return integer positions (List[int])

    Returns:
      selected: pd.Index of obs_names (if return_names) else List[int] positions
      parsed_where: dict of conditions
      matched_count: matched before sampling
    """
    parsed = _parse_where(where)
    obs = adata.obs

    mask = pd.Series(True, index=obs.index)
    for col, val in parsed.items():
        if col not in obs.columns:
            raise KeyError(f"Column '{col}' not found in adata.obs.")
        mask &= (obs[col].astype(str) == str(val))

    matched = int(mask.sum())
    if matched == 0:
        raise ValueError(f"No cells match conditions: {parsed}")

    matched_names = obs.index[mask]

    if n is not None:
        n = min(int(n), matched)
        matched_names = obs.loc[matched_names].sample(n=n, random_state=seed).index

    if return_names:
        return pd.Index(matched_names), parsed, matched

    positions = [adata.obs_names.get_loc(name) for name in matched_names]
    return positions, parsed, matched


# vapor/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

def dataset_from_adata(
    adata,
    *,
    # time_label=None,
    root_indices=None,
    terminal_indices=None,
    scale=True,
    # spatial_key=None,     # e.g. "spatial" in adata.obsm
    # batch_key=None,       # e.g. "batch" in adata.obs
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
        # time_labels=time_labels,
        root_indices=root_indices,
        terminal_indices=terminal_indices,
        # spatial=spatial,
        # spatial_mean=spatial_mean,
        # spatial_std=spatial_std,
        # batch_ids=batch_ids,
    )
    return dataset

class AnnDataDataset(Dataset):
    def __init__(
        self,
        X,
        obs_names=None,
        # time_labels=None,
        root_indices=None,
        terminal_indices=None,
        # spatial=None,
        # spatial_mean=None,
        # spatial_std=None,
        # batch_ids=None,
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
