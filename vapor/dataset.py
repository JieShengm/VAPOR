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


# =========================
# Dataset factory
# =========================

def dataset_from_adata(
    adata,
    *,
    time_label: Optional[str] = None,
    root_indices: Optional[Union[List[Union[int, str]], pd.Index, np.ndarray]] = None,
    terminal_indices: Optional[Union[List[Union[int, str]], pd.Index, np.ndarray]] = None,
    # New (more flexible) selection interface:
    root_where: WhereClause = None,
    terminal_where: WhereClause = None,
    root_n: int = 200,
    terminal_n: int = 200,
    seed: int = 0,
    scale: bool = True,
    spatial_key: Optional[str] = None,
):
    """
    Build an AnnDataDataset from an AnnData object.

    You can specify root/terminal cells either by:
      - root_indices / terminal_indices (names or integer positions), OR
      - root_where / terminal_where (list of "COLUMN=VALUE" AND-ed), plus root_n/terminal_n + seed.

    If indices are provided explicitly, they take precedence over where-clauses.
    """
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Ensure ndarray
    X = np.asarray(X)

    # Time labels
    if time_label is not None:
        raw = adata.obs[time_label].values
        try:
            raw = raw.astype(float)
        except Exception as e:
            raise ValueError(
                f"time_label column '{time_label}' could not be converted to float. "
                f"Got dtype={getattr(raw, 'dtype', None)}."
            ) from e

        scaler = MinMaxScaler(feature_range=(0, 1))
        time_labels = scaler.fit_transform(raw.reshape(-1, 1)).flatten()
        print(f"Time Range: ({np.min(time_labels):.3f}, {np.max(time_labels):.3f})")
    else:
        time_labels = None
        print("No time label provided; proceeding without time supervision.")

    # Row-wise scaling (per-cell z-score)
    if scale:
        try:
            row_means = X.mean(axis=1, keepdims=True)
            row_stds = X.std(axis=1, keepdims=True, ddof=0)
            # avoid divide-by-zero
            row_stds = np.where(row_stds == 0, 1.0, row_stds)
            X = (X - row_means) / row_stds
            print("Data scaled per row.")
        except AttributeError as e:
            raise RuntimeError(
                "Scaling is not supported for backed or sparse .X in your current configuration. "
                "Either load the AnnData fully (no backed='r'), "
                "or call with scale=False."
            ) from e

    # If where-clauses provided and explicit indices not given, compute indices (as names).
    if root_indices is None and root_where is not None and len(root_where) > 0:
        root_indices, parsed, matched = select_obs_indices(
            adata, root_where, n=root_n, seed=seed, return_names=True
        )
        print(f"Root selection: matched={matched}, sampled={len(root_indices)}, where={parsed}")

    if terminal_indices is None and terminal_where is not None and len(terminal_where) > 0:
        terminal_indices, parsed, matched = select_obs_indices(
            adata, terminal_where, n=terminal_n, seed=seed, return_names=True
        )
        print(f"Terminal selection: matched={matched}, sampled={len(terminal_indices)}, where={parsed}")

    spatial = None
    spatial_mean = None
    spatial_std = None

    if spatial_key is not None:
        if spatial_key not in adata.obsm:
            raise KeyError(f"{spatial_key} not found in adata.obsm")
        spatial = np.asarray(adata.obsm[spatial_key])

        # global zscore stats（推荐）
        mu = spatial.mean(axis=0, keepdims=True)
        sd = spatial.std(axis=0, keepdims=True, ddof=0)
        sd[sd == 0] = 1.0

        spatial_mean = mu
        spatial_std = sd

    dataset = AnnDataDataset(
        X,
        obs_names=adata.obs_names,
        time_labels=time_labels,
        root_indices=root_indices,
        terminal_indices=terminal_indices,
        spatial=spatial,
        spatial_mean=spatial_mean,
        spatial_std=spatial_std,
    )
    return dataset


# =========================
# Torch Dataset
# =========================

class AnnDataDataset(Dataset):
    """
    Returns:
      sample: (n_genes,) float tensor
      time_label: float tensor scalar (0.0 if no time_label)
      is_root: bool
      is_terminal: bool
    """

    def __init__(
        self,
        X: np.ndarray,
        obs_names: Optional[pd.Index] = None,
        time_labels: Optional[np.ndarray] = None,
        root_indices: Optional[Union[List[Union[int, str]], pd.Index, np.ndarray]] = None,
        terminal_indices: Optional[Union[List[Union[int, str]], pd.Index, np.ndarray]] = None,
        spatial=None, 
        spatial_mean=None, 
        spatial_std=None
    ):
        self.data = torch.tensor(np.asarray(X), dtype=torch.float32)

        self.time_labels = (
            torch.tensor(time_labels, dtype=torch.float32)
            if time_labels is not None
            else None
        )

        self.spatial = torch.tensor(spatial, dtype=torch.float32) if spatial is not None else None
        self.spatial_mean = torch.tensor(spatial_mean, dtype=torch.float32) if spatial_mean is not None else None
        self.spatial_std = torch.tensor(spatial_std, dtype=torch.float32) if spatial_std is not None else None

        # Normalize obs_names
        if obs_names is not None and not isinstance(obs_names, pd.Index):
            obs_names = pd.Index(obs_names)
        self.obs_names = obs_names

        self.root_indices = self._normalize_indices(root_indices, kind="root")
        self.terminal_indices = self._normalize_indices(terminal_indices, kind="terminal")

    def _normalize_indices(
        self,
        indices: Optional[Union[List[Union[int, str]], pd.Index, np.ndarray]],
        kind: str,
    ) -> set:
        """
        Convert indices input into a set of integer positions for fast membership checks.
        Accepts:
          - None
          - list/np array/pd.Index of ints (positions)
          - list/np array/pd.Index of strings (cell names, requires obs_names)
        """
        if indices is None:
            return set()

        # Convert pd.Index / np.ndarray to list
        if isinstance(indices, pd.Index):
            indices_list = indices.tolist()
        elif isinstance(indices, np.ndarray):
            indices_list = indices.tolist()
        else:
            indices_list = list(indices)

        if len(indices_list) == 0:
            return set()

        first = indices_list[0]

        # Names -> positions
        if isinstance(first, str):
            if self.obs_names is None:
                raise ValueError(f"obs_names must be provided when {kind}_indices are cell names.")
            return set(int(self.obs_names.get_loc(name)) for name in indices_list)

        # Positions
        try:
            return set(int(i) for i in indices_list)
        except Exception as e:
            raise ValueError(
                f"Unsupported {kind}_indices type/values. Provide a list of ints (positions) "
                f"or list of str (cell names). Got example element: {type(first)}"
            ) from e

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        time_label = (
            self.time_labels[idx]
            if self.time_labels is not None
            else torch.tensor(0.0, dtype=torch.float32)
        )
        is_root = idx in self.root_indices
        is_terminal = idx in self.terminal_indices

        coords = self.spatial[idx] if self.spatial is not None else None
        if coords is None:
            return sample, time_label, is_root, is_terminal
        return sample, time_label, is_root, is_terminal, coords