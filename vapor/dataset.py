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

# def dataset_from_adata(
#     adata,
#     *,
#     time_label: Optional[str] = None,
#     root_indices: Optional[Union[List[Union[int, str]], pd.Index, np.ndarray]] = None,
#     terminal_indices: Optional[Union[List[Union[int, str]], pd.Index, np.ndarray]] = None,
#     # New (more flexible) selection interface:
#     root_where: WhereClause = None,
#     terminal_where: WhereClause = None,
#     root_n: int = 200,
#     terminal_n: int = 200,
#     seed: int = 0,
#     scale: bool = True,
#     spatial_key: Optional[str] = None,
# ):
#     """
#     Build an AnnDataDataset from an AnnData object.

#     You can specify root/terminal cells either by:
#       - root_indices / terminal_indices (names or integer positions), OR
#       - root_where / terminal_where (list of "COLUMN=VALUE" AND-ed), plus root_n/terminal_n + seed.

#     If indices are provided explicitly, they take precedence over where-clauses.
#     """
#     X = adata.X
#     if hasattr(X, "toarray"):
#         X = X.toarray()

#     # Ensure ndarray
#     X = np.asarray(X)

#     # Time labels
#     if time_label is not None:
#         raw = adata.obs[time_label].values
#         try:
#             raw = raw.astype(float)
#         except Exception as e:
#             raise ValueError(
#                 f"time_label column '{time_label}' could not be converted to float. "
#                 f"Got dtype={getattr(raw, 'dtype', None)}."
#             ) from e

#         scaler = MinMaxScaler(feature_range=(0, 1))
#         time_labels = scaler.fit_transform(raw.reshape(-1, 1)).flatten()
#         print(f"Time Range: ({np.min(time_labels):.3f}, {np.max(time_labels):.3f})")
#     else:
#         time_labels = None
#         print("No time label provided; proceeding without time supervision.")

#     # Row-wise scaling (per-cell z-score)
#     if scale:
#         try:
#             row_means = X.mean(axis=1, keepdims=True)
#             row_stds = X.std(axis=1, keepdims=True, ddof=0)
#             # avoid divide-by-zero
#             row_stds = np.where(row_stds == 0, 1.0, row_stds)
#             X = (X - row_means) / row_stds
#             print("Data scaled per row.")
#         except AttributeError as e:
#             raise RuntimeError(
#                 "Scaling is not supported for backed or sparse .X in your current configuration. "
#                 "Either load the AnnData fully (no backed='r'), "
#                 "or call with scale=False."
#             ) from e

#     # If where-clauses provided and explicit indices not given, compute indices (as names).
#     if root_indices is None and root_where is not None and len(root_where) > 0:
#         root_indices, parsed, matched = select_obs_indices(
#             adata, root_where, n=root_n, seed=seed, return_names=True
#         )
#         print(f"Root selection: matched={matched}, sampled={len(root_indices)}, where={parsed}")

#     if terminal_indices is None and terminal_where is not None and len(terminal_where) > 0:
#         terminal_indices, parsed, matched = select_obs_indices(
#             adata, terminal_where, n=terminal_n, seed=seed, return_names=True
#         )
#         print(f"Terminal selection: matched={matched}, sampled={len(terminal_indices)}, where={parsed}")

#     spatial = None
#     spatial_mean = None
#     spatial_std = None

#     if spatial_key is not None:
#         if spatial_key not in adata.obsm:
#             raise KeyError(f"{spatial_key} not found in adata.obsm")
#         spatial = np.asarray(adata.obsm[spatial_key])

#         # global zscore stats（推荐）
#         mu = spatial.mean(axis=0, keepdims=True)
#         sd = spatial.std(axis=0, keepdims=True, ddof=0)
#         sd[sd == 0] = 1.0

#         spatial_mean = mu
#         spatial_std = sd

#     dataset = AnnDataDataset(
#         X,
#         obs_names=adata.obs_names,
#         time_labels=time_labels,
#         root_indices=root_indices,
#         terminal_indices=terminal_indices,
#         spatial=spatial,
#         spatial_mean=spatial_mean,
#         spatial_std=spatial_std,
#     )
#     return dataset


# # =========================
# # Torch Dataset
# # =========================

# class AnnDataDataset(Dataset):
#     """
#     Returns:
#       sample: (n_genes,) float tensor
#       time_label: float tensor scalar (0.0 if no time_label)
#       is_root: bool
#       is_terminal: bool
#     """

#     def __init__(
#         self,
#         X: np.ndarray,
#         obs_names: Optional[pd.Index] = None,
#         time_labels: Optional[np.ndarray] = None,
#         root_indices: Optional[Union[List[Union[int, str]], pd.Index, np.ndarray]] = None,
#         terminal_indices: Optional[Union[List[Union[int, str]], pd.Index, np.ndarray]] = None,
#         spatial=None, 
#         spatial_mean=None, 
#         spatial_std=None
#     ):
#         self.data = torch.tensor(np.asarray(X), dtype=torch.float32)

#         self.time_labels = (
#             torch.tensor(time_labels, dtype=torch.float32)
#             if time_labels is not None
#             else None
#         )

#         self.spatial = torch.tensor(spatial, dtype=torch.float32) if spatial is not None else None
#         self.spatial_mean = torch.tensor(spatial_mean, dtype=torch.float32) if spatial_mean is not None else None
#         self.spatial_std = torch.tensor(spatial_std, dtype=torch.float32) if spatial_std is not None else None

#         # Normalize obs_names
#         if obs_names is not None and not isinstance(obs_names, pd.Index):
#             obs_names = pd.Index(obs_names)
#         self.obs_names = obs_names

#         self.root_indices = self._normalize_indices(root_indices, kind="root")
#         self.terminal_indices = self._normalize_indices(terminal_indices, kind="terminal")

#     def _normalize_indices(
#         self,
#         indices: Optional[Union[List[Union[int, str]], pd.Index, np.ndarray]],
#         kind: str,
#     ) -> set:
#         """
#         Convert indices input into a set of integer positions for fast membership checks.
#         Accepts:
#           - None
#           - list/np array/pd.Index of ints (positions)
#           - list/np array/pd.Index of strings (cell names, requires obs_names)
#         """
#         if indices is None:
#             return set()

#         # Convert pd.Index / np.ndarray to list
#         if isinstance(indices, pd.Index):
#             indices_list = indices.tolist()
#         elif isinstance(indices, np.ndarray):
#             indices_list = indices.tolist()
#         else:
#             indices_list = list(indices)

#         if len(indices_list) == 0:
#             return set()

#         first = indices_list[0]

#         # Names -> positions
#         if isinstance(first, str):
#             if self.obs_names is None:
#                 raise ValueError(f"obs_names must be provided when {kind}_indices are cell names.")
#             return set(int(self.obs_names.get_loc(name)) for name in indices_list)

#         # Positions
#         try:
#             return set(int(i) for i in indices_list)
#         except Exception as e:
#             raise ValueError(
#                 f"Unsupported {kind}_indices type/values. Provide a list of ints (positions) "
#                 f"or list of str (cell names). Got example element: {type(first)}"
#             ) from e

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, idx: int):
#         sample = self.data[idx]
#         time_label = (
#             self.time_labels[idx]
#             if self.time_labels is not None
#             else torch.tensor(0.0, dtype=torch.float32)
#         )
#         is_root = idx in self.root_indices
#         is_terminal = idx in self.terminal_indices

#         coords = self.spatial[idx] if self.spatial is not None else None
#         if coords is None:
#             return sample, time_label, is_root, is_terminal
#         return sample, time_label, is_root, is_terminal, coords

# vapor/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

def dataset_from_adata(
    adata,
    *,
    time_label=None,
    root_indices=None,
    terminal_indices=None,
    scale=True,
    spatial_key=None,     # e.g. "spatial" in adata.obsm
    batch_key=None,       # e.g. "batch" in adata.obs
):
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    # time label (optional)
    if time_label is not None:
        time_labels = np.asarray(adata.obs[time_label].values)
        scaler = MinMaxScaler(feature_range=(0, 1))
        time_labels = scaler.fit_transform(time_labels.reshape(-1, 1)).flatten()
        print(f"Time Range: ({np.min(time_labels):.3f}, {np.max(time_labels):.3f})")
    else:
        time_labels = None
        print("No time label provided; proceeding without time supervision.")

    # row-wise z-score scaling (optional)
    if scale:
        row_means = X.mean(axis=1, keepdims=True)
        row_stds = X.std(axis=1, keepdims=True, ddof=0)
        row_stds[row_stds == 0] = 1.0
        X = (X - row_means) / row_stds
        print("Data scaled per row.")

    # spatial coords (optional)
    spatial = None
    spatial_mean = None
    spatial_std = None
    if spatial_key is not None:
        if spatial_key not in adata.obsm:
            raise KeyError(f"spatial_key='{spatial_key}' not found in adata.obsm")
        spatial = np.asarray(adata.obsm[spatial_key])
        if spatial.ndim != 2:
            raise ValueError(f"adata.obsm['{spatial_key}'] must be 2D, got shape {spatial.shape}")

        # global stats (useful if you want zscore_mode='global')
        mu = spatial.mean(axis=0, keepdims=True)
        sd = spatial.std(axis=0, keepdims=True, ddof=0)
        sd[sd == 0] = 1.0
        spatial_mean, spatial_std = mu, sd

    # batch_id (optional but recommended for grouped batching)
    batch_ids = None
    if batch_key is not None:
        if batch_key not in adata.obs:
            raise KeyError(f"batch_key='{batch_key}' not found in adata.obs")
        batch_ids = np.asarray(adata.obs[batch_key].astype(str).values)  # keep as string labels

    dataset = AnnDataDataset(
        X,
        obs_names=adata.obs_names,
        time_labels=time_labels,
        root_indices=root_indices,
        terminal_indices=terminal_indices,
        spatial=spatial,
        spatial_mean=spatial_mean,
        spatial_std=spatial_std,
        batch_ids=batch_ids,
    )
    return dataset

class AnnDataDataset(Dataset):
    def __init__(
        self,
        X,
        obs_names=None,
        time_labels=None,
        root_indices=None,
        terminal_indices=None,
        spatial=None,
        spatial_mean=None,
        spatial_std=None,
        batch_ids=None,
    ):
        self.data = torch.tensor(X, dtype=torch.float32)

        self.time_labels = (
            torch.tensor(time_labels, dtype=torch.float32)
            if time_labels is not None
            else None
        )

        # root / terminal indices: can be int indices or obs_names strings
        self.root_indices = self._normalize_indices(root_indices, obs_names, "root_indices")
        self.terminal_indices = self._normalize_indices(terminal_indices, obs_names, "terminal_indices")

        # optional spatial
        self.spatial = torch.tensor(spatial, dtype=torch.float32) if spatial is not None else None
        self.spatial_mean = torch.tensor(spatial_mean, dtype=torch.float32) if spatial_mean is not None else None
        self.spatial_std = torch.tensor(spatial_std, dtype=torch.float32) if spatial_std is not None else None

        # optional batch labels (store as Python list of str)
        self.batch_ids = list(batch_ids) if batch_ids is not None else None
        
         # flags
        self.has_spatial = (self.spatial is not None)
        self.has_batch = (self.batch_ids is not None)

        self.output_fields = ["x", "t", "is_root", "is_terminal"]
        if self.has_spatial:
            self.output_fields.append("coords")
        if self.has_batch:
            self.output_fields.append("batch_id")

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
        t = self.time_labels[idx] if self.time_labels is not None else torch.tensor(0.0, dtype=torch.float32)
        is_root = idx in self.root_indices
        is_terminal = idx in self.terminal_indices
        
        out = [x, t, is_root, is_terminal]

        if self.has_spatial:
            out.append(self.spatial[idx])         # coords tensor

        if self.has_batch:
            out.append(self.batch_ids[idx])       # string (NOT None)

        return tuple(out)

        # batch_id = self.batch_ids[idx] if self.batch_ids is not None else None

        # if self.spatial is None:
        #     # 5-tuple
        #     return x, t, is_root, is_terminal, batch_id

        # coords = self.spatial[idx]
        # # 6-tuple
        # return x, t, is_root, is_terminal, coords, batch_id
    

# vapor/dataset.py 里（或 vapor/samplers.py）
import math
import random
from collections import defaultdict
from torch.utils.data import Sampler, Subset

class GroupedBatchSampler(Sampler):
    """
    Yield batches where each batch contains indices from only ONE group (batch/slice).
    Works with both Dataset and torch.utils.data.Subset.
    Requires underlying dataset to have `batch_ids` (list-like length N).
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False, seed: int = 0):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        # ---- handle Subset ----
        if isinstance(dataset, Subset):
            base = dataset.dataset
            indices = list(dataset.indices)
        else:
            base = dataset
            indices = None

        if getattr(base, "batch_ids", None) is None:
            raise ValueError("GroupedBatchSampler requires base_dataset.batch_ids (pass batch_key to dataset_from_adata).")

        self.dataset = dataset
        self.base = base

        # group -> list of *subset indices* (indices into `dataset`, not base)
        groups = defaultdict(list)

        if indices is None:
            # dataset is the base dataset
            for i, bid in enumerate(base.batch_ids):
                groups[bid].append(i)
        else:
            # dataset is a Subset: i is position in subset, base_i is index in base dataset
            for subset_i, base_i in enumerate(indices):
                bid = base.batch_ids[base_i]
                groups[bid].append(subset_i)

        self.groups = dict(groups)

        # precompute number of batches
        self._num_batches = 0
        for _, idxs in self.groups.items():
            n = len(idxs)
            self._num_batches += (n // self.batch_size) if self.drop_last else math.ceil(n / self.batch_size)

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        rng = random.Random(self.seed)

        group_batches = []
        for bid, idxs in self.groups.items():
            idxs = list(idxs)
            if self.shuffle:
                rng.shuffle(idxs)

            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                group_batches.append(batch)

        if self.shuffle:
            rng.shuffle(group_batches)

        for batch in group_batches:
            yield batch