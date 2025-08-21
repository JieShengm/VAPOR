import torch
from torch.utils.data import Dataset
import numpy as np
import anndata as ad
from sklearn.preprocessing import MinMaxScaler

def dataset_from_adata(
    adata,
    *,
    time_label=None,
    root_indices=None,
    terminal_indices=None,
    scale=True
):
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    if time_label is not None:
        time_labels = adata.obs[time_label].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        time_labels = scaler.fit_transform(time_labels.reshape(-1, 1)).flatten()
        print(f"Time Range: ({np.min(time_labels):.3f}, {np.max(time_labels):.3f})")
    else:
        time_labels = None
        print("No time label provided; proceeding without time supervision.")

    if scale:
        try:
            row_means = X.mean(axis=1, keepdims=True)
            row_stds = X.std(axis=1, keepdims=True, ddof=0)
            X = (X - row_means) / row_stds
            print("Data scaled per row.")
        except AttributeError:
            raise RuntimeError(
                "Scaling is not supported for backed or sparse .X. "
                "Either load the AnnData fully (no backed='r'), "
                "or call with scale=False."
            )

    dataset = AnnDataDataset(
        X,
        obs_names=adata.obs_names,
        time_labels=time_labels,
        root_indices=root_indices,
        terminal_indices=terminal_indices
    )
    return dataset

class AnnDataDataset(Dataset):
    def __init__(self, X, obs_names=None, time_labels=None, root_indices=None, terminal_indices=None):
        self.data = torch.tensor(X, dtype=torch.float32)
        self.time_labels = (torch.tensor(time_labels, dtype=torch.float32)
                            if time_labels is not None else None)
        
        if root_indices is not None:
            if isinstance(root_indices[0], str):
                if obs_names is None:
                    raise ValueError("obs_names must be provided when root_indices are cell names.")
                self.root_indices = set([obs_names.get_loc(name) for name in root_indices])
            else:
                self.root_indices = set(root_indices)
        else:
            self.root_indices = set()
        
        if terminal_indices is not None:
            if isinstance(terminal_indices[0], str):
                if obs_names is None:
                    raise ValueError("obs_names must be provided when terminal_indices are cell names.")
                self.terminal_indices = set([obs_names.get_loc(name) for name in terminal_indices])
            else:
                self.terminal_indices = set(terminal_indices)
        else:
            self.terminal_indices = set()
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        time_label = (self.time_labels[idx] if self.time_labels is not None 
                      else torch.tensor(0.0, dtype=torch.float32))
        is_root = idx in self.root_indices
        is_terminal = idx in self.terminal_indices
        return sample, time_label, is_root, is_terminal
