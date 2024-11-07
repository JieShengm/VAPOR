import os
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
# import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
from model import VAE, construct_pairs
# from utilities import load_checkpoint

class Dataset(Dataset):
    def __init__(self, data_path, header=None, transform=None):
        self.file = None
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path, header=header).to_numpy()
            print(f'n_OBS: {self.data.shape[0]}; n_VAR: {self.data.shape[1]}')
        elif data_path.endswith('.h5ad'):
            # import anndata as ad
            h5ad_data = ad.read_h5ad(data_path)
            self.data = h5ad_data.X
            if issparse(h5ad_data.X):
                self.data = h5ad_data.X.toarray()
            print(f'n_OBS: {self.data.shape[0]}; n_VAR: {self.data.shape[1]}')
        else:
            raise AssertionError("The file format is not supported/available now.")

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx, :]
        
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_dataloader(data_path, batch_size, shuffle=True,  header=None, transform=None, return_input_dim = True):
    dataset = Dataset(data_path=data_path, header=header, transform=transform)
    input_dim = dataset.data.shape[1]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last = True)
    if return_input_dim:
        return dataloader, input_dim
    else:
        return dataloader 

def load_checkpoint(checkpoint_path, model, device, optimizer=None):
    """
    Loads model state (and optimizer state) from a file.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', -1)  # Return the last completed epoch number, if available

def load_data(data_path):
    """Load single-cell data from the specified path."""
    return ad.read_h5ad(data_path)

def load_model(model_path, input_dim, device=None,
               latent_dim=2, encoder_dims=[256, 64, 8],
               decoder_dims=[8, 64, 256], psi_M=4):
    """Initialize and load the VAE model with pretrained weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the VAE model
    vae = VAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_dims=encoder_dims,
        decoder_dims=decoder_dims,
        psi_M=psi_M
    )
    vae.to(device)
    vae.eval()

    # Load the model checkpoint
    load_checkpoint(model_path, vae, device)

    # Load psi from the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    psi = checkpoint['psi']

    return vae, psi

def construct_VAPOR_adata(data_path, model_path, device=None,
                          latent_dim=2, encoder_dims=[256, 64, 8],
                          decoder_dims=[8, 64, 256], psi_M=4):
    """Construct adata_VAPOR using the data and the trained model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    adata = load_data(data_path)
    input_dim = adata.shape[1]

    # Load model
    vae, psi = load_model(
        model_path, input_dim, device,
        latent_dim, encoder_dims, decoder_dims, psi_M
    )

    # Convert adata.X to a PyTorch tensor and move to device
    X_tensor = torch.tensor(adata.X, dtype=torch.float32, device=device)

    # Encode the data
    with torch.no_grad():
        mu, _ = vae.Encode(X_tensor)

    # Construct pairs for the VAE model
    pairs = construct_pairs(mu, 15, psi)

    # Run the VAE forward pass
    with torch.no_grad():
        _, z, mu_ast, _ = vae(X_tensor, pairs[1][1:])

    # Create adata_VAPOR and populate it with computed values
    mu_np = mu.cpu().numpy()
    adata_VAPOR = ad.AnnData(mu_np)
    adata_VAPOR.obs = adata.obs.copy()
    adata_VAPOR.obsm['X_mu'] = mu_np
    # adata_VAPOR.obsm['X_mu_ast'] = mu_ast.cpu().numpy()
    adata_VAPOR.obsm['X_z'] = z.cpu().numpy()

    # Compute mu1 and velocity
    with torch.no_grad():
        psi_exp = torch.matrix_exp(psi.permute(2, 0, 1))
        mu1 = torch.einsum('mik, bk -> mbi', psi_exp, mu)
        velocity = mu1 - mu

    # Move mu1 and velocity to CPU and store them in adata_VAPOR layers
    mu1_np = mu1.cpu().numpy()
    velocity_np = velocity.cpu().numpy()
    for i in range(psi.shape[-1]):
        mu1_key = f'mu1_psi{i}'
        v_key = f'v_psi{i}'
        adata_VAPOR.layers[mu1_key] = mu1_np[i]
        adata_VAPOR.layers[v_key] = velocity_np[i]

    return [adata_VAPOR, adata]

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import scanpy as sc
def compute_transition_matrix(current_state, velocity, n_neighbors=30, eps=1e-5):
    n_cells = current_state.shape[0]
    
    if np.any(np.isnan(velocity)) or np.any(np.isinf(velocity)):
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=1e10, neginf=-1e10)
    
    zero_vel_rows = np.where(np.all(velocity == 0, axis=1))[0]
    if len(zero_vel_rows) > 0:
        velocity[zero_vel_rows] = eps

    vel_similarities = 1 - squareform(pdist(velocity, metric='cosine'))
    vel_similarities = np.clip(vel_similarities, 0, 1 - eps)
    state_distances = squareform(pdist(current_state, metric='euclidean'))
    state_distances = np.maximum(state_distances, eps)
    transition_probs = vel_similarities / state_distances

    for i in range(n_cells):
        indices = np.argsort(transition_probs[i])[::-1]
        transition_probs[i, indices[n_neighbors:]] = 0
        
    zero_rows = np.where(np.sum(transition_probs, axis=1) == 0)[0]
    for row in zero_rows:
        transition_probs[row, np.argmax(vel_similarities[row])] = eps
    
    row_sums = transition_probs.sum(axis=1)
    transition_matrix = transition_probs / row_sums[:, np.newaxis]
    
    
    return transition_matrix

def compute_pseudotime(transition_matrix, start_cell=None, eps=1e-10):
    n_cells = transition_matrix.shape[0]
    if start_cell is None:
        start_cell = np.argmax(transition_matrix.sum(axis=1))
        print(f"Automatically selected start cell: {start_cell}")
    else:
        print(f"Using specified start cell: {start_cell}")
    pseudotime = np.zeros(n_cells)
    current_cell = start_cell
    current_time = 0

    visited = set([start_cell])
    while len(visited) < n_cells:
        probs = transition_matrix[current_cell]
        next_cell = np.argmax(probs)
        if next_cell in visited:
            unvisited = list(set(range(n_cells)) - visited)
            next_cell = unvisited[np.argmax(probs[unvisited])]
        
        time_step = 1 / (probs[next_cell] + eps)
        time_step = min(time_step, 1000)
        current_time += time_step
        pseudotime[next_cell] = current_time
        visited.add(next_cell)
        current_cell = next_cell

    pseudotime = (pseudotime - np.min(pseudotime)) / (np.max(pseudotime) - np.min(pseudotime))
    
    return pseudotime

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
def diffusion_pseudotime(transition_matrix, start_cell=None):
    graph = csr_matrix(transition_matrix)
    n_components, labels = connected_components(csgraph=graph, directed=True, return_labels=True)
    if start_cell is None:
        start_cell = np.argmax(transition_matrix.sum(axis=1))
        print(f"Automatically selected start cell: {start_cell}")
    else:
        print(f"Using specified start cell: {start_cell}")
        
    distances, predecessors = dijkstra(csgraph=graph, directed=True, 
                                     indices=start_cell, 
                                     return_predecessors=True)

    unreachable = np.isinf(distances)
    
    if unreachable.sum() > 0:
        max_finite_distance = np.max(distances[~np.isinf(distances)])
        distances[unreachable] = max_finite_distance * 1.1  
    
    normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    
    return normalized_distances
