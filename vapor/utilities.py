import os
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import anndata as ad
from scipy.sparse import issparse
from .model import VAE, construct_pairs
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

def construct_VAPOR_adata(data_path, 
                          model_path, 
                          device=None,
                          latent_dim=3, 
                          encoder_dims=[1024, 512, 256, 128],
                          decoder_dims=[128, 256, 512, 1024],
                          psi_M=4):
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

    X_tensor = torch.tensor(adata.X, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu, _ = vae.Encode(X_tensor)
        
    pairs = construct_pairs(mu, 50, psi)
    with torch.no_grad():
        _, z, _, _ = vae(X_tensor, pairs[1][1:])

    mu_np = mu.cpu().numpy()
    adata_VAPOR = ad.AnnData(mu_np)
    adata_VAPOR.obs = adata.obs.copy()
    adata_VAPOR.obsm['X_mu'] = mu_np
    # adata_VAPOR.obsm['X_mu_ast'] = mu_ast.cpu().numpy()
    adata_VAPOR.obsm['X_z'] = z.cpu().numpy()

    with torch.no_grad():
        psi_exp = torch.matrix_exp(psi)
        mu1 = torch.einsum('mij, bj -> mbi', psi_exp, mu)
        velocity = mu1 - mu

    # mu1_np = mu1.cpu().numpy()
    velocity_np = velocity.cpu().numpy()
    for i in range(psi.shape[0]):
        # mu1_key = f'mu1_psi{i}'
        key = f'psi{i+1}'
        # adata_VAPOR.layers[mu1_key] = mu1_np[i]
        adata_VAPOR.layers['v_'+key] = velocity_np[i]
        adata_VAPOR.uns[key] =psi[i] 

    return [adata_VAPOR, adata]
