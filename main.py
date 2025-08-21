import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import MinMaxScaler
import torch

import vapor
from vapor.config import parse_arguments

def main():
    config = parse_arguments()
    print("========== Parsed Arguments ==========")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("======================================\n")

    device = config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    adata = ad.read_h5ad(config.adata_file)
    X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

    if config.time_label is not None:
        time_labels = adata.obs[config.time_label].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        time_labels_scaled = scaler.fit_transform(time_labels.reshape(-1, 1)).flatten()
        time_labels = time_labels_scaled
        print(f"Time Range: {(np.min(time_labels), np.max(time_labels))}")
    else:
        time_labels = None
        print("No time label provided; proceeding without time supervision.")

    row_means = X.mean(axis=1, keepdims=True)
    row_stds = X.std(axis=1, keepdims=True, ddof=0)
    X = (X - row_means) / row_stds
    print(f"Data shape: {X.shape}")

    dataset = vapor.AnnDataDataset(
        X,
        obs_names=adata.obs_names, 
        time_labels=time_labels, 
        root_indices=config.root_indices, 
        terminal_indices=config.terminal_indices
    )
    # input_dim = X.shape[1]

    # vae = vapor.VAE(
    #     input_dim=input_dim,
    #     latent_dim=config.latent_dim,
    #     encoder_dims=config.encoder_dims,
    #     decoder_dims=config.decoder_dims
    # )
    
    # transport_op = vapor.TransportOperator(
    #     latent_dim=config.latent_dim, 
    #     n_dynamics=config.n_dynamics
    # )
     
    # model = vapor.VAPOR(vae=vae, transport_op=transport_op)
    model = vapor.initialize_model(adata.n_vars,model_config)
    trained_model = vapor.train_model(model, dataset_supervised)
    # trained_model = vapor.train_model(
    #     model=model,
    #     dataset=dataset,
    #     epochs=config.epochs,
    #     lr=config.lr,
    #     batch_size=config.batch_size,
    #     alpha=config.alpha,
    #     beta=config.beta,
    #     gamma=config.gamma,
    #     eta=config.eta,
    #     t_max=config.t_max,
    #     prune=config.prune,
    #     device=device,
    # )

    torch.save(trained_model.state_dict(), config.save_path)
    print(f"Model saved to {config.save_path}")

if __name__ == "__main__":
    main()
