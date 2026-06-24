# VAPOR

Variational Autoencoder with transPort OpeRators — disentangle co-occurring biological processes in time and space.

## Installation

1. Clone the repository

    ```bash
    git clone https://github.com/JieShengm/VAPOR.git
    cd VAPOR
    ```

2. Create a virtual environment (choose one)

    ### Option 1: Using conda

    ```bash
    conda create -n vapor-env python=3.10 -y
    conda activate vapor-env
    ```

    ### Option 2: Using venv

    ```bash
    python3 -m venv vapor-env
    source vapor-env/bin/activate
    ```

3. Install VAPOR

    ```bash
    pip install -e .
    ```

    With Jupyter support:

    ```bash
    pip install -e ".[notebook]"
    python3 -m ipykernel install --user --name vapor-env --display-name "Python (vapor-env)"
    ```

For GPU acceleration, install the appropriate PyTorch version for your CUDA setup (see [PyTorch installation guide](https://pytorch.org/get-started/locally/)).

## Usage

```python
import vapor
import scanpy as sc
from vapor.config import VAPORConfig

# ── 1. Load data ──────────────────────────────────────────────
# Input should be normalized gene expression (e.g. log1p-normalized),
# not raw counts. VAPOR applies per-cell z-score scaling internally.
adata = sc.read_h5ad("your_data.h5ad")

# ── 2. Create dataset ────────────────────────────────────────
# Root/terminal cells guide directionality (optional but recommended).
# Provide as integer indices (row positions) or cell names (obs_names).
dataset = vapor.dataset_from_adata(
    adata,
    root_indices=[0, 1, 2],          # or None for unsupervised
    terminal_indices=[100, 101],     # or None
    scale=True,
)

# You can also select root/terminal cells by metadata:
from vapor.dataset import select_obs_indices

root_idx, _ = select_obs_indices(
    adata, where={"celltype": "Radial Glia", "Age": "pcw16"}, n=200, return_names=False
)
terminal_idx, _ = select_obs_indices(
    adata, where={"celltype": "Neuron", "Age": "pcw24"}, n=200, return_names=False
)
dataset = vapor.dataset_from_adata(
    adata, root_indices=root_idx, terminal_indices=terminal_idx
)

# ── 3. Configure & initialize ────────────────────────────────
config = VAPORConfig(
    latent_dim=64,
    n_dynamics=10,
    total_steps=20000,
    batch_size=512,
    lr=3e-5,
    beta=0.01,
)

model = vapor.initialize_model(adata.n_vars, config=config)

# ── 4. Train ─────────────────────────────────────────────────
model = vapor.train_model(model, dataset, config=config)

# ── 5. Extract latents & dynamics ─────────────────────────────
adata_vapor = vapor.extract_latents_and_dynamics(model, adata, device="cpu")

# adata_vapor contains:
#   .obsm["X_VAPOR"]        latent embedding (mu)
#   .layers["v_VAPOR"]      vector field
#   .obs["pw_1"], etc.      per-process mixture weights
#   .layers["v_psi1"], etc. per-process unit directions

# ── 6. Directional gene scoring ──────────────────────────────
from vapor.inference import directional_gene_scores_jvp_progress, run_enrichment

directional_gene_scores_jvp_progress(model, adata_vapor)

# ── 7. Gene set enrichment (optional) ────────────────────────
results = run_enrichment(
    adata_vapor,
    organism="Human",
    gene_sets=("GO_Biological_Process_2025",),
)
```

### Save & load checkpoints

```python
# Save
vapor.save_checkpoint(model, config, "model.pt")

# Load — one call, no setup needed
model, ckpt = vapor.load_checkpoint("model.pt")

# Or load into an existing model
model, ckpt = vapor.load_checkpoint("model.pt", model=my_model)
```

### Key parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `latent_dim` | Latent space dimensionality | 64 |
| `n_dynamics` | Number of transport operators (processes to disentangle) | 10 |
| `total_steps` | Training steps | 20000 |
| `beta` | KL divergence weight | 0.01 |
| `t_max` | Max integration time for trajectory loss | 5 |
| `root_indices` | Cell indices marking trajectory start | None |
| `terminal_indices` | Cell indices marking trajectory end | None |
