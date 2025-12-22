# VAPOR

Variational Autoencoder with transPort OpeRators disentangle co-occurring biological processes in development

## Installation


1. Clone the repository

    ```bash
    git clone https://github.com/JieShengm/VAPOR.git
    cd VAPOR
    ```

2. Create a virtual environment (choose one)

    ### Option 1: Using conda

    * **Step 1. Create environment**
    
        ```bash
        conda create -n vapor-env python=3.10 -y
        conda activate vapor-env
        ```
    
    * **Step 2. Verify Python Version**
    
        ```bash
        python --version
        # Python 3.10.x
        ```

    ### Option 2: Using venv (Python built-in)

    * **Step 0. Check Python version**

        ```bash
        python3 --version
        ```
        If the version is 3.10 or newer, proceed to Step 2.
        Otherwise, install Python 3.10+.

    * **Step 1. Install Python 3.10+ (Skip if already installed)**
   
        * macOS (Homebrew)
            ```bash
            brew install python@3.10
            ```
    
        * Ubuntu
            ```bash
            sudo apt update
            sudo apt install python3.10 python3.10-venv
            ```

    * **Step 2. Create and activate a virtual environment with Python 3.10+**
    
        ```bash
        python3 -m venv vapor-env
        source vapor-env/bin/activate
        ```

3. Install VAPOR

   * Core library only
   
       ```bash
       pip install -e .
       ```

   * With Jupyter / VS Code Notebook support (recommended)
       ```bash
       pip install -e ".[notebook]"
       ```
       Register the Jupyter kernel (one-time)
       ```bash
       python3 -m ipykernel install \
         --user \
         --name vapor-env \
         --display-name "Python (vapor-env)"
      ```
    
For GPU acceleration, make sure you have a working CUDA setup and install the appropriate `PyTorch` version (see [PyTorch installation guide](https://pytorch.org/get-started/locally/)).

## Notebook Usage

For a complete tutorial, please refer to [`examples/pancreas.ipynb`](examples/pancreas.ipynb).

```python
import vapor
import anndata as ad
from vapor.config import VAPORConfig

# Load data
adata = ad.read_h5ad("your_data.h5ad")

# Create dataset (unsupervised by default)
dataset = vapor.dataset_from_adata(
    adata,
    root_indices=None,         # can be integer indices (rows of adata)
    terminal_indices=None,     # or cell names (from adata.obs_names)
    scale=True
)

# Or: create dataset using selection rules (supervised)
dataset = vapor.dataset_from_adata(
    adata,
    root_where=["celltype=Early RG", "Age=pcw16"],
    terminal_where=["celltype=Glutamatergic", "Age=pcw24"],
    root_n=200,
    terminal_n=200,
    seed=42,
    scale=True
)

# Config
config = VAPORConfig(
    latent_dim=64,
    n_dynamics=10,
    lr=5e-4,
    beta=0.01,
    eta=1.0,
    t_max=4,
    epochs=500,
    batch_size=512,
)

# Initialize model
model = vapor.initialize_model(adata.n_vars, config=config)

# Train
trained_model = vapor.train_model(model, dataset, config=config)
```

#### Guidelines for `root_indices` / `terminal_indices`

Either `None`: runs in unsupervised mode (no supervision on trajectory start/end).

If provided, they can be:

- Integer indices: row positions in adata (e.g., 0,1,2,3).

- Cell names: values from adata.obs_names (e.g., cellA,cellB).

### Command Line Training [[[need to upd main.py]]]

#### Basic (unsupervised)

```bash
python main.py \
    --adata_file your_data.h5ad \
    --root_indices None \
    --terminal_indices None \
    --epochs 500 \
    --batch_size 512
```

#### Supervised

```bash
python main.py \
    --adata_file your_data.h5ad \
    --root_indices 0,1,2 \
    --terminal_indices 100,101 \
    --epochs 500
```
