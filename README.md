# VAPOR

Variational Autoencoder with transPort OpeRators disentangle co-occurring biological processes in development

## Installation

1. Clone the repository

```bash
git clone https://github.com/JieShengm/VAPOR.git
cd VAPOR
```

2. (Recommended) Create a virtual environment

Using **conda**:

```bash
conda create -n vapor-env python=3.10 -y
conda activate vapor-env
```

Or using **venv**:

```bash
python -m venv vapor-env
source vapor-env/bin/activate   # Linux/Mac
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Install the package

```bash
pip install -e .
```

### Notes

If you plan to run notebooks, also install `Jupyter`:

```bash
pip install jupyterlab
```

For GPU acceleration, make sure you have a working CUDA setup and install the appropriate `PyTorch` version (see [PyTorch installation guide](https://pytorch.org/get-started/locally/)).

## Usage

### Command Line Training

**Basic (unsupervised)**

```bash
python main.py \
    --adata_file your_data.h5ad \
    --root_indices None \
    --terminal_indices None \
    --epochs 500 \
    --batch_size 512
```

Guidelines for `root_indices` / `terminal_indices`

Either `None`: runs in unsupervised mode (no supervision on trajectory start/end).

If provided, they can be:

- Integer indices: row positions in adata (e.g., 0,1,2,3).

- Cell names: values from adata.obs_names (e.g., cellA,cellB).

**Supervised**

```bash
python main.py --adata_file data.h5ad \
    --root_indices 0,1,2 \
    --terminal_indices 100,101 \
    --epochs 500
```


### Notebook Usage

```python
import vapor
import anndata as ad

# Load data
adata = ad.read_h5ad("your_data.h5ad")

# Create dataset (unsupervised by default)
dataset = vapor.dataset_from_adata(
    adata,
    root_indices=None,         # can be integer indices (rows of adata)
    terminal_indices=None,     # or cell names (from adata.obs_names)
    scale=True
)

# Initialize model
model = vapor.initialize_model(adata.n_vars, lr=5e-5)

# Train model
trained_model = vapor.train_model(model, dataset, epochs=500)
```
