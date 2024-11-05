# Variational autoencoder with transport operators decouples co-occurring biological processes in development

Emerging single-cell and spatial transcriptomic data enable the investigation of gene expression dynamics of various biological processes, especially for development. To this end, existing computational methods typically infer trajectories that sequentially order cells for revealing gene expression changes in development, e.g., to assign a pseudotime to each cell indicating the ordering. However, these trajectories can aggregate different biological processes that cells undergo simultaneously such as maturation for specialized function and differentiation into specific cell types that do not occur on the same timescale. Therefore, a single pseudotime axis may not distinguish gene expression dynamics from co-occurring processes. 

We introduce a method, VAPOR (variational autoencoder with transport operators), to decouple dynamic patterns from developmental gene expression data. Particularly, VAPOR learns a latent space for gene expression dynamics and decomposes the space into multiple subspaces. The dynamics on each subspace are governed by an ordinary differential equation model, attempting to recapitulate specific biological processes. Furthermore, we can infer the process-specific pseudotimes, revealing multifaceted timescales of distinct processes in which cells may simultaneously be involved during development. 

VAPOR is open source for general use to parameterize and infer developmental gene expression dynamics. It can be further extended for other single-cell and spatial omics such as chromatin accessibility to reveal developmental epigenomic dynamics.

![fig1](https://github.com/JieShengm/VAPOR/blob/main/figures/fig1.png)

# Usage

First, clone and navigate to the repository.

```
git clone https://github.com/JieShengm/VAPOR
cd VAPOR
```

Create and activate VAPOR environment using python 3.9 with `conda`,

```
conda env create -f environment.yml
conda activate VAPOR_env
```

To train a VAPOR model, use

```
python VAPOR/main.py --data_path /PATH/TO/FILE
```
