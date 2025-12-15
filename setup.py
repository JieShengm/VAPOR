from setuptools import setup, find_packages

setup(
    name="vapor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision",
        "torchdiffeq",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "umap-learn",
        "anndata",
        "scanpy",
        "scvelo",
        "notebook",
        "ipykernel"
    ],
    python_requires=">=3.10",
    author="JS",
    description="VAPOR: Variational Autoencoder with Transport Operators",
    long_description=open("README.md").read() if "README.md" else "",
    long_description_content_type="text/markdown",
)
