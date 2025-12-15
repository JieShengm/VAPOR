from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
readme = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

setup(
    name="vapor",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
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
    ],
    extras_require={
        "notebook": [
            "ipykernel",
            "jupyter-client",
            "pyzmq",
            "tornado",
            "notebook",
            # or "jupyterlab"
        ],
        "dev": [
            "pytest",
            "ruff",
            "black",
        ],
    },
    author="JS",
    description="VAPOR: Variational Autoencoder with Transport Operators",
    long_description=readme,
    long_description_content_type="text/markdown",
)
