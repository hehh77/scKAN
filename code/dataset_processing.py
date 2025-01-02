import scanpy as sc
import numpy as np
from pathlib import Path


data_dir = Path("../data/muto2021")  
adata = sc.read(data_dir / "Muto-2021-RNA.h5ad")


seed = 1
np.random.seed(seed)

n_cells = adata.n_obs
n_train = int(n_cells * 0.8)
indices = np.random.permutation(n_cells)
train_idx, test_idx = indices[:n_train], indices[n_train:]

adata_train = adata[train_idx].copy()
adata_test = adata[test_idx].copy()

adata_train.write(data_dir / f"train_data.h5ad")
adata_test.write(data_dir / f"test_data.h5ad")
