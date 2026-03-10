import h5py
import numpy as np
from pathlib import Path

ROOT = Path.cwd().parent
DATA_DIR = ROOT/"data"
db_path = DATA_DIR/"sift-128-euclidean.hdf5"


def load_sift(path=db_path, n_subset=10000):
    """Load SIFT-128-euclidean. Returns train subset as np.float64."""
    with h5py.File(path, "r") as f:
        # Standard ANN-benchmarks format: train/query/neighbors/distances
        train = f["train"][:n_subset].astype(np.float64)
    print(f"Loaded {train.shape} SIFT subset")
    return train

