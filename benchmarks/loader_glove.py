import numpy as np
import h5py


def load_corpus(path: str, n_items: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        data = f["train"][:n_items]
    return data.astype(np.float64)


def load_queries(path: str, n_queries: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        queries = f["test"][:n_queries].astype(np.float64)
        ground_truth = f["neighbors"][:n_queries].astype(np.int32)
    return queries, ground_truth
