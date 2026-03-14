import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path.cwd().parent
DATA_DIR = ROOT / "data"
# Updated path to point to your parquet file
db_path = DATA_DIR / "cve1999-2025.parquet"

def load_cve(path = db_path, n_subset = None):
    """
    Load vector dataset from Parquet. 
    Returns the first n_subset rows as a np.float64 numpy array.
    """
    df = pd.read_parquet(path)

    train = df.iloc[:len(df)].to_numpy(dtype = np.float64)
    
    print(f"Loaded {train.shape} subset from Parquet")
    return train

