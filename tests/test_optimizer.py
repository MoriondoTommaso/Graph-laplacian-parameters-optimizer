import numpy as np
import optuna
import os
from unittest.mock import patch

from graphlaplacianoptimizer._optimizer import make_synthetic_dataset, main


# --- Test 1: Synthetic dataset has correct shape and dtype ---

def test_make_synthetic_dataset_shape_and_dtype():
    items = make_synthetic_dataset(n_items=51, n_features=24, seed=42)

    # Shape must match what was requested.
    assert items.shape == (51, 24), f"Unexpected shape: {items.shape}"

    # dtype must be float64 — mandatory for Rust FFI safety.
    assert items.dtype == np.float64, f"Unexpected dtype: {items.dtype}"

    # Must be a numpy array — never a list.
    assert isinstance(items, np.ndarray)


# --- Test 2: Dataset is reproducible with the same seed ---

def test_make_synthetic_dataset_reproducible():
    items_a = make_synthetic_dataset(seed=42)
    items_b = make_synthetic_dataset(seed=42)

    # Same seed must produce identical arrays every time.
    # This ensures benchmark results are reproducible across runs.
    np.testing.assert_array_equal(items_a, items_b)


# --- Test 3: main() runs end to end without crashing ---

def test_main_runs_without_crashing(tmp_path):
    # tmp_path is a pytest built-in fixture that provides a temporary
    # directory unique to this test. We use it to store the SQLite study
    # database so tests never pollute the real study.db file.
    db_path = tmp_path / "test_study.db"

    # Patch the storage path and n_trials inside main() so the test
    # completes in seconds (2 trials) rather than the full 50.
    with patch(
        "graphlaplacianoptimizer._optimizer.STORAGE",
        f"sqlite:///{db_path}"
    ), patch(
        "graphlaplacianoptimizer._optimizer.N_TRIALS",
        2
    ):
        # Must complete without raising any exception.
        main()
