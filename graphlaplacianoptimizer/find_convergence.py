#!/usr/bin/env python3
"""
convergence.py — Topological convergence threshold finder for ArrowSpace.

Usage:
    uv run python benchmarks/convergence.py
"""

import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed

from benchmarks.cve_loader import load_cve
from graphlaplacianoptimizer._build_direct import build_arrowspace


# ── Worker ────────────────────────────────────────────────────────────────────
def _run_single_seed(arr: np.ndarray, graph_params: dict) -> float:
    aspace, _ = build_arrowspace(graph_params, arr.copy())
    lams = aspace.lambdas()
    return float(np.var(lams))


# ── Multi-seed variance estimate ──────────────────────────────────────────────
def _variance_at_size(
    arr: np.ndarray,  # FIX 3: Changed to np.ndarray
    graph_params: dict,
    size: int,
    n_seeds: int,
) -> tuple[float, float]:
    seed_arrays = [
        arr[np.random.default_rng(s).choice(arr.shape[0], size=size, replace=False)]
        .astype(np.float64)
        for s in range(n_seeds)
    ]
    
    workers = min(n_seeds, multiprocessing.cpu_count())

    variances = Parallel(
        n_jobs=workers,
        backend="loky",
        prefer="processes",
        timeout=600,
    )(
        # FIX 2: Renamed iterator variable from arr to sub_arr to avoid shadowing
        delayed(_run_single_seed)(sub_arr, graph_params) for sub_arr in seed_arrays
    )

    return float(np.mean(variances)), float(np.std(variances))


# ── Main convergence finder ───────────────────────────────────────────────────
def find_convergence_threshold_robust(
    arr: np.ndarray,  # FIX 3: Changed to np.ndarray
    graph_params: dict,
    start_size:    int   = 100,
    step_factor:   float = 2.0,
    tolerance_pct: float = 0.05,
    seed_spread:   float = 0.10,
    patience:      int   = 2,
    n_seeds:       int   = 3,
    max_size_cap:  int   = 100000,  # FIX 1: Added max size cap parameter
) -> int:
    """
    Finds the minimum sample size at which the ArrowSpace Graph Laplacian's
    spectral variance has converged.

    Returns
    -------
    int : first sample size at which stability was confirmed
    """
    max_samples        = len(arr)
    current_size       = start_size
    prev_variance      = float("inf")
    consecutive_stable = 0
    first_stable_size  = None

    header = f"{'Size':>10} | {'Mean Var':>12} | {'Var Std':>10} | {'Rel Δ (%)':>12} | {'Stable':>8}"
    print(f"\nConvergence search  tolerance={tolerance_pct*100:.0f}%  "
          f"patience={patience}  seeds={n_seeds}  cap={max_size_cap:,}")
    print("─" * len(header))
    print(header)
    print("─" * len(header))

    while current_size <= max_samples:
        
        # FIX 1: The OOM Safety Net
        if current_size > max_size_cap:
            print("─" * len(header))
            print(f"⚠️  Reached safety cap ({max_size_cap:,}). Stopping to prevent OOM crash.")
            return first_stable_size if first_stable_size else max_size_cap

        var_mean, var_std = _variance_at_size(arr, graph_params, current_size, n_seeds)

        if prev_variance != float("inf"):
            rel_delta = abs(prev_variance - var_mean) / (prev_variance + 1e-12)
        else:
            rel_delta = float("inf")

        seed_cv   = var_std / (var_mean + 1e-12)
        is_stable = rel_delta < tolerance_pct and seed_cv < seed_spread

        if is_stable:
            consecutive_stable += 1
            if first_stable_size is None:
                first_stable_size = current_size
        else:
            consecutive_stable = 0
            first_stable_size  = None

        print(f"{current_size:>10} | {var_mean:>12.6f} | {var_std:>10.6f} | "
              f"{rel_delta*100:>11.2f}% | {consecutive_stable:>8}")

        if consecutive_stable >= patience:
            print("─" * len(header))
            pct = first_stable_size / max_samples * 100
            print(f"✅  Converged at {first_stable_size:,} items  "
                  f"({pct:.1f}% of corpus).")
            print(f"    Variance stable for {patience} consecutive "
                  f"doubling steps from {first_stable_size:,} → {current_size:,}.")
            return first_stable_size

        prev_variance = var_mean

        if current_size >= max_samples:
            break
        current_size = min(int(current_size * step_factor), max_samples)

    print("─" * len(header))
    print("⚠️  No convergence detected across full dataset.")
    return max_samples


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    GRAPH_PARAMS = {
        "eps":   3.180912,
        "k":     17,
        "topk":  204,
        "p":     2.24425,
        "sigma": 0.985255,
    }

    # Replace with your actual data loader
    arr = load_cve()

    threshold = find_convergence_threshold_robust(
        arr           = arr,
        graph_params  = GRAPH_PARAMS,
        start_size    = 10,
        step_factor   = 2.0,
        tolerance_pct = 0.05,
        seed_spread   = 0.10,
        patience      = 2,
        n_seeds       = 3,           
        max_size_cap  = 100      # Safety net engaged
    )

    print(f"\nFinal threshold: {threshold:,} items")