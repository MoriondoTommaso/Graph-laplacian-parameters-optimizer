import numpy as np
import optuna

from graphlaplacianoptimizer._isolated_build import run_isolated_build


def make_objective(items: np.ndarray):

    assert items.dtype == np.float64, f"items must be np.float64, got {items.dtype}"
    
    n_items, n_features = items.shape
    
    # SAFE BOUNDS ← ADD THIS BLOCK
    max_edges = 30000  
    safe_k_max = min(n_features // 2, max_edges // max(1, n_items))
    
    def objective(trial: optuna.Trial) -> float:

        eps = trial.suggest_float("eps", 0.01, 0.3, log=True)  # tighter too
        
        k = trial.suggest_int("k", 2, safe_k_max)  # ← CHANGED
        
        topk = trial.suggest_int("topk", 1, n_items - 1)

        p = trial.suggest_float("p", 1.0, 3.0)

        sigma = trial.suggest_float("sigma", 0.01, 1.0, log=True)
        
        graph_params = {
            "eps":   eps,
            "k":     k,
            "topk":  topk,
            "p":     p,
            "sigma": sigma,
        }

        # --- Build the graph in an isolated subprocess ---
        # run_isolated_build handles all FFI safety (spawn context, copy).
        # Returns list[float] of eigenvalues, or None on subprocess crash.
        lambdas = run_isolated_build(graph_params, items)

        # Subprocess crashed entirely (exitcode != 0).
        # Prune so Optuna skips this region of param space.
        if lambdas is None:
            raise optuna.exceptions.TrialPruned()

        # Disconnected graph: λ₁ = 0.0 means no edges formed.
        # A disconnected graph has no useful spectral structure for search.
        # Prune so Optuna avoids this region.
        if lambdas[1] == 0.0:
            raise optuna.exceptions.TrialPruned()

        # Need at least 2 eigenvalues to compute both terms of the score.
        if len(lambdas) < 2:
            raise optuna.exceptions.TrialPruned()

        # --- Scoring ---
        # λ₁ (Fiedler value): how well-connected the graph is.
        # Higher = harder to disconnect = more robust topology.
        fiedler = lambdas[0]

        # λ₂ - λ₁ (spectral gap): how clearly clusters are separated.
        # Higher = more distinct community structure on the manifold.
        spectral_gap = lambdas[1] - lambdas[0]

        # The score is the sum of both terms.
        # Optuna maximises this — it will push toward params that produce
        # both a well-connected graph AND well-separated clusters.
        return float(fiedler + spectral_gap)

    # Return the closure — items is bound inside it.
    return objective
