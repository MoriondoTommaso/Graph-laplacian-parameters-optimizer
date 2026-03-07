import numpy as np
import optuna

from graphlaplacianoptimizer._isolated_build import run_isolated_build


def make_objective(items: np.ndarray):
    # Validate dtype here once, at factory time, not on every trial.
    # This enforces the FFI safety rule before Optuna even starts.
    assert items.dtype == np.float64, (
        f"items must be np.float64, got {items.dtype}"
    )

    # Read dataset shape once. These values are used to set param bounds
    # that are safe for this specific dataset.
    n_items, n_features = items.shape

    def objective(trial: optuna.Trial) -> float:
        # --- Parameter suggestion ---
        # Each suggest_* call asks Optuna to pick a value within the given
        # bounds. Optuna uses past trial results (Bayesian inference via TPE)
        # to pick values more likely to score well — not random sampling.

        # eps: log=True means Optuna searches on a log scale.
        # This is correct because eps matters more at small values
        # (0.01 vs 0.05 is a big difference) than at large ones
        # (0.4 vs 0.45 barely differs).
        eps = trial.suggest_float("eps", 0.01, 0.5, log=True)

        # k: number of nearest neighbours for graph wiring.
        # Lower bound 2: minimum for a connected graph.
        # Upper bound n_features // 2: matches the example in project docs.
        k = trial.suggest_int("k", 2, n_features // 2)

        # topk: how many results the search returns per query.
        # Must be at least 1 and less than total items.
        topk = trial.suggest_int("topk", 1, n_items - 1)

        # p: Minkowski exponent. p=1 is Manhattan, p=2 is Euclidean.
        # We search between 1.0 and 3.0 — values beyond 3 are rarely useful.
        p = trial.suggest_float("p", 1.0, 3.0)

        # sigma: Gaussian kernel bandwidth. Controls edge weight smoothness.
        # log=True for the same reason as eps — small values matter more.
        # This is the most sensitive parameter for the Fiedler value.
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
