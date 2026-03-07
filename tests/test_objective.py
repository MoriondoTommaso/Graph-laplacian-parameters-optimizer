import numpy as np
import optuna
from unittest.mock import patch

from graphlaplacianoptimizer._objective import make_objective

# 51 items, 24 features — large enough to form edges in 24D space.
# Clustered around 3 centroids to give the Laplacian real structure.
rng = np.random.default_rng(42)
centers = rng.random((3, 24)).astype(np.float64)
ITEMS = np.vstack([
    centers[i] + rng.normal(0, 0.05, (17, 24))  # tight cluster, low noise
    for i in range(3)
]).astype(np.float64)


# --- Test 1: Factory returns a callable ---

def test_make_objective_returns_callable():
    # make_objective must return a function that Optuna can call.
    objective = make_objective(ITEMS)
    assert callable(objective)


# --- Test 2: Objective returns a positive score on valid params ---

def test_objective_returns_positive_score():
    objective = make_objective(ITEMS)

    study = optuna.create_study(direction="maximize")

    # Force known-good params via enqueue_trial — no randomness.
    study.enqueue_trial({
        "eps":   0.5,
        "k":     12,
        "topk":  5,
        "p":     2.0,
        "sigma": 0.5,
    })
    study.optimize(objective, n_trials=1)

    # Trial must have completed — not pruned, not failed.
    assert study.best_trial.state == optuna.trial.TrialState.COMPLETE

    # Score must be positive — λ₁ + spectral gap > 0 for a connected graph.
    assert study.best_value > 0.0


# --- Test 3: Disconnected graph gets pruned ---

def test_objective_prunes_disconnected_graph():
    objective = make_objective(ITEMS)

    study = optuna.create_study(direction="maximize")

    study.enqueue_trial({
        "eps":   0.1,
        "k":     4,
        "topk":  3,
        "p":     2.0,
        "sigma": 0.1,
    })

    # Mock run_isolated_build to return [0.0, 0.0, ...] — two zero eigenvalues
    # signals a disconnected graph (multiple components) regardless of params.
    # This tests the pruning logic in _objective.py directly and cleanly.
    with patch(
        "graphlaplacianoptimizer._objective.run_isolated_build",
        return_value=[0.0, 0.0, 0.1, 0.2]
    ):
        study.optimize(objective, n_trials=1)

    pruned = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]
    assert len(pruned) == 1, "Expected 1 pruned trial for disconnected graph"
