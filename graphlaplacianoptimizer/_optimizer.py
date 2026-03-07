import numpy as np
import optuna

from graphlaplacianoptimizer._objective import make_objective

# Module-level constants — defined here so tests can patch them cleanly
# without needing to reach inside main()'s local scope.
STORAGE = "sqlite:///study.db"
N_TRIALS = 50


def make_synthetic_dataset(
    n_items: int = 51,
    n_features: int = 24,
    seed: int = 42
) -> np.ndarray:
    # Fixed seed ensures the dataset is identical across every run.
    # This is critical for reproducibility — if the dataset changes
    # between runs, trial scores are not comparable.
    rng = np.random.default_rng(seed)

    # 3 cluster centroids randomly placed in n_features-dimensional space.
    # These define the "true" structure the Laplacian should recover.
    n_clusters = 3
    centers = rng.random((n_clusters, n_features)).astype(np.float64)

    # Each cluster gets n_items // n_clusters points scattered around
    # its centroid with small Gaussian noise (std=0.05).
    # Low noise = tight clusters = clear community structure for the Laplacian.
    points_per_cluster = n_items // n_clusters
    clusters = [
        centers[i] + rng.normal(0, 0.05, (points_per_cluster, n_features))
        for i in range(n_clusters)
    ]

    # Stack all clusters into a single (n_items, n_features) array.
    # astype(np.float64) enforces the Rust FFI dtype requirement.
    return np.vstack(clusters).astype(np.float64)


def main() -> None:
    # Suppress Optuna's info logs during the optimization loop.
    # We print our own summary at the end instead.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    items = make_synthetic_dataset()

    # create_study with load_if_exists=True means:
    # - First run: creates a new study and saves it to STORAGE (SQLite).
    # - Subsequent runs: loads the existing study and continues from
    #   where it left off. Trials accumulate across restarts.
    study = optuna.create_study(
        direction="maximize",
        study_name="graph_laplacian_opt",
        storage=STORAGE,
        load_if_exists=True,
    )

    # n_startup_trials=10 tells the TPE sampler to explore randomly
    # for the first 10 trials before switching to Bayesian inference.
    # This prevents early convergence to a local optimum.
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, seed=42)
    study.sampler = sampler

    print(f"Starting optimization: {N_TRIALS} trials")
    print(f"Dataset shape: {items.shape}, dtype: {items.dtype}")
    print(f"Storage: {STORAGE}\n")

    study.optimize(
        make_objective(items),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # Report results.
    print("\n=== Best trial ===")
    print(f"  Score (λ₁ + spectral gap): {study.best_value:.6f}")
    print(f"  Params:")
    for key, val in study.best_params.items():
        print(f"    {key}: {val}")

    # Report how many trials were pruned vs completed.
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]
    print(f"\n  Completed trials: {len(completed)}")
    print(f"  Pruned trials:    {len(pruned)}")


if __name__ == "__main__":
    main()
