#!/usr/bin/env python
import optuna
from pathlib import Path

from benchmarks.sift_loader import load_sift
from benchmarks.spectral_diag import compute_spectral_diag
from graphlaplacianoptimizer._objective import make_objective
from save_results import save_spectral_test, save_trial_log

def main():
    # Full dataset for evaluation
    print("=== STEP 1: BASELINE ===")
    full_items = load_sift(n_subset=4000)
    baseline_params = {
    "eps":   250.0,
    "k":     8,      # << Conservative k=8
    "topk":  50,
    "p":     2.0,
    "sigma": 100.0,
}

    baseline_diag = compute_spectral_diag(baseline_params, full_items)
    print(f"BASELINE: {baseline_diag}")
    

    print("\n=== STEP 2: OPTIMIZE ON SUBSET ===")
    subset_items = load_sift(n_subset=4000)  # Fast tuning
    study = optuna.create_study(
        direction="maximize",
        study_name="sift_topo",
        storage="sqlite:///sift_study.db",
        load_if_exists=True,
    )
    study.optimize(make_objective(subset_items), n_trials=20)
    opt_params = study.best_params | {"topk": 100}  # Fill missing
    print(f"OPTIMIZED: score={study.best_value:.6f}")
    print(f"Params: {opt_params}")
    
    print("\n=== STEP 3: OPTIMIZED ON FULL ===")
    opt_diag = compute_spectral_diag(opt_params, full_items)
    print(f"OPTIMIZED: {opt_diag}")
    
    print("\n=== COMPARISON ===")
    print(f"Δ Fiedler:     {opt_diag['fiedler'] - baseline_diag['fiedler']:+.6f}")
    print(f"Δ Spectral Gap:{opt_diag['spectral_gap'] - baseline_diag['spectral_gap']:+.6f}")
    print(f"Δ Score:       {opt_diag['score'] - baseline_diag['score']:+.6f}")
    
    dataset_info = {
        "name": "SIFT-128-euclidean",
        "n_baseline": full_items.shape[0],
        "n_optuna": subset_items.shape[0],
        "dim": full_items.shape[1],
    }
    
    from save_results import save_spectral_test, save_trial_log
    trials_df = study.trials_dataframe()
    save_trial_log(trials_df, dataset_info)
    save_spectral_test(baseline_diag, opt_diag, opt_params, dataset_info)

if __name__ == "__main__":
    main()
