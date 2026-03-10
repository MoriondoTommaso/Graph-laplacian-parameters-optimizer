import optuna
from pathlib import Path
import numpy as np

from benchmarks.sift_loader import load_sift
from graphlaplacianoptimizer._build_direct import build_direct  # Your fixed version
from graphlaplacianoptimizer._objective import make_objective
from save_results import save_spectral_test, save_trial_log

def spectral_score(lambdas):  # FIXED: proper normalized gap
    """Maximize spectral gap / Fiedler = λ₂/λ₁ (cluster quality)."""
    l1, l2 = lambdas[1], lambdas[2]
    return (l2 - l1) / l1 if l1 > 0 else 0.0

def main():
    full_n = 5000
    print("=== STEP 1: BASELINE ===")
    
    full_items = load_sift(n_subset=full_n).astype(np.float64)
    
    baseline_params = {'eps': 0.05, 'k': 4, 'topk': 50, 'p': 2.0, 'sigma': 0.1}
    
    baseline_lambdas_raw = build_direct(baseline_params, full_items)
    baseline_lambdas = np.array(baseline_lambdas_raw).flatten()
    
    baseline_diag = {
        'fiedler': float(baseline_lambdas[1]),
        'spectral_gap': float(baseline_lambdas[2] - baseline_lambdas[1]),
        'score': spectral_score(baseline_lambdas)
    }
    print(f"BASELINE λ[:5]: {baseline_lambdas[:5]}")
    print(f"BASELINE: {baseline_diag}")
    
    print("\n=== STEP 2: OPTIMIZE ===")
    study = optuna.create_study(
        direction="maximize",
        study_name="sift_topo_5k",
        storage="sqlite:///sift_study_5k.db",
        load_if_exists=True,
    )
    
    objective = make_objective(full_items)  # Assumes it uses build_direct + spectral_score
    study.optimize(objective, n_trials=20)  # Continues from existing trials
    
    opt_params = study.best_params  # NO topk override
    print(f"OPTIMIZED: score={study.best_value:.6f}")
    print(f"Best params: {opt_params}")
    
    print("\n=== STEP 3: BEST PARAMS VALIDATION ===")
    opt_lambdas_raw = build_direct(opt_params, full_items)
    opt_lambdas = np.array(opt_lambdas_raw).flatten()
    
    opt_diag = {
        'fiedler': float(opt_lambdas[1]),
        'spectral_gap': float(opt_lambdas[2] - opt_lambdas[1]),
        'score': spectral_score(opt_lambdas)
    }
    print(f"OPTIMIZED λ[:5]: {opt_lambdas[:5]}")
    print(f"OPTIMIZED: {opt_diag}")
    
    print("\n=== COMPARISON ===")
    print(f"Δ Fiedler:     {opt_diag['fiedler'] - baseline_diag['fiedler']:+.6f}")
    print(f"Δ Spectral Gap:{opt_diag['spectral_gap'] - baseline_diag['spectral_gap']:+.6f}")
    print(f"Δ Score:       {opt_diag['score'] - baseline_diag['score']:+.6f}")
    
    dataset_info = {
        "name": "SIFT-128-euclidean",
        "n": full_items.shape[0],
        "dim": full_items.shape[1],
        "safe_k_max": 6,
        "single_thread": True
    }
    
    trials_df = study.trials_dataframe()
    save_trial_log(trials_df, dataset_info)
    save_spectral_test(baseline_diag, opt_diag, opt_params, dataset_info)
    
    print("\n✅ 5k COMPLETE: stable single-thread tuning!")

if __name__ == "__main__":
    main()
