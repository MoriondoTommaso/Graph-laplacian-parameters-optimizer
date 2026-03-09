#!/usr/bin/env python
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path.cwd().parent

def save_spectral_test(baseline_diag, opt_diag, opt_params, dataset_info, trial_log=None):
    """Salva test completo in folder dedicato con dimensioni dataset."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_base = dataset_info.get("n_baseline", "N/A")
    n_opt = dataset_info.get("n_optuna", "N/A")
    dim = dataset_info.get("dim", "N/A")
    
    # Crea folder benchmark dedicato
    bench_folder = ROOT / "results" / f"bench_n{n_base}_n{n_opt}_d{dim}_{timestamp}"
    bench_folder.mkdir(parents=True, exist_ok=True)
    
    # Enrich dataset info
    dataset_enriched = dataset_info.copy()
    dataset_enriched["n_samples_baseline"] = n_base
    dataset_enriched["n_samples_optuna"] = n_opt
    dataset_enriched["dimension"] = dim
    
    # Master results JSON
    master = {
        "timestamp": timestamp,
        "dataset": dataset_enriched,
        "baseline": baseline_diag,
        "optimized": opt_diag,
        "opt_params": opt_params,
        "improvement": {
            "delta_score": opt_diag["score"] - baseline_diag["score"],
            "pct_improvement": (opt_diag["score"] / baseline_diag["score"] - 1) * 100 
                               if baseline_diag["score"] > 0 else float('inf'),
            "fiedler_improvement": opt_diag["spectral_gap"] - baseline_diag["spectral_gap"],
        },
        "trial_log": trial_log,
    }
    
    # Salva in folder dedicato
    (bench_folder / "spectral_test.json").write_text(json.dumps(master, indent=2))
    
    # CSV comparison
    df_cmp = pd.DataFrame([baseline_diag, opt_diag])
    df_cmp["config"] = ["baseline", "optimized"]
    df_cmp["n_samples"] = [n_base, n_opt]
    df_cmp["dim"] = dim
    df_cmp.to_csv(bench_folder / "spectral_comparison.csv", index=False)
    
    # Params CSV
    df_params = pd.DataFrame([opt_params])
    df_params.to_csv(bench_folder / "best_parameters.csv", index=False)
    
    print(f"✅ Saved benchmark folder: {bench_folder}")
    print(f"   Δ Score: {master['improvement']['delta_score']:+.6f}")
    print(f"   Improvement: {master['improvement']['pct_improvement']:.1f}x")


def save_trial_log(trials_df, dataset_info):
    """Salva log trials in folder dedicato."""
    n_opt = dataset_info.get("n_optuna", "N/A")
    dim = dataset_info.get("dim", 128)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    trial_folder = ROOT / "results" / f"trials_n{n_opt}_d{dim}_{timestamp}"
    trial_folder.mkdir(parents=True, exist_ok=True)
    
    trials_df["dataset_n"] = n_opt
    trials_df["dataset_dim"] = dim
    trials_df.to_csv(trial_folder / "optuna_trials.csv", index=False)
    
    print(f"📊 Saved trials folder: {trial_folder} ({len(trials_df)} trials)")
