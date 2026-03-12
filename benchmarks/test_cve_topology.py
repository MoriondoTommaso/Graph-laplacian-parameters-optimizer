import optuna
import numpy as np

from benchmarks.cve_loader import load_cve
from graphlaplacianoptimizer._build_direct import build_direct
from graphlaplacianoptimizer._objective import make_objective
from graph_params_opt.save_results import save_spectral_test, save_trial_log

  # define once, shared by objective and dataset_info


def spectral_score(lambdas):
    """Normalized gap (λ₃ − λ₂) / λ₂ — proxy for cluster separation quality.
    lambdas[0] = 0.0 trivial, lambdas[1] = Fiedler (λ₂), lambdas[2] = λ₃.
    """
    l2, l3 = lambdas[1], lambdas[2]
    return (l3 - l2) / l2 if l2 > 0 else 0.0


def extract_lambdas(raw) -> np.ndarray:
    """
    lambdas_sorted() → list[tuple[lambda: float, rank: int]]
    tuple[0] is the eigenvalue, tuple[1] is the node rank integer.
    """
    if raw is None:
        raise RuntimeError("build_direct returned None — graph construction failed.")
    return np.array([lam[0] for lam in raw])  # ← lam[0] is the float eigenvalue


def main():
    full_n = 50000

    print("=== STEP 1: BASELINE ===")

    full_items = load_cve(n_subset=full_n).astype(np.float64)

    baseline_params = {'eps': 3, 'k': 10, 'topk': 50, 'p': 2.0, 'sigma': 1}

    baseline_lambdas = extract_lambdas(build_direct(baseline_params, full_items))

    baseline_diag = {
        'fiedler':      float(baseline_lambdas[1]),
        'spectral_gap': float(baseline_lambdas[2] - baseline_lambdas[1]),
        'score':        spectral_score(baseline_lambdas)
    }
    print(f"BASELINE λ[:5]: {baseline_lambdas[:5]}")
    print(f"BASELINE: {baseline_diag}")

    print("\n=== STEP 2: OPTIMIZE ===")

    try:
        study = optuna.create_study(
            direction="maximize",
            study_name="cve_topo_50k",
            storage="sqlite:///cve_topo_50k.db",
            load_if_exists=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Optuna study creation failed (DB locked?): {exc}") from exc

    objective = make_objective(full_items)  # FIX: pass explicitly
    study.optimize(objective, n_trials=20)

    opt_params = study.best_params

    if 'topk' not in opt_params:
        opt_params['topk'] = baseline_params['topk']

    print(f"OPTIMIZED: score={study.best_value:.6f}")
    print(f"Best params: {opt_params}")

    print("\n=== STEP 3: BEST PARAMS VALIDATION ===")

    opt_lambdas = extract_lambdas(build_direct(opt_params, full_items))

    opt_diag = {
        'fiedler':      float(opt_lambdas[1]),
        'spectral_gap': float(opt_lambdas[2] - opt_lambdas[1]),
        'score':        spectral_score(opt_lambdas)
    }
    print(f"OPTIMIZED λ[:5]: {opt_lambdas[:5]}")
    print(f"OPTIMIZED: {opt_diag}")

    print("\n=== COMPARISON ===")
    print(f"Δ Fiedler:      {opt_diag['fiedler']      - baseline_diag['fiedler']:+.6f}")
    print(f"Δ Spectral Gap: {opt_diag['spectral_gap'] - baseline_diag['spectral_gap']:+.6f}")
    print(f"Δ Score:        {opt_diag['score']        - baseline_diag['score']:+.6f}")

    dataset_info = {
        "name":          "CVE-dataset",
        "n":             full_items.shape[0],
        "dim":           full_items.shape[1],
        "safe_k_max":    SAFE_K_MAX,   # consistent with what was passed to make_objective
        "single_thread": True
    }

    trials_df = study.trials_dataframe()
    save_trial_log(trials_df, dataset_info)
    save_spectral_test(baseline_diag, opt_diag, opt_params, dataset_info)

    print(f"\n✅ {full_n} COMPLETE: stable single-thread tuning!")


if __name__ == "__main__":
    main()
