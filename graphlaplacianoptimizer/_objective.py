import optuna
import numpy as np
from graphlaplacianoptimizer._build_direct import build_direct


def make_objective(items: np.ndarray, safe_k_max: int = 50):   # FIX 1: accept param
    assert items.dtype == np.float64
    n_items, n_features = items.shape

    # FIX 2: removed broken formula — safe_k_max is now caller-controlled.
    # The old formula (30000 // n_items) returns 0 for n_items > 30000.
    if safe_k_max < 2:
        raise ValueError(f"safe_k_max must be >= 2, got {safe_k_max}")

    def objective(trial: optuna.Trial) -> float:
        eps   = trial.suggest_float("eps",   0.5,  12.0,              log=True)
        k     = trial.suggest_int(  "k",     5,    min(30, safe_k_max))
        topk  = trial.suggest_int(  "topk",  50,   min(500, n_items - 1))
        p     = trial.suggest_float("p",     1.0,  3.0)
        sigma = trial.suggest_float("sigma", 0.1,  3.0,               log=True)
        params = {"eps": eps, "k": k, "topk": topk, "p": p, "sigma": sigma}

        lambdas_raw = build_direct(params, items)

        if not lambdas_raw:
            raise optuna.TrialPruned()

        # lam[0] is the float eigenvalue, lam[1] is the integer node rank
        lambdas = [lam[0] for lam in lambdas_raw]

        if len(lambdas) < 3 or lambdas[1] == 0.0:
            raise optuna.TrialPruned()

        return (lambdas[2] - lambdas[1]) / lambdas[1]

    return objective
