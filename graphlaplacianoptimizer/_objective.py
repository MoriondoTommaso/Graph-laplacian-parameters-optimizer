import optuna
import numpy as np
from graphlaplacianoptimizer._build_direct import build_direct  # ← CAMBIO

def make_objective(items: np.ndarray):
    assert items.dtype == np.float64
    n_items, n_features = items.shape
    safe_k_max = min(n_features // 2, 30000 // max(1, n_items))
    
    def objective(trial: optuna.Trial) -> float:
        eps = trial.suggest_float("eps", 0.01, 0.3, log=True)
        k = trial.suggest_int("k", 2, safe_k_max)
        topk = trial.suggest_int("topk", 1, n_items - 1)
        p = trial.suggest_float("p", 1.0, 3.0)
        sigma = trial.suggest_float("sigma", 0.01, 1.0, log=True)
        
        params = {"eps": eps, "k": k, "topk": topk, "p": p, "sigma": sigma}
        
        # SINGLE THREAD build → NO deadlock!
        lambdas_raw = build_direct(params, items)  # returns lambdas list
        
        lambdas = [lam[0] for lam in lambdas_raw] if lambdas_raw else []
        
        if len(lambdas) < 2 or lambdas[1] == 0.0:
            raise optuna.TrialPruned()
            
        return lambdas[0] + (lambdas[1] - lambdas[0])
    return objective
