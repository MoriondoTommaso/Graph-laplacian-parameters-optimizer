import optuna
import numpy as np
from graphlaplacianoptimizer._build_direct import build_direct

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
        lambdas_raw = build_direct(params, items)
        
        if not lambdas_raw:
            raise optuna.TrialPruned()
            
        # Extract only the eigenvalues (index 0 of the tuple)
        lambdas = [lam[0] for lam in lambdas_raw]
        
        # We need at least 3 eigenvalues to calculate spectral gap (λ2 - λ1)
        # Prune if the graph is disconnected (λ1 == 0.0) or too small
        if len(lambdas) < 3 or lambdas[1] == 0.0:
            raise optuna.TrialPruned()
            
        # Maximize the normalized spectral gap: (λ2 - λ1) / λ1
        return (lambdas[2] - lambdas[1]) / lambdas[1]
        
    return objective