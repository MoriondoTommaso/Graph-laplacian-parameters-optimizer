import optuna
import numpy as np
import warnings
from optuna.exceptions import TrialPruned

from graphlaplacianoptimizer.find_convergence import find_convergence_threshold_robust
from graphlaplacianoptimizer._build_direct import build_arrowspace

# Nascondi i warning di autovalori complessi se il grafo si rompe
warnings.filterwarnings("ignore")

def _evaluate_graph(arr: np.ndarray, params: dict, size: int) -> float:
    """Estrae un subset deterministico e calcola il punteggio topologico (Fiedler + Gap)."""
    rng = np.random.default_rng(42)
    indices = rng.choice(len(arr), size=size, replace=False)
    subset = arr[indices].astype(np.float64)

    try:
        aspace, _ = build_arrowspace(params, subset.copy())
        lams = [v for v, _ in aspace.lambdas_sorted()]
        
        if len(lams) > 1:
            fiedler = float(lams[1])
            spectral_gap = float(lams[1] - lams[0])
            return fiedler + spectral_gap
    except Exception:
        pass # Se la libreria Rust crasha per parametri assurdi, restituiamo 0
        
    return 0.0

# ── Optuna objective (Factory che accetta il base_size dinamico) ──────────────
def make_objective(arr: np.ndarray, dynamic_base_size: int):

    def objective(trial: optuna.Trial) -> float:
        params = {
            "eps":   trial.suggest_float("eps",   0.5, 15.0),
            "k":     trial.suggest_int  ("k",     10,  30),
            "topk":  trial.suggest_int  ("topk",  100, 500),
            "p":     trial.suggest_float("p",     1.0, 3.0),
            "sigma": trial.suggest_float("sigma", 0.2, 1.2),
        }

        # ── FASE 1: FAST PASS (Alla risoluzione minima del dataset) ───────────
        score_base = _evaluate_graph(arr, params, dynamic_base_size)

        # Se il grafo è disconnesso/spazzatura, uccidi il trial in 10 secondi
        if score_base < 1e-6:
            raise TrialPruned("Grafo disconnesso al Fast-Pass.")

        # Leggi il record attuale (Se è il primissimo trial, il record è -inf)
        try:
            best_score = trial.study.best_value
        except ValueError:
            best_score = float('-inf')

        # ── FASE 2: VALIDAZIONE AD ALTA RISOLUZIONE (Multi-Fidelity) ──────────
        # Se lo score batte o si avvicina al 90% del record attuale...
        if score_base >= (best_score * 0.90):
            print(f"\n🌟 Trial {trial.number} PROMETTENTE! Score Fast-Pass: {score_base:.5f} (Record: {best_score:.5f})")
            print("Avvio il calcolo della convergenza per validare questi parametri...")
            
            # Troviamo la convergenza ESATTA per questi nuovi parametri
            true_size = find_convergence_threshold_robust(
                arr           = arr,
                graph_params  = params,
                start_size    = dynamic_base_size, # Partiamo avvantaggiati!
                step_factor   = 1.5,
                tolerance_pct = 0.10,
                seed_spread   = 0.15,
                patience      = 2,
                n_seeds       = 2,
                max_size_cap  = 100_000,
            )
            
            trial.set_user_attr("convergence_size", true_size)
            
            # Ricalcoliamo lo score definitivo sulla taglia perfetta
            if true_size > dynamic_base_size:
                print(f"🔄 Ricalcolo lo score definitivo su {true_size:,} item...")
                final_score = _evaluate_graph(arr, params, true_size)
                return final_score
            else:
                return score_base
                
        else:
            # I parametri non sono competitivi. Scartiamoli per non inquinare la matematica di Optuna.
            raise TrialPruned("Score mediocre. Non merita il calcolo di convergenza.")

    return objective

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from benchmarks.cve_loader import load_cve

    arr = load_cve()
    
    print("==========================================================")
    print("FASE 0: PRE-FLIGHT CHECK (Calibrazione del Dataset)")
    print("Calcolo la risoluzione base (Fast-Pass) per questo dataset...")
    print("==========================================================")
    
    # Usiamo parametri "Averaged" generici per misurare il dataset a freddo
    baseline_params = {'eps': 3.18, 'k': 17, 'topk': 204, 'p': 2.24, 'sigma': 0.98}
    
    dynamic_base_size = find_convergence_threshold_robust(
        arr           = arr,
        graph_params  = baseline_params,
        start_size    = 8_000,
        step_factor   = 2.0,
        tolerance_pct = 0.10,
        seed_spread   = 0.15,
        patience      = 2,
        n_seeds       = 3,
        max_size_cap  = 100000,
    )
    
    # Margine di sicurezza: non esploriamo mai sotto i 16k se il dataset è minuscolo
    dynamic_base_size = max(dynamic_base_size, 16_000)
    print(f"\n🎯 Calibrazione Completata. Il Fast-Pass userà: {dynamic_base_size:,} item.\n")

    # Inizializza Optuna
    study = optuna.create_study(
        direction  = "maximize",
        study_name = "cve_topo_adaptive_v2",
        storage    = "sqlite:///study.db",
        load_if_exists=True,
    )

    # Inietta la taglia dinamica nella factory
    study.optimize(make_objective(arr, dynamic_base_size), n_trials=20)