"""
GloVe-25 Bayesian optimization for ArrowSpace.

Objective (maximise):
    L = 0.5 * NDCG@10  +  0.3 * Tail/Head  +  0.2 * Fiedler_norm
    → 0.0  if graph is disconnected (λ₂ = 0.0)
    → 0.0  if NDCG@10 < NDCG_FLOOR

Search space:
    eps   ∈ [1.0, 8.0]   float
    k     ∈ [5,   30]    int
    sigma ∈ [0.1, 3.0]   float
    tau   ∈ [0.3, 1.0]   float
    p     = 2.0           fixed

Usage:
    uv run benchmarks/optimize_glove.py
    uv run benchmarks/optimize_glove.py --trials 50 --queries 10
"""
import os
import time
import pickle
import tempfile
import argparse
import multiprocessing as mp

import numpy as np
import optuna
from optuna.samplers import TPESampler

from benchmarks.loader_glove import load_corpus, load_queries

# ── Constants (module-level → patchable by tests) ─────────────────────────────
GLOVE_PATH     = "data/glove-25-angular.hdf5"
STORAGE        = "sqlite:///glove_study.db"
N_TRIALS       = 100
N_OPT_QUERIES  = 10      # queries per trial (speed vs accuracy trade-off)
N_CORPUS       = 1_183_514
NDCG_FLOOR     = 0.85    # hard floor — trials below this score 0.0
K              = 10

# Objective weights
W_NDCG    = 0.5
W_TAIL    = 0.3
W_FIEDLER = 0.2

# Fixed params
P_FIXED = 2.0

# Search space bounds
EPS_LOW,   EPS_HIGH   = 1.0, 8.0
K_LOW,     K_HIGH     = 5,   30
SIGMA_LOW, SIGMA_HIGH = 0.1, 3.0
TAU_LOW,   TAU_HIGH   = 0.3, 1.0


# ── Spawn worker ──────────────────────────────────────────────────────────────

def _worker(graph_params: dict, tau: float,
            corpus: np.ndarray, queries: np.ndarray,
            result_path: str):
    """
    Isolated spawn worker.
    Builds the ArrowSpace index, extracts Fiedler λ₂, runs searches,
    computes NDCG@10 and Tail/Head ratio, writes results to pickle.
    """
    from arrowspace import ArrowSpaceBuilder
    from sklearn.metrics import ndcg_score as sklearn_ndcg

    # ── Build ──────────────────────────────────────────────────────────────
    index, gl = ArrowSpaceBuilder().build(graph_params, corpus.copy())

    # ── Fiedler (λ₂) ──────────────────────────────────────────────────────
    raw     = index.lambdas()
    lambdas = np.array([v[0] if isinstance(v, tuple) else v for v in raw])
    lambdas_sorted = np.sort(lambdas)

    # λ₀ is always 0.0 (trivial); λ₁ is Fiedler
    fiedler = float(lambdas_sorted[1]) if len(lambdas_sorted) > 1 else 0.0
    disconnected = fiedler == 0.0

    if disconnected:
        with open(result_path, "wb") as f:
            pickle.dump({"disconnected": True}, f)
        return

    # Normalise Fiedler to [0, 1] using max eigenvalue as reference
    lambda_max   = float(lambdas_sorted[-1]) if lambdas_sorted[-1] > 1e-9 else 1.0
    fiedler_norm = min(fiedler / lambda_max, 1.0)

    # Topo scores for MRR (reuse lambdas clipped)
    lambdas_clipped = np.clip(lambdas, 1e-9, None)
    topo_scores     = {i: float(lambdas_clipped[i]) for i in range(len(lambdas_clipped))}

    # ── Search at cosine (reference) and trial tau ─────────────────────────
    ndcg_vals  = []
    tail_vals  = []

    for q in queries:
        res_cos = index.search(q.copy(), gl, tau=1.0)
        res_tau = index.search(q.copy(), gl, tau=tau)

        # NDCG@K — tau results vs cosine reference
        ref_map    = {idx: K - i for i, (idx, _) in enumerate(res_cos[:K])}
        pred_idx   = [idx for idx, _ in res_tau[:K]]
        true_rel   = [ref_map.get(idx, 0) for idx in pred_idx]

        if sum(true_rel) > 0:
            pred_scores = np.array([s for _, s in res_tau[:K]])
            if pred_scores.max() > 0:
                pred_scores /= pred_scores.max()
            try:
                ndcg = sklearn_ndcg(
                    np.array([true_rel]).reshape(1, -1),
                    pred_scores.reshape(1, -1), k=K
                )
            except Exception:
                ndcg = 0.0
        else:
            ndcg = 0.0
        ndcg_vals.append(ndcg)

        # Tail/Head ratio
        scores  = [s for _, s in res_tau]
        k_head  = 3
        if len(scores) > k_head:
            head = np.mean(scores[:k_head])
            tail = np.mean(scores[k_head:])
            thr  = tail / head if head > 1e-10 else 0.0
        else:
            thr = 0.0
        tail_vals.append(thr)

    with open(result_path, "wb") as f:
        pickle.dump({
            "disconnected": False,
            "fiedler_norm": fiedler_norm,
            "ndcg":         float(np.mean(ndcg_vals)),
            "tail_head":    float(np.mean(tail_vals)),
        }, f)


def _run_worker(graph_params, tau, corpus, queries):
    """Launch worker in spawn context, return result dict or raise."""
    ctx         = mp.get_context("spawn")
    result_path = tempfile.mktemp(suffix=".pkl")
    p           = ctx.Process(target=_worker,
                               args=(graph_params, tau, corpus, queries, result_path))
    p.start()
    p.join(timeout=300)

    if p.is_alive():
        p.kill()
        raise RuntimeError("Worker timed out")
    if p.exitcode != 0:
        raise RuntimeError(f"Worker exited with code {p.exitcode}")
    if not os.path.exists(result_path):
        raise RuntimeError("Result file missing after worker exit")

    with open(result_path, "rb") as f:
        result = pickle.load(f)
    os.unlink(result_path)
    return result


# ── Objective ─────────────────────────────────────────────────────────────────

def make_objective(corpus: np.ndarray, queries: np.ndarray):
    """
    Returns the Optuna objective closure.
    Corpus and queries are captured once — not re-loaded per trial.
    """
    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ─────────────────────────────────────────
        eps   = trial.suggest_float("eps",   EPS_LOW,   EPS_HIGH)
        k     = trial.suggest_int  ("k",     K_LOW,     K_HIGH)
        sigma = trial.suggest_float("sigma", SIGMA_LOW, SIGMA_HIGH)
        tau   = trial.suggest_float("tau",   TAU_LOW,   TAU_HIGH)

        graph_params = {
            "eps":   eps,
            "k":     k,
            "topk":  100,
            "p":     P_FIXED,
            "sigma": sigma,
        }

        t0 = time.perf_counter()

        try:
            result = _run_worker(graph_params, tau, corpus, queries)
        except RuntimeError as e:
            print(f"  [trial {trial.number}] worker error: {e}")
            return 0.0

        elapsed = time.perf_counter() - t0

        # ── Disconnected graph → prune immediately ─────────────────────────
        if result["disconnected"]:
            print(f"  [trial {trial.number}] disconnected graph → score=0.0")
            raise optuna.TrialPruned()

        ndcg        = result["ndcg"]
        tail_head   = result["tail_head"]
        fiedler_norm = result["fiedler_norm"]

        # ── Hard NDCG floor ────────────────────────────────────────────────
        if ndcg < NDCG_FLOOR:
            print(f"  [trial {trial.number}] NDCG={ndcg:.4f} < floor={NDCG_FLOOR} → score=0.0"
                  f"  (eps={eps:.2f}, k={k}, sigma={sigma:.3f}, tau={tau:.3f})")
            return 0.0

        # ── Composite score ────────────────────────────────────────────────
        score = W_NDCG * ndcg + W_TAIL * tail_head + W_FIEDLER * fiedler_norm

        print(f"  [trial {trial.number:>3}] "
              f"eps={eps:.2f} k={k:>2} sigma={sigma:.3f} tau={tau:.3f} | "
              f"NDCG={ndcg:.4f} T/H={tail_head:.4f} λ₂={fiedler_norm:.4f} | "
              f"score={score:.4f}  ({elapsed:.1f}s)")

        return score

    return objective


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_trials: int, n_queries: int):
    print(f"Loading corpus ({N_CORPUS:,} vectors)...")
    corpus  = load_corpus(GLOVE_PATH, n_items=N_CORPUS).astype(np.float64)

    print(f"Loading {n_queries} queries...")
    queries, _ = load_queries(GLOVE_PATH, n_queries=n_queries)
    queries     = queries.astype(np.float64)

    print(f"\nObjective weights: NDCG×{W_NDCG}  Tail/Head×{W_TAIL}  Fiedler×{W_FIEDLER}")
    print(f"NDCG floor: {NDCG_FLOOR}  |  Trials: {n_trials}  |  Storage: {STORAGE}\n")

    sampler = TPESampler(n_startup_trials=10, seed=42)
    study   = optuna.create_study(
        study_name="glove25_arrowspace",
        storage=STORAGE,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    objective = make_objective(corpus, queries)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── Report best ───────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST TRIAL:  #{best.number}  score={best.value:.4f}")
    print(f"  eps   = {best.params['eps']:.4f}")
    print(f"  k     = {best.params['k']}")
    print(f"  sigma = {best.params['sigma']:.4f}")
    print(f"  tau   = {best.params['tau']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=N_TRIALS,      help="Number of Optuna trials")
    parser.add_argument("--queries", type=int, default=N_OPT_QUERIES,  help="Queries per trial")
    args = parser.parse_args()
    main(args.trials, args.queries)
