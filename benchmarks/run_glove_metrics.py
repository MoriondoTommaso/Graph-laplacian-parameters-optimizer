"""
GloVe-25 multi-metric benchmark for ArrowSpace.
Metrics: Recall@10, MRR-Top0, NDCG@10, Spearman ρ, Kendall τ, Tail/Head ratio.
Three tau modes: 1.0 (cosine), 0.72 (hybrid), 0.42 (taumode/spectral).
"""
import os
import time
import pickle
import tempfile
import multiprocessing as mp
import numpy as np
import csv
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score

from benchmarks.loader_glove import load_corpus, load_queries


GLOVE_PATH = "data/glove-25-angular.hdf5"
N_CORPUS   = 1_183_514
N_QUERIES  = 50
K          = 10

TAU_COSINE  = 1.0
TAU_HYBRID  = 0.72
TAU_TAUMODE = 0.42
TAU_LABELS  = [f"Cosine(τ={TAU_COSINE})", f"Hybrid(τ={TAU_HYBRID})", f"Taumode(τ={TAU_TAUMODE})"]

GRAPH_PARAMS = {
    "eps":   5.0,
    "k":     12,
    "topk":  100,
    "p":     2.0,
    "sigma": 1.5,
}


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(corpus: np.ndarray, queries: np.ndarray, result_path: str):
    from arrowspace import ArrowSpaceBuilder

    index, gl = ArrowSpaceBuilder().build(GRAPH_PARAMS, corpus.copy())

    raw     = index.lambdas()
    lambdas = np.clip(
        np.array([v[0] if isinstance(v, tuple) else v for v in raw]),
        1e-9, None
    )
    topo_scores = {i: float(lambdas[i]) for i in range(len(lambdas))}

    all_tau = []
    for tau in (TAU_COSINE, TAU_HYBRID, TAU_TAUMODE):
        tau_results = []
        for q in queries:
            tau_results.append(index.search(q.copy(), gl, tau))
        all_tau.append(tau_results)

    with open(result_path, "wb") as f:
        pickle.dump((all_tau, topo_scores), f)


# ── Metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(retrieved, ground_truth_row, k):
    gt = set(ground_truth_row[:k].tolist())
    return len({idx for idx, _ in retrieved[:k]} & gt) / k


def compute_mrr_top0(results, topo_scores):
    if not results:
        return 0.0
    return sum(topo_scores.get(idx, 0.0) / r
               for r, (idx, _) in enumerate(results, 1)) / len(results)


def compute_ndcg(results_pred, results_ref, k=10):
    ref_map    = {idx: k - i for i, (idx, _) in enumerate(results_ref[:k])}
    pred_idx   = [idx for idx, _ in results_pred[:k]]
    true_rel   = [ref_map.get(idx, 0) for idx in pred_idx]
    if sum(true_rel) == 0:
        return 0.0
    pred_scores = np.array([s for _, s in results_pred[:k]])
    if pred_scores.max() > 0:
        pred_scores /= pred_scores.max()
    try:
        return ndcg_score(np.array([true_rel]).reshape(1, -1),
                          pred_scores.reshape(1, -1), k=k)
    except Exception:
        return 0.0


def compute_rank_corr(results_a, results_b):
    idx_a  = [idx for idx, _ in results_a]
    idx_b  = [idx for idx, _ in results_b]
    shared = list(set(idx_a) & set(idx_b))
    if len(shared) < 2:
        return 0.0, 0.0
    ra = [idx_a.index(i) for i in shared]
    rb = [idx_b.index(i) for i in shared]
    rho, _ = spearmanr(ra, rb)
    tau, _ = kendalltau(ra, rb)
    return rho, tau


def tail_head_ratio(results, k_head=3):
    scores = [s for _, s in results]
    if len(scores) <= k_head:
        return 0.0
    head = np.mean(scores[:k_head])
    tail = np.mean(scores[k_head:])
    return tail / head if head > 1e-10 else 0.0


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_summary_csv(rows, path="glove_summary.csv"):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "tau", "mean", "std"])
        w.writeheader()
        w.writerows(rows)
    print(f"Summary saved → {path}")


def save_per_query_csv(records, path="glove_per_query.csv"):
    if not records:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"Per-query saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {N_CORPUS:,} corpus vectors and {N_QUERIES} queries...")
    corpus  = load_corpus(GLOVE_PATH, n_items=N_CORPUS).astype(np.float64)
    queries, ground_truth = load_queries(GLOVE_PATH, n_queries=N_QUERIES)
    queries = queries.astype(np.float64)

    print("Building index (isolated worker)...")
    ctx         = mp.get_context("spawn")
    result_path = tempfile.mktemp(suffix=".pkl")
    p           = ctx.Process(target=_worker, args=(corpus, queries, result_path))

    t0 = time.perf_counter()
    p.start()
    p.join(timeout=600)
    elapsed = time.perf_counter() - t0

    if p.is_alive():
        p.kill()
        raise RuntimeError("Worker timed out after 600s")
    if p.exitcode != 0:
        raise RuntimeError(f"Worker exited with code {p.exitcode}")
    if not os.path.exists(result_path):
        raise RuntimeError("Result file missing — silent crash")

    with open(result_path, "rb") as f:
        all_tau, topo_scores = pickle.load(f)
    os.unlink(result_path)
    print(f"Build + search done in {elapsed:.1f}s\n")

    accum = {label: {"recall": [], "mrr": [], "ndcg": [],
                     "spearman": [], "kendall": [], "tail_head": []}
             for label in TAU_LABELS}
    per_query = []

    for qi in range(N_QUERIES):
        res_cos = all_tau[0][qi]
        res_hyb = all_tau[1][qi]
        res_tau = all_tau[2][qi]
        gt_row  = ground_truth[qi]

        for label, res in zip(TAU_LABELS, [res_cos, res_hyb, res_tau]):
            rec        = recall_at_k(res, gt_row, K)
            mrr        = compute_mrr_top0(res, topo_scores)
            ndcg       = compute_ndcg(res, res_cos, k=K) if label != TAU_LABELS[0] else None
            rho, tau_k = compute_rank_corr(res, res_cos)
            thr        = tail_head_ratio(res)

            accum[label]["recall"].append(rec)
            accum[label]["mrr"].append(mrr)
            if ndcg is not None:
                accum[label]["ndcg"].append(ndcg)
            accum[label]["spearman"].append(rho)
            accum[label]["kendall"].append(tau_k)
            accum[label]["tail_head"].append(thr)

            per_query.append({
                "query_id":  qi + 1,
                "tau":       label,
                "recall@10": f"{rec:.4f}",
                "mrr_top0":  f"{mrr:.4f}",
                "ndcg@10":   f"{ndcg:.4f}" if ndcg is not None else "ref",
                "spearman":  f"{rho:.4f}",
                "kendall":   f"{tau_k:.4f}",
                "tail_head": f"{thr:.4f}",
            })

    metric_keys = {
        "recall":    "Recall@10",
        "mrr":       "MRR-Top0",
        "ndcg":      "NDCG@10 vs cos",
        "spearman":  "Spearman ρ",
        "kendall":   "Kendall τ",
        "tail_head": "Tail/Head ratio",
    }

    print(f"{'Metric':<20} {TAU_LABELS[0]:>18} {TAU_LABELS[1]:>18} {TAU_LABELS[2]:>18}")
    print("-" * 76)

    summary_rows = []
    for key, label_str in metric_keys.items():
        row = f"{label_str:<20}"
        for label in TAU_LABELS:
            vals = accum[label][key]
            if vals:
                m, s = np.mean(vals), np.std(vals)
                row += f"  {m:.4f}±{s:.4f}"
                summary_rows.append({"metric": label_str, "tau": label,
                                     "mean": f"{m:.6f}", "std": f"{s:.6f}"})
            else:
                row += f"  {'ref':>13}"
                summary_rows.append({"metric": label_str, "tau": label,
                                     "mean": "ref", "std": "ref"})
        print(row)

    save_summary_csv(summary_rows)
    save_per_query_csv(per_query)


if __name__ == "__main__":
    main()
