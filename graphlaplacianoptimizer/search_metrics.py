

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score
import logging

logging.basicConfig(level=logging.DEBUG)

# ============================================================================
# Metrics
# ============================================================================
def compute_ranking_metrics(results_a, results_b):
    """Spearman ρ and Kendall τ between two ranked lists."""
    indices_a = [idx for idx, _ in results_a]
    indices_b = [idx for idx, _ in results_b]
    shared = set(indices_a) & set(indices_b)
    if len(shared) < 2:
        return 0.0, 0.0
    rank_a = [indices_a.index(idx) for idx in shared]
    rank_b = [indices_b.index(idx) for idx in shared]
    spearman_rho, _ = spearmanr(rank_a, rank_b)
    kendall_tau_v, _ = kendalltau(rank_a, rank_b)
    return spearman_rho, kendall_tau_v


def compute_ndcg(results_pred, results_ref, k=10):
    """NDCG@k treating reference ranking as ground truth."""
    ref_indices    = [idx for idx, _ in results_ref[:k]]
    relevance_map  = {idx: k - i for i, idx in enumerate(ref_indices)}
    pred_indices   = [idx for idx, _ in results_pred[:k]]
    true_relevance = [relevance_map.get(idx, 0) for idx in pred_indices]

    if sum(true_relevance) == 0:
        return 0.0
    try:
        pred_scores = np.array([score for _, score in results_pred[:k]])
        if pred_scores.max() > 0:
            pred_scores = pred_scores / pred_scores.max()
        return ndcg_score(
            np.array([true_relevance]).reshape(1, -1),
            np.array([pred_scores]).reshape(1, -1),
            k=k,
        )
    except Exception:
        return 0.0



def compute_mrr_top0(lmb_sorted: list[tuple[int, float]]) -> float:
    """
    Need fix.

    MRR-Top0: topology-weighted reciprocal rank over the full top-k.

    Formula (label-agnostic, all returned items are Rel(q)):
        MRR-Top0 = Σ_{i ∈ results} T_{q,i} / rank(i)

    Normalised by the number of items so scores are comparable across
    queries with different result-set sizes.

    Parameters
    ----------
    lmb_sorted     : list[(item_idx, score)]  ranked from best to worst

    Returns
    -------
    float
    """
    if not lmb_sorted:
        return 0.0
    total = 0.0
    for rank, (_, T_qi) in enumerate(lmb_sorted):
        total += T_qi / float(rank)
    return total / len(lmb_sorted)


def analyze_tail_distribution(results_list, labels, k_head=3, k_tail=20):
    """Score distribution statistics for head vs tail positions."""
    min_length = min(len(r) for r in results_list)
    if min_length <= k_head:
        return {}

    actual_k_tail = min(k_tail, min_length)
    metrics = {}

    for results, label in zip(results_list, labels):
        seg         = results[:actual_k_tail]
        head_scores = [s for _, s in seg[:k_head]]
        tail_scores = [s for _, s in seg[k_head:actual_k_tail]]

        if not tail_scores or not head_scores:
            continue

        tail_mean = np.mean(tail_scores)
        tail_std  = np.std(tail_scores)
        head_mean = np.mean(head_scores)

        metrics[label] = {
            "head_mean":          head_mean,
            "tail_mean":          tail_mean,
            "tail_std":           tail_std,
            "tail_to_head_ratio": tail_mean / head_mean if head_mean > 1e-10 else 0.0,
            "tail_cv":            tail_std / tail_mean if tail_mean > 1e-10 else 0.0,
            "tail_decay_rate":    (tail_scores[0] - tail_scores[-1]) / len(tail_scores)
                                  if len(tail_scores) > 1 else 0.0,
            "n_tail_items":       len(tail_scores),
            "total_items":        actual_k_tail,
        }
    return metrics


