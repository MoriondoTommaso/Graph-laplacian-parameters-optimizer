from scipy.stats import spearmanr, kendalltau
import pytest
import numpy as np

from graphlaplacianoptimizer.search_metrics import compute_ranking_metrics, compute_ndcg, compute_mrr_top0, analyze_tail_distribution


def test_identical_rankings():
    results_a = [(1, 0.9), (2, 0.8), (3, 0.7)]
    results_b = [(1, 0.95), (2, 0.85), (3, 0.75)]

    rho, tau = compute_ranking_metrics(results_a, results_b)

    assert pytest.approx(rho) == 1.0
    assert pytest.approx(tau) == 1.0


def test_reversed_rankings():
    results_a = [(1, 0.9), (2, 0.8), (3, 0.7)]
    results_b = [(3, 0.95), (2, 0.85), (1, 0.75)]

    rho, tau = compute_ranking_metrics(results_a, results_b)

    assert pytest.approx(rho) == -1.0
    assert pytest.approx(tau) == -1.0


def test_ndcg_perfect_ranking():
    """Test ndcg score metrics ncdg = 1"""
    results_ref = [(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7)]
    results_pred = [(1, 0.99), (2, 0.95), (3, 0.9), (4, 0.85)]

    ndcg = compute_ndcg(results_pred, results_ref, k=4)

    assert pytest.approx(ndcg, rel=1e-5) == 1.0

def test_ndcg_reversed_ranking():
    """Test ndcg score metrics ncdg < 1"""
    results_ref = [(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7)]
    results_pred = [(4, 0.99), (3, 0.95), (2, 0.9), (1, 0.85)]

    ndcg = compute_ndcg(results_pred, results_ref, k=4)
    print(ndcg)
    assert ndcg < 1.0


def test_monotonic_decay():
    """Scores decrease smoothly from head to tail."""
    results = [(i, 10 - i) for i in range(10)]  # scores: 10..1
    metrics = analyze_tail_distribution([results], ["model"], k_head=3, k_tail=10)

    m = metrics["model"]

    assert pytest.approx(m["head_mean"]) == np.mean([10, 9, 8])
    assert pytest.approx(m["tail_mean"]) == np.mean([7, 6, 5, 4, 3, 2, 1])
    assert m["tail_std"] > 0
    assert m["tail_decay_rate"] > 0
    assert m["n_tail_items"] == 7
    assert m["total_items"] == 10


def test_flat_scores():
    """All scores identical → std=0 and cv=0."""
    results = [(i, 5.0) for i in range(10)]
    metrics = analyze_tail_distribution([results], ["model"], k_head=3, k_tail=10)

    m = metrics["model"]

    assert m["head_mean"] == 5.0
    assert m["tail_mean"] == 5.0
    assert m["tail_std"] == 0.0
    assert m["tail_cv"] == 0.0
    assert m["tail_to_head_ratio"] == 1.0


def test_tail_higher_than_head():
    """Tail scores larger than head."""
    results = [(0,1),(1,2),(2,3),(3,10),(4,9),(5,8),(6,7)]
    metrics = analyze_tail_distribution([results], ["model"], k_head=3, k_tail=7)

    m = metrics["model"]

    assert m["tail_mean"] > m["head_mean"]
    assert m["tail_to_head_ratio"] > 1


def test_multiple_models():
    """Function handles multiple result lists."""
    r1 = [(i, 10 - i) for i in range(10)]
    r2 = [(i, i) for i in range(10)]

    metrics = analyze_tail_distribution(
        [r1, r2],
        ["model_a", "model_b"],
        k_head=3,
        k_tail=10,
    )

    assert "model_a" in metrics
    assert "model_b" in metrics
    assert metrics["model_a"]["head_mean"] != metrics["model_b"]["head_mean"]


# -----------------------------
# Edge cases
# -----------------------------

def test_results_too_short():
    """If list shorter than head size → empty dict."""
    results = [(0, 1.0), (1, 0.9)]

    metrics = analyze_tail_distribution([results], ["model"], k_head=3)

    assert metrics == {}


def test_zero_head_mean():
    """Head scores near zero → ratio must safely return 0."""
    results = [(0,0.0),(1,0.0),(2,0.0),(3,5.0),(4,4.0),(5,3.0)]

    metrics = analyze_tail_distribution([results], ["model"], k_head=3, k_tail=6)

    m = metrics["model"]

    assert m["head_mean"] == 0.0
    assert m["tail_to_head_ratio"] == 0.0