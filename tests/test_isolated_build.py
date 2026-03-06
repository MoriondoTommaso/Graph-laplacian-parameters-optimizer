import numpy as np
from graphlaplacianoptimizer._isolated_build import run_isolated_build

# A minimal but valid dataset: 10 items, 24 features, dtype=float64.
# dtype=float64 is mandatory for the Rust FFI bridge.
ITEMS = np.random.default_rng(42).random((10, 24)).astype(np.float64)

# A known-good param set.
# k=12 = n_features // 2, topk=3, p=2.0 (Euclidean), sigma moderate.
VALID_PARAMS = {
    "eps":   0.5,
    "k":     12,
    "topk":  3,
    "p":     2.0,
    "sigma": 0.3,
}


# --- Test 1: Happy path ---

def test_returns_lambdas_on_valid_input():
    result = run_isolated_build(VALID_PARAMS, ITEMS)

    # Subprocess must have exited cleanly and returned something.
    assert result is not None, "Expected lambdas list, got None"

    # Must be a plain Python list — no tuples, no numpy arrays.
    assert isinstance(result, list), f"Expected list, got {type(result)}"

    # Need at least 2 eigenvalues to compute fiedler + spectral gap.
    assert len(result) >= 2, f"Expected ≥2 lambdas, got {len(result)}"

    # Every element must be a plain float after tuple extraction in _isolated_build.
    for val in result:
        assert isinstance(val, float), f"Non-float lambda: {val}"


# --- Test 2: Degenerate params produce λ₁ = 0.0 ---

def test_degenerate_params_produce_zero_fiedler():
    # eps and sigma so small that the graph is effectively disconnected.
    # ArrowSpace does not crash — it returns a valid but trivial Laplacian
    # where λ₁ = 0.0 (the trivial eigenvalue indicating disconnection).
    degenerate_params = {
        "eps":   0.00001,
        "k":     12,
        "topk":  3,
        "p":     2.0,
        "sigma": 0.00001,
    }

    result = run_isolated_build(degenerate_params, ITEMS)

    # Must not crash — spawn isolation must hold even for degenerate inputs.
    assert result is not None, "Expected result, got None"

    # The first eigenvalue must be 0.0 — the signature of a disconnected graph.
    # This is the condition Optuna uses to prune bad trials.
    assert result[0] == 0.0, f"Expected fiedler=0.0, got {result[0]}"


# --- Test 3: Repeated calls do not crash (double-free guard) ---

def test_repeated_calls_do_not_crash():
    # 3 sequential calls with identical inputs.
    # If spawn isolation is broken, call 2 or 3 triggers:
    # "double free detected in tcache 2" — fatal, unrecoverable.
    results = []
    for i in range(3):
        result = run_isolated_build(VALID_PARAMS, ITEMS)
        results.append(result)

    for i, result in enumerate(results):
        assert result is not None, f"Call {i+1} returned None unexpectedly"
        assert len(result) >= 2, f"Call {i+1} returned fewer than 2 lambdas"
