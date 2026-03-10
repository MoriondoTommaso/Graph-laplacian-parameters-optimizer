from graphlaplacianoptimizer._isolated_build import run_isolated_build


def compute_spectral_diag(graph_params, items):
    """FFI-safe spectral diagnostics."""
    print(f"DEBUG: Testing params={graph_params}, items.shape={items.shape}")
    lambdas = run_isolated_build(graph_params, items)
    
    if lambdas is None:
        print("DEBUG: run_isolated_build returned None (OOM/timeout/crash)")
        return {"status": "failed", "fiedler": 0.0, "spectral_gap": 0.0, "score": -1e6, "prefix": []}
    
    if len(lambdas) < 2 or lambdas[1] == 0.0:
        print(f"DEBUG: Bad spectrum: len={len(lambdas)}, λ₂={lambdas[1] if len(lambdas)>1 else 'N/A'}")
        return {"status": "bad_spectrum", "fiedler": 0.0, "spectral_gap": 0.0, "score": -1e6, "prefix": []}
    
    fiedler = lambdas[0]
    gap = lambdas[1] - lambdas[0]
    
    return {
        "status": "success",
        "fiedler": fiedler,
        "spectral_gap": gap,
        "score": fiedler + gap,
        "prefix": lambdas[:5],
    }
