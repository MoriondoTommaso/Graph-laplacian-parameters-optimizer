from arrowspace import ArrowSpaceBuilder
import traceback
import numpy as np
import time


def build_arrowspace(graph_params: dict,items: np.ndarray):
    t0 = time.perf_counter()
    builder = (ArrowSpaceBuilder()
            .with_dims_reduction(False, None)
            .with_sampling("simple", 1.0)
        )
    aspace, gl = builder.build(graph_params, items.copy())
    print(f"build time: {(time.perf_counter() - t0):2f}s ")
    return aspace, gl 


def build_direct(graph_params: dict, items: np.ndarray):
    try:
        print(f"Trial params: {graph_params}")  # debug
        aspace, _ = build_arrowspace(graph_params,items)
        lambdas_raw = aspace.lambdas_sorted()
        return lambdas_raw  # ← returns lambdas!
    except Exception as e:
        print(f"Build fail: {e}")
        return None