from arrowspace import ArrowSpaceBuilder
import traceback
import numpy as np
def build_direct(graph_params: dict, items: np.ndarray):
    try:
        print(f"Trial params: {graph_params}")  # debug
        builder = (ArrowSpaceBuilder()
            .with_dims_reduction(False, None)
            .with_sampling("simple", 1.0)
        )
        aspace, _gl = builder.build(graph_params, items.copy())
        lambdas_raw = aspace.lambdas_sorted()
        return lambdas_raw  # ← returns lambdas!
    except Exception as e:
        print(f"Build fail: {e}")
        return None
