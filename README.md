
# graphlaplacianoptimizer

A Bayesian optimization pipeline for tuning the graph parameters of the
Laplacian matrix in [pyarrowspace](https://github.com/tuned-org-uk/pyarrowspace),
a spectral vector database that uses graph Laplacian eigenstructures for
topology-aware vector search.

## What It Does

`pyarrowspace` builds a graph Laplacian over the feature space of a vector
dataset. The quality of that Laplacian — and therefore the quality of
retrieval — depends on five graph construction parameters:

| Parameter | Role |
|-----------|------|
| `eps`     | Edge creation radius (ε-graph threshold) |
| `k`       | kNN neighbour count for graph wiring |
| `topk`    | Number of results returned per search |
| `p`       | Minkowski distance exponent |
| `sigma`   | Gaussian kernel bandwidth for edge weights |

This project uses [Optuna](https://optuna.org/) to find the combination of
these parameters that maximises an intrinsic spectral quality score derived
from the Fiedler value and spectral gap of the resulting Laplacian.

## Mathematical Objective

The optimizer maximises:

```
score = fiedler_value + spectral_gap
```

where `fiedler_value = λ₁` (the smallest non-zero eigenvalue) and
`spectral_gap = λ₂ - λ₁`, both read from `aspace.lambdas_sorted()`.
A large Fiedler value indicates a well-connected, robust graph.
A large spectral gap indicates well-separated clusters on the manifold.

## Requirements

- Linux (Ubuntu)
- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
git clone https://github.com/MoriondoTommaso/Graph-laplacian-parameters-optimizer.git
cd graphlaplacianoptimizer
uv sync
```

## Verify FFI Bridge

Before running anything, confirm the Rust backend is linked correctly:

```bash
uv run python -c "import numpy as np; from arrowspace import ArrowSpaceBuilder, GraphLaplacian; print('FFI OK')"
```

Expected output: `FFI OK`

## Project Structure

```
graphlaplacianoptimizer/
├── __init__.py
├── _isolated_build.py   # Rust FFI worker — runs in a spawned subprocess
├── _objective.py        # Optuna trial logic and intrinsic scoring
└── _optimizer.py        # Entry point: study creation and execution
```

## Safety Constraints

The `pyarrowspace` Rust backend allocates memory outside Python's GC.
Three rules are enforced throughout this codebase:

1. **Never** pass a Python `list` to `ArrowSpaceBuilder` — always `np.ndarray`
   with `dtype=np.float64`.
2. **Always** call `items.copy()` before passing the array to the builder.
3. **Always** run the builder inside a `multiprocessing` worker using the
   `"spawn"` context (`_isolated_build.py`). Never call the Rust builder
   directly from the main thread in a loop.

Violating any of these can cause a fatal `double free detected in tcache 2`
crash.

## References

- [ArrowSpace — Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.09002.pdf)
- [MRR-Top0: Topology-Aware MRR Extension](docs/mrr-top0-paper.pdf)


