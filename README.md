
# graphlaplacianoptimizer

A Bayesian optimization pipeline for tuning the graph construction parameters
of the Laplacian matrix in [pyarrowspace](https://github.com/tuned-org-uk/pyarrowspace),
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

This project uses [Optuna](https://optuna.org/) (TPE sampler) to find the
combination of these parameters that maximises an intrinsic spectral quality
score derived from the graph Laplacian's eigenstructure.

## Mathematical Objective

The optimizer maximises:

```
score = λ₁ + (λ₂ - λ₁) = λ₂
```

where `λ₁` is the **Fiedler value** (graph connectivity robustness) and
`λ₂ - λ₁` is the **spectral gap** (cluster separation quality on the manifold).
Both are read from `aspace.lambdas_sorted()` after each graph build.

Trials where `λ₂ = 0.0` (disconnected graph — multiple components) are
pruned immediately so Optuna avoids that region of parameter space.

## Requirements

- Linux (Ubuntu)
- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
git clone https://github.com/MoriondoTommaso/Graph-laplacian-parameters-optimizer.git
cd arrowspace-opt
uv sync
uv pip install -e .
```

## Verify FFI Bridge

Before running anything, confirm the Rust backend is linked correctly:

```bash
uv run python -c "import numpy as np; from arrowspace import ArrowSpaceBuilder, GraphLaplacian; print('FFI OK')"
```

Expected output: `FFI OK`

## Run the Optimizer

```bash
uv run graphlaplacianoptimizer/_optimizer.py
```

Results are saved to `study.db` (SQLite). Restarting the command resumes
from where it left off — trials accumulate across runs.

## Run the Tests

```bash
uv run pytest tests/ -v
```

Expected: 9/9 passing in ~3.5 seconds.

## Project Structure

```
arrowspace-opt/
├── graphlaplacianoptimizer/
│   ├── __init__.py              # empty
│   ├── _isolated_build.py       # Rust FFI worker — spawn-safe subprocess
│   ├── _objective.py            # Optuna trial logic and spectral scoring
│   └── _optimizer.py            # Entry point: study creation and execution
├── tests/
│   ├── test_isolated_build.py   # 3 tests: FFI safety, degenerate params, double-free guard
│   ├── test_objective.py        # 3 tests: callable, positive score, pruning
│   └── test_optimizer.py        # 3 tests: dataset shape/dtype, reproducibility, end-to-end
├── docs/
│   ├── mrr-top0-paper.pdf
│   └── Topological Transformer-uploaded version.pdf
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

## Rust FFI Safety Rules

The `pyarrowspace` Rust backend allocates memory outside Python's GC.
Three rules are enforced throughout this codebase:

1. **Never** pass a Python `list` to `ArrowSpaceBuilder` — always `np.ndarray`
   with `dtype=np.float64`
2. **Always** call `items.copy()` before passing the array to the builder
3. **Always** run the builder inside a `multiprocessing` worker using the
   `"spawn"` context (`_isolated_build.py`) — never in the main thread loop

Violating any of these causes a fatal `double free detected in tcache 2` crash.

## Key Technical Notes

- `lambdas_sorted()` returns `list[tuple[float, int]]` — eigenvalue + original index
- `λ₁ = 0.0` is always present (trivial eigenvalue) — normal for any valid Laplacian
- `λ₂ = 0.0` signals a disconnected graph — this is the pruning condition
- Test dataset: 51 clustered items (3 centroids × 17 points, Gaussian noise std=0.05)

Yes, good idea. Here is the section to add to `README.md` — insert it after the **Project Structure** section and before the **Rust FFI Safety Rules** section:


## How the Pipeline Works

The three modules form a strict one-way dependency chain:

### `_isolated_build.py` — The Safety Boundary
Every call to the Rust backend happens here and only here. When the optimizer
needs to evaluate a set of graph parameters, it spawns a completely fresh
Python subprocess using the `"spawn"` context. Inside that subprocess,
`ArrowSpaceBuilder().build()` is called with `items.copy()` to prevent
memory ownership conflicts between Python's GC and the Rust allocator.
The subprocess returns the Laplacian eigenvalues (`lambdas_sorted()`) via
a `multiprocessing.Queue` as plain Python floats, then exits cleanly.
The main process never touches Rust memory directly.

```
run_isolated_build(graph_params, items)
    │
    └── spawns child process
            │
            └── ArrowSpaceBuilder().build(graph_params, items.copy())
                    │
                    └── returns [λ₀, λ₁, λ₂, ...] via Queue
```

### `_objective.py` — The Scoring Logic
Wraps the isolated build in an Optuna-compatible objective function.
For each trial, it asks Optuna to suggest values for `eps`, `k`, `topk`,
`p`, and `sigma`, passes them to `run_isolated_build()`, and scores the
result using the Fiedler value and spectral gap:

```
score = λ₁ + (λ₂ - λ₁)
```

If the build returns `λ₂ = 0.0` (disconnected graph), the trial is pruned
immediately — Optuna learns to avoid that region of parameter space.

### `_optimizer.py` — The Entry Point
Creates the synthetic dataset, initialises the Optuna study with SQLite
persistence, and runs the optimization loop. The TPE sampler explores
randomly for the first 10 trials, then switches to Bayesian inference —
using the scores of past trials to pick parameter values more likely to
produce high-quality graph topologies.

```
main()
  │
  ├── make_synthetic_dataset()     → 51 clustered items, np.float64
  ├── optuna.create_study()        → loads or creates study.db
  └── study.optimize(objective)    → 50 trials, TPE sampler
          │
          └── each trial calls _objective.py → _isolated_build.py → Rust
```
**SIFT‑128‑Euclidean dataset link** + **updated documentation**:

## Dataset SIFT‑1M (your file)

```
data/sift-128-euclidean.hdf5
├── train: (1,000,000, 128) float32 → your subsets
├── query: (10,000, 128) 
├── neighbors: (10,000, 100) int32 ground truth
└── distances: (10,000, 100) float32
```

**Direct download** (backup):
```
https://ann-benchmarks.com/datasets/sift-128-euclidean.hdf5  # 501MB [web:6]
# or
huggingface.co/datasets/open-vdb/sift-128-euclidean [web:149]
```

## Updated README.md section

# Graph Laplacian Parameter Optimizer
Bayesian tuning of ArrowSpace graphs for spectral topology (Fiedler + gap).

## Stage A: Spectral Topology ✓ VALIDATED

### SIFT‑128‑Euclidean Results

| Dataset | Baseline λ₂ | Best λ₂ | Improvement | Best Params |
|---------|-------------|---------|-------------|-------------|
| 1k      | 5.8e-5     | **0.202** | **347x** | eps=0.027, k=20 [file:120] |
| 4k      | 5.6e-5     | **0.049** | **86x**  | eps=0.050, k=24 [file:135] |

**Pattern**: Optuna finds tight `eps`/`sigma` → dense/connected graphs.

### Scaling Limits
```
4k: ✅ Stable (20 trials ~10min)
5k+: ⚠️ Rust sparse heap (~40k edges)
1M: 🔄 Sharding 250×4k
```

### Production Workflow
```
1. Tune 4k subset (10min)
2. Best params → 1M full build (sharding overnight)
3. Precompute lambda scalars → O(1) tau-mode query
```

## Usage
```bash
uv run python test_sift_topology.py  # Auto-save results/
```

## Next: Stage 2 Retrieval 
MRR-Top0/NDCG/tail metrics with best spectral params vs cosine baseline.


## Verify dataset

```bash
# File info
h5dump -H data/sift-128-euclidean.hdf5 | head -20

# Fresh download 
wget https://ann-benchmarks.com/datasets/sift-128-euclidean.hdf5 -O data/sift-backup.hdf5

## References

- [ArrowSpace — Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.09002.pdf)
- [MRR-Top0: Topology-Aware MRR Extension](docs/mrr-top0-paper.pdf)




