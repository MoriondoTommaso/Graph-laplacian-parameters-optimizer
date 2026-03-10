# Graph Laplacian Parameter Optimizer

A Bayesian optimization pipeline for tuning the graph construction parameters of the Laplacian matrix in **pyarrowspace**, a spectral vector database that uses graph Laplacian eigenstructures for topology-aware vector search.

---

## 📖 Overview

**pyarrowspace** builds a graph Laplacian over the feature space of a vector dataset. The quality of that Laplacian — and therefore the quality of retrieval — depends on five critical graph construction parameters:

| Parameter | Role |
| --- | --- |
| `eps` | Edge creation radius ($\epsilon$-graph threshold) |
| `k` | kNN neighbor count for graph wiring |
| `topk` | Number of results returned per search |
| `p` | Minkowski distance exponent |
| `sigma` | Gaussian kernel bandwidth for edge weights |

This project uses **Optuna (TPE sampler)** to find the combination of these parameters that maximizes an intrinsic spectral quality score derived from the graph Laplacian's eigenstructure.

---

## 🧠 Mathematical Objective

The optimizer evaluates the topological structure by maximizing the **Normalized Spectral Gap**:

$$Score = \frac{\lambda_2 - \lambda_1}{\lambda_1}$$

**Where:**

* **$\lambda_1$ (Fiedler Value / Algebraic Connectivity):** Represents the bottleneck of the graph. A smaller $\lambda_1$ indicates that the graph has distinct, well-separated community structures (clusters).
* **$\lambda_2 - \lambda_1$ (Spectral Gap):** Represents the internal density of those clusters. A larger gap indicates that nodes within the same cluster are strongly and densely connected.

By maximizing this ratio, the optimizer forces the graph to form tight, dense communities that are clearly separated from one another. Trials where $\lambda_1 = 0.0$ (disconnected graph) are pruned immediately.

---

## 📊 Stage A: Spectral Topology ✓ VALIDATED

### SIFT-128-Euclidean Results

| Dataset | Baseline Score | Best Score | Improvement | Best Params |
| --- | --- | --- | --- | --- |
| **1k** | $5.8 \times 10^{-5}$ | 0.202 | **347x** | `eps=0.027`, `k=20` |
| **4k** | $5.6 \times 10^{-5}$ | 0.049 | **86x** | `eps=0.050`, `k=24` |
| **5k** | 0.533 | 0.965 | **+81%** | `eps=0.113`, `k=4`, `sigma=0.012` |

> **Pattern:** Optuna finds tight `eps`/`sigma` combinations that create dense, perfectly connected manifold graphs.

### Scaling Limits & Workflow

* **4k - 5k:** ✅ Stable (20 trials ~10min)
* **5k+:** ⚠️ Rust sparse heap limits (~40k edges)
* **1M:** 🔄 Sharding $250 \times 4\text{k}$

**Production Workflow:**

1. Tune on a 4k/5k subset (10 min).
2. Apply best params to the 1M full build (sharding overnight).
3. Precompute lambda scalars $\rightarrow$ $O(1)$ tau-mode query.

---

## 💿 Dataset SIFT-1M Setup

We use the standard SIFT-128-euclidean dataset for benchmarks.

```text
data/sift-128-euclidean.hdf5
├── train: (1,000,000, 128) float32 → your subsets
├── query: (10,000, 128) 
├── neighbors: (10,000, 100) int32 ground truth
└── distances: (10,000, 100) float32

```

### Download Instructions

```bash
mkdir -p data
wget https://ann-benchmarks.com/datasets/sift-128-euclidean.hdf5 -O data/sift-128-euclidean.hdf5

# Verify the download
h5dump -H data/sift-128-euclidean.hdf5 | head -20

```

---

## 🚀 Installation & Usage

**Requirements:** Linux (Ubuntu), Python >= 3.13, `uv`.

```bash
git clone https://github.com/MoriondoTommaso/Graph-laplacian-parameters-optimizer.git
cd arrowspace-opt
uv sync
uv pip install -e .

```

### Verify FFI Bridge

```bash
uv run python -c "import numpy as np; from arrowspace import ArrowSpaceBuilder; print('FFI OK')"

```

### Run the Optimizer (SIFT Benchmark)

```bash
uv run python benchmarks/test_sift_topology.py

```

*Results, CSV comparisons, and parameters are automatically saved to the `results/` folder.*

---

## ⚙️ How the Pipeline Works

The codebase relies on a clean dependency chain to safely interface Python with Rust:

1. **`_build_direct.py` — The Rust FFI Interface** Every call to the Rust backend happens here. The `ArrowSpaceBuilder().build()` is called in single-thread from Python; parallelization is handled entirely natively by Rust. The builder takes the params, constructs the graph, and returns the eigenvalues via `lambdas_sorted()`.
2. **`_objective.py` — The Scoring Logic** Wraps the builder in an Optuna-compatible objective function. For each trial, Optuna suggests topological parameters. The function extracts the pure eigenvalues, verifies the graph isn't disconnected ($\lambda_1 > 0.0$), and returns the normalized spectral gap.
3. **`benchmarks/test_sift_topology.py` — The Entry Point** Loads the SIFT subset, evaluates the baseline parameters, initializes the SQLite-backed Optuna study, runs the optimization loop, and dumps the final validations into `benchmarks/save_results.py`.

---

## 🦀 Rust FFI Safety Rules

The pyarrowspace Rust backend allocates memory outside Python's GC. Two strict rules apply:

1. **Never pass a Python list** to `ArrowSpaceBuilder` — always use `np.ndarray` with `dtype=np.float64`.
2. **Always call `items.copy()**` before passing the array to the builder to prevent memory ownership conflicts.**

---

## 📚 References

* *ArrowSpace — Journal of Open Source Software*
* *MRR-Top0: Topology-Aware MRR Extension*

**(Next: Stage 2 Retrieval — Evaluating MRR-Top0/NDCG/tail metrics with best spectral params vs cosine baseline).**

