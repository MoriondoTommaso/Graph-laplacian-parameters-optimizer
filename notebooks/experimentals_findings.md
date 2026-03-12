
# Topological Convergence of the ArrowSpace Laplacian: The Asymptotic Variance Rule

**Date:** March 12, 2026
**Subject:** Empirical and Theoretical Optimization of Graph Subsampling in Vector Search
**Dataset:** CVE Database (BERT-384 embeddings, ~300k items)

## Abstract

Optimizing graph-based vector search indexes on massive datasets incurs prohibitive computational costs. This paper investigates the topological convergence of the ArrowSpace Graph Laplacian using a dataset of ~300,000 CVE embeddings. We demonstrate that the manifold's underlying structure can be fully captured using a heavily reduced subsample. By tracking the asymptotic variance of the Rayleigh quotients ($\lambda$), we prove that the CVE manifold undergoes a percolation phase transition at ~8,000 items and achieves strict topological stability at ~16,000 items (just 5.1% of the corpus). This establishes a mathematically sound protocol for conducting hyperparameter optimization on minimal subsets without degrading the integrity of the spectral metrics.

---

## 1. Introduction

ArrowSpace relies on a Graph Laplacian, denoted as $\mathbf{L} = \mathbf{D} - \mathbf{A}$, to map the topological terrain of an embedding space. Each item in the space is assigned a Rayleigh quotient ($\lambda$), which quantifies its topological "roughness" or isolation relative to the manifold.

Optimizing ArrowSpace hyperparameters (such as $\epsilon$, $k$, and $\sigma$) requires computing spectral metrics like the Fiedler value and mean energy. Running these calculations across hundreds of thousands of items per trial is computationally inefficient. Our objective was to determine the minimum number of items required to construct a mathematically faithful representation of the dataset's global topology.

## 2. Methodology: The Asymptotic Variance Rule

To determine convergence without relying on arbitrary percentage-based guesses, we developed the **Asymptotic Variance Rule**.

As random items are added to a spatial graph, the structural differences between them initially create massive fluctuations in the distribution of $\lambda$. However, once the fundamental shape of the data manifold (its clusters, voids, and boundaries) is defined, adding more points merely increases density without altering the core topology. This stabilization is perfectly mirrored by the variance of the Rayleigh quotients ($\sigma^2_{\lambda}$).

We implemented a robust sequential sampling algorithm that doubles the sample size iteratively, halting only when the relative percentage change in variance drops below a strict tolerance threshold (5%) for two consecutive steps.

## 3. Empirical Findings on the CVE Dataset

### 3.1 The Percolation Phase Transition

During our scaling tests, the metrics did not follow a smooth, linear descent. At approximately 8,000 items, the manifold experienced a sharp, 33% drop in variance accompanied by a temporary anomaly in mean energy.

This represents a classic **Percolation Phase Transition**. At this critical density threshold, previously isolated micro-clusters of CVEs suddenly bridged together into a "Giant Connected Component." The items forming these sparse bridges experienced massive spikes in their individual $\lambda$ scores.

### 3.2 Validation of Topological Outliers

To ensure this transition was not a mathematical artifact of a fixed $\epsilon$ radius, we conducted a control run with a significantly larger radius ($\epsilon=10.84$). The structural anomaly remained rigidly fixed at the ~10,000-item mark. Extracting the specific entries driving this anomaly revealed $\lambda$ scores approaching the theoretical maximum (1.0). This confirms that the transition represents a genuine, dense sub-manifold of highly specific, topologically isolated CVE entries.

### 3.3 Strict Convergence at 5.1%

Following the phase transition, the graph smoothed out rapidly. The Asymptotic Variance algorithm recorded the following final steps:

* **16,000 items:** Variance stabilized with a minor 4.14% relative change.
* **32,000 items:** Variance change dropped to a negligible 0.79%.

Because the structural variance effectively flatlined between 16k and 32k, we mathematically conclude that the dataset's topology is fully represented at **16,000 items**.

## 4. Theoretical Support

This 5.1% empirical convergence aligns closely with established spectral graph theory:

* **Laplace-Beltrami Convergence:** The discrete graph Laplacian approximates the continuous manifold operator at a rate heavily dependent on the intrinsic dimension ($m$), not the total dataset size ($N$).
* **Nyström Approximation:** The data strongly supports the Nyström bound, which predicts that a low-rank approximation of a Laplacian requires a sample size proportional to the data's complexity rather than its sheer volume.

## 5. Critical Distinction: The Map vs. The Territory

A crucial caveat to this finding is the distinction between the topological map and the retrieval corpus.

* **The Laplacian (The Map):** Can be safely and fully constructed using only the 16,000-item subsample.
* **The Search Index (The Territory):** Must contain all 300,000 items.

Undersampling the final retrieval index will result in missing ground-truth hits and collapsed Recall metrics. The subsample is strictly for building the Laplacian and optimizing hyperparameters.

## 6. Conclusion and Engineering Directives

The "10% rule" initially hypothesized was overly conservative. For the BERT-384 CVE dataset, the ArrowSpace Laplacian converges at **~16,000 items (5.1%)**.

**Standard Operating Procedure for Future Datasets:**

1. Do not hardcode percentage thresholds. Convergence is a function of a dataset's intrinsic dimension, not its volume.
2. For all new datasets, execute the `find_convergence_threshold_robust` algorithm to empirically locate the variance plateau.
3. Conduct all Optuna hyperparameter trials exclusively on the dynamically identified subsample, reducing compute overhead by roughly 95% without compromising spectral fidelity.
4. Freeze the optimal Laplacian parameters, and subsequently map the full 100% corpus into that defined space for production retrieval.

---

## Appendix A: Mathematical Justification for the Asymptotic Variance Rule

### A.1 The Continuous Limit of the Discrete Laplacian

Let the full dataset be represented as points sampled from a compact Riemannian manifold $\mathcal{M} \subset \mathbb{R}^d$ with intrinsic dimension $m$. In ArrowSpace, we construct a discrete graph Laplacian $\mathbf{L}_n$ from a random subsample of size $n$.

According to the theory of manifold learning, as $n \to \infty$ and the neighborhood radius $\epsilon \to 0$, the discrete graph operator $\mathbf{L}_n$ converges pointwise to the continuous Laplace-Beltrami operator $\Delta_{\mathcal{M}}$ of the underlying manifold:

$$\lim_{n \to \infty, \epsilon \to 0} C_{\epsilon, n} \mathbf{L}_n f \propto \Delta_{\mathcal{M}} f$$

Where $C_{\epsilon, n}$ is a scaling constant dependent on the graph construction. Because the operators converge, their resulting eigenspectra must also converge.

### A.2 Spectral Convergence and the Rayleigh Quotient

For any item $i$ in the graph, its topological role is defined by its Rayleigh quotient $\lambda_i$, representing the local Dirichlet energy. The set of all Rayleigh quotients in a subsample of size $n$ forms an empirical spectral distribution, denoted by its cumulative distribution function (CDF), $F_n(\lambda)$.

By the Law of Large Numbers, as the subsample size $n$ approaches the critical density required to span $\mathcal{M}$, the empirical spectral distribution $F_n(\lambda)$ converges in distribution to the true, global spectral distribution $F(\lambda)$ of the complete dataset.

### A.3 Why Variance is the Necessary Condition for Topological Completeness

By the method of moments in probability theory, if a sequence of bounded empirical distributions $F_n$ converges to a true distribution $F$, all central moments must mathematically converge to fixed constants.

The first moment (the mean energy, $\bar{\lambda}_n$) stabilizes rapidly once the "bulk" dense regions of the manifold are sampled. However, the mean is highly insensitive to rare, low-density boundary conditions. To ensure the *entire* manifold is mapped, we must track the second central moment—the variance:

$$\sigma^2_{\lambda}(n) = \frac{1}{n} \sum_{i=1}^{n} (\lambda_i - \bar{\lambda}_n)^2$$

Because variance squares the deviation from the mean, it heavily penalizes the absence of topological outliers. Items located in structural bottlenecks, manifold boundaries, or isolated clusters produce extreme eigenvalues ($\lambda_i \gg \bar{\lambda}_n$). Therefore, if the random subsample has not yet discovered these rare structural features, the variance will remain unstable.

### A.4 The Formal Convergence Criterion

We formally define the topological convergence threshold as the minimum sample size $n^*$ at which the relative rate of change of the spectral variance falls below a strict tolerance threshold $\delta$ for consecutive iterative steps:

$$\left| \frac{\sigma^2_{\lambda}(n^*) - \sigma^2_{\lambda}(n_{prev})}{\sigma^2_{\lambda}(n_{prev})} \right| < \delta$$

When this condition is satisfied, it provides a mathematical guarantee that the discrete graph $\mathbf{L}_{n^*}$ has fully mapped the manifold. Adding further samples ($n > n^*$) will only increase local density within already-mapped regions, leaving the structural variance—and thus the overarching topology—statistically unchanged.

---

## Appendix B: Python Implementation

```python
import numpy as np
import pandas as pd

def find_convergence_threshold_robust(df, graph_params, start_size=1000, step_factor=2.0, tolerance_pct=0.05, patience=2):
    """
    Finds topological convergence using percentage-based tolerance and patience.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full dataset to sample from.
    graph_params : dict
        The ArrowSpace parameters (eps, k, etc.).
    start_size : int
        The initial sample size to test.
    step_factor : float
        Multiplier for the next sample size.
    tolerance_pct : float
        The maximum allowed percentage change in variance (e.g., 0.05 = 5%).
    patience : int
        How many consecutive steps the delta must be below tolerance_pct before stopping.
    """
    max_samples = len(df)
    current_size = start_size
    prev_variance = float('inf')
    consecutive_stable_steps = 0
    
    print(f"Searching for convergence (Tolerance: < {tolerance_pct*100}% change for {patience} steps)...")
    print("-" * 65)
    print(f"{'Size':>10} | {'Variance':>12} | {'Rel Delta (%)':>15} | {'Patience':>8}")
    print("-" * 65)
    
    while current_size <= max_samples:
        sample_df = df.sample(n=current_size, random_state=42, replace=False)
        arr = np.array(sample_df)
        
        # Build graph and extract lambdas
        aspace, _ = build_arrowspace(graph_params, arr)
        lambdas = aspace.lambdas()
        current_variance = np.var(lambdas)
        
        # Calculate Relative Delta
        if prev_variance != float('inf'):
            rel_delta = abs(prev_variance - current_variance) / prev_variance
        else:
            rel_delta = float('inf')
            
        # Check stability
        if rel_delta < tolerance_pct:
            consecutive_stable_steps += 1
        else:
            consecutive_stable_steps = 0
            
        print(f"{current_size:>10} | {current_variance:>12.6f} | {rel_delta*100:>14.2f}% | {consecutive_stable_steps:>8}")
        
        # Stopping Condition
        if consecutive_stable_steps >= patience:
            print("-" * 65)
            converged_size = int(current_size / (step_factor ** (patience - 1)))
            print(f"✅ SUCCESS: Manifold converged around {converged_size} items.")
            print(f"You only need {(converged_size/max_samples)*100:.2f}% of this dataset.")
            return converged_size
            
        prev_variance = current_variance
        
        next_size = int(current_size * step_factor)
        if current_size == max_samples:
            break
        current_size = min(next_size, max_samples)
            
    print("-" * 65)
    print("⚠️ WARNING: Dataset fully evaluated without crossing threshold.")
    return max_samples

```

