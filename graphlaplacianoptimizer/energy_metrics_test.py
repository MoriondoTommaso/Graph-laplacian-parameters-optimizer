import numpy as np
import pandas as pd
from graphlaplacianoptimizer._build_direct import build_arrowspace
from benchmarks.cve_loader import load_cve
from pathlib import Path
import os
import json
import pickle
from multiprocessing import get_context, cpu_count

ROOT = Path.cwd()
DATA_DIR = ROOT / "energy_results"


def generate_arrowspace_trials(n_trials=5, seed=42):
    rng = np.random.default_rng(seed)

    trials = []
    for i in range(n_trials):
        eps = rng.uniform(0.5, 12)
        k = rng.integers(10, 30)
        topk = rng.integers(80, 500)
        p = rng.uniform(1.2, 3.0)
        sigma = rng.uniform(0.1, 1.0)

        # occasional outliers
        if rng.random() < 0.1:
            eps *= rng.uniform(2, 5)
        if rng.random() < 0.1:
            k = rng.integers(30, 60)
        if rng.random() < 0.1:
            topk = rng.integers(500, 1000)
        if rng.random() < 0.1:
            p = rng.uniform(3, 6)
        if rng.random() < 0.1:
            sigma = rng.uniform(1, 3)

        trials.append({
            "trial": i,
            "eps": float(eps),
            "k": int(k),
            "topk": int(topk),
            "p": float(p),
            "sigma": float(sigma),
        })

    return trials


def _sample_array(df, size, seed):
    """
    Deterministic bootstrap sample based only on (size, seed).

    Important:
    This means all parameter trials are tested on the SAME sampled data
    for the same (size, seed), which makes the comparison fairer.
    """
    combined_seed = int((seed * 1_000_003 + size * 97) % (2**32 - 1))
    rng = np.random.default_rng(combined_seed)
    idx = rng.choice(len(df), size=size, replace=True)
    return df[idx]


def _run_single_experiment(args):
    df, trial, size, seed = args

    graph_params = {
        "eps": trial["eps"],
        "k": trial["k"],
        "topk": trial["topk"],
        "p": trial["p"],
        "sigma": trial["sigma"],
    }

    arr = _sample_array(df, size=size, seed=seed)

    aspace, _ = build_arrowspace(graph_params, arr)
    lmb = aspace.lambdas()

    metric_record = {
        "trial": trial["trial"],
        "sample_size": size,
        "seed": seed,
        "Total Energy": float(np.sum(lmb)),
        "Mean Energy": float(np.mean(lmb)),
        "Variance Energy": float(np.var(lmb)),
    }

    lambda_record = {
        "trial": trial["trial"],
        "sample_size": size,
        "seed": seed,
        "lambdas": lmb,
    }

    return metric_record, lambda_record


def compute_energy_metrics(df, param_trials, sample_sizes, seeds, n_jobs=4):
    """
    Runs ArrowSpace energy experiments across parameter sets, sample sizes and seeds.

    Returns
    -------
    metrics_df : pd.DataFrame
        Indexed by (trial, sample_size, seed)
    lambda_records : list[dict]
        Each element contains trial, sample_size, seed, and raw lambdas
    """
    jobs = []
    for trial in param_trials:
        for seed in seeds:
            for size in sample_sizes:
                jobs.append((df, trial, size, seed))

    total_jobs = len(jobs)
    print(f"Total ArrowSpace builds to run: {total_jobs}")

    if n_jobs == 1:
        outputs = []
        for i, job in enumerate(jobs, start=1):
            outputs.append(_run_single_experiment(job))
            if i % 5 == 0 or i == total_jobs:
                print(f"Completed {i}/{total_jobs} jobs")
    else:
        n_jobs = min(n_jobs, cpu_count() or 1)
        print(f"Running in parallel with {n_jobs} processes")

        # fork is efficient on Linux because of copy-on-write
        ctx = get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            outputs = []
            for i, out in enumerate(pool.imap_unordered(_run_single_experiment, jobs), start=1):
                outputs.append(out)
                if i % 5 == 0 or i == total_jobs:
                    print(f"Completed {i}/{total_jobs} jobs")

    metric_records = [m for m, _ in outputs]
    lambda_records = [l for _, l in outputs]

    metrics_df = pd.DataFrame(metric_records)
    metrics_df = metrics_df.set_index(["trial", "sample_size", "seed"]).sort_index()

    return metrics_df, lambda_records


def save_energy_experiment(
    results_df,
    lambda_records,
    param_trials,
    sample_sizes,
    seeds,
    out_dir=DATA_DIR,
):
    """
    Saves ArrowSpace energy experiment results in a reusable format.

    Outputs
    -------
    metrics.parquet        -> metrics dataframe
    lambdas.pkl            -> raw eigenvalue spectra with metadata
    params.json            -> parameter trials
    experiment_config.json -> seeds and sample sizes
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.parquet"
    results_df.to_parquet(metrics_path)

    lambdas_path = out_dir / "lambdas.pkl"
    with open(lambdas_path, "wb") as f:
        pickle.dump(lambda_records, f)

    params_path = out_dir / "params.json"
    with open(params_path, "w") as f:
        json.dump(param_trials, f, indent=2)

    config = {
        "sample_sizes": sample_sizes,
        "seeds": seeds,
        "n_trials": len(param_trials),
    }

    config_path = out_dir / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Experiment saved to: {out_dir}")


sample_sizes = [100, 500, 2000, 8000, 30000, 100000]
seeds = [55, 23, 43]


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_cve()

    print(f"Dataset type: {type(df)}")
    print(f"Dataset size: {len(df)}")

    print("Generating parameter trials...")
    param_trials = generate_arrowspace_trials(n_trials=5, seed=42)

    print(f"Trials: {len(param_trials)}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Seeds: {seeds}")

    print("Running ArrowSpace energy experiment...")
    results_df, lambda_records = compute_energy_metrics(
        df,
        param_trials=param_trials,
        sample_sizes=sample_sizes,
        seeds=seeds,
        n_jobs=4,   # start with 4 on 16 GB RAM
    )

    print("Saving results...")
    save_energy_experiment(
        results_df,
        lambda_records,
        param_trials,
        sample_sizes,
        seeds,
    )

    print("Done.")