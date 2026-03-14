import numpy as np
import pandas as pd
from graphlaplacianoptimizer._build_direct import build_arrowspace
from benchmarks.cve_loader import load_cve
from pathlib import Path
import json
import pickle
from multiprocessing import get_context, cpu_count


ROOT = Path.cwd()

# old run folder: the one that generated 100..100000
BASE_RESULTS_DIR = ROOT / "energy_results"

# new extension folder: only 200k and 300k
EXT_RESULTS_DIR = ROOT / "energy_results_ext_200k_300k"

# optional merged folder
MERGED_RESULTS_DIR = ROOT / "energy_results_merged"


def generate_arrowspace_trials(n_trials=5, seed=42):
    rng = np.random.default_rng(seed)

    trials = []
    for i in range(n_trials):
        eps = rng.uniform(0.5, 12)
        k = rng.integers(10, 30)
        topk = rng.integers(80, 500)
        p = rng.uniform(1.2, 3.0)
        sigma = rng.uniform(0.1, 1.0)

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


def load_existing_params_and_config(results_dir: Path):
    params_path = results_dir / "params.json"
    config_path = results_dir / "experiment_config.json"

    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.json in {results_dir}")

    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment_config.json in {results_dir}")

    with open(params_path, "r") as f:
        param_trials = json.load(f)

    with open(config_path, "r") as f:
        config = json.load(f)

    return param_trials, config


def _sample_array(df, size, seed):
    """
    Deterministic bootstrap sample based only on (size, seed).
    Must remain identical to the original experiment for mergeability.
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


def compute_energy_metrics(df, param_trials, sample_sizes, seeds, n_jobs=1):
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
            print(f"Completed {i}/{total_jobs} jobs")
    else:
        n_jobs = min(n_jobs, cpu_count() or 1)
        print(f"Running in parallel with {n_jobs} processes")

        ctx = get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            outputs = []
            for i, out in enumerate(pool.imap_unordered(_run_single_experiment, jobs), start=1):
                outputs.append(out)
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
    out_dir: Path,
):
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
        "sample_sizes": list(sample_sizes),
        "seeds": list(seeds),
        "n_trials": len(param_trials),
    }

    config_path = out_dir / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Experiment saved to: {out_dir}")


def merge_energy_result_dirs(base_dir: Path, ext_dir: Path, merged_dir: Path):
    # ---------- load ----------
    base_metrics = pd.read_parquet(base_dir / "metrics.parquet")
    ext_metrics = pd.read_parquet(ext_dir / "metrics.parquet")

    with open(base_dir / "lambdas.pkl", "rb") as f:
        base_lambdas = pickle.load(f)

    with open(ext_dir / "lambdas.pkl", "rb") as f:
        ext_lambdas = pickle.load(f)

    with open(base_dir / "params.json", "r") as f:
        base_params = json.load(f)

    with open(ext_dir / "params.json", "r") as f:
        ext_params = json.load(f)

    with open(base_dir / "experiment_config.json", "r") as f:
        base_cfg = json.load(f)

    with open(ext_dir / "experiment_config.json", "r") as f:
        ext_cfg = json.load(f)

    # ---------- validate ----------
    if base_params != ext_params:
        raise ValueError("Cannot merge: params.json differs between runs")

    if sorted(base_cfg["seeds"]) != sorted(ext_cfg["seeds"]):
        raise ValueError("Cannot merge: seeds differ between runs")

    # ---------- merge metrics ----------
    merged_metrics = pd.concat([base_metrics, ext_metrics], axis=0)
    merged_metrics = (
        merged_metrics
        .reset_index()
        .drop_duplicates(subset=["trial", "sample_size", "seed"], keep="last")
        .set_index(["trial", "sample_size", "seed"])
        .sort_index()
    )

    # ---------- merge lambdas ----------
    combined_lambdas = base_lambdas + ext_lambdas
    dedup = {}
    for rec in combined_lambdas:
        key = (rec["trial"], rec["sample_size"], rec["seed"])
        dedup[key] = rec
    merged_lambdas = list(dedup.values())

    # ---------- merge config ----------
    merged_sample_sizes = sorted(set(base_cfg["sample_sizes"]) | set(ext_cfg["sample_sizes"]))
    merged_cfg = {
        "sample_sizes": merged_sample_sizes,
        "seeds": base_cfg["seeds"],
        "n_trials": base_cfg["n_trials"],
    }

    # ---------- save ----------
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_metrics.to_parquet(merged_dir / "metrics.parquet")

    with open(merged_dir / "lambdas.pkl", "wb") as f:
        pickle.dump(merged_lambdas, f)

    with open(merged_dir / "params.json", "w") as f:
        json.dump(base_params, f, indent=2)

    with open(merged_dir / "experiment_config.json", "w") as f:
        json.dump(merged_cfg, f, indent=2)

    print(f"Merged results saved to: {merged_dir}")


if __name__ == "__main__":
    # run ONLY the extension
    sample_sizes = [150000]

    print("Loading dataset...")
    df = load_cve()
    print(f"Dataset type: {type(df)}")
    print(f"Dataset size: {len(df)}")

    print("Loading params + config from base run...")
    param_trials, base_config = load_existing_params_and_config(BASE_RESULTS_DIR)

    # reuse exactly the same seeds as the original run
    seeds = base_config["seeds"]

    print(f"Loaded {len(param_trials)} parameter trials from base run")
    print(f"Seeds reused from base run: {seeds}")
    print(f"Extension sample sizes: {sample_sizes}")

    print("Running ArrowSpace extension experiment...")
    results_df, lambda_records = compute_energy_metrics(
        df,
        param_trials=param_trials,
        sample_sizes=sample_sizes,
        seeds=seeds,
        n_jobs=2,   # use 1 for safety on 200k / 300k with 16 GB RAM
    )

    print("Saving extension results...")
    save_energy_experiment(
        results_df,
        lambda_records,
        param_trials,
        sample_sizes,
        seeds,
        out_dir=EXT_RESULTS_DIR,
    )

    print("Merging base + extension results...")
    merge_energy_result_dirs(
        base_dir=BASE_RESULTS_DIR,
        ext_dir=EXT_RESULTS_DIR,
        merged_dir=MERGED_RESULTS_DIR,
    )

    print("Done.")