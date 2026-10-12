"""Run all models over 5 seeds, log to MLflow, report mean +/- SD.

Usage:  python -m src.run_experiment
Structure in MLflow: one parent run per model, one nested child run per seed.
The parent run logs <metric>_mean and <metric>_sd (sample SD, ddof=1).
"""
import argparse
import json
import os
import platform
import random
import subprocess
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .config import EXPERIMENT_NAME, SEEDS, TRACKING_URI
from .data import load_data, split
from .fingerprint import data_fingerprint
from .pipeline import make_search

MODELS = ("dummy", "ridge", "random_forest")
METRICS = ("rmse", "mae", "r2")
warnings.filterwarnings("ignore")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_one(model: str, seed: int, X, y) -> dict:
    set_seed(seed)
    X_tr, X_te, y_tr, y_te = split(X, y, seed)
    search = make_search(model, seed).fit(X_tr, y_tr)   # CV only on train
    pred = search.predict(X_te)                         # test touched once
    return {
        "rmse": root_mean_squared_error(y_te, pred),
        "mae": mean_absolute_error(y_te, pred),
        "r2": r2_score(y_te, pred),
        "cv_rmse": -search.best_score_,
        "best_params": search.best_params_,
    }


def main(models=MODELS, seeds=SEEDS, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)  # fresh machine: output folder may not exist
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    X, y = load_data()
    fp = data_fingerprint(X, y)
    py = platform.python_version()
    rows, summary = [], {}

    for model in models:
        with mlflow.start_run(run_name=model):
            mlflow.log_params({"model": model, "seeds": list(seeds),
                               "n_seeds": len(seeds),
                               "git_commit": git_commit(),
                               "data_sha256": fp, "python": py})
            per_seed = []
            for seed in seeds:
                with mlflow.start_run(run_name=f"{model}_seed{seed}", nested=True):
                    r = run_one(model, seed, X, y)
                    mlflow.log_param("seed", seed)
                    mlflow.log_params({k.replace("model__", "hp_"): v
                                       for k, v in r["best_params"].items()})
                    mlflow.log_metrics({m: r[m] for m in (*METRICS, "cv_rmse")})
                per_seed.append(r)
                rows.append({"model": model, "seed": seed,
                             **{m: r[m] for m in (*METRICS, "cv_rmse")}})
            summary[model] = {}
            for m in METRICS:
                v = np.array([r[m] for r in per_seed])
                mean, sd = float(v.mean()), float(v.std(ddof=1))
                mlflow.log_metrics({f"{m}_mean": mean, f"{m}_sd": sd})
                summary[model][m] = {"mean": mean, "sd": sd}

    pd.DataFrame(rows).to_csv(f"{out_dir}/per_seed_metrics.csv", index=False,
                              float_format="%.6f")
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    lines = ["| model | RMSE (mean ± SD) | MAE (mean ± SD) | R² (mean ± SD) |",
             "|---|---|---|---|"]
    for model, s in summary.items():
        lines.append(f"| {model} | " + " | ".join(
            f"{s[m]['mean']:.3f} ± {s[m]['sd']:.3f}" for m in METRICS) + " |")
    table = "\n".join(lines)
    with open(f"{out_dir}/summary.md", "w") as f:
        f.write(table + "\n")
    print(f"Data SHA-256: {fp}\nPython: {py}")
    print(f"Seeds: {list(seeds)}  (SD = sample SD, ddof=1)\n")
    print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    main(a.models, a.seeds, a.out_dir)
