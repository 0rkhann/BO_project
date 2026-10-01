#!/usr/bin/env python3
"""Summarize BO / random-search results into Markdown tables.

Reads only files that are already in the repository:
  results/**/bo_iteration_history.csv
  results/*_random_search/seed*/random_search_history.csv
  ml_results/*/*_predictions.csv  (hold-out RMSE / R², True_Energy vs prediction)

Usage: python scripts/summarize_results.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from recompute_holdout_metrics import collect as holdout_metrics_table

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATASETS = ["dft_descriptors", "dft_chemberta2", "dft_mordred"]
METHODS = ["fabo", "opls", "pca", "pls"]


def summarize_run(path, col):
    """Best (minimum) value in a run and the first iteration at which it was reached."""
    df = pd.read_csv(path)
    best = df[col].cummin()
    final = best.iloc[-1]
    it = int(df.loc[best <= final, "iter"].iloc[0])
    return final, it, float(df["best_pool_min"].iloc[-1]), len(df)


def aggregate(dataset, method, files, col):
    runs = [summarize_run(f, col) for f in files]
    vals = np.array([r[0] for r in runs])
    its = np.array([r[1] for r in runs])
    pool = runs[0][2]
    return {
        "dataset": dataset,
        "method": method,
        "n_seeds": len(runs),
        "best_mean": vals.mean(),
        "best_std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
        "iter_mean": its.mean(),
        "iter_std": its.std(ddof=1) if len(its) > 1 else np.nan,
        "pool_min": pool,
        "n_iter": runs[0][3],
        "n_hit_pool_min": int(np.sum(np.isclose(vals, pool))),
    }


def collect():
    rows = []
    for ds in DATASETS:
        for m in METHODS:
            files = sorted(
                (RESULTS / ds / f"{m}_Matern_EI").glob("seed*/**/bo_iteration_history.csv")
            )
            if files:
                rows.append(aggregate(ds, m, files, "best_bo"))
        files = sorted(
            (RESULTS / f"{ds}_random_search").glob("seed*/random_search_history.csv")
        )
        if files:
            rows.append(aggregate(ds, "random", files, "best_random"))
    return pd.DataFrame(rows)


def main():
    t = collect()
    print(
        "| Representation | Method | Seeds | Best energy (mean ± std) "
        "| Iteration reached (mean ± std) | Seeds hitting pool optimum |"
    )
    print("|---|---|---|---|---|---|")
    for _, r in t.iterrows():
        print(
            f"| {r.dataset} | {r.method} | {r.n_seeds} "
            f"| {r.best_mean:.2f} ± {r.best_std:.2f} "
            f"| {r.iter_mean:.1f} ± {r.iter_std:.1f} "
            f"| {r.n_hit_pool_min}/{r.n_seeds} |"
        )
    print()
    for ds, g in t.groupby("dataset", sort=False):
        print(f"{ds}: pool optimum {g.pool_min.iloc[0]:.4f}, iterations per run {sorted(set(g.n_iter))}")
    print()
    ml = holdout_metrics_table()
    print("| Representation | Model | Hold-out RMSE | Hold-out R² |")
    print("|---|---|---|---|")
    for _, r in ml.iterrows():
        print(f"| {r.Representation} | {r.Model} | {r.Test_RMSE:.2f} | {r.Test_R2:.3f} |")
    print(f"\nHold-out set size per model: {sorted(set(ml.n_test))}")


if __name__ == "__main__":
    main()
