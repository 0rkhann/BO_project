#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ——— CONFIG ———
DESKTOP = os.path.expanduser("~/Desktop")
RESULTS_ROOT = os.path.expanduser("/home/orkhan/Desktop/bo_project/results")
ENERGY_CACHE = os.path.expanduser("/home/orkhan/Desktop/bo_project/data/dft_G.json")
INCLUDE_BASELINES = True


# ——— HELPERS ———
def aggregate(runs, low_p=15, high_p=85):
    """Stack per-seed arrays and compute median + [low_p,high_p] interval."""
    arr = np.stack(runs, axis=0)
    med = np.median(arr, axis=0)
    low = np.percentile(arr, low_p, axis=0)
    high = np.percentile(arr, high_p, axis=0)
    return med, low, high


def plot_curve(curves, ylabel, title, fname):
    plt.figure(figsize=(8, 5))
    for label, (med, low_percentile, high_percentile) in curves.items():
        it = np.arange(len(med))
        plt.plot(it, med, label=label)
        plt.fill_between(it, low_percentile, high_percentile, alpha=0.2)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(linestyle=":", linewidth=0.5)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.savefig(os.path.join(DESKTOP, fname), bbox_inches="tight")
    plt.close()


# ——— LOAD GLOBAL DISTRIBUTION ———
with open(ENERGY_CACHE, "r") as f:
    energy_data = json.load(f)
energies = np.array([item["energy"] for item in energy_data])

# ——— COLLECT ALL METRICS & TRAJECTORIES ———
ae_runs = {}
rmse_runs = {}
regret_runs = {}
r2_runs = {}
pearson_runs = {}
bestbo_runs = {}
random_runs = {}  # For baseline comparison

for method_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*"))):
    label = os.path.basename(method_dir)
    ae_list, rmse_list, reg_list = [], [], []
    r2_list, pr_list, bb_list = [], [], []
    rand_list = []

    for seed_dir in sorted(glob.glob(os.path.join(method_dir, "seed*"))):
        # Try BO results first
        csv = os.path.join(seed_dir, "bo_iteration_history.csv")
        if os.path.isfile(csv):
            df = pd.read_csv(csv)

            # AE
            if "ae" in df.columns:
                ae_list.append(df["ae"].values)
            # RMSE
            if "rmse" in df.columns:
                rmse_list.append(df["rmse"].values)
            # Regret = best_bo - pool best
            if {"best_bo", "best_pool_min"}.issubset(df.columns):
                bb = np.minimum.accumulate(df["best_bo"].values)
                pb = np.minimum.accumulate(df["best_pool_min"].values)
                reg = np.clip(bb - pb, a_min=0.0, a_max=None)
                reg_list.append(reg)
            # Held‑out R²
            if "r2" in df.columns:
                r2_list.append(df["r2"].values)
            # Held‑out Pearson r
            if "pearson_r" in df.columns:
                pr_list.append(df["pearson_r"].values)
            # Best‑found trajectory (absolute)
            if "best_bo" in df.columns:
                bb_list.append(np.minimum.accumulate(df["best_bo"].values))

        # Try random search results
        random_csv = os.path.join(seed_dir, "random_search_history.csv")
        if os.path.isfile(random_csv):
            df_rand = pd.read_csv(random_csv)
            if "best_random" in df_rand.columns:
                rand_list.append(np.minimum.accumulate(df_rand["best_random"].values))

    if ae_list:
        ae_runs[label] = ae_list
    if rmse_list:
        rmse_runs[label] = rmse_list
    if reg_list:
        regret_runs[label] = reg_list
    if r2_list:
        r2_runs[label] = r2_list
    if pr_list:
        pearson_runs[label] = pr_list
    if bb_list:
        bestbo_runs[label] = bb_list
    if rand_list:
        random_runs[label] = rand_list

# ——— AGGREGATE ———
ae_curves = {lbl: aggregate(v) for lbl, v in ae_runs.items()}
rmse_curves = {lbl: aggregate(v) for lbl, v in rmse_runs.items()}
reg_curves = {lbl: aggregate(v) for lbl, v in regret_runs.items()}
r2_curves = {lbl: aggregate(v) for lbl, v in r2_runs.items()}
pearson_curves = {lbl: aggregate(v) for lbl, v in pearson_runs.items()}
bestbo_curves = {lbl: aggregate(v) for lbl, v in bestbo_runs.items()}
random_curves = {lbl: aggregate(v) for lbl, v in random_runs.items()}

# ——— SMOOTH AE ———
win = 10
ae_sm = {}
for lbl, (med_val, low_val, high_val) in ae_curves.items():
    s_m = pd.Series(med_val).rolling(win, center=True, min_periods=1).mean().values
    s_l = pd.Series(low_val).rolling(win, center=True, min_periods=1).mean().values
    s_h = pd.Series(high_val).rolling(win, center=True, min_periods=1).mean().values
    ae_sm[lbl] = (s_m, s_l, s_h)

# ——— PLOT METRICS ———
plot_curve(
    ae_sm,
    "Absolute Error",
    "AE over Iterations\n(10‑point MA; median ± 70% interval)",
    "AE_over_iterations.png",
)
plot_curve(
    rmse_curves,
    "Held‑out RMSE",
    "RMSE over Iterations\n(median ± 70% interval)",
    "RMSE_over_iterations.png",
)
plot_curve(
    reg_curves,
    "Regret = best_bo − pool_best",
    "Regret over Iterations\n(median ± 70% interval)",
    "Regret_over_iterations.png",
)
plot_curve(
    r2_curves,
    "R²",
    "R² (actual vs. pred) over Iterations\n(median ± 70% interval)",
    "R2_over_iterations.png",
)
plot_curve(
    pearson_curves,
    "Pearson r",
    "Pearson r (actual vs. pred) over Iterations\n(median ± 70% interval)",
    "Pearson_r_over_iterations.png",
)

# ——— PLOT DISTRIBUTION + BO TRAJECTORY ———
for lbl, (med, low, high) in bestbo_curves.items():
    iters = np.arange(len(med))
    fig, ax0 = plt.subplots(figsize=(8, 5))
    ax1 = ax0.twinx()

    # histogram of full energy distribution
    ax0.hist(energies, bins=30, density=True, alpha=0.6)
    ax0.set_xlabel("Energy")
    ax0.set_ylabel("Density")
    ax0.set_title(f"{lbl}: Energy Dist. & BO Best‑Found Trajectory")

    # BO best trajectory
    ax1.plot(iters, med, color="C1", label="median best_bo")
    ax1.fill_between(iters, low, high, color="C1", alpha=0.2)
    ax1.set_ylabel("Best BO energy")
    ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(DESKTOP, f"{lbl}_dist_vs_traj.png"), bbox_inches="tight")
    plt.close()

# ——— BASELINE COMPARISON PLOTS ———
if INCLUDE_BASELINES and (bestbo_runs or random_runs):
    # Combine BO and Random Search for comparison
    combined_data = []

    # Add BO data
    for lbl, runs in bestbo_runs.items():
        for seed_idx, run in enumerate(runs):
            for it, val in enumerate(run):
                combined_data.append(
                    {
                        "Method": lbl,
                        "Type": "BO",
                        "Seed": seed_idx,
                        "Iteration": it,
                        "Best_Value": val,
                    }
                )

    # Add Random Search data
    for lbl, runs in random_runs.items():
        for seed_idx, run in enumerate(runs):
            for it, val in enumerate(run):
                combined_data.append(
                    {
                        "Method": f"{lbl}_Random",
                        "Type": "Random",
                        "Seed": seed_idx,
                        "Iteration": it,
                        "Best_Value": val,
                    }
                )

    if combined_data:
        combined_df = pd.DataFrame(combined_data)

        # Plot 1: BO vs Random comparison
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=combined_df,
            x="Iteration",
            y="Best_Value",
            hue="Method",
            style="Type",
            ci=95,
            estimator=np.median,
        )
        plt.xlabel("Iteration")
        plt.ylabel("Best Value Found")
        plt.title("BO Methods vs Random Search Baseline\n(median ± 95% CI)")
        plt.grid(True, alpha=0.3, linestyle=":")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(DESKTOP, "BO_vs_Random_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot 2: Final values distribution (violin plot)
        final_values = (
            combined_df.groupby(["Method", "Type", "Seed"])["Best_Value"]
            .last()
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        sns.violinplot(
            data=final_values, x="Method", y="Best_Value", hue="Type", split=False
        )
        plt.xlabel("Method")
        plt.ylabel("Final Best Value")
        plt.title("Distribution of Final Best Values: BO vs Random Search")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, linestyle=":", axis="y")
        plt.legend(title="Type")
        plt.tight_layout()
        plt.savefig(
            os.path.join(DESKTOP, "Final_values_violin.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot 3: Box plot for statistical comparison
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=final_values, x="Method", y="Best_Value", hue="Type")
        plt.xlabel("Method")
        plt.ylabel("Final Best Value")
        plt.title("Statistical Comparison: Final Best Values")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, linestyle=":", axis="y")
        plt.legend(title="Type")
        plt.tight_layout()
        plt.savefig(
            os.path.join(DESKTOP, "Final_values_boxplot.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

print("✅ Saved:")
print("   • AE_over_iterations.png")
print("   • RMSE_over_iterations.png")
print("   • Regret_over_iterations.png")
print("   • R2_over_iterations.png")
print("   • Pearson_r_over_iterations.png")
print("   • <method>_dist_vs_traj.png (one per method)")
if INCLUDE_BASELINES and combined_data:
    print("   • BO_vs_Random_comparison.png")
    print("   • Final_values_violin.png")
    print("   • Final_values_boxplot.png")
