#!/usr/bin/env python3
"""
Comprehensive Comparison Plots for BO Methods + Random Search
Creates publication-quality comparison plots across all methods and datasets.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# ——— CONFIG ———
RESULTS_ROOT = "/home/orkhan/Desktop/bo_project/results"
OUTPUT_DIR = "/home/orkhan/Desktop/bo_project/plots/comprehensive_comparison"
CACHE_FILES = {
    "dft": "/home/orkhan/Desktop/bo_project/data/dft_G.json",
    "xtb": "/home/orkhan/Desktop/bo_project/data/xtb_G.json",
}

# Color scheme for methods
METHOD_COLORS = {
    "fabo": "#e74c3c",  # Red
    "opls": "#f39c12",  # Orange
    "pca": "#9b59b6",  # Purple
    "pls": "#3498db",  # Blue
    "vanilla": "#2ecc71",  # Green
    "random": "#95a5a6",  # Gray
}

# Matplotlib settings for publication quality
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["figure.dpi"] = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_global_best(cache_file):
    """Load the global best energy from cache."""
    with open(cache_file, "r") as f:
        energy_data = json.load(f)
    energies = [item["energy"] for item in energy_data]
    return min(energies)


def parse_method_label(label):
    """Extract dataset and method from directory label."""
    parts = label.split("_")

    # Handle random search
    if "random_search" in label:
        dataset = "_".join(parts[:-2])  # Everything before 'random_search'
        method = "random"
        return dataset, method

    # Handle BO methods (format: dataset_method_kernel_acquisition)
    if len(parts) >= 4:
        # Last 3 are method_kernel_acquisition
        method = parts[-3]
        dataset = "_".join(parts[:-3])
        return dataset, method

    return None, None


def aggregate_runs(runs, low_p=25, high_p=75):
    """Stack per-seed arrays and compute median + percentile interval."""
    if not runs:
        return None, None, None

    arr = np.stack(runs, axis=0)
    med = np.median(arr, axis=0)
    low = np.percentile(arr, low_p, axis=0)
    high = np.percentile(arr, high_p, axis=0)
    return med, low, high


def collect_data_by_dataset():
    """Collect BO and random search data organized by dataset."""
    data_by_dataset = {}
    random_trajectories = []  # Store canonical random search baseline

    # Structure: results/{dataset}/{method}_{kernel}_{acquisition}/seed{N}/...
    # or results/{dataset}_random_search/seed{N}/random_search_history.csv

    # First pass: collect random search data (use first found as canonical)
    for dataset_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*_random_search"))):
        if not random_trajectories:  # Only load first random search found
            for seed_dir in sorted(glob.glob(os.path.join(dataset_dir, "seed*"))):
                random_csv = os.path.join(seed_dir, "random_search_history.csv")
                if os.path.isfile(random_csv):
                    df = pd.read_csv(random_csv)
                    if "best_random" in df.columns:
                        best_trajectory = np.minimum.accumulate(
                            df["best_random"].values
                        )
                        random_trajectories.append(best_trajectory)

    # Second pass: collect BO data for all datasets
    for dataset_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*"))):
        dataset_label = os.path.basename(dataset_dir)

        # Skip random search directories in this pass
        if "random_search" in dataset_label:
            continue
        else:
            # Handle BO methods
            dataset = dataset_label

            for method_dir in sorted(glob.glob(os.path.join(dataset_dir, "*"))):
                method_label = os.path.basename(method_dir)
                parts = method_label.split("_")

                if len(parts) >= 3:
                    method = parts[0]  # fabo, opls, pca, pls, vanilla

                    if dataset not in data_by_dataset:
                        data_by_dataset[dataset] = {}
                    if method not in data_by_dataset[dataset]:
                        data_by_dataset[dataset][method] = []

                    for seed_dir in sorted(
                        glob.glob(os.path.join(method_dir, "seed*"))
                    ):
                        # Check for nested subdirectory structure
                        csv_paths = [
                            os.path.join(seed_dir, "bo_iteration_history.csv"),
                            *glob.glob(
                                os.path.join(
                                    seed_dir, "*", "seed*", "bo_iteration_history.csv"
                                )
                            ),
                        ]

                        for csv_path in csv_paths:
                            if os.path.isfile(csv_path):
                                df = pd.read_csv(csv_path)
                                if "best_bo" in df.columns:
                                    best_trajectory = np.minimum.accumulate(
                                        df["best_bo"].values
                                    )
                                    data_by_dataset[dataset][method].append(
                                        best_trajectory
                                    )
                                    break

    # Apply canonical random search baseline to all datasets
    if random_trajectories:
        for dataset in data_by_dataset.keys():
            data_by_dataset[dataset]["random"] = random_trajectories

    return data_by_dataset


def plot_dataset_comparison(dataset, methods_data, global_best, output_path):
    """
    Create comparison plot for one dataset with all methods.

    Args:
        dataset: Dataset name (e.g., 'dft_mordred')
        methods_data: Dict mapping method name -> list of trajectories
        global_best: Global best energy in the pool
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort methods for consistent ordering (vanilla, then alphabetical, then random)
    method_order = []
    for method in ["fabo", "opls", "pca", "pls", "vanilla", "random"]:
        if method in methods_data and methods_data[method]:
            method_order.append(method)

    legend_elements = []

    for method in method_order:
        trajectories = methods_data[method]
        if not trajectories:
            continue

        med, low, high = aggregate_runs(trajectories)
        if med is None:
            continue

        iterations = np.arange(len(med))
        color = METHOD_COLORS.get(method, "#34495e")

        # Format label
        if method == "random":
            label = "Random Search"
            linestyle = "--"
            linewidth = 2
        else:
            label = f"{dataset.split('_')[1]}_{method}"
            linestyle = "-"
            linewidth = 2.5

        # Plot median line
        line = ax.plot(
            iterations,
            med,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.9,
        )[0]

        # Plot confidence interval
        ax.fill_between(iterations, low, high, color=color, alpha=0.25)

        legend_elements.append(line)

    # Add global best line
    ax.axhline(
        y=global_best,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Global Best Pool Energy: {global_best:.4f}",
        alpha=0.8,
    )

    # Labels and title
    descriptor_type = dataset.split("_")[-1].capitalize()
    ax.set_xlabel("BO Iterations", fontsize=14, fontweight="bold")
    ax.set_ylabel("Best BO Energy", fontsize=14, fontweight="bold")

    title_map = {
        "mordred": "Mordred",
        "chemberta2": "Chemberta2",
        "descriptors": "DFT Descriptors",
    }
    desc_name = title_map.get(dataset.split("_")[-1], descriptor_type)
    ax.set_title(
        f"BO Performance Comparison - {desc_name} Methods + Random Search",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Styling
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax.set_xlim(0, len(med) - 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def calculate_convergence_metrics(data_by_dataset, global_bests):
    """
    Calculate convergence rate metrics for all methods.

    Returns DataFrame with convergence analysis.
    """
    convergence_data = []

    for dataset, methods_data in data_by_dataset.items():
        global_best = global_bests.get(dataset)
        if global_best is None:
            continue

        for method, trajectories in methods_data.items():
            if not trajectories:
                continue

            for seed_idx, trajectory in enumerate(trajectories):
                # Calculate metrics
                final_value = trajectory[-1]
                gap_to_global = abs(final_value - global_best)

                # Find iteration where within X% of global best
                thresholds = [0.01, 0.05, 0.10]  # 1%, 5%, 10%
                convergence_iters = {}

                for threshold in thresholds:
                    target = global_best + abs(global_best) * threshold
                    iters_to_target = np.where(trajectory <= target)[0]
                    convergence_iters[f"iter_to_{int(threshold*100)}pct"] = (
                        iters_to_target[0]
                        if len(iters_to_target) > 0
                        else len(trajectory)
                    )

                # Calculate improvement rate (first 20 iterations)
                if len(trajectory) >= 20:
                    initial_20 = trajectory[:20]
                    improvement_rate = (trajectory[0] - trajectory[19]) / 20
                else:
                    improvement_rate = 0

                convergence_data.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed_idx,
                        "final_value": final_value,
                        "gap_to_global": gap_to_global,
                        "improvement_rate": improvement_rate,
                        **convergence_iters,
                    }
                )

    return pd.DataFrame(convergence_data)


def plot_convergence_analysis(convergence_df, output_dir):
    """Print convergence summary statistics."""

    # Print summary statistics
    print("\n" + "=" * 80)
    print("CONVERGENCE RATE SUMMARY")
    print("=" * 80)

    plot_data = convergence_df.copy()
    summary = (
        plot_data.groupby(["dataset", "method"])
        .agg(
            {
                "final_value": ["mean", "std"],
                "gap_to_global": ["mean", "std", "min"],
                "iter_to_5pct": ["mean", "std", "min"],
                "improvement_rate": ["mean", "std"],
            }
        )
        .round(4)
    )

    print(summary)


def plot_best_across_datasets(data_by_dataset, global_bests, output_path):
    """
    Plot comparison of best performing method from each dataset + random search.

    Args:
        data_by_dataset: All collected data organized by dataset
        global_bests: Global best values for each dataset
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Find best BO method for each dataset (excluding random)
    best_methods = {}
    random_methods = {}

    for dataset, methods_data in data_by_dataset.items():
        best_final_value = float("inf")
        best_method = None
        best_trajectories = None

        for method, trajectories in methods_data.items():
            if not trajectories:
                continue

            # Store random search separately
            if method == "random":
                random_methods[dataset] = trajectories
                continue

            # Calculate median final value for BO methods
            final_values = [traj[-1] for traj in trajectories]
            median_final = np.median(final_values)

            if median_final < best_final_value:
                best_final_value = median_final
                best_method = method
                best_trajectories = trajectories

        if best_method:
            best_methods[dataset] = {
                "method": best_method,
                "trajectories": best_trajectories,
                "final_value": best_final_value,
            }

    # Plot the best BO methods
    for dataset, info in sorted(best_methods.items()):
        method = info["method"]
        trajectories = info["trajectories"]

        med, low, high = aggregate_runs(trajectories)
        if med is None:
            continue

        iterations = np.arange(len(med))
        color = METHOD_COLORS.get(method, "#34495e")

        # Format label
        title_map = {
            "mordred": "Mordred",
            "chemberta2": "Chemberta2",
            "descriptors": "DFT Descriptors",
        }
        descriptor_type = title_map.get(
            dataset.split("_")[-1], dataset.split("_")[-1].capitalize()
        )
        label = (
            f"{descriptor_type} - {method.upper()} (best: {info['final_value']:.2f})"
        )

        # Plot median line
        ax.plot(
            iterations,
            med,
            label=label,
            color=color,
            linestyle="-",
            linewidth=2.5,
            alpha=0.9,
        )

        # Plot confidence interval
        ax.fill_between(iterations, low, high, color=color, alpha=0.25)

    # Plot random search once (same for all datasets)
    if random_methods:
        # Get random search from any dataset (they're all the same)
        trajectories = list(random_methods.values())[0]
        med, low, high = aggregate_runs(trajectories)

        if med is not None:
            iterations = np.arange(len(med))
            color = METHOD_COLORS.get("random", "#95a5a6")

            final_value = np.median([traj[-1] for traj in trajectories])
            label = f"RANDOM (best: {final_value:.2f})"

            # Plot median line (dashed for random)
            ax.plot(
                iterations,
                med,
                label=label,
                color=color,
                linestyle="--",
                linewidth=2.0,
                alpha=0.7,
            )

            # Plot confidence interval
            ax.fill_between(iterations, low, high, color=color, alpha=0.15)

    # Add global minimum line (all DFT datasets use same cache, so same global min)
    if global_bests:
        # Get global best from any DFT dataset (they all use dft_G.json)
        global_min = list(global_bests.values())[0]
        ax.axhline(
            y=global_min,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Global Pool Minimum: {global_min:.4f}",
            alpha=0.8,
        )

    # Labels and title
    ax.set_xlabel("BO Iterations", fontsize=14, fontweight="bold")
    ax.set_ylabel("Best BO Energy", fontsize=14, fontweight="bold")
    ax.set_title(
        "Best Performing Methods Across All Datasets\n(Median ± IQR across 5 seeds)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Styling
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {output_path}")


def main():
    """Generate all comprehensive comparison plots."""

    print("=" * 80)
    print("GENERATING COMPREHENSIVE COMPARISON PLOTS")
    print("=" * 80)

    # Collect data
    print("\n📊 Collecting data from all experiments...")
    data_by_dataset = collect_data_by_dataset()

    # Load global bests
    global_bests = {}
    for dataset in data_by_dataset.keys():
        # Determine which cache file to use
        if "dft" in dataset:
            cache_file = CACHE_FILES["dft"]
        elif "xtb" in dataset:
            cache_file = CACHE_FILES["xtb"]
        else:
            continue

        global_bests[dataset] = load_global_best(cache_file)

    # Generate comparison plots for each dataset
    print("\n📈 Generating dataset comparison plots...")
    for dataset, methods_data in sorted(data_by_dataset.items()):
        global_best = global_bests.get(dataset)
        if global_best is None:
            continue

        output_path = os.path.join(OUTPUT_DIR, f"comparison_{dataset}.png")
        plot_dataset_comparison(dataset, methods_data, global_best, output_path)

    # Generate best methods comparison across datasets
    print("\n🏆 Generating best methods comparison across datasets...")
    best_output_path = os.path.join(OUTPUT_DIR, "best_methods_comparison.png")
    plot_best_across_datasets(data_by_dataset, global_bests, best_output_path)

    # Print convergence summary
    print("\n⚡ Calculating convergence statistics...")
    convergence_df = calculate_convergence_metrics(data_by_dataset, global_bests)
    plot_convergence_analysis(convergence_df, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print(f"✅ ALL PLOTS SAVED TO: {OUTPUT_DIR}")
    print("=" * 80)

    # List all generated plots
    print("\nGenerated plots:")
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        if filename.endswith(".png"):
            print(f"  • {filename}")


if __name__ == "__main__":
    main()
