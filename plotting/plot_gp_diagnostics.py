#!/usr/bin/env python3
"""
GP Hyperparameter Diagnostics Plotting

Visualizes the evolution of GP hyperparameters, convergence, and stability
over Bayesian Optimization iterations.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def aggregate_gp_data(dfs_list):
    """Aggregate GP data across multiple seeds."""
    if not dfs_list:
        return None

    # Get common columns
    common_cols = set(dfs_list[0].columns)
    for df in dfs_list[1:]:
        common_cols &= set(df.columns)

    # Create aggregated data structure
    aggregated = {}

    for col in common_cols:
        if col == "iter":
            aggregated[col] = dfs_list[0][col].values
        elif col in ["l", "sigma2", "noise", "lml", "r2", "rmse", "ae"]:
            # Numeric columns - compute median and percentiles
            values = []
            for df in dfs_list:
                if col == "l":
                    # Handle length-scale specially (can be list)
                    l_means = []
                    for l_val in df[col]:
                        if isinstance(l_val, list):
                            l_means.append(np.mean(l_val))
                        else:
                            try:
                                import ast

                                l_means.append(np.mean(ast.literal_eval(str(l_val))))
                            except:
                                l_means.append(float(l_val))
                    values.append(l_means)
                else:
                    values.append(df[col].values)

            values_array = np.array(values)
            aggregated[f"{col}_median"] = np.median(values_array, axis=0)
            aggregated[f"{col}_p25"] = np.percentile(values_array, 25, axis=0)
            aggregated[f"{col}_p75"] = np.percentile(values_array, 75, axis=0)
        elif col == "gp_converged":
            # Boolean - compute rate
            values = np.array([df[col].astype(int).values for df in dfs_list])
            aggregated[f"{col}_rate"] = np.mean(values, axis=0)

    return aggregated


def plot_hyperparameter_evolution_aggregated(
    aggregated_data: dict, output_path: str, method_name: str = ""
):
    """
    Plot how GP hyperparameters evolve over BO iterations (aggregated across seeds).

    Args:
        aggregated_data: Dict with aggregated GP data across seeds
        output_path: Path to save the plot
        method_name: Name of the method for the title
    """
    if aggregated_data is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"GP Hyperparameter Evolution: {method_name}\n(Median ± IQR across 5 seeds)",
        fontsize=14,
        fontweight="bold",
    )

    iterations = aggregated_data["iter"]

    # Length-scale
    if "l_median" in aggregated_data:
        axes[0, 0].plot(
            iterations, aggregated_data["l_median"], "b-", label="Median", linewidth=2
        )
        axes[0, 0].fill_between(
            iterations,
            aggregated_data["l_p25"],
            aggregated_data["l_p75"],
            alpha=0.3,
            label="IQR (25-75%)",
        )
        axes[0, 0].set_title("Length-Scale Evolution")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Length-scale")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale("log")

    # Signal variance
    if "sigma2_median" in aggregated_data:
        axes[0, 1].plot(
            iterations,
            aggregated_data["sigma2_median"],
            "g-",
            linewidth=2,
            label="Median",
        )
        axes[0, 1].fill_between(
            iterations,
            aggregated_data["sigma2_p25"],
            aggregated_data["sigma2_p75"],
            alpha=0.3,
            label="IQR",
        )
        axes[0, 1].set_title("Signal Variance (σ²) Evolution")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("σ²")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Noise
    if "noise_median" in aggregated_data:
        axes[0, 2].plot(
            iterations,
            aggregated_data["noise_median"],
            "r-",
            linewidth=2,
            label="Median",
        )
        axes[0, 2].fill_between(
            iterations,
            aggregated_data["noise_p25"],
            aggregated_data["noise_p75"],
            alpha=0.3,
            label="IQR",
        )
        axes[0, 2].set_title("Noise Level Evolution")
        axes[0, 2].set_xlabel("Iteration")
        axes[0, 2].set_ylabel("Noise")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # Log Marginal Likelihood
    if "lml_median" in aggregated_data:
        axes[1, 0].plot(
            iterations,
            aggregated_data["lml_median"],
            "purple",
            linewidth=2,
            label="Median",
        )
        axes[1, 0].fill_between(
            iterations,
            aggregated_data["lml_p25"],
            aggregated_data["lml_p75"],
            alpha=0.3,
            label="IQR",
        )
        axes[1, 0].set_title("Log Marginal Likelihood")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("LML")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Convergence status
    if "gp_converged_rate" in aggregated_data:
        conv_rate = aggregated_data["gp_converged_rate"]
        axes[1, 1].plot(
            iterations,
            conv_rate,
            "o-",
            linewidth=2,
            markersize=3,
            label="Convergence Rate",
        )
        axes[1, 1].set_title("GP Optimization Convergence Rate")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Convergence Rate (across seeds)")
        axes[1, 1].set_ylim(-0.1, 1.1)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        # Add summary text
        overall_rate = conv_rate.mean() * 100
        axes[1, 1].text(
            0.05,
            0.95,
            f"Overall convergence: {overall_rate:.1f}%",
            transform=axes[1, 1].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # R² trajectory (to check for overfitting/underfitting)
    if "r2_median" in aggregated_data:
        axes[1, 2].plot(
            iterations,
            aggregated_data["r2_median"],
            "orange",
            linewidth=2,
            label="Median",
        )
        axes[1, 2].fill_between(
            iterations,
            aggregated_data["r2_p25"],
            aggregated_data["r2_p75"],
            alpha=0.3,
            label="IQR",
        )
        axes[1, 2].axhline(y=0, color="k", linestyle="--", alpha=0.3, label="Baseline")
        axes[1, 2].set_title("R² Evolution (Test Set)")
        axes[1, 2].set_xlabel("Iteration")
        axes[1, 2].set_ylabel("R²")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # Add final R² value
        final_r2 = aggregated_data["r2_median"][-1]
        axes[1, 2].text(
            0.05,
            0.05,
            f"Final R² (median): {final_r2:.4f}",
            transform=axes[1, 2].transAxes,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_stability_summary(results_root: str, output_dir: str):
    """
    Create summary plots showing stability issues across all methods.

    Args:
        results_root: Root directory containing all experiment results
        output_dir: Directory to save summary plots
    """
    os.makedirs(output_dir, exist_ok=True)

    stability_data = []

    # Collect stability data from all experiments
    # Structure: results/{dataset}/{method}_{kernel}_{acquisition}/seed{N}/...
    for dataset_dir in sorted(glob.glob(os.path.join(results_root, "*"))):
        dataset_name = os.path.basename(dataset_dir)

        # Skip random search directories
        if "random_search" in dataset_name:
            continue

        for method_dir in sorted(glob.glob(os.path.join(dataset_dir, "*"))):
            method_name = os.path.basename(method_dir)

            for seed_dir in sorted(glob.glob(os.path.join(method_dir, "seed*"))):
                # Check for nested subdirectory structure
                csv_paths = [
                    os.path.join(seed_dir, "bo_iteration_history.csv"),  # Direct path
                    *glob.glob(
                        os.path.join(seed_dir, "*", "seed*", "bo_iteration_history.csv")
                    ),  # Nested
                ]

                for csv_path in csv_paths:
                    if not os.path.isfile(csv_path):
                        continue

                    df = pd.read_csv(csv_path)

                    if "gp_converged" in df.columns and "gp_stable" in df.columns:
                        seed = os.path.basename(seed_dir)

                        convergence_rate = df["gp_converged"].astype(int).mean()
                        stability_rate = (
                            df["gp_stable"].astype(int).mean()
                            if "gp_stable" in df.columns
                            else 1.0
                        )

                        # Count stability issues
                        if "gp_stability_issues" in df.columns:
                            n_issues = df["gp_stability_issues"].notna().sum()
                        else:
                            n_issues = 0

                        stability_data.append(
                            {
                                "method": f"{dataset_name}_{method_name}",
                                "seed": seed,
                                "convergence_rate": convergence_rate,
                                "stability_rate": stability_rate,
                                "n_stability_issues": n_issues,
                                "final_r2": (
                                    df["r2"].iloc[-1] if "r2" in df.columns else np.nan
                                ),
                            }
                        )
                        break  # Only process one file per seed

    if not stability_data:
        print("No stability data found")
        return

    stability_df = pd.DataFrame(stability_data)

    # Plot 1: Convergence rate by method
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=stability_df, x="method", y="convergence_rate", ax=axes[0])
    axes[0].set_title("GP Convergence Rate by Method")
    axes[0].set_xlabel("Method")
    axes[0].set_ylabel("Convergence Rate")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Plot 2: Stability issues by method
    sns.boxplot(data=stability_df, x="method", y="n_stability_issues", ax=axes[1])
    axes[1].set_title("Number of Stability Issues by Method")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Number of Issues")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "gp_stability_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"✅ Stability summary saved to {output_dir}/gp_stability_summary.png")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("GP CONVERGENCE & STABILITY SUMMARY")
    print("=" * 60)
    summary = (
        stability_df.groupby("method")
        .agg(
            {
                "convergence_rate": ["mean", "std"],
                "stability_rate": ["mean", "std"],
                "n_stability_issues": "sum",
                "final_r2": ["mean", "std"],
            }
        )
        .round(4)
    )
    print(summary)


def main():
    """Generate GP diagnostic plots for all experiments."""
    # Configuration
    repo_root = Path(__file__).resolve().parents[1]
    RESULTS_ROOT = str(repo_root / "results")
    OUTPUT_DIR = str(repo_root / "plots" / "gp_diagnostics")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating GP hyperparameter diagnostic plots...")

    # Collect data by dataset/method across all seeds
    method_data = {}  # {(dataset, method): [df1, df2, ...]}

    # Structure: results/{dataset}/{method}_{kernel}_{acquisition}/seed{N}/...
    for dataset_dir in sorted(glob.glob(os.path.join(RESULTS_ROOT, "*"))):
        dataset_name = os.path.basename(dataset_dir)

        # Skip random search directories
        if "random_search" in dataset_name:
            continue

        for method_dir in sorted(glob.glob(os.path.join(dataset_dir, "*"))):
            method_name = os.path.basename(method_dir)
            key = (dataset_name, method_name)

            if key not in method_data:
                method_data[key] = []

            for seed_dir in sorted(glob.glob(os.path.join(method_dir, "seed*"))):
                # Check for nested subdirectory structure
                csv_paths = [
                    os.path.join(seed_dir, "bo_iteration_history.csv"),  # Direct path
                    *glob.glob(
                        os.path.join(seed_dir, "*", "seed*", "bo_iteration_history.csv")
                    ),  # Nested
                ]

                for csv_path in csv_paths:
                    if not os.path.isfile(csv_path):
                        continue

                    try:
                        df = pd.read_csv(csv_path)

                        # Check if new columns exist
                        if "lml" not in df.columns:
                            continue

                        method_data[key].append(df)
                        break  # Only process one file per seed

                    except Exception as e:
                        print(f"Error processing {csv_path}: {e}")

    # Generate aggregated plots
    n_plots = 0
    for (dataset_name, method_name), dfs in sorted(method_data.items()):
        if len(dfs) == 0:
            continue

        try:
            # Aggregate data across seeds
            aggregated = aggregate_gp_data(dfs)

            if aggregated is not None:
                output_filename = f"{dataset_name}_{method_name}_gp_diagnostics.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)

                plot_hyperparameter_evolution_aggregated(
                    aggregated, output_path, f"{dataset_name}_{method_name}"
                )
                n_plots += 1
                print(f"  ✓ {dataset_name}_{method_name} ({len(dfs)} seeds)")

        except Exception as e:
            print(f"  ✗ Error processing {dataset_name}_{method_name}: {e}")

    print(f"\n✅ Generated {n_plots} aggregated hyperparameter evolution plots")
    print(f"\n📊 All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
