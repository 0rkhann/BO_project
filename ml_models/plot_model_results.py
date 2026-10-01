#!/usr/bin/env python3
"""
Create publication-quality plots from trained ML model results.
Recreates the plots shown in the reference images.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_predictions(results_dir):
    """Load prediction results from all datasets and models."""
    results = {}

    datasets = ["mordred", "chemberta2", "dft"]
    models = ["randomforest", "xgboost", "neuralnetwork"]

    for dataset in datasets:
        dataset_dir = results_dir / dataset
        if not dataset_dir.exists():
            print(f"Warning: {dataset_dir} not found, skipping...")
            continue

        results[dataset] = {}

        for model in models:
            pred_file = dataset_dir / f"{model}_predictions.csv"
            if pred_file.exists():
                df = pd.read_csv(pred_file)
                results[dataset][model] = df
            else:
                print(f"Warning: {pred_file} not found, skipping...")

    return results


def create_scatter_plots(results, output_dir):
    """Create 2x3 grid of scatter plots comparing predictions to actual values."""
    # Set style
    sns.set_style("white")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10

    # Dataset and model names for display
    dataset_names = {"mordred": "Mordred", "chemberta2": "ChemBERTa2", "dft": "DFT"}

    model_names = {
        "randomforest": "Random Forest",
        "xgboost": "XGBoost",
    }

    datasets = ["mordred", "chemberta2", "dft"]
    models = ["randomforest", "xgboost"]

    # Create figure (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "ML Model Predictions vs Actual xTB Energies",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for col_idx, dataset in enumerate(datasets):
        for row_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]

            if dataset in results and model in results[dataset]:
                df = results[dataset][model]

                # Get actual and predicted values
                y_true = df["True_Energy"].values
                y_pred_col = [col for col in df.columns if "Predicted" in col][0]
                y_pred = df[y_pred_col].values

                # Calculate metrics
                from sklearn.metrics import r2_score, mean_squared_error

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                # Scatter plot
                ax.scatter(
                    y_true,
                    y_pred,
                    alpha=0.4,
                    s=8,
                    color="#FF6B6B",
                    edgecolors="none",
                    rasterized=True,
                )

                # Perfect prediction line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax.plot(
                    [min_val, max_val], [min_val, max_val], "r--", lw=1.5, alpha=0.6
                )

                # Labels and title
                ax.set_xlabel("Actual Energy (kcal/mol)", fontsize=9)
                ax.set_ylabel("Predicted Energy (kcal/mol)", fontsize=9)

                title = f"{dataset_names[dataset]} - {model_names[model]}\n"
                title += f"R² = {r2:.3f}, RMSE = {rmse:.1f}"
                ax.set_title(title, fontsize=9, fontweight="bold")

                ax.grid(True, alpha=0.2, linewidth=0.5)
                ax.tick_params(labelsize=8)

                # Set equal aspect for square plots
                ax.set_aspect("equal", adjustable="box")
            else:
                ax.axis("off")

    plt.tight_layout()
    output_path = output_dir / "ml_predictions_vs_actual.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_bar_charts(results, output_dir):
    """Create bar charts comparing RMSE and R² across datasets and models."""
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300

    dataset_names = {"mordred": "Mordred", "chemberta2": "ChemBERTa2", "dft": "DFT"}

    model_names = {
        "randomforest": "Random Forest",
        "xgboost": "XGBoost",
    }

    datasets = ["mordred", "chemberta2", "dft"]
    models = ["randomforest", "xgboost"]

    # Collect metrics
    rmse_data = {model: [] for model in models}
    r2_data = {model: [] for model in models}

    for dataset in datasets:
        for model in models:
            if dataset in results and model in results[dataset]:
                df = results[dataset][model]
                y_true = df["True_Energy"].values
                y_pred_col = [col for col in df.columns if "Predicted" in col][0]
                y_pred = df[y_pred_col].values

                from sklearn.metrics import r2_score, mean_squared_error

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                rmse_data[model].append(rmse)
                r2_data[model].append(r2)
            else:
                rmse_data[model].append(0)
                r2_data[model].append(0)

    # Create figure with extra space for legend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(
        "ML Model Performance with xTB Energies", fontsize=13, fontweight="bold"
    )

    x = np.arange(len(datasets))
    width = 0.35
    colors = ["#FFB6C1", "#DAA520"]

    # RMSE comparison
    for idx, model in enumerate(models):
        offset = (idx - len(models) / 2 + 0.5) * width
        ax1.bar(
            x + offset,
            rmse_data[model],
            width,
            label=model_names[model],
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
        )

    ax1.set_xlabel("Dataset", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RMSE (kcal/mol)", fontsize=11, fontweight="bold")
    ax1.set_title("Test RMSE Comparison", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([dataset_names[d] for d in datasets])
    ax1.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    ax1.set_axisbelow(True)

    # R² comparison
    for idx, model in enumerate(models):
        offset = (idx - len(models) / 2 + 0.5) * width
        ax2.bar(
            x + offset,
            r2_data[model],
            width,
            label=model_names[model],
            color=colors[idx],
            edgecolor="black",
            linewidth=0.5,
        )

    ax2.set_xlabel("Dataset", fontsize=11, fontweight="bold")
    ax2.set_ylabel("R²", fontsize=11, fontweight="bold")
    ax2.set_title("Test R² Comparison", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([dataset_names[d] for d in datasets])
    ax2.grid(True, alpha=0.3, axis="y", linewidth=0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_axisbelow(True)

    # Add single legend outside the plot area
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path = output_dir / "ml_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_summary_table(results, output_dir):
    """Create a summary table of all results."""
    dataset_names = {"mordred": "Mordred", "chemberta2": "ChemBERTa2", "dft": "DFT"}

    model_names = {
        "randomforest": "Random Forest",
        "xgboost": "XGBoost",
    }

    datasets = ["mordred", "chemberta2", "dft"]
    models = ["randomforest", "xgboost"]

    summary_data = []

    for dataset in datasets:
        for model in models:
            if dataset in results and model in results[dataset]:
                df = results[dataset][model]
                y_true = df["True_Energy"].values
                y_pred_col = [col for col in df.columns if "Predicted" in col][0]
                y_pred = df[y_pred_col].values

                from sklearn.metrics import r2_score, mean_squared_error

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                summary_data.append(
                    {
                        "Dataset": dataset_names[dataset],
                        "Model": model_names[model],
                        "Test_RMSE": round(rmse, 1),
                        "Test_R²": round(r2, 3),
                    }
                )

    summary_df = pd.DataFrame(summary_data)
    output_path = output_dir / "ml_model_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")

    print("\n" + "=" * 70)
    print("Model Performance Summary:")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Create plots from trained ML model results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="ml_results",
        help="Directory containing model results (default: ml_results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml_plots",
        help="Directory to save plots (default: ml_plots)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Loading ML Model Results and Creating Plots")
    print(f"{'='*70}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Load results
    print("\nLoading prediction results...")
    results = load_predictions(results_dir)

    if not results:
        print("❌ No results found! Please train models first.")
        print("\nTo train models, run:")
        print("  python -m ml_models.train_models --input-csv data/dft_mordred.csv \\")
        print("      --cache-path data/dft_G.json --output-dir ml_results/mordred")
        return

    print(f"Loaded results for {len(results)} dataset(s)")

    # Create plots
    print("\n📊 Creating scatter plots...")
    create_scatter_plots(results, output_dir)

    print("\n📊 Creating bar charts...")
    create_bar_charts(results, output_dir)

    print("\n📊 Creating summary table...")
    create_summary_table(results, output_dir)

    print(f"\n{'='*70}")
    print("✅ All plots created successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
