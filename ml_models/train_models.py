"""
Script to train and evaluate all ML models.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_io import load_descriptors, load_or_init_cache
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel


def train_and_evaluate_models(
    input_csv: str,
    cache_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train and evaluate all ML models."""
    # Load data
    df = load_descriptors(Path(input_csv))
    smiles = df["SMILES"].tolist()
    X = df.drop(columns="SMILES").values

    # Load energies from cache
    cache = load_or_init_cache(cache_path)
    y = np.array([cache[smi] for smi in smiles])

    # Train-test split
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X, y, smiles, test_size=test_size, random_state=random_state
    )

    # Model configurations
    rf_config = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": random_state,
    }

    xgb_config = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": random_state,
    }

    # Initialize models
    models = {
        "RandomForest": RandomForestModel(rf_config),
        "XGBoost": XGBoostModel(xgb_config),
    }

    results = []

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Cross-validation
        cv_results = model.cross_validate(X_train, y_train, n_splits=5)

        # Train on full training set
        model.train(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Save model
        model_path = output_dir / f"{name.lower()}_model.joblib"
        model.save_model(model_path)

        # Save predictions
        predictions_df = pd.DataFrame(
            {
                "SMILES": smiles_test,
                "True_Energy": y_test,
                f"{name}_Predicted_Energy": y_test_pred,
            }
        )
        predictions_df.to_csv(
            output_dir / f"{name.lower()}_predictions.csv", index=False
        )

        # Collect results
        results.append(
            {
                "Model": name,
                "Train_RMSE": np.sqrt(np.mean((y_train - y_train_pred) ** 2)),
                "Test_RMSE": np.sqrt(np.mean((y_test - y_test_pred) ** 2)),
                "Train_R2": cv_results["r2_train_mean"],
                "Test_R2": cv_results["r2_val_mean"],
                "CV_RMSE_mean": cv_results["rmse_val_mean"],
                "CV_RMSE_std": cv_results["rmse_val_std"],
                "CV_R2_mean": cv_results["r2_val_mean"],
                "CV_R2_std": cv_results["r2_val_std"],
            }
        )

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    print("\nResults saved to:", output_dir)

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ML models")
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to input CSV file with descriptors",
    )
    parser.add_argument(
        "--cache-path", type=str, required=True, help="Path to energy cache file"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state (default: 42)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_and_evaluate_models(
        input_csv=args.input_csv,
        cache_path=Path(args.cache_path),
        output_dir=output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
