#!/usr/bin/env python3
"""Recompute hold-out RMSE / R² from the saved ML predictions (no retraining).

Reads ml_results/<dataset>/<model>_predictions.csv (True_Energy vs the
*_Predicted_Energy column) and updates ml_results/<dataset>/model_comparison.csv
so that it matches the current ml_models/train_models.py output:
  Test_RMSE, Test_R2      -> hold-out values from the predictions file
  Train_R2 (was CV mean)  -> renamed CV_Train_R2_mean
The old Test_R2 duplicated CV_R2_mean, so it is simply replaced.

Usage: python scripts/recompute_holdout_metrics.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
ML_RESULTS = ROOT / "ml_results"
DATASETS = {"dft": "DFT descriptors", "mordred": "Mordred", "chemberta2": "ChemBERTa-2"}
MODELS = {"RandomForest": "Random Forest", "XGBoost": "XGBoost"}


def holdout_metrics(dataset, model):
    df = pd.read_csv(ML_RESULTS / dataset / f"{model.lower()}_predictions.csv")
    y_true = df["True_Energy"].to_numpy()
    y_pred = df[f"{model}_Predicted_Energy"].to_numpy()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))), float(r2_score(y_true, y_pred)), len(df)


def collect():
    """Hold-out metrics for every dataset/model as a DataFrame."""
    rows = []
    for ds, label in DATASETS.items():
        for model, model_label in MODELS.items():
            rmse, r2, n = holdout_metrics(ds, model)
            rows.append(
                {"dataset": ds, "Representation": label, "model": model,
                 "Model": model_label, "Test_RMSE": rmse, "Test_R2": r2, "n_test": n}
            )
    return pd.DataFrame(rows)


def update_model_comparison(metrics):
    for ds, g in metrics.groupby("dataset"):
        path = ML_RESULTS / ds / "model_comparison.csv"
        mc = pd.read_csv(path)
        if "Train_R2" in mc.columns:
            mc = mc.rename(columns={"Train_R2": "CV_Train_R2_mean"})
        for _, r in g.iterrows():
            i = mc.index[mc["Model"] == r.model][0]
            mc.loc[i, "Test_RMSE"] = r.Test_RMSE
            mc.loc[i, "Test_R2"] = r.Test_R2
        cols = ["Model", "Train_RMSE", "Test_RMSE", "Test_R2", "CV_Train_R2_mean",
                "CV_RMSE_mean", "CV_RMSE_std", "CV_R2_mean", "CV_R2_std"]
        mc[cols].to_csv(path, index=False)


if __name__ == "__main__":
    m = collect()
    update_model_comparison(m)
    print(m.to_string(index=False))
