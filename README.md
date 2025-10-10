# Bayesian Optimization for N-heterocyclic carbenes (NHCs) design

This project implements Bayesian Optimization with multiple feature selection methods for discovering molecules with optimal binding free energies

## Quick Start

### Local Execution

```bash
# 1. Setup Python environment (first time only)
python -m venv .bo_project_env
source .bo_project_env/bin/activate
pip install -r requirements.txt

# 2. Run all DFT experiments (60 BO + 15 Random = 75 total)
./run_all_dft_experiments.sh

# 3. Train ML baseline models (optional)
./train_all_ml_models.sh

# 4. Generate comparison plots
python3 plotting/plot_comprehensive_comparison.py
python3 plotting/plot_gp_diagnostics.py
```

### Cluster Execution (Hábrók)

For running on SLURM cluster with job arrays:

```bash
# 1. Connect and setup
ssh <your_id>@login2.hb.hpc.rug.nl
cd ~/bo_project
module load Python/3.11.3-GCCcore-12.3.0
python -m venv .bo_project_env
source .bo_project_env/bin/activate
pip install -r requirements.txt

# 2. Submit experiments (array job for parallel execution)
sbatch submit_all_experiments.sh
```

**Note:** `submit_all_experiments.sh` runs a different configuration:
- 5 methods (includes vanilla), 2 kernels, 2 acquisitions, 1 seed = 60 experiments
- Good for exploring hyperparameter space
- Runs in parallel using SLURM array jobs

For the validated configuration (4 methods, 5 seeds, Matern+EI only), use `run_all_dft_experiments.sh` instead.

## Experiments Overview

**Current implementation** (recommended configuration):
- **3 DFT datasets**: dft_descriptors (30 features), dft_chemberta2 (384 features), dft_mordred (1,469 features)
- **4 feature selection methods**: fabo, pls, pca, opls
- **1 GP kernel**: Matern
- **1 acquisition function**: EI (Expected Improvement)
- **5 random seeds**: 42, 43, 44, 45, 46
- **Plus Random Search baseline** for comparison

**Total: 75 experiments**
- 60 BO experiments (3 datasets × 4 methods × 5 seeds)
- 15 Random Search experiments (3 datasets × 5 seeds)

Each experiment: 100 iterations, 10 initial samples, validated with held-out test set.

## Project Structure

```
bo_project/
├── data/                              # Molecular datasets
│   ├── dft_G.json                    # DFT target binding energies
│   ├── dft_descriptors.csv           # DFT-based descriptors (30 features)
│   ├── dft_chemberta2.csv            # ChemBERTa2 embeddings (384 features)
│   ├── dft_mordred.csv               # Mordred descriptors (1,469 features)
│   ├── xtb_G.json                    # XTB target energies
│   └── xtb_mordred.csv               # XTB + Mordred descriptors
├── src/                               # Source code
│   ├── pipelines/                    # BO pipelines for each method
│   ├── feature_selection/            # Feature selection implementations
│   ├── kernels/                      # GP kernels (Matern, RBF, RQ)
│   └── acquisition/                  # Acquisition functions (EI, UCB, PI)
├── baselines/                         # Baseline methods
│   └── random_search.py              # Random search baseline
├── ml_models/                         # Supervised ML baselines
│   ├── train_models.py               # Training script
│   ├── random_forest.py              # Random Forest model
│   └── xgboost_model.py              # XGBoost model
├── plotting/                          # Visualization scripts
│   ├── plot_comprehensive_comparison.py  # Main comparison plots
│   └── plot_gp_diagnostics.py        # GP hyperparameter diagnostics
├── run_all_dft_experiments.sh        # Run all DFT experiments (BO + Random)
├── train_all_ml_models.sh            # Train ML baseline models
├── submit_all_experiments.sh         # SLURM cluster submission
├── results/                           # Experiment results (generated)
│   ├── dft_descriptors/              # DFT descriptors results
│   ├── dft_chemberta2/               # ChemBERTa2 results
│   └── dft_mordred/                  # Mordred results
├── plots/                             # Generated plots
│   ├── comprehensive_comparison/     # Method comparison plots
│   └── gp_diagnostics/               # GP hyperparameter diagnostics
└── ml_results/                        # ML model outputs
    ├── dft/                          # DFT descriptors ML results
    ├── chemberta2/                   # ChemBERTa2 ML results
    └── mordred/                      # Mordred ML results
```

## Feature Selection Methods

1. **Vanilla**: No feature selection (baseline)
2. **FABO**: Feature-Aware BO using Spearman correlation + cross-validation
3. **PCA**: Principal Component Analysis
4. **PLS**: Partial Least Squares
5. **OPLS**: Orthogonal Partial Least Squares

## Results

Results are saved in: `results/<dataset>/<method>_<kernel>_<acquisition>/seed*/`

Each result directory contains:
- `bo_iteration_history.csv` - BO iteration metrics including:
  - GP hyperparameters (length-scale, signal variance, noise)
  - Test set performance (RMSE, R², AE, Pearson r)
  - Best energies found (best_bo, best_pool_min)
  - GP convergence status and stability
  - Memory usage tracking

## Visualization

After experiments complete, generate performance plots:

```bash
# Generate comprehensive comparison plots (recommended)
python3 plotting/plot_comprehensive_comparison.py

# Generate GP hyperparameter diagnostics
python3 plotting/plot_gp_diagnostics.py
```

### Comprehensive Comparison Plots (`plots/comprehensive_comparison/`)

**4 plots generated:**
- `comparison_dft_descriptors.png` - All methods vs Random Search (DFT descriptors)
- `comparison_dft_chemberta2.png` - All methods vs Random Search (ChemBERTa2)
- `comparison_dft_mordred.png` - All methods vs Random Search (Mordred)
- `best_methods_comparison.png` - Best performing method from each dataset

All plots show **median ± IQR (25th-75th percentile) across 5 seeds**.

### GP Diagnostics Plots (`plots/gp_diagnostics/`)

**12 plots** (one per dataset/method combination, aggregated across 5 seeds):
- `{dataset}_{method}_Matern_EI_gp_diagnostics.png` - 6-panel diagnostics showing:
  - Length-scale evolution
  - Signal variance (σ²) evolution  
  - Noise level evolution
  - Log Marginal Likelihood (LML)
  - GP convergence rate (100% across all methods)
  - R² on test set

## ML Baseline Models

To assess the intrinsic difficulty of the optimization problem, train supervised ML models on the full dataset:

```bash
# Train Random Forest and XGBoost models on all datasets
./train_all_ml_models.sh
```

This script trains models on:
- **DFT descriptors** (30 features)
- **ChemBERTa2 embeddings** (384 features)  
- **Mordred descriptors** (1,469 features)

**Generated outputs:**
- `ml_results/{dataset}/` - Trained models (.joblib) and predictions
- `ml_plots/ml_performance_comparison.png` - Performance comparison across datasets
- `ml_plots/ml_predictions_vs_actual.png` - Prediction quality visualization
- `ml_plots/ml_model_summary.csv` - Detailed metrics (RMSE, R², CV scores)

**Interpretation:**
- **High R² (>0.7)**: Property is highly predictable → easier for BO
- **Low R² (<0.3)**: Noisy/complex property → harder for BO, but BO success more significant
- Current binding free energy: **R² ≈ 0.17-0.27** (challenging optimization problem)

## Resource Requirements

**Cluster Storage**: ~250 GB required because vanilla method requires a lot of memory
