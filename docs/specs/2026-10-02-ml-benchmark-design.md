# Tuned ML benchmark with Chemprop and tabular foundation models

Date: 2026-10-02 · Status: design approved, awaiting spec review

## Goal

Find out whether the low hold-out R² of the current surrogates (0.13 to 0.28) is a ceiling of the data or of the models. To test it, add stronger models and tune every model with one shared protocol. The output is a benchmark section in the README.

Out of scope: replacing the GP inside the BO loop. `src/` and `results/` do not change.

## Decisions

| Topic | Decision |
|---|---|
| Purpose | Benchmark only. No BO integration. |
| Models | Random Forest, XGBoost, TabPFN-3, TabICLv2, Chemprop |
| Chemprop inputs | Two variants: SMILES only, and SMILES + the 29 DFT descriptors as extra molecule features |
| Tabular representations | DFT descriptors (29), ChemBERTa-2 (384), Mordred (1,469) |
| Evaluation | Nested: 5 outer folds, with tuning on inner folds of the outer training part only |
| Tuning | Optuna TPE for every model, minimising inner-CV RMSE |
| Model comparison | Nadeau–Bengio corrected resampled t-test, Holm correction |
| Compute | GitHub Actions CPU, one job per (model, representation, outer fold) |
| Target | `data/dft_G.json`: xTB binding free energies in kJ/mol, 6,850 molecules |

## Dependencies

These go in a separate pinned `requirements-ml.txt`, so the BO install does not grow:

| Package | Version | Licence |
|---|---|---|
| `chemprop` | 2.3.1 | MIT. Needs Python ≥ 3.11. |
| `tabpfn` | 9.0.0 | Code Apache-2.0. TabPFN-3 weights are under a non-commercial licence; research use is allowed. The weights are not gated. |
| `tabicl` | 2.2.0 | BSD-3-Clause |
| `optuna` | 5.0.0 | MIT |

XGBoost and scikit-learn stay at their current versions. The README states the TabPFN-3 weight licence.

## Layout

```
ml_models/benchmark/
  data.py        load a representation as (X, y, SMILES), aligned by SMILES; build the outer folds once
  models/        one adapter per model:
                   fit(X_tr, y_tr, params), predict(X), search_space(trial) -> params
    rf.py  xgb.py  tabpfn.py  tabicl.py  chemprop.py
  tune.py        Optuna loop on inner folds; trial cap plus wall-time cap; returns best params and the trial log
  run.py         CLI: --model --rep --fold; tune, refit on the outer training part, predict the outer test fold
  summarize.py   combine folds; metrics, corrected t-tests, summary tables
.github/workflows/ml-benchmark.yml
requirements-ml.txt
```

## Protocol

1. **Outer folds.**
   - `KFold(n_splits=5, shuffle=True, random_state=0)` over the 6,850 molecules, in the row order of `data/dft_G.json`.
   - The folds are stored by SMILES in `ml_results/benchmark/folds.json`, so every model and representation uses the same split.
   - Each feature table is aligned to the folds by SMILES, never by row position.
2. **Tuning.** This step uses the outer training part only.
   - Optuna's TPE sampler (`seed=0`) minimises inner-CV RMSE.
   - Each run stops at its trial cap or its wall-time cap, whichever comes first.
3. **Refit.** The best parameters are refit on the whole outer training part, and that model predicts the outer test fold once.
4. **Metrics.** R², RMSE, MAE and Spearman on the outer test fold.

### Search spaces and budgets (per outer fold)

| Model | Space | Budget | Inner validation |
|---|---|---|---|
| Random Forest | `n_estimators` 300–1000; `max_features` 0.1–1.0; `min_samples_leaf` 1–10; `max_depth` {None, 10–40} | 50 trials | 3-fold |
| XGBoost | `n_estimators` 200–2000; `learning_rate` 0.01–0.3 (log); `max_depth` 3–10; `min_child_weight` 1–20; `subsample` 0.5–1; `colsample_bytree` 0.3–1; `reg_lambda`, `reg_alpha` 1e-3–10 (log) | 50 trials | 3-fold |
| TabPFN-3 | `n_estimators` {8, 16} × `softmax_temperature` {0.75, 0.9}; default regression checkpoint | 4-config grid | 3-fold |
| TabICLv2 | `n_estimators` {8, 16, 32}; default checkpoint | 3-config grid | 3-fold |
| Chemprop | `depth` 2–6; `message_hidden_dim` 300–1200; `ffn_num_layers` 1–3; `ffn_hidden_dim` 300–1200; `dropout` 0–0.4; `max_lr` 1e-4–3e-3 (log); up to 100 epochs, early stopping with patience 15 | 20 trials | one 90/10 split |

- **Chemprop refit:** the final model keeps 10% of the outer training part for early stopping.
- **Descriptor scaling:** Chemprop uses its own feature scaling for the descriptor variant.
- **Grid sizes for the foundation models are provisional.** The first implementation task times one inner fit per model and representation on the Actions runner. Grids are shrunk only by that rule, and the README records any reduction. If a cell cannot fit even one configuration within the job limit, it is reported as "not run (CPU budget)".

## Outputs

```
ml_results/benchmark/
  folds.json
  <model>/<rep>/fold<k>/metrics.json      # status, metrics, n_trials, wall time, versions
  <model>/<rep>/fold<k>/predictions.csv   # SMILES, true, predicted (outer test fold)
  <model>/<rep>/fold<k>/best_params.json
  <model>/<rep>/fold<k>/trials.csv
  summary.csv        # mean ± sd per (model, rep), and folds completed
  comparisons.csv    # corrected t-tests: model vs tuned RF per rep; best vs others per rep
  oof_predictions/<model>_<rep>.csv   # out-of-fold predictions for all 6,850 molecules
```

For Chemprop, `<rep>` is `smiles` or `smiles+dft_descriptors`.

## Statistics

- Per (model, representation): mean ± sd over the 5 outer folds of R², RMSE, MAE and Spearman.
- **Nadeau–Bengio corrected resampled t-test**, with variance correction `1/k + n_test/n_train` (0.25 here).
- **Families:**
  - every model against tuned Random Forest within each representation;
  - the best model of each representation against the others.
- Holm correction within each family. Effect sizes are reported as ΔR² and ΔRMSE.

## Failure handling

- **Failed folds stay visible.** A failed run still writes `metrics.json` with `status: "failed"` and the error message. The summary shows folds completed per cell. Cells with missing folds are left out of the tests and marked in the README.
- **Input validation:** SMILES must align between features and targets, there must be no NaN targets, and every SMILES must parse in RDKit before Chemprop starts. A failed check stops the job with a clear error.
- **Time cap:** hitting the time cap is not a failure. `trials.csv` records the trials that finished.

## Compute

- **Workflow:** `ml-benchmark.yml` is started manually (`workflow_dispatch`).
- **Matrix:** (4 tabular models × 3 representations + 2 Chemprop variants) × 5 folds = 70 jobs, with up to 20 running in parallel and a 340-minute timeout each.
- **Clean artifacts:** each job deletes `results/` and `ml_results/` from its checkout before running.
- **Collect job:** checks out the latest `main` and commits only `ml_results/benchmark/` to `experiment-results/ml-<run id>`. The run summary links that branch.
- **Caching:** pip downloads and the TabPFN and TabICL weights are cached between jobs.

## Figures and README

- **Figures:** light and dark variants, in the existing style.
  - **ML figure 1:** R² and Spearman, mean ± sd, per model and representation.
  - **ML figure 2:** parity plot of the best model's out-of-fold predictions on all 6,850 molecules. It replaces the current parity plot.
- **README:**
  - The ML section is rewritten with the summary table and both figures. It ends with one line on whether the R² ceiling moved.
  - The "hard to learn" limitation is updated.
  - Reproduction gets the workflow dispatch and a single-fold command.
  - A TabPFN-3 licence note is added.

## Cleanup (after the new results exist)

- **Delete:**
  - the old single-split models: `ml_models/train_models.py`, `random_forest.py`, `xgboost_model.py`, `base_model.py`, `plot_model_results.py`;
  - their results in `ml_results/{dft,chemberta2,mordred}/` and `ml_plots/ml_model_summary.csv`;
  - `train_all_ml_models.sh`, which runs the old models.
- **Repoint:** `scripts/analyze_results.py` and `scripts/summarize_results.py` read the new `ml_results/benchmark/summary.csv`.
- **Remove:** `scripts/recompute_holdout_metrics.py` exists only for the old tables.
- **Keep:** `scripts/diagnose_ml_r2.py`, as the explanation of the R² ceiling.

## Tests (written before the code)

- The folds are deterministic, disjoint and cover all 6,850 SMILES. Two feature tables in different row order map to the same folds.
- Each adapter fits and predicts on a small synthetic set. TabPFN, TabICL and Chemprop use `pytest.importorskip`, so the BO suite runs without `requirements-ml.txt`.
- `tune.py` stops at its time cap and records the trials that finished.
- `run.py` works end to end with Random Forest on synthetic data, and a forced error writes `status: "failed"`.
- The corrected t-test matches a hand-computed example.

## Risks

- **CPU time** for TabPFN-3 and TabICL on Mordred (1,469 features) and for Chemprop tuning. The timing task above handles this.
- **Licence:** TabPFN-3 weights are non-commercial. The project is research, so this is acceptable, and the README states it.
- **Few folds:** with 5 outer folds, the corrected t-test is conservative, so small gaps between models will not reach significance. The README reports effect sizes alongside p-values.
