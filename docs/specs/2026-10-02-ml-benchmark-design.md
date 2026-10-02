# Tuned ML benchmark with Chemprop and tabular foundation models

Date: 2026-10-02 · Status: design approved and refined for gaps, awaiting spec review

## Goal

Find out whether the low hold-out R² of the current surrogates (0.13 to 0.28) is a ceiling of the data or of the models. To test it, compare a broad set of model families under one shared, tuned protocol: a constant baseline, linear models, trees, the BO loop's own Gaussian process, a graph neural network and tabular foundation models. The benchmark also reports how well each model ranks the low-energy tail, which is what BO needs. The output is a benchmark section in the README.

Out of scope: replacing the GP inside the BO loop. `src/` and `results/` do not change.

## Decisions

| Topic | Decision |
|---|---|
| Purpose | Benchmark only. No BO integration. |
| Models | Mean predictor, Ridge, Random Forest, XGBoost, exact GP (the BO surrogate), TabPFN-3, TabICLv2, Chemprop |
| Chemprop inputs | Two variants: SMILES only, and SMILES + the 29 DFT descriptors as extra molecule features |
| Tabular representations | DFT descriptors (29), ChemBERTa-2 (384), Mordred (1,469 columns, 1,197 non-constant) |
| Evaluation | Nested: 5 outer folds, with tuning on inner splits of the outer training part only |
| Tuning | Optuna TPE for the tunable models, minimising inner-CV RMSE. The GP fits its hyperparameters by marginal likelihood, as in the BO loop. |
| Metrics | R², RMSE, MAE, Spearman, plus two tail metrics: top-1% recall at 5% and RMSE on the lowest-energy decile |
| Model comparison | Nadeau–Bengio corrected resampled t-test, Holm correction |
| Headline rule | "The ceiling moved" only if the best tuned model beats tuned Random Forest by ΔR² ≥ 0.05 with Holm p < 0.05 on at least one representation. Otherwise the README states that the ceiling holds. |
| Compute | GitHub Actions CPU, one job per (model, representation, outer fold) |
| TabPFN-3 on CPU | Run with the CPU size guard overridden. If too slow, first reduce to the default configuration only; if even that does not fit, report "not run (CPU budget)". |
| CI | A pytest workflow on every pull request |
| Target | `data/dft_G.json`: xTB binding free energies in kJ/mol, 6,850 molecules |

## Data facts (checked on 2026-10-02)

- All 6,850 SMILES parse in RDKit (13–51 heavy atoms). There are no canonical duplicates, and no duplicates even when stereochemistry is ignored, so a plain random split cannot leak the same molecule into train and test.
- All three feature tables contain exactly the 6,850 target SMILES, with no NaN and no non-numeric cells. The descriptor table is in a different row order from the other two, so alignment must go by SMILES.
- Mordred has 272 constant columns.
- No descriptor column leaks the target: the strongest single correlation is PA at |r| = 0.32.
- NMR, `f+`, `f-` and `fdual` are almost perfectly collinear (a known data issue, unchanged here).

The tests assert the first two facts on the real data, so a changed data file fails loudly.

## Dependencies

These go in a separate pinned `requirements-ml.txt`, plus a fully resolved lock file `requirements-ml.lock` compiled for Python 3.11. The BO install stays as it is.

| Package | Version | Licence |
|---|---|---|
| `chemprop` | 2.3.1 | MIT. Needs Python ≥ 3.11. |
| `tabpfn` | 9.0.0 | Code Apache-2.0. TabPFN-3 weights are under a non-commercial licence; research use is allowed. The weights are not gated. |
| `tabicl` | 2.2.0 | BSD-3-Clause |
| `optuna` | 5.0.0 | MIT |
| `botorch`, `gpytorch` | the versions the BO code uses, re-resolved against torch 2.14 | MIT |

A trial resolution on 2026-10-02 (without BoTorch) gave torch 2.14.1+cpu, numpy 2.4.6, scikit-learn 1.9.1, xgboost 3.2.0, lightning 2.6.6 and rdkit 2026.3.6. The first implementation task re-resolves the set with BoTorch included. If BoTorch does not resolve, the GP baseline runs in the BO environment as a separate job type rather than forcing versions. The README states the TabPFN-3 weight licence.

## Layout

```
ml_models/benchmark/
  data.py        load a representation as (X, y, SMILES), aligned by SMILES; build the outer folds once
  models/        one adapter per model:
                   fit(X_tr, y_tr, params), predict(X), search_space(trial) -> params
    mean.py  ridge.py  rf.py  xgb.py  gp.py  tabpfn.py  tabicl.py  chemprop.py
  tune.py        Optuna loop on inner splits; trial cap plus wall-time cap; returns best params and the trial log
  run.py         CLI: --model --rep --fold; tune, refit on the outer training part, predict the outer test fold
  summarize.py   combine folds; metrics, corrected t-tests, summary tables; merges partial reruns
.github/workflows/ml-benchmark.yml
.github/workflows/tests.yml
requirements-ml.txt  requirements-ml.lock
```

## Protocol

1. **Outer folds.**
   - `KFold(n_splits=5, shuffle=True, random_state=0)` over the 6,850 molecules, in the row order of `data/dft_G.json`.
   - The folds are stored by SMILES in `ml_results/benchmark/folds.json`, so every model and representation uses the same split.
   - Each feature table is aligned to the folds by SMILES, never by row position.
2. **Preprocessing inside the split.** Every data-dependent step is fitted on the training part of the current split only, in both the inner and the outer loop:
   - dropping zero-variance columns (this removes Mordred's 272 constant columns and any column that becomes constant within a fold);
   - feature standardisation where a model needs it (Ridge, GP);
   - Chemprop's descriptor and target scaling.

   Tree models and the foundation models receive the unscaled columns; the foundation models do their own preprocessing.
3. **Tuning.** This step uses the outer training part only.
   - Optuna's TPE sampler (`seed=0`) minimises inner-CV RMSE.
   - Each run stops at its trial cap or its wall-time cap, whichever comes first.
4. **Refit.** The best parameters are refit on the whole outer training part, and that model predicts the outer test fold once.
5. **Metrics on the outer test fold:**
   - R², RMSE, MAE and Spearman;
   - **top-1% recall at 5%**: of the molecules in the lowest 1% of true energy in the test fold (about 14), the share that are among the lowest 5% of predicted energy;
   - **tail RMSE**: RMSE on the 10% of test molecules with the lowest true energy.

### Search spaces and budgets (per outer fold)

| Model | Space | Budget | Inner validation |
|---|---|---|---|
| Mean predictor | none | — | — |
| Ridge | `alpha` 1e-3–1e3 (log) | 30 trials | 3-fold |
| Random Forest | `n_estimators` 300–1000; `max_features` 0.1–1.0; `min_samples_leaf` 1–10; `max_depth` {None, 10–40} | 50 trials | 3-fold |
| XGBoost | `n_estimators` 200–2000; `learning_rate` 0.01–0.3 (log); `max_depth` 3–10; `min_child_weight` 1–20; `subsample` 0.5–1; `colsample_bytree` 0.3–1; `reg_lambda`, `reg_alpha` 1e-3–10 (log) | 50 trials | 3-fold |
| Exact GP | the BO loop's `build_gp` and `fit_gp` (Matern 5/2, ARD, same priors); hyperparameters fitted by marginal likelihood | one fit | — |
| TabPFN-3 | `n_estimators` {8, 16} × `softmax_temperature` {0.75, 0.9}; default regression checkpoint; `ignore_pretraining_limits=True` | 4-config grid | 3-fold |
| TabICLv2 | `n_estimators` {8, 16, 32}; default checkpoint | 3-config grid | 3-fold |
| Chemprop | `depth` 2–6; `message_hidden_dim` 300–1200; `ffn_num_layers` 1–3; `ffn_hidden_dim` 300–1200; `dropout` 0–0.4; `max_lr` 1e-4–3e-3 (log); up to 100 epochs, early stopping with patience 15 | 20 trials | one 90/10 split |

- **GP on Mordred:** uses `--prune-correlated 0.95` (1,469 → 523 columns), the same as vanilla BO on Mordred, so the benchmark GP matches the BO surrogate.
- **Chemprop splits:** Chemprop is given our fold indices explicitly; its own splitting is never used. The final model keeps 10% of the outer training part for early stopping.
- **Foundation-model grids are provisional:**
  - The first implementation task times one inner fit per model and representation on the Actions runner.
  - TabPFN raises an error on CPU above 1,000 training rows by default (`MAX_CPU_SAMPLES = 1000`), so `ignore_pretraining_limits=True` is required.
  - If the grid does not fit in a job, the cell runs the default configuration only, without inner tuning. If even that does not fit, the cell is reported as "not run (CPU budget)".
  - The README records every reduction.
- **Refit reserve:** each job's tuning cap is its 340-minute budget minus twice the measured refit time, so tuning can never consume the time needed for the final fit.

## Outputs

```
ml_results/benchmark/
  folds.json
  <model>/<rep>/fold<k>/metrics.json      # status, all metrics, n_trials, wall time, peak RSS,
                                           # package versions, git SHA, checkpoint name, Actions run id
  <model>/<rep>/fold<k>/predictions.csv   # SMILES, true, predicted (outer test fold)
  <model>/<rep>/fold<k>/best_params.json
  <model>/<rep>/fold<k>/trials.csv
  summary.csv        # mean ± sd per (model, rep), and folds completed
  comparisons.csv    # corrected t-tests; ceiling-rule outcome
  oof_predictions/<model>_<rep>.csv   # out-of-fold predictions for all 6,850 molecules
```

For Chemprop, `<rep>` is `smiles` or `smiles+dft_descriptors`. Out-of-fold predictions combine five separately tuned models, one per fold, and the README says so where they are plotted.

## Statistics

- Per (model, representation): mean ± sd over the 5 outer folds of every metric.
- **Nadeau–Bengio corrected resampled t-test**, with variance correction `1/k + n_test/n_train` (0.25 here), on R² and on top-1% recall.
- **Families:**
  - every model against tuned Random Forest within each representation;
  - the best model of each representation against the others.
- Holm correction within each family. Effect sizes are reported as ΔR², ΔRMSE and Δrecall.
- `comparisons.csv` records the outcome of the headline rule.

## Failure handling

- **Failed folds stay visible.** A failed run still writes `metrics.json` with `status: "failed"` and the error message. The summary shows folds completed per cell. Cells with missing folds are left out of the tests and marked in the README.
- **Input validation:** the data facts above are checked at load time (SMILES alignment, no NaN targets, RDKit parsing). A failed check stops the job with a clear error.
- **Time cap:** hitting the time cap is not a failure. `trials.csv` records the trials that finished.
- **Partial reruns:**
  - The workflow can rerun selected models, representations or folds and merge them into the existing `ml_results/benchmark/`, replacing only the cells it reran.
  - `summarize.py` refuses to combine folds from different package versions within one cell.

## Compute

- **Workflow:** `ml-benchmark.yml` is started manually (`workflow_dispatch`), with inputs to select models, representations and folds.
- **Matrix:**
  - 7 tabular models × 3 representations + 2 Chemprop variants = 23 cells, × 5 folds = 115 jobs.
  - The mean predictor and Ridge are run together in one job per representation and fold, giving 100 jobs.
  - Up to 20 jobs run in parallel, each with a 340-minute timeout.
- **Threads:** `n_jobs` and the torch thread count are set to the runner's core count; Optuna runs trials one at a time.
- **Clean artifacts:** each job deletes `results/` and `ml_results/` from its checkout before running.
- **Collect job:** checks out the latest `main` and commits only `ml_results/benchmark/` to `experiment-results/ml-<run id>`. The run summary links that branch. In merge mode, the collect job first copies in the existing `ml_results/benchmark/` from `main`.
- **Caching:** pip downloads and the TabPFN and TabICL weights are cached between jobs.

## CI

`tests.yml` runs on every pull request:
- the BO test suite, with the BO requirements;
- the fast ML tests (folds, mean, Ridge, Random Forest, tuner, summary, real-data facts).

The TabPFN, TabICL, Chemprop and GP adapter tests skip with `pytest.importorskip` when their packages are absent. The fast job has a 15-minute timeout.

## Figures and README

- **Figures:** light and dark variants, in the existing style.
  - **ML figure 1:** R², Spearman and top-1% recall, mean ± sd, per model and representation. The mean predictor and random-ranking expectation (5%) are drawn as reference lines.
  - **ML figure 2:** parity plot of the best model's out-of-fold predictions on all 6,850 molecules. It replaces the current parity plot.
- **README:**
  - The ML section is rewritten with the summary table, both figures and the outcome of the headline rule.
  - It notes that the new 5-fold numbers replace the old single 80/20 split.
  - The "hard to learn" limitation is updated with the result.
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

- **Folds:** deterministic, disjoint, covering all 6,850 SMILES. Two feature tables in different row order map to the same folds.
- **Real-data facts:** the checks listed under "Data facts" pass on the committed data.
- **No leakage through preprocessing:** a zero-variance filter or scaler fitted on a training split never sees the test rows. The test uses a column that is constant only in the training part.
- **Adapters:** each adapter fits and predicts on a small synthetic set. TabPFN, TabICL, Chemprop and GP use `pytest.importorskip`.
- **Chemprop splits:** the training set Chemprop sees equals our fold's training indices.
- **Tuner:** stops at its time cap, leaves the refit reserve, and records the trials that finished.
- **End to end:** `run.py` works with Random Forest on synthetic data, and a forced error writes `status: "failed"`.
- **Metrics:** top-1% recall and tail RMSE match hand-computed examples, including a fold where fewer than one molecule falls in the top 1% (the count is rounded up to at least one).
- **Statistics:** the corrected t-test and the headline rule match hand-computed examples.
- **Partial reruns:** merging replaces only the rerun cells, and mixing package versions within a cell is refused.

## Risks

- **CPU time** for TabPFN-3 and TabICL on Mordred and for Chemprop tuning. The timing task and the reduction rule handle this, and the README reports any reduction.
- **Licence:** TabPFN-3 weights are non-commercial. The project is research, so this is acceptable, and the README states it.
- **Few folds:** with 5 outer folds, the corrected t-test is conservative, so small gaps between models will not reach significance. The README reports effect sizes alongside p-values.
- **Small tail sample:** each test fold has about 14 top-1% molecules, so recall moves in steps of about 0.07. It is reported with its fold-to-fold spread and is never used alone for the headline.
- **Noise ceiling unmeasured:** the data has no repeated calculations, so the xTB noise level, and with it the true R² ceiling, cannot be estimated here. The README says so.
