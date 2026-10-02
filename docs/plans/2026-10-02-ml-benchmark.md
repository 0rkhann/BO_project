# Tuned ML Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark eight model families under one nested, tuned protocol to test whether the hold-out R² of about 0.28 is a data ceiling, and connect the result to the BO story.

**Architecture:** A new `ml_models/benchmark/` package. A data layer builds fixed SMILES-keyed folds; model adapters share one `fit`/`predict`/`search_space` interface; one runner tunes with Optuna on inner splits, refits, and writes per-cell results. A summariser combines cells with provenance checks and corrected t-tests. A manual GitHub Actions matrix runs one job per (model, representation, fold).

**Tech Stack:** Python 3.11, scikit-learn, XGBoost, BoTorch/GPyTorch (the BO loop's GP), TabPFN 9.0.0 (TabPFN-3 weights), TabICL 2.2.0, Chemprop 2.3.1, Optuna 5.0.0, pytest, GitHub Actions.

**Spec:** `docs/specs/2026-10-02-ml-benchmark-design.md`. Read it before starting any task.

## Global Constraints

- Python 3.11 for everything in `requirements-ml.txt`. Pins: `chemprop==2.3.1`, `tabpfn==9.0.0`, `tabicl==2.2.0`, `optuna==5.0.0`; BoTorch and GPyTorch as resolved in `requirements-ml.lock` (resolved 2026-10-02: botorch 0.18.1, gpytorch 1.15.2, torch 2.14.1+cpu).
- `src/` and `results/` are not modified, except for importing `src.gp_model` and `src.data_io` from the GP adapter.
- Outer folds: `KFold(n_splits=5, shuffle=True, random_state=0)` over `data/dft_G.json` row order, stored by SMILES. Every feature table is aligned by SMILES, never by row position.
- Every data-dependent preprocessing step is fitted on the training part of the current split only.
- Targets are xTB binding free energies in kJ/mol from `data/dft_G.json`.
- TabPFN-3 checkpoint: `Prior-Labs/tabpfn_3`, file `tabpfn-v3-regressor-v3_default.ckpt`, revision `24a16a89d245878b846555110985634aa2e656d7`. TabICL checkpoint: `jingang/TabICL`, file `tabicl-regressor-v2-20260212.ckpt`, revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19`.
- Results layout: `ml_results/benchmark/<model>/<rep>/fold<k>/{metrics.json, predictions.csv, best_params.json, trials.csv}`.
- Commit messages, PR text, code comments and repository files must not name the tool that wrote them, and must not contain co-author or "generated with" lines.
- Every task is test-first: write the failing test, run it and see it fail, implement, run it and see it pass, then commit.
- Heavy-model tests use `pytest.importorskip`, so the BO test suite runs without `requirements-ml.txt`.

## Review Focus

These five inputs are implied by the spec but not covered by any task's main tests. Each line has its test added to the owning task.

1. **Every Optuna trial fails** (for example, a Chemprop learning rate that diverges to NaN). Expected: the tuner keeps going through all trials, and if none succeeds the cell writes `status: "failed"` with the last error, never a crash or a fake best. Test in Task 5.
2. **A job killed by the 340-minute timeout** leaves a cell folder with no `metrics.json`. Expected: `summarize.py` counts it as a missing fold and reports it, without crashing. Test in Task 9.
3. **Tied predictions** (the mean predictor predicts one value for every molecule) in top-1% recall. Expected: ties are broken at random with seed 0, so a constant predictor scores about the 5% chance level instead of 0 or 1. Test in Task 3.
4. **A checkpoint download fails** (network or Hugging Face outage). Expected: the cell writes `status: "failed"` with the download error, and the rest of the matrix is unaffected. Test in Task 7.
5. **A feature table no longer contains every SMILES in `folds.json`** (for example, after a regenerated export). Expected: loading stops with an error naming how many SMILES are missing, before any fitting. Test in Task 2.

---

## File Structure

| File | Responsibility |
|---|---|
| `requirements-ml.txt`, `requirements-ml.lock` | Pinned ML dependencies and their full resolution |
| `setup.py` | `ml` extra, corrected `python_requires` |
| `ml_models/__init__.py`, `ml_models/benchmark/__init__.py` | Package markers |
| `ml_models/benchmark/data.py` | Load target and representations, SMILES alignment, data checks, folds, data hashes |
| `ml_models/benchmark/metrics.py` | Fold metrics, including tail metrics |
| `ml_models/benchmark/stats.py` | Corrected t-test, Holm, headline rule |
| `ml_models/benchmark/preprocess.py` | In-fold zero-variance filter and optional scaler |
| `ml_models/benchmark/models/__init__.py` | Model registry |
| `ml_models/benchmark/models/{mean,ridge,rf,xgb,gp,tabpfn,tabicl,chemprop,outlier}.py` | One adapter each |
| `ml_models/benchmark/tune.py` | Optuna loop with trial, time and refit-reserve caps; grid mode |
| `ml_models/benchmark/run.py` | CLI for one cell: tune, refit, predict, write outputs, learning curve |
| `ml_models/benchmark/timing.py` | Timing spike: one inner fit per model and representation |
| `ml_models/benchmark/summarize.py` | Combine cells, provenance checks, summary, comparisons, OOF predictions, story-link table |
| `.github/workflows/tests.yml` | pytest on pull requests |
| `.github/workflows/ml-benchmark.yml` | Manual matrix for timing and benchmark runs |
| `plotting/make_figures.py` | Two new ML figures; parity plot replaced |
| `tests/ml/` | All new tests |
| `CHANGELOG.md`, `README.md` | History and the benchmark section |

---

### Task 1: Environment, packaging and CI

**Files:**
- Create: `requirements-ml.txt`, `requirements-ml.lock`, `ml_models/__init__.py`, `ml_models/benchmark/__init__.py`, `tests/ml/__init__.py`, `tests/ml/test_packaging.py`, `.github/workflows/tests.yml`
- Modify: `setup.py` (`python_requires`, `extras_require`), `.gitignore`, `README.md` (Python badge)

**Interfaces:**
- Produces: importable package `ml_models.benchmark`; the extra `pip install -e ".[ml]"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_packaging.py
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_package_imports():
    import ml_models.benchmark  # noqa: F401


def test_setup_declares_ml_extra_and_python_floor():
    tree = ast.parse((ROOT / "setup.py").read_text())
    src = (ROOT / "setup.py").read_text()
    assert 'python_requires=">=3.10"' in src
    assert '"ml"' in src and "requirements-ml.txt" in src


def test_ml_requirements_are_pinned():
    pins = dict(l.split("==") for l in (ROOT / "requirements-ml.txt").read_text().split()
                if "==" in l)
    assert pins == {"chemprop": "2.3.1", "tabpfn": "9.0.0", "tabicl": "2.2.0", "optuna": "5.0.0"}
```

- [ ] **Step 2: Run it and see it fail**

Run: `python -m pytest tests/ml/test_packaging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_models.benchmark'`.

- [ ] **Step 3: Implement**

`requirements-ml.txt`:
```
-r requirements.txt
chemprop==2.3.1
tabpfn==9.0.0
tabicl==2.2.0
optuna==5.0.0
huggingface_hub
```

Create the lock file with the CPU PyTorch index:
```bash
uv pip compile --python-version 3.11 --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  requirements-ml.txt -o requirements-ml.lock
```

`ml_models/__init__.py` and `ml_models/benchmark/__init__.py`: one docstring line each, for example `"""Supervised ML benchmark for the xTB binding free energies."""`.

`setup.py`: change `python_requires=">=3.8"` to `python_requires=">=3.10"`, and add to `extras_require`:
```python
        "ml": [l.strip() for l in open("requirements-ml.txt") if "==" in l],
```

`.gitignore`, append:
```
# Model checkpoints from the ML benchmark
*.ckpt
lightning_logs/
chemprop_runs/
```

`README.md`: change the Python badge from `python-3.8%2B` to `python-3.10%2B`.

`.github/workflows/tests.yml`:
```yaml
name: Tests

on:
  pull_request:

permissions:
  contents: read

jobs:
  pytest:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 1
          lfs: true # real-data checks read data/*.csv
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install
        run: |
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt
      - name: Test
        run: python -m pytest -q
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `python -m pytest -q`
Expected: all existing tests and the 3 new ones pass. Also run `uvx --from actionlint-py actionlint .github/workflows/tests.yml` and expect no output.

- [ ] **Step 5: Commit**

```bash
git add requirements-ml.txt requirements-ml.lock setup.py .gitignore README.md ml_models/__init__.py ml_models/benchmark/__init__.py tests/ml .github/workflows/tests.yml
git commit -m "Add the ML benchmark package skeleton, pinned dependencies and a pytest workflow"
```

---

### Task 2: Data layer, folds and data hashes

**Files:**
- Create: `ml_models/benchmark/data.py`, `tests/ml/test_data.py`

**Interfaces:**
- Produces:
  - `REPS: dict[str, str]` mapping rep names to CSV files: `{"dft_descriptors": "dft_descriptors.csv", "dft_chemberta2": "dft_chemberta2.csv", "dft_mordred": "dft_mordred.csv"}`
  - `load_target(path=DATA/"dft_G.json") -> pd.Series` (index SMILES, float energies)
  - `load_rep(rep: str, smiles: list[str]) -> np.ndarray` (rows in the order of `smiles`, raises `ValueError` if any SMILES is missing)
  - `make_folds(smiles: list[str], k=5, seed=0) -> list[list[str]]` (test SMILES per fold)
  - `save_folds(folds, path)`, `load_folds(path) -> dict` with keys `folds`, `data_hashes`
  - `sha256(path) -> str`
  - `split(folds: list[list[str]], k: int, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]` (train and test row indices into `smiles`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_data.py
import json
import numpy as np
import pandas as pd
import pytest

from ml_models.benchmark import data


def test_folds_are_deterministic_disjoint_and_complete():
    smi = [f"S{i}" for i in range(103)]
    a, b = data.make_folds(smi), data.make_folds(smi)
    assert a == b
    flat = [s for f in a for s in f]
    assert sorted(flat) == sorted(smi) and len(flat) == len(set(flat))
    assert len(a) == 5


def test_split_maps_by_smiles_not_row_order():
    smi = [f"S{i}" for i in range(50)]
    folds = data.make_folds(smi)
    shuffled = list(reversed(smi))
    tr, te = data.split(folds, 0, shuffled)
    assert {shuffled[i] for i in te} == set(folds[0])
    assert set(tr).isdisjoint(te)


def test_load_rep_fails_on_missing_smiles(tmp_path, monkeypatch):
    pd.DataFrame({"SMILES": ["A", "B"], "f": [1.0, 2.0]}).to_csv(tmp_path / "x.csv", index=False)
    monkeypatch.setattr(data, "DATA", tmp_path)
    monkeypatch.setitem(data.REPS, "toy", "x.csv")
    with pytest.raises(ValueError, match="1 SMILES"):
        data.load_rep("toy", ["A", "B", "C"])


def test_real_data_facts():
    y = data.load_target()
    assert len(y) == 6850 and y.index.is_unique and not y.isna().any()
    for rep in data.REPS:
        X = data.load_rep(rep, list(y.index))
        assert X.shape[0] == 6850 and np.isfinite(X).all()


def test_all_smiles_parse_in_rdkit():
    Chem = pytest.importorskip("rdkit.Chem")
    bad = [s for s in data.load_target().index if Chem.MolFromSmiles(s) is None]
    assert bad == []


def test_folds_file_records_data_hashes(tmp_path):
    folds = data.make_folds([f"S{i}" for i in range(20)])
    data.save_folds(folds, tmp_path / "folds.json")
    saved = data.load_folds(tmp_path / "folds.json")
    assert saved["folds"] == folds
    assert set(saved["data_hashes"]) == {"dft_G.json", *data.REPS.values()}
```

- [ ] **Step 2: Run them and see them fail**

Run: `python -m pytest tests/ml/test_data.py -v`
Expected: FAIL with `ImportError: cannot import name 'data'`.

- [ ] **Step 3: Implement**

```python
# ml_models/benchmark/data.py
"""Targets, representations and the shared outer folds, all keyed by SMILES."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPS = {"dft_descriptors": "dft_descriptors.csv", "dft_chemberta2": "dft_chemberta2.csv",
        "dft_mordred": "dft_mordred.csv"}


def load_target(path: Path | None = None) -> pd.Series:
    rows = json.loads((path or DATA / "dft_G.json").read_text())
    y = pd.Series({r["SMILES"]: float(r["energy"]) for r in rows})
    if not y.index.is_unique or y.isna().any():
        raise ValueError("dft_G.json has duplicate SMILES or missing energies")
    return y


def load_rep(rep: str, smiles: list[str]) -> np.ndarray:
    df = pd.read_csv(DATA / REPS[rep]).set_index("SMILES")
    missing = [s for s in smiles if s not in df.index]
    if missing:
        raise ValueError(f"{rep}: {len(missing)} SMILES from the folds are missing from {REPS[rep]}")
    X = df.loc[smiles].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError(f"{rep}: non-finite feature values")
    return X


def make_folds(smiles: list[str], k: int = 5, seed: int = 0) -> list[list[str]]:
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [[smiles[i] for i in test] for _, test in kf.split(smiles)]


def split(folds: list[list[str]], k: int, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    test = set(folds[k])
    is_test = np.array([s in test for s in smiles])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_hashes() -> dict[str, str]:
    files = ["dft_G.json", *REPS.values()]
    return {f: sha256(DATA / f) for f in files}


def save_folds(folds: list[list[str]], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"folds": folds, "data_hashes": data_hashes()}, indent=1))


def load_folds(path: Path) -> dict:
    return json.loads(Path(path).read_text())
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `python -m pytest tests/ml/test_data.py -v`
Expected: 6 passed (the RDKit check skips where RDKit is not installed). The real-data tests need the Git LFS files; in CI, `lfs: true` provides them.

- [ ] **Step 5: Commit**

```bash
git add ml_models/benchmark/data.py tests/ml/test_data.py
git commit -m "Add SMILES-keyed data loading, shared outer folds and data hashes"
```

---

### Task 3: Metrics and statistics

**Files:**
- Create: `ml_models/benchmark/metrics.py`, `ml_models/benchmark/stats.py`, `tests/ml/test_metrics.py`, `tests/ml/test_stats.py`

**Interfaces:**
- Produces:
  - `metrics.fold_metrics(y_true, y_pred, ranking_only=False) -> dict` with keys `r2, rmse, mae, spearman, top1_recall, tail_rmse` (only `spearman, top1_recall` when `ranking_only`)
  - `metrics.top1_recall(y_true, y_pred, seed=0) -> float`
  - `stats.corrected_ttest(a: np.ndarray, b: np.ndarray, test_train_ratio=0.25) -> tuple[float, float]` returns `(mean_diff, p_value)`
  - `stats.holm(pvals: list[float]) -> list[float]`
  - `stats.ceiling_moved(delta_r2: float, p_holm: float) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_metrics.py
import numpy as np
from ml_models.benchmark import metrics


def test_top1_recall_perfect_and_worst():
    y = np.arange(200, dtype=float)          # lowest 2 molecules are the top 1%
    assert metrics.top1_recall(y, y) == 1.0
    assert metrics.top1_recall(y, -y) == 0.0


def test_top1_recall_counts_at_least_one_molecule():
    y = np.arange(30, dtype=float)           # 1% of 30 rounds up to 1
    assert metrics.top1_recall(y, y) == 1.0


def test_constant_prediction_scores_near_chance():
    rng = np.random.default_rng(1)
    vals = [metrics.top1_recall(rng.normal(size=1400), np.zeros(1400), seed=s) for s in range(40)]
    assert 0.0 < np.mean(vals) < 0.15        # chance level is 5%


def test_tail_rmse_uses_lowest_decile():
    y = np.arange(100, dtype=float)
    p = y.copy(); p[:10] += 2.0              # error only in the lowest 10
    assert metrics.fold_metrics(y, p)["tail_rmse"] == 2.0


def test_ranking_only_keys():
    y = np.arange(100, dtype=float)
    assert set(metrics.fold_metrics(y, y, ranking_only=True)) == {"spearman", "top1_recall"}
```

```python
# tests/ml/test_stats.py
import numpy as np
from scipy import stats as st
from ml_models.benchmark import stats


def test_corrected_ttest_matches_hand_computation():
    a = np.array([0.30, 0.28, 0.31, 0.27, 0.29]); b = np.array([0.25, 0.26, 0.24, 0.27, 0.23])
    d = a - b
    t = d.mean() / np.sqrt((1 / 5 + 0.25) * d.var(ddof=1))
    expected_p = 2 * st.t.sf(abs(t), df=4)
    diff, p = stats.corrected_ttest(a, b)
    assert np.isclose(diff, d.mean()) and np.isclose(p, expected_p)


def test_holm():
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_ceiling_rule():
    assert stats.ceiling_moved(0.06, 0.01)
    assert not stats.ceiling_moved(0.04, 0.001)
    assert not stats.ceiling_moved(0.10, 0.2)
```

- [ ] **Step 2: Run them and see them fail**

Run: `python -m pytest tests/ml/test_metrics.py tests/ml/test_stats.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# ml_models/benchmark/metrics.py
"""Per-fold metrics. Lower energy is better, so the 'top' molecules have the lowest values."""
import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def top1_recall(y_true, y_pred, seed: int = 0) -> float:
    """Share of the lowest-1% true energies that are among the lowest 5% predictions.
    Ties in the predictions are broken at random with a fixed seed."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    n = len(y_true)
    top = set(np.argsort(y_true, kind="stable")[: max(1, math.ceil(0.01 * n))])
    jitter = np.random.default_rng(seed).random(n)
    order = np.lexsort((jitter, y_pred))           # sort by prediction, then by jitter
    picked = set(order[: max(1, math.ceil(0.05 * n))])
    return len(top & picked) / len(top)


def fold_metrics(y_true, y_pred, ranking_only: bool = False) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {"spearman": float(spearmanr(y_true, y_pred)[0]), "top1_recall": top1_recall(y_true, y_pred)}
    if ranking_only:
        return out
    tail = np.argsort(y_true, kind="stable")[: max(1, math.ceil(0.10 * len(y_true)))]
    out.update(r2=float(r2_score(y_true, y_pred)),
               rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
               mae=float(mean_absolute_error(y_true, y_pred)),
               tail_rmse=float(np.sqrt(np.mean((y_true[tail] - y_pred[tail]) ** 2))))
    return out
```

```python
# ml_models/benchmark/stats.py
"""Nadeau-Bengio corrected resampled t-test, Holm correction and the headline rule."""
import numpy as np
from scipy import stats as st

CEILING_DELTA_R2 = 0.05
ALPHA = 0.05


def corrected_ttest(a, b, test_train_ratio: float = 0.25) -> tuple[float, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float(d.mean()), (1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / np.sqrt((1 / k + test_train_ratio) * var)
    return float(d.mean()), float(2 * st.t.sf(abs(t), df=k - 1))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def ceiling_moved(delta_r2: float, p_holm: float) -> bool:
    return delta_r2 >= CEILING_DELTA_R2 and p_holm < ALPHA
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `python -m pytest tests/ml/test_metrics.py tests/ml/test_stats.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add ml_models/benchmark/metrics.py ml_models/benchmark/stats.py tests/ml/test_metrics.py tests/ml/test_stats.py
git commit -m "Add fold metrics with tail recall, the corrected t-test and the headline rule"
```

---

### Task 4: In-fold preprocessing and the fast adapters

**Files:**
- Create: `ml_models/benchmark/preprocess.py`, `ml_models/benchmark/models/__init__.py`, `ml_models/benchmark/models/{mean,ridge,rf,xgb}.py`, `tests/ml/test_adapters_fast.py`

**Interfaces:**
- Produces:
  - `preprocess.InFold(scale: bool)` with `.fit(X_tr) -> self`, `.transform(X) -> np.ndarray`; drops columns with zero variance in `X_tr`, then optionally standardises with statistics from `X_tr`.
  - Adapter protocol, one class per model with class attributes and methods:
    ```python
    class Adapter:
        name: str                      # registry key, e.g. "rf"
        tunable: bool                  # False -> fit once with default params
        grid: list[dict] | None        # set -> grid search instead of TPE
        def search_space(self, trial) -> dict: ...
        def fit(self, X, y, params: dict, smiles: list[str] | None = None) -> "Adapter": ...
        def predict(self, X, smiles: list[str] | None = None) -> np.ndarray: ...
    ```
  - `models.REGISTRY: dict[str, type]` and `models.get(name) -> Adapter instance`. Heavy adapters are imported lazily inside `get`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_adapters_fast.py
import numpy as np
import optuna
import pytest

from ml_models.benchmark import models
from ml_models.benchmark.preprocess import InFold


def toy(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)); X[:, 5] = 3.0          # a constant column
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.1, n)
    return X, y


def test_infold_uses_training_statistics_only():
    X_tr = np.c_[np.ones(10), np.arange(10.0)]          # column 0 constant in train only
    X_te = np.c_[np.arange(5.0) + 100, np.arange(5.0)]
    pre = InFold(scale=True).fit(X_tr)
    out = pre.transform(X_te)
    assert out.shape == (5, 1)                          # column 0 dropped by the training rule
    assert np.allclose(out[:, 0], (np.arange(5.0) - 4.5) / np.arange(10.0).std())


@pytest.mark.parametrize("name", ["mean", "ridge", "rf", "xgb"])
def test_fast_adapters_fit_and_predict(name):
    X, y = toy()
    a = models.get(name)
    params = a.search_space(optuna.trial.FixedTrial(a.defaults())) if a.tunable else {}
    pred = a.fit(X[:100], y[:100], params).predict(X[100:])
    assert pred.shape == (20,) and np.isfinite(pred).all()
    if name != "mean":
        assert np.corrcoef(pred, y[100:])[0, 1] > 0.8


def test_mean_predictor_predicts_training_mean():
    X, y = toy()
    pred = models.get("mean").fit(X[:100], y[:100], {}).predict(X[100:])
    assert np.allclose(pred, y[:100].mean())
```

- [ ] **Step 2: Run them and see them fail**

Run: `python -m pytest tests/ml/test_adapters_fast.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# ml_models/benchmark/preprocess.py
"""Preprocessing fitted on the training part of a split only."""
import numpy as np


class InFold:
    def __init__(self, scale: bool):
        self.scale = scale

    def fit(self, X_tr: np.ndarray) -> "InFold":
        sd = X_tr.std(axis=0)
        self.keep = sd > 0
        self.mu = X_tr[:, self.keep].mean(axis=0)
        self.sd = sd[self.keep]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X[:, self.keep]
        return (X - self.mu) / self.sd if self.scale else X
```

```python
# ml_models/benchmark/models/__init__.py
"""Model registry. Heavy adapters are imported only when requested."""
import importlib

REGISTRY = {"mean": "mean.Mean", "ridge": "ridge.Ridge", "rf": "rf.RF", "xgb": "xgb.XGB",
            "gp": "gp.GP", "tabpfn": "tabpfn.TabPFN", "tabicl": "tabicl.TabICL",
            "chemprop": "chemprop.Chemprop", "outlier": "outlier.OutlierRank"}


def get(name: str):
    module, cls = REGISTRY[name].split(".")
    return getattr(importlib.import_module(f"{__name__}.{module}"), cls)()
```

```python
# ml_models/benchmark/models/mean.py
import numpy as np


class Mean:
    name, tunable, grid = "mean", False, None

    def defaults(self): return {}
    def search_space(self, trial): return {}

    def fit(self, X, y, params, smiles=None):
        self.mu = float(np.mean(y)); return self

    def predict(self, X, smiles=None):
        return np.full(len(X), self.mu)
```

```python
# ml_models/benchmark/models/ridge.py
from sklearn.linear_model import Ridge as _Ridge

from ml_models.benchmark.preprocess import InFold


class Ridge:
    name, tunable, grid = "ridge", True, None

    def defaults(self): return {"alpha": 1.0}

    def search_space(self, trial):
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}

    def fit(self, X, y, params, smiles=None):
        self.pre = InFold(scale=True).fit(X)
        self.m = _Ridge(alpha=params["alpha"]).fit(self.pre.transform(X), y)
        return self

    def predict(self, X, smiles=None):
        return self.m.predict(self.pre.transform(X))
```

```python
# ml_models/benchmark/models/rf.py
import os

from sklearn.ensemble import RandomForestRegressor

from ml_models.benchmark.preprocess import InFold


class RF:
    name, tunable, grid = "rf", True, None

    def defaults(self):
        return {"n_estimators": 300, "max_features": 0.5, "min_samples_leaf": 1, "max_depth": 0}

    def search_space(self, trial):
        return {"n_estimators": trial.suggest_int("n_estimators", 300, 1000),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_depth": trial.suggest_categorical("max_depth", [0, 10, 20, 30, 40])}  # 0 = None

    def fit(self, X, y, params, smiles=None):
        p = dict(params); p["max_depth"] = p["max_depth"] or None
        self.pre = InFold(scale=False).fit(X)
        self.m = RandomForestRegressor(**p, n_jobs=os.cpu_count(), random_state=0)
        self.m.fit(self.pre.transform(X), y)
        return self

    def predict(self, X, smiles=None):
        return self.m.predict(self.pre.transform(X))
```

```python
# ml_models/benchmark/models/xgb.py
import os

from xgboost import XGBRegressor

from ml_models.benchmark.preprocess import InFold


class XGB:
    name, tunable, grid = "xgb", True, None

    def defaults(self):
        return {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 1e-3}

    def search_space(self, trial):
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True)}

    def fit(self, X, y, params, smiles=None):
        self.pre = InFold(scale=False).fit(X)
        self.m = XGBRegressor(**params, tree_method="hist", n_jobs=os.cpu_count(), random_state=0)
        self.m.fit(self.pre.transform(X), y)
        return self

    def predict(self, X, smiles=None):
        return self.m.predict(self.pre.transform(X))
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `python -m pytest tests/ml/test_adapters_fast.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add ml_models/benchmark/preprocess.py ml_models/benchmark/models tests/ml/test_adapters_fast.py
git commit -m "Add in-fold preprocessing and the mean, Ridge, Random Forest and XGBoost adapters"
```

---

### Task 5: Tuner and single-cell runner

**Files:**
- Create: `ml_models/benchmark/tune.py`, `ml_models/benchmark/run.py`, `tests/ml/test_tune.py`, `tests/ml/test_run.py`

**Interfaces:**
- Consumes: `data.*` (Task 2), `metrics.fold_metrics` (Task 3), `models.get` (Task 4).
- Produces:
  - `tune.tune(adapter, X, y, smiles, inner: str, n_trials: int, time_cap_s: float) -> tuple[dict, pd.DataFrame]` where `inner` is `"cv3"` (3-fold) or `"holdout10"` (one 90/10 split). Returns best params and the trial log. Raises `tune.AllTrialsFailed` if no trial succeeds.
  - `run.run_cell(model: str, rep: str, fold: int, out_root: Path, n_trials: int, job_budget_s: float, learning_curve: bool = False, folds_path: Path = ...) -> dict` (the `metrics.json` content).
  - CLI: `python -m ml_models.benchmark.run --model rf --rep dft_descriptors --fold 0 [--n-trials 50] [--job-budget-min 340] [--learning-curve]`. For Chemprop, `--rep` is `smiles` or `smiles+dft_descriptors`.
  - `metrics.json` keys: `status` (`"ok"` or `"failed"`), `error`, `model`, `rep`, `fold`, `metrics`, `n_trials_done`, `wall_s`, `peak_rss_mb`, `versions`, `git_sha`, `data_hashes`, `checkpoint`, `run_id`, and `learning_curve` when requested.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_tune.py
import time
import numpy as np
import pytest

from ml_models.benchmark import models, tune


def toy(n=90):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4)); y = X[:, 0] + rng.normal(0, 0.1, n)
    return X, y, [f"S{i}" for i in range(n)]


def test_tune_respects_trial_cap():
    X, y, s = toy()
    best, log = tune.tune(models.get("ridge"), X, y, s, inner="cv3", n_trials=5, time_cap_s=60)
    assert len(log) == 5 and "alpha" in best


def test_tune_stops_at_time_cap():
    X, y, s = toy()
    t0 = time.time()
    _, log = tune.tune(models.get("rf"), X, y, s, inner="cv3", n_trials=10_000, time_cap_s=3)
    assert time.time() - t0 < 30 and 1 <= len(log) < 10_000


class Exploding:
    name, tunable, grid = "boom", True, None
    def defaults(self): return {"a": 1.0}
    def search_space(self, trial): return {"a": trial.suggest_float("a", 0, 1)}
    def fit(self, X, y, params, smiles=None): raise FloatingPointError("loss is NaN")
    def predict(self, X, smiles=None): raise AssertionError


def test_all_trials_failing_raises_with_last_error():           # Review Focus 1
    X, y, s = toy()
    with pytest.raises(tune.AllTrialsFailed, match="loss is NaN"):
        tune.tune(Exploding(), X, y, s, inner="cv3", n_trials=4, time_cap_s=60)


def test_grid_mode_tries_every_config():
    X, y, s = toy()
    a = models.get("ridge"); a.grid = [{"alpha": 0.1}, {"alpha": 10.0}]
    best, log = tune.tune(a, X, y, s, inner="cv3", n_trials=99, time_cap_s=60)
    assert len(log) == 2 and best["alpha"] in (0.1, 10.0)
```

```python
# tests/ml/test_run.py
import json
import numpy as np
import pandas as pd

from ml_models.benchmark import data, run


def fake_data(tmp_path, monkeypatch, n=150):
    rng = np.random.default_rng(0)
    smi = [f"C{'C' * (i % 7)}O{i}" for i in range(n)]
    X = rng.normal(size=(n, 5)); y = 3 * X[:, 0] + rng.normal(0, 0.2, n)
    (tmp_path / "dft_G.json").write_text(json.dumps([{"SMILES": s, "energy": float(v)} for s, v in zip(smi, y)]))
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(5)]); df.insert(0, "SMILES", smi)
    for f in data.REPS.values():
        df.to_csv(tmp_path / f, index=False)
    monkeypatch.setattr(data, "DATA", tmp_path)
    folds = tmp_path / "folds.json"
    data.save_folds(data.make_folds(smi), folds)
    return folds


def test_run_cell_writes_outputs(tmp_path, monkeypatch):
    folds = fake_data(tmp_path, monkeypatch)
    out = tmp_path / "out"
    m = run.run_cell("rf", "dft_descriptors", 0, out, n_trials=3, job_budget_s=120, folds_path=folds)
    cell = out / "rf" / "dft_descriptors" / "fold0"
    assert m["status"] == "ok" and m["metrics"]["r2"] > 0.5
    for f in ["metrics.json", "predictions.csv", "best_params.json", "trials.csv"]:
        assert (cell / f).exists()
    assert set(m["data_hashes"]) == set(data.load_folds(folds)["data_hashes"])


def test_run_cell_records_failure(tmp_path, monkeypatch):
    folds = fake_data(tmp_path, monkeypatch)
    monkeypatch.setattr(run, "_fit_and_score", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    m = run.run_cell("rf", "dft_descriptors", 0, tmp_path / "out", n_trials=2, job_budget_s=60, folds_path=folds)
    saved = json.loads((tmp_path / "out/rf/dft_descriptors/fold0/metrics.json").read_text())
    assert m["status"] == saved["status"] == "failed" and "boom" in saved["error"]


def test_learning_curve_uses_nested_training_subsets(tmp_path, monkeypatch):
    folds = fake_data(tmp_path, monkeypatch)
    m = run.run_cell("ridge", "dft_descriptors", 1, tmp_path / "out", n_trials=2, job_budget_s=60,
                     learning_curve=True, folds_path=folds)
    lc = m["learning_curve"]
    assert [p["fraction"] for p in lc] == [0.1, 0.25, 0.5, 1.0]
    assert lc[-1]["n_train"] == 120 and all(p["n_train"] < 120 for p in lc[:-1])
```

- [ ] **Step 2: Run them and see them fail**

Run: `python -m pytest tests/ml/test_tune.py tests/ml/test_run.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# ml_models/benchmark/tune.py
"""Optuna tuning on inner splits of the outer training part."""
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)


class AllTrialsFailed(RuntimeError):
    pass


def _inner_splits(n: int, inner: str):
    idx = np.arange(n)
    if inner == "cv3":
        return list(KFold(3, shuffle=True, random_state=0).split(idx))
    if inner == "holdout10":
        tr, va = train_test_split(idx, test_size=0.1, random_state=0)
        return [(tr, va)]
    raise ValueError(f"unknown inner scheme {inner!r}")


def tune(adapter, X, y, smiles, inner: str, n_trials: int, time_cap_s: float):
    splits = _inner_splits(len(y), inner)
    smiles = np.asarray(smiles, dtype=object)
    last_error: list[str] = []

    def objective(trial):
        params = adapter.search_space(trial)
        rmses = []
        try:
            for tr, va in splits:
                m = type(adapter)()
                if adapter.grid is not None:
                    m.grid = adapter.grid
                pred = m.fit(X[tr], y[tr], params, smiles=list(smiles[tr])).predict(X[va], smiles=list(smiles[va]))
                rmses.append(float(np.sqrt(np.mean((y[va] - pred) ** 2))))
        except Exception as exc:                      # a failed trial must not stop the study
            last_error[:] = [f"{type(exc).__name__}: {exc}"]
            raise optuna.TrialPruned(last_error[0])
        return float(np.mean(rmses))

    if adapter.grid is not None:
        keys = sorted({k for g in adapter.grid for k in g})
        sampler = optuna.samplers.GridSampler({k: sorted({g[k] for g in adapter.grid}) for k in keys}, seed=0)
        n_trials = len(adapter.grid)
    else:
        sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=time_cap_s, catch=())
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    log = study.trials_dataframe()
    if not done:
        raise AllTrialsFailed(last_error[0] if last_error else "no trial completed within the time cap")
    return dict(study.best_trial.params), log
```

`GridSampler` works with the adapters' own `search_space`: it returns the grid value for each suggested parameter, as long as the value lies inside the suggested range. That is why the grid test can reuse `Ridge.search_space` unchanged.

```python
# ml_models/benchmark/run.py
"""Run one benchmark cell: tune on the outer training part, refit, score the outer test fold."""
import argparse
import importlib.metadata as md
import json
import os
import resource
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ml_models.benchmark import data, metrics, models, tune

ROOT = data.ROOT
FOLDS = ROOT / "ml_results" / "benchmark" / "folds.json"
OUT = ROOT / "ml_results" / "benchmark"
TRIALS = {"ridge": 30, "rf": 50, "xgb": 50, "chemprop": 20}
INNER = {"chemprop": "holdout10"}
LC_FRACTIONS = [0.1, 0.25, 0.5, 1.0]
PACKAGES = ["numpy", "scikit-learn", "xgboost", "torch", "botorch", "gpytorch", "tabpfn", "tabicl",
            "chemprop", "optuna"]


def _versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except OSError:
        return ""


def _features(rep: str, smiles: list[str]) -> np.ndarray:
    if rep == "smiles":
        return np.zeros((len(smiles), 0))
    if rep == "smiles+dft_descriptors":
        return data.load_rep("dft_descriptors", smiles)
    return data.load_rep(rep, smiles)


def _fit_and_score(adapter, params, X, y, smiles, tr, te):
    s = list(np.asarray(smiles, dtype=object)[tr]), list(np.asarray(smiles, dtype=object)[te])
    pred = adapter.fit(X[tr], y[tr], params, smiles=s[0]).predict(X[te], smiles=s[1])
    return pred


def run_cell(model, rep, fold, out_root=OUT, n_trials=None, job_budget_s=340 * 60,
             learning_curve=False, folds_path=FOLDS):
    t0 = time.time()
    cell = Path(out_root) / model / rep / f"fold{fold}"
    cell.mkdir(parents=True, exist_ok=True)
    saved = data.load_folds(folds_path)
    result = {"model": model, "rep": rep, "fold": fold, "status": "ok", "error": None,
              "versions": _versions(), "git_sha": _git_sha(), "data_hashes": saved["data_hashes"],
              "run_id": os.environ.get("GITHUB_RUN_ID", ""), "checkpoint": None, "n_trials_done": 0}
    try:
        y_all = data.load_target()
        smiles = list(y_all.index)
        y = y_all.to_numpy()
        X = _features(rep, smiles)
        tr, te = data.split(saved["folds"], fold, smiles)
        adapter = models.get(model)
        result["checkpoint"] = getattr(adapter, "checkpoint", None)
        params = {}
        if adapter.tunable:
            refit_reserve = getattr(adapter, "refit_estimate_s", 60) * 2
            cap = max(60.0, job_budget_s - (time.time() - t0) - refit_reserve)
            params, log = tune.tune(adapter, X[tr], y[tr], list(np.asarray(smiles, dtype=object)[tr]),
                                    INNER.get(model, "cv3"), n_trials or TRIALS.get(model, 50), cap)
            log.to_csv(cell / "trials.csv", index=False)
            result["n_trials_done"] = int((log["state"] == "COMPLETE").sum())
        else:
            pd.DataFrame().to_csv(cell / "trials.csv", index=False)
        (cell / "best_params.json").write_text(json.dumps(params, indent=1))
        pred = _fit_and_score(adapter, params, X, y, smiles, tr, te)
        pd.DataFrame({"SMILES": np.asarray(smiles, dtype=object)[te], "true": y[te], "pred": pred}).to_csv(
            cell / "predictions.csv", index=False)
        result["metrics"] = metrics.fold_metrics(y[te], pred, ranking_only=(model == "outlier"))
        if learning_curve:
            rng = np.random.default_rng(0)
            order = rng.permutation(tr)                 # nested subsets: prefixes of one permutation
            result["learning_curve"] = []
            for f in LC_FRACTIONS:
                sub = order[: max(2, int(round(f * len(tr))))]
                p = _fit_and_score(models.get(model), params, X, y, smiles, sub, te)
                result["learning_curve"].append({"fraction": f, "n_train": int(len(sub)),
                                                 **metrics.fold_metrics(y[te], p)})
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}")
    result["wall_s"] = round(time.time() - t0, 1)
    result["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    (cell / "metrics.json").write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(models.REGISTRY))
    ap.add_argument("--rep", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--job-budget-min", type=float, default=340)
    ap.add_argument("--learning-curve", action="store_true")
    a = ap.parse_args()
    if not FOLDS.exists():
        y = data.load_target()
        data.save_folds(data.make_folds(list(y.index)), FOLDS)
    r = run_cell(a.model, a.rep, a.fold, n_trials=a.n_trials, job_budget_s=a.job_budget_min * 60,
                 learning_curve=a.learning_curve)
    print(json.dumps({k: r[k] for k in ("model", "rep", "fold", "status", "wall_s")}))
    raise SystemExit(0 if r["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
```

Notes for the implementer:
- In the failure test, the monkeypatched `_fit_and_score` raises on the final refit. The tuner calls adapters directly, so tuning still succeeds and the failure is recorded at refit, which is what the test checks.
- `folds.json` is committed once, by Task 12, before the full run. `main()` creates it only when it is missing, so every job reads the same file.

- [ ] **Step 4: Run the tests and see them pass**

Run: `python -m pytest tests/ml/test_tune.py tests/ml/test_run.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add ml_models/benchmark/tune.py ml_models/benchmark/run.py tests/ml/test_tune.py tests/ml/test_run.py
git commit -m "Add the Optuna tuner and the single-cell benchmark runner"
```

---

### Task 6: GP adapter (the BO surrogate)

**Files:**
- Create: `ml_models/benchmark/models/gp.py`, `tests/ml/test_adapter_gp.py`

**Interfaces:**
- Consumes: `src.gp_model.build_gp(train_x, train_y, kernel_name="matern")`, `src.gp_model.fit_gp(model, ...) -> dict`, `src.data_io.prune_columns(X: pd.DataFrame, threshold: float) -> list[str]`, `preprocess.InFold`.
- Produces: adapter `gp` (`tunable = False`). On inputs with more than 600 columns, it prunes at 0.95 using the training part only, matching vanilla BO on Mordred.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_adapter_gp.py
import numpy as np
import pytest

pytest.importorskip("botorch")
from ml_models.benchmark import models


def test_gp_fits_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(80, 3)); y = np.sin(4 * X[:, 0]) + 0.05 * rng.normal(size=80)
    pred = models.get("gp").fit(X[:60], y[:60], {}).predict(X[60:])
    assert pred.shape == (20,) and np.corrcoef(pred, y[60:])[0, 1] > 0.8


def test_gp_prunes_wide_inputs_on_training_rows_only():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 5))
    X = np.c_[base, base[:, [0]] * 2.0, rng.normal(size=(50, 700))]   # column 5 duplicates column 0
    gp = models.get("gp").fit(X[:40], base[:40, 0], {})
    assert 5 not in gp.kept_columns and len(gp.kept_columns) < X.shape[1]
```

- [ ] **Step 2: Run it and see it fail**

Run: `python -m pytest tests/ml/test_adapter_gp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_models.benchmark.models.gp'`.

- [ ] **Step 3: Implement**

```python
# ml_models/benchmark/models/gp.py
"""Exact GP with the BO loop's own kernel, priors and fitting code."""
import numpy as np
import pandas as pd
import torch

from ml_models.benchmark.preprocess import InFold
from src.data_io import prune_columns
from src.gp_model import build_gp, fit_gp

WIDE = 600          # vanilla BO prunes Mordred; the 384-dim ChemBERTa-2 table is left unpruned there
PRUNE_AT = 0.95


class GP:
    name, tunable, grid = "gp", False, None
    refit_estimate_s = 1800

    def defaults(self): return {}
    def search_space(self, trial): return {}

    def fit(self, X, y, params, smiles=None):
        self.kept_columns = list(range(X.shape[1]))
        if X.shape[1] > WIDE:
            df = pd.DataFrame(X, columns=range(X.shape[1]))
            self.kept_columns = prune_columns(df, PRUNE_AT)
        X = X[:, self.kept_columns]
        self.pre = InFold(scale=True).fit(X)
        Xs = torch.tensor(self.pre.transform(X), dtype=torch.double)
        self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y))
        ys = torch.tensor((y - self.y_mu) / self.y_sd, dtype=torch.double).unsqueeze(-1)
        self.model = build_gp(Xs, ys, kernel_name="matern")
        fit_gp(self.model)
        return self

    def predict(self, X, smiles=None):
        Xs = torch.tensor(self.pre.transform(X[:, self.kept_columns]), dtype=torch.double)
        self.model.eval()
        with torch.no_grad():
            mean = torch.cat([self.model.posterior(c).mean.squeeze(-1) for c in Xs.split(512)])
        return mean.numpy() * self.y_sd + self.y_mu
```

Check `build_gp`'s expected input range before finishing. The BO loop feeds MinMax-scaled inputs in [0, 1]. If its lengthscale prior assumes that, use `sklearn.preprocessing.MinMaxScaler` fitted on the training part instead of standardisation, and say which in the commit message.

- [ ] **Step 4: Run the test and see it pass**

Run: `python -m pytest tests/ml/test_adapter_gp.py -v`
Expected: 2 passed (or skipped without BoTorch installed).

- [ ] **Step 5: Commit**

```bash
git add ml_models/benchmark/models/gp.py tests/ml/test_adapter_gp.py
git commit -m "Add the exact GP adapter that reuses the BO loop's surrogate"
```

---

### Task 7: Foundation-model adapters and the timing spike

**Files:**
- Create: `ml_models/benchmark/models/tabpfn.py`, `ml_models/benchmark/models/tabicl.py`, `ml_models/benchmark/weights.py`, `ml_models/benchmark/timing.py`, `tests/ml/test_adapters_foundation.py`

**Interfaces:**
- Produces:
  - `weights.fetch(repo: str, filename: str, revision: str) -> Path`, wrapping `huggingface_hub.hf_hub_download` with the cache directory `os.environ.get("BENCHMARK_WEIGHTS", ".cache/weights")`.
  - Adapters `tabpfn` and `tabicl` with `grid` set, `checkpoint` attribute `"<repo>/<file>@<revision>"`.
  - `python -m ml_models.benchmark.timing --model tabpfn --rep dft_mordred` prints JSON `{"model","rep","fit_predict_s","n_train"}` for one inner fit at the inner-CV training size (two thirds of an outer training part, 3,653 rows).

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_adapters_foundation.py
import numpy as np
import pytest

from ml_models.benchmark import models, weights


def test_download_failure_is_reported(monkeypatch):            # Review Focus 4
    def offline(**kw):
        raise OSError("connection reset")
    monkeypatch.setattr(weights, "_download", offline)
    with pytest.raises(OSError, match="connection reset"):
        weights.fetch("Prior-Labs/tabpfn_3", "x.ckpt", "abc")


@pytest.mark.parametrize("name", ["tabpfn", "tabicl"])
def test_foundation_adapter_fit_predict(name):
    pytest.importorskip(name)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5)); y = X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=120)
    a = models.get(name)
    params = a.grid[0]
    pred = a.fit(X[:100], y[:100], params).predict(X[100:])
    assert pred.shape == (20,) and np.corrcoef(pred, y[100:])[0, 1] > 0.8
    assert "@" in a.checkpoint
```

- [ ] **Step 2: Run them and see them fail**

Run: `python -m pytest tests/ml/test_adapters_foundation.py -v`
Expected: FAIL with `ImportError: cannot import name 'weights'`.

- [ ] **Step 3: Implement**

```python
# ml_models/benchmark/weights.py
"""Pinned model checkpoints from Hugging Face."""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download as _download


def fetch(repo: str, filename: str, revision: str) -> Path:
    cache = os.environ.get("BENCHMARK_WEIGHTS", ".cache/weights")
    return Path(_download(repo_id=repo, filename=filename, revision=revision, cache_dir=cache))
```

```python
# ml_models/benchmark/models/tabpfn.py
"""TabPFN-3 (weights under a non-commercial licence; research use only)."""
import os

import torch

from ml_models.benchmark import weights
from ml_models.benchmark.preprocess import InFold

REPO, FILE, REV = "Prior-Labs/tabpfn_3", "tabpfn-v3-regressor-v3_default.ckpt", "24a16a89d245878b846555110985634aa2e656d7"
GRID = [{"n_estimators": n, "softmax_temperature": t} for n in (8, 16) for t in (0.75, 0.9)]


class TabPFN:
    name, tunable = "tabpfn", True
    checkpoint = f"{REPO}/{FILE}@{REV}"
    refit_estimate_s = 900

    def __init__(self):
        self.grid = list(GRID)

    def defaults(self): return dict(self.grid[0])

    def search_space(self, trial):
        return {"n_estimators": trial.suggest_categorical("n_estimators", sorted({g["n_estimators"] for g in self.grid})),
                "softmax_temperature": trial.suggest_categorical(
                    "softmax_temperature", sorted({g["softmax_temperature"] for g in self.grid}))}

    def fit(self, X, y, params, smiles=None):
        from tabpfn import TabPFNRegressor
        torch.set_num_threads(os.cpu_count())
        self.pre = InFold(scale=False).fit(X)
        self.m = TabPFNRegressor(model_path=str(weights.fetch(REPO, FILE, REV)), device="cpu",
                                 ignore_pretraining_limits=True, random_state=0, **params)
        self.m.fit(self.pre.transform(X), y)
        return self

    def predict(self, X, smiles=None):
        return self.m.predict(self.pre.transform(X))
```

```python
# ml_models/benchmark/models/tabicl.py
"""TabICLv2 (BSD-3-Clause)."""
import os

import torch

from ml_models.benchmark import weights
from ml_models.benchmark.preprocess import InFold

REPO, FILE, REV = "jingang/TabICL", "tabicl-regressor-v2-20260212.ckpt", "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"


class TabICL:
    name, tunable = "tabicl", True
    checkpoint = f"{REPO}/{FILE}@{REV}"
    refit_estimate_s = 900

    def __init__(self):
        self.grid = [{"n_estimators": n} for n in (8, 16, 32)]

    def defaults(self): return dict(self.grid[0])

    def search_space(self, trial):
        return {"n_estimators": trial.suggest_categorical("n_estimators", [g["n_estimators"] for g in self.grid])}

    def fit(self, X, y, params, smiles=None):
        from tabicl import TabICLRegressor
        torch.set_num_threads(os.cpu_count())
        self.pre = InFold(scale=False).fit(X)
        self.m = TabICLRegressor(model_path=str(weights.fetch(REPO, FILE, REV)), allow_auto_download=False,
                                 device="cpu", **params)
        self.m.fit(self.pre.transform(X), y)
        return self

    def predict(self, X, smiles=None):
        return self.m.predict(self.pre.transform(X))
```

If `TabICLRegressor` or `TabPFNRegressor` 9.0.0 rejects a keyword used here, check its signature (`inspect.signature`), adjust the call, and keep the pinned checkpoint path. Do not let either package download its own default weights.

```python
# ml_models/benchmark/timing.py
"""Time one inner fit+predict per model and representation, to size the grids."""
import argparse
import json
import time

import numpy as np

from ml_models.benchmark import data, models
from ml_models.benchmark.run import _features

INNER_TRAIN = 3653           # two thirds of an outer training part (5,480 rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rep", required=True)
    a = ap.parse_args()
    y_all = data.load_target(); smiles = list(y_all.index); y = y_all.to_numpy()
    X = _features(a.rep, smiles)
    idx = np.random.default_rng(0).permutation(len(y))
    tr, va = idx[:INNER_TRAIN], idx[INNER_TRAIN:INNER_TRAIN + 1827]
    m = models.get(a.model)
    t0 = time.time()
    m.fit(X[tr], y[tr], m.defaults(), smiles=[smiles[i] for i in tr]).predict(X[va], smiles=[smiles[i] for i in va])
    print(json.dumps({"model": a.model, "rep": a.rep, "fit_predict_s": round(time.time() - t0, 1),
                      "n_train": INNER_TRAIN}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `python -m pytest tests/ml/test_adapters_foundation.py -v`
Expected: the download test passes; the adapter tests pass with `requirements-ml.txt` installed (first run downloads both checkpoints), otherwise skip.

- [ ] **Step 5: Commit**

```bash
git add ml_models/benchmark/weights.py ml_models/benchmark/models/tabpfn.py ml_models/benchmark/models/tabicl.py ml_models/benchmark/timing.py tests/ml/test_adapters_foundation.py
git commit -m "Add TabPFN-3 and TabICLv2 adapters with pinned checkpoints, and the timing spike"
```

---

### Task 8: Chemprop adapter

**Files:**
- Create: `ml_models/benchmark/models/chemprop.py`, `tests/ml/test_adapter_chemprop.py`

**Interfaces:**
- Consumes: Chemprop 2.3.1 Python API: `chemprop.data.MoleculeDatapoint.from_smi(smi, y, x_d=...)`, `data.MoleculeDataset(dps, featurizer)`, `dataset.normalize_targets(scaler=None)`, `dataset.normalize_inputs("X_d", scaler=None)`, `data.build_dataloader(dataset, batch_size, shuffle)`, `nn.BondMessagePassing(d_h, depth, dropout)`, `nn.MeanAggregation()`, `nn.RegressionFFN(input_dim, hidden_dim, n_layers, dropout, output_transform)`, `nn.UnscaleTransform.from_standard_scaler`, `nn.ScaleTransform.from_standard_scaler`, `models.MPNN(mp, agg, ffn, batch_norm, X_d_transform, max_lr, init_lr, final_lr)`, `lightning.pytorch.Trainer`.
- Produces: adapter `chemprop`. `X` has 0 columns for `rep="smiles"` or 29 columns for `rep="smiles+dft_descriptors"`. `smiles` is required in `fit` and `predict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_adapter_chemprop.py
import numpy as np
import optuna
import pytest

pytest.importorskip("chemprop")
from ml_models.benchmark import models

SMI = ["CCO", "CCCO", "CCCCO", "CCN", "CCCN", "CCCCN", "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCOC",
       "CCCOC", "CC(C)O", "CC(C)N", "OCCO", "NCCN", "CC=O", "CCC=O", "CC(=O)O", "CCC(=O)O", "COC"] * 3


def small_params():
    return {"depth": 2, "message_hidden_dim": 64, "ffn_num_layers": 1, "ffn_hidden_dim": 64,
            "dropout": 0.0, "max_lr": 1e-3, "epochs": 15}


@pytest.mark.parametrize("n_xd", [0, 3])
def test_chemprop_fits_with_and_without_descriptors(n_xd):
    rng = np.random.default_rng(0)
    y = np.array([len(s) for s in SMI], dtype=float) + 0.1 * rng.normal(size=len(SMI))
    X = rng.normal(size=(len(SMI), n_xd))
    m = models.get("chemprop").fit(X[:50], y[:50], small_params(), smiles=SMI[:50])
    pred = m.predict(X[50:], smiles=SMI[50:])
    assert pred.shape == (10,) and np.isfinite(pred).all()


def test_chemprop_trains_only_on_given_rows():
    m = models.get("chemprop").fit(np.zeros((40, 0)), np.arange(40.0), small_params(), smiles=SMI[:40])
    assert m.n_train_seen + m.n_val_seen == 40      # its own split is never used


def test_search_space_keys():
    a = models.get("chemprop")
    p = a.search_space(optuna.trial.FixedTrial(a.defaults()))
    assert {"depth", "message_hidden_dim", "ffn_num_layers", "ffn_hidden_dim", "dropout", "max_lr"} <= set(p)
```

- [ ] **Step 2: Run them and see them fail**

Run: `python -m pytest tests/ml/test_adapter_chemprop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml_models.benchmark.models.chemprop'`.

- [ ] **Step 3: Implement**

```python
# ml_models/benchmark/models/chemprop.py
"""Chemprop D-MPNN on SMILES, optionally with molecule-level descriptors (x_d)."""
import os

import numpy as np


class Chemprop:
    name, tunable, grid = "chemprop", True, None
    refit_estimate_s = 1200
    MAX_EPOCHS, PATIENCE = 100, 15

    def defaults(self):
        return {"depth": 3, "message_hidden_dim": 300, "ffn_num_layers": 1, "ffn_hidden_dim": 300,
                "dropout": 0.0, "max_lr": 1e-3}

    def search_space(self, trial):
        return {"depth": trial.suggest_int("depth", 2, 6),
                "message_hidden_dim": trial.suggest_int("message_hidden_dim", 300, 1200, step=100),
                "ffn_num_layers": trial.suggest_int("ffn_num_layers", 1, 3),
                "ffn_hidden_dim": trial.suggest_int("ffn_hidden_dim", 300, 1200, step=100),
                "dropout": trial.suggest_float("dropout", 0.0, 0.4),
                "max_lr": trial.suggest_float("max_lr", 1e-4, 3e-3, log=True)}

    def _dataset(self, smiles, y, X):
        from chemprop import data, featurizers
        x_d = X if X.shape[1] else None
        dps = [data.MoleculeDatapoint.from_smi(s, None if y is None else [float(v)],
                                               x_d=None if x_d is None else x_d[i])
               for i, (s, v) in enumerate(zip(smiles, y if y is not None else [0.0] * len(smiles)))]
        return data.MoleculeDataset(dps, featurizers.SimpleMoleculeMolGraphFeaturizer())

    def fit(self, X, y, params, smiles=None):
        import lightning.pytorch as pl
        import torch
        from chemprop import data, models, nn
        from lightning.pytorch.callbacks import EarlyStopping

        torch.manual_seed(0); torch.set_num_threads(os.cpu_count())
        idx = np.random.default_rng(0).permutation(len(y))
        n_val = max(1, int(round(0.1 * len(y))))
        va, tr = idx[:n_val], idx[n_val:]
        self.n_train_seen, self.n_val_seen = len(tr), len(va)
        smiles = list(smiles)
        train = self._dataset([smiles[i] for i in tr], y[tr], X[tr])
        val = self._dataset([smiles[i] for i in va], y[va], X[va])
        y_scaler = train.normalize_targets(); val.normalize_targets(y_scaler)
        xd_tf = None
        if X.shape[1]:
            xd_scaler = train.normalize_inputs("X_d"); val.normalize_inputs("X_d", xd_scaler)
            xd_tf = nn.ScaleTransform.from_standard_scaler(xd_scaler)
        mp = nn.BondMessagePassing(d_h=params["message_hidden_dim"], depth=params["depth"], dropout=params["dropout"])
        ffn = nn.RegressionFFN(input_dim=mp.output_dim + X.shape[1], hidden_dim=params["ffn_hidden_dim"],
                               n_layers=params["ffn_num_layers"], dropout=params["dropout"],
                               output_transform=nn.UnscaleTransform.from_standard_scaler(y_scaler))
        self.model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=True, X_d_transform=xd_tf,
                                 max_lr=params["max_lr"], init_lr=params["max_lr"] / 10,
                                 final_lr=params["max_lr"] / 10)
        self.trainer = pl.Trainer(max_epochs=params.get("epochs", self.MAX_EPOCHS), accelerator="cpu",
                                  logger=False, enable_checkpointing=False, enable_progress_bar=False,
                                  callbacks=[EarlyStopping("val_loss", patience=self.PATIENCE)],
                                  deterministic=False)
        self.trainer.fit(self.model, data.build_dataloader(train, shuffle=True, seed=0),
                         data.build_dataloader(val, shuffle=False))
        return self

    def predict(self, X, smiles=None):
        from chemprop import data
        ds = self._dataset(list(smiles), None, X)        # X_d is scaled by the model's X_d_transform
        out = self.trainer.predict(self.model, data.build_dataloader(ds, shuffle=False))
        return np.concatenate([o.numpy().ravel() for o in out])
```

Before finishing, check two details against the Chemprop 2.3.1 documentation (`chemprop.readthedocs.io`, "Training a regression model" and "Extra features and descriptors"):
- the keyword names in `build_dataloader` (`seed`, `shuffle`);
- that `X_d_transform` scales descriptors at prediction time, so test data must not be normalised again.

A test that fails because of an API name mismatch is fixed in the adapter, never by loosening the test.

- [ ] **Step 4: Run the tests and see them pass**

Run: `python -m pytest tests/ml/test_adapter_chemprop.py -v`
Expected: 4 passed (or skipped without Chemprop installed).

- [ ] **Step 5: Commit**

```bash
git add ml_models/benchmark/models/chemprop.py tests/ml/test_adapter_chemprop.py
git commit -m "Add the Chemprop adapter with fixed inner splits and optional DFT descriptors"
```

---

### Task 9: Summariser, provenance checks and partial-rerun merge

**Files:**
- Create: `ml_models/benchmark/summarize.py`, `ml_models/benchmark/models/outlier.py`, `tests/ml/test_summarize.py`, `tests/ml/test_outlier_ranker.py`

**Interfaces:**
- Consumes: the cell layout and `metrics.json` keys from Task 5; `stats.*` from Task 3.
- Produces:
  - `summarize.collect(root: Path) -> pd.DataFrame`, one row per cell folder, with `status` set to `"missing"` when `metrics.json` is absent.
  - `summarize.check_provenance(cells: pd.DataFrame) -> None`, which raises `ValueError` if one (model, rep) cell mixes `data_hashes` or package versions.
  - `summarize.summary(cells) -> pd.DataFrame`, written to `summary.csv`: mean ± sd of every metric, `folds_ok`, `cpu_minutes`.
  - `summarize.comparisons(cells) -> pd.DataFrame`, written to `comparisons.csv`, with the families from the spec and a `ceiling_moved` row.
  - `summarize.oof(root, model, rep) -> pd.DataFrame`, written to `oof_predictions/<model>_<rep>.csv`.
  - `summarize.story_table(cells, bo_summary: Path = ROOT/"analysis/summary.csv") -> pd.DataFrame`, written to `story_links.csv`.
  - `summarize.merge(new_root: Path, existing_root: Path) -> None`: copies new cell folders over existing ones, leaving other cells untouched.
  - Adapter `outlier` (`tunable = False`): ranks by max |z| with z from training statistics; predictions are negative scores, so the most extreme molecules rank as the "lowest energy".

- [ ] **Step 1: Write the failing tests**

```python
# tests/ml/test_summarize.py
import json
from pathlib import Path

import numpy as np
import pytest

from ml_models.benchmark import summarize


def write_cell(root, model, rep, fold, r2, hashes=None, version="1.0", status="ok"):
    d = Path(root) / model / rep / f"fold{fold}"; d.mkdir(parents=True)
    m = {"r2": r2, "rmse": 1 - r2, "mae": 1 - r2, "spearman": r2, "top1_recall": r2, "tail_rmse": 1 - r2}
    (d / "metrics.json").write_text(json.dumps({
        "model": model, "rep": rep, "fold": fold, "status": status, "metrics": m, "wall_s": 60,
        "data_hashes": hashes or {"dft_G.json": "a"}, "versions": {"scikit-learn": version}}))
    return d


def test_missing_metrics_file_counts_as_missing_fold(tmp_path):      # Review Focus 2
    for k in range(4):
        write_cell(tmp_path, "rf", "dft_descriptors", k, 0.3)
    (tmp_path / "rf" / "dft_descriptors" / "fold4").mkdir()            # killed job, no metrics.json
    cells = summarize.collect(tmp_path)
    s = summarize.summary(cells)
    row = s[(s.model == "rf") & (s.rep == "dft_descriptors")].iloc[0]
    assert row.folds_ok == 4 and (cells.status == "missing").sum() == 1


def test_mixed_data_hashes_are_refused(tmp_path):
    write_cell(tmp_path, "rf", "dft_descriptors", 0, 0.3, hashes={"dft_G.json": "a"})
    write_cell(tmp_path, "rf", "dft_descriptors", 1, 0.3, hashes={"dft_G.json": "b"})
    with pytest.raises(ValueError, match="data"):
        summarize.check_provenance(summarize.collect(tmp_path))


def test_mixed_versions_are_refused(tmp_path):
    write_cell(tmp_path, "rf", "dft_descriptors", 0, 0.3, version="1.0")
    write_cell(tmp_path, "rf", "dft_descriptors", 1, 0.3, version="2.0")
    with pytest.raises(ValueError, match="version"):
        summarize.check_provenance(summarize.collect(tmp_path))


def test_comparisons_and_ceiling_rule(tmp_path):
    for k in range(5):
        write_cell(tmp_path, "rf", "dft_descriptors", k, 0.28 + 0.001 * k)
        write_cell(tmp_path, "tabpfn", "dft_descriptors", k, 0.40 + 0.002 * k)
    c = summarize.comparisons(summarize.collect(tmp_path))
    vs_rf = c[(c.family == "vs_rf") & (c.model == "tabpfn")].iloc[0]
    assert vs_rf.delta_r2 > 0.1
    assert bool(c[c.family == "ceiling"].iloc[0].ceiling_moved)


def test_merge_replaces_only_rerun_cells(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    write_cell(old, "rf", "dft_descriptors", 0, 0.1); write_cell(old, "xgb", "dft_descriptors", 0, 0.2)
    write_cell(new, "rf", "dft_descriptors", 0, 0.3)
    summarize.merge(new, old)
    cells = summarize.collect(old).set_index("model")
    assert cells.loc["rf", "r2"] == 0.3 and cells.loc["xgb", "r2"] == 0.2
```

```python
# tests/ml/test_outlier_ranker.py
import numpy as np
from ml_models.benchmark import models


def test_outlier_z_uses_training_statistics_only():
    X_tr = np.c_[np.zeros(10) + np.arange(10) * 0.1]
    X_te = np.array([[0.45], [100.0]])                 # the extreme test value must not shift z
    m = models.get("outlier").fit(X_tr, np.zeros(10), {})
    p = m.predict(X_te)
    assert p[1] < p[0]                                 # more extreme -> ranked as lower energy
    z_expected = abs(100.0 - X_tr.mean()) / X_tr.std()
    assert np.isclose(-p[1], z_expected)
```

- [ ] **Step 2: Run them and see them fail**

Run: `python -m pytest tests/ml/test_summarize.py tests/ml/test_outlier_ranker.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# ml_models/benchmark/models/outlier.py
"""The BO 'outlier' control as a ranker: no training, max |z| with training-part statistics."""
import numpy as np


class OutlierRank:
    name, tunable, grid = "outlier", False, None

    def defaults(self): return {}
    def search_space(self, trial): return {}

    def fit(self, X, y, params, smiles=None):
        sd = X.std(axis=0)
        self.keep = sd > 0
        self.mu, self.sd = X[:, self.keep].mean(axis=0), sd[self.keep]
        return self

    def predict(self, X, smiles=None):
        z = np.abs((X[:, self.keep] - self.mu) / self.sd).max(axis=1)
        return -z                                       # most extreme = predicted lowest energy
```

```python
# ml_models/benchmark/summarize.py
"""Combine benchmark cells into summary tables, with provenance checks."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ml_models.benchmark import stats

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "ml_results" / "benchmark"
METRICS = ["r2", "rmse", "mae", "spearman", "top1_recall", "tail_rmse"]
TABPFN_NOTICE = ("Predictions in this folder were produced with TabPFN-3 weights, released under the TabPFN-3\n"
                 "non-commercial licence (https://huggingface.co/Prior-Labs/tabpfn_3). They may only be used for\n"
                 "non-commercial purposes.\n")


def collect(root: Path = BENCH) -> pd.DataFrame:
    rows = []
    for cell in sorted(Path(root).glob("*/*/fold*")):
        model, rep, fold = cell.parts[-3], cell.parts[-2], int(cell.name[4:])
        f = cell / "metrics.json"
        if not f.exists():
            rows.append({"model": model, "rep": rep, "fold": fold, "status": "missing"})
            continue
        m = json.loads(f.read_text())
        rows.append({"model": model, "rep": rep, "fold": fold, "status": m["status"],
                     "wall_s": m.get("wall_s", np.nan), "data_hashes": json.dumps(m.get("data_hashes"), sort_keys=True),
                     "versions": json.dumps(m.get("versions"), sort_keys=True), **(m.get("metrics") or {})})
    return pd.DataFrame(rows)


def check_provenance(cells: pd.DataFrame) -> None:
    ok = cells[cells.status == "ok"]
    for (model, rep), g in ok.groupby(["model", "rep"]):
        if g.data_hashes.nunique() > 1:
            raise ValueError(f"{model}/{rep}: folds were computed on different data files")
        if g.versions.nunique() > 1:
            raise ValueError(f"{model}/{rep}: folds were computed with different package versions")


def summary(cells: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, rep), g in cells.groupby(["model", "rep"]):
        ok = g[g.status == "ok"]
        row = {"model": model, "rep": rep, "folds_ok": len(ok), "folds_total": len(g),
               "cpu_minutes": round(ok.wall_s.sum() / 60, 1) if "wall_s" in ok else np.nan}
        for m in METRICS:
            if m in ok and ok[m].notna().any():
                row[f"{m}_mean"], row[f"{m}_sd"] = ok[m].mean(), ok[m].std(ddof=1)
        out.append(row)
    return pd.DataFrame(out)


def _paired(cells, model_a, rep_a, model_b, rep_b, metric):
    a = cells[(cells.model == model_a) & (cells.rep == rep_a) & (cells.status == "ok")].set_index("fold")[metric]
    b = cells[(cells.model == model_b) & (cells.rep == rep_b) & (cells.status == "ok")].set_index("fold")[metric]
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return None
    return stats.corrected_ttest(a[common].to_numpy(), b[common].to_numpy())


def comparisons(cells: pd.DataFrame) -> pd.DataFrame:
    s = summary(cells)
    full = s[(s.folds_ok == 5) & s.get("r2_mean", pd.Series(dtype=float)).notna()]
    rows = []
    for rep, g in full.groupby("rep"):
        for model in g.model:
            if model == "rf":
                continue
            for metric in ("r2", "top1_recall"):
                r = _paired(cells, model, rep, "rf", rep, metric)
                if r:
                    rows.append({"family": "vs_rf", "rep": rep, "model": model, "ref": "rf", "metric": metric,
                                 f"delta_{metric}": r[0], "delta": r[0], "p": r[1]})
        best = g.sort_values("r2_mean").iloc[-1].model
        for model in g.model:
            if model != best:
                r = _paired(cells, best, rep, model, rep, "r2")
                if r:
                    rows.append({"family": "best_vs_others", "rep": rep, "model": best, "ref": model,
                                 "metric": "r2", "delta": r[0], "p": r[1]})
    for model, g in full.groupby("model"):
        if len(g) > 1:
            best_rep = g.sort_values("r2_mean").iloc[-1].rep
            for rep in g.rep:
                if rep != best_rep:
                    r = _paired(cells, model, best_rep, model, rep, "r2")
                    if r:
                        rows.append({"family": "across_reps", "rep": best_rep, "model": model, "ref": rep,
                                     "metric": "r2", "delta": r[0], "p": r[1]})
    c = pd.DataFrame(rows)
    if c.empty:
        return c
    c["p_holm"] = np.nan
    for fam, idx in c.groupby(["family", "metric"]).groups.items():
        c.loc[idx, "p_holm"] = stats.holm(c.loc[idx, "p"].tolist())
    vr = c[(c.family == "vs_rf") & (c.metric == "r2")]
    moved = bool(((vr.delta >= stats.CEILING_DELTA_R2) & (vr.p_holm < stats.ALPHA)).any())
    c["delta_r2"] = np.where(c.metric == "r2", c.delta, np.nan)
    return pd.concat([c, pd.DataFrame([{"family": "ceiling", "ceiling_moved": moved}])], ignore_index=True)


def oof(root: Path, model: str, rep: str) -> pd.DataFrame:
    parts = [pd.read_csv(f) for f in sorted(Path(root, model, rep).glob("fold*/predictions.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def story_table(cells: pd.DataFrame, bo_summary: Path = ROOT / "analysis" / "summary.csv") -> pd.DataFrame:
    s = summary(cells)
    bo = pd.read_csv(bo_summary)
    bo = bo[~bo.method.isin(["random", "outlier"])].groupby("rep").auc_median.min()
    rows = []
    for rep, g in s[s.rep.isin(bo.index)].groupby("rep"):
        trained = g[~g.model.isin(["outlier", "mean"])].dropna(subset=["top1_recall_mean"])
        best = trained.sort_values("top1_recall_mean").iloc[-1] if len(trained) else None
        gp = g[g.model == "gp"]; out = g[g.model == "outlier"]
        rows.append({"rep": rep, "best_model": None if best is None else best.model,
                     "best_top1_recall": None if best is None else best.top1_recall_mean,
                     "gp_top1_recall": gp.top1_recall_mean.iloc[0] if len(gp) else np.nan,
                     "outlier_top1_recall": out.top1_recall_mean.iloc[0] if len(out) else np.nan,
                     "bo_best_regret_auc": bo[rep]})
    return pd.DataFrame(rows)


def merge(new_root: Path, existing_root: Path) -> None:
    for cell in Path(new_root).glob("*/*/fold*"):
        dest = Path(existing_root) / cell.relative_to(new_root)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(cell, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=BENCH)
    a = ap.parse_args()
    cells = collect(a.root)
    check_provenance(cells)
    summary(cells).to_csv(a.root / "summary.csv", index=False)
    comparisons(cells).to_csv(a.root / "comparisons.csv", index=False)
    story_table(cells).to_csv(a.root / "story_links.csv", index=False)
    if (a.root / "tabpfn").is_dir():
        (a.root / "tabpfn" / "NOTICE").write_text(TABPFN_NOTICE)
    (a.root / "oof_predictions").mkdir(exist_ok=True)
    for (model, rep), _ in cells[cells.status == "ok"].groupby(["model", "rep"]):
        oof(a.root, model, rep).to_csv(a.root / "oof_predictions" / f"{model}_{rep}.csv", index=False)
    print(summary(cells).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `python -m pytest tests/ml/test_summarize.py tests/ml/test_outlier_ranker.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add ml_models/benchmark/summarize.py ml_models/benchmark/models/outlier.py tests/ml/test_summarize.py tests/ml/test_outlier_ranker.py
git commit -m "Add the benchmark summariser with provenance checks, rerun merge and the outlier ranker"
```

---

### Task 10: ML benchmark workflow

**Files:**
- Create: `.github/workflows/ml-benchmark.yml`, `tests/ml/test_workflow_plan.py`

**Interfaces:**
- Consumes: the CLIs from Tasks 5, 7 and 9.
- Produces: a manual workflow with inputs:
  - `mode`: `timing` or `benchmark`;
  - `models`, default `mean_ridge rf xgb gp tabpfn tabicl outlier chemprop`;
  - `reps`, default `dft_descriptors dft_chemberta2 dft_mordred`;
  - `folds`, default `0 1 2 3 4`;
  - `merge` (boolean, default `false`);
  - `learning_curve_models`, default empty.

  `mean_ridge` runs `mean` then `ridge` in one job. `chemprop` expands to the reps `smiles` and `smiles+dft_descriptors` only.

- [ ] **Step 1: Write the failing test**

The planner lives in a small Python file so it can be tested: `ml_models/benchmark/plan_matrix.py`.

```python
# tests/ml/test_workflow_plan.py
from ml_models.benchmark import plan_matrix


def test_default_benchmark_matrix_has_100_jobs():
    jobs = plan_matrix.plan("benchmark", "mean_ridge rf xgb gp tabpfn tabicl outlier chemprop",
                            "dft_descriptors dft_chemberta2 dft_mordred", "0 1 2 3 4")
    assert len(jobs) == (6 * 3 + 2) * 5      # 6 tabular job types x 3 reps + 2 Chemprop reps
    assert {"model": "chemprop", "rep": "smiles+dft_descriptors", "fold": 0} in jobs


def test_timing_mode_runs_one_job_per_model_and_rep():
    jobs = plan_matrix.plan("timing", "tabpfn tabicl", "dft_mordred", "0 1 2 3 4")
    assert jobs == [{"model": "tabpfn", "rep": "dft_mordred", "fold": -1},
                    {"model": "tabicl", "rep": "dft_mordred", "fold": -1}]


def test_bad_names_are_rejected():
    import pytest
    with pytest.raises(SystemExit):
        plan_matrix.plan("benchmark", "rf; rm -rf /", "dft_descriptors", "0")
```

- [ ] **Step 2: Run it and see it fail**

Run: `python -m pytest tests/ml/test_workflow_plan.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

```python
# ml_models/benchmark/plan_matrix.py
"""Turn workflow inputs into a validated job matrix."""
import json
import re
import sys

NAME = re.compile(r"^[a-z0-9_+]+$")
CHEMPROP_REPS = ["smiles", "smiles+dft_descriptors"]


def plan(mode: str, models: str, reps: str, folds: str) -> list[dict]:
    m, r, f = models.split(), reps.split(), folds.split()
    bad = [v for v in m + r if not NAME.match(v)] + [x for x in f if not x.isdigit()]
    if bad or mode not in ("timing", "benchmark"):
        raise SystemExit(f"invalid input: {bad or mode}")
    jobs = []
    for model in m:
        for rep in (CHEMPROP_REPS if model == "chemprop" else r):
            if mode == "timing":
                jobs.append({"model": model, "rep": rep, "fold": -1})
            else:
                jobs.extend({"model": model, "rep": rep, "fold": int(k)} for k in f)
    if len(jobs) > 256:
        raise SystemExit(f"{len(jobs)} jobs exceeds the 256-job matrix limit")
    return jobs


if __name__ == "__main__":
    print("matrix=" + json.dumps({"include": plan(*sys.argv[1:5])}))
```

`.github/workflows/ml-benchmark.yml`:
```yaml
name: ML benchmark

on:
  workflow_dispatch:
    inputs:
      mode:
        description: "timing (one inner fit per model and representation) or benchmark"
        default: "benchmark"
      models:
        description: "Space-separated models (mean_ridge runs mean and Ridge in one job)"
        default: "mean_ridge rf xgb gp tabpfn tabicl outlier chemprop"
      reps:
        description: "Space-separated tabular representations"
        default: "dft_descriptors dft_chemberta2 dft_mordred"
      folds:
        description: "Space-separated outer folds"
        default: "0 1 2 3 4"
      merge:
        description: "Merge into the existing ml_results/benchmark/ on main instead of replacing it"
        type: boolean
        default: false
      learning_curve_models:
        description: "Space-separated models that also run the learning curve"
        default: ""

permissions:
  contents: read

jobs:
  plan:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.plan.outputs.matrix }}
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 1
      - id: plan
        env:
          MODE: ${{ inputs.mode }}
          MODELS: ${{ inputs.models }}
          REPS: ${{ inputs.reps }}
          FOLDS: ${{ inputs.folds }}
        run: python3 ml_models/benchmark/plan_matrix.py "$MODE" "$MODELS" "$REPS" "$FOLDS" >> "$GITHUB_OUTPUT"

  run:
    needs: plan
    runs-on: ubuntu-latest
    timeout-minutes: 350
    strategy:
      fail-fast: false
      max-parallel: 20
      matrix: ${{ fromJSON(needs.plan.outputs.matrix) }}
    name: ${{ matrix.model }} / ${{ matrix.rep }} / fold ${{ matrix.fold }}
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 1
          lfs: true # data/*.csv are Git LFS files
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - uses: actions/cache@v4
        with:
          path: .cache/weights
          key: weights-tabpfn3-24a16a8-tabicl-4dcd344
      - name: Install
        run: pip install -r requirements-ml.lock
      - name: Run
        env:
          MODE: ${{ inputs.mode }}
          MODEL: ${{ matrix.model }}
          REP: ${{ matrix.rep }}
          FOLD: ${{ matrix.fold }}
          LC_MODELS: ${{ inputs.learning_curve_models }}
          BENCHMARK_WEIGHTS: .cache/weights
        run: |
          set -o pipefail
          # Only this job's output may end up in the artifact.
          rm -rf results ml_results/benchmark/*/
          mkdir -p logs
          if [ "$MODE" = timing ]; then
            for m in ${MODEL/mean_ridge/mean ridge}; do
              python -m ml_models.benchmark.timing --model "$m" --rep "$REP" | tee -a logs/timing.jsonl
            done
            exit 0
          fi
          status=0
          for m in ${MODEL/mean_ridge/mean ridge}; do
            LC=(); case " $LC_MODELS " in *" $m "*) LC=(--learning-curve);; esac
            python -m ml_models.benchmark.run --model "$m" --rep "$REP" --fold "$FOLD" "${LC[@]}" \
              2>&1 | tee "logs/${m}_${REP}_fold${FOLD}.log" || status=1
          done
          exit $status
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: ml-${{ strategy.job-index }}
          path: |
            ml_results/benchmark/
            logs/
          if-no-files-found: warn
          retention-days: 14

  collect:
    needs: run
    if: ${{ !cancelled() && inputs.mode == 'benchmark' }}
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      # Base the results commit on the latest main (see the BO rerun workflow for why).
      - uses: actions/checkout@v6
        with:
          ref: main
          fetch-depth: 1
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - uses: actions/download-artifact@v4
        with:
          pattern: ml-*
          path: new
          merge-multiple: true
      - name: Commit results to a new branch
        env:
          MERGE: ${{ inputs.merge }}
          BRANCH: experiment-results/ml-${{ github.run_id }}
        run: |
          pip install -q pandas numpy scipy scikit-learn
          if [ "$MERGE" != true ]; then
            git rm -rq --ignore-unmatch ml_results/benchmark
            mkdir -p ml_results/benchmark
            cp -r new/ml_results/benchmark/. ml_results/benchmark/
          else
            python -c "from ml_models.benchmark.summarize import merge; merge('new/ml_results/benchmark', 'ml_results/benchmark')"
          fi
          python -m ml_models.benchmark.summarize
          git switch -c "$BRANCH"
          git add ml_results/benchmark
          git -c user.name="github-actions[bot]" -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
            commit -qm "ML benchmark results (workflow run ${GITHUB_RUN_ID})"
          git push -q origin "$BRANCH"
          echo "Results: ${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/compare/main...${BRANCH}?expand=1" >> "$GITHUB_STEP_SUMMARY"
```

`folds.json` is committed to `main` by Task 12 before the first benchmark dispatch, and the run step keeps it: `rm -rf ml_results/benchmark/*/` deletes cell folders only.

- [ ] **Step 4: Run the tests and lint**

Run: `python -m pytest tests/ml/test_workflow_plan.py -v && uvx --from actionlint-py actionlint .github/workflows/ml-benchmark.yml`
Expected: 3 passed, and no actionlint output.

- [ ] **Step 5: Commit**

```bash
git add ml_models/benchmark/plan_matrix.py tests/ml/test_workflow_plan.py .github/workflows/ml-benchmark.yml
git commit -m "Add the manual ML benchmark workflow with timing and merge modes"
```

---

### Task 11: Tag the internship state

**Files:** none (a git tag).

- [ ] **Step 1: Check the target commit**

Run: `git show --no-patch --format='%h %ad %s' --date=short 0a6bf4a`
Expected: `0a6bf4a 2025-10-11 Clean up whitespace`.

- [ ] **Step 2: Create and push the tag**

```bash
git tag -a v0.1.0-internship 0a6bf4a -m "State of the repository at the end of the internship, before the pipeline fixes"
git push origin v0.1.0-internship
```

This step is run by the maintainer's session, not by the workflow agent, because pushing tags needs repository write access outside a branch.

---

### Task 12: Timing spike, grid sizing and the full benchmark run

This task runs the code from Tasks 1–10 and records decisions; it adds no new code.

- [ ] **Step 1: Commit the outer folds to `main`**

```bash
python -c "from ml_models.benchmark import data; y=data.load_target(); data.save_folds(data.make_folds(list(y.index)), data.ROOT/'ml_results/benchmark/folds.json')"
git add ml_results/benchmark/folds.json
git commit -m "Add the fixed outer folds for the ML benchmark"
```

- [ ] **Step 2: Run the timing spike**

Dispatch `ML benchmark` with `mode=timing`, `models="gp tabpfn tabicl chemprop"`, and the default reps. Read `logs/timing.jsonl` from the artifacts.

- [ ] **Step 3: Apply the reduction rule from the spec**

For each foundation model and representation:
- the grid cost is `n_configs × 3 inner folds × fit_predict_s`, plus a refit;
- if that exceeds 340 minutes minus a 20% margin, the grid becomes the default configuration only (no inner tuning);
- if even one fit exceeds the budget, the cell is listed as "not run (CPU budget)" and removed from the dispatch.

Record each decision in `ml_results/benchmark/README_budget.md` with the measured times, and set `refit_estimate_s` in the adapters from the measured values. Commit with the message "Size the foundation-model grids from the timing spike".

- [ ] **Step 4: Run the benchmark**

Dispatch `ML benchmark` with `mode=benchmark` and default inputs. Open a PR from the results branch.

- [ ] **Step 5: Rerun failed cells**

If any cell failed, rerun only those with `merge=true` and the matching `models`, `reps` and `folds`.

- [ ] **Step 6: Learning curves for the best model per representation**

From `summary.csv`, take the model with the highest `r2_mean` for each tabular representation, and for the Chemprop pair. For each one, dispatch `ML benchmark` with `merge=true`, `models` set to that model, `reps` set to that representation, and `learning_curve_models` set to the same model. This reruns those cells with the learning curve, so they reproduce the earlier tuned results and add the curve.

---

### Task 13: Figures, README, cleanup and CHANGELOG

**Files:**
- Modify: `plotting/make_figures.py` (two new figures, parity replaced), `README.md`, `scripts/analyze_results.py`, `scripts/summarize_results.py`
- Create: `CHANGELOG.md`, `tests/ml/test_figures_inputs.py`
- Delete: `ml_models/{train_models,random_forest,xgboost_model,base_model,plot_model_results}.py`, `ml_results/{dft,chemberta2,mordred}/`, `ml_plots/ml_model_summary.csv`, `train_all_ml_models.sh`, `scripts/recompute_holdout_metrics.py`

**Interfaces:**
- Consumes: `ml_results/benchmark/{summary.csv, comparisons.csv, story_links.csv, oof_predictions/}`.
- Produces: `ml_plots/ml_benchmark.svg` (+ `-dark.svg`, `.pdf`) and `ml_plots/ml_parity.svg` (+ variants), both from the best model by mean R².

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/test_figures_inputs.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_old_single_split_ml_code_is_gone():
    for p in ["ml_models/train_models.py", "ml_models/random_forest.py", "scripts/recompute_holdout_metrics.py",
              "train_all_ml_models.sh"]:
        assert not (ROOT / p).exists(), p


def test_tabpfn_outputs_carry_the_licence_notice():
    notice = (ROOT / "ml_results/benchmark/tabpfn/NOTICE").read_text()
    assert "non-commercial" in notice


def test_analysis_scripts_read_the_new_summary():
    for p in ["scripts/analyze_results.py", "scripts/summarize_results.py"]:
        src = (ROOT / p).read_text()
        assert "ml_results/benchmark" in src and "model_comparison.csv" not in src, p
```

- [ ] **Step 2: Run it and see it fail**

Run: `python -m pytest tests/ml/test_figures_inputs.py -v`
Expected: FAIL, because the old files still exist.

- [ ] **Step 3: Implement**

- **Delete** the files listed above.
- **NOTICE:** `summarize.py` writes `ml_results/benchmark/tabpfn/NOTICE` on every run (Task 9), so a rerun that replaces the folder keeps it. Check that it is present in the merged results.
- **Analysis scripts:** in `scripts/analyze_results.py` and `scripts/summarize_results.py`, replace every read of `ml_results/<ds>/*_predictions.csv` or `model_comparison.csv` with a read of `ml_results/benchmark/summary.csv`, keeping their printed tables in the same format, but with the new models and the mean ± sd columns.
- **Figures:** in `plotting/make_figures.py`, replace `fig_parity()` with two functions, drawn for both themes like the others:
  - `fig_ml_benchmark()`: three panels (R², Spearman, top-1% recall). Each is a dot plot of mean ± sd per model, with a colour per representation, a reference line at 0 for R² and at 0.05 for recall, and the outlier ranker shown only in the ranking panels.
  - `fig_ml_parity()`: out-of-fold predictions of the best model by mean R² on all 6,850 molecules, one panel per representation.
- **README ML section:**
  - the summary table (folds completed, mean ± sd);
  - both figures;
  - the outcome of the headline rule from `comparisons.csv`, in one sentence;
  - the story-links table from `story_links.csv`;
  - the learning-curve numbers;
  - the reductions from `README_budget.md`;
  - the TabPFN-3 licence note.
- **README claim recheck:** recheck every statement listed in the spec's "README claims to recheck" against the new numbers, and list in the PR description which were kept, reworded or removed.
- **README reproduction:** add the `ML benchmark` dispatch and a single-cell command: `python -m ml_models.benchmark.run --model rf --rep dft_descriptors --fold 0`.
- **`CHANGELOG.md`:**
  ```markdown
  # Changelog

  ## v1.0.0

  Numbers in the internship report correspond to `v0.1.0-internship`. Every BO and ML result changed in this release.

  ### Fixed (BO pipeline)
  - Feature selectors were fitted on scaled inputs but applied to raw inputs from the first iteration on.
  - FABO, PCA, PLS and OPLS were fitted once on the 10 initial molecules and never refitted.
  - "OPLS" ran as one-component PLS with no orthogonal component.
  - The first BO pick was wasted (`best_f = inf`) and the initial design was ignored when tracking the best value.
  - The MinMax scaler was fitted on the 10 initial molecules only.
  - FABO command-line options were ignored.
  - "Seeds at pool optimum" used one seed's optimum for all seeds.
  - Scoring the whole pool in one batch made memory grow with every iteration.

  ### Changed
  - BO results: 20 seeds, two control methods (vanilla GP, outlier heuristic) and paired statistics.
  - ML results: a nested 5-fold, tuned benchmark replaces the single 80/20 split.
  - Energies are labelled as xTB binding free energies (kJ/mol); only the 29 descriptors are DFT-level.
  ```
  Then link `CHANGELOG.md` from the README, under the credit line.

- [ ] **Step 4: Run all tests and regenerate the figures**

Run: `python -m pytest -q && python plotting/make_figures.py`
Expected: every test passes, and the figure script writes the new ML figures without errors. Check both figures at 900 px width on light and dark backgrounds, as with the BO figures.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Report the tuned ML benchmark in the README, replace the old ML code and add a changelog"
```

---

### Task 14: Release tag

- [ ] **Step 1:** Right after the Task 13 PR merges, and before anything else lands on `main`, tag it:

```bash
git fetch origin main
git tag -a v1.0.0 origin/main -m "Fixed BO pipeline, 20-seed results and the tuned ML benchmark"
git push origin v1.0.0
```

Like Task 11, this runs in the maintainer's session.

---

## Execution on GitHub

The repository's agent runs Tasks 1–10 and 13, one issue per task, each started with `/agent`. Each issue contains the task's section of this plan and the global constraints. The agent opens a PR per task. The maintainer's session:
1. reviews it;
2. reruns the tests locally;
3. merges with a clean squash message;
4. only then opens the next task.

Tasks 11, 12 and 14 (tags and workflow dispatches) run from the maintainer's session.
