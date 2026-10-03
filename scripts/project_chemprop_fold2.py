"""Expected result for the unfinished Chemprop SMILES+DFT-descriptor fold 2 (see ml_models/benchmark/BUDGET.md).

Every complete cell on DFT descriptors, plus Chemprop on SMILES, scores its fold 2 above or below its own
mean of the other four folds by a similar amount. That fold-2 offset, added to Chemprop's four-fold mean,
gives the expected fold-2 value; the smallest and largest offsets give the range. The corrected t-tests are
then rerun with the expected value. These are projections, not measurements. Run from the repository root:
python scripts/project_chemprop_fold2.py
"""
import json
from pathlib import Path

import numpy as np

from ml_models.benchmark import stats

ROOT = Path("ml_results/benchmark")
TARGET = ("chemprop", "smiles+dft_descriptors")
REFS = [("rf", "dft_descriptors"), ("tabpfn", "dft_descriptors"), ("chemprop", "smiles")]
OTHER = (0, 1, 3, 4)

cells = {}
for p in ROOT.glob("*/*/fold*/metrics.json"):
    m = json.loads(p.read_text())
    if m["status"] == "ok" and m["model"] not in ("mean", "outlier"):
        cells.setdefault((m["model"], m["rep"]), {})[m["fold"]] = m["metrics"]
assert 2 not in cells[TARGET] and len(cells[TARGET]) == 4, "fold 2 exists: report the measured value instead"
peers = [k for k, f in cells.items() if len(f) == 5 and (k[1] == "dft_descriptors" or k[0] == "chemprop")]

for metric in ("r2", "top1_recall", "spearman"):
    off = np.array([cells[k][2][metric] - np.mean([cells[k][f][metric] for f in OTHER]) for k in peers])
    base = np.mean([cells[TARGET][f][metric] for f in OTHER])
    print(f"\n{metric}: four-fold mean {base:.3f}; fold-2 offset {off.mean():+.3f} "
          f"(min {off.min():+.3f}, max {off.max():+.3f}, {len(peers)} cells)")
    for label, v in (("expected", base + off.mean()), ("low", base + off.min()), ("high", base + off.max())):
        a = [cells[TARGET][f][metric] if f != 2 else v for f in range(5)]
        tests = [stats.corrected_ttest(a, [cells[r][f][metric] for f in range(5)]) for r in REFS]
        p_holm = stats.holm([p for _, p in tests])
        vs = "; ".join(f"vs {m}/{r}: {d:+.3f} (Holm p {h:.3f})" for (m, r), (d, _), h in zip(REFS, tests, p_holm))
        print(f"  {label:8s} fold 2 {v:.3f}, 5-fold mean {np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}; {vs}")
