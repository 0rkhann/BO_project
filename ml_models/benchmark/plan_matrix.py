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
