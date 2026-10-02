# tests/ml/test_workflow_plan.py
from ml_models.benchmark import plan_matrix


def test_default_benchmark_matrix_has_115_jobs():
    jobs = plan_matrix.plan("benchmark", "mean_ridge rf xgb gp tabpfn tabicl outlier chemprop",
                            "dft_descriptors dft_chemberta2 dft_mordred", "0 1 2 3 4")
    assert len(jobs) == (7 * 3 + 2) * 5      # 7 tabular job types (incl. the outlier ranker) x 3 reps + 2 Chemprop reps
    assert {"model": "chemprop", "rep": "smiles+dft_descriptors", "fold": 0} in jobs


def test_timing_mode_runs_one_job_per_model_and_rep():
    jobs = plan_matrix.plan("timing", "tabpfn tabicl", "dft_mordred", "0 1 2 3 4")
    assert jobs == [{"model": "tabpfn", "rep": "dft_mordred", "fold": -1},
                    {"model": "tabicl", "rep": "dft_mordred", "fold": -1}]


def test_bad_names_are_rejected():
    import pytest
    with pytest.raises(SystemExit):
        plan_matrix.plan("benchmark", "rf; rm -rf /", "dft_descriptors", "0")
