#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.pipelines.vanilla_pipeline import VanillaPipeline
from src.pipelines.pls_pipeline import PLSPipeline
from src.pipelines.fabo_pipeline import FABOPipeline
from src.pipelines.pca_pipeline import PCAPipeline
from src.pipelines.opls_pipeline import OPLSPipeline
from baselines.random_search import RandomSearchPipeline
from src.simulation import EnergySimulator


def parse_list(arg_str, type_fn=str):
    return [type_fn(x) for x in arg_str.split(",") if x]


def main():
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization with various kernels, acquisitions, and seeds"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="CSV with SMILES + descriptors"
    )
    parser.add_argument(
        "--output-dir", "-o", default="results", help="Base results directory"
    )
    parser.add_argument(
        "--cache", "-c", required=True, help="Path to energy cache JSON"
    )
    parser.add_argument(
        "--n-initial", type=int, required=True, help="Number of initial random samples"
    )
    parser.add_argument(
        "--n-iter", type=int, required=True, help="Number of BO iterations per run"
    )
    parser.add_argument("--seed", type=int, required=True, help="Starting random seed")
    parser.add_argument(
        "--repeats", type=int, default=1, help="How many successive seeds to run"
    )
    parser.add_argument(
        "--mode",
        choices=["vanilla", "pls", "fabo", "pca", "opls", "random"],
        required=True,
        help="Which feature-selection mode to use",
    )
    parser.add_argument(
        "--kernels",
        "-k",
        type=str,
        default="Matern",
        help="Comma-separated GP kernels, e.g. Matern,RBF",
    )
    parser.add_argument(
        "--acquisitions",
        "-a",
        type=str,
        default="UCB",
        help="Comma-separated acquisition functions, e.g. UCB,EI",
    )
    parser.add_argument(
        "--betas",
        "-b",
        type=str,
        default="1.0",
        help="Comma-separated β values for UCB (ignored by EI)",
    )
    # FABO-specific parameters
    parser.add_argument(
        "--fabo-threshold",
        type=float,
        default=0.3,
        help="Spearman correlation cutoff for FABO (default: 0.3)",
    )
    parser.add_argument(
        "--fabo-k",
        type=int,
        default=None,
        help="Number of top features for FABO; if omitted, k will be chosen automatically",
    )
    parser.add_argument(
        "--pls-n-components",
        type=int,
        default=None,
        help="Fix the number of PLS components (skip CV tuning if set)",
    )
    parser.add_argument(
        "--pca-n-components",
        type=float,
        default=None,
        help="If set, number of PCA components (int) or fraction of variance (0–1) to keep.",
    )
    parser.add_argument(
        "--pca-whiten",
        action="store_true",
        help="Whiten PCA components (default: False)",
    )

    parser.add_argument(
        "--opls-n-components",
        type=int,
        default=1,
        help="Number of predictive components for OPLS (default: 1)",
    )
    parser.add_argument(
        "--opls-orthogonal",
        type=int,
        default=1,
        help="Number of orthogonal components for OPLS (default: 1)",
    )
    parser.add_argument(
        "--opls-scale",
        action="store_true",
        help="Scale X before OPLS (default: True)",
    )
    args = parser.parse_args()

    # parse lists
    kernels = parse_list(args.kernels, str)
    acquisitions = parse_list(args.acquisitions, str)
    betas = parse_list(args.betas, float)

    # choose pipeline class
    pipeline_map = {
        "vanilla": VanillaPipeline,
        "pls": PLSPipeline,
        "fabo": FABOPipeline,
        "pca": PCAPipeline,
        "opls": OPLSPipeline,
        "random": RandomSearchPipeline,
    }
    PipelineClass = pipeline_map[args.mode]

    # instantiate simulator once
    simulator = EnergySimulator()

    # Random search has a different interface (no kernel/acquisition loops)
    if args.mode == "random":
        for run_idx in range(args.repeats):
            seed = args.seed + run_idx
            config = {
                "n_initial": args.n_initial,
                "n_iter": args.n_iter,
                "random_state": seed,
                "results_dir": args.output_dir,
                "test_frac": 0.1,
            }
            print(f"=== RUN: mode=random, seed={seed} ===")
            pipeline = PipelineClass(config)
            pipeline.run(
                input_csv=args.input,
                cache_path=Path(args.cache),
                simulator=simulator,
            )
    else:
        # loop over all combinations for BO methods
        for kernel in kernels:
            for acq in acquisitions:
                for beta in betas:
                    for run_idx in range(args.repeats):
                        seed = args.seed + run_idx
                        # build config
                        config = {
                            "data": {
                                "n_initial": args.n_initial,
                                "test_frac": 0.1,
                            },
                            "optimization": {
                                "n_iter": args.n_iter,
                                "kernel": kernel,
                                "acquisition": acq,
                                "beta": beta,
                                "batch_size": 1,
                            },
                            "random_state": seed,
                            "results_dir": args.output_dir,
                            "feature_selection": {
                                "pls_components": args.pls_n_components,
                                "pca_components": args.pca_n_components,
                                "pca_whiten": args.pca_whiten,
                                "opls_components": args.opls_n_components,
                                "opls_orthogonal": args.opls_orthogonal,
                                "opls_scale": args.opls_scale,
                            },
                            "fabo_threshold": args.fabo_threshold,
                            "fabo_k": args.fabo_k,
                        }
                        print(
                            f"=== RUN: mode={args.mode}, kernel={kernel}, acq={acq}, beta={beta}, seed={seed} ==="
                        )
                        pipeline = PipelineClass(config)
                        pipeline.run(
                            input_csv=args.input,
                            cache_path=Path(args.cache),
                            simulator=simulator,
                        )


if __name__ == "__main__":
    main()
