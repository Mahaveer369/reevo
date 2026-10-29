#!/usr/bin/env python3
"""
Two-stage ACO hyperparameter tuning for ReEvo TSP.

Stage 1 (coarse):
    - all 108 configurations
    - frozen LLM heuristic only
    - small, deterministic validation subset
    - multiple seeds

Stage 2 (confirmation):
    - top-K configurations from stage 1
    - baseline ACO and LLM+ACO
    - larger validation subset
    - more seeds
    - same seeds for both methods

Final selection is based on LLM+ACO validation performance, with
baseline performance reported as a control.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import distance_matrix


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
TSP_ACO_DIR = REPO_ROOT / "problems" / "tsp_aco"
DATASET_DIR = TSP_ACO_DIR / "dataset"
RESULTS_DIR = BENCHMARK_DIR / "results"
HEURISTIC_PATH = (
    BENCHMARK_DIR / "heuristics" / "qwen_reevo_best_tsp.py"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(TSP_ACO_DIR) not in sys.path:
    sys.path.insert(0, str(TSP_ACO_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aco import ACO


# ---------------------------------------------------------------------
# Fixed experiment design
# ---------------------------------------------------------------------

ALPHAS = (0.5, 1.0, 2.0)
BETAS = (0.5, 1.0, 2.0)
DECAYS = (0.7, 0.8, 0.9, 0.95)
N_ANTS_VALUES = (20, 30, 50)

N_ITERATIONS = 100
VALIDATION_SIZES = (20, 50, 100)

DEFAULT_CONFIG = {
    "alpha": 1.0,
    "beta": 1.0,
    "decay": 0.9,
    "n_ants": 30,
    "n_iterations": N_ITERATIONS,
}

METHOD_LLM = "aco_llm"
METHOD_BASELINE = "aco_baseline"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_llm_heuristic():
    spec = importlib.util.spec_from_file_location(
        "qwen_reevo_best_tsp",
        HEURISTIC_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {HEURISTIC_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "heuristics_v2"):
        raise AttributeError(
            "Frozen heuristic must define heuristics_v2()."
        )
    return module.heuristics_v2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_instance_indices(
    n_instances: int,
    count: int,
    size: int,
    base_seed: int = 20261004,
) -> list[int]:
    """Deterministic pseudo-random subset; larger subsets contain smaller ones."""
    count = min(count, n_instances)
    if count <= 0:
        raise ValueError("count must be positive")

    rng = np.random.default_rng(base_seed + size)
    permutation = rng.permutation(n_instances)
    return sorted(permutation[:count].tolist())


def prepare_validation_distance(node_pos: np.ndarray) -> np.ndarray:
    dist = distance_matrix(node_pos, node_pos).astype(np.float64, copy=True)
    np.fill_diagonal(dist, 1.0)
    return dist


def safe_heuristic(heuristic_fn, dist: np.ndarray) -> np.ndarray:
    h = np.asarray(heuristic_fn(dist.copy()), dtype=np.float64)
    h = np.nan_to_num(h, nan=1e-9, posinf=1e9, neginf=1e-9)
    h = h + 1e-9
    h[h < 1e-9] = 1e-9
    return h


def run_one(
    dist: np.ndarray,
    heuristic_fn,
    seed: int,
    *,
    alpha: float,
    beta: float,
    decay: float,
    n_ants: int,
    n_iterations: int,
    device: str,
) -> tuple[float, float]:
    """
    Returns:
        cost, runtime_sec
    """
    set_seed(seed)

    heuristic = safe_heuristic(heuristic_fn, dist)

    aco = ACO(
        dist,
        heuristic,
        n_ants=n_ants,
        decay=decay,
        alpha=alpha,
        beta=beta,
        device=device,
    )

    start = time.perf_counter()
    objective = aco.run(n_iterations)
    elapsed = time.perf_counter() - start

    if isinstance(objective, torch.Tensor):
        cost = float(objective.item())
    else:
        cost = float(objective)

    return cost, elapsed


def make_grid() -> list[dict]:
    grid = []
    for alpha, beta, decay, n_ants in itertools.product(
        ALPHAS, BETAS, DECAYS, N_ANTS_VALUES
    ):
        grid.append(
            {
                "alpha": float(alpha),
                "beta": float(beta),
                "decay": float(decay),
                "n_ants": int(n_ants),
                "n_iterations": N_ITERATIONS,
            }
        )

    assert len(grid) == 108
    return grid


def config_key(config: dict) -> tuple[float, float, float, int]:
    return (
        float(config["alpha"]),
        float(config["beta"]),
        float(config["decay"]),
        int(config["n_ants"]),
    )


def config_label(config: dict) -> str:
    return (
        f"a={config['alpha']:g},"
        f"b={config['beta']:g},"
        f"d={config['decay']:g},"
        f"ants={config['n_ants']}"
    )


def load_validation_data(
    instances_per_size: int,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    data = {}

    for size in VALIDATION_SIZES:
        dataset_path = DATASET_DIR / f"val{size}_dataset.npy"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Missing validation dataset: {dataset_path}"
            )

        node_positions = np.load(dataset_path)
        selected = select_instance_indices(
            n_instances=node_positions.shape[0],
            count=instances_per_size,
            size=size,
        )

        data[size] = [
            (idx, prepare_validation_distance(node_positions[idx]))
            for idx in selected
        ]

        print(
            f"val{size}: {node_positions.shape[0]} total instances; "
            f"using indices {selected}"
        )

    return data


def is_default(config: dict) -> bool:
    return (
        float(config["alpha"]) == DEFAULT_CONFIG["alpha"]
        and float(config["beta"]) == DEFAULT_CONFIG["beta"]
        and float(config["decay"]) == DEFAULT_CONFIG["decay"]
        and int(config["n_ants"]) == DEFAULT_CONFIG["n_ants"]
        and int(config["n_iterations"]) == DEFAULT_CONFIG["n_iterations"]
    )


# ---------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------

def run_coarse(
    *,
    instances_per_size: int,
    seeds: list[int],
    device: str,
) -> None:
    llm_heuristic = load_llm_heuristic()
    validation = load_validation_data(instances_per_size)
    grid = make_grid()

    print(f"\nCoarse grid size: {len(grid)} configurations")
    print(f"Methods: {METHOD_LLM} only")
    print(f"Seeds: {seeds}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Device: {device}\n")

    rows = []
    total = len(grid)
    overall_start = time.perf_counter()

    for config_number, config in enumerate(grid, start=1):
        config_start = time.perf_counter()
        print(
            f"[{config_number:03d}/{total}] "
            f"{config_label(config)}",
            flush=True,
        )

        for size in VALIDATION_SIZES:
            for instance_idx, dist in validation[size]:
                for seed in seeds:
                    cost, runtime = run_one(
                        dist,
                        llm_heuristic,
                        seed,
                        alpha=config["alpha"],
                        beta=config["beta"],
                        decay=config["decay"],
                        n_ants=config["n_ants"],
                        n_iterations=N_ITERATIONS,
                        device=device,
                    )

                    rows.append(
                        {
                            "alpha": config["alpha"],
                            "beta": config["beta"],
                            "decay": config["decay"],
                            "n_ants": config["n_ants"],
                            "n_iterations": N_ITERATIONS,
                            "method": METHOD_LLM,
                            "size": size,
                            "instance_index": instance_idx,
                            "seed": seed,
                            "cost": cost,
                            "runtime_sec": runtime,
                        }
                    )

        print(
            f"    completed in {time.perf_counter() - config_start:.3f}s",
            flush=True,
        )

    detailed_path = RESULTS_DIR / "aco_tuning_coarse_detailed.csv"

    with detailed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "alpha",
                "beta",
                "decay",
                "n_ants",
                "n_iterations",
                "method",
                "size",
                "instance_index",
                "seed",
                "cost",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Default-config cost for paired normalization.
    default_costs = {}
    for row in rows:
        if (
            is_default(row)
            and row["method"] == METHOD_LLM
        ):
            default_costs[
                (
                    row["size"],
                    row["instance_index"],
                    row["seed"],
                    row["method"],
                )
            ] = float(row["cost"])

    grouped = defaultdict(list)
    for row in rows:
        key = (
            float(row["alpha"]),
            float(row["beta"]),
            float(row["decay"]),
            int(row["n_ants"]),
        )

        ref = default_costs[
            (
                row["size"],
                row["instance_index"],
                row["seed"],
                row["method"],
            )
        ]

        grouped[key].append(
            {
                "cost": float(row["cost"]),
                "runtime_sec": float(row["runtime_sec"]),
                "relative": float(row["cost"]) / ref,
                "size": int(row["size"]),
            }
        )

    summary_rows = []

    for key, values in grouped.items():
        alpha, beta, decay, n_ants = key
        relative_values = [v["relative"] for v in values]
        costs = [v["cost"] for v in values]
        runtimes = [v["runtime_sec"] for v in values]

        size_relative = {}
        for size in VALIDATION_SIZES:
            x = [v["relative"] for v in values if v["size"] == size]
            size_relative[size] = float(np.mean(x))

        summary_rows.append(
            {
                "alpha": alpha,
                "beta": beta,
                "decay": decay,
                "n_ants": n_ants,
                "n_iterations": N_ITERATIONS,
                "llm_relative_to_default": float(np.mean(relative_values)),
                "llm_relative_std": float(np.std(relative_values)),
                "llm_mean_cost": float(np.mean(costs)),
                "llm_runtime_sec": float(np.mean(runtimes)),
                "llm_relative_val20": size_relative[20],
                "llm_relative_val50": size_relative[50],
                "llm_relative_val100": size_relative[100],
            }
        )

    summary_rows.sort(key=lambda r: r["llm_relative_to_default"])

    summary_path = RESULTS_DIR / "aco_tuning_coarse_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 100)
    print("TOP 10 COARSE CONFIGURATIONS")
    print("=" * 100)
    print(
        f"{'rank':>4} {'alpha':>6} {'beta':>6} {'decay':>7} "
        f"{'ants':>5} {'LLM rel.':>12} {'std':>10}"
    )

    for rank, row in enumerate(summary_rows[:10], start=1):
        print(
            f"{rank:>4} "
            f"{row['alpha']:>6g} "
            f"{row['beta']:>6g} "
            f"{row['decay']:>7g} "
            f"{row['n_ants']:>5d} "
            f"{row['llm_relative_to_default']:>12.6f} "
            f"{row['llm_relative_std']:>10.6f}"
        )

    print("\nSaved:")
    print(detailed_path)
    print(summary_path)
    print(
        f"\nTotal coarse elapsed: "
        f"{time.perf_counter() - overall_start:.3f}s"
    )


# ---------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------

def read_top_configs(top_k: int) -> list[dict]:
    summary_path = RESULTS_DIR / "aco_tuning_coarse_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Run coarse tuning first: {summary_path}"
        )

    with summary_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: float(r["llm_relative_to_default"]))

    configs = []
    for row in rows[:top_k]:
        configs.append(
            {
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "decay": float(row["decay"]),
                "n_ants": int(row["n_ants"]),
                "n_iterations": N_ITERATIONS,
            }
        )

    # Always include the default configuration as a control.
    if DEFAULT_CONFIG not in configs:
        configs.append(dict(DEFAULT_CONFIG))

    return configs


def run_confirmation(
    *,
    instances_per_size: int,
    seeds: list[int],
    top_k: int,
    device: str,
) -> None:
    llm_heuristic = load_llm_heuristic()
    validation = load_validation_data(instances_per_size)
    candidate_configs = read_top_configs(top_k)

    methods = {
        METHOD_BASELINE: lambda dist: 1.0 / dist,
        METHOD_LLM: llm_heuristic,
    }

    print(f"\nConfirmation configurations: {len(candidate_configs)}")
    for i, config in enumerate(candidate_configs, start=1):
        print(f"  {i}. {config_label(config)}")
    print(f"Seeds: {seeds}")
    print(f"Iterations: {N_ITERATIONS}")
    print(f"Device: {device}\n")

    rows = []
    start_all = time.perf_counter()

    for config_number, config in enumerate(candidate_configs, start=1):
        print(
            f"[CONFIRM {config_number}/{len(candidate_configs)}] "
            f"{config_label(config)}",
            flush=True,
        )

        for size in VALIDATION_SIZES:
            for instance_idx, dist in validation[size]:
                for seed in seeds:
                    for method_name, heuristic_fn in methods.items():
                        cost, runtime = run_one(
                            dist,
                            heuristic_fn,
                            seed,
                            alpha=config["alpha"],
                            beta=config["beta"],
                            decay=config["decay"],
                            n_ants=config["n_ants"],
                            n_iterations=N_ITERATIONS,
                            device=device,
                        )

                        rows.append(
                            {
                                "alpha": config["alpha"],
                                "beta": config["beta"],
                                "decay": config["decay"],
                                "n_ants": config["n_ants"],
                                "n_iterations": N_ITERATIONS,
                                "method": method_name,
                                "size": size,
                                "instance_index": instance_idx,
                                "seed": seed,
                                "cost": cost,
                                "runtime_sec": runtime,
                            }
                        )

    detailed_path = RESULTS_DIR / "aco_tuning_confirmation_detailed.csv"

    with detailed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # Build paired lookup and default references.
    lookup = {}
    default_lookup = {}

    for row in rows:
        key = (
            float(row["alpha"]),
            float(row["beta"]),
            float(row["decay"]),
            int(row["n_ants"]),
            row["method"],
            int(row["size"]),
            int(row["instance_index"]),
            int(row["seed"]),
        )
        lookup[key] = float(row["cost"])

        if (
            is_default(row)
            and row["method"] in {METHOD_BASELINE, METHOD_LLM}
        ):
            default_lookup[
                (
                    row["method"],
                    int(row["size"]),
                    int(row["instance_index"]),
                    int(row["seed"]),
                )
            ] = float(row["cost"])

    grouped = defaultdict(list)

    for row in rows:
        cfg_key = (
            float(row["alpha"]),
            float(row["beta"]),
            float(row["decay"]),
            int(row["n_ants"]),
        )
        grouped[cfg_key].append(row)

    summary = []

    for cfg_key, cfg_rows in grouped.items():
        alpha, beta, decay, n_ants = cfg_key

        metrics = {}

        for method in (METHOD_BASELINE, METHOD_LLM):
            method_rows = [
                r for r in cfg_rows if r["method"] == method
            ]

            relative = []
            costs = []
            runtimes = []

            for r in method_rows:
                ref = default_lookup[
                    (
                        method,
                        r["size"],
                        r["instance_index"],
                        r["seed"],
                    )
                ]
                relative.append(float(r["cost"]) / ref)
                costs.append(float(r["cost"]))
                runtimes.append(float(r["runtime_sec"]))

            metrics[method] = {
                "relative": float(np.mean(relative)),
                "relative_std": float(np.std(relative)),
                "mean_cost": float(np.mean(costs)),
                "mean_runtime": float(np.mean(runtimes)),
            }

        # Paired baseline vs LLM comparison.
        paired_improvements = []

        for size in VALIDATION_SIZES:
            for instance_idx, _dist in validation[size]:
                for seed in seeds:
                    baseline = lookup[
                        (
                            alpha,
                            beta,
                            decay,
                            n_ants,
                            METHOD_BASELINE,
                            size,
                            instance_idx,
                            seed,
                        )
                    ]
                    llm = lookup[
                        (
                            alpha,
                            beta,
                            decay,
                            n_ants,
                            METHOD_LLM,
                            size,
                            instance_idx,
                            seed,
                        )
                    ]
                    paired_improvements.append(
                        (baseline - llm) / baseline * 100.0
                    )

        summary.append(
            {
                "alpha": alpha,
                "beta": beta,
                "decay": decay,
                "n_ants": n_ants,
                "n_iterations": N_ITERATIONS,
                "baseline_relative_to_default": metrics[METHOD_BASELINE]["relative"],
                "baseline_relative_std": metrics[METHOD_BASELINE]["relative_std"],
                "baseline_mean_cost": metrics[METHOD_BASELINE]["mean_cost"],
                "baseline_runtime_sec": metrics[METHOD_BASELINE]["mean_runtime"],
                "llm_relative_to_default": metrics[METHOD_LLM]["relative"],
                "llm_relative_std": metrics[METHOD_LLM]["relative_std"],
                "llm_mean_cost": metrics[METHOD_LLM]["mean_cost"],
                "llm_runtime_sec": metrics[METHOD_LLM]["mean_runtime"],
                "llm_vs_baseline_improvement_percent": float(
                    np.mean(paired_improvements)
                ),
            }
        )

    # Primary selection: best LLM relative-to-default score.
    # Secondary: lower variance, then lower baseline relative score.
    summary.sort(
        key=lambda r: (
            r["llm_relative_to_default"],
            r["llm_relative_std"],
            r["baseline_relative_to_default"],
        )
    )

    summary_path = RESULTS_DIR / "aco_tuning_confirmation_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)

    winner = summary[0]
    winner_config = {
        "alpha": float(winner["alpha"]),
        "beta": float(winner["beta"]),
        "decay": float(winner["decay"]),
        "n_ants": int(winner["n_ants"]),
        "n_iterations": N_ITERATIONS,
    }

    winner_path = RESULTS_DIR / "aco_selected_config.json"
    with winner_path.open("w", encoding="utf-8") as f:
        json.dump(winner_config, f, indent=2)

    print("\n" + "=" * 110)
    print("CONFIRMATION RESULTS")
    print("=" * 110)
    print(
        f"{'rank':>4} {'alpha':>6} {'beta':>6} {'decay':>7} "
        f"{'ants':>5} {'LLM rel.':>12} {'BASE rel.':>12} "
        f"{'LLM vs BASE':>14}"
    )

    for rank, row in enumerate(summary, start=1):
        print(
            f"{rank:>4} "
            f"{row['alpha']:>6g} "
            f"{row['beta']:>6g} "
            f"{row['decay']:>7g} "
            f"{row['n_ants']:>5d} "
            f"{row['llm_relative_to_default']:>12.6f} "
            f"{row['baseline_relative_to_default']:>12.6f} "
            f"{row['llm_vs_baseline_improvement_percent']:>14.3f}%"
        )

    print("\nSELECTED CONFIGURATION")
    print(json.dumps(winner_config, indent=2))

    print("\nSaved:")
    print(detailed_path)
    print(summary_path)
    print(winner_path)
    print(
        f"\nTotal confirmation elapsed: "
        f"{time.perf_counter() - start_all:.3f}s"
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=("coarse", "confirm"),
        default="coarse",
    )
    parser.add_argument(
        "--instances-per-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.instances_per_size <= 0:
        raise ValueError("--instances-per-size must be positive")

    if not args.seeds:
        raise ValueError("At least one seed is required")

    print("=" * 100)
    print("ACO HYPERPARAMETER TUNING")
    print("=" * 100)
    print(
        "NOTE: 'decay' is the parameter used by the current ACO code "
        "as pheromone retention: pheromone *= decay."
    )

    if args.mode == "coarse":
        run_coarse(
            instances_per_size=args.instances_per_size,
            seeds=args.seeds,
            device=args.device,
        )
    else:
        run_confirmation(
            instances_per_size=args.instances_per_size,
            seeds=args.seeds,
            top_k=args.top_k,
            device=args.device,
        )


if __name__ == "__main__":
    main()
