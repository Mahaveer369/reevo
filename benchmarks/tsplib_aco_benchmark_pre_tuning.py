from __future__ import annotations

import csv
import importlib.util
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
TSPLIB_DIR = BENCHMARK_DIR / "tsplib"
HEURISTIC_PATH = (
    BENCHMARK_DIR / "heuristics" / "qwen_reevo_best_tsp.py"
)
RESULTS_DIR = BENCHMARK_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Make ReEvo's tsp_aco package importable.
TSP_ACO_DIR = REPO_ROOT / "problems" / "tsp_aco"

if str(TSP_ACO_DIR) not in sys.path:
    sys.path.insert(0, str(TSP_ACO_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aco import ACO
from tsplib_parser import load_instance


# ---------------------------------------------------------
# Load frozen LLM heuristic
# ---------------------------------------------------------

def load_llm_heuristic():
    spec = importlib.util.spec_from_file_location(
        "qwen_reevo_best_tsp",
        HEURISTIC_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load heuristic from {HEURISTIC_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "heuristics_v2"):
        raise AttributeError(
            "Frozen heuristic must define heuristics_v2()."
        )

    return module.heuristics_v2


# ---------------------------------------------------------
# Prepare distance matrix
# ---------------------------------------------------------

def prepare_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Match ReEvo's evaluator convention.

    ReEvo replaces diagonal zeros with 1 before
    passing the matrix to the heuristic.
    """
    dist = distance_matrix.astype(np.float64, copy=True)

    diagonal = np.diag_indices_from(dist)
    dist[diagonal] = 1.0

    return dist


# ---------------------------------------------------------
# Run one ACO experiment
# ---------------------------------------------------------

def run_aco(
    distance_matrix: np.ndarray,
    heuristic_fn,
    seed: int,
    device: str = "cpu",
    n_ants: int = 30,
    n_iterations: int = 100,
) -> dict:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    dist = prepare_distance_matrix(distance_matrix)

    heuristic = heuristic_fn(dist.copy())

    # Match ReEvo evaluator's numerical safeguard.
    heuristic = np.asarray(heuristic, dtype=np.float64)
    heuristic = heuristic + 1e-9
    heuristic[heuristic < 1e-9] = 1e-9

    aco = ACO(
        dist,
        heuristic,
        n_ants=n_ants,
        device=device,
    )

    start_time = time.perf_counter()

    objective = aco.run(n_iterations)

    elapsed = time.perf_counter() - start_time

    if isinstance(objective, torch.Tensor):
        objective_value = float(objective.item())
    else:
        objective_value = float(objective)

    route = None

    if aco.shortest_path is not None:
        route = (
            aco.shortest_path.detach()
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )

    return {
        "cost": objective_value,
        "runtime_sec": elapsed,
        "route": route,
    }


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

INSTANCES = [
    "att48",
    "berlin52",
    "st70",
    "pr76",
    "kroA100",
    "rat195",
]

SEEDS = list(range(10))

N_ANTS = 30
N_ITERATIONS = 100

# CPU is safest for the first benchmark.
# We can move this to GPU after validating correctness.
DEVICE = "cpu"


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    llm_heuristic = load_llm_heuristic()

    detailed_rows = []

    summary_rows = []

    for instance_name in INSTANCES:

        tsp_path = TSPLIB_DIR / f"{instance_name}.tsp"
        opt_path = TSPLIB_DIR / f"{instance_name}.opt.tour"

        instance = load_instance(
            tsp_path,
            opt_path if opt_path.exists() else None,
        )

        distance_matrix = instance["distance_matrix"]
        optimal_cost = instance["optimal_cost"]

        print("\n" + "=" * 70)
        print(f"Instance: {instance_name}")
        print(f"Nodes: {instance['dimension']}")
        print(f"Distance type: {instance['edge_weight_type']}")
        print(f"Known optimum: {optimal_cost}")
        print("=" * 70)

        methods = {
            "aco_baseline": lambda d: 1.0 / d,
            "aco_llm": llm_heuristic,
        }

        instance_summary = {
            "instance": instance_name,
            "nodes": instance["dimension"],
            "optimal_cost": optimal_cost,
        }

        for method_name, heuristic_fn in methods.items():

            costs = []
            runtimes = []
            best_route = None
            best_cost = float("inf")

            print(f"\n--- {method_name} ---")

            for seed in SEEDS:

                result = run_aco(
                    distance_matrix,
                    heuristic_fn,
                    seed=seed,
                    device=DEVICE,
                    n_ants=N_ANTS,
                    n_iterations=N_ITERATIONS,
                )

                cost = result["cost"]
                runtime = result["runtime_sec"]
                route = result["route"]

                costs.append(cost)
                runtimes.append(runtime)

                if cost < best_cost:
                    best_cost = cost
                    best_route = route

                gap = None

                if optimal_cost is not None:
                    gap = (
                        (cost - optimal_cost)
                        / optimal_cost
                        * 100.0
                    )

                print(
                    f"seed={seed:02d} "
                    f"cost={cost:.3f} "
                    f"time={runtime:.3f}s "
                    f"gap={gap if gap is not None else 'NA'}"
                )

                detailed_rows.append({
                    "instance": instance_name,
                    "nodes": instance["dimension"],
                    "method": method_name,
                    "seed": seed,
                    "cost": cost,
                    "runtime_sec": runtime,
                    "optimal_cost": optimal_cost,
                    "gap_percent": gap,
                    "route": route,
                })

            mean_cost = float(np.mean(costs))
            std_cost = float(np.std(costs))
            best_cost = float(np.min(costs))
            mean_runtime = float(np.mean(runtimes))

            mean_gap = None

            if optimal_cost is not None:
                mean_gap = (
                    (mean_cost - optimal_cost)
                    / optimal_cost
                    * 100.0
                )

            prefix = (
                "baseline"
                if method_name == "aco_baseline"
                else "llm"
            )

            instance_summary[f"{prefix}_mean"] = mean_cost
            instance_summary[f"{prefix}_std"] = std_cost
            instance_summary[f"{prefix}_best"] = best_cost
            instance_summary[f"{prefix}_runtime"] = mean_runtime
            instance_summary[f"{prefix}_gap"] = mean_gap
            instance_summary[f"{prefix}_best_route"] = best_route

        baseline_mean = instance_summary["baseline_mean"]
        llm_mean = instance_summary["llm_mean"]

        improvement = (
            (baseline_mean - llm_mean)
            / baseline_mean
            * 100.0
        )

        instance_summary["llm_improvement_percent"] = improvement

        summary_rows.append(instance_summary)

    # -----------------------------------------------------
    # Save detailed CSV
    # -----------------------------------------------------

    detailed_path = RESULTS_DIR / "tsplib_detailed_results.csv"

    with detailed_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance",
                "nodes",
                "method",
                "seed",
                "cost",
                "runtime_sec",
                "optimal_cost",
                "gap_percent",
                "route",
            ],
        )

        writer.writeheader()
        writer.writerows(detailed_rows)

    # -----------------------------------------------------
    # Save summary CSV
    # -----------------------------------------------------

    summary_path = RESULTS_DIR / "tsplib_summary.csv"

    summary_fields = [
        "instance",
        "nodes",
        "optimal_cost",
        "baseline_mean",
        "baseline_std",
        "baseline_best",
        "baseline_runtime",
        "baseline_gap",
        "llm_mean",
        "llm_std",
        "llm_best",
        "llm_runtime",
        "llm_gap",
        "llm_improvement_percent",
        "baseline_best_route",
        "llm_best_route",
    ]

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields,
        )

        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    # -----------------------------------------------------
    # Print final table
    # -----------------------------------------------------

    print("\n\n" + "=" * 100)
    print("FINAL TSPLIB SUMMARY")
    print("=" * 100)

    print(
        f"{'Instance':<12}"
        f"{'n':>5}"
        f"{'OPT':>12}"
        f"{'ACO Mean':>14}"
        f"{'LLM Mean':>14}"
        f"{'ACO Gap':>12}"
        f"{'LLM Gap':>12}"
        f"{'Improvement':>14}"
    )

    for row in summary_rows:

        opt = (
            str(row["optimal_cost"])
            if row["optimal_cost"] is not None
            else "NA"
        )

        aco_gap = (
            f"{row['baseline_gap']:.2f}%"
            if row["baseline_gap"] is not None
            else "NA"
        )

        llm_gap = (
            f"{row['llm_gap']:.2f}%"
            if row["llm_gap"] is not None
            else "NA"
        )

        print(
            f"{row['instance']:<12}"
            f"{row['nodes']:>5}"
            f"{opt:>12}"
            f"{row['baseline_mean']:>14.3f}"
            f"{row['llm_mean']:>14.3f}"
            f"{aco_gap:>12}"
            f"{llm_gap:>12}"
            f"{row['llm_improvement_percent']:>13.2f}%"
        )

    print("\nSaved:")
    print(detailed_path)
    print(summary_path)


if __name__ == "__main__":
    main()
