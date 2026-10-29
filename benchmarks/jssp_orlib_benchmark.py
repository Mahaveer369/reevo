"""
Reproducible OR-Library JSSP Benchmark Runner (`benchmarks/jssp_orlib_benchmark.py`).
Executes all classical dispatching baselines, ReEvo-Qwen priority, VRPAgent-Ensemble priority,
and Critical-Path Block-Swap Local Search hybrids across all 12 canonical OR-Library instances,
validating 100% schedule feasibility and saving CSV artifacts to `benchmarks/results/`.
"""

from __future__ import annotations
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems.jssp_constructive.benchmark_runner import run_benchmark_suite


def main() -> None:
    results_dir = ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Running live JSSP OR-Library benchmark suite (12 instances, 3 seeds, K=5 starts, 25 CP-LS steps)...")
    suite = run_benchmark_suite(n_seeds=3, n_starts=5, ls_steps=25)

    summary_csv = results_dir / "jssp_orlib_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "name",
                "mean_cmax",
                "mean_gap",
                "sd_gap",
                "median_gap",
                "best_gap",
                "mean_runtime_ms",
                "mean_evaluations",
                "code_lines",
                "vrpagent_fitness",
                "feasibility_rate",
            ],
        )
        writer.writeheader()
        for row in suite["summary_rows"]:
            writer.writerow(row)

    detailed_csv = results_dir / "jssp_orlib_detailed.csv"
    with detailed_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["instance", "family", "size", "optimum", "lower_bound", "solver_id", "best_cmax", "mean_cmax", "best_gap", "mean_gap"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for inst_entry in suite["per_instance"]:
            for solver_id, sdata in inst_entry["solvers"].items():
                writer.writerow({
                    "instance": inst_entry["name"],
                    "family": inst_entry["family"],
                    "size": inst_entry["size"],
                    "optimum": inst_entry["optimum"],
                    "lower_bound": inst_entry["lower_bound"],
                    "solver_id": solver_id,
                    "best_cmax": sdata["best_cmax"],
                    "mean_cmax": sdata["mean_cmax"],
                    "best_gap": sdata["best_gap"],
                    "mean_gap": sdata["mean_gap"],
                })

    print(f"\nSaved summary CSV  -> {summary_csv}")
    print(f"Saved detailed CSV -> {detailed_csv}\n")
    print(f"{'Solver ID':<26} | {'Mean Gap':>8} | {'Best Gap':>8} | {'Median':>8} | {'Time (ms)':>9} | {'Lines':>5} | {'Feasible':>8}")
    print("-" * 88)
    for r in suite["summary_rows"]:
        print(
            f"{r['id']:<26} | {r['mean_gap']:>7.2f}% | {r['best_gap']:>7.2f}% | {r['median_gap']:>7.2f}% | "
            f"{r['mean_runtime_ms']:>7.2f}ms | {r['code_lines']:>5d} | {r['feasibility_rate']:>7.1f}%"
        )


if __name__ == "__main__":
    main()
