"""
Standalone Berth Allocation System (BAP / CBAP / C/T-V BACAP / DCBAP) Benchmark Suite.

Executes all 7 solvers across all 7 benchmark instances, verifies 100% of schedules
via `verify_schedule()`, runs the Event-Driven vs Time-Driven Rolling-Horizon
comparison (Chen et al. 2026 Section 4.3), and writes:
  - /benchmarks/results/bap_benchmark_detailed.csv
  - /benchmarks/results/bap_benchmark_summary.csv
  - /benchmarks/results/bap_rolling_horizon_comparison.csv
  - /benchmarks/results/bap_initial_state_cache.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from problems.berth_allocation.eval import DEFAULT_CUSTOM_BAP_CODE
from problems.berth_allocation.instances import get_all_bap_instances, get_bap_instance
from problems.berth_allocation.model import verify_schedule
from problems.berth_allocation.solvers import (
    BAP_SOLVER_METADATA,
    BAP_SOLVER_REGISTRY,
    compare_rolling_horizon_strategies,
)

RESULTS_DIR = ROOT_DIR / "benchmarks" / "results"
DETAILED_CSV = RESULTS_DIR / "bap_benchmark_detailed.csv"
SUMMARY_CSV = RESULTS_DIR / "bap_benchmark_summary.csv"
ROLLING_CSV = RESULTS_DIR / "bap_rolling_horizon_comparison.csv"
BAP_CACHE_JSON = RESULTS_DIR / "bap_initial_state_cache.json"


def run_bap_benchmarks(save_files: bool = True) -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    instances = get_all_bap_instances()
    solver_names = list(BAP_SOLVER_REGISTRY.keys())

    detailed_rows: List[Dict[str, Any]] = []
    schedules_by_inst_solver: Dict[str, Dict[str, Any]] = {}

    for inst_name, inst in instances.items():
        schedules_by_inst_solver[inst_name] = {}
        fifp_sched = BAP_SOLVER_REGISTRY["fifp_greedy"](inst)
        fifp_obj = fifp_sched.total_objective

        for s_name in solver_names:
            solver_fn = BAP_SOLVER_REGISTRY[s_name]
            sched = (
                fifp_sched if s_name == "fifp_greedy" else solver_fn(inst)
            )
            ok, violations = verify_schedule(inst, sched)
            if not ok:
                raise RuntimeError(
                    f"Infeasible schedule from {s_name} on {inst_name}: {violations}"
                )

            imp_vs_fifp = round(
                (fifp_obj - sched.total_objective) / max(1e-6, fifp_obj) * 100.0,
                2,
            )
            row = {
                "instance": inst_name,
                "instance_title": inst.title,
                "paper_ref": inst.paper_ref,
                "num_vessels": len(inst.vessels),
                "quay_length_m": inst.quay_length,
                "total_cranes": inst.total_cranes,
                "use_crane_allocation": inst.use_crane_allocation,
                "solver": s_name,
                "solver_label": BAP_SOLVER_METADATA[s_name]["label"],
                "solver_paper": BAP_SOLVER_METADATA[s_name]["paper"],
                "total_objective": round(sched.total_objective, 2),
                "lower_bound": round(sched.lower_bound, 2),
                "optimality_gap_pct": round(sched.optimality_gap_pct, 2),
                "improvement_vs_fifp_pct": imp_vs_fifp,
                "total_waiting_time_h": round(sched.total_waiting_time, 2),
                "total_service_time_h": round(sched.total_service_time, 2),
                "total_berth_deviation_m": round(sched.total_berth_deviation, 1),
                "makespan_h": round(sched.makespan, 2),
                "runtime_ms": round(sched.runtime_ms, 2),
                "feasible": sched.feasible,
            }
            detailed_rows.append(row)
            schedules_by_inst_solver[inst_name][s_name] = sched.to_dict()

    # Compute solver summary across all instances
    best_obj_per_inst: Dict[str, float] = {}
    for inst_name in instances:
        best_obj_per_inst[inst_name] = min(
            r["total_objective"]
            for r in detailed_rows
            if r["instance"] == inst_name
        )

    summary_rows: List[Dict[str, Any]] = []
    for s_name in solver_names:
        s_rows = [r for r in detailed_rows if r["solver"] == s_name]
        n = len(s_rows)
        wins = sum(
            1
            for r in s_rows
            if abs(r["total_objective"] - best_obj_per_inst[r["instance"]]) <= 1e-4
        )
        summary_rows.append(
            {
                "solver": s_name,
                "solver_label": BAP_SOLVER_METADATA[s_name]["label"],
                "solver_paper": BAP_SOLVER_METADATA[s_name]["paper"],
                "category": BAP_SOLVER_METADATA[s_name]["category"],
                "instances_tested": n,
                "best_or_tied_wins": f"{wins}/{n}",
                "mean_objective": round(
                    sum(r["total_objective"] for r in s_rows) / n, 2
                ),
                "mean_optimality_gap_pct": round(
                    sum(r["optimality_gap_pct"] for r in s_rows) / n, 2
                ),
                "mean_improvement_vs_fifp_pct": round(
                    sum(r["improvement_vs_fifp_pct"] for r in s_rows) / n, 2
                ),
                "mean_waiting_time_h": round(
                    sum(r["total_waiting_time_h"] for r in s_rows) / n, 2
                ),
                "mean_berth_deviation_m": round(
                    sum(r["total_berth_deviation_m"] for r in s_rows) / n, 1
                ),
                "mean_runtime_ms": round(
                    sum(r["runtime_ms"] for r in s_rows) / n, 2
                ),
            }
        )

    # Rolling-Horizon comparison across Chen et al. (2026) dynamic instances
    rolling_results: Dict[str, Any] = {}
    rolling_csv_rows: List[Dict[str, Any]] = []
    for rh_inst_name in ("chen26_table10", "chen26_n20", "xu25_bacap_20"):
        rh_data = compare_rolling_horizon_strategies(get_bap_instance(rh_inst_name))
        rolling_results[rh_inst_name] = rh_data
        for st in rh_data["strategies"]:
            rolling_csv_rows.append(
                {
                    "instance": rh_inst_name,
                    "strategy": st["strategy"],
                    "total_objective": st["total_objective"],
                    "total_waiting_time": st["total_waiting_time"],
                    "total_service_time": st["total_service_time"],
                    "total_berth_deviation": st["total_berth_deviation"],
                    "makespan": st["makespan"],
                    "optimality_gap_pct": st["optimality_gap_pct"],
                    "avg_replan_berth_shift_m": st["avg_replan_berth_shift_m"],
                    "num_triggers": st["num_triggers"],
                    "runtime_ms": st["runtime_ms"],
                    "feasible": st["feasible"],
                }
            )

    if save_files:
        with DETAILED_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detailed_rows)

        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        with ROLLING_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rolling_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rolling_csv_rows)

    payload = {
        "instances": [inst.to_dict() for inst in instances.values()],
        "solvers": BAP_SOLVER_METADATA,
        "summary": summary_rows,
        "detailed": detailed_rows,
        "rolling_horizon": rolling_results,
        "default_instance": "xu25_bacap_20",
        "default_comparison": {
            "instance": instances["xu25_bacap_20"].to_dict(),
            "baseline_schedule": schedules_by_inst_solver["xu25_bacap_20"]["fifp_greedy"],
            "hma_schedule": schedules_by_inst_solver["xu25_bacap_20"]["hma_xu2025"],
            "reevo_schedule": schedules_by_inst_solver["xu25_bacap_20"]["reevo_vrpagent_bap"],
        },
        "precomputed_schedules": schedules_by_inst_solver,
        "default_custom_code": DEFAULT_CUSTOM_BAP_CODE,
    }

    if save_files:
        BAP_CACHE_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


if __name__ == "__main__":
    t_start = time.perf_counter()
    res = run_bap_benchmarks(save_files=True)
    dt = (time.perf_counter() - t_start) * 1000.0
    print(f"=== Berth Allocation Benchmark Suite Completed in {dt:.1f} ms ===")
    print("\n--- Summary Across All 7 Benchmark Instances ---")
    for s in res["summary"]:
        print(
            f"{s['solver_label']:38s} | Wins: {s['best_or_tied_wins']:4s} | "
            f"Mean Obj: {s['mean_objective']:8.2f} | "
            f"Imp vs FIFP: +{s['mean_improvement_vs_fifp_pct']:5.2f}% | "
            f"Wait: {s['mean_waiting_time_h']:6.2f}h | "
            f"Dev: {s['mean_berth_deviation_m']:6.1f}m | "
            f"Time: {s['mean_runtime_ms']:6.2f}ms"
        )
    print("\n--- Rolling-Horizon Comparison (chen26_table10) ---")
    for st in res["rolling_horizon"]["chen26_table10"]["strategies"]:
        print(
            f"{st['strategy']:18s} | Obj: {st['total_objective']:7.2f} | "
            f"Wait: {st['total_waiting_time']:5.2f}h | "
            f"Shift: {st['avg_replan_berth_shift_m']:5.1f}m | "
            f"Triggers: {st['num_triggers']} | Feasible: {st['feasible']}"
        )
