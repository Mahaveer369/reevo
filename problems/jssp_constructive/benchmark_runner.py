"""Controlled experimental harness for JSSP benchmarks, ablations, and TSP-to-JSSP diagnostic audit."""

from __future__ import annotations
import math
import statistics
import time
from typing import Any, Callable, Dict, List, Optional
from .parser import JSSPInstance, list_benchmark_instances, load_benchmark_instance
from .schedule import ScheduleResult
from .baselines import (
    CLASSICAL_RULES,
    PriorityFunc,
    rule_fifo,
    rule_lpt,
    rule_mor,
    rule_mwkr,
    rule_pt_plus_winq,
    rule_spt,
)
from .scheduler import build_active_schedule_gt
from .local_search import critical_path_local_search, multi_start_hybrid_solve
from .reevo_heuristics import (
    REEVO_QWEN_CODE,
    compile_custom_priority_code,
    rule_ablation_no_winq,
    rule_ablation_zero_shot,
    rule_reevo_qwen,
)


def run_benchmark_suite(
    instance_names: Optional[List[str]] = None,
    n_seeds: int = 5,
    n_starts: int = 10,
    ls_steps: int = 35,
    custom_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute controlled comparison across:
      A. Conventional Dispatching Rules (SPT, LPT, FIFO, MOR, MWKR, PT+WINQ)
      D1. Single-Pass ReEvo-Qwen Giffler-Thompson Priority Rule
      B. Classical Metaheuristic (MWKR + Critical-Path Tabu/Block-Swap Local Search)
      C. Multi-Start GT (MWKR Prior) + Critical-Path Local Search
      D2. Proposed Hybrid: Multi-Start GT (ReEvo-Qwen Prior) + Critical-Path Local Search
      Ablation 1: Zero-Shot LLM (No ReEvo Reflective Evolution) + Critical-Path LS
      Ablation 2: ReEvo-Qwen Prior WITHOUT Downstream WINQ Lookahead + Critical-Path LS
      Ablation 3: Multi-Start GT (ReEvo-Qwen Prior) WITHOUT Critical-Path Local Search
    """
    if not instance_names:
        instance_names = ["ft06", "ft10", "ft20", "la01", "la02", "la03", "la04", "la05", "la16", "orb01", "abz5", "abz6"]

    instances: List[JSSPInstance] = [load_benchmark_instance(name) for name in instance_names]
    seeds = [100 + s for s in range(max(1, n_seeds))]

    custom_fn: Optional[PriorityFunc] = None
    custom_error: Optional[str] = None
    if custom_code and custom_code.strip():
        try:
            custom_fn = compile_custom_priority_code(custom_code)
        except Exception as exc:
            custom_error = str(exc)

    solver_specs: List[Dict[str, Any]] = [
        {
            "id": "spt_gt",
            "category": "A. Classical Dispatching",
            "name": "SPT (Giffler–Thompson Active)",
            "mode": "single",
            "fn": rule_spt,
            "ls": False,
            "llm_calls": 0,
            "llm_tokens": 0,
        },
        {
            "id": "fifo_gt",
            "category": "A. Classical Dispatching",
            "name": "FIFO (Earliest Ready Time)",
            "mode": "single",
            "fn": rule_fifo,
            "ls": False,
            "llm_calls": 0,
            "llm_tokens": 0,
        },
        {
            "id": "mor_gt",
            "category": "A. Classical Dispatching",
            "name": "MOR (Most Operations Remaining)",
            "mode": "single",
            "fn": rule_mor,
            "ls": False,
            "llm_calls": 0,
            "llm_tokens": 0,
        },
        {
            "id": "mwkr_gt",
            "category": "A. Classical Dispatching",
            "name": "MWKR (Most Work Remaining)",
            "mode": "single",
            "fn": rule_mwkr,
            "ls": False,
            "llm_calls": 0,
            "llm_tokens": 0,
        },
        {
            "id": "pt_winq_gt",
            "category": "A. Classical Dispatching",
            "name": "PT + WINQ (Holthaus & Rajendran)",
            "mode": "single",
            "fn": rule_pt_plus_winq,
            "ls": False,
            "llm_calls": 0,
            "llm_tokens": 0,
        },
        {
            "id": "reevo_gt_single",
            "category": "D1. Single-Pass LLM Rule",
            "name": "ReEvo-Qwen Priority (Single-Pass GT)",
            "mode": "single",
            "fn": rule_reevo_qwen,
            "ls": False,
            "llm_calls": 60,
            "llm_tokens": 28400,
        },
        {
            "id": "mwkr_cpls_single",
            "category": "B. Classical Metaheuristic",
            "name": "Single-Start MWKR + Critical-Path LS (N1/N5)",
            "mode": "single",
            "fn": rule_mwkr,
            "ls": True,
            "llm_calls": 0,
            "llm_tokens": 0,
        },
        {
            "id": "multistart_mwkr_cpls",
            "category": "C. Non-LLM Multi-Start + LS",
            "name": f"Multi-Start GT (MWKR Prior, K={n_starts}) + CP-LS",
            "mode": "multistart",
            "fn": rule_mwkr,
            "ls": True,
            "llm_calls": 0,
            "llm_tokens": 0,
        },
        {
            "id": "multistart_reevo_cpls",
            "category": "D2. Proposed Hybrid (Full)",
            "name": f"Multi-Start GT (ReEvo-Qwen Prior, K={n_starts}) + CP-LS",
            "mode": "multistart",
            "fn": rule_reevo_qwen,
            "ls": True,
            "llm_calls": 60,
            "llm_tokens": 28400,
        },
        {
            "id": "ablation_zeroshot_cpls",
            "category": "Ablation 1 (No Evolution)",
            "name": f"Zero-Shot LLM Rule (No Reflection, K={n_starts}) + CP-LS",
            "mode": "multistart",
            "fn": rule_ablation_zero_shot,
            "ls": True,
            "llm_calls": 10,
            "llm_tokens": 4200,
        },
        {
            "id": "ablation_nowinq_cpls",
            "category": "Ablation 2 (No WINQ/Slack)",
            "name": f"ReEvo Rule w/o Downstream WINQ (K={n_starts}) + CP-LS",
            "mode": "multistart",
            "fn": rule_ablation_no_winq,
            "ls": True,
            "llm_calls": 60,
            "llm_tokens": 28400,
        },
        {
            "id": "ablation_reevo_nols",
            "category": "Ablation 3 (No Local Search)",
            "name": f"Multi-Start GT (ReEvo-Qwen Prior, K={n_starts}) w/o CP-LS",
            "mode": "multistart",
            "fn": rule_reevo_qwen,
            "ls": False,
            "llm_calls": 60,
            "llm_tokens": 28400,
        },
    ]

    if custom_fn is not None:
        solver_specs.append({
            "id": "custom_user_cpls",
            "category": "Custom Sandbox Heuristic",
            "name": f"Custom Sandbox Priority (K={n_starts}) + CP-LS",
            "mode": "multistart",
            "fn": custom_fn,
            "ls": True,
            "llm_calls": 0,
            "llm_tokens": 0,
        })

    rows: List[Dict[str, Any]] = []
    per_instance_matrix: Dict[str, Dict[str, Any]] = {
        inst.name: {
            "name": inst.name,
            "family": inst.family,
            "size": f"{inst.n_jobs}x{inst.n_machines}",
            "optimum": inst.optimum,
            "lower_bound": inst.lower_bound,
            "solvers": {},
        }
        for inst in instances
    }

    for spec in solver_specs:
        all_gaps: List[float] = []
        best_gaps_per_inst: List[float] = []
        all_cmax: List[float] = []
        all_runtimes: List[float] = []
        all_evals: List[int] = []
        feasible_count = 0
        total_runs = 0

        for inst in instances:
            ref = inst.optimum or inst.lower_bound
            seed_gaps: List[float] = []
            seed_cmax: List[int] = []
            run_seeds = [seeds[0]] if spec["mode"] == "single" else seeds

            for s in run_seeds:
                total_runs += 1
                if spec["mode"] == "single":
                    sched = build_active_schedule_gt(
                        instance=inst,
                        priority_fn=spec["fn"],
                        solver_name=spec["name"],
                        validate=True,
                    )
                    if spec["ls"]:
                        sched = critical_path_local_search(
                            instance=inst,
                            initial_schedule=sched,
                            max_steps=ls_steps * 2,
                            neighborhood="N1",
                            solver_name=spec["name"],
                            validate=True,
                        )
                else:
                    sched = multi_start_hybrid_solve(
                        instance=inst,
                        priority_fn=spec["fn"],
                        n_starts=n_starts,
                        ls_steps=ls_steps if spec["ls"] else 0,
                        temperature=0.20,
                        seed=s,
                        solver_name=spec["name"],
                        use_local_search=spec["ls"],
                    )

                if sched.validation and sched.validation.feasible:
                    feasible_count += 1
                gap = 100.0 * (sched.makespan - ref) / float(ref)
                seed_gaps.append(gap)
                seed_cmax.append(sched.makespan)
                all_gaps.append(gap)
                all_cmax.append(float(sched.makespan))
                all_runtimes.append(sched.runtime_ms)
                all_evals.append(sched.evaluations)

            best_gaps_per_inst.append(min(seed_gaps))
            per_instance_matrix[inst.name]["solvers"][spec["id"]] = {
                "mean_cmax": round(statistics.mean(seed_cmax), 1),
                "best_cmax": min(seed_cmax),
                "mean_gap": round(statistics.mean(seed_gaps), 2),
                "best_gap": round(min(seed_gaps), 2),
            }

        mean_gap = statistics.mean(all_gaps)
        sd_gap = statistics.stdev(all_gaps) if len(all_gaps) > 1 else 0.0
        median_gap = statistics.median(all_gaps)
        best_mean_gap = statistics.mean(best_gaps_per_inst)

        rows.append({
            "id": spec["id"],
            "category": spec["category"],
            "name": spec["name"],
            "mean_cmax": round(statistics.mean(all_cmax), 1),
            "mean_gap": round(mean_gap, 2),
            "sd_gap": round(sd_gap, 2),
            "median_gap": round(median_gap, 2),
            "best_gap": round(best_mean_gap, 2),
            "mean_runtime_ms": round(statistics.mean(all_runtimes), 2),
            "mean_evaluations": round(statistics.mean(all_evals), 1),
            "llm_calls": spec["llm_calls"],
            "llm_tokens": spec["llm_tokens"],
            "feasibility_rate": round(100.0 * feasible_count / float(max(1, total_runs)), 1),
        })

    return {
        "instance_names": instance_names,
        "n_seeds": n_seeds,
        "n_starts": n_starts,
        "ls_steps": ls_steps,
        "custom_error": custom_error,
        "summary_rows": rows,
        "per_instance": list(per_instance_matrix.values()),
    }


def inspect_single_instance_schedule(
    instance_name: str = "la01",
    solver_id: str = "multistart_reevo_cpls",
    n_starts: int = 10,
    ls_steps: int = 40,
    seed: int = 100,
    custom_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Solve a single instance and return full Gantt chart & critical-path telemetry for visual inspection."""
    inst = load_benchmark_instance(instance_name)

    solver_map = {
        "spt_gt": ("SPT (Giffler–Thompson Active)", rule_spt, False, False),
        "mwkr_gt": ("MWKR (Giffler–Thompson Active)", rule_mwkr, False, False),
        "pt_winq_gt": ("PT + WINQ (Giffler–Thompson Active)", rule_pt_plus_winq, False, False),
        "reevo_gt_single": ("ReEvo-Qwen Priority (Single-Pass GT)", rule_reevo_qwen, False, False),
        "mwkr_cpls_single": ("Single-Start MWKR + Critical-Path LS", rule_mwkr, False, True),
        "multistart_mwkr_cpls": ("Multi-Start GT (MWKR) + Critical-Path LS", rule_mwkr, True, True),
        "multistart_reevo_cpls": ("Multi-Start GT (ReEvo-Qwen) + Critical-Path LS", rule_reevo_qwen, True, True),
        "ablation_zeroshot_cpls": ("Zero-Shot LLM + Critical-Path LS", rule_ablation_zero_shot, True, True),
    }

    if solver_id == "custom_user_cpls" and custom_code:
        fn = compile_custom_priority_code(custom_code)
        label, is_multi, use_ls = ("Custom Sandbox Priority + Critical-Path LS", True, True)
    else:
        label, fn, is_multi, use_ls = solver_map.get(
            solver_id,
            ("Multi-Start GT (ReEvo-Qwen) + Critical-Path LS", rule_reevo_qwen, True, True),
        )

    if is_multi:
        sched = multi_start_hybrid_solve(
            instance=inst,
            priority_fn=fn,
            n_starts=n_starts,
            ls_steps=ls_steps if use_ls else 0,
            temperature=0.20,
            seed=seed,
            solver_name=label,
            use_local_search=use_ls,
        )
    else:
        sched = build_active_schedule_gt(inst, priority_fn=fn, solver_name=label, validate=True)
        if use_ls:
            sched = critical_path_local_search(inst, sched, max_steps=ls_steps * 2, solver_name=label, validate=True)

    # Also compute single-pass MWKR baseline for immediate delta comparison
    mwkr_base = build_active_schedule_gt(inst, priority_fn=rule_mwkr, solver_name="MWKR Baseline", validate=True)

    operations_list = []
    for j in range(inst.n_jobs):
        for k in range(inst.n_machines):
            op = (j, k)
            m = inst.machines[j][k]
            p = inst.durations[j][k]
            s = sched.start_times[op]
            e = sched.end_times[op]
            operations_list.append({
                "job": j,
                "op": k,
                "machine": m,
                "duration": p,
                "start": s,
                "end": e,
                "is_critical": op in sched.critical_ops,
            })

    blocks_formatted = [
        {
            "machine": m,
            "operations": [{"job": j, "op": k} for (j, k) in blk],
        }
        for (m, blk) in sched.critical_blocks
    ]

    ref = inst.optimum or inst.lower_bound
    return {
        "instance": {
            "name": inst.name,
            "family": inst.family,
            "n_jobs": inst.n_jobs,
            "n_machines": inst.n_machines,
            "optimum": inst.optimum,
            "lower_bound": inst.lower_bound,
        },
        "solver_id": solver_id,
        "solver_name": sched.solver_name,
        "makespan": sched.makespan,
        "mwkr_baseline_makespan": mwkr_base.makespan,
        "improvement_over_mwkr": mwkr_base.makespan - sched.makespan,
        "gap_percent": round(100.0 * (sched.makespan - ref) / float(ref), 2),
        "runtime_ms": round(sched.runtime_ms, 2),
        "evaluations": sched.evaluations,
        "critical_op_count": len(sched.critical_ops),
        "critical_block_count": len(sched.critical_blocks),
        "critical_blocks": blocks_formatted,
        "validation": {
            "feasible": sched.validation.feasible if sched.validation else False,
            "precedence_ok": sched.validation.precedence_ok if sched.validation else False,
            "machine_non_overlap_ok": sched.validation.machine_non_overlap_ok if sched.validation else False,
            "duration_exact_ok": sched.validation.duration_exact_ok if sched.validation else False,
            "completeness_ok": sched.validation.completeness_ok if sched.validation else False,
            "total_operations_checked": sched.validation.total_operations_checked if sched.validation else 0,
            "errors": sched.validation.errors if sched.validation else [],
        },
        "operations": operations_list,
    }
