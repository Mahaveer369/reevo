"""Giffler-Thompson (GT) Active Schedule Constructor (deterministic & stochastic multi-start)."""

from __future__ import annotations
import math
import random
import time
from typing import Dict, List, Optional, Tuple
from .parser import JSSPInstance
from .schedule import OperationID, ScheduleResult, evaluate_machine_sequences, validate_schedule
from .baselines import PriorityFunc, compute_conflict_features, rule_mwkr


def sanitize_scores(raw_scores: list, n_expected: int) -> List[float]:
    """Ensure LLM-generated priority scores are finite floats of exact length |C|."""
    if not isinstance(raw_scores, (list, tuple)) or len(raw_scores) != n_expected:
        return [0.0] * n_expected
    cleaned: List[float] = []
    for val in raw_scores:
        try:
            fval = float(val)
            if math.isnan(fval) or math.isinf(fval):
                fval = 0.0
            cleaned.append(max(-1e9, min(1e9, fval)))
        except Exception:
            cleaned.append(0.0)
    return cleaned


def build_active_schedule_gt(
    instance: JSSPInstance,
    priority_fn: PriorityFunc = rule_mwkr,
    solver_name: str = "GT-Active",
    stochastic_temperature: float = 0.0,
    rng: Optional[random.Random] = None,
    validate: bool = True,
) -> ScheduleResult:
    """
    Construct a guaranteed feasible active schedule using the Giffler-Thompson (1960) algorithm:
      1. Maintain the set E of currently precedence-eligible operations (one per incomplete job).
      2. Find o* in E with minimum earliest completion time c* = min_{o in E} (max(job_ready[j], mach_ready[m_o]) + p_o).
      3. Identify bottleneck machine m* = m_{o*} and conflict set C = {o in E : m_o == m* and earliest_start(o) < c*}.
      4. If |C| == 1, schedule that unique operation; otherwise evaluate priority_fn(features(C)) and select argmax
         (or sample via Boltzmann softmax when stochastic_temperature > 0).
    """
    t0 = time.perf_counter()
    n_jobs = instance.n_jobs
    n_machines = instance.n_machines
    total_ops = instance.total_ops

    job_next_op = [0] * n_jobs
    job_ready = [0] * n_jobs
    machine_ready = [0] * n_machines

    job_rem_work = [sum(instance.durations[j]) for j in range(n_jobs)]
    machine_rem_work = [0] * n_machines
    for j in range(n_jobs):
        for k in range(n_machines):
            machine_rem_work[instance.machines[j][k]] += instance.durations[j][k]

    mean_pt = max(1.0, sum(job_rem_work) / float(max(1, total_ops)))
    max_job_work = float(max(job_rem_work) if job_rem_work else 1.0)
    max_mach_work = float(max(machine_rem_work) if machine_rem_work else 1.0)

    start_times: Dict[OperationID, int] = {}
    end_times: Dict[OperationID, int] = {}
    machine_sequences: List[List[OperationID]] = [[] for _ in range(n_machines)]

    for scheduled_count in range(total_ops):
        # 1. Collect eligible operations and find minimum earliest completion c*
        min_c = None
        bottleneck_machine = -1
        eligible_by_machine: Dict[int, List[Tuple[OperationID, int, int]]] = {}

        for j in range(n_jobs):
            k = job_next_op[j]
            if k < n_machines:
                m = instance.machines[j][k]
                p = instance.durations[j][k]
                es = job_ready[j] if job_ready[j] >= machine_ready[m] else machine_ready[m]
                ec = es + p
                if m not in eligible_by_machine:
                    eligible_by_machine[m] = []
                eligible_by_machine[m].append(((j, k), es, ec))
                if min_c is None or ec < min_c:
                    min_c = ec
                    bottleneck_machine = m

        # 2. Form Giffler-Thompson conflict set C on bottleneck_machine
        candidates = [
            op for (op, es, _ec) in eligible_by_machine[bottleneck_machine]
            if es < min_c
        ]

        if len(candidates) == 1:
            chosen_op = candidates[0]
        else:
            features = compute_conflict_features(
                instance=instance,
                conflict_ops=candidates,
                job_ready=job_ready,
                machine_ready=machine_ready,
                job_rem_work=job_rem_work,
                job_next_op=job_next_op,
                machine_rem_work=machine_rem_work,
                scheduled_count=scheduled_count,
                mean_pt=mean_pt,
                max_job_work=max_job_work,
                max_mach_work=max_mach_work,
            )
            try:
                raw_scores = priority_fn(features)
            except Exception:
                raw_scores = rule_mwkr(features)
            scores = sanitize_scores(raw_scores, len(candidates))

            if stochastic_temperature <= 1e-6 or rng is None:
                # Deterministic tie-breaking by score then job index
                best_idx = max(range(len(candidates)), key=lambda i: (scores[i], -candidates[i][0]))
                chosen_op = candidates[best_idx]
            else:
                max_s = max(scores)
                weights = [math.exp(max(-30.0, (s - max_s) / stochastic_temperature)) for s in scores]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0.0
                chosen_op = candidates[-1]
                for idx_c, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        chosen_op = candidates[idx_c]
                        break

        # 3. Schedule chosen_op
        j_sel, k_sel = chosen_op
        m_sel = instance.machines[j_sel][k_sel]
        p_sel = instance.durations[j_sel][k_sel]
        s_sel = max(job_ready[j_sel], machine_ready[m_sel])
        e_sel = s_sel + p_sel

        start_times[chosen_op] = s_sel
        end_times[chosen_op] = e_sel
        machine_sequences[m_sel].append(chosen_op)

        job_ready[j_sel] = e_sel
        machine_ready[m_sel] = e_sel
        job_next_op[j_sel] += 1
        job_rem_work[j_sel] -= p_sel
        machine_rem_work[m_sel] -= p_sel

    sched = evaluate_machine_sequences(
        instance=instance,
        machine_sequences=machine_sequences,
        solver_name=solver_name,
        compute_critical=True,
    )
    assert sched is not None, "Giffler-Thompson active schedule is guaranteed acyclic"
    sched.runtime_ms = (time.perf_counter() - t0) * 1000.0
    if validate:
        sched.validation = validate_schedule(instance, sched)
    return sched
