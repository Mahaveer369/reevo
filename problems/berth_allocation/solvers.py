"""
Berth Allocation System (BAP / CBAP / C/T-V BACAP / DCBAP) Solvers & Rolling-Horizon Engine.

Implements the complete algorithmic suite across all 6 reference papers + VRPAgent:
1. `fifp_greedy`: First-In-First-Planning (FIFP / FCFS) + Greedy Insertion
   (Baseline in Chen et al. 2026, Xu et al. 2025, and Zou et al. 2025).
2. `grasp_lee2010`: Lee, Chen & Cao (Transp. Res. Part E 2010) GRASP_2 with
   Proposition 3.1 candidate insertion points and Pair-Wise Exchange (Guan & Cheung 2004).
3. `hma_xu2025`: Xu et al. (IEEE TITS 2025) Heuristic Memetic Algorithm (HMA) with
   3-Stage Heuristic Decoding (Initial Berth -> Time-Variant Quay Crane Assignment ->
   Horizontal Berth Offset Rectification) and Target-Guided Local Search (TGLS).
4. `i2hcso_zou2025`: Zou et al. (IEEE TETCI 2025) Swarm Optimization with Intra- and
   Inter-Hierarchical Competition + Local Neighborhood [-eps, +eps] Berth/QC Repair.
5. `alns_martin2024`: Martin-Iradi, Pacino & Ropke (EJOR 2024) Adaptive Large
   Neighborhood Search (Shaw Removal, Cost-Time Removal, Regret-k Insertion,
   Strict-Packing Insertion, and Ejection-Chain Local Search).
6. `reevo_ub_bb_chen2026`: Chen et al. (Ocean Engineering 2026) Initial-Bound
   Optimization + Branch-and-Bound (UB-BB) with Analytical Lower Bound Eq. (19).
7. `reevo_vrpagent_bap`: Proposed Flagship Framework — ReEvo-Evolved Spatiotemporal
   Priority & Hole-Tightness Scoring + Xu et al. (2025) 3-Stage Time-Variant QC
   Decoding + VRPAgent-Discovered Destroy/Repair Portfolio + Chen et al. (2026)
   Initial-Bound Branch-and-Bound Refinement.
"""

from __future__ import annotations

import copy
import math
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from problems.berth_allocation.model import (
    BerthInstance,
    BerthSchedule,
    ScheduledVessel,
    Vessel,
    compute_analytical_lower_bound,
    compute_vessel_cost,
    effective_handling_time,
    rectangles_conflict,
    spatial_clearance,
    verify_schedule,
)

PriorityScoringFn = Callable[[Vessel, BerthInstance, float, float], float]
InsertionScoringFn = Callable[
    [Vessel, BerthInstance, float, float, float, List[ScheduledVessel]], float
]


# ---------------------------------------------------------------------------
# 1. Core 2D Spatiotemporal & Time-Variant QC Decoding Primitives
# ---------------------------------------------------------------------------


def _earliest_feasible_start_at_berth(
    vessel: Vessel,
    berth_pos: float,
    duration: float,
    placed: Sequence[ScheduledVessel],
    instance: BerthInstance,
    v_lookup: Dict[int, Vessel],
) -> float:
    """
    Given a fixed horizontal berth position `berth_pos` and service `duration`,
    returns the earliest start time t >= vessel.arrival_time such that the
    rectangle [berth_pos, berth_pos + length] x [t, t + duration] does not
    overlap any already-placed vessel (including safety clearance).
    Follows Proposition 3.1 of Lee, Chen & Cao (2010) and Algorithm 2 of Xu et al. (2025).
    """
    t = vessel.arrival_time
    horizontally_overlapping: List[ScheduledVessel] = []
    for sv in placed:
        other_v = v_lookup[sv.vessel_id]
        clr = spatial_clearance(vessel, other_v, instance)
        if not (
            berth_pos + vessel.length + clr <= sv.berth_pos + 1e-9
            or sv.berth_pos + other_v.length + clr <= berth_pos + 1e-9
        ):
            horizontally_overlapping.append(sv)

    if not horizontally_overlapping:
        return t

    horizontally_overlapping.sort(key=lambda s: s.start_time)
    changed = True
    while changed:
        changed = False
        end_t = t + duration
        for sv in horizontally_overlapping:
            if not (
                end_t + instance.safety_time_gap <= sv.start_time + 1e-9
                or sv.end_time + instance.safety_time_gap <= t + 1e-9
            ):
                new_t = sv.end_time + instance.safety_time_gap
                if new_t > t + 1e-9:
                    t = new_t
                    changed = True
                    break
    return t


def _build_used_cranes_profile(
    placed: Sequence[ScheduledVessel], max_horizon: int = 1200
) -> List[int]:
    used_cranes = [0] * max_horizon
    for sv in placed:
        t0 = int(round(sv.start_time))
        for idx, q_val in enumerate(sv.qc_profile):
            slot = t0 + idx
            if 0 <= slot < max_horizon:
                used_cranes[slot] += q_val
    return used_cranes


def _allocate_time_variant_cranes_and_start(
    vessel: Vessel,
    berth_pos: float,
    placed: Sequence[ScheduledVessel],
    instance: BerthInstance,
    v_lookup: Dict[int, Vessel],
    target_efficiency_bias: float = 0.85,
    precomputed_used_cranes: Optional[List[int]] = None,
) -> Tuple[float, float, List[int], float]:
    """
    Implements Stage 2 of Xu et al. (IEEE TITS 2025, Algorithm 3):
    Time-Variant Quay Crane Assignment.
    """
    if not instance.use_crane_allocation:
        dur = effective_handling_time(vessel, berth_pos, instance)
        t_s = _earliest_feasible_start_at_berth(
            vessel, berth_pos, dur, placed, instance, v_lookup
        )
        avg_q = 0.5 * (vessel.q_min + vessel.q_max)
        steps = max(1, int(round(dur)))
        return t_s, t_s + dur, [int(round(avg_q))] * steps, avg_q

    max_horizon = 1200
    used_cranes = (
        precomputed_used_cranes
        if precomputed_used_cranes is not None
        else _build_used_cranes_profile(placed, max_horizon)
    )

    dev_mult = 1.0 + instance.berth_deviation_factor * abs(
        berth_pos - vessel.preferred_berth
    )
    eff_workload = vessel.workload * dev_mult

    min_dur_est = math.ceil(eff_workload / vessel.q_max)
    t_cand = int(
        math.ceil(
            _earliest_feasible_start_at_berth(
                vessel, berth_pos, float(min_dur_est), placed, instance, v_lookup
            )
            - 1e-9
        )
    )

    horiz_conflicts: List[ScheduledVessel] = []
    for sv in placed:
        other_v = v_lookup[sv.vessel_id]
        clr = spatial_clearance(vessel, other_v, instance)
        if not (
            berth_pos + vessel.length + clr <= sv.berth_pos + 1e-9
            or sv.berth_pos + other_v.length + clr <= berth_pos + 1e-9
        ):
            horiz_conflicts.append(sv)

    desired_max_q = max(
        vessel.q_min,
        min(
            vessel.q_max,
            int(
                round(
                    vessel.q_min
                    + target_efficiency_bias * (vessel.q_max - vessel.q_min)
                )
            ),
        ),
    )

    while t_cand < max_horizon - 60:
        rem_work = eff_workload
        cur_slot = t_cand
        qc_prof: List[int] = []
        feasible = True

        while rem_work > 1e-6:
            if cur_slot >= max_horizon:
                feasible = False
                t_cand = max_horizon
                break
            slot_start = float(t_cand)
            slot_end = float(cur_slot + 1)
            for sv in horiz_conflicts:
                if not (
                    slot_end + instance.safety_time_gap <= sv.start_time + 1e-9
                    or sv.end_time + instance.safety_time_gap <= slot_start + 1e-9
                ):
                    feasible = False
                    t_cand = max(
                        t_cand + 1,
                        int(math.ceil(sv.end_time + instance.safety_time_gap - 1e-9)),
                    )
                    break
            if not feasible:
                break

            avail_q = instance.total_cranes - used_cranes[cur_slot]
            if avail_q < vessel.q_min:
                feasible = False
                t_cand = cur_slot + 1
                break

            cap_q = vessel.q_max if avail_q >= vessel.q_max + 2 else desired_max_q
            q_assign = min(avail_q, max(vessel.q_min, cap_q))
            qc_prof.append(int(q_assign))
            rem_work -= float(q_assign)
            cur_slot += 1

        if feasible and qc_prof:
            dur = float(len(qc_prof))
            avg_q = sum(qc_prof) / dur
            return float(t_cand), float(t_cand) + dur, qc_prof, avg_q

    dur = float(math.ceil(eff_workload / vessel.q_min))
    t_s = _earliest_feasible_start_at_berth(
        vessel, berth_pos, dur, placed, instance, v_lookup
    )
    steps = max(1, int(round(dur)))
    return t_s, t_s + dur, [vessel.q_min] * steps, float(vessel.q_min)


def rectify_berth_offset_stage3(
    sv: ScheduledVessel,
    vessel: Vessel,
    placed_others: Sequence[ScheduledVessel],
    instance: BerthInstance,
    v_lookup: Dict[int, Vessel],
) -> ScheduledVessel:
    """
    Implements Stage 3 of Xu et al. (IEEE TITS 2025, Algorithm 4):
    Horizontal Berth Offset Rectification.
    Slides `berth_pos` horizontally toward `vessel.preferred_berth` without
    colliding with any temporally overlapping vessel or violating crane workload.
    """
    if vessel.status == 0 and vessel.locked_berth is not None:
        return sv

    left_bound = 0.0
    right_bound = instance.quay_length - vessel.length

    for other in placed_others:
        if other.vessel_id == sv.vessel_id:
            continue
        if not (
            sv.end_time + instance.safety_time_gap <= other.start_time + 1e-9
            or other.end_time + instance.safety_time_gap <= sv.start_time + 1e-9
        ):
            other_v = v_lookup[other.vessel_id]
            clr = spatial_clearance(vessel, other_v, instance)
            if other.berth_pos + other_v.length <= sv.berth_pos + 1e-6:
                left_bound = max(left_bound, other.berth_pos + other_v.length + clr)
            elif other.berth_pos >= sv.berth_pos + vessel.length - 1e-6:
                right_bound = min(right_bound, other.berth_pos - clr - vessel.length)
            else:
                return sv

    if left_bound > right_bound + 1e-9:
        return sv

    step = instance.berth_step
    target = min(right_bound, max(left_bound, vessel.preferred_berth))
    rectified_b = round(target / step) * step
    if rectified_b < left_bound - 1e-9:
        rectified_b = math.ceil(left_bound / step - 1e-9) * step
    if rectified_b > right_bound + 1e-9:
        rectified_b = math.floor(right_bound / step + 1e-9) * step
    if rectified_b < left_bound - 1e-9 or rectified_b > right_bound + 1e-9:
        return sv

    if abs(rectified_b - sv.berth_pos) < 1e-6:
        return sv

    if instance.use_crane_allocation:
        # Verify delivered crane-hours still cover workload at rectified_b
        dev_mult = 1.0 + instance.berth_deviation_factor * abs(
            rectified_b - vessel.preferred_berth
        )
        if sum(sv.qc_profile) + 1e-6 < vessel.workload * dev_mult:
            return sv
        new_end = sv.end_time
    else:
        new_dur = effective_handling_time(vessel, rectified_b, instance)
        new_end = sv.start_time + new_dur

    # Double-check zero conflict with all other placed vessels
    for other in placed_others:
        if other.vessel_id == sv.vessel_id:
            continue
        other_v = v_lookup[other.vessel_id]
        if rectangles_conflict(
            rectified_b,
            vessel.length,
            sv.start_time,
            new_end,
            vessel,
            other,
            other_v,
            instance,
        ):
            return sv

    new_cost = compute_vessel_cost(
        vessel, rectified_b, sv.start_time, new_end, instance
    )
    return ScheduledVessel(
        vessel_id=sv.vessel_id,
        berth_pos=rectified_b,
        start_time=sv.start_time,
        end_time=new_end,
        assigned_cranes=sv.assigned_cranes,
        qc_profile=list(sv.qc_profile),
        waiting_time=max(0.0, sv.start_time - vessel.arrival_time),
        berth_deviation=abs(rectified_b - vessel.preferred_berth),
        weighted_cost=new_cost,
        length=vessel.length,
    )


def generate_candidate_berth_positions(
    vessel: Vessel,
    placed: Sequence[ScheduledVessel],
    instance: BerthInstance,
    v_lookup: Dict[int, Vessel],
    dense_grid: bool = False,
) -> List[float]:
    """
    Combines Proposition 3.1 of Lee, Chen & Cao (2010), preferred berth neighborhood,
    and spatial grid points along the quay.
    """
    max_b = instance.quay_length - vessel.length
    step = instance.berth_step
    cands = {0.0, max_b}

    pref_aligned = min(max_b, max(0.0, round(vessel.preferred_berth / step) * step))
    cands.add(pref_aligned)
    for mult in (-6, -4, -3, -2, -1, 1, 2, 3, 4, 6):
        b_nb = pref_aligned + mult * step
        if 0.0 <= b_nb <= max_b:
            cands.add(b_nb)

    for sv in placed:
        other_v = v_lookup[sv.vessel_id]
        clr = spatial_clearance(vessel, other_v, instance)
        right_edge = (
            math.ceil((sv.berth_pos + other_v.length + clr) / step - 1e-9) * step
        )
        if 0.0 <= right_edge <= max_b:
            cands.add(right_edge)
        left_edge = (
            math.floor((sv.berth_pos - clr - vessel.length) / step + 1e-9) * step
        )
        if 0.0 <= left_edge <= max_b:
            cands.add(left_edge)

    stride = max(step * 4, 25.0) if dense_grid else max(step * 6, 40.0)
    num_steps = int(round(max_b / stride))
    for k in range(num_steps + 1):
        b_grid = min(max_b, round((k * stride) / step) * step)
        cands.add(b_grid)

    return sorted(cands)


def best_insertion_for_vessel(
    vessel: Vessel,
    placed: Sequence[ScheduledVessel],
    instance: BerthInstance,
    v_lookup: Dict[int, Vessel],
    insertion_scorer: Optional[InsertionScoringFn] = None,
    dense_grid: bool = False,
    efficiency_bias: float = 0.85,
) -> ScheduledVessel:
    """
    Finds the best feasible 2D (berth, start_time, QC_profile) insertion for `vessel`
    given `placed` vessels, and applies Stage 3 Horizontal Berth Rectification.
    """
    if (
        vessel.status == 0
        and vessel.locked_berth is not None
        and vessel.locked_start is not None
    ):
        b0 = vessel.locked_berth
        t0 = vessel.locked_start
        locked_prof = getattr(vessel, "locked_qc_profile", None)
        locked_end = getattr(vessel, "locked_end", None)
        if locked_prof is not None and locked_end is not None:
            dur = float(locked_end - t0)
            qc_prof = list(locked_prof)
            avg_q = sum(qc_prof) / max(1, len(qc_prof))
        elif instance.use_crane_allocation:
            q_val = max(vessel.q_min, min(vessel.q_max, int(math.ceil(0.5 * (vessel.q_min + vessel.q_max)))))
            steps = max(1, int(math.ceil(vessel.workload / max(1, q_val))))
            dur = float(steps)
            qc_prof = [q_val] * steps
            avg_q = float(q_val)
        else:
            dur = effective_handling_time(vessel, b0, instance)
            avg_q = 0.5 * (vessel.q_min + vessel.q_max)
            steps = max(1, int(round(dur)))
            qc_prof = [int(round(avg_q))] * steps
        cost = compute_vessel_cost(vessel, b0, t0, t0 + dur, instance)
        return ScheduledVessel(
            vessel_id=vessel.id,
            berth_pos=b0,
            start_time=t0,
            end_time=t0 + dur,
            assigned_cranes=avg_q,
            qc_profile=qc_prof,
            waiting_time=max(0.0, t0 - vessel.arrival_time),
            berth_deviation=abs(b0 - vessel.preferred_berth),
            weighted_cost=cost,
            length=vessel.length,
        )

    cands = generate_candidate_berth_positions(
        vessel, placed, instance, v_lookup, dense_grid=dense_grid
    )
    best_sv: Optional[ScheduledVessel] = None
    best_score = float("inf")
    used_cranes_prof = (
        _build_used_cranes_profile(placed, 1200)
        if instance.use_crane_allocation
        else None
    )

    for b_pos in cands:
        t_s, t_e, qc_prof, avg_q = _allocate_time_variant_cranes_and_start(
            vessel,
            b_pos,
            placed,
            instance,
            v_lookup,
            target_efficiency_bias=efficiency_bias,
            precomputed_used_cranes=used_cranes_prof,
        )
        obj_cost = compute_vessel_cost(vessel, b_pos, t_s, t_e, instance)
        if insertion_scorer is not None:
            score = insertion_scorer(vessel, instance, b_pos, t_s, t_e, list(placed))
        else:
            score = obj_cost

        if score < best_score - 1e-9 or (
            abs(score - best_score) <= 1e-9
            and best_sv is not None
            and obj_cost < best_sv.weighted_cost
        ):
            best_score = score
            best_sv = ScheduledVessel(
                vessel_id=vessel.id,
                berth_pos=b_pos,
                start_time=t_s,
                end_time=t_e,
                assigned_cranes=avg_q,
                qc_profile=qc_prof,
                waiting_time=max(0.0, t_s - vessel.arrival_time),
                berth_deviation=abs(b_pos - vessel.preferred_berth),
                weighted_cost=obj_cost,
                length=vessel.length,
            )

    assert best_sv is not None
    best_sv = rectify_berth_offset_stage3(best_sv, vessel, placed, instance, v_lookup)
    return best_sv


def decode_sequence_to_schedule(
    vessel_order: Sequence[Vessel],
    instance: BerthInstance,
    solver_name: str,
    insertion_scorer: Optional[InsertionScoringFn] = None,
    dense_grid: bool = False,
    apply_stage3_pass: bool = True,
) -> BerthSchedule:
    """
    Decodes an ordered sequence of vessels into a feasible 2D BerthSchedule.
    Locked vessels (status == 0) are always placed first at their locked positions.
    """
    v_lookup = {v.id: v for v in instance.vessels}
    placed: List[ScheduledVessel] = []

    locked = [
        v
        for v in vessel_order
        if v.status == 0 and v.locked_berth is not None and v.locked_start is not None
    ]
    unlocked = [
        v
        for v in vessel_order
        if not (
            v.status == 0 and v.locked_berth is not None and v.locked_start is not None
        )
    ]

    for v in locked:
        placed.append(
            best_insertion_for_vessel(
                v, placed, instance, v_lookup, insertion_scorer, dense_grid=dense_grid
            )
        )
    for v in unlocked:
        placed.append(
            best_insertion_for_vessel(
                v, placed, instance, v_lookup, insertion_scorer, dense_grid=dense_grid
            )
        )

    if apply_stage3_pass:
        for idx in range(len(placed)):
            sv = placed[idx]
            v = v_lookup[sv.vessel_id]
            placed[idx] = rectify_berth_offset_stage3(
                sv, v, placed, instance, v_lookup
            )

    return _finalize_schedule(placed, instance, solver_name)


def _finalize_schedule(
    placed: Sequence[ScheduledVessel],
    instance: BerthInstance,
    solver_name: str,
    runtime_ms: float = 0.0,
) -> BerthSchedule:
    v_lookup = {v.id: v for v in instance.vessels}
    ordered = sorted(placed, key=lambda s: s.vessel_id)
    for s in ordered:
        s.length = v_lookup[s.vessel_id].length
    total_obj = round(sum(s.weighted_cost for s in ordered), 4)
    total_wait = round(sum(s.waiting_time for s in ordered), 4)
    total_service = round(sum(s.end_time - s.start_time for s in ordered), 4)
    total_dev = round(sum(s.berth_deviation for s in ordered), 4)
    makespan = round(max((s.end_time for s in ordered), default=0.0), 4)
    lb = compute_analytical_lower_bound(instance)
    gap = round(max(0.0, (total_obj - lb) / max(1e-6, lb) * 100.0), 3)

    sched = BerthSchedule(
        instance_name=instance.name,
        solver_name=solver_name,
        scheduled_vessels=ordered,
        total_objective=total_obj,
        total_waiting_time=total_wait,
        total_service_time=total_service,
        total_berth_deviation=total_dev,
        makespan=makespan,
        lower_bound=lb,
        optimality_gap_pct=gap,
        runtime_ms=round(runtime_ms, 3),
        feasible=True,
        violations=[],
    )
    ok, viols = verify_schedule(instance, sched)
    sched.feasible = ok
    sched.violations = viols
    return sched


# ---------------------------------------------------------------------------
# 2. Local Search Operators: TGLS (Xu et al. 2025) & Ejection Chain (Martin-Iradi 2024)
# ---------------------------------------------------------------------------


def target_guided_local_search(
    schedule: BerthSchedule,
    instance: BerthInstance,
    insertion_scorer: Optional[InsertionScoringFn] = None,
    max_passes: int = 2,
) -> BerthSchedule:
    """
    Implements Target-Guided Local Search (TGLS, Xu et al. IEEE TITS 2025 Section IV-E)
    combined with Ejection-Reinsertion (Martin-Iradi et al. EJOR 2024 Algorithm 4).
    """
    v_lookup = {v.id: v for v in instance.vessels}
    best_sched = schedule

    for _ in range(max_passes):
        improved = False
        ranked = sorted(
            [
                sv
                for sv in best_sched.scheduled_vessels
                if not (
                    v_lookup[sv.vessel_id].status == 0
                    and v_lookup[sv.vessel_id].locked_berth is not None
                )
            ],
            key=lambda s: s.weighted_cost,
            reverse=True,
        )

        for sv_target in ranked[: min(7, len(ranked))]:
            others = [
                s
                for s in best_sched.scheduled_vessels
                if s.vessel_id != sv_target.vessel_id
            ]
            v_target = v_lookup[sv_target.vessel_id]
            new_sv = best_insertion_for_vessel(
                v_target,
                others,
                instance,
                v_lookup,
                insertion_scorer=insertion_scorer,
                dense_grid=True,
            )
            if new_sv.weighted_cost + 1e-6 < sv_target.weighted_cost:
                cand_list = others + [new_sv]
                for idx in range(len(cand_list)):
                    s_i = cand_list[idx]
                    cand_list[idx] = rectify_berth_offset_stage3(
                        s_i, v_lookup[s_i.vessel_id], cand_list, instance, v_lookup
                    )
                cand_sched = _finalize_schedule(
                    cand_list, instance, best_sched.solver_name
                )
                if (
                    cand_sched.feasible
                    and cand_sched.total_objective + 1e-6 < best_sched.total_objective
                ):
                    best_sched = cand_sched
                    improved = True

        target_ids = {sv.vessel_id for sv in ranked[:4]}
        cur_order = [
            v_lookup[s.vessel_id]
            for s in sorted(
                best_sched.scheduled_vessels,
                key=lambda s: (s.start_time, -v_lookup[s.vessel_id].weight),
            )
        ]
        n = len(cur_order)
        for i in range(n):
            if cur_order[i].id not in target_ids:
                continue
            if cur_order[i].status == 0 and cur_order[i].locked_berth is not None:
                continue
            for j in range(max(0, i - 2), min(n, i + 3)):
                if i == j:
                    continue
                if cur_order[j].status == 0 and cur_order[j].locked_berth is not None:
                    continue
                if abs(cur_order[i].arrival_time - cur_order[j].arrival_time) > 8.0:
                    continue
                trial_order = list(cur_order)
                trial_order[i], trial_order[j] = trial_order[j], trial_order[i]
                cand_sched = decode_sequence_to_schedule(
                    trial_order,
                    instance,
                    best_sched.solver_name,
                    insertion_scorer=insertion_scorer,
                    dense_grid=False,
                )
                if (
                    cand_sched.feasible
                    and cand_sched.total_objective + 1e-6 < best_sched.total_objective
                ):
                    best_sched = cand_sched
                    cur_order = trial_order
                    improved = True
                    break
        if not improved:
            break

    return best_sched


# ---------------------------------------------------------------------------
# 3. Baseline 1: FIFP / FCFS Greedy (Chen et al. 2026 / Xu et al. 2025 / Zou et al. 2025)
# ---------------------------------------------------------------------------


def solve_fifp_greedy(instance: BerthInstance) -> BerthSchedule:
    """
    First-In-First-Planning (FIFP / FCFS) + Earliest Start Greedy Insertion.
    """
    t0 = time.perf_counter()
    v_lookup = {v.id: v for v in instance.vessels}
    ordered = sorted(
        instance.vessels,
        key=lambda v: (
            0 if (v.status == 0 and v.locked_berth is not None) else 1,
            v.arrival_time,
            v.id,
        ),
    )

    def fifp_scorer(
        v: Vessel,
        inst: BerthInstance,
        b_pos: float,
        t_s: float,
        t_e: float,
        _placed: List[ScheduledVessel],
    ) -> float:
        wait = max(0.0, t_s - v.arrival_time)
        dev = abs(b_pos - v.preferred_berth)
        return 100.0 * wait + 10.0 * (t_e - t_s) + 0.15 * dev + 0.01 * b_pos

    placed: List[ScheduledVessel] = []
    for v in ordered:
        placed.append(
            best_insertion_for_vessel(
                v,
                placed,
                instance,
                v_lookup,
                insertion_scorer=fifp_scorer,
                dense_grid=False,
                efficiency_bias=0.65,
            )
        )

    elapsed = (time.perf_counter() - t0) * 1000.0
    return _finalize_schedule(placed, instance, "fifp_greedy", elapsed)


# ---------------------------------------------------------------------------
# 4. Baseline 2: GRASP_2 (Lee, Chen & Cao, Transp. Res. Part E 2010)
# ---------------------------------------------------------------------------


def solve_grasp_lee2010(instance: BerthInstance, seed: int = 42) -> BerthSchedule:
    """
    Implements GRASP_2 from Lee, Chen & Cao (Transp. Res. Part E 46, 2010).
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    best_sched: Optional[BerthSchedule] = None

    unlocked_base = [
        v
        for v in instance.vessels
        if not (v.status == 0 and v.locked_berth is not None)
    ]
    locked_base = [
        v for v in instance.vessels if v.status == 0 and v.locked_berth is not None
    ]

    n_restarts = 4
    rcl_alpha = 0.25

    for r in range(n_restarts):
        if r == 0:
            seq = sorted(
                unlocked_base,
                key=lambda v: v.arrival_time - 0.8 * v.weight,
            )
        else:
            rem = list(unlocked_base)
            seq = []
            cur_clock = 0.0
            while rem:
                scores = []
                for v in rem:
                    wait_est = max(0.0, cur_clock - v.arrival_time)
                    urgency = (v.weight * (1.0 + 0.2 * wait_est)) / (
                        1.0 + max(0.0, v.arrival_time - cur_clock)
                    )
                    scores.append((urgency, v))
                scores.sort(key=lambda x: x[0], reverse=True)
                max_s = scores[0][0]
                min_s = scores[-1][0]
                thresh = max_s - rcl_alpha * (max_s - min_s)
                rcl = [v for s, v in scores if s >= thresh - 1e-9]
                chosen = rng.choice(rcl)
                seq.append(chosen)
                rem.remove(chosen)
                cur_clock = max(cur_clock, chosen.arrival_time) + 0.3

        cand = decode_sequence_to_schedule(
            locked_base + seq,
            instance,
            "grasp_lee2010",
            dense_grid=False,
            apply_stage3_pass=False,
        )
        if best_sched is None or cand.total_objective < best_sched.total_objective:
            best_sched = cand

    assert best_sched is not None
    best_sched = target_guided_local_search(best_sched, instance, max_passes=1)
    best_sched.runtime_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    best_sched.solver_name = "grasp_lee2010"
    return best_sched


# ---------------------------------------------------------------------------
# 5. Baseline 3: HMA (Xu et al., IEEE TITS 2025)
# ---------------------------------------------------------------------------


def solve_hma_xu2025(instance: BerthInstance, seed: int = 42) -> BerthSchedule:
    """
    Implements the Heuristic Memetic Algorithm (HMA) of Xu et al. (IEEE TITS 2025).
    """
    t0 = time.perf_counter()
    rng = random.Random(seed + 101)
    locked = [
        v for v in instance.vessels if v.status == 0 and v.locked_berth is not None
    ]
    unlocked = [
        v
        for v in instance.vessels
        if not (v.status == 0 and v.locked_berth is not None)
    ]

    pop_size = 5
    generations = 2
    population: List[Tuple[List[Vessel], BerthSchedule]] = []

    seed_orders = [
        sorted(unlocked, key=lambda v: (v.arrival_time, -v.weight)),
        sorted(
            unlocked,
            key=lambda v: v.arrival_time - 0.6 * v.weight - 0.002 * v.length,
        ),
        sorted(
            unlocked,
            key=lambda v: (v.arrival_time + 0.15 * v.handling_time, -v.workload),
        ),
    ]
    while len(seed_orders) < pop_size:
        base = list(seed_orders[len(seed_orders) % 3])
        for _ in range(max(1, len(base) // 5)):
            i = rng.randrange(len(base))
            j = min(len(base) - 1, max(0, i + rng.choice([-2, -1, 1, 2])))
            base[i], base[j] = base[j], base[i]
        seed_orders.append(base)

    for order in seed_orders:
        sched = decode_sequence_to_schedule(
            locked + order,
            instance,
            "hma_xu2025",
            dense_grid=False,
            apply_stage3_pass=True,
        )
        population.append((order, sched))

    population.sort(key=lambda x: x[1].total_objective)

    def ox_crossover(p1: List[Vessel], p2: List[Vessel]) -> List[Vessel]:
        n = len(p1)
        if n <= 2:
            return list(p1)
        a, b = sorted(rng.sample(range(n), 2))
        slice_ids = {v.id for v in p1[a : b + 1]}
        child: List[Optional[Vessel]] = [None] * n
        for idx in range(a, b + 1):
            child[idx] = p1[idx]
        fill = [v for v in p2 if v.id not in slice_ids]
        f_idx = 0
        for idx in range(n):
            if child[idx] is None:
                child[idx] = fill[f_idx]
                f_idx += 1
        return [v for v in child if v is not None]

    for _gen in range(generations):
        offspring: List[Tuple[List[Vessel], BerthSchedule]] = []
        for _ in range(pop_size // 2):
            p1 = population[rng.randrange(min(4, len(population)))][0]
            p2 = population[rng.randrange(len(population))][0]
            child_seq = ox_crossover(p1, p2)
            if rng.random() < 0.45 and len(child_seq) >= 2:
                i = rng.randrange(len(child_seq))
                j = min(len(child_seq) - 1, max(0, i + rng.choice([-2, -1, 1, 2])))
                child_seq[i], child_seq[j] = child_seq[j], child_seq[i]
            child_sched = decode_sequence_to_schedule(
                locked + child_seq,
                instance,
                "hma_xu2025",
                dense_grid=False,
                apply_stage3_pass=True,
            )
            offspring.append((child_seq, child_sched))

        population.extend(offspring)
        population.sort(key=lambda x: x[1].total_objective)
        population = population[:pop_size]

    best_sched = target_guided_local_search(
        population[0][1], instance, max_passes=1
    )
    best_sched.runtime_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    best_sched.solver_name = "hma_xu2025"
    return best_sched


# ---------------------------------------------------------------------------
# 6. Baseline 4: I2HCSO (Zou et al., IEEE TETCI 2025)
# ---------------------------------------------------------------------------


def solve_i2hcso_zou2025(instance: BerthInstance, seed: int = 42) -> BerthSchedule:
    """
    Implements Swarm Optimization With Intra- and Inter-Hierarchical Competition
    (I2HCSO, Zou et al., IEEE TETCI 2025).
    """
    t0 = time.perf_counter()
    rng = random.Random(seed + 202)
    locked = [
        v for v in instance.vessels if v.status == 0 and v.locked_berth is not None
    ]
    unlocked = [
        v
        for v in instance.vessels
        if not (v.status == 0 and v.locked_berth is not None)
    ]
    n_u = len(unlocked)

    swarm_size = 6
    iterations = 2
    positions: List[List[float]] = []
    velocities: List[List[float]] = []

    for p_idx in range(swarm_size):
        vec: List[float] = []
        for v in unlocked:
            base_key = (
                v.arrival_time
                - (0.4 + 0.15 * (p_idx % 4)) * v.weight
                - 0.0015 * v.length
                + (rng.uniform(-0.8, 0.8) if p_idx > 0 else 0.0)
            )
            vec.append(base_key)
        positions.append(vec)
        velocities.append([0.0] * n_u)

    def eval_particle(pos_vec: List[float]) -> BerthSchedule:
        paired = sorted(zip(pos_vec, unlocked), key=lambda x: (x[0], x[1].id))
        order = [v for _, v in paired]
        return decode_sequence_to_schedule(
            locked + order,
            instance,
            "i2hcso_zou2025",
            dense_grid=False,
            apply_stage3_pass=True,
        )

    evaluated = [eval_particle(pos) for pos in positions]
    best_idx = min(range(swarm_size), key=lambda idx: evaluated[idx].total_objective)
    global_best_pos = list(positions[best_idx])
    global_best_sched = evaluated[best_idx]

    for _it in range(iterations):
        order_idx = sorted(
            range(swarm_size), key=lambda idx: evaluated[idx].total_objective
        )
        half = swarm_size // 2
        winners = order_idx[:half]
        losers = order_idx[half:]

        for k in range(half):
            w_i = winners[k]
            l_i = losers[k]
            elite_i = order_idx[0]
            for d in range(n_u):
                r1, r2, r3 = rng.random(), rng.random(), rng.random()
                velocities[l_i][d] = (
                    r1 * velocities[l_i][d]
                    + r2 * (positions[w_i][d] - positions[l_i][d])
                    + 0.5 * r3 * (global_best_pos[d] - positions[l_i][d])
                )
                positions[l_i][d] += velocities[l_i][d]
                if w_i != elite_i:
                    velocities[w_i][d] = 0.4 * velocities[w_i][d] + 0.6 * rng.random() * (
                        global_best_pos[d] - positions[w_i][d]
                    )
                    positions[w_i][d] += velocities[w_i][d]

            evaluated[l_i] = eval_particle(positions[l_i])
            if evaluated[l_i].total_objective < global_best_sched.total_objective:
                global_best_sched = evaluated[l_i]
                global_best_pos = list(positions[l_i])

            if w_i != elite_i:
                evaluated[w_i] = eval_particle(positions[w_i])
                if evaluated[w_i].total_objective < global_best_sched.total_objective:
                    global_best_sched = evaluated[w_i]
                    global_best_pos = list(positions[w_i])

    global_best_sched = target_guided_local_search(
        global_best_sched, instance, max_passes=1
    )
    global_best_sched.runtime_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    global_best_sched.solver_name = "i2hcso_zou2025"
    return global_best_sched


# ---------------------------------------------------------------------------
# 7. Baseline 5: ALNS (Martin-Iradi, Pacino & Ropke, EJOR 2024)
# ---------------------------------------------------------------------------


def _shaw_relatedness(
    sv_i: ScheduledVessel,
    sv_j: ScheduledVessel,
    v_i: Vessel,
    v_j: Vessel,
    instance: BerthInstance,
) -> float:
    time_diff = abs(sv_i.start_time - sv_j.start_time) / max(
        1.0, instance.quay_length / 50.0
    )
    berth_diff = abs(sv_i.berth_pos - sv_j.berth_pos) / instance.quay_length
    len_diff = abs(v_i.length - v_j.length) / instance.quay_length
    weight_diff = abs(v_i.weight - v_j.weight)
    return 0.40 * time_diff + 0.35 * berth_diff + 0.15 * len_diff + 0.10 * weight_diff


def _alns_destroy_and_repair(
    current_sched: BerthSchedule,
    instance: BerthInstance,
    destroy_mode: str,
    repair_mode: str,
    rng: random.Random,
    insertion_scorer: Optional[InsertionScoringFn] = None,
) -> BerthSchedule:
    v_lookup = {v.id: v for v in instance.vessels}
    locked_svs = [
        sv
        for sv in current_sched.scheduled_vessels
        if v_lookup[sv.vessel_id].status == 0
        and v_lookup[sv.vessel_id].locked_berth is not None
    ]
    unlocked_svs = [
        sv
        for sv in current_sched.scheduled_vessels
        if not (
            v_lookup[sv.vessel_id].status == 0
            and v_lookup[sv.vessel_id].locked_berth is not None
        )
    ]

    if len(unlocked_svs) <= 2:
        return current_sched

    q_remove = max(3, min(8, int(round(0.28 * len(unlocked_svs)))))
    removed_ids = set()

    if destroy_mode == "shaw":
        by_cost = sorted(unlocked_svs, key=lambda s: s.weighted_cost, reverse=True)
        seed_sv = by_cost[rng.randrange(min(4, len(by_cost)))]
        removed_ids.add(seed_sv.vessel_id)
        seed_v = v_lookup[seed_sv.vessel_id]
        rels = [
            (
                _shaw_relatedness(
                    seed_sv, other_sv, seed_v, v_lookup[other_sv.vessel_id], instance
                ),
                other_sv.vessel_id,
            )
            for other_sv in unlocked_svs
            if other_sv.vessel_id != seed_sv.vessel_id
        ]
        rels.sort(key=lambda x: x[0])
        for _, vid in rels[: q_remove - 1]:
            removed_ids.add(vid)

    elif destroy_mode == "cost_time":
        by_excess = sorted(
            unlocked_svs,
            key=lambda s: (
                s.weighted_cost
                - compute_vessel_cost(
                    v_lookup[s.vessel_id],
                    v_lookup[s.vessel_id].preferred_berth,
                    v_lookup[s.vessel_id].arrival_time,
                    v_lookup[s.vessel_id].arrival_time
                    + v_lookup[s.vessel_id].handling_time,
                    instance,
                )
            )
            * rng.uniform(0.85, 1.15),
            reverse=True,
        )
        for sv in by_excess[:q_remove]:
            removed_ids.add(sv.vessel_id)

    else:
        center_sv = rng.choice(unlocked_svs)
        dists = [
            (
                abs(sv.start_time - center_sv.start_time)
                + 0.015 * abs(sv.berth_pos - center_sv.berth_pos),
                sv.vessel_id,
            )
            for sv in unlocked_svs
        ]
        dists.sort(key=lambda x: x[0])
        for _, vid in dists[:q_remove]:
            removed_ids.add(vid)

    partial_placed: List[ScheduledVessel] = list(locked_svs)
    kept_unlocked = sorted(
        [sv for sv in unlocked_svs if sv.vessel_id not in removed_ids],
        key=lambda s: (s.start_time, s.berth_pos),
    )
    for sv in kept_unlocked:
        v = v_lookup[sv.vessel_id]
        partial_placed.append(
            best_insertion_for_vessel(
                v,
                partial_placed,
                instance,
                v_lookup,
                insertion_scorer=insertion_scorer,
                dense_grid=False,
            )
        )

    unassigned = [v_lookup[vid] for vid in removed_ids]

    if repair_mode == "regret_2":
        while unassigned:
            best_v: Optional[Vessel] = None
            best_sv_choice: Optional[ScheduledVessel] = None
            best_regret = -float("inf")

            used_cranes_prof = (
                _build_used_cranes_profile(partial_placed, 1200)
                if instance.use_crane_allocation
                else None
            )
            for v in unassigned:
                cands = generate_candidate_berth_positions(
                    v, partial_placed, instance, v_lookup, dense_grid=False
                )
                evals: List[Tuple[float, ScheduledVessel]] = []
                for b_pos in cands:
                    t_s, t_e, qc_prof, avg_q = _allocate_time_variant_cranes_and_start(
                        v,
                        b_pos,
                        partial_placed,
                        instance,
                        v_lookup,
                        precomputed_used_cranes=used_cranes_prof,
                    )
                    cost = compute_vessel_cost(v, b_pos, t_s, t_e, instance)
                    score = (
                        insertion_scorer(v, instance, b_pos, t_s, t_e, partial_placed)
                        if insertion_scorer is not None
                        else cost
                    )
                    evals.append(
                        (
                            score,
                            ScheduledVessel(
                                vessel_id=v.id,
                                berth_pos=b_pos,
                                start_time=t_s,
                                end_time=t_e,
                                assigned_cranes=avg_q,
                                qc_profile=qc_prof,
                                waiting_time=max(0.0, t_s - v.arrival_time),
                                berth_deviation=abs(b_pos - v.preferred_berth),
                                weighted_cost=cost,
                                length=v.length,
                            ),
                        )
                    )
                evals.sort(key=lambda x: x[0])
                best_score_1, sv_1 = evals[0]
                second_score = best_score_1 + 5.0
                for sc, sv_alt in evals[1:]:
                    if abs(sv_alt.berth_pos - sv_1.berth_pos) >= instance.berth_step * 2:
                        second_score = sc
                        break
                regret_val = (
                    (second_score - best_score_1)
                    + 1.5 * v.weight
                    - 0.35 * v.arrival_time
                )
                if regret_val > best_regret:
                    best_regret = regret_val
                    best_v = v
                    best_sv_choice = rectify_berth_offset_stage3(
                        sv_1, v, partial_placed, instance, v_lookup
                    )

            assert best_v is not None and best_sv_choice is not None
            partial_placed.append(best_sv_choice)
            unassigned.remove(best_v)
    else:
        unassigned.sort(
            key=lambda v: (v.arrival_time - 0.75 * v.weight - 0.002 * v.length)
        )
        for v in unassigned:
            partial_placed.append(
                best_insertion_for_vessel(
                    v,
                    partial_placed,
                    instance,
                    v_lookup,
                    insertion_scorer=insertion_scorer,
                    dense_grid=True,
                )
            )

    for idx in range(len(partial_placed)):
        s_i = partial_placed[idx]
        partial_placed[idx] = rectify_berth_offset_stage3(
            s_i, v_lookup[s_i.vessel_id], partial_placed, instance, v_lookup
        )

    return _finalize_schedule(partial_placed, instance, current_sched.solver_name)


def solve_alns_martin2024(instance: BerthInstance, seed: int = 42) -> BerthSchedule:
    """
    Implements the Adaptive Large Neighborhood Search (ALNS) heuristic of
    Martin-Iradi, Pacino & Ropke (EJOR 316, 2024, 152-167).
    """
    t0 = time.perf_counter()
    rng = random.Random(seed + 303)

    init_order = sorted(
        instance.vessels,
        key=lambda v: (
            0 if (v.status == 0 and v.locked_berth is not None) else 1,
            v.arrival_time - 0.5 * v.weight,
            v.id,
        ),
    )
    current = decode_sequence_to_schedule(
        init_order, instance, "alns_martin2024", dense_grid=False
    )
    best = current

    destroy_ops = ["shaw", "cost_time", "spatiotemporal_cluster"]
    repair_ops = ["regret_2", "strict_packing"]
    d_weights = {op: 1.0 for op in destroy_ops}
    r_weights = {op: 1.0 for op in repair_ops}

    temp = max(1.0, 0.02 * best.total_objective)
    cooling = 0.85

    for _ in range(5):
        d_op = rng.choices(destroy_ops, weights=[d_weights[o] for o in destroy_ops])[0]
        r_op = rng.choices(repair_ops, weights=[r_weights[o] for o in repair_ops])[0]

        cand = _alns_destroy_and_repair(
            current, instance, d_op, r_op, rng, insertion_scorer=None
        )
        if cand.total_objective + 1e-6 < best.total_objective:
            best = cand
            current = cand
            d_weights[d_op] += 0.5
            r_weights[r_op] += 0.5
        elif cand.total_objective + 1e-6 < current.total_objective:
            current = cand
            d_weights[d_op] += 0.2
            r_weights[r_op] += 0.2
        else:
            delta = cand.total_objective - current.total_objective
            if rng.random() < math.exp(-delta / max(1e-6, temp)):
                current = cand
        temp *= cooling

    best = target_guided_local_search(best, instance, max_passes=1)
    best.runtime_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    best.solver_name = "alns_martin2024"
    return best


# ---------------------------------------------------------------------------
# 8. Baseline 6: Chen et al. (Ocean Engineering 2026) Initial-Bound BB (`UB-BB`)
# ---------------------------------------------------------------------------


def _branch_and_bound_window_polish(
    schedule: BerthSchedule,
    instance: BerthInstance,
    insertion_scorer: Optional[InsertionScoringFn] = None,
    window_size: int = 4,
    max_nodes: int = 32,
) -> BerthSchedule:
    """
    Implements the Initial-Bound Optimization + Branch-and-Bound (`UB-BB`) mechanism
    from Chen et al. (Ocean Engineering 364, 2026, Section 3.4.1 & Eq. 19).
    """
    v_lookup = {v.id: v for v in instance.vessels}
    ordered_svs = sorted(
        schedule.scheduled_vessels,
        key=lambda s: (
            0
            if (
                v_lookup[s.vessel_id].status == 0
                and v_lookup[s.vessel_id].locked_berth is not None
            )
            else 1,
            s.start_time,
            s.vessel_id,
        ),
    )

    locked_count = sum(
        1
        for s in ordered_svs
        if v_lookup[s.vessel_id].status == 0
        and v_lookup[s.vessel_id].locked_berth is not None
    )
    n_total = len(ordered_svs)
    if n_total - locked_count <= 2:
        return schedule

    w_len = min(window_size, n_total - locked_count)
    best_w_start = locked_count
    best_w_excess = -1.0

    for start_idx in range(locked_count, n_total - w_len + 1):
        excess = 0.0
        for k in range(start_idx, start_idx + w_len):
            sv = ordered_svs[k]
            v = v_lookup[sv.vessel_id]
            ideal = compute_vessel_cost(
                v,
                v.preferred_berth,
                v.arrival_time,
                v.arrival_time + v.handling_time,
                instance,
            )
            excess += sv.weighted_cost - ideal
        if excess > best_w_excess:
            best_w_excess = excess
            best_w_start = start_idx

    prefix_svs = ordered_svs[:best_w_start]
    window_vessels = [
        v_lookup[s.vessel_id]
        for s in ordered_svs[best_w_start : best_w_start + w_len]
    ]
    suffix_vessels = [
        v_lookup[s.vessel_id] for s in ordered_svs[best_w_start + w_len :]
    ]

    prefix_cost = sum(s.weighted_cost for s in prefix_svs)
    ideal_vessel_lb = {
        v.id: compute_vessel_cost(
            v,
            v.preferred_berth,
            v.arrival_time,
            v.arrival_time
            + (
                float(math.ceil(v.workload / v.q_max))
                if instance.use_crane_allocation
                else v.handling_time
            ),
            instance,
        )
        for v in instance.vessels
    }
    suffix_lb = sum(ideal_vessel_lb[v.id] for v in suffix_vessels)

    best_ub = schedule.total_objective
    best_sched = schedule
    nodes_visited = 0
    leaves_evaluated = 0

    def dfs(
        rem_window: List[Vessel],
        placed_so_far: List[ScheduledVessel],
        acc_cost: float,
    ) -> None:
        nonlocal best_ub, best_sched, nodes_visited, leaves_evaluated
        if nodes_visited >= max_nodes or leaves_evaluated >= 3:
            return
        nodes_visited += 1

        rem_lb = sum(ideal_vessel_lb[v.id] for v in rem_window) + suffix_lb
        if acc_cost + rem_lb >= best_ub - 1e-6:
            return

        if not rem_window:
            leaves_evaluated += 1
            full_placed = list(placed_so_far)
            for sv_v in suffix_vessels:
                sv_ins = best_insertion_for_vessel(
                    sv_v,
                    full_placed,
                    instance,
                    v_lookup,
                    insertion_scorer=insertion_scorer,
                    dense_grid=False,
                )
                full_placed.append(sv_ins)
                if sum(x.weighted_cost for x in full_placed) >= best_ub - 1e-6:
                    return
            for idx in range(len(full_placed)):
                s_i = full_placed[idx]
                full_placed[idx] = rectify_berth_offset_stage3(
                    s_i, v_lookup[s_i.vessel_id], full_placed, instance, v_lookup
                )
            cand_sched = _finalize_schedule(
                full_placed, instance, schedule.solver_name
            )
            if cand_sched.feasible and cand_sched.total_objective < best_ub - 1e-6:
                best_ub = cand_sched.total_objective
                best_sched = cand_sched
            return

        candidates_to_branch = sorted(
            rem_window,
            key=lambda v: v.arrival_time - 0.7 * v.weight - 0.002 * v.length,
        )
        for v_next in candidates_to_branch:
            next_rem = [v for v in rem_window if v.id != v_next.id]
            sv_ins = best_insertion_for_vessel(
                v_next,
                placed_so_far,
                instance,
                v_lookup,
                insertion_scorer=insertion_scorer,
                dense_grid=False,
            )
            dfs(
                next_rem,
                placed_so_far + [sv_ins],
                acc_cost + sv_ins.weighted_cost,
            )

    dfs(window_vessels, list(prefix_svs), prefix_cost)
    return best_sched


def solve_reevo_ub_bb_chen2026(instance: BerthInstance) -> BerthSchedule:
    """
    Implements Chen et al. (Ocean Engineering 364, 2026) Initial-Bound Optimization
    + Branch-and-Bound (`UB-BB`).
    """
    t0 = time.perf_counter()
    policy_order = sorted(
        instance.vessels,
        key=lambda v: (
            0 if (v.status == 0 and v.locked_berth is not None) else 1,
            v.arrival_time - 0.65 * v.weight - 0.0018 * v.length,
            v.id,
        ),
    )
    ub_sched = decode_sequence_to_schedule(
        policy_order,
        instance,
        "reevo_ub_bb_chen2026",
        dense_grid=True,
        apply_stage3_pass=True,
    )
    ub_sched = target_guided_local_search(ub_sched, instance, max_passes=1)
    refined = _branch_and_bound_window_polish(
        ub_sched, instance, window_size=3, max_nodes=10
    )
    refined.runtime_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    refined.solver_name = "reevo_ub_bb_chen2026"
    return refined


# ---------------------------------------------------------------------------
# 9. Proposed Flagship: ReEvo-VRPAgent Unified Berth Allocation Framework
# ---------------------------------------------------------------------------


def reevo_spatiotemporal_insertion_scorer(
    vessel: Vessel,
    instance: BerthInstance,
    berth_pos: float,
    start_time: float,
    end_time: float,
    placed: List[ScheduledVessel],
) -> float:
    """
    ReEvo-Evolved Spatiotemporal 2D Packing & Lookahead Insertion Scorer.
    """
    base_cost = compute_vessel_cost(vessel, berth_pos, start_time, end_time, instance)

    min_ship_len = 110.0
    left_gap = berth_pos
    right_gap = instance.quay_length - (berth_pos + vessel.length)
    frag_penalty = 0.0
    if 1e-3 < left_gap < min_ship_len:
        frag_penalty += (
            0.25 * instance.alpha_wait * (min_ship_len - left_gap) / min_ship_len
        )
    if 1e-3 < right_gap < min_ship_len:
        frag_penalty += (
            0.25 * instance.alpha_wait * (min_ship_len - right_gap) / min_ship_len
        )

    corner_touch_bonus = 0.0
    for sv in placed:
        if not (end_time <= sv.start_time or sv.end_time <= start_time):
            gap_l = berth_pos - (sv.berth_pos + sv.length)
            gap_r = sv.berth_pos - (berth_pos + vessel.length)
            if 0.0 <= gap_l <= 25.0 or 0.0 <= gap_r <= 25.0:
                corner_touch_bonus -= 0.12 * instance.alpha_wait
            elif (25.0 < gap_l < min_ship_len) or (25.0 < gap_r < min_ship_len):
                frag_penalty += 0.18 * instance.alpha_wait

    return base_cost + frag_penalty + corner_touch_bonus


def reevo_priority_key(vessel: Vessel, instance: BerthInstance) -> Tuple[int, float, int]:
    is_unlocked = 0 if (vessel.status == 0 and v_is_locked(vessel)) else 1
    mean_w = (
        sum(v.weight for v in instance.vessels) / max(1, len(instance.vessels))
    )
    w_norm = vessel.weight / max(1e-6, mean_w)
    crane_pressure = vessel.workload / max(1.0, float(vessel.q_max))
    score = (
        vessel.arrival_time
        - 0.85 * w_norm
        - 0.0022 * vessel.length
        - 0.08 * crane_pressure
    )
    return (is_unlocked, score, vessel.id)


def v_is_locked(v: Vessel) -> bool:
    return v.status == 0 and v.locked_berth is not None and v.locked_start is not None


def solve_reevo_vrpagent_bap(
    instance: BerthInstance,
    seed: int = 42,
    custom_insertion_scorer: Optional[InsertionScoringFn] = None,
    fast_rolling_mode: bool = False,
) -> BerthSchedule:
    """
    Proposed Flagship Solver: `reevo_vrpagent_bap` (100% Stateless & Standalone).

    Four-Phase Unified Architecture synthesizing all 6 papers + ReEvo + VRPAgent:
      Phase 1: Scale-Invariant ReEvo Multi-Attribute Priority Portfolio (releasing FIFP,
               Zou et al. 2025) decoded via 3-Stage Time-Variant QC + Stage-3 Berth Offset
               Rectification (Xu et al. 2025) & Proposition 3.1 exact vertices (Lee et al. 2010),
               scored by our ReEvo Spatiotemporal Hole-Tightness Scorer.
      Phase 2: Bottleneck-Driven Priority Elevation (VRPAgent-style targeted sequence
               exploration promoting highest excess-cost vessels across congestion waves).
      Phase 3: Adaptive Large Neighborhood Search (Martin-Iradi et al. 2024 + VRPAgent)
               with Shaw (Eq. 11), Cost-Time, and Spatiotemporal Cluster destroy operators
               paired with 2-Regret and Strict-Packing repair operators.
      Phase 4: Dual-Scorer Target-Guided Local Search (`TGLS`, Xu et al. 2025 + Guan & Cheung
               2004 pair-wise exchange) followed by Initial-Bound Branch-and-Bound (`UB-BB`,
               Chen et al. 2026 Eq. 19) exact window polish.
    """
    t0 = time.perf_counter()
    rng = random.Random(seed + 707)
    scorer = custom_insertion_scorer or reevo_spatiotemporal_insertion_scorer
    mean_w = sum(v.weight for v in instance.vessels) / max(1, len(instance.vessels))

    def _wn(v: Vessel) -> float:
        return v.weight / max(1e-6, mean_w)

    # --- Phase 1: Scale-Invariant ReEvo Priority Portfolio (Proposition 3.1 Vertex Decoding) ---
    candidate_specs = [
        (sorted(instance.vessels, key=lambda v: reevo_priority_key(v, instance)), scorer),
        (
            sorted(
                instance.vessels,
                key=lambda v: (
                    0 if v_is_locked(v) else 1,
                    v.arrival_time,
                    -v.weight,
                    v.id,
                ),
            ),
            None,
        ),
        (
            sorted(
                instance.vessels,
                key=lambda v: (
                    0 if v_is_locked(v) else 1,
                    v.arrival_time - 0.50 * _wn(v),
                    -v.length,
                    v.id,
                ),
            ),
            None,
        ),
    ]
    if not fast_rolling_mode:
        candidate_specs.extend(
            [
                (
                    sorted(
                        instance.vessels,
                        key=lambda v: (
                            0 if v_is_locked(v) else 1,
                            v.arrival_time - 0.65 * _wn(v) - 0.0018 * v.length,
                            v.id,
                        ),
                    ),
                    scorer,
                ),
                (
                    sorted(
                        instance.vessels,
                        key=lambda v: (
                            0 if v_is_locked(v) else 1,
                            v.arrival_time - 1.25 * _wn(v) + 0.08 * v.handling_time,
                            v.id,
                        ),
                    ),
                    scorer,
                ),
                (
                    sorted(
                        instance.vessels,
                        key=lambda v: (
                            0 if v_is_locked(v) else 1,
                            v.arrival_time + 0.15 * v.handling_time - 0.55 * _wn(v),
                            -v.workload,
                            v.id,
                        ),
                    ),
                    None,
                ),
            ]
        )

    best_sched: Optional[BerthSchedule] = None
    best_order: List[Vessel] = list(instance.vessels)
    for order, sc_fn in candidate_specs:
        cand = decode_sequence_to_schedule(
            order,
            instance,
            "reevo_vrpagent_bap",
            insertion_scorer=sc_fn,
            dense_grid=False,
            apply_stage3_pass=True,
        )
        if best_sched is None or cand.total_objective < best_sched.total_objective:
            best_sched = cand
            best_order = list(order)

    assert best_sched is not None

    # --- Phase 2: Bottleneck-Driven Priority Elevation (VRPAgent Targeted Search) ---
    if not fast_rolling_mode:
        locked_prefix = [v for v in best_order if v_is_locked(v)]
        unlocked_seq = [v for v in best_order if not v_is_locked(v)]
        n_u = len(unlocked_seq)
        if n_u >= 4:
            sv_by_id = {sv.vessel_id: sv for sv in best_sched.scheduled_vessels}
            excess_ranked = sorted(
                unlocked_seq,
                key=lambda v: sv_by_id[v.id].weighted_cost
                - compute_vessel_cost(
                    v,
                    v.preferred_berth,
                    v.arrival_time,
                    v.arrival_time
                    + (
                        float(math.ceil(v.workload / v.q_max))
                        if instance.use_crane_allocation
                        else v.handling_time
                    ),
                    instance,
                ),
                reverse=True,
            )
            for bottleneck_v in excess_ranked[: min(4, n_u)]:
                idx = next(
                    (k for k, x in enumerate(unlocked_seq) if x.id == bottleneck_v.id),
                    None,
                )
                if idx is None or idx == 0:
                    continue
                for shift in (1, 2, 3):
                    new_idx = max(0, idx - shift)
                    if new_idx == idx:
                        continue
                    trial_seq = list(unlocked_seq)
                    item = trial_seq.pop(idx)
                    trial_seq.insert(new_idx, item)
                    cand = decode_sequence_to_schedule(
                        locked_prefix + trial_seq,
                        instance,
                        "reevo_vrpagent_bap",
                        insertion_scorer=None,
                        dense_grid=False,
                        apply_stage3_pass=True,
                    )
                    if (
                        cand.feasible
                        and cand.total_objective + 1e-6 < best_sched.total_objective
                    ):
                        best_sched = cand
                        unlocked_seq = trial_seq
                        break

    # --- Phase 3: Adaptive Large Neighborhood Search (Shaw / Cost-Time / Cluster + Regret-2 / Strict) ---
    operator_pairs = (
        [("cost_time", "regret_2"), ("shaw", "strict_packing")]
        if fast_rolling_mode
        else [
            ("cost_time", "regret_2"),
            ("shaw", "strict_packing"),
            ("spatiotemporal_cluster", "regret_2"),
            ("cost_time", "strict_packing"),
            ("shaw", "regret_2"),
            ("spatiotemporal_cluster", "strict_packing"),
        ]
    )
    current = best_sched
    temp = max(1.0, 0.015 * best_sched.total_objective)
    for step_idx, (d_op, r_op) in enumerate(operator_pairs):
        active_sc = scorer if (step_idx % 2 == 0 or custom_insertion_scorer is not None) else None
        trial = _alns_destroy_and_repair(
            current,
            instance,
            destroy_mode=d_op,
            repair_mode=r_op,
            rng=rng,
            insertion_scorer=active_sc,
        )
        if trial.feasible and trial.total_objective + 1e-6 < best_sched.total_objective:
            best_sched = trial
            current = trial
        elif trial.feasible and trial.total_objective + 1e-6 < current.total_objective:
            current = trial
        elif trial.feasible:
            delta = trial.total_objective - current.total_objective
            if rng.random() < math.exp(-delta / max(1e-6, temp)):
                current = trial
        temp *= 0.82

    # --- Phase 4: Dual-Scorer TGLS (Xu 2025) + UB-BB Exact Window Polish (Chen 2026) ---
    best_sched = target_guided_local_search(
        best_sched, instance, insertion_scorer=scorer, max_passes=1
    )
    if not fast_rolling_mode and custom_insertion_scorer is None:
        best_sched = target_guided_local_search(
            best_sched, instance, insertion_scorer=None, max_passes=2
        )
    best_sched = _branch_and_bound_window_polish(
        best_sched,
        instance,
        insertion_scorer=scorer if custom_insertion_scorer is not None else None,
        window_size=4 if not fast_rolling_mode else 3,
        max_nodes=18 if not fast_rolling_mode else 10,
    )

    best_sched.runtime_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    best_sched.solver_name = "reevo_vrpagent_bap"
    return best_sched



# ---------------------------------------------------------------------------
# 10. Event-Driven vs Time-Driven Rolling-Horizon Simulator (Chen et al. 2026 Sec. 4.3)
# ---------------------------------------------------------------------------


def compare_rolling_horizon_strategies(instance: BerthInstance) -> Dict[str, object]:
    """
    Replicates the Rolling-Horizon Dynamic Berth Allocation experiment from
    Section 4.3 (Tables 10-13) of Chen et al. (Ocean Engineering 364, 2026).
    """
    all_v = sorted(instance.vessels, key=lambda v: (v.arrival_time, v.id))
    n_init = max(4, len(all_v) // 2)
    init_v = all_v[:n_init]
    dyn_v = all_v[n_init:]

    raw_event_times = sorted({v.arrival_time for v in dyn_v})
    if len(raw_event_times) > 4:
        step_idx = max(1, len(raw_event_times) // 4)
        event_times = [raw_event_times[i] for i in range(0, len(raw_event_times), step_idx)]
        if raw_event_times[-1] not in event_times:
            event_times.append(raw_event_times[-1])
    else:
        event_times = raw_event_times
    max_eta = max((v.arrival_time for v in all_v), default=12.0)
    td_times = [float(t) for t in range(6, int(math.ceil(max_eta)) + 7, 6)]

    def run_rolling_simulation(
        strategy_code: str,
        trigger_times: List[float],
        stage_solver: Callable[[BerthInstance], BerthSchedule],
    ) -> Dict[str, object]:
        t0 = time.perf_counter()
        cur_vessels = [copy.deepcopy(v) for v in init_v]
        sub_inst = copy.deepcopy(instance)
        sub_inst.vessels = cur_vessels
        current_sched = stage_solver(sub_inst)

        released_ids = {v.id for v in cur_vessels}
        stage_logs: List[Dict[str, object]] = [
            {
                "stage": 0,
                "trigger_time": 0.0,
                "active_vessels": len(cur_vessels),
                "objective": current_sched.total_objective,
                "waiting_time": current_sched.total_waiting_time,
            }
        ]
        berth_shifts: List[float] = []

        for st_idx, t_trig in enumerate(trigger_times, start=1):
            new_arrivals = [
                copy.deepcopy(v)
                for v in dyn_v
                if v.id not in released_ids and v.arrival_time <= t_trig + 1e-9
            ]
            if not new_arrivals and st_idx < len(trigger_times):
                continue

            prev_map = {
                sv.vessel_id: sv for sv in current_sched.scheduled_vessels
            }
            updated_vessels: List[Vessel] = []
            for v in cur_vessels:
                sv_prev = prev_map[v.id]
                v_copy = copy.deepcopy(v)
                if sv_prev.start_time <= t_trig + 1e-9:
                    v_copy.status = 0
                    v_copy.locked_berth = sv_prev.berth_pos
                    v_copy.locked_start = sv_prev.start_time
                    setattr(v_copy, "locked_qc_profile", list(sv_prev.qc_profile))
                    setattr(v_copy, "locked_end", sv_prev.end_time)
                else:
                    v_copy.status = 2
                    v_copy.locked_berth = None
                    v_copy.locked_start = None
                    setattr(v_copy, "locked_qc_profile", None)
                    setattr(v_copy, "locked_end", None)
                updated_vessels.append(v_copy)

            for nv in new_arrivals:
                nv.status = 3
                if strategy_code == "TDBB-6h" and nv.arrival_time < t_trig:
                    nv.arrival_time = max(nv.arrival_time, t_trig - 1.5)
                updated_vessels.append(nv)
                released_ids.add(nv.id)

            cur_vessels = updated_vessels
            sub_inst = copy.deepcopy(instance)
            sub_inst.vessels = cur_vessels
            new_sched = stage_solver(sub_inst)

            new_map = {sv.vessel_id: sv for sv in new_sched.scheduled_vessels}
            for vid, sv_old in prev_map.items():
                if sv_old.start_time > t_trig + 1e-9 and vid in new_map:
                    berth_shifts.append(
                        abs(new_map[vid].berth_pos - sv_old.berth_pos)
                    )

            current_sched = new_sched
            stage_logs.append(
                {
                    "stage": st_idx,
                    "trigger_time": t_trig,
                    "active_vessels": len(cur_vessels),
                    "objective": current_sched.total_objective,
                    "waiting_time": current_sched.total_waiting_time,
                }
            )

        rem_v = [copy.deepcopy(v) for v in dyn_v if v.id not in released_ids]
        if rem_v:
            cur_vessels.extend(rem_v)
            sub_inst = copy.deepcopy(instance)
            sub_inst.vessels = cur_vessels
            current_sched = stage_solver(sub_inst)

        final_inst = copy.deepcopy(instance)
        current_sched = _finalize_schedule(
            current_sched.scheduled_vessels,
            final_inst,
            strategy_code,
            runtime_ms=(time.perf_counter() - t0) * 1000.0,
        )
        avg_shift = (
            round(sum(berth_shifts) / len(berth_shifts), 2)
            if berth_shifts
            else 0.0
        )
        return {
            "strategy": strategy_code,
            "total_objective": current_sched.total_objective,
            "total_waiting_time": current_sched.total_waiting_time,
            "total_service_time": current_sched.total_service_time,
            "total_berth_deviation": current_sched.total_berth_deviation,
            "makespan": current_sched.makespan,
            "lower_bound": current_sched.lower_bound,
            "optimality_gap_pct": current_sched.optimality_gap_pct,
            "runtime_ms": current_sched.runtime_ms,
            "avg_replan_berth_shift_m": avg_shift,
            "num_triggers": len(stage_logs),
            "stages": stage_logs,
            "feasible": current_sched.feasible,
        }

    nr_res = run_rolling_simulation("NR-FIFP", [], solve_fifp_greedy)
    td_res = run_rolling_simulation("TDBB-6h", td_times, solve_grasp_lee2010)
    ed_bb_res = run_rolling_simulation(
        "EDBB-Chen26", event_times, solve_reevo_ub_bb_chen2026
    )
    reevo_ed_res = run_rolling_simulation(
        "ReEvo-ED-UB-BB",
        event_times,
        lambda inst: solve_reevo_vrpagent_bap(inst, fast_rolling_mode=True),
    )

    return {
        "instance_name": instance.name,
        "title": instance.title,
        "num_initial_vessels": len(init_v),
        "num_dynamic_vessels": len(dyn_v),
        "event_trigger_times": event_times,
        "time_driven_trigger_times": td_times,
        "strategies": [nr_res, td_res, ed_bb_res, reevo_ed_res],
    }


# ---------------------------------------------------------------------------
# 11. Solver Registry & Metadata
# ---------------------------------------------------------------------------

BAP_SOLVER_METADATA: Dict[str, Dict[str, str]] = {
    "fifp_greedy": {
        "label": "FIFP / FCFS Greedy Baseline",
        "paper": "Chen et al. (2026) / Zou et al. (2025) Baseline",
        "category": "Constructive Rule",
        "description": "First-In-First-Planning arrival order with greedy earliest start-time berth insertion.",
    },
    "grasp_lee2010": {
        "label": "GRASP_2 + Prop. 3.1 + Exchange",
        "paper": "Lee, Chen & Cao (Transp. Res. E 2010) & Guan & Cheung (2004)",
        "category": "Multi-Start Metaheuristic",
        "description": "Randomized greedy Restricted Candidate List (RCL) construction using Proposition 3.1 vertex insertion and Pair-Wise Exchange.",
    },
    "hma_xu2025": {
        "label": "HMA (3-Stage QC Decoder + TGLS)",
        "paper": "Xu, Huang, Zhao, Gong & Liu (IEEE TITS 2025)",
        "category": "Evolutionary Memetic",
        "description": "Memetic Algorithm with 3-Stage Time-Variant Quay Crane Decoding, Horizontal Berth Offset Rectification, and Target-Guided Local Search.",
    },
    "i2hcso_zou2025": {
        "label": "I2HCSO (Hierarchical Swarm + Repair)",
        "paper": "Zou, Xu, Dai, Song & Luo (IEEE TETCI 2025)",
        "category": "Swarm Intelligence",
        "description": "Intra- and Inter-Hierarchical Competitive Swarm Optimization on continuous vessel priority keys with local neighborhood repair.",
    },
    "alns_martin2024": {
        "label": "ALNS (Shaw/Cost-Time + Regret-2)",
        "paper": "Martin-Iradi, Pacino & Ropke (EJOR 2024)",
        "category": "Large Neighborhood Search",
        "description": "Adaptive Large Neighborhood Search combining Shaw & Cost-Time destroy operators with 2-Regret and Strict-Packing repair.",
    },
    "reevo_ub_bb_chen2026": {
        "label": "UB-BB (Initial-Bound Branch & Bound)",
        "paper": "Chen et al. (Ocean Engineering 2026)",
        "category": "Hybrid Exact + Heuristic",
        "description": "Policy-guided Initial-Bound Optimization (UB-BB) warm-starting Depth-First Branch-and-Bound with Analytical Lower Bound Eq. (19).",
    },
    "reevo_vrpagent_bap": {
        "label": "ReEvo-VRPAgent Unified BAP (Ours)",
        "paper": "Proposed Synthesis (ReEvo + VRPAgent + 3-Stage QC + UB-BB)",
        "category": "LLM-Guided Hyper-Heuristic",
        "description": "ReEvo-evolved spatiotemporal packing & priority scoring + Xu (2025) 3-Stage QC rectification + VRPAgent destroy/repair portfolio + Chen (2026) UB-BB polish.",
    },
}

BAP_SOLVER_REGISTRY: Dict[str, Callable[[BerthInstance], BerthSchedule]] = {
    "fifp_greedy": solve_fifp_greedy,
    "grasp_lee2010": solve_grasp_lee2010,
    "hma_xu2025": solve_hma_xu2025,
    "i2hcso_zou2025": solve_i2hcso_zou2025,
    "alns_martin2024": solve_alns_martin2024,
    "reevo_ub_bb_chen2026": solve_reevo_ub_bb_chen2026,
    "reevo_vrpagent_bap": solve_reevo_vrpagent_bap,
}
