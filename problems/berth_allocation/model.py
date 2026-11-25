"""
Mathematical Models, Analytical Lower Bounds, and Feasibility Verifiers for Berth Allocation Problems.

Synthesizes the mathematical formulations from:
1. Chen et al. (Ocean Engineering 364, 2026, 127072): Dynamic Continuous BAP (DCBAP) &
   Event-Driven Rolling-Horizon with Analytical Lower Bound Eq. (19).
2. Martin-Iradi, Pacino & Ropke (EJOR 316, 2024, 152-167): Continuous BAP with
   position-dependent handling times h_i = (1 + beta * |p_i - b_i^0|) * h_i^0.
3. Xu, Huang, Zhao, Gong & Liu (IEEE TITS 26(3), 2025, 3387-3401): Continuous Berth
   Allocation and Time-Variant Quay Crane Assignment (C/T-V BACAP).
4. Zou, Xu, Dai, Song & Luo (IEEE TETCI 9(2), 2025, 1307-1321): Large-Scale BACAP with
   vessel priority encoding and safety distance constraints.
5. Guan & Cheung (OR Spectrum 26, 2004, 75-92) & Lee, Chen & Cao (TRE 46, 2010, 1017-1029):
   Relative Position Formulation (RPF) & Proposition 3.1 staircase/vertex packing properties.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Vessel:
    """Represents a single vessel requiring berth and quay crane allocation."""

    id: int
    name: str
    length: float
    arrival_time: float
    handling_time: float
    workload: float
    q_min: int
    q_max: int
    preferred_berth: float
    weight: float = 1.0
    status: int = 2  # 0=Working/Berthed, 1=Departed, 2=Scheduled Unberthed, 3=Dynamic Arrival
    locked_berth: Optional[float] = None
    locked_start: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScheduledVessel:
    """Allocation decision for a single vessel."""

    vessel_id: int
    berth_pos: float
    start_time: float
    end_time: float
    assigned_cranes: float
    qc_profile: List[int] = field(default_factory=list)
    waiting_time: float = 0.0
    berth_deviation: float = 0.0
    weighted_cost: float = 0.0
    length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BerthInstance:
    """Complete specification of a CBAP / C/T-V BACAP / DCBAP benchmark instance."""

    name: str
    title: str
    paper_ref: str
    description: str
    quay_length: float
    total_cranes: int
    berth_step: float
    time_step: float
    safety_Clearance_ratio: float
    safety_time_gap: float
    alpha_wait: float
    alpha_handle: float
    beta_dev: float
    berth_deviation_factor: float
    use_crane_allocation: bool
    vessels: List[Vessel]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "paper_ref": self.paper_ref,
            "description": self.description,
            "quay_length": self.quay_length,
            "total_cranes": self.total_cranes,
            "berth_step": self.berth_step,
            "time_step": self.time_step,
            "safety_clearance_ratio": self.safety_Clearance_ratio,
            "safety_time_gap": self.safety_time_gap,
            "alpha_wait": self.alpha_wait,
            "alpha_handle": self.alpha_handle,
            "beta_dev": self.beta_dev,
            "berth_deviation_factor": self.berth_deviation_factor,
            "use_crane_allocation": self.use_crane_allocation,
            "num_vessels": len(self.vessels),
            "vessels": [v.to_dict() for v in self.vessels],
        }


@dataclass
class BerthSchedule:
    """Complete solution for a BerthInstance with decomposed objective metrics."""

    instance_name: str
    solver_name: str
    scheduled_vessels: List[ScheduledVessel]
    total_objective: float
    total_waiting_time: float
    total_service_time: float
    total_berth_deviation: float
    makespan: float
    lower_bound: float
    optimality_gap_pct: float
    runtime_ms: float
    feasible: bool = True
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_name": self.instance_name,
            "solver_name": self.solver_name,
            "total_objective": self.total_objective,
            "total_waiting_time": self.total_waiting_time,
            "total_service_time": self.total_service_time,
            "total_berth_deviation": self.total_berth_deviation,
            "makespan": self.makespan,
            "lower_bound": self.lower_bound,
            "optimality_gap_pct": self.optimality_gap_pct,
            "runtime_ms": self.runtime_ms,
            "feasible": self.feasible,
            "violations": self.violations,
            "scheduled_vessels": [s.to_dict() for s in self.scheduled_vessels],
        }


def effective_handling_time(
    vessel: Vessel, berth_pos: float, instance: BerthInstance
) -> float:
    """
    Computes position-dependent handling time:
    h_i = (1 + beta_pos * |p_i - b_i^0|) * h_i^0 (Martin-Iradi et al. 2024 Eq. 26).
    """
    dev = abs(berth_pos - vessel.preferred_berth)
    raw = vessel.handling_time * (1.0 + instance.berth_deviation_factor * dev)
    if instance.time_step > 0:
        return round(math.ceil(raw / instance.time_step - 1e-9) * instance.time_step, 4)
    return round(raw, 4)


def spatial_clearance(v1: Vessel, v2: Vessel, instance: BerthInstance) -> float:
    """
    Minimum required horizontal clearance between two simultaneously moored vessels:
    >= safety_clearance_ratio * max(l_1, l_2) (Xu et al. 2025 Eq. 5).
    """
    if instance.safety_Clearance_ratio <= 0.0:
        return 0.0
    raw = instance.safety_Clearance_ratio * max(v1.length, v2.length)
    return math.ceil(raw / instance.berth_step - 1e-9) * instance.berth_step


def rectangles_conflict(
    b1: float,
    l1: float,
    s1: float,
    e1: float,
    v1: Vessel,
    sv2: ScheduledVessel,
    v2: Vessel,
    instance: BerthInstance,
) -> bool:
    """Returns True if two 2D (berth x time) vessel rectangles overlap."""
    if (
        e1 + instance.safety_time_gap <= sv2.start_time + 1e-9
        or sv2.end_time + instance.safety_time_gap <= s1 + 1e-9
    ):
        return False
    clr = spatial_clearance(v1, v2, instance)
    if (
        b1 + l1 + clr <= sv2.berth_pos + 1e-9
        or sv2.berth_pos + v2.length + clr <= b1 + 1e-9
    ):
        return False
    return True


def compute_vessel_cost(
    vessel: Vessel,
    berth_pos: float,
    start_time: float,
    end_time: float,
    instance: BerthInstance,
) -> float:
    """
    Computes the weighted objective contribution of a single vessel:
    w_i * (alpha_wait * (ts_i - a_i) + alpha_handle * (te_i - ts_i) + beta_dev * |b_i - p_i|).
    Unifies Eq. (1) of Chen et al. (2026), Eq. (1) of Xu et al. (2025), and Eq. (2) of Martin-Iradi (2024).
    """
    wait = max(0.0, start_time - vessel.arrival_time)
    dur = max(0.0, end_time - start_time)
    dev = abs(berth_pos - vessel.preferred_berth)
    return round(
        vessel.weight
        * (
            instance.alpha_wait * wait
            + instance.alpha_handle * dur
            + instance.beta_dev * dev
        ),
        4,
    )


def compute_analytical_lower_bound(instance: BerthInstance) -> float:
    """
    Computes the Analytical Lower Bound (Eq. 19 of Chen et al. 2026 & Guan & Cheung 2004):
    Each vessel is evaluated at its conflict-free earliest arrival time (or locked berth/start
    if Status 0) with maximum crane speed `q_max` (for BACAP) or ideal berth handling duration.
    """
    total_lb = 0.0
    for v in instance.vessels:
        if v.status == 0 and v.locked_berth is not None and v.locked_start is not None:
            b0 = v.locked_berth
            t0 = v.locked_start
            dur = (
                math.ceil(v.workload / v.q_max)
                if instance.use_crane_allocation
                else effective_handling_time(v, b0, instance)
            )
            total_lb += compute_vessel_cost(v, b0, t0, t0 + dur, instance)
        else:
            min_dur = (
                float(math.ceil(v.workload / v.q_max))
                if instance.use_crane_allocation
                else v.handling_time
            )
            total_lb += compute_vessel_cost(
                v, v.preferred_berth, v.arrival_time, v.arrival_time + min_dur, instance
            )
    return round(total_lb, 4)


def verify_schedule(
    instance: BerthInstance, schedule: BerthSchedule
) -> Tuple[bool, List[str]]:
    """
    Strictly verifies all operational, spatial, temporal, crane, and rolling-horizon constraints:
      1. Every vessel in `instance.vessels` is scheduled exactly once.
      2. Arrival constraint: `start_time >= arrival_time`.
      3. Quay boundary constraint: `0 <= berth_pos` and `berth_pos + length <= quay_length`.
      4. Handling duration / crane workload constraint.
      5. Locked vessel invariance (Status 0 in rolling horizon).
      6. Pairwise 2D spatiotemporal non-overlap with required safety clearance.
      7. Time-variant quay crane capacity `sum_i q_{i,t} <= total_cranes` in every time slice.
    """
    violations: List[str] = []
    v_map = {v.id: v for v in instance.vessels}
    allocs = schedule.scheduled_vessels

    if len(allocs) != len(instance.vessels):
        violations.append(
            f"Expected {len(instance.vessels)} scheduled vessels, got {len(allocs)}."
        )

    seen = set()
    for sv in allocs:
        if sv.vessel_id in seen:
            violations.append(f"Duplicate vessel_id {sv.vessel_id}.")
        seen.add(sv.vessel_id)
        v = v_map.get(sv.vessel_id)
        if v is None:
            violations.append(f"Unknown vessel_id {sv.vessel_id}.")
            continue
        # Ensure length is populated on ScheduledVessel
        sv.length = v.length

        if sv.start_time + 1e-6 < v.arrival_time:
            violations.append(
                f"Vessel {v.name} starts at {sv.start_time} before arrival {v.arrival_time}."
            )
        if sv.berth_pos < -1e-6 or sv.berth_pos + v.length > instance.quay_length + 1e-6:
            violations.append(
                f"Vessel {v.name} berth [{sv.berth_pos}, {sv.berth_pos + v.length}] outside quay [0, {instance.quay_length}]."
            )

        if instance.use_crane_allocation:
            dev_mult = 1.0 + instance.berth_deviation_factor * abs(
                sv.berth_pos - v.preferred_berth
            )
            req_work = v.workload * dev_mult
            delivered = float(sum(sv.qc_profile))
            if delivered + 1e-5 < req_work:
                violations.append(
                    f"Vessel {v.name} delivered crane-hours {delivered} < required {req_work:.2f}."
                )
            for q_h in sv.qc_profile:
                if q_h < v.q_min or q_h > v.q_max:
                    violations.append(
                        f"Vessel {v.name} hourly crane count {q_h} outside [{v.q_min}, {v.q_max}]."
                    )
                    break
        else:
            req_dur = effective_handling_time(v, sv.berth_pos, instance)
            if (sv.end_time - sv.start_time) + 1e-5 < req_dur:
                violations.append(
                    f"Vessel {v.name} duration {sv.end_time - sv.start_time} < required {req_dur}."
                )

        if v.status == 0 and v.locked_berth is not None and v.locked_start is not None:
            if (
                abs(sv.berth_pos - v.locked_berth) > 1e-5
                or abs(sv.start_time - v.locked_start) > 1e-5
            ):
                violations.append(
                    f"Locked vessel {v.name} moved from ({v.locked_berth}, {v.locked_start}) to ({sv.berth_pos}, {sv.start_time})."
                )

    # Pairwise 2D non-overlap check
    n = len(allocs)
    for i in range(n):
        sv1 = allocs[i]
        v1 = v_map[sv1.vessel_id]
        for j in range(i + 1, n):
            sv2 = allocs[j]
            v2 = v_map[sv2.vessel_id]
            if rectangles_conflict(
                sv1.berth_pos,
                v1.length,
                sv1.start_time,
                sv1.end_time,
                v1,
                sv2,
                v2,
                instance,
            ):
                violations.append(
                    f"2D overlap between {v1.name} ([{sv1.berth_pos}, {sv1.berth_pos+v1.length}]x[{sv1.start_time}, {sv1.end_time}]) "
                    f"and {v2.name} ([{sv2.berth_pos}, {sv2.berth_pos+v2.length}]x[{sv2.start_time}, {sv2.end_time}])."
                )

    # Quay crane capacity check across all integer hours
    if instance.use_crane_allocation:
        max_t = int(math.ceil(max((s.end_time for s in allocs), default=0.0))) + 2
        for tau in range(max_t):
            used_q = 0
            for sv in allocs:
                t0 = int(round(sv.start_time))
                idx = tau - t0
                if 0 <= idx < len(sv.qc_profile):
                    used_q += sv.qc_profile[idx]
            if used_q > instance.total_cranes:
                violations.append(
                    f"Quay crane capacity exceeded at hour {tau}: {used_q} > {instance.total_cranes}."
                )
                break

    return (len(violations) == 0), violations
