"""Dimensionless state-feature extraction and classical JSSP dispatching rules."""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple
from .parser import JSSPInstance

OperationID = Tuple[int, int]
PriorityFunc = Callable[[Dict[str, list]], List[float]]


def compute_conflict_features(
    instance: JSSPInstance,
    conflict_ops: List[OperationID],
    job_ready: List[int],
    machine_ready: List[int],
    job_rem_work: List[int],
    job_next_op: List[int],
    machine_rem_work: List[int],
    scheduled_count: int,
    mean_pt: float,
    max_job_work: float,
    max_mach_work: float,
) -> Dict[str, list]:
    """
    Compute dimensionless, scale-invariant dynamic state features for all operations
    in the current Giffler-Thompson conflict set C.
    """
    n_machines = instance.n_machines
    total_ops = instance.total_ops
    progress = float(scheduled_count) / float(max(1, total_ops))

    # Compute current Work in Next Queue (WINQ) for each machine:
    # sum of durations of currently eligible operations waiting on machine m'
    current_queue_work = [0] * n_machines
    for j in range(instance.n_jobs):
        k_cur = job_next_op[j]
        if k_cur < n_machines:
            m_cur = instance.machines[j][k_cur]
            current_queue_work[m_cur] += instance.durations[j][k_cur]

    current_lb = max(
        max((job_ready[j] + job_rem_work[j] for j in range(instance.n_jobs)), default=1),
        max((machine_ready[m] + machine_rem_work[m] for m in range(n_machines)), default=1),
        1,
    )

    pt_norm: List[float] = []
    rem_work_norm: List[float] = []
    rem_ops_norm: List[float] = []
    winq_norm: List[float] = []
    slack_norm: List[float] = []
    wait_norm: List[float] = []
    machine_load_norm: List[float] = []
    earliest_start_norm: List[float] = []
    job_ids: List[int] = []

    for (j, k) in conflict_ops:
        m = instance.machines[j][k]
        p = instance.durations[j][k]
        r_o = max(job_ready[j], machine_ready[m])
        wait_o = max(0, machine_ready[m] - job_ready[j])
        rem_w = job_rem_work[j]
        rem_ops = n_machines - k

        if k + 1 < n_machines:
            next_m = instance.machines[j][k + 1]
            winq = current_queue_work[next_m]
        else:
            winq = 0

        slack = float(current_lb - (r_o + rem_w)) / float(max_job_work)

        pt_norm.append(float(p) / mean_pt)
        rem_work_norm.append(float(rem_w) / max_job_work)
        rem_ops_norm.append(float(rem_ops) / float(n_machines))
        winq_norm.append(float(winq) / max_job_work)
        slack_norm.append(slack)
        wait_norm.append(float(wait_o) / mean_pt)
        machine_load_norm.append(float(machine_rem_work[m]) / max_mach_work)
        earliest_start_norm.append(float(r_o) / max_job_work)
        job_ids.append(j)

    return {
        "pt_norm": pt_norm,
        "rem_work_norm": rem_work_norm,
        "rem_ops_norm": rem_ops_norm,
        "winq_norm": winq_norm,
        "slack_norm": slack_norm,
        "wait_norm": wait_norm,
        "machine_load_norm": machine_load_norm,
        "earliest_start_norm": earliest_start_norm,
        "progress": progress,
        "job_ids": job_ids,
    }


# Classical Human-Designed Dispatching Rules (Higher score = selected first)

def rule_spt(features: Dict[str, list]) -> List[float]:
    """Shortest Processing Time (SPT)."""
    return [-p for p in features["pt_norm"]]


def rule_lpt(features: Dict[str, list]) -> List[float]:
    """Longest Processing Time (LPT)."""
    return [p for p in features["pt_norm"]]


def rule_mwkr(features: Dict[str, list]) -> List[float]:
    """Most Work Remaining (MWKR) with SPT tie-breaker."""
    return [m - 1e-4 * p for m, p in zip(features["rem_work_norm"], features["pt_norm"])]


def rule_lwkr(features: Dict[str, list]) -> List[float]:
    """Least Work Remaining (LWKR)."""
    return [-m for m in features["rem_work_norm"]]


def rule_mor(features: Dict[str, list]) -> List[float]:
    """Most Operations Remaining (MOR) with MWKR tie-breaker."""
    return [o + 1e-3 * m for o, m in zip(features["rem_ops_norm"], features["rem_work_norm"])]


def rule_fifo(features: Dict[str, list]) -> List[float]:
    """First-In First-Out / Earliest Ready Time (FIFO)."""
    return [-es for es in features["earliest_start_norm"]]


def rule_pt_plus_winq(features: Dict[str, list]) -> List[float]:
    """Holthaus & Rajendran PT + WINQ (Shortest Processing Time + Work in Next Queue)."""
    return [-(p + 2.0 * w) for p, w in zip(features["pt_norm"], features["winq_norm"])]


CLASSICAL_RULES: Dict[str, PriorityFunc] = {
    "SPT": rule_spt,
    "LPT": rule_lpt,
    "MWKR": rule_mwkr,
    "LWKR": rule_lwkr,
    "MOR": rule_mor,
    "FIFO": rule_fifo,
    "PT+WINQ": rule_pt_plus_winq,
}
