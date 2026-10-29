"""Schedule representation, disjunctive graph critical-path computation, and independent feasibility validator."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from .parser import JSSPInstance

OperationID = Tuple[int, int]  # (job_idx, op_idx_in_job)


@dataclass
class ValidationReport:
    feasible: bool
    makespan: int
    total_operations_checked: int
    precedence_ok: bool
    machine_non_overlap_ok: bool
    duration_exact_ok: bool
    completeness_ok: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class ScheduleResult:
    instance_name: str
    solver_name: str
    makespan: int
    # start_times[(j, k)] = S_{j,k}
    start_times: Dict[OperationID, int]
    # end_times[(j, k)] = C_{j,k}
    end_times: Dict[OperationID, int]
    # machine_sequences[m] = ordered list of (j, k) processed on machine m
    machine_sequences: List[List[OperationID]]
    # critical_ops: set of (j, k) on the critical path (r + p + q == Cmax)
    critical_ops: Set[OperationID] = field(default_factory=set)
    # critical_blocks: list of (machine_id, [op1, op2, ...]) contiguous critical operations
    critical_blocks: List[Tuple[int, List[OperationID]]] = field(default_factory=list)
    validation: Optional[ValidationReport] = None
    runtime_ms: float = 0.0
    evaluations: int = 1

    def gap_percent(self, reference: Optional[int]) -> Optional[float]:
        if reference is None or reference <= 0:
            return None
        return 100.0 * (self.makespan - reference) / float(reference)


def validate_schedule(instance: JSSPInstance, result: ScheduleResult) -> ValidationReport:
    """
    Independent mathematical feasibility validator for any JSSP schedule.
    Verifies:
      1. Completeness: all n_jobs * n_machines operations scheduled exactly once.
      2. Exact durations & non-negativity: S_{j,k} >= 0 and C_{j,k} - S_{j,k} == p_{j,k}.
      3. Job precedence: S_{j,k} >= C_{j,k-1} for all k >= 1.
      4. Machine disjunctive non-overlap: on each machine m, sorted operations satisfy S_{r+1} >= C_r.
      5. Makespan consistency: result.makespan == max_{j,k} C_{j,k}.
    """
    errors: List[str] = []
    n_jobs = instance.n_jobs
    n_machines = instance.n_machines
    expected_ops = {(j, k) for j in range(n_jobs) for k in range(n_machines)}

    completeness_ok = True
    if set(result.start_times.keys()) != expected_ops or set(result.end_times.keys()) != expected_ops:
        completeness_ok = False
        errors.append("Start/end time keys do not match exact set of (job, op) operations.")

    seen_in_machines: List[OperationID] = []
    for m, seq in enumerate(result.machine_sequences):
        for op in seq:
            seen_in_machines.append(op)
            if op in expected_ops and instance.machines[op[0]][op[1]] != m:
                completeness_ok = False
                errors.append(f"Operation {op} assigned to machine {m}, expected {instance.machines[op[0]][op[1]]}.")
    if len(seen_in_machines) != len(expected_ops) or set(seen_in_machines) != expected_ops:
        completeness_ok = False
        errors.append("Machine sequences do not form a complete partition of all operations.")

    duration_exact_ok = True
    for (j, k) in expected_ops:
        if (j, k) not in result.start_times or (j, k) not in result.end_times:
            continue
        s = result.start_times[(j, k)]
        e = result.end_times[(j, k)]
        p = instance.durations[j][k]
        if s < 0:
            duration_exact_ok = False
            errors.append(f"Negative start time S[{j},{k}] = {s}.")
        if e - s != p:
            duration_exact_ok = False
            errors.append(f"Duration mismatch on op ({j},{k}): C - S = {e - s} != {p}.")

    precedence_ok = True
    for j in range(n_jobs):
        for k in range(1, n_machines):
            if (j, k) in result.start_times and (j, k - 1) in result.end_times:
                if result.start_times[(j, k)] < result.end_times[(j, k - 1)]:
                    precedence_ok = False
                    errors.append(
                        f"Job precedence violation in job {j}: op {k} starts at {result.start_times[(j, k)]} "
                        f"before op {k - 1} ends at {result.end_times[(j, k - 1)]}."
                    )

    machine_non_overlap_ok = True
    for m in range(n_machines):
        ops_on_m = [
            (result.start_times[(j, k)], result.end_times[(j, k)], (j, k))
            for (j, k) in expected_ops
            if (j, k) in result.start_times and instance.machines[j][k] == m
        ]
        ops_on_m.sort(key=lambda x: (x[0], x[1]))
        for idx in range(len(ops_on_m) - 1):
            s1, e1, op1 = ops_on_m[idx]
            s2, e2, op2 = ops_on_m[idx + 1]
            if s2 < e1:
                machine_non_overlap_ok = False
                errors.append(
                    f"Machine {m} overlap between {op1} [{s1}, {e1}] and {op2} [{s2}, {e2}]."
                )

    computed_cmax = max(result.end_times.values()) if result.end_times else 0
    if computed_cmax != result.makespan:
        errors.append(f"Reported makespan {result.makespan} != actual max completion {computed_cmax}.")

    feasible = (
        completeness_ok
        and duration_exact_ok
        and precedence_ok
        and machine_non_overlap_ok
        and (computed_cmax == result.makespan)
        and len(errors) == 0
    )
    return ValidationReport(
        feasible=feasible,
        makespan=computed_cmax,
        total_operations_checked=len(expected_ops),
        precedence_ok=precedence_ok,
        machine_non_overlap_ok=machine_non_overlap_ok,
        duration_exact_ok=duration_exact_ok,
        completeness_ok=completeness_ok,
        errors=errors,
    )


def evaluate_machine_sequences(
    instance: JSSPInstance,
    machine_sequences: List[List[OperationID]],
    solver_name: str = "Solver",
    compute_critical: bool = True,
) -> Optional[ScheduleResult]:
    """
    Given machine processing sequences (a disjunctive graph orientation),
    compute earliest start/end times via topological longest paths.
    Returns None if the disjunctive graph contains a cycle (infeasible orientation).
    """
    n_jobs = instance.n_jobs
    n_machines = instance.n_machines
    total_ops = n_jobs * n_machines

    # Map each operation (j, k) to its position on its assigned machine
    mach_pos: Dict[OperationID, int] = {}
    for m, seq in enumerate(machine_sequences):
        for pos, op in enumerate(seq):
            mach_pos[op] = pos

    in_degree: Dict[OperationID, int] = {}
    for j in range(n_jobs):
        for k in range(n_machines):
            deg = 0
            if k > 0:
                deg += 1
            if mach_pos[(j, k)] > 0:
                deg += 1
            in_degree[(j, k)] = deg

    queue: List[OperationID] = [op for op, d in in_degree.items() if d == 0]
    topo_order: List[OperationID] = []
    heads: Dict[OperationID, int] = {op: 0 for op in in_degree}

    idx = 0
    while idx < len(queue):
        u = queue[idx]
        idx += 1
        topo_order.append(u)
        j, k = u
        p_u = instance.durations[j][k]
        finish_u = heads[u] + p_u

        # Job successor
        if k + 1 < n_machines:
            v_job = (j, k + 1)
            if finish_u > heads[v_job]:
                heads[v_job] = finish_u
            in_degree[v_job] -= 1
            if in_degree[v_job] == 0:
                queue.append(v_job)

        # Machine successor
        m = instance.machines[j][k]
        pos = mach_pos[u]
        if pos + 1 < len(machine_sequences[m]):
            v_mach = machine_sequences[m][pos + 1]
            if finish_u > heads[v_mach]:
                heads[v_mach] = finish_u
            in_degree[v_mach] -= 1
            if in_degree[v_mach] == 0:
                queue.append(v_mach)

    if len(topo_order) != total_ops:
        # Cycle detected in disjunctive graph
        return None

    end_times: Dict[OperationID, int] = {
        (j, k): heads[(j, k)] + instance.durations[j][k]
        for (j, k) in topo_order
    }
    makespan = max(end_times.values()) if end_times else 0

    critical_ops: Set[OperationID] = set()
    critical_blocks: List[Tuple[int, List[OperationID]]] = []

    if compute_critical and total_ops > 0:
        # Backward pass for tails q_u (longest path from completion of u to makespan)
        tails: Dict[OperationID, int] = {op: 0 for op in topo_order}
        for u in reversed(topo_order):
            j, k = u
            p_u = instance.durations[j][k]
            max_after = 0
            if k + 1 < n_machines:
                v_job = (j, k + 1)
                cand = instance.durations[v_job[0]][v_job[1]] + tails[v_job]
                if cand > max_after:
                    max_after = cand
            m = instance.machines[j][k]
            pos = mach_pos[u]
            if pos + 1 < len(machine_sequences[m]):
                v_mach = machine_sequences[m][pos + 1]
                cand = instance.durations[v_mach[0]][v_mach[1]] + tails[v_mach]
                if cand > max_after:
                    max_after = cand
            tails[u] = max_after
            if heads[u] + p_u + tails[u] == makespan:
                critical_ops.add(u)

        # Extract critical blocks on each machine (contiguous critical ops linked by tight machine arcs)
        for m, seq in enumerate(machine_sequences):
            current_block: List[OperationID] = []
            for op in seq:
                if op in critical_ops:
                    if not current_block:
                        current_block = [op]
                    else:
                        prev_op = current_block[-1]
                        if end_times[prev_op] == heads[op]:
                            current_block.append(op)
                        else:
                            if len(current_block) >= 2:
                                critical_blocks.append((m, current_block))
                            current_block = [op]
                else:
                    if len(current_block) >= 2:
                        critical_blocks.append((m, current_block))
                    current_block = []
            if len(current_block) >= 2:
                critical_blocks.append((m, current_block))

    sched = ScheduleResult(
        instance_name=instance.name,
        solver_name=solver_name,
        makespan=makespan,
        start_times=heads,
        end_times=end_times,
        machine_sequences=[list(seq) for seq in machine_sequences],
        critical_ops=critical_ops,
        critical_blocks=critical_blocks,
    )
    return sched
