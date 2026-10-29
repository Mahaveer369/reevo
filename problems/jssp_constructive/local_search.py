"""Critical-Path Block-Swap Local Search (Nowicki-Smutnicki N5 / Balas N1) and Multi-Start Hybrid Solver."""

from __future__ import annotations
import random
import time
from typing import List, Optional, Tuple
from .parser import JSSPInstance
from .schedule import OperationID, ScheduleResult, evaluate_machine_sequences, validate_schedule
from .baselines import PriorityFunc, rule_mwkr
from .scheduler import build_active_schedule_gt


def generate_critical_block_neighbors(
    schedule: ScheduleResult,
    neighborhood: str = "N1",
) -> List[Tuple[int, int, int]]:
    """
    Generate candidate adjacent swaps (machine_id, pos_a, pos_b) on critical blocks.
    - 'N5' (Nowicki & Smutnicki, 1996): swaps only the first two and last two operations of each critical block.
    - 'N1' (Balas, 1969): swaps any adjacent pair of critical operations within each critical block.
    Analogous to candidate-list restriction in 2-opt for TSP: non-critical swaps can never reduce Cmax.
    """
    moves: List[Tuple[int, int, int]] = []
    seen = set()

    for m, block in schedule.critical_blocks:
        if len(block) < 2:
            continue
        seq = schedule.machine_sequences[m]
        pos_map = {op: idx for idx, op in enumerate(seq)}

        if neighborhood == "N5":
            # Border swaps only: first pair and last pair in the critical block
            pairs = [(block[0], block[1])]
            if len(block) > 2:
                pairs.append((block[-2], block[-1]))
        else:
            # All adjacent pairs inside the critical block
            pairs = [(block[i], block[i + 1]) for i in range(len(block) - 1)]

        for op_u, op_v in pairs:
            # Two operations from the same job cannot be swapped
            if op_u[0] == op_v[0]:
                continue
            idx_u = pos_map[op_u]
            idx_v = pos_map[op_v]
            if idx_v == idx_u + 1:
                key = (m, idx_u, idx_v)
                if key not in seen:
                    seen.add(key)
                    moves.append(key)
    return moves


def critical_path_local_search(
    instance: JSSPInstance,
    initial_schedule: ScheduleResult,
    max_steps: int = 80,
    tabu_tenure: int = 7,
    neighborhood: str = "N5",
    solver_name: Optional[str] = None,
    validate: bool = True,
) -> ScheduleResult:
    """
    Critical-Path Tabu / Descent Local Search over the disjunctive graph.
    Explores critical-block adjacent swaps, guaranteeing 100% schedule feasibility
    by rejecting any swap that creates a cycle in evaluate_machine_sequences.
    """
    t0 = time.perf_counter()
    current = initial_schedule
    best = initial_schedule
    eval_count = initial_schedule.evaluations
    tabu_dict = {}
    label = solver_name or f"{initial_schedule.solver_name}+CP-LS"

    for step in range(max_steps):
        candidate_moves = generate_critical_block_neighbors(current, neighborhood=neighborhood)
        if not candidate_moves:
            break

        best_neighbor_cmax: Optional[int] = None
        best_trial_seqs: Optional[List[List[OperationID]]] = None
        best_move_ops: Optional[Tuple[OperationID, OperationID]] = None

        for (m, idx_u, idx_v) in candidate_moves:
            op_u = current.machine_sequences[m][idx_u]
            op_v = current.machine_sequences[m][idx_v]
            is_tabu = tabu_dict.get((op_v, op_u), -1) > step

            trial_seqs = [list(seq) for seq in current.machine_sequences]
            trial_seqs[m][idx_u], trial_seqs[m][idx_v] = trial_seqs[m][idx_v], trial_seqs[m][idx_u]

            eval_count += 1
            cand_sched = evaluate_machine_sequences(
                instance=instance,
                machine_sequences=trial_seqs,
                solver_name=label,
                compute_critical=False,
            )
            if cand_sched is None:
                continue

            # Aspiration criterion: accept tabu move if it strictly improves global best
            if is_tabu and cand_sched.makespan >= best.makespan:
                continue

            if best_neighbor_cmax is None or cand_sched.makespan < best_neighbor_cmax:
                best_neighbor_cmax = cand_sched.makespan
                best_trial_seqs = trial_seqs
                best_move_ops = (op_u, op_v)

        if best_trial_seqs is None:
            break

        if best_move_ops is not None:
            tabu_dict[best_move_ops] = step + tabu_tenure

        next_sched = evaluate_machine_sequences(
            instance=instance,
            machine_sequences=best_trial_seqs,
            solver_name=label,
            compute_critical=True,
        )
        if next_sched is None:
            break

        current = next_sched
        if current.makespan < best.makespan:
            best = current
            if instance.optimum is not None and best.makespan <= instance.optimum:
                break

    best.solver_name = label
    best.runtime_ms = initial_schedule.runtime_ms + (time.perf_counter() - t0) * 1000.0
    best.evaluations = eval_count
    if validate:
        best.validation = validate_schedule(instance, best)
    return best


def multi_start_hybrid_solve(
    instance: JSSPInstance,
    priority_fn: PriorityFunc = rule_mwkr,
    n_starts: int = 12,
    ls_steps: int = 40,
    temperature: float = 0.22,
    seed: int = 42,
    solver_name: str = "MultiStart-GT+CP-LS",
    use_local_search: bool = True,
) -> ScheduleResult:
    """
    Matched-budget Multi-Start Giffler-Thompson + Critical-Path Local Search solver.
    Start 0 is deterministic greedy (temperature=0); Starts 1..n_starts-1 sample conflict sets
    using Boltzmann softmax over priority_fn(features).
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    best_sched: Optional[ScheduleResult] = None
    total_evals = 0

    for start_idx in range(n_starts):
        temp = 0.0 if start_idx == 0 else temperature
        init_sched = build_active_schedule_gt(
            instance=instance,
            priority_fn=priority_fn,
            solver_name=solver_name,
            stochastic_temperature=temp,
            rng=rng,
            validate=False,
        )
        total_evals += 1
        if use_local_search and ls_steps > 0:
            refined = critical_path_local_search(
                instance=instance,
                initial_schedule=init_sched,
                max_steps=ls_steps,
                tabu_tenure=5,
                neighborhood="N5",
                solver_name=solver_name,
                validate=False,
            )
            total_evals += (refined.evaluations - 1)
        else:
            refined = init_sched

        if best_sched is None or refined.makespan < best_sched.makespan:
            best_sched = refined
            if instance.optimum is not None and best_sched.makespan <= instance.optimum:
                break

    assert best_sched is not None
    best_sched.solver_name = solver_name
    best_sched.runtime_ms = (time.perf_counter() - t0) * 1000.0
    best_sched.evaluations = total_evals
    best_sched.validation = validate_schedule(instance, best_sched)
    return best_sched
