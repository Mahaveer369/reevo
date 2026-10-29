"""ReEvo evaluation entrypoint for JSSP constructive + critical-path local search heuristic evolution."""

from __future__ import annotations
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from problems.jssp_constructive.parser import generate_training_instances, list_benchmark_instances
from problems.jssp_constructive.scheduler import build_active_schedule_gt
from problems.jssp_constructive.local_search import critical_path_local_search
from problems.jssp_constructive.reevo_heuristics import compile_custom_priority_code, rule_reevo_qwen


def load_candidate_heuristic():
    gpt_path = os.path.join(os.path.dirname(__file__), "gpt.py")
    if os.path.exists(gpt_path):
        with open(gpt_path, "r", encoding="utf-8") as f:
            code = f.read()
        return compile_custom_priority_code(code)
    return rule_reevo_qwen


def evaluate_instances(instances, priority_fn, use_ls: bool = True) -> float:
    gaps = []
    for inst in instances:
        sched = build_active_schedule_gt(inst, priority_fn=priority_fn, validate=False)
        if use_ls:
            sched = critical_path_local_search(inst, sched, max_steps=35, validate=False)
        ref = inst.optimum or inst.lower_bound
        gap = 100.0 * (sched.makespan - ref) / float(ref)
        gaps.append(gap)
    return sum(gaps) / float(max(1, len(gaps)))


if __name__ == "__main__":
    print("[*] Running JSSP evaluation ...")
    problem_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    mood = sys.argv[3] if len(sys.argv) > 3 else "train"
    priority_fn = load_candidate_heuristic()

    if mood == "train":
        train_instances = generate_training_instances(n_instances=6, n_jobs=max(6, problem_size), n_machines=min(10, max(5, problem_size)), seed=1234)
        avg_gap = evaluate_instances(train_instances, priority_fn, use_ls=True)
        print("[*] Average:")
        print(f"{avg_gap:.6f}")
    else:
        val_instances = list_benchmark_instances(split="val") + list_benchmark_instances(split="debug")
        avg_gap = evaluate_instances(val_instances, priority_fn, use_ls=True)
        print(f"[*] Average for val: {avg_gap:.6f}")
        print(f"{avg_gap:.6f}")
