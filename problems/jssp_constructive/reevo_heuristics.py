"""ReEvo LLM-Discovered Priority Heuristics, Ablation Variants, and Safe Custom Code Executor."""

from __future__ import annotations
import math
from typing import Callable, Dict, List


REEVO_QWEN_CODE = '''def priority(features: dict) -> list:
    """
    ReEvo-Qwen Discovered Conflict-Set Priority Rule (Evolved on synthetic 10x10 training instances).
    Combines:
      1. Critical tail urgency: Most Work Remaining (`rem_work_norm`), Most Operations Remaining (`rem_ops_norm`),
         and minimum job slack (`-slack_norm`).
      2. Stage-adaptive non-delay bottleneck protection: penalizes operations with late earliest-start (`earliest_start_norm`)
         that force the bottleneck machine to sit idle, strengthening as `progress` advances.
      3. Downstream congestion avoidance (`-winq_norm`, Work in Next Queue) and mild SPT tie-breaking (`-pt_norm`).
    """
    pt = features['pt_norm']
    mwkr = features['rem_work_norm']
    mor = features['rem_ops_norm']
    winq = features['winq_norm']
    slack = features['slack_norm']
    es = features['earliest_start_norm']
    prog = features['progress']

    scores = []
    for p, m, o, w, s, e in zip(pt, mwkr, mor, winq, slack, es):
        tail_urgency = 1.60 * m + 0.35 * o - 0.85 * s
        idle_and_queue_penalty = (0.90 + 0.40 * prog) * e + 0.18 * w + 0.06 * p
        scores.append(tail_urgency - idle_and_queue_penalty)
    return scores
'''


def rule_reevo_qwen(features: Dict[str, list]) -> List[float]:
    """ReEvo-Qwen evolved state-dependent conflict-resolution priority function."""
    pt = features["pt_norm"]
    mwkr = features["rem_work_norm"]
    mor = features["rem_ops_norm"]
    winq = features["winq_norm"]
    slack = features["slack_norm"]
    es = features["earliest_start_norm"]
    prog = float(features.get("progress", 0.5))

    return [
        (1.60 * m + 0.35 * o - 0.85 * s) - (0.90 + 0.40 * prog) * e - 0.18 * w - 0.06 * p
        for p, m, o, w, s, e in zip(pt, mwkr, mor, winq, slack, es)
    ]


def rule_ablation_no_winq(features: Dict[str, list]) -> List[float]:
    """Ablation 1: ReEvo-Qwen rule with downstream queue lookahead (WINQ) & slack removed."""
    pt = features["pt_norm"]
    mwkr = features["rem_work_norm"]
    mor = features["rem_ops_norm"]
    es = features["earliest_start_norm"]
    prog = float(features.get("progress", 0.5))
    return [
        (1.60 * m + 0.35 * o) - (0.90 + 0.40 * prog) * e - 0.06 * p
        for p, m, o, e in zip(pt, mwkr, mor, es)
    ]


def rule_ablation_zero_shot(features: Dict[str, list]) -> List[float]:
    """Ablation 2: Zero-Shot LLM / Unreflected static linear combination (no ReEvo reflection/crossover)."""
    pt = features["pt_norm"]
    mwkr = features["rem_work_norm"]
    winq = features["winq_norm"]
    return [1.20 * m - 0.15 * w - 0.12 * p for p, m, w in zip(pt, mwkr, winq)]


def rule_vrpagent_ensemble(features: Dict[str, list]) -> List[float]:
    """
    VRPAgent-Inspired Multi-Regime Ensemble Priority Operator (Hottung, Berto, Hua, Ye, Park, Tierney, TMLR 2026).
    Switches/blends between three complementary scoring regimes based on conflict-set congestion & schedule progress:
      - Regime 1 (Early / Uncongested): Heavy global work-remaining (MWKR) + operation count (MOR) + slack urgency.
      - Regime 2 (High Downstream Queue Congestion): Strong WINQ lookahead relief + non-delay bottleneck protection.
      - Regime 3 (Late Tail / Tight Finishing): Aggressive non-delay + SPT machine release speed.
    """
    pt = features["pt_norm"]
    mwkr = features["rem_work_norm"]
    mor = features["rem_ops_norm"]
    winq = features["winq_norm"]
    slack = features["slack_norm"]
    es = features["earliest_start_norm"]
    prog = float(features.get("progress", 0.5))
    mean_winq = sum(winq) / float(len(winq)) if winq else 0.0

    scores: List[float] = []
    for p, m, o, w, s, e in zip(pt, mwkr, mor, winq, slack, es):
        r1_tail = 1.75 * m + 0.45 * o - 0.90 * s - 0.75 * e
        r2_congestion = 1.40 * m + 0.30 * o - 1.15 * e - 0.38 * w - 0.10 * p
        r3_finishing = 1.45 * m - 1.35 * e - 0.22 * p - 0.70 * s - 0.15 * w
        if mean_winq > 0.45:
            score = 0.35 * r1_tail + 0.50 * r2_congestion + 0.15 * r3_finishing
        elif prog < 0.65:
            score = 0.60 * r1_tail + 0.25 * r2_congestion + 0.15 * r3_finishing
        else:
            score = 0.25 * r1_tail + 0.20 * r2_congestion + 0.55 * r3_finishing
        scores.append(score)
    return scores



def compile_custom_priority_code(code_str: str) -> Callable[[Dict[str, list]], List[float]]:
    """
    Safely compile user- or LLM-provided Python source defining `priority*` into a callable PriorityFunc.
    Raises ValueError if syntax is invalid or no callable priority function is found.
    """
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "float": float,
        "int": int,
        "round": round,
        "sorted": sorted,
        "math": math,
    }
    namespace: dict = {"__builtins__": safe_builtins, "math": math}
    try:
        exec(code_str, namespace, namespace)
    except Exception as exc:
        raise ValueError(f"Failed to compile custom priority function: {exc}") from exc

    candidate_fn = None
    for name in ("priority", "priority_v1", "priority_v2", "priority_v3", "heuristics"):
        if name in namespace and callable(namespace[name]):
            candidate_fn = namespace[name]
            break
    if candidate_fn is None:
        for k, v in namespace.items():
            if callable(v) and not k.startswith("__") and v is not math:
                candidate_fn = v
                break
    if candidate_fn is None:
        raise ValueError("No callable `def priority(features: dict)` function found in code.")

    def wrapped(features: Dict[str, list]) -> List[float]:
        res = candidate_fn(features)
        return list(res)

    return wrapped
