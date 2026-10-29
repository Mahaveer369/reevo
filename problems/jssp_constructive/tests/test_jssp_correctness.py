"""Unit test suite for JSSP parser, independent feasibility validator, Giffler-Thompson scheduler, and Critical-Path LS."""

from __future__ import annotations
import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import hydra
from omegaconf import DictConfig
from problems.jssp_constructive.parser import (
    generate_training_instances,
    list_benchmark_instances,
    load_benchmark_instance,
)
from problems.jssp_constructive.schedule import (
    ScheduleResult,
    evaluate_machine_sequences,
    validate_schedule,
)
from problems.jssp_constructive.baselines import (
    CLASSICAL_RULES,
    rule_mwkr,
    rule_pt_plus_winq,
    rule_spt,
)
from problems.jssp_constructive.scheduler import build_active_schedule_gt
from problems.jssp_constructive.local_search import (
    critical_path_local_search,
    multi_start_hybrid_solve,
)
from problems.jssp_constructive.reevo_heuristics import (
    REEVO_QWEN_CODE,
    compile_custom_priority_code,
    rule_reevo_qwen,
)


class TestJSSPCorrectness(unittest.TestCase):

    def test_hydra_and_omegaconf_compat(self) -> None:
        cfg = hydra.compose_config(config_path=os.path.join(ROOT_DIR, "cfg"), config_name="config")
        self.assertIsInstance(cfg, DictConfig)
        self.assertEqual(cfg.algorithm, "reevo")
        self.assertIn("problem", cfg)

    def test_orlib_parser_integrity(self) -> None:
        ft06 = load_benchmark_instance("ft06")
        self.assertEqual(ft06.n_jobs, 6)
        self.assertEqual(ft06.n_machines, 6)
        self.assertEqual(ft06.optimum, 55)

        ft10 = load_benchmark_instance("ft10")
        self.assertEqual(ft10.n_jobs, 10)
        self.assertEqual(ft10.n_machines, 10)
        self.assertEqual(ft10.optimum, 930)

        all_insts = list_benchmark_instances()
        self.assertGreaterEqual(len(all_insts), 12)

    def test_giffler_thompson_active_schedules_100_percent_feasible(self) -> None:
        for name in ("ft06", "ft10", "la01", "orb01", "abz5"):
            inst = load_benchmark_instance(name)
            for rule_name, rule_fn in CLASSICAL_RULES.items():
                sched = build_active_schedule_gt(inst, priority_fn=rule_fn, solver_name=rule_name, validate=True)
                self.assertIsNotNone(sched.validation)
                self.assertTrue(
                    sched.validation.feasible,
                    f"Schedule for {name} under {rule_name} failed validation: {sched.validation.errors}",
                )
                self.assertGreaterEqual(sched.makespan, inst.optimum or inst.lower_bound)

    def test_independent_validator_catches_violations(self) -> None:
        inst = load_benchmark_instance("ft06")
        valid_sched = build_active_schedule_gt(inst, priority_fn=rule_mwkr, validate=True)
        self.assertTrue(valid_sched.validation.feasible)

        # Inject a precedence / overlap violation by shifting an operation's start time to 0
        corrupt_starts = dict(valid_sched.start_times)
        corrupt_ends = dict(valid_sched.end_times)
        corrupt_starts[(0, 3)] = 0
        corrupt_ends[(0, 3)] = inst.durations[0][3]
        corrupt_sched = ScheduleResult(
            instance_name=inst.name,
            solver_name="Corrupt",
            makespan=valid_sched.makespan,
            start_times=corrupt_starts,
            end_times=corrupt_ends,
            machine_sequences=valid_sched.machine_sequences,
        )
        rep = validate_schedule(inst, corrupt_sched)
        self.assertFalse(rep.feasible)
        self.assertFalse(rep.precedence_ok)
        self.assertGreater(len(rep.errors), 0)

    def test_critical_path_local_search_improves_and_reaches_ft06_optimum(self) -> None:
        inst = load_benchmark_instance("ft06")
        init_sched = build_active_schedule_gt(inst, priority_fn=rule_spt, validate=True)
        refined = critical_path_local_search(inst, init_sched, max_steps=60, validate=True)
        self.assertTrue(refined.validation.feasible)
        self.assertLessEqual(refined.makespan, init_sched.makespan)

        hybrid = multi_start_hybrid_solve(
            inst,
            priority_fn=rule_reevo_qwen,
            n_starts=10,
            ls_steps=40,
            seed=42,
        )
        self.assertTrue(hybrid.validation.feasible)
        self.assertEqual(hybrid.makespan, 55)  # Known optimum C* = 55 for ft06

    def test_reevo_qwen_beats_classical_single_pass_average(self) -> None:
        test_names = ["ft06", "ft10", "ft20", "la01", "la02", "la03", "la04", "la05", "orb01", "abz5", "abz6"]
        mwkr_gaps = []
        spt_gaps = []
        reevo_gaps = []
        for name in test_names:
            inst = load_benchmark_instance(name)
            s_mwkr = build_active_schedule_gt(inst, priority_fn=rule_mwkr, validate=True)
            s_spt = build_active_schedule_gt(inst, priority_fn=rule_spt, validate=True)
            s_reevo = build_active_schedule_gt(inst, priority_fn=rule_reevo_qwen, validate=True)
            self.assertTrue(s_reevo.validation.feasible)
            mwkr_gaps.append(s_mwkr.gap_percent(inst.optimum))
            spt_gaps.append(s_spt.gap_percent(inst.optimum))
            reevo_gaps.append(s_reevo.gap_percent(inst.optimum))

        self.assertLess(sum(reevo_gaps) / len(reevo_gaps), sum(mwkr_gaps) / len(mwkr_gaps))
        self.assertLess(sum(reevo_gaps) / len(reevo_gaps), sum(spt_gaps) / len(spt_gaps))

    def test_custom_code_compiler(self) -> None:
        fn = compile_custom_priority_code(REEVO_QWEN_CODE)
        inst = load_benchmark_instance("la01")
        sched = build_active_schedule_gt(inst, priority_fn=fn, validate=True)
        self.assertTrue(sched.validation.feasible)


if __name__ == "__main__":
    unittest.main()
