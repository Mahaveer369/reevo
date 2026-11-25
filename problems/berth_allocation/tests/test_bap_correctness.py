"""
Unit test suite for the Berth Allocation System (BAP / CBAP / C/T-V BACAP / DCBAP).
Verifies mathematical feasibility, non-overlap constraints, time-variant quay crane
capacity limits, analytical lower bounds, rolling-horizon state transitions, and
solver superiority.
"""

from __future__ import annotations

import unittest

from problems.berth_allocation.eval import (
    DEFAULT_CUSTOM_BAP_CODE,
    evaluate_custom_bap_heuristic,
)
from problems.berth_allocation.instances import get_all_bap_instances, get_bap_instance
from problems.berth_allocation.model import (
    compute_analytical_lower_bound,
    verify_schedule,
)
from problems.berth_allocation.solvers import (
    BAP_SOLVER_REGISTRY,
    compare_rolling_horizon_strategies,
    solve_fifp_greedy,
    solve_reevo_vrpagent_bap,
)


class TestBerthAllocationSystem(unittest.TestCase):
    def test_all_instances_and_analytical_lower_bounds(self) -> None:
        instances = get_all_bap_instances()
        self.assertGreaterEqual(len(instances), 7)
        for name, inst in instances.items():
            lb = compute_analytical_lower_bound(inst)
            self.assertGreater(lb, 0.0, f"Lower bound must be positive for {name}")
            self.assertEqual(len(inst.vessels), len({v.id for v in inst.vessels}))

    def test_all_solvers_produce_100pct_feasible_schedules(self) -> None:
        test_names = ["chen26_table10", "chen26_n20", "xu25_bacap_20", "martin24_cbap_25"]
        for inst_name in test_names:
            inst = get_bap_instance(inst_name)
            lb = compute_analytical_lower_bound(inst)
            for solver_name, solver_fn in BAP_SOLVER_REGISTRY.items():
                sched = solver_fn(inst)
                ok, violations = verify_schedule(inst, sched)
                self.assertTrue(
                    ok,
                    f"Solver {solver_name} failed verification on {inst_name}: {violations}",
                )
                self.assertEqual(len(sched.scheduled_vessels), len(inst.vessels))
                self.assertGreaterEqual(
                    sched.total_objective + 1e-5,
                    lb,
                    f"Schedule objective {sched.total_objective} below analytical LB {lb}",
                )

    def test_reevo_vrpagent_strictly_beats_fifp_baseline(self) -> None:
        for inst_name in get_all_bap_instances():
            inst = get_bap_instance(inst_name)
            fifp = solve_fifp_greedy(inst)
            ours = solve_reevo_vrpagent_bap(inst)
            self.assertTrue(ours.feasible)
            self.assertLess(
                ours.total_objective,
                fifp.total_objective,
                f"Expected ReEvo-VRPAgent ({ours.total_objective}) < FIFP ({fifp.total_objective}) on {inst_name}",
            )

    def test_rolling_horizon_simulation_chen26(self) -> None:
        inst = get_bap_instance("chen26_table10")
        rh_res = compare_rolling_horizon_strategies(inst)
        strategies = {s["strategy"]: s for s in rh_res["strategies"]}
        self.assertIn("NR-FIFP", strategies)
        self.assertIn("TDBB-6h", strategies)
        self.assertIn("EDBB-Chen26", strategies)
        self.assertIn("ReEvo-ED-UB-BB", strategies)
        for s_name, s_data in strategies.items():
            self.assertTrue(s_data["feasible"], f"Rolling strategy {s_name} infeasible")
        self.assertLessEqual(
            strategies["ReEvo-ED-UB-BB"]["total_objective"],
            strategies["EDBB-Chen26"]["total_objective"],
        )
        self.assertLess(
            strategies["ReEvo-ED-UB-BB"]["total_objective"],
            strategies["NR-FIFP"]["total_objective"],
        )

    def test_custom_bap_heuristic_sandbox(self) -> None:
        res = evaluate_custom_bap_heuristic(
            DEFAULT_CUSTOM_BAP_CODE, instance_name="xu25_bacap_20"
        )
        self.assertTrue(res["custom_direct"]["feasible"])
        self.assertTrue(res["custom_with_vrpagent_ub_bb"]["feasible"])
        self.assertGreater(res["custom_with_vrpagent_ub_bb"]["improvement_vs_fifp_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
