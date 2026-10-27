import time
import numpy as np
import torch

from benchmarks.tsplib_parser import load_instance
from benchmarks.tsplib_aco_benchmark import (
    TSPLIB_DIR,
    run_aco,
    load_llm_heuristic,
)

INSTANCES = ["berlin52", "pr76"]
SEEDS = [0, 1, 2, 3, 4]

N_ANTS = 50
N_ITERATIONS = 100
ALPHA = 1.0
BETA = 2.0
DECAY = 0.8
DEVICE = "cpu"

def main():
    llm_heuristic = load_llm_heuristic()

    for name in INSTANCES:
        instance = load_instance(
            TSPLIB_DIR / f"{name}.tsp",
            (TSPLIB_DIR / f"{name}.opt.tour")
            if (TSPLIB_DIR / f"{name}.opt.tour").exists()
            else None,
        )
        distance_matrix = instance["distance_matrix"]
        optimal_cost = instance["optimal_cost"]

        methods = {
            "aco_baseline": lambda d: 1.0 / d,
            "aco_llm": llm_heuristic,
        }

        for method, heuristic in methods.items():
            for seed in SEEDS:
                start = time.perf_counter()
                result = run_aco(
                    distance_matrix,
                    heuristic,
                    seed=seed,
                    device=DEVICE,
                    n_ants=N_ANTS,
                    n_iterations=N_ITERATIONS,
                    alpha=ALPHA,
                    beta=BETA,
                    decay=DECAY,
                )
                elapsed = time.perf_counter() - start
                cost = result["cost"]
                gap = (
                    100 * (cost - optimal_cost) / optimal_cost
                    if optimal_cost is not None
                    else None
                )
                print(
                    f"{name} {method} seed={seed} "
                    f"cost={cost:.3f} elapsed={elapsed:.2f}s "
                    f"reported_runtime={result['runtime_sec']:.2f}s "
                    f"gap={gap}",
                    flush=True,
                )

if __name__ == "__main__":
    main()
