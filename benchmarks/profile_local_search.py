import random
import time
import numpy as np
import torch

from benchmarks.tsplib_parser import load_instance
from benchmarks.tsplib_aco_benchmark import TSPLIB_DIR, prepare_distance_matrix
from problems.tsp_aco.aco import ACO

SEED = 42
N_ANTS = 20
N_ITERATIONS = 20

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    instance = load_instance(
        TSPLIB_DIR / "berlin52.tsp",
        TSPLIB_DIR / "berlin52.opt.tour",
    )

    dist = prepare_distance_matrix(instance["distance_matrix"])
    heuristic = 1.0 / dist
    heuristic = np.asarray(heuristic, dtype=np.float64) + 1e-9
    heuristic[heuristic < 1e-9] = 1e-9

    for use_local_search in [False, True]:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        aco = ACO(
            dist,
            heuristic,
            n_ants=N_ANTS,
            alpha=1.0,
            beta=2.0,
            decay=0.8,
            device="cpu",
            candidate_list_size=20,
            local_search=use_local_search,
        )

        start = time.perf_counter()
        objective = aco.run(N_ITERATIONS)
        elapsed = time.perf_counter() - start

        if isinstance(objective, torch.Tensor):
            cost = objective.item()
        else:
            cost = float(objective)

        print(
            f"local_search={use_local_search} "
            f"cost={cost:.3f} runtime={elapsed:.2f}s",
            flush=True,
        )

if __name__ == "__main__":
    main()
