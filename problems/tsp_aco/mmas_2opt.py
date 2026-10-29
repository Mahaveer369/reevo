"""
Improved TSP Max-Min Ant System (MMAS) + Heuristic-Guided Candidate-List 2-Opt Solver.

Why standard LLM-ACO + 2-Opt had mixed results:
  1. In upstream ReEvo (`problems/tsp_aco/eval.py`), heuristics are evolved WITHOUT 2-opt in the loop.
     The LLM therefore evolves an overly greedy edge prior (like `exp(-8 * z)`), which kills tour diversity
     once 2-opt is turned on.
  2. Standard top-5 2-opt builds its candidate list using only raw distance D[i, j], completely ignoring
     the LLM heuristic prior during local search.

How this module fixes TSP MMAS + 2-Opt:
  1. Uses true MMAS pheromone bounds [tau_min, tau_max] with global-best / iteration-best schedule.
  2. Builds the 2-opt candidate list from the TOP-K highest (pheromone^alpha * heuristic^beta) edges,
     so the LLM heuristic directly guides which 2-opt swaps are explored.
  3. Evolves / evaluates heuristics WITH 2-opt active in the evaluation loop.
"""

from __future__ import annotations
import math
import random
import time
from typing import Callable, List, Optional, Tuple


DistanceMatrix = List[List[float]]
HeuristicMatrix = List[List[float]]


def heuristic_inverse_distance(dist: DistanceMatrix) -> HeuristicMatrix:
    """Baseline heuristic: eta[i][j] = 1 / d[i][j]."""
    n = len(dist)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] > 1e-9:
                out[i][j] = 1.0 / dist[i][j]
    return out


def heuristic_qwen_mutual_knn_regret(dist: DistanceMatrix) -> HeuristicMatrix:
    """
    Improved TSP Heuristic for MMAS + 2-Opt:
    Instead of an overly greedy exp(-8*d) that collapses diversity, combines:
      1. Mutual k-nearest-neighbor rank boost (preserving non-myopic bridge edges between clusters),
      2. Second-nearest regret bonus (prioritizing nodes whose alternative connections are expensive).
    """
    n = len(dist)
    out = [[0.0] * n for _ in range(n)]

    # Compute each node's sorted neighbor distances and 2nd-best regret
    ranks = [[0] * n for _ in range(n)]
    regret = [0.0] * n
    for i in range(n):
        order = sorted((dist[i][j], j) for j in range(n) if j != i)
        if len(order) >= 2:
            regret[i] = order[1][0] - order[0][0]
        for r, (_d, j) in enumerate(order):
            ranks[i][j] = r + 1

    mean_regret = sum(regret) / float(max(1, n)) + 1e-9

    for i in range(n):
        for j in range(n):
            if i == j or dist[i][j] <= 1e-9:
                continue
            inv_d = 1.0 / dist[i][j]
            mutual_rank = 0.5 * (ranks[i][j] + ranks[j][i])
            rank_factor = math.exp(-mutual_rank / 6.0)
            regret_bonus = 1.0 + 0.25 * ((regret[i] + regret[j]) / (2.0 * mean_regret))
            out[i][j] = inv_d * (0.45 + 0.55 * rank_factor) * regret_bonus
    return out


def tour_cost(dist: DistanceMatrix, tour: List[int]) -> float:
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def candidate_restricted_two_opt(
    dist: DistanceMatrix,
    tour: List[int],
    candidates: List[List[int]],
    max_passes: int = 15,
) -> Tuple[List[int], float]:
    """
    Fast candidate-list 2-opt local search (O(K * n) per pass).
    For each node u = tour[i], only considers 2-opt swaps where u connects to v in candidates[u].
    """
    n = len(tour)
    cur_tour = list(tour)
    pos = [0] * n
    for idx, node in enumerate(cur_tour):
        pos[node] = idx

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            u = cur_tour[i]
            u_next = cur_tour[(i + 1) % n]
            d_u_unext = dist[u][u_next]

            for v in candidates[u]:
                j = pos[v]
                if j == i or j == (i + 1) % n or j == (i - 1) % n:
                    continue
                v_next = cur_tour[(j + 1) % n]
                delta = (dist[u][v] + dist[u_next][v_next]) - (d_u_unext + dist[v][v_next])
                if delta < -1e-6:
                    # Reverse segment between i+1 and j
                    a, b = (i + 1) % n, j
                    if a < b:
                        cur_tour[a : b + 1] = reversed(cur_tour[a : b + 1])
                        for k in range(a, b + 1):
                            pos[cur_tour[k]] = k
                    else:
                        # Wrap-around reversal: normalize so tour[0] stays fixed
                        cur_tour = cur_tour[i + 1 :] + cur_tour[: i + 1]
                        for k, node in enumerate(cur_tour):
                            pos[node] = k
                    improved = True
                    break
    return cur_tour, tour_cost(dist, cur_tour)


def run_mmas_tsp(
    dist: DistanceMatrix,
    heuristic_fn: Callable[[DistanceMatrix], HeuristicMatrix] = heuristic_qwen_mutual_knn_regret,
    n_ants: int = 16,
    n_iterations: int = 25,
    rho: float = 0.85,
    alpha: float = 1.0,
    beta: float = 2.0,
    candidate_k: int = 8,
    use_two_opt: bool = True,
    seed: int = 42,
) -> dict:
    """
    Max-Min Ant System (MMAS) with Heuristic-Guided Candidate-List 2-Opt.
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    n = len(dist)
    heu = heuristic_fn(dist)

    # Build top-K candidate lists per node ranked by heuristic prior
    candidates: List[List[int]] = []
    for i in range(n):
        ranked = sorted(
            (j for j in range(n) if j != i),
            key=lambda j: heu[i][j],
            reverse=True,
        )
        candidates.append(ranked[: min(candidate_k, n - 1)])

    # Initial pheromone bounds
    tau_max = 1.0
    tau_min = tau_max / (2.0 * n)
    pheromone = [[tau_max] * n for _ in range(n)]

    best_tour: Optional[List[int]] = None
    best_cost = float("inf")

    for it in range(n_iterations):
        iter_best_tour: Optional[List[int]] = None
        iter_best_cost = float("inf")

        for _ant in range(n_ants):
            start = rng.randrange(n)
            visited = [False] * n
            visited[start] = True
            tour = [start]
            cur = start

            for _step in range(n - 1):
                # Prefer unvisited nodes inside candidate list first
                pool = [j for j in candidates[cur] if not visited[j]]
                if not pool:
                    pool = [j for j in range(n) if not visited[j]]

                weights = [
                    (pheromone[cur][j] ** alpha) * (max(1e-9, heu[cur][j]) ** beta)
                    for j in pool
                ]
                tot = sum(weights)
                if tot <= 0:
                    nxt = rng.choice(pool)
                else:
                    r = rng.random() * tot
                    acc = 0.0
                    nxt = pool[-1]
                    for j_cand, w in zip(pool, weights):
                        acc += w
                        if r <= acc:
                            nxt = j_cand
                            break
                tour.append(nxt)
                visited[nxt] = True
                cur = nxt

            if use_two_opt:
                tour, cost = candidate_restricted_two_opt(dist, tour, candidates, max_passes=10)
            else:
                cost = tour_cost(dist, tour)

            if cost < iter_best_cost:
                iter_best_cost = cost
                iter_best_tour = tour

        if iter_best_cost < best_cost and iter_best_tour is not None:
            best_cost = iter_best_cost
            best_tour = list(iter_best_tour)

        # Update MMAS bounds [tau_min, tau_max]
        tau_max = 1.0 / (max(1e-3, 1.0 - rho) * best_cost)
        tau_min = tau_max / (2.0 * n)

        # Alternate iteration-best and global-best deposit
        deposit_tour = best_tour if (it % 3 == 0 and best_tour is not None) else iter_best_tour
        deposit_cost = best_cost if (it % 3 == 0) else iter_best_cost
        delta = 1.0 / max(1e-6, deposit_cost)

        for i in range(n):
            for j in range(n):
                pheromone[i][j] = max(tau_min, min(tau_max, pheromone[i][j] * rho))

        if deposit_tour is not None:
            for idx in range(n):
                u = deposit_tour[idx]
                v = deposit_tour[(idx + 1) % n]
                pheromone[u][v] = min(tau_max, pheromone[u][v] + delta)
                pheromone[v][u] = min(tau_max, pheromone[v][u] + delta)

    return {
        "best_cost": round(best_cost, 4),
        "best_tour": best_tour,
        "runtime_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
