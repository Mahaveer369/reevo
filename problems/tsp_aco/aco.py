import torch


class ACO:
    """
    TSP Ant Colony Optimization.

    This implementation uses:
      - MAX-MIN Ant System (MMAS)
      - pheromone lower/upper bounds
      - nearest-neighbor candidate lists
      - global-best pheromone reinforcement
      - 2-opt local search

    The public constructor remains compatible with the original ReEvo TSP ACO.
    """

    def __init__(
        self,
        distances,
        heuristic,
        n_ants=30,
        decay=0.9,
        alpha=1.0,
        beta=1.0,
        device="cpu",
        candidate_list_size=20,
        local_search=True,
        p_best=0.05,
    ):
        self.device = torch.device(device)

        self.distances = (
            distances
            if isinstance(distances, torch.Tensor)
            else torch.tensor(distances, dtype=torch.float32, device=self.device)
        )

        self.distances = self.distances.to(
            device=self.device, dtype=torch.float32
        )

        self.heuristic = (
            heuristic
            if isinstance(heuristic, torch.Tensor)
            else torch.tensor(
                heuristic,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.heuristic = self.heuristic.to(
            device=self.device, dtype=torch.float32
        )

        self.problem_size = self.distances.shape[0]
        self.n_ants = int(n_ants)

        self.alpha = float(alpha)
        self.beta = float(beta)

        # IMPORTANT:
        # Existing ReEvo code calls this parameter "decay", meaning
        # pheromone retention. Therefore:
        #
        #   pheromone *= decay
        #
        # corresponds to conventional MMAS evaporation:
        #
        #   rho = 1 - decay
        #
        self.decay = float(decay)
        self.rho = 1.0 - self.decay

        # Keep rho in a numerically sensible range.
        self.rho = min(max(self.rho, 1e-4), 0.999)

        self.candidate_list_size = min(
            int(candidate_list_size),
            max(1, self.problem_size - 1),
        )

        self.local_search = bool(local_search)
        self.p_best = float(p_best)

        self.shortest_path = None
        self.lowest_cost = float("inf")

        # Build nearest-neighbor candidate lists.
        self.candidate_lists = self._build_candidate_lists()

        # Initialize pheromone from a nearest-neighbor tour.
        nn_tour = self._nearest_neighbor_tour()
        nn_cost = self._tour_cost(nn_tour)

        self.tau_max = 1.0 / (self.rho * max(nn_cost, 1e-8))

        # MMAS tau_min initialization using p_best.
        p = min(max(self.p_best, 1e-6), 0.999999)
        p_root = p ** (1.0 / max(self.problem_size, 2))

        denominator = (
            max(self.problem_size / 2.0 - 1.0, 1e-6)
            * p_root
        )

        tau_min = (
            self.tau_max
            * (1.0 - p_root)
            / max(denominator, 1e-6)
        )

        self.tau_min = max(
            tau_min,
            self.tau_max * 1e-6,
        )

        self.pheromone = torch.full_like(
            self.distances,
            float(self.tau_max),
        )

        # No self transitions.
        self.pheromone.fill_diagonal_(0.0)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_candidate_lists(self):
        """
        For each node, store its nearest unvisited neighbors.

        Candidate lists substantially reduce the branching factor while
        preserving a fallback to all unvisited nodes when necessary.
        """
        d = self.distances.clone()

        # Never choose the current node as its own successor.
        d.fill_diagonal_(float("inf"))

        _, nearest = torch.topk(
            d,
            k=self.candidate_list_size,
            dim=1,
            largest=False,
        )

        return nearest

    @torch.no_grad()
    def _nearest_neighbor_tour(self):
        """
        Construct a deterministic nearest-neighbor tour.
        """
        n = self.problem_size

        tour = torch.empty(
            n,
            dtype=torch.long,
            device=self.device,
        )

        visited = torch.zeros(
            n,
            dtype=torch.bool,
            device=self.device,
        )

        current = torch.tensor(
            0,
            dtype=torch.long,
            device=self.device,
        )

        tour[0] = current
        visited[current] = True

        for step in range(1, n):
            distances = self.distances[current].clone()
            distances[visited] = float("inf")

            current = torch.argmin(distances)
            tour[step] = current
            visited[current] = True

        return tour

    @torch.no_grad()
    def _tour_cost(self, tour):
        """
        Calculate closed-tour cost.
        """
        return float(
            self.distances[
                tour,
                torch.roll(tour, shifts=-1),
            ].sum().item()
        )

    # ------------------------------------------------------------------
    # Main ACO loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(int(n_iterations)):
            paths = self.gen_path(require_prob=False)

            # Apply local search before evaluating/reinforcing.
            if self.local_search:
                paths = self._apply_2opt(paths)

            costs = self.gen_path_costs(paths)

            iteration_best_cost, iteration_best_idx = costs.min(dim=0)

            # Update global best.
            if float(iteration_best_cost.item()) < self.lowest_cost:
                self.lowest_cost = float(iteration_best_cost.item())
                self.shortest_path = paths[
                    :,
                    iteration_best_idx,
                ].clone()

            # MMAS update.
            self.update_pheronome(
                paths,
                costs,
                iteration_best_idx,
            )

        return self.lowest_cost

    # ------------------------------------------------------------------
    # MMAS pheromone update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_pheronome(
        self,
        paths,
        costs,
        iteration_best_idx=None,
    ):
        """
        MMAS pheromone update.

        Evaporation:
            tau <- (1-rho) * tau

        Reinforcement:
            global-best tour only

        Pheromone is always clipped to [tau_min, tau_max].
        """

        # Evaporation.
        self.pheromone.mul_(1.0 - self.rho)

        if self.shortest_path is None:
            return

        # Global-best reinforcement.
        best_tour = self.shortest_path
        best_cost = max(float(self.lowest_cost), 1e-8)

        deposit = 1.0 / best_cost

        edges_a = best_tour
        edges_b = torch.roll(best_tour, shifts=-1)

        self.pheromone[edges_a, edges_b] += deposit
        self.pheromone[edges_b, edges_a] += deposit

        # Enforce MMAS bounds.
        self.pheromone.clamp_(
            min=self.tau_min,
            max=self.tau_max,
        )

        self.pheromone.fill_diagonal_(0.0)

    # ------------------------------------------------------------------
    # Tour construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def gen_path(self, require_prob=False):
        """
        Construct one tour for every ant.

        Returns:
            paths:
                shape = (problem_size, n_ants)
        """

        n = self.problem_size
        device = self.device

        start = torch.randint(
            low=0,
            high=n,
            size=(self.n_ants,),
            device=device,
        )

        mask = torch.ones(
            (self.n_ants, n),
            dtype=torch.bool,
            device=device,
        )

        ant_ids = torch.arange(
            self.n_ants,
            device=device,
        )

        mask[ant_ids, start] = False

        paths = [start]
        log_probs = []

        previous = start

        for _ in range(n - 1):
            actions, action_log_probs = self.pick_move(
                previous,
                mask,
                require_prob,
            )

            paths.append(actions)

            if require_prob:
                log_probs.append(action_log_probs)

            mask = mask.clone()
            mask[ant_ids, actions] = False

            previous = actions

        path_tensor = torch.stack(paths)

        if require_prob:
            return (
                path_tensor,
                torch.stack(log_probs),
            )

        return path_tensor

    @torch.no_grad()
    def pick_move(
        self,
        prev,
        mask,
        require_prob=False,
    ):
        """
        Choose the next city.

        Candidate-list cities are considered first.
        If no candidate-list city remains unvisited, all remaining
        cities are considered.
        """

        n_ants = prev.shape[0]
        device = self.device

        actions = torch.empty(
            n_ants,
            dtype=torch.long,
            device=device,
        )

        log_probs = torch.zeros(
            n_ants,
            dtype=torch.float32,
            device=device,
        )

        for ant in range(n_ants):
            current = int(prev[ant].item())

            candidate_nodes = self.candidate_lists[current]

            candidate_mask = mask[ant, candidate_nodes]

            if candidate_mask.any():
                choices = candidate_nodes[candidate_mask]
            else:
                choices = torch.nonzero(
                    mask[ant],
                    as_tuple=False,
                ).flatten()

            pheromone = self.pheromone[current, choices]
            heuristic = self.heuristic[current, choices]

            # Protect against numerical problems.
            pheromone = torch.clamp(
                pheromone,
                min=1e-12,
            )

            heuristic = torch.clamp(
                heuristic,
                min=1e-12,
            )

            weights = (
                pheromone.pow(self.alpha)
                * heuristic.pow(self.beta)
            )

            weights = torch.nan_to_num(
                weights,
                nan=0.0,
                posinf=1e30,
                neginf=0.0,
            )

            total = weights.sum()

            if (
                not torch.isfinite(total)
                or total <= 0
            ):
                probabilities = torch.ones_like(weights)
                probabilities /= probabilities.numel()
            else:
                probabilities = weights / total

            distribution = torch.distributions.Categorical(
                probs=probabilities
            )

            selected_idx = distribution.sample()

            actions[ant] = choices[selected_idx]

            if require_prob:
                log_probs[ant] = distribution.log_prob(
                    selected_idx
                )

        if require_prob:
            return actions, log_probs

        return actions, None

    # ------------------------------------------------------------------
    # Cost evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def gen_path_costs(self, paths):
        """
        Calculate closed-tour costs.

        paths shape:
            (problem_size, n_ants)
        """

        assert paths.shape == (
            self.problem_size,
            self.n_ants,
        )

        tours = paths.T

        next_nodes = torch.roll(
            tours,
            shifts=-1,
            dims=1,
        )

        return self.distances[
            tours,
            next_nodes,
        ].sum(dim=1)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 2-opt local search
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_2opt(self, paths):
        """Apply delta-based 2-opt to the five shortest tours."""
        if paths.shape != (self.problem_size, self.n_ants):
            raise ValueError("Unexpected paths shape")

        costs = self.gen_path_costs(paths)
        k = min(5, self.n_ants)
        best_ants = torch.argsort(costs)[:k]

        improved_paths = paths.clone()

        for ant_idx in best_ants.tolist():
            improved_paths[:, ant_idx] = self._two_opt(
                paths[:, ant_idx].clone()
            )

        return improved_paths

    @torch.no_grad()
    def _two_opt(self, tour):
        """First-improvement 2-opt for symmetric TSPs."""
        n = self.problem_size

        if n < 4:
            return tour

        while True:
            improved = False

            for i in range(1, n - 1):
                a = int(tour[i - 1].item())
                b = int(tour[i].item())

                for j in range(i + 1, n):
                    if i == 1 and j == n - 1:
                        continue

                    c = int(tour[j].item())
                    d = int(tour[(j + 1) % n].item())

                    delta = (
                        self.distances[a, c]
                        + self.distances[b, d]
                        - self.distances[a, b]
                        - self.distances[c, d]
                    )

                    if delta < 0:
                        tour[i:j + 1] = torch.flip(
                            tour[i:j + 1], dims=[0]
                        )
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                return tour
