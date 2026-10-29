from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def parse_tsp_file(path: str | Path) -> dict:
    """
    Parse a standard TSPLIB TSP file with NODE_COORD_SECTION.

    Currently supports:
        - EUC_2D
        - ATT
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"TSP file not found: {path}")

    metadata: dict[str, str] = {}
    coordinates: dict[int, tuple[float, float]] = {}

    reading_coords = False

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue

            if line in {"EOF", "DISPLAY_DATA_SECTION"}:
                break

            if reading_coords:
                parts = line.split()

                if len(parts) < 3:
                    continue

                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])

                coordinates[node_id] = (x, y)

            elif ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

    if "DIMENSION" not in metadata:
        raise ValueError(f"DIMENSION missing in {path}")

    if "EDGE_WEIGHT_TYPE" not in metadata:
        raise ValueError(f"EDGE_WEIGHT_TYPE missing in {path}")

    dimension = int(metadata["DIMENSION"])
    edge_weight_type = metadata["EDGE_WEIGHT_TYPE"].upper()

    if len(coordinates) != dimension:
        raise ValueError(
            f"{path}: expected {dimension} coordinates, "
            f"found {len(coordinates)}"
        )

    # TSPLIB node IDs are generally 1..N.
    # We convert to zero-based NumPy indexing.
    ordered_coords = np.array(
        [coordinates[i] for i in range(1, dimension + 1)],
        dtype=np.float64,
    )

    return {
        "name": metadata.get("NAME", path.stem),
        "type": metadata.get("TYPE", ""),
        "dimension": dimension,
        "edge_weight_type": edge_weight_type,
        "coordinates": ordered_coords,
        "metadata": metadata,
    }


def tsplib_distance_matrix(
    coordinates: np.ndarray,
    edge_weight_type: str,
) -> np.ndarray:
    """
    Construct a TSPLIB-compatible distance matrix.

    Supported:
        EUC_2D
        ATT

    Returns:
        NxN integer-valued distance matrix.
    """

    edge_weight_type = edge_weight_type.upper()

    if edge_weight_type == "EUC_2D":
        diff = coordinates[:, None, :] - coordinates[None, :, :]
        euclidean = np.sqrt(np.sum(diff * diff, axis=2))

        # TSPLIB EUC_2D:
        # round to nearest integer using floor(x + 0.5)
        distance = np.floor(euclidean + 0.5).astype(np.int64)

        np.fill_diagonal(distance, 0)
        return distance

    if edge_weight_type == "ATT":
        diff = coordinates[:, None, :] - coordinates[None, :, :]
        squared = np.sum(diff * diff, axis=2)

        rij = np.sqrt(squared / 10.0)
        tij = np.floor(rij + 0.5)

        distance = np.where(
            tij < rij,
            tij + 1,
            tij,
        ).astype(np.int64)

        np.fill_diagonal(distance, 0)
        return distance

    raise NotImplementedError(
        f"EDGE_WEIGHT_TYPE '{edge_weight_type}' is not supported yet."
    )


def parse_opt_tour(path: str | Path) -> list[int]:
    """
    Parse a TSPLIB .opt.tour file.

    Returns:
        Zero-based node indices in tour order.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Tour file not found: {path}")

    tour: list[int] = []
    reading_tour = False

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line == "TOUR_SECTION":
                reading_tour = True
                continue

            if not reading_tour:
                continue

            if line == "-1":
                break

            if line == "EOF":
                break

            tour.append(int(line) - 1)

    return tour


def tour_cost(
    tour: list[int],
    distance_matrix: np.ndarray,
) -> int:
    """
    Calculate a closed TSP tour cost.

    The return edge from the final node back to the first
    node is included.
    """
    if not tour:
        raise ValueError("Tour is empty.")

    if len(set(tour)) != len(tour):
        raise ValueError("Tour contains duplicate nodes.")

    n = distance_matrix.shape[0]

    if len(tour) != n:
        raise ValueError(
            f"Tour contains {len(tour)} nodes, expected {n}."
        )

    cost = 0

    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        cost += int(distance_matrix[a, b])

    return cost


def load_instance(
    tsp_path: str | Path,
    opt_tour_path: str | Path | None = None,
) -> dict:
    """
    Complete loader for a TSPLIB TSP instance.
    """
    tsp = parse_tsp_file(tsp_path)

    distance_matrix = tsplib_distance_matrix(
        tsp["coordinates"],
        tsp["edge_weight_type"],
    )

    result = {
        **tsp,
        "distance_matrix": distance_matrix,
        "optimal_tour": None,
        "optimal_cost": None,
    }

    if opt_tour_path is not None and Path(opt_tour_path).exists():
        optimal_tour = parse_opt_tour(opt_tour_path)

        result["optimal_tour"] = optimal_tour
        result["optimal_cost"] = tour_cost(
            optimal_tour,
            distance_matrix,
        )

    return result


if __name__ == "__main__":
    base = Path(__file__).parent / "tsplib"

    examples = [
        ("att48", "att48.opt.tour"),
        ("berlin52", "berlin52.opt.tour"),
        ("pr76", "pr76.opt.tour"),
        ("kroA100", "kroA100.opt.tour"),
        ("st70", None),
        ("rat195", None),
    ]

    for name, opt_file in examples:
        tsp_path = base / f"{name}.tsp"
        opt_path = base / opt_file if opt_file else None

        instance = load_instance(tsp_path, opt_path)

        print("=" * 60)
        print(f"Instance        : {instance['name']}")
        print(f"Dimension       : {instance['dimension']}")
        print(f"Edge weight     : {instance['edge_weight_type']}")
        print(
            f"Optimal cost    : "
            f"{instance['optimal_cost']}"
        )

        print(
            "Distance matrix :",
            instance["distance_matrix"].shape,
        )
