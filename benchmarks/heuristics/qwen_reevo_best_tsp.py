import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:

    max_distance = np.max(distance_matrix)
    normalized_distances = distance_matrix / max_distance

    node_degrees = np.sum(distance_matrix > 0, axis=0)

    sparsity_factor = (
        (1 - normalized_distances)
        * (node_degrees / np.max(node_degrees))
    )

    sparse_matrix = np.where(
        sparsity_factor > 0.5,
        sparsity_factor,
        0
    )

    combined_priors = (
        sparse_matrix / (normalized_distances + 1e-6)
    )

    return combined_priors
