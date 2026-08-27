"""Distance metrics for segmentation surfaces and clustered vectors."""

import numpy as np
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances


def compute_3d_bld(volume1: np.ndarray, volume2: np.ndarray):
    """
    Compute bidirectional local distances and corresponding surface points.

    Args:
        volume1 (np.ndarray): 3D binary array representing the first segmentation surface.
        volume2 (np.ndarray): 3D binary array representing the second segmentation surface.
    
    Returns:
        bld (np.ndarray): 3D array of the same shape as volume1 and volume2, containing the bidirectional local distances.
        matches (np.ndarray): 4D array of shape (X, Y, Z, 3) containing the coordinates of the corresponding surface points in volume2 for each point in volume1.
    
    
    """
    from skimage.measure import label

    if volume1.shape != volume2.shape:
        raise ValueError("volume1 and volume2 must have the same shape")
    components = []
    for volume in (volume1, volume2):
        labeled = label(volume)
        components.append([(labeled == index) for index in np.unique(labeled) if index != 0])
    volumes1, volumes2 = components
    if not volumes1 or not volumes2:
        return np.zeros(volume1.shape), np.zeros((*volume1.shape, 3))
    if len(volumes1) > 1 and len(volumes2) == 1:
        volumes1 = [np.any(volumes1, axis=0)]
    elif len(volumes2) > 1 and len(volumes1) == 1:
        volumes2 = [np.any(volumes2, axis=0)]

    bld = np.zeros(volume1.shape)
    matches = np.zeros((*volume1.shape, 3))
    for component1, component2 in zip(volumes1, volumes2):
        coords1 = np.argwhere(component1)
        coords2 = np.argwhere(component2)
        forward = distance.cdist(coords1, coords2)
        backward = distance.cdist(coords2, coords1)
        forward_min = forward.min(axis=1)
        forward_matches = coords2[forward.argmin(axis=1)]
        backward_min = backward.min(axis=1)
        backward_matches = coords1[backward.argmin(axis=1)]
        for index, coordinate in enumerate(coords1):
            candidates = np.where(np.all(backward_matches == coordinate, axis=1))[0]
            if len(candidates):
                backward_index = candidates[np.argmax(backward_min[candidates])]
                backward_value = backward_min[backward_index]
            else:
                backward_index, backward_value = None, -np.inf
            if forward_min[index] >= backward_value:
                value, match = forward_min[index], forward_matches[index]
            else:
                value, match = backward_value, coords2[backward_index]
            bld[tuple(coordinate)] = value
            matches[tuple(coordinate)] = match
    return bld, matches


def mean_intra_cluster_distance(features: np.ndarray, labels: np.ndarray) -> float:
    """
    
    Return the mean pairwise distance within non-singleton clusters.
    
    Args:
        features (np.ndarray): 2D array of shape (n_samples, n_features) containing the feature vectors to evaluate.
        labels (np.ndarray): 1D array of shape (n_samples,) containing the cluster labels for each sample. Noise points are labeled as -1.
    
    Returns:
        float: The mean intra-cluster distance, where lower values indicate better clustering quality.
        
    """
    distances = []
    for cluster in np.unique(labels):
        points = features[labels == cluster]
        if len(points) > 1:
            matrix = pairwise_distances(points)
            distances.append(matrix[np.triu_indices_from(matrix, k=1)].mean())
    return float(np.mean(distances)) if distances else float("nan")


compute_3d_BLD = compute_3d_bld