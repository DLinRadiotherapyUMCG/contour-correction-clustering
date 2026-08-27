import numpy as np
from scipy.spatial.distance import cdist, pdist


def compute_spatial_cohesion(features, labels):
    """
    Evaluate compactness relative to separation between non-noise clusters.
    
    Args:
        features (np.ndarray): 2D array of shape (n_samples, n_features) containing the feature vectors to evaluate.
        labels (np.ndarray): 1D array of shape (n_samples,) containing the cluster labels for each sample. Noise points are labeled as -1.
    
    Returns:
        float: The spatial cohesion score, where higher values indicate better clustering quality.
    
    """
    unique_clusters = np.unique(labels[labels != -1])
    if len(unique_clusters) < 2:
        return 0

    total_intra_distance = 0
    for cluster in unique_clusters:
        cluster_points = features[labels == cluster]
        if len(cluster_points) > 1:
            intra_distances = pdist(cluster_points, metric="euclidean")
            score = (1 / (1 + np.mean(intra_distances))) * len(cluster_points)
            if np.max(intra_distances) > 10:
                score -= len(cluster_points) * 0.1
            total_intra_distance += score

    intra_distance_score = total_intra_distance / len(unique_clusters)
    total_inter_distance = 0
    number_of_pairs = 0
    for index, cluster_i in enumerate(unique_clusters):
        for cluster_j in unique_clusters[index + 1:]:
            inter_distances = cdist(features[labels == cluster_i], features[labels == cluster_j])
            total_inter_distance += np.sum(inter_distances)
            number_of_pairs += inter_distances.size
    if number_of_pairs == 0:
        return 0
    inter_distance = total_inter_distance / number_of_pairs
    return 0 if inter_distance == 0 else intra_distance_score / inter_distance


def compute_relative_feature_coherence(features, labels):
    """
    Compute average within-cluster variance relative to total variance.
    
    Args:
        features (np.ndarray): 2D array of shape (n_samples, n_features) containing the feature vectors to evaluate.
        labels (np.ndarray): 1D array of shape (n_samples,) containing the cluster labels for each sample. Noise points are labeled as -1.
    
    Returns:
        float: The relative feature coherence score, where higher values indicate better clustering quality.
    
    """
    total_variance = np.var(features, axis=0)
    total_variance = np.where(total_variance == 0, 1, total_variance)
    scores = []
    for cluster in np.unique(labels):
        if cluster != -1:
            scores.append(np.mean(np.var(features[labels == cluster], axis=0) / total_variance))
    return float(np.mean(scores)) if scores else float("nan")
