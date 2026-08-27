from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans


def _validate_features(features: np.ndarray):
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one sample")


def run_hdbscan(features: np.ndarray, min_cluster_size: int, min_samples: Optional[int] = None) :
    """
    Cluster feature rows with HDBSCAN and return one label per row.
    
    Args:
        features (np.ndarray): 2D array of shape (n_samples, n_features) containing the feature vectors to cluster.
        min_cluster_size (int): The minimum size of clusters; smaller clusters will be considered noise.
        min_samples (Optional[int]): The number of samples in a neighbourhood for a point to be considered a core point. 
                                     If None, defaults to the value of min_cluster_size.
    Returns:
        np.ndarray: 1D array of shape (n_samples,) containing the cluster labels for each sample. Noise points are labeled as -1.
    """
    _validate_features(features)

    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1")
    if min_samples is not None and min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, n_jobs=-1)

    return model.fit_predict(features)


def run_dbscan(features: np.ndarray, eps: float, min_samples: int):
    """
    Cluster feature rows with DBSCAN and return one label per row.
    
    Args: 
        features (np.ndarray): 2D array of shape (n_samples, n_features) containing the feature vectors to cluster.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighbourhood.
        min_samples (int): The number of samples in a neighbourhood for a point to be considered a core point.
    
    Returns:
        np.ndarray: 1D array of shape (n_samples,) containing the cluster labels for each sample. Noise points are labeled as -1.

    """
    _validate_features(features)
    if eps <= 0:
        raise ValueError("eps must be greater than zero")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)


def run_kmeans(features: np.ndarray, n_clusters: int, random_state: int = 42):
    """
    Cluster feature rows with KMeans and return one label per row.
    
    Args:
        features (np.ndarray): 2D array of shape (n_samples, n_features)
        n_clusters (int): The number of clusters to form.
        random_state (int): Determines random number generation for centroid initialisation
    Returns:
        np.ndarray: 1D array of shape (n_samples,) containing the cluster labels for each sample.
    """


    _validate_features(features)
    if n_clusters < 1 or n_clusters > features.shape[0]:
        raise ValueError("n_clusters must be between 1 and the number of samples")
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit_predict(features)
