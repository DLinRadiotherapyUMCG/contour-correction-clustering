from collections.abc import Mapping
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

from geometry.surfaces import compute_surface
from geometry.bounds import calculate_margins
from geometry.vectors import get_edit_vectors
from contour_alternatives.alternatives.regions import clusters_to_regions
from contour_alternatives.alternatives.generation import create_contour_alternative
from clustering.preprocessing import scale_features
from contour_alternatives.evaluation.robustness import (performance_improvement, spatial_analysis_clusters, volume_analysis_clusters)
from clustering.algorithms import run_dbscan, run_hdbscan, run_kmeans
from data_io.patient import load_patient_data

DEFAULT_CLUSTERING_PARAMETERS = {
    "weight_spatial": 12.58,
    "weight_radial_distance": 1.07,
    "weight_angle": 0.97,
    "min_cluster_size": 2,
    "min_samples": 4,
}

LOGGER = logging.getLogger(__name__)


def load_tuning_parameters(path: Path):
    """
    Load tuning results and validate required columns.

    Args:
        path (Path): Path to the tuning results CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the tuning results.
    
    """
    tuning = pd.read_csv(path, sep=";")
    required = {"Patient", "OAR", "score", *DEFAULT_CLUSTERING_PARAMETERS}
    missing = required - set(tuning.columns)
    if missing:
        raise ValueError(f"Tuning results are missing columns: {sorted(missing)}")
    return tuning


def select_clustering_parameters(patient: object, oar: int, tuning: pd.DataFrame | None, defaults: Mapping[str, float] = DEFAULT_CLUSTERING_PARAMETERS):
    """
    Select the highest-scoring tuned row or return defaults.
    
    Args:
        patient (object): Patient identifier.
        oar (int): OAR value.
        tuning (pd.DataFrame | None): DataFrame containing tuning results or None.
        defaults (Mapping[str, float]): Default clustering parameters.
    
    Returns:
        dict[str, float]: Dictionary of clustering parameters for the given patient and OAR.

    """
    if tuning is None:
        return dict(defaults)
    matches = tuning[(tuning["Patient"] == patient) & (tuning["OAR"] == oar)]
    if matches.empty:
        return dict(defaults)
    best = matches.loc[matches["score"].idxmax()]
    return {name: float(best[name]) for name in defaults}


def generate_contour_alternatives(config: dict):
    """
    Generate contour alternatives for every configured patient and OAR.
    
    Args:
        config (dict): Configuration dictionary containing workflow settings.
    
    Returns:
        dict[str, pd.DataFrame]: Dictionary containing DataFrames of spatial, volume, and accuracy results.
    """

    paths = config["paths"]

    tuning = None

    if config.get("use_tuned_hyperparameters", True):
        tuning = load_tuning_parameters(paths["tuning_results"])

    # Prepare lists to store results of spatial, volume, and accuracy analyses
    spatial_results = []
    region_results = []
    volume_results = []
    accuracy_results = []

    for ct_file in sorted(paths["ct"].glob("*_0000.nii*")):
        patient = ct_file.name.removesuffix("_0000.nii.gz").removesuffix("_0000.nii")
        LOGGER.info("Processing patient %s", patient)

        loaded = load_patient_data(ct_file.name, str(paths["ct"]), str(paths["ground_truth"]), str(paths["prediction"]))

        _, gt, pred, _, pred_affine, _, _, gt_header, _ = loaded
        bounding_box = calculate_margins(gt, config.get("margin", 5))

        for oar_value in np.unique(pred).astype(int):

            if oar_value == 0: # skip background
                continue
            
            parameters = select_clustering_parameters(patient, oar_value, tuning)
            filtered_edits, all_edits = get_edit_vectors(patient, gt, pred, oar_value, config.get("margin", 5), config.get("edit_threshold", 1.0))

            if len(filtered_edits) < 2:
                LOGGER.warning(f"Not enough edits for patient {patient}, OAR {oar_value}")
                continue

            features = scale_features(filtered_edits, list(config["spatial_features"]), list(config["edit_features"]), parameters["weight_spatial"], parameters["weight_radial_distance"], parameters["weight_angle"])
            min_samples = min(int(parameters["min_samples"]), len(features))

            clustering_algorithm = config.get("clustering_algorithm", "HDBSCAN")
            if clustering_algorithm == "DBSCAN":
                    labels = run_dbscan(features, config.get("dbscan_eps", 0.5), min_samples)
            elif clustering_algorithm == "HDBSCAN":
                    labels = run_hdbscan(features, 2, min_samples)
            elif clustering_algorithm == "KMeans":
                    labels = run_kmeans(features, 2)

            filtered_edits = filtered_edits.copy()
            filtered_edits["Cluster"] = labels

            edges = compute_surface((pred == oar_value).astype(np.uint8))

            regions = clusters_to_regions(filtered_edits, all_edits, edges, config.get("dilation_size", 2), noNoise=True)

            for cluster in np.unique(filtered_edits["Cluster"]):
                if cluster == -1: # skip noise
                    continue

                edit_region = regions[regions["Cluster"] == cluster]

                alternative = create_contour_alternative(pred, gt, edit_region, neighbour_mode="full")
                alternative_name = f"{patient}_{oar_value}_{cluster}" 
                nib.save(nib.Nifti1Image(alternative.astype(np.int32), pred_affine), paths["alternatives"] / f"{alternative_name}.nii.gz")

                changed = ((pred != 0) != (alternative != 0)).astype(np.int32)
                nib.save(nib.Nifti1Image(changed, pred_affine), paths["alternatives"] / f"{alternative_name}_changed.nii.gz")

                accuracy = performance_improvement(alternative, (pred != 0).astype(np.uint8), (gt != 0).astype(np.uint8), gt_header.get_zooms())
                accuracy_results.append({"Patient": patient, "OAR": oar_value, "Cluster": cluster, **accuracy})

            cluster_result = spatial_analysis_clusters(filtered_edits, bounding_box)
            cluster_result["Patient"], cluster_result["OAR"] = patient, oar_value
            spatial_results.append(cluster_result)
            region_result = spatial_analysis_clusters(regions, bounding_box)
            region_result["Patient"], region_result["OAR"] = patient, oar_value
            region_results.append(region_result)
            volume_result = volume_analysis_clusters(regions, pred, gt, oar_value)
            volume_result["Patient"] = patient
            volume_results.append(volume_result)

    results = {
        "clusters_spatial": pd.concat(spatial_results, ignore_index=True),
        "regions_spatial": pd.concat(region_results, ignore_index=True),
        "clusters_volume": pd.concat(volume_results, ignore_index=True),
        "accuracy": pd.DataFrame(accuracy_results),
        }

    with pd.ExcelWriter(paths["alternatives"] / "Statistics.xlsx") as writer:
        for sheet, frame in results.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    return results


def run_generation(config: dict):
    """Run the contour alternative generation workflow."""
    return generate_contour_alternatives(config)


