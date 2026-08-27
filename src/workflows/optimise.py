import logging
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import davies_bouldin_score, silhouette_score

from geometry.vectors import get_edit_vectors
from data_io.patient import load_patient_data
from geometry.metrics import mean_intra_cluster_distance
from clustering.preprocessing import scale_features
from clustering.algorithms import run_hdbscan, run_dbscan, run_kmeans
from workflows.generate import DEFAULT_CLUSTERING_PARAMETERS

LOGGER = logging.getLogger(__name__)


def optimise_clustering(config: dict, n_trials: int = 50, clustering_algorithm: str = "HDBSCAN", optimisation_goal: str = "custom", alpha: float = 0.8):
    """Tune clustering parameters for every patient and OAR.

    The resulting CSV can be passed back to generation of the clusters (step 2) as
    ``paths.tuning_results``. 

    Args:
        config (dict): Configuration dictionary containing workflow settings.
        n_trials (int): Number of trials for hyperparameter optimisation.
        clustering_algorithm (str): Clustering algorithm to use, one of "HDBSCAN", "DBSCAN", or "KMeans".
        optimisation_goal (str): Optimisation goal, one of "silhouette", "db", or "custom".
        alpha (float): Weighting factor for custom optimisation, between 0 and 1.
    
    Returns:
        pd.DataFrame: DataFrame containing the tuning results for each patient and OAR.

    """

    paths = config["paths"]

    if optimisation_goal not in {"silhouette", "db", "custom"}:
        raise ValueError("optimisation_goal must be 'silhouette', 'db', or 'custom'")
    elif clustering_algorithm not in {"HDBSCAN", "DBSCAN", "KMeans"}:
        raise ValueError("clustering_algorithm must be 'HDBSCAN', 'DBSCAN', or 'KMeans'")

    spatial_features = list(config["spatial_features"])
    edit_features = list(config["edit_features"])

    rows = [] # List to store tuning results for each patient and OAR

    for ct_file in sorted(paths["ct"].glob("*_0000.nii*")):
        patient = ct_file.name.removesuffix("_0000.nii.gz").removesuffix("_0000.nii")
        loaded = load_patient_data(ct_file.name, str(paths["ct"]), str(paths["ground_truth"]), str(paths["prediction"]))

        _, gt, pred, *_ = loaded

        for oar in np.unique(pred).astype(int):

            if oar == 0: # skip background
                continue

            edits, _ = get_edit_vectors(patient, gt, pred, oar, config.get("margin", 5), config.get("edit_threshold", 1.0))

            if len(edits) < 2: # skip OARs with less than 2 edits, these cannot be clustered
                continue

            def objective(trial):
                weights = {
                    "weight_spatial": trial.suggest_float("weight_spatial", 1, 10),
                    "weight_radial_distance": trial.suggest_float("weight_radial_distance", 0.1, 2),
                    "weight_angle": trial.suggest_float("weight_angle", 0.1, 2),
                    }
                
                min_samples = trial.suggest_int("min_samples", 2, min(10, len(edits)))

                features = scale_features(edits, spatial_features, edit_features, **weights)

                if clustering_algorithm == "DBSCAN":
                    labels = run_dbscan(features, config.get("dbscan_eps", 0.5), min_samples)
                elif clustering_algorithm == "HDBSCAN":
                    labels = run_hdbscan(features, 2, min_samples)
                elif clustering_algorithm == "KMeans":
                    labels = run_kmeans(features, 2)

                mask = labels != -1 # Exclude noise points (i.e. -1 labels) for silhouette and Davies-Bouldin scores

                clusters = np.unique(labels[mask])

                # Compute the optimisation score based on the selected goal
                if len(clusters) < 2:
                    return float("-inf") if optimisation_goal != "db" else float("inf")
                if optimisation_goal == "silhouette":
                    return silhouette_score(features[mask], labels[mask])
                if optimisation_goal == "db":
                    return davies_bouldin_score(features[mask], labels[mask])
                spatial_score = (silhouette_score(edits.loc[mask, spatial_features], labels[mask]) + 1) / 2
                edit_score = 1 / (1 + mean_intra_cluster_distance(edits.loc[mask, edit_features].to_numpy(), labels[mask]))
                return alpha * spatial_score + (1 - alpha) * edit_score

            # Create an Optuna study and optimise the objective function for the specified number of trials
            direction = "minimize" if optimisation_goal == "db" else "maximize"
            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=n_trials)
            for trial in study.trials:
                rows.append({
                    "Patient": patient,
                    "OAR": oar,
                    "trial_number": trial.number,
                    "score": trial.value,
                    "min_cluster_size": 2,
                    "min_samples": trial.params.get("min_samples"),
                    "weight_spatial": trial.params.get("weight_spatial"),
                    "weight_radial_distance": trial.params.get("weight_radial_distance"),
                    "weight_angle": trial.params.get("weight_angle"),
                })

    results = pd.DataFrame(rows)
    output = paths.get("tuning_output") or (paths["alternatives"] / "custom_tuning_results.csv")

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output, index=False, sep=";")

    LOGGER.info("Wrote tuning results to %s", output)

    return results