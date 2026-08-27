import numpy as np
import pandas as pd

from geometry.bounds import calculate_margins
from geometry.metrics import compute_3d_bld
from geometry.surfaces import compute_surface


def get_edit_vectors(patient, gt, pred, oar, margin, edit_threshold):
    """
    Return all edit vectors between the predicted and reference surfaces for a given OAR, along with a DataFrame containing all edit vectors (including those below the threshold).

    Args:
        patient (str): Patient identifier.
        gt (np.ndarray): Ground truth segmentation array.
        pred (np.ndarray): Predicted segmentation array.
        oar (int): Organ at risk label.
        margin (int): Margin size to add around the bounding box of the OAR.
        edit_threshold (float): Minimum length of edit vectors to include in the output.

    Returns:
        filtered_dataframe (pd.DataFrame): DataFrame containing all filterededit vectors and their properties.
        dataframe (pd.DataFrame): DataFrame containing all edit vectors (including those below the threshold).
    """
    calculate_margins(gt, margin)
    gt_oar = (gt == oar).astype(np.uint8)
    pred_oar = (pred == oar).astype(np.uint8)
    edges_gt = compute_surface(gt_oar)
    edges_pred = compute_surface(pred_oar)
    bld, matches = compute_3d_bld(edges_pred, edges_gt)
    voxel_indices = np.argwhere(edges_pred).astype(int)
    matched = matches[tuple(voxel_indices.T)].astype(int)
    valid = ~(matched == 0).all(axis=1)
    origins = voxel_indices[valid]
    ends = matched[valid]
    vectors = ends - origins
    lengths = np.linalg.norm(vectors, axis=1)
    xy = vectors[:, 0] ** 2 + vectors[:, 1] ** 2
    dataframe = pd.DataFrame({
        "Patient": patient, "OAR": oar,
        "origin": list(map(tuple, origins)), "end": list(map(tuple, ends)),
        "edit vector": list(map(tuple, vectors)), "edit length": lengths,
        "Origin_x": origins[:, 0], "Origin_y": origins[:, 1], "Origin_z": origins[:, 2],
        "End_x": ends[:, 0], "End_y": ends[:, 1], "End_z": ends[:, 2],
        "radial distance": np.sqrt(xy + vectors[:, 2] ** 2),
        "elevation angle": np.arctan2(vectors[:, 2], np.sqrt(xy)),
        "angle of rotation": np.arctan2(vectors[:, 1], vectors[:, 0]),
    })

    filtered_dataframe = dataframe[dataframe["edit length"] > edit_threshold].copy()
    return filtered_dataframe, dataframe
