from itertools import product

import numpy as np
from skimage.draw import line_nd

NEIGHBOUR_OFFSETS = {
    "none": ((0, 0, 0),),
    "minimal": tuple(
        offset
        for offset in product((-1, 0, 1), repeat=3)
        if sum(abs(value) for value in offset) == 1
    ),
    "full": tuple(product((-1, 0, 1), repeat=3)),
}
NEIGHBOR_OFFSETS = NEIGHBOUR_OFFSETS


def create_contour_alternative(pred: np.ndarray, gt: np.ndarray, vectors: object, neighbor_mode: str = "full", neighbour_mode: str | None = None):
    """Adjust a prediction along edit-vector paths toward the reference mask.

    Args
        pred (np.ndarray): Predicted segmentation array.
        gt (np.ndarray): Ground truth segmentation array.
        vectors (pd.DataFrame): DataFrame containing edit vectors with required columns: "Origin_x", "Origin_y", "Origin_z", "End_x", "End_y", "End_z".
        neighbour_mode (str): Mode for considering neighboring voxels. Options are "none", "minimal", or "full".

    Returns
        np.ndarray: Adjusted prediction array with the same shape as `pred`.

    """
    if neighbour_mode is not None:
        neighbor_mode = neighbour_mode
    if neighbor_mode not in NEIGHBOR_OFFSETS:
        raise ValueError(f"Unknown neighbor_mode: {neighbor_mode}")

    required_columns = {
        "Origin_x",
        "Origin_y",
        "Origin_z",
        "End_x",
        "End_y",
        "End_z",
    }

    columns = set(getattr(vectors, "columns", ()))
    missing_columns = required_columns - columns
    if missing_columns:
        raise ValueError(f"vectors is missing columns: {sorted(missing_columns)}")

    refined = (pred != 0).astype(np.uint8) # Initialise the refined mask where the predicted mask is non-zero
    modified = np.zeros(pred.shape, dtype=bool)
    shape = pred.shape

    for start, end in zip(vectors[["Origin_x", "Origin_y", "Origin_z"]].to_numpy(), vectors[["End_x", "End_y", "End_z"]].to_numpy()):
        coordinates = line_nd(start, end, endpoint=True) # Get the coordinates of the line between the start and end points
        valid = (
            (coordinates[0] >= 0)
            & (coordinates[0] < shape[0])
            & (coordinates[1] >= 0)
            & (coordinates[1] < shape[1])
            & (coordinates[2] >= 0)
            & (coordinates[2] < shape[2])
        )
        # For each valid coordinate along the line, apply the neighbour offsets to refine the mask, 
        # neighbour offsets are used to include neighboring voxels based on the specified neighbour_mode
        for row, column, depth in zip(*(axis[valid] for axis in coordinates)):
            for row_offset, column_offset, depth_offset in NEIGHBOR_OFFSETS[neighbor_mode]:
                target = (row + row_offset, column + column_offset, depth + depth_offset)
                if all(0 <= index < size for index, size in zip(target, shape)) and not modified[target]:
                    refined[target] = np.uint8(gt[target] != 0)
                    modified[target] = True

    return np.where((refined == 1) & (gt == 0) & (pred == 0), 0, refined).astype(np.uint8)