import numpy as np


def calculate_margins(mask: np.ndarray, margin: int = 5):
    """
    Return a clipped ``(x0, x1, y0, y1, z0, z1)`` box around a mask.

    Coordinates use the historical project convention: x corresponds to axis 1,
    y to axis 0, and z to axis 2. Upper bounds are exclusive.

    Args:
        mask (np.ndarray): A three-dimensional binary mask.
        margin (int): The number of voxels to expand the bounding box in each direction.
    Returns:
        tuple: A tuple of six integers representing the bounding box coordinates:
            (x0, x1, y0, y1, z0, z1).

    """
    if mask.ndim != 3:
        raise ValueError("mask must be three-dimensional")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("mask must contain at least one foreground voxel")

    y0, x0, z0 = coordinates.min(axis=0)
    y1, x1, z1 = coordinates.max(axis=0) + 1
    return (
        max(0, int(x0 - margin)),
        min(mask.shape[1], int(x1 + margin)),
        max(0, int(y0 - margin)),
        min(mask.shape[0], int(y1 + margin)),
        max(0, int(z0 - margin)),
        min(mask.shape[2], int(z1 + margin)),
    )