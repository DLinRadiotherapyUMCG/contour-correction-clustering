import numpy as np
from scipy.ndimage import binary_dilation, grey_dilation, grey_erosion


def compute_surface(volume: np.ndarray, method: str = "6_connectivity"):
    """
    Return surface voxels for a three-dimensional mask.
    
    Args:
        volume (np.ndarray): Three-dimensional mask.
        method (str): Surface extraction method.

    Returns:
        np.ndarray: Surface voxels.

    """

    if method not in {"canny", "erosion_dilation", "6_connectivity"}:
        raise ValueError("method must be 'canny', 'erosion_dilation', or '6_connectivity'")

    mask = volume.astype(bool)

    if method == "canny":
        try:
            import itk
        except ImportError as error:
            raise ImportError("The 'canny' method requires the optional itk dependency") from error
        itk_image = itk.image_from_array(mask.astype(np.float32))
        canny_filter = itk.CannyEdgeDetectionImageFilter.New(Input=itk_image)
        canny_filter.Update()
        return itk.array_from_image(canny_filter.GetOutput()).astype(bool)
    elif method == "erosion_dilation":
        return (grey_dilation(mask, size=(3, 3, 3)) != grey_erosion(mask, size=(3, 3, 3)))
    elif method == "6_connectivity":
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, 1] = True
        structure[0, 1, 1] = True
        structure[2, 1, 1] = True
        structure[1, 0, 1] = True
        structure[1, 2, 1] = True
        structure[1, 1, 0] = True
        structure[1, 1, 2] = True

        return binary_dilation(mask, structure=structure) ^ mask  # 6-connectivity surface