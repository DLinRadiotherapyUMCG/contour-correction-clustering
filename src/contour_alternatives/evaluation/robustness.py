import numpy as np
import pandas as pd
from contour_alternatives.alternatives.generation import create_contour_alternative

import sys
sys.path.append(r'C:\Users\AalstJE\OneDrive - UMCG\Documents\PhD\Code\segmentation-evaluation\src')
from contour_alternatives.evaluation.geometric import geometricMeasures

def calculate_spread(points, x0, x1, y0, y1, z0, z1):
    """
    Helper function to calculate the spread of points in x, y, z dimensions.
    Returns the relative spread for each dimension.

    Args:
        points (np.ndarray): Array of shape (N, 3) containing the coordinates of the points.
        x0, x1, y0, y1, z0, z1 (float): The bounding box coordinates.
    
    Returns:
        tuple: Relative spread in x, y, z dimensions.
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    zmin = np.min(points[:, 2])
    zmax = np.max(points[:, 2])
    
    relative_spread_x = (xmax - xmin) / (x1 - x0)
    relative_spread_y = (ymax - ymin) / (y1 - y0)
    relative_spread_z = (zmax - zmin) / (z1 - z0)
    
    return relative_spread_x, relative_spread_y, relative_spread_z

def spatial_analysis_clusters(df_regions: pd.DataFrame, bounding_box: list):
    """
    Performs spatial analysis on clusters of regions defined in `df_regions`. For each cluster, 
    the function calculates various spatial properties, including the spread of the region's origin 
    and end points, as well as vector properties like radial distance and angles (phi, theta).

    Parameters:
    -----------
    df_regions : pd.DataFrame
        A DataFrame containing the region information, with columns including:
            - 'cluster': the cluster identifier
            - 'origin': the 3D coordinates of the origin point of the vector
            - 'end': the 3D coordinates of the end point of the vector
            - 'radial distance': the radial distance of the vector
            - 'elevation angle': the elevation angle of the vector
            - 'angle of rotation': the rotation angle of the vector

    bounding_box : list
        A list containing the coordinates defining the bounding box: [x0, x1, y0, y1, z0, z1]

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the computed spatial analysis results for each cluster, including:
            - Cluster: the cluster identifier
            - Size [voxels]: the number of voxels in the cluster
            - Relative spread values for x, y, and z coordinates for both origin and end points
            - Minimum and maximum values for radial distance, elevation angle (phi), and rotation angle (theta)
            - Relative spread values for edit length, theta, and phi angles for the cluster
    """
    
    if len(bounding_box) != 6:
        raise ValueError("Bounding box must have 6 elements: [x0, x1, y0, y1, z0, z1]")
    
    # Unpack bounding box
    x0, x1, y0, y1, z0, z1 = bounding_box
    
    # Global variables for the full dataset
    all_min_edit_length = np.min(df_regions['radial distance'])
    all_max_edit_length = np.max(df_regions['radial distance'])
    all_min_phi = np.min(df_regions['elevation angle'])
    all_max_phi = np.max(df_regions['elevation angle'])
    all_min_theta = np.min(df_regions['angle of rotation'])
    all_max_theta = np.max(df_regions['angle of rotation'])

    # List to collect result rows
    results = []

    for cluster in np.unique(df_regions['Cluster']):
        # Extract points for the current cluster
        points_origin = np.stack(df_regions[df_regions['Cluster'] == cluster]['origin'].values, axis=0)
        points_end = np.stack(df_regions[df_regions['Cluster'] == cluster]['end'].values, axis=0)

        # Number of voxels in the cluster
        no_voxels = len(points_origin)
        
        # Calculate the relative spread for origin and end points
        relative_spread_x_origin, relative_spread_y_origin, relative_spread_z_origin = calculate_spread(points_origin, x0, x1, y0, y1, z0, z1)
        relative_spread_x_end, relative_spread_y_end, relative_spread_z_end = calculate_spread(points_end, x0, x1, y0, y1, z0, z1)
        
        # Calculate spread in vector properties
        min_edit_length = np.min(df_regions[df_regions['Cluster'] == cluster]['radial distance'])
        max_edit_length = np.max(df_regions[df_regions['Cluster'] == cluster]['radial distance'])
        min_phi = np.min(df_regions[df_regions['Cluster'] == cluster]['elevation angle'])
        max_phi = np.max(df_regions[df_regions['Cluster'] == cluster]['elevation angle'])
        min_theta = np.min(df_regions[df_regions['Cluster'] == cluster]['angle of rotation'])
        max_theta = np.max(df_regions[df_regions['Cluster'] == cluster]['angle of rotation'])

        relative_edit_length = (max_edit_length - min_edit_length) / (all_max_edit_length - all_min_edit_length)
        relative_phi = (max_phi - min_phi) / (all_max_phi - all_min_phi)
        relative_theta = (max_theta - min_theta) / (all_max_theta - all_min_theta)

        # Collecting results
        results.append({
            'Cluster': cluster,
            'Size [voxels]': no_voxels,
            'Minimum x value origin:': np.min(df_regions[df_regions['Cluster'] == cluster]['Origin_x']),
            'Maximum x value origin': np.max(df_regions[df_regions['Cluster'] == cluster]['Origin_x']),
            'Minimum y value origin': np.min(df_regions[df_regions['Cluster'] == cluster]['Origin_y']),
            'Maximum y value origin': np.max(df_regions[df_regions['Cluster'] == cluster]['Origin_y']),
            'Minimum z value origin': np.min(df_regions[df_regions['Cluster'] == cluster]['Origin_z']),
            'Maximum z value origin': np.max(df_regions[df_regions['Cluster'] == cluster]['Origin_z']),
            'Mean x value origin': np.mean(df_regions[df_regions['Cluster'] == cluster]['Origin_x']),
            'Std x value origin': np.std(df_regions[df_regions['Cluster'] == cluster]['Origin_x']),
            'Mean y value origin': np.mean(df_regions[df_regions['Cluster'] == cluster]['Origin_y']),
            'Std y value origin': np.std(df_regions[df_regions['Cluster'] == cluster]['Origin_y']),
            'Mean z value origin': np.mean(df_regions[df_regions['Cluster'] == cluster]['Origin_z']),
            'Std z value origin': np.std(df_regions[df_regions['Cluster'] == cluster]['Origin_z']),

            'Minimum x value end:': np.min(df_regions[df_regions['Cluster'] == cluster]['End_x']),
            'Maximum x value end': np.max(df_regions[df_regions['Cluster'] == cluster]['End_x']),
            'Minimum y value end': np.min(df_regions[df_regions['Cluster'] == cluster]['End_y']),
            'Maximum y value end': np.max(df_regions[df_regions['Cluster'] == cluster]['End_y']),
            'Minimum z value end': np.min(df_regions[df_regions['Cluster'] == cluster]['End_z']),
            'Maximum z value end': np.max(df_regions[df_regions['Cluster'] == cluster]['End_z']),
            'Mean x value end': np.mean(df_regions[df_regions['Cluster'] == cluster]['End_x']),
            'Std x value end': np.std(df_regions[df_regions['Cluster'] == cluster]['End_x']),
            'Mean y value end': np.mean(df_regions[df_regions['Cluster'] == cluster]['End_y']),
            'Std y value end': np.std(df_regions[df_regions['Cluster'] == cluster]['End_y']),
            'Mean z value end': np.mean(df_regions[df_regions['Cluster'] == cluster]['End_z']),
            'Std z value end': np.std(df_regions[df_regions['Cluster'] == cluster]['End_z']),

            'Relative spread x origin': relative_spread_x_origin,
            'Relative spread y origin': relative_spread_y_origin,
            'Relative spread z origin': relative_spread_z_origin,
            'Relative spread x end': relative_spread_x_end,
            'Relative spread y end': relative_spread_y_end,
            'Relative spread z end': relative_spread_z_end,
            'Min edit length': min_edit_length,
            'Max edit length': max_edit_length,
            'Min edit angle (theta)': min_theta,
            'Max edit angle (theta)': max_theta, 
            'Min edit angle (phi)': min_phi,
            'Max edit angle (phi)': max_phi,
            'Relative spread edit length': relative_edit_length,
            'Relative spread edit angle (theta)': relative_theta,
            'Relative spread edit angle (phi)': relative_phi
        })

    # Convert results list to DataFrame
    df_results_spatial_analysis = pd.DataFrame(results)

    return df_results_spatial_analysis


import numpy as np
import pandas as pd
import scipy.ndimage as ndi

def volume_analysis_clusters(df_regions: pd.DataFrame, pred: np.ndarray, gt: np.ndarray, OAR: int):
    """
    Performs volume analysis on clusters, including changed volume, hole detection, and connectivity check.

    Parameters:
    -----------
    df_regions : pd.DataFrame
        A DataFrame containing the region information, with at least the following columns:
            - 'cluster': The cluster identifier
            - 'Origin_x', 'Origin_y', 'Origin_z': Coordinates of the origin of the vector path
            - 'End_x', 'End_y', 'End_z': Coordinates of the end of the vector path
    
    pred : np.ndarray
        The predicted volume (3D array).
    
    gt : np.ndarray
        The ground truth volume (3D array).
    
    OAR : int
        The Organ at Risk (OAR) identifier to be associated with the analysis.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the volume analysis results for each cluster, including:
            - 'changed_voxel_count': The number of changed voxels
            - 'changed_volume_ratio': The ratio of changed voxels to total voxels
            - 'hole_count': The number of holes detected in the changed region
            - 'num_components': The number of connected components in the changed volume
            - 'cluster': The cluster identifier
            - 'OAR': The Organ at Risk (OAR) identifier
    """
    
    # List to store results for each cluster
    df_results_volume_analysis = []  
    total_voxels = np.prod(pred.shape)  # Compute total voxels only once

    for cluster in np.unique(df_regions['Cluster']):
        # Extract the region of interest (ROI) for the current cluster
        df_edit_region = df_regions[df_regions['Cluster'] == cluster]
        
        results_volume_check = {}

        # Compute changed volume by toggling voxel states between predicted and ground truth
        pred_alt = create_contour_alternative(pred, gt, df_edit_region, neighbor_mode='full')
        changed_volume = (pred != 0).astype(int) != (pred_alt != 0).astype(int)

        # Calculate changed voxel count and ratio
        changed_voxel_count = np.sum(changed_volume)
        results_volume_check["changed_voxel_count"] = changed_voxel_count
        results_volume_check["changed_volume_ratio"] = changed_voxel_count / total_voxels

        # Hole detection: A region completely enclosed by unchanged voxels
        filled = ndi.binary_fill_holes(changed_volume)
        holes = filled & ~changed_volume
        hole_count = np.sum(holes)
        results_volume_check["hole_count"] = hole_count

        # Connectivity check (26-connectivity)
        structure_26 = np.ones((3, 3, 3), dtype=np.uint8)  # Full 26-connectivity
        _, num_components = ndi.label(changed_volume, structure=structure_26)
        results_volume_check["num_components"] = num_components
        print(f"This is {OAR} of type {type(OAR)}")
        # Ensure OAR is provided (integer check for clarity)
        if OAR is None:
            raise ValueError(f"Missing or invalid OAR: {OAR}. Please provide a valid integer identifier for OAR.")
        
        # Register cluster and OAR information
        results_volume_check["Cluster"] = cluster
        results_volume_check["OAR"] = OAR
        
        # Append the volume analysis results for the current cluster/OAR combination
        df_results_volume_analysis.append(results_volume_check)

    # Convert the results list to a DataFrame
    df_results_volume_analysis = pd.DataFrame(df_results_volume_analysis)
    
    return df_results_volume_analysis


def performance_improvement(alt_pred: np.ndarray, pred: np.ndarray, gt: np.ndarray, spacing: np.ndarray):
    """Compute the geometric measures for both original and alternative predictions

    Args:
        alt_pred (np.ndarray): Alternative prediction array.
        pred (np.ndarray): Original prediction array.
        gt (np.ndarray): Ground truth array.
        spacing (np.ndarray): Spacing of the image.
    
    Returns:
        dict: Dictionary containing the geometric measures for both original and alternative predictions.
    """

    # Compute the geometric measures for both original and alternative predictions
    og_accuracy = geometricMeasures(pred, gt, spacing)
    new_accuracy = geometricMeasures(alt_pred, gt, spacing)

    # Flatten the result dictionary
    results = {
        "Original_Volumetric Dice score": og_accuracy["Volumetric Dice score"],
        "Original_Hausdorff distance [mm]": og_accuracy["Hausdorff distance [mm]"],  # Assuming list with one value
        "Original_Hausdorff distance (95%) [mm]": og_accuracy["Hausdorff distance (95%) [mm]"],  # Assuming list with one value
        "Original_Surface Dice score": og_accuracy["Surface Dice score"],
        "Original_Added path length [pixels]": og_accuracy["Added path length [pixels]"]
    }

    results.update({
        "Alternative_Volumetric Dice score": new_accuracy["Volumetric Dice score"],
        "Alternative_Hausdorff distance [mm]": new_accuracy["Hausdorff distance [mm]"],  # Assuming list with one value
        "Alternative_Hausdorff distance (95%) [mm]": new_accuracy["Hausdorff distance (95%) [mm]"],  # Assuming list with one value
        "Alternative_Surface Dice score": new_accuracy["Surface Dice score"],
        "Alternative_Added path length [pixels]": new_accuracy["Added path length [pixels]"]
    })

    return results
