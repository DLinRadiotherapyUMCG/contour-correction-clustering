import numpy as np
from tqdm import tqdm
from skimage.morphology import isotropic_dilation
import pandas as pd
from skimage.draw import line_nd
import os

def clusters_to_regions(df_clusters: pd.DataFrame, df_edits: pd.DataFrame, edges_pred: np.ndarray, dilation_size: int = 2, noNoise: bool = True, ):

    """
    This function processes clustered data to create regions by applying isotropic dilation 
    and associating them with edit vectors from df_edits. The regions are stored in the 
    df_regions DataFrame.

    Args:
        df_clusters (DataFrame): The input dataframe containing clustered data with information 
                                  about the edit vectors for all surface voxels with edit vectors 
                                  whose edit length is greater than the threshold specified in the main code.
        df_edits (DataFrame): A DataFrame with details of edits for all surface voxels of the predicted segmentation. 
                              Necessary to copy information to the new dataframe.
        edges_pred (3D array): A 3D array representing the predicted surface (used to filter out non-relevant regions).
        noNoise (bool): Flag to indicate whether noise clusters (e.g., cluster labeled -1) should be removed. Default is True.
        dilation_size (int): The size of the isotropic dilation to be applied for expanding the regions. Default is 2.

    Returns:
        DataFrame: A dataframe with information on which voxels belong to each dilated cluster
    """

    # Create the columns dynamically by combining df_edits columns with 'cluster' and 'dilation size'
    # This ensures df_regions has the same structure as df_edits, plus additional columns for cluster and dilation size
    df_regions = pd.DataFrame(columns=list(df_edits.columns) + ['Cluster', 'Dilation size'])

    # Get the unique clusters from the df_clusters
    clusters = np.unique(df_clusters["Cluster"])

    # Optionally remove the noise clusters (e.g., -1 clusters) if noNoise is True
    if noNoise:
        df_clusters = df_clusters[df_clusters["Cluster"].isin(clusters[0:])]  # Only keep valid clusters
    
    # Get the bounds of the predicted edges array to define the voxel grid dimensions
    # This will be used for mapping the clustered points to the voxel grid
    x_min, x_max = 0, np.shape(edges_pred)[0]
    y_min, y_max = 0, np.shape(edges_pred)[1]
    z_min, z_max = 0, np.shape(edges_pred)[2]

    # Define the voxel grid shape, which is necessary for performing isotropic dilation
    grid_shape = (x_max - x_min, y_max - y_min, z_max - z_min)
    voxel_grid = np.zeros(grid_shape, dtype=int)  # Initialize a voxel grid with zeros

    # Map the clustered points into the voxel grid, assigning them a cluster ID
    # +1 is used to differentiate from the background (0) of the voxel_grid, and +2 for noisy clusters (which is assigned cluster -1)
    for i, row in df_clusters.iterrows():
        x, y, z, c = row["Origin_x"], row["Origin_y"], row["Origin_z"], row["Cluster"]
        if noNoise:
            voxel_grid[x, y, z] = c + 2  # Shift cluster IDs to avoid confusion with background
        else:
            voxel_grid[x,y,z] = c + 1 # Shift cluster IDs to avoid confusion with background, noisy channely (-1) becomes 0 and is removed skipped in the for loop below

    # Loop over each unique cluster in the voxel grid to perform region expansion
    for cluster in np.unique(voxel_grid)[1:]:  # Process all clusters, exclude background
        print(f"\n Processing cluster {cluster-1} / {len(clusters)}")
        
        # Create a temporary binary voxel grid for the current cluster
        voxel_grid_temp = np.where(voxel_grid == cluster, 1, 0)
        
        # Apply isotropic dilation to expand the region of the current cluster
        region_mask = isotropic_dilation(voxel_grid_temp, dilation_size)
        
        # Filter out the dilated region by keeping only the voxels that are part of the predicted edges
        region_dilated = np.where(edges_pred != 0, region_mask, 0)
        
        print(f"Region size before dilation: {np.unique_counts(voxel_grid_temp)}")
        print(f"Region size after dilation: {np.unique_counts(region_mask)}")
        print(f"Region size after applying edge filter: {np.unique_counts(region_dilated)}")

        # Keep track of which voxels are part of the new dilated region
        non_zero_voxels = np.argwhere(region_dilated)  # Extract the coordinates of non-zero voxels

        # Loop over each voxel in the dilated region to add to the dataframe that it was part of the dilated region
        for i, voxel in enumerate(non_zero_voxels):
            
            # Find the corresponding row in the df_edits DataFrame for the current voxel
            matching_row = df_edits.loc[df_edits['origin'].apply(set) == set(voxel)].copy()
            # Add cluster information and dilation size to the matching row
            # Reset cluster offset necessary for dilating
            if noNoise:
                cluster_offset = -2
            else:
                cluster_offset = -1
            matching_row['Cluster'] = cluster + cluster_offset
            matching_row['dilation size'] = dilation_size + cluster_offset
            
            # Add voxel to df_regions (storing the information in the new df)
            print(matching_row)
            row_dict = matching_row.iloc[0].to_dict()  # Convert row to dictionary
            df_regions.loc[len(df_regions)] = row_dict  # Add the row to the df_regions DataFrame
    
    return df_regions

def create_contour_alternative(pred, gt, df_vectors, neighbor_mode='full'):
    """
    Adjusts the predicted segmentation (`pred`) towards the ground truth (`gt`) 
    by modifying voxels along specified vector paths.

    Parameters:
        pred (np.ndarray): 3D binary/label prediction array.
        gt (np.ndarray): 3D binary/label ground truth array.
        df_vectors (pd.DataFrame): DataFrame with 'Origin_x', 'Origin_y', 'Origin_z', 
                                   'End_x', 'End_y', 'End_z' columns defining vector paths.
        neighbor_mode (str): Defines how many neighbors to consider. 
                             Options: 'none' (only line), 'minimal' (6-neighbors), 
                             'full' (26-neighbors, default).

    Returns:
        np.ndarray: The refined prediction array.
    """
    # Clone the prediction to avoid modifying it in-place
    refined_pred = (pred != 0).astype(int)  # Ensure binary format

    # Track modified voxels to avoid redundant updates
    modified_mask = np.zeros_like(refined_pred, dtype=bool)  

    # Define neighbor search based on mode
    neighbor_dict = {
        'none': [(0, 0, 0)],  # Only the line itself
        'minimal': [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],  # 6-connectivity
        'full': [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
                 (-1, -1, 0), (1, 1, 0), (-1, 1, 0), (1, -1, 0),
                 (-1, 0, -1), (1, 0, 1), (0, -1, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1),
                 (-1, -1, -1), (1, 1, 1), (-1, 1, 1), (1, -1, -1)] #26 connectivity
    }
    directions = neighbor_dict.get(neighbor_mode, neighbor_dict['full'])

    # Process all vectors at once
    for start, end in zip(df_vectors[['Origin_x', 'Origin_y', 'Origin_z']].values, 
                          df_vectors[['End_x', 'End_y', 'End_z']].values):
        rr, cc, zz = line_nd(start, end, endpoint=True)  # Get voxel coordinates along the line

        valid_mask = (rr >= 0) & (rr < pred.shape[0]) & \
                     (cc >= 0) & (cc < pred.shape[1]) & \
                     (zz >= 0) & (zz < pred.shape[2])

        # Modify voxels along the line + neighbors
        for r, c, z in zip(rr[valid_mask], cc[valid_mask], zz[valid_mask]):
            for dr, dc, dz in directions:
                nr, nc, nz = r + dr, c + dc, z + dz

                if 0 <= nr < pred.shape[0] and 0 <= nc < pred.shape[1] and 0 <= nz < pred.shape[2]:
                    if not modified_mask[nr, nc, nz]:  # Avoid redundant modifications
                        refined_pred[nr, nc, nz] = (gt[nr, nc, nz] > 0).astype(int)
                        modified_mask[nr, nc, nz] = True  # Mark as modified

    # Final cleanup: remove false positives outside GT
    refined_pred = np.where((refined_pred == 1) & (gt == 0) & (pred == 0), 0, refined_pred)

    # Return final refined prediction
    return refined_pred
