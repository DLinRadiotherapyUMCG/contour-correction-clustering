from skimage.draw import line_nd
import numpy as np

def toggle_voxel_and_neighbors(pred, gt, df_vectors, neighbor_mode='full'):
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