import seg_metrics.seg_metrics as sg
import surface_distance
import numpy as np

def getEdgeOfMask(mask):
    '''
    Computes and returns edge of a segmentation mask

    Args:
        mask (ndarray): A binary NumPy array representing the segmentation mask. with 0 = background
    
    Returns:
        edge (ndarray): A binary NumPy array representing the edge of the segmentation mask. with 0 = background

    '''
    # edge has the pixels which are at the edge of the mask
    edge = np.zeros_like(mask)
    
    # mask_pixels has the pixels which are inside the mask of the automated segmentation result
    mask_pixels = np.where(mask > 0)

    for idx in range(0,mask_pixels[0].size):

        x = mask_pixels[0][idx]
        y = mask_pixels[1][idx]
        z = mask_pixels[2][idx]

        # Count # pixels in 3x3 neighborhood that are in the mask
        # If sum < 27, then (x, y, z) is on the edge of the mask
        if mask[x-1:x+2, y-1:y+2, z-1:z+2].sum() < 27:
            edge[x,y,z] = 1
            
    return edge

def AddedPathLength(seg1, seg2):
    '''
    Returns the added path length, in pixels
    
    Steps:
    1. Find pixels at the edge of the mask for both seg1 and seg2
    2. Count # pixels on the edge of seg2 that are not in the edge of seg1

    Args:
        seg1 (ndarray): A binary NumPy array representing the first segmentation mask. with 0 = background
        seg2 (ndarray): A binary NumPy array representing the second segmentation mask. with 0 = background
    
    Returns:
        apl (int): The added path length, in pixels
    '''
    
    # Check if seg1 and seg2 have same dimensions. If not, then raise a ValueError
    if seg1.shape != seg2.shape:
        raise ValueError('Shape of seg1 and seg2 must be identical!')

    # edge_auto has the pixels which are at the edge of the automated segmentation result
    edge_auto = getEdgeOfMask(seg1)
    # edge_gt has the pixels which are at the edge of the ground truth segmentation
    edge_gt = getEdgeOfMask(seg2)
    
    # Count # pixels on the edge of seg2 that are on not in the edge of seg1
    apl = (edge_gt > edge_auto).astype(int).sum()
    
    return apl 


def compute_dsc(seg1, seg2):
    """
    Computes the Dice Similarity Coefficient (DSC) between two binary segmentation masks.
    NOTE: we do not use the dsc from seg_metrics because it takes way too long (> 6 minutes), I don't know why it takes so long.

    Args:
        seg1 (ndarray): A binary NumPy array representing the first segmentation mask. with 0 = background
        seg2 (ndarray): A binary NumPy array representing the second segmentation mask. with 0 = background

    Returns:
        float: The Dice Similarity Coefficient, a value between 0 and 1.
    """
    seg1 = np.where(seg1 > 0, 1, 0).astype(int)  # Set non-zero values to 1, 0 remains 0
    seg2 = np.where(seg2 > 0, 1, 0).astype(int)  # Set non-zero values to 1, 0 remains 0

    dsc = np.sum(seg1[seg2==1])*2.0 / (np.sum(seg1) + np.sum(seg2))
    return dsc

def geometricMeasures(seg1, seg2, spacing): 
    """
    Computes various geometric similarity measures between two binary segmentation masks.

    This function calculates:
    - Dice Similarity Coefficient (DSC)
    - Hausdorff Distance (HD) and 95th percentile HD (HD95)
    - Surface Dice Score at 1mm tolerance
    - Added Path Length (APL)

    Args:
        seg1 (ndarray): A binary NumPy array representing the first segmentation mask (ground truth).
        seg2 (ndarray): A binary NumPy array representing the second segmentation mask (prediction).
        spacing (tuple or list): The voxel spacing in each dimension, required for surface distance computations.

    Returns:
        dict: A dictionary containing the following geometric measures:
            - 'Volumetric Dice score' (float): Dice Similarity Coefficient.
            - 'Hausdorff distance [mm]' (float): Maximum Hausdorff distance.
            - 'Hausdorff distance (95%) [mm]' (float): 95th percentile Hausdorff distance.
            - 'Surface Dice score' (float): Surface Dice score at 1mm tolerance.
            - 'Added path length [pixels]' (float): Added path length metric.
    """

    ##### function to compute the dice score, hd distance, surface dice and added path length
    ##### input:    seg 1 and seg2 should be numpy arrays 
    ##### output:   array contianing the four geometric metrics
    
    #compute volumetric measures
    measures = sg.write_metrics(labels=[1],  # exclude background
                      gdth_img=seg1,
                      pred_img=seg2, metrics=['hd', 'hd95'])
    dice = compute_dsc(seg1, seg2)

    #print(measures)  # a list of dictionaries which includes the metrics for each pair of image.
    ##### compute surface measures
    surface_distances = surface_distance.compute_surface_distances(seg1.astype(bool), seg2.astype(bool), spacing_mm= spacing)

    expected_distances = {
        'surfel_areas_gt': surface_distances["surfel_areas_gt"],
        'surfel_areas_pred': surface_distances["surfel_areas_pred"],
        'distances_gt_to_pred': surface_distances["distances_gt_to_pred"],
        'distances_pred_to_gt': surface_distances["distances_pred_to_gt"],
    }

    structDict = { # r'Parotid_L_pCT\w+->\w+CT\w+'
        "average_surface_distance" : (surface_distance.compute_average_surface_distance(expected_distances)),
        "surface_overlap_at_1mm" : (surface_distance.compute_surface_overlap_at_tolerance(expected_distances, tolerance_mm=1)),
        "surface_dice_at_1mm" : (surface_distance.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=1))
        }

    surface_measures = [structDict]

    #print(surface_measures)

    ##### compute added path length 
    apl = AddedPathLength(seg1, seg2)


    ##### measures summary
    measures_summary = {
        'Volumetric Dice score' : dice,
        'Hausdorff distance [mm]' : measures[0]['hd'][0],
        'Hausdorff distance (95%) [mm]' : measures[0]['hd95'][0],
        'Surface Dice score': surface_measures[0]['surface_dice_at_1mm'],
        'Added path length [pixels]': apl
        }
    
    #print(measures_summary)

    return measures_summary

