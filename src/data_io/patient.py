import os
import nibabel as nib
# from pydicom import dcmread
# import glob
# from rt_utils import RTStructBuilder
# import pandas as pd
# import numpy as np
# from misc.paths import locateCTandRTSTRUCT, locateRTDOSE
# from misc.dicom import loadCTasSITK, loadRTDOSE, resampleRTDOSEasCT


# def calculate_margins(arr, margin=5):
#     """

#     Calculate the margins (bounding box with an offset) around non-zero elements in the array.

#     Parameters:
#         arr (np.ndarray): 3D binary array.
#         margin (int): Margin size to add around the bounding box.

#     Returns:
#         list: [x0, x1, y0, y1, z0, z1] representing the margin boundaries.
#     """
        
#     x0 = np.min(np.argwhere(arr)[:,1]) - margin
#     x1 = np.max(np.argwhere(arr)[:,1]) + margin
#     y0 = np.min(np.argwhere(arr)[:,0]) - margin
#     y1 = np.max(np.argwhere(arr)[:,0]) + margin
#     z0 = np.min(np.argwhere(arr)[:,2]) - margin
#     z1 = np.max(np.argwhere(arr)[:,2]) + margin
#     return [x0, x1, y0, y1, z0, z1]

# def create_case_patient_mapping(key_file, sheet_name):
#     """
#     Creates a mapping between Case IDs and Patient numbers from an Excel file.

#     Args:
#         key_file (str): Path to the Excel file containing the mapping.
#         sheet_name (str): Name of the sheet in the Excel file.

#     Returns:
#         pd.DataFrame: DataFrame with columns "Case ID" and "Patient number".
#     """
#     # Load the Excel file
#     df_key_file = pd.read_excel(io=key_file, sheet_name=sheet_name)

#     # Initialize the mapping DataFrame
#     df_key = pd.DataFrame(columns=["Case ID", "Patient number"])

#     # Populate the DataFrame
#     for _, row in df_key_file.iterrows():
#         if 'missing' in row['Filename CT'].lower():
#             df_key = pd.concat([df_key, pd.DataFrame([{'Case ID': 'missing', 'Patient number': row['Patient number']}])], ignore_index=True)
#         else:
#             case_id = int(row['Filename CT'].split('_')[1])
#             df_key = pd.concat([df_key, pd.DataFrame([{'Case ID': case_id, 'Patient number': int(row['Patient number'])}])], ignore_index=True)

#     return df_key


def load_patient_data(patientfile, CTMainNifti, TrueRTSTRUCTMain, PredRTSTRUCTMain):
    """
    Load CT, GT segmentation and predicted segmentation of a patient 

    Args:
        patientfile (str): Identifier of the patient(e.g., "Patient_001") that appears in the CT (Patient_001_0000.nii.gz) and True RTSTRUCT (Patient_001.nii.gz) and Predicted RTSTRUCT (Patient_001_pred.nii.gz) filenames.

    Returns:
        ct_arr (numpy.ndarray): CT image data.
        gt_arr (numpy.ndarray): Ground truth segmentation data
        pred_arr (numpy.ndarray): Predicted segmentation data
        ct.affine (numpy.ndarray): Affine transformation matrix for the CT image.
        gt_img.affine (numpy.ndarray): Affine transformation matrix for the ground truth segmentation.
        pred_img.affine (numpy.ndarray): Affine transformation matrix for the predicted segmentation.
        ct_header (nibabel.Nifti1Header): Header information for the CT image
        gt_header (nibabel.Nifti1Header): Header information for the ground truth segmentation.
        pred_header (nibabel.Nifti1Header): Header information for the predicted segmentation.

    """
    ct_file = os.path.join(CTMainNifti, patientfile)
    gt_file = os.path.join(TrueRTSTRUCTMain, patientfile.replace("_0000", ""))
    pred_file = os.path.join(PredRTSTRUCTMain, patientfile.replace("_0000", ""))
    
    if not os.path.exists(ct_file) or not os.path.exists(gt_file) or not os.path.exists(pred_file):
        print(f"Files for patient {patientfile} do not exist.")
        return None, None, None

    ct = nib.load(ct_file)
    ct_arr = ct.get_fdata()
    ct_header = ct.header
    gt_img = nib.load(gt_file)
    gt_arr = gt_img.get_fdata()
    gt_header = gt_img.header
    pred_img = nib.load(pred_file)
    pred_arr = pred_img.get_fdata()
    pred_header = pred_img.header

    return ct_arr, gt_arr, pred_arr, ct.affine, gt_img.affine, pred_img.affine, ct_header, gt_header, pred_header


# def load_patient(patient, CTMainNifti, TrueRTSTRUCTMain, PredRTSTRUCTMain, DICOMMain, dfKey, tumorFolder):
#     """
#     Load patient data including CT, ground truth, prediction, and dose information.

#     Args:
#         patient (int): Patient ID.
#         CTMainNifti (str): Path to the CT NIfTI files.
#         TrueRTSTRUCTMain (str): Path to the ground truth RTSTRUCT files.
#         PredRTSTRUCTMain (str): Path to the predicted RTSTRUCT files.
#         DICOMMain (str): Path to the DICOM directory.
#         dfKey (pd.DataFrame): DataFrame mapping Case ID to Patient number.

#     Returns:
#         dict: Dictionary containing loaded data arrays and dose information.
#     """
#     # File paths
#     ct_file = os.path.join(CTMainNifti, f'HNC-B_{patient}_0000.nii.gz')
#     gt_file = os.path.join(TrueRTSTRUCTMain, f'HNC-B_{patient}.nii.gz')
#     pred_file = os.path.join(PredRTSTRUCTMain, f'HNC-B_{patient}.nii.gz')

#     # Load NIfTI files
#     ct_nii = nib.load(ct_file)
#     gt_nii = nib.load(gt_file)
#     pred_nii = nib.load(pred_file)

#     # Extract data arrays
#     ct_arr = ct_nii.get_fdata()
#     gt_arr = gt_nii.get_fdata()
#     pred_arr = pred_nii.get_fdata()

#     # Get patient number
#     PatientNumber = str(dfKey[dfKey['Case ID'] == patient]['Patient number'].values[0])

#     # Locate DICOM directories
#     patientSubFolder = locateCTandRTSTRUCT(os.path.join(DICOMMain, PatientNumber))
#     dicomDirCT = os.path.join(DICOMMain, PatientNumber, patientSubFolder, "CT")
#     doseFolder = locateRTDOSE(os.path.join(DICOMMain, PatientNumber))
#     dicomDirDose = os.path.join(DICOMMain, PatientNumber, doseFolder, 'RTDOSE')

#     # Load CT and dose data
#     ct = loadCTasSITK(dicomDirCT)
#     ds = dcmread(os.path.join(dicomDirDose, os.listdir(dicomDirDose)[0]))
#     rtdose = loadRTDOSE(dicomDirDose)

#     # Resample dose to match CT
#     rtdose_dtype = 'float32'
#     resampled_arrRTDOSE, arr_CT, arr_RTDOSE, resampled_imgRTDOSE = resampleRTDOSEasCT(rtdose, ct)
#     resampled_arrRTDOSE = (resampled_arrRTDOSE * float(ds.DoseGridScaling)).astype(rtdose_dtype)

#     # Load rtstruct
#     tumor_files = glob.glob(tumorFolder, recursive=True)
#     tumor_file = tumor_files[0]
#     rtstruct = RTStructBuilder.create_from(dicom_series_path=dicomDirCT, rt_struct_path=tumor_file)

#     # Return all loaded data
#     return {
#         "ct_arr": ct_arr,
#         "gt_arr": gt_arr,
#         "pred_arr": pred_arr,
#         "resampled_arrRTDOSE": resampled_arrRTDOSE,
#         "rtstruct": rtstruct
#     }

# def is_valid_roi(rtstruct, roi_name):
#     try:
#         mask = rtstruct.get_roi_mask_by_name(roi_name)
#         return mask.any()
#     except Exception:
#         print(f"{roi_name} is an invalid structure")
#         return False
