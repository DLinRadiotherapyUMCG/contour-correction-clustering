import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_features(dataframe: pd.DataFrame, spatial_features: list[str], edit_features: list[str], weight_spatial: float, weight_radial_distance: float, weight_angle: float):
    """
    Scale selected columns and apply spatial/edit feature weights.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the features to be scaled.
        spatial_features (list[str]): List of column names corresponding to spatial features.
        edit_features (list[str]): List of column names corresponding to edit features.
        weight_spatial (float): Weighting factor for spatial features.
        weight_radial_distance (float): Weighting factor for radial distance feature.
        weight_angle (float): Weighting factor for angle feature.
    
    Returns:
        scaled (np.ndarray): Scaled feature array with applied weights.

    """
    feature_names = [*spatial_features, *edit_features]

    if not feature_names:
        raise ValueError("At least one feature is required")

    missing = set(feature_names) - set(dataframe.columns)

    if missing:
        raise ValueError(f"Missing clustering features: {sorted(missing)}")
    
    selected = dataframe[feature_names]

    scaled = MinMaxScaler().fit_transform(selected)

    for feature in spatial_features:
        scaled[:, selected.columns.get_loc(feature)] *= weight_spatial
    for feature in edit_features:
        index = selected.columns.get_loc(feature)
        scaled[:, index] *= weight_radial_distance if feature == "radial distance" else weight_angle

    return scaled
