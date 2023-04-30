import pandas as pd

# to install pycytominer, run `poetry run pip install <path_to_pycytominer>`
from pycytominer import normalize, feature_select

config = {
    "MAD_EPSILON": 0.0,
    "INC_IMAGE_FEATURES": True,
    "FEAT_SELECT_OPS": ["variance_threshold", "correlation_threshold", "drop_na_columns", "blocklist"]
}

def preprocess_profiles(ann_dframe, remove_nan_rows=True):

    ann_dframe = normalize(ann_dframe, method="mad_robustize",
                           mad_robustize_epsilon=config["MAD_EPSILON"],
                           image_features=config["INC_IMAGE_FEATURES"])
    
    ann_dframe = feature_select(ann_dframe, operation=config["FEAT_SELECT_OPS"],
                                image_features=config["INC_IMAGE_FEATURES"])
    
    # remove rows with any NaNs in features
    if remove_nan_rows:
        ann_dframe = ann_dframe[~ann_dframe.filter(regex="^(?!Metadata_)").isnull().T.any()]

    return ann_dframe