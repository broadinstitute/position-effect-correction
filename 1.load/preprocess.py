import pandas as pd

# to install pycytominer, run `poetry run pip install <path_to_pycytominer>`
from pycytominer import normalize, feature_select

config = {
    "NORMALIZE_METHOD": "mad_robustize",
    "MAD_EPSILON": 0.0,
    "INC_IMAGE_FEATURES": True,
    "FEAT_SELECT_OPS": [
        "variance_threshold",
        "correlation_threshold",
        "drop_na_columns",
        "blocklist",
    ],
}


def preprocess_profiles(
    ann_dframe,
    remove_nan_rows=True,
    normalize_kwargs=None,
    feature_select_kwargs=None,
):
    """
    Preprocess profiles by normalizing and feature selecting.

    Parameters
    ----------
    ann_dframe : pandas.core.frame.DataFrame
        DataFrame of annotated profiles.
    remove_nan_rows : bool, default True
        Whether to remove rows with any NaNs in features.
    normalize_kwargs : dict, optional
        Keyword arguments to pass to `pycytominer.normalize.normalize`.
    feature_select_kwargs : dict, optional
        Keyword arguments to pass to `pycytominer.feature_select.feature_select`.

    Returns
    -------
    ann_dframe : pandas.core.frame.DataFrame
        DataFrame of annotated profiles after normalization and feature selection.
    """
    # normalize profiles
    normalize_kwargs = normalize_kwargs or {}
    if "method" not in normalize_kwargs:
        normalize_kwargs["method"] = config["NORMALIZE_METHOD"]
    if "mad_robustize_epsilon" not in normalize_kwargs:
        normalize_kwargs["mad_robustize_epsilon"] = config["MAD_EPSILON"]
    if "image_features" not in normalize_kwargs:
        normalize_kwargs["image_features"] = config["INC_IMAGE_FEATURES"]

    ann_dframe = normalize(ann_dframe, **normalize_kwargs)

    # feature select
    feature_select_kwargs = feature_select_kwargs or {}
    if "operation" not in feature_select_kwargs:
        feature_select_kwargs["operation"] = config["FEAT_SELECT_OPS"]
    if "image_features" not in feature_select_kwargs:
        feature_select_kwargs["image_features"] = config["INC_IMAGE_FEATURES"]

    ann_dframe = feature_select(ann_dframe, **feature_select_kwargs)

    # remove rows with any NaNs in features
    if remove_nan_rows:
        ann_dframe = ann_dframe[
            ~ann_dframe.filter(regex="^(?!Metadata_)").isnull().T.any()
        ]

    return ann_dframe
