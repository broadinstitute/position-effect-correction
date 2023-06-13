"""Functions for preprocessing profiles."""
from typing import Optional

import pandas as pd

# to install pycytominer, run `poetry run pip install <path_to_pycytominer>`
from pycytominer import normalize, feature_select


def drop_na_feature_rows(ann_dframe: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NA values in non-feature columns.

    Parameters
    ----------
    ann_dframe : pd.DataFrame
        DataFrame of annotated profiles.

    Returns
    -------
    ann_dframe : pd.DataFrame
        DataFrame of annotated profiles after dropping rows with NA values in feature columns.
    """
    ann_dframe = ann_dframe[~ann_dframe.filter(regex="^(?!Metadata_)").isnull().T.any()]
    ann_dframe.reset_index(drop=True, inplace=True)
    return ann_dframe


def normalize_profiles(
    ann_dframe: pd.DataFrame,
    normalize_group: Optional[str] = None,
    normalize_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Normalize profiles.

    Parameters
    ----------
    ann_dframe : pd.DataFrame
        DataFrame of annotated profiles.
    normalize_group : str, optional
        Column name to group by for normalization.
    normalize_kwargs : dict, optional
        Keyword arguments to pass to `pycytominer.normalize.normalize`.

    Returns
    -------
    ann_dframe : pd.DataFrame
        DataFrame of annotated profiles after normalization.
    """
    normalize_kwargs = normalize_kwargs or {}

    if normalize_group is not None:
        ann_dframe = ann_dframe.groupby(normalize_group, group_keys=True).apply(
            lambda x: normalize(x, **normalize_kwargs)
        )
        ann_dframe.reset_index(drop=True, inplace=True)
    else:
        ann_dframe = normalize(ann_dframe, **normalize_kwargs)

    return ann_dframe


def select_features(
    ann_dframe: pd.DataFrame,
    feature_select_kwargs: Optional[dict] = None,
    feature_whitelist: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Select features.

    Parameters
    ----------
    ann_dframe : pd.DataFrame
        DataFrame of annotated profiles.
    feature_select_kwargs : dict, optional
        Keyword arguments to pass to `pycytominer.feature_select.feature_select`.
    feature_select_whitelist : list[str], optional
        List of features to preserve.

    Returns
    -------
    ann_dframe : pd.DataFrame
        DataFrame of annotated profiles after feature selection.
    """
    feature_select_kwargs = feature_select_kwargs or {}

    # preserve features in whitelist
    if feature_whitelist is not None:
        whitelist_features = ann_dframe[feature_whitelist]
        ann_dframe = feature_select(
            ann_dframe.drop(columns=feature_whitelist), **feature_select_kwargs
        )
        ann_dframe = pd.concat([ann_dframe, whitelist_features], axis="columns")
    else:
        ann_dframe = feature_select(ann_dframe, **feature_select_kwargs)

    return ann_dframe


def preprocess_profiles(
    ann_dframe: pd.DataFrame,
    remove_nan_rows: bool = True,
    normalize_group: Optional[str] = None,
    normalize_kwargs: Optional[dict] = None,
    feature_select_kwargs: Optional[dict] = None,
    feature_select_whitelist: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Preprocess profiles by normalizing and selecting features.

    Parameters
    ----------
    ann_dframe : pd.DataFrame
        DataFrame of annotated profiles.
    remove_nan_rows : bool, optional
        Whether to remove rows with any NaNs in features.
    normalize_group : str, optional
        Column name to group by for normalization.
    normalize_kwargs : dict, optional
        Keyword arguments to pass to `pycytominer.normalize.normalize`.
    feature_select_kwargs : dict, optional
        Keyword arguments to pass to `pycytominer.feature_select.feature_select`.
    feature_select_whitelist : list[str], optional
        List of features to preserve.

    Returns
    -------
    ann_dframe : pd.DataFrame
        DataFrame of annotated profiles after preprocessing.
    """
    ann_dframe = normalize_profiles(ann_dframe, normalize_group, normalize_kwargs)

    ann_dframe = select_features(
        ann_dframe, feature_select_kwargs, feature_select_whitelist
    )

    # remove rows with any NaNs in features
    if remove_nan_rows:
        ann_dframe = drop_na_feature_rows(ann_dframe)

    return ann_dframe
