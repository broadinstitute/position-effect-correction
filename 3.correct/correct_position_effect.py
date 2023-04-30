import pandas as pd

def subtract_well_mean(ann_df: pd.DataFrame) -> pd.DataFrame:
    '''Subtract the mean of each feature per each well.
    
    Parameters
    ----------
    ann_df : pandas.DataFrame
        Dataframe with features and metadata.
    
    Returns
    -------
    pandas.DataFrame
        Dataframe with features and metadata, with each feature subtracted by the mean of that feature per well.
    '''
    feature_cols = ann_df.filter(regex="^(?!Metadata_)").columns
    ann_df[feature_cols] = ann_df.groupby("Metadata_Well")[feature_cols].transform(lambda x: x - x.mean())
    return ann_df