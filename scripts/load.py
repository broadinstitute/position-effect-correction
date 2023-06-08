"""Functions for loading metadata and profiles."""

import logging
from typing import Optional, Union

from pathlib import Path
from functools import reduce
from omegaconf import OmegaConf, dictconfig

import pandas as pd

import s3fs
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow.dataset import DirectoryPartitioning

import fire

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_config(config_path: Union[Path, str]) -> dictconfig.DictConfig:
    """
    Load configs from a directory or file.

    Parameters
    ----------
    config_path : Union[Path, str]
        Path to config directory or file.

    Returns
    -------
    dict
        Dictionary of configs.
    """
    if isinstance(config_path, (str)):
        config_path = Path(config_path)

    if config_path.is_dir():
        configs = [
            OmegaConf.load(meta_conf) for meta_conf in config_path.glob("*.yaml")
        ]
        return OmegaConf.merge(*configs)
    else:
        return OmegaConf.load(config_path)


def read_config_data(config: dictconfig.DictConfig) -> pd.DataFrame:
    """
    Read metadata from a directory or file.

    Parameters
    ----------
    metadata_config : dict
        Metadata config.

    Returns
    -------
    pd.DataFrame
        Metadata dataframe.
    """
    data_path = Path(config.path).glob(config.files)
    data = (
        pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        for f in data_path
    )
    data = pd.concat(data, ignore_index=True)
    if "drop" in config:
        data = data.drop(config.drop, axis=1)
    if "rename" in config:
        data = data.rename(columns=config.rename)
    if "filter" in config:
        data = data.query(config.filter)
        data.reset_index(drop=True, inplace=True)
    return data


def merge_metadata(meta_config: dictconfig.DictConfig) -> pd.DataFrame:
    """
    Merge metadata from multiple sources.

    Parameters
    ----------
    meta_config : dictconfig.DictConfig
        Metadata config.

    Returns
    -------
    pd.DataFrame
    """
    dataframes = []
    merge_on_fields = []
    merge_orders = []

    def inner_merge(left, right):
        return pd.merge(left, right[0], on=right[1], how="inner")

    for config in meta_config.values():
        dataframes.append(read_config_data(config))
        merge_on_fields.append(OmegaConf.to_object(config["merge_on"]))
        merge_orders.append(config["merge_order"])

    # zip the three lists together
    zipped_list = sorted(
        list(zip(dataframes, merge_on_fields, merge_orders)), key=lambda x: x[2]
    )
    merged_data = reduce(inner_merge, zipped_list[1:], zipped_list[0][0])
    return merged_data


def load_data(
    dataset: str,
    source: str,
    component: str,
    batch: str = None,
    plate: str = None,
    columns: list = None,
    output: str = None,
) -> Optional[pd.DataFrame]:
    """Load data components from a source directory, optionally filtering by batch and plate.

    Parameters
    ----------
    dataset : str
        Dataset name.

    source : str
        Source directory.

    component : str
        Component name (currently only "profiles" or "load_data_csv").

    batch : str, optional
        Batch name, by default None

    plate : str, optional
        Plate name, by default None

    columns : list, optional
        Columns to load, by default None

    output : str, optional
        Output Parquet file, by default None (return pandas.DataFrame)

    Returns
    -------
    Optional[pd.DataFrame]
        Pandas DataFrame if `output` is None, else None.

    Examples
    --------
    >>> python load.py \
    >>> cpg0016-jump \
    >>> source_4 \
    >>> --columns "[Metadata_Source,Metadata_Plate,Metadata_Well,Cells_AreaShape_Eccentricity,Nuclei_AreaShape_Area]" \
    >>> --batch 2021_06_14_Batch6 \
    >>> --plate BR00121429 \
    >>> --output ~/Desktop/test.parquet

    Print the top 5 rows
    >>> python -c "import pandas as pd; print(pd.read_parquet('~/Desktop/test.parquet').head())"
    """
    # Checks
    if output is not None:
        assert isinstance(output, str), "`output` must be a string"
        assert output.endswith(".parquet"), "`output` must end with .parquet"

    if columns is not None:
        assert isinstance(columns, list), "`columns` must be a list"
        logging.info(f"Loading columns: {columns}")

    assert component in [
        "profiles",
        "load_data_csv",
    ], "`component` must be 'profiles' or 'load_data_csv'"

    if batch is None:
        assert plate is None, "`plate` must be None if `batch` is None"

    dataset_source = f"cellpainting-gallery/{dataset}/{source}/workspace/profiles"

    if batch is not None:
        dataset_source += f"/{batch}"
        if plate is not None:
            dataset_source += f"/{plate}"

    logging.info(f"Loading profiles from {dataset_source}")

    dataset = ds.dataset(
        source=dataset_source,
        # use s3fs for faster download, see https://github.com/apache/arrow/issues/14336
        filesystem=s3fs.S3FileSystem(anon=True),
        partitioning=DirectoryPartitioning(
            pa.schema(
                [
                    ("dataset", pa.string()),
                    ("source", pa.string()),
                    ("workspace", pa.string()),
                    (component, pa.string()),
                    ("batch", pa.string()),
                    ("plate", pa.string()),
                ],
            )
        ),
        format="parquet",
        exclude_invalid_files=True,
    )

    logging.info(f"Found {len(dataset.files)} files")

    logging.info(f"Load {component}...")

    df = dataset.to_table(columns=columns)

    if output is not None:
        logging.info(f"Writing {component} to {output}")
        pq.write_table(df, output)
    else:
        return df.to_pandas()


if __name__ == "__main__":
    fire.Fire(load_data)
