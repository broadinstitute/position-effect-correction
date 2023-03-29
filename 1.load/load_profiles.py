# -*- coding: utf-8 -*-

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow.dataset import DirectoryPartitioning
import logging

logging.basicConfig(level=logging.INFO)


def load_profiles(
    dataset: str,
    source: str,
    batch: str = None,
    plate: str = None,
    columns: list = None,
    output: str = None,
):
    """Load profiles from a source directory, optionally filtering by batch and plate.

    Parameters
    ----------
    dataset : str
        Dataset name.

    source : str
        Source directory.

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
    pandas.DataFrame
        Profiles.
    """

    # Checks
    if output is not None:
        assert isinstance(output, str), "`output` must be a string"
        assert output.endswith(".parquet"), "`output` must end with .parquet"

    if columns is not None:
        assert isinstance(columns, list), "`columns` must be a list"
        logging.info(f"Loading columns: {columns}")

    dataset_source = f"/{dataset}/{source}/workspace/profiles"

    if batch is not None:
        dataset_source += f"/{batch}"
        if plate is not None:
            dataset_source += f"/{plate}"

    logging.info(f"Loading profiles from {dataset_source}")

    dataset = ds.dataset(
        source=dataset_source,
        filesystem="s3://cellpainting-gallery",
        partitioning=DirectoryPartitioning(
            pa.schema(
                [
                    ("dataset", pa.string()),
                    ("source", pa.string()),
                    ("workspace", pa.string()),
                    ("profiles", pa.string()),
                    ("batch", pa.string()),
                    ("plate", pa.string()),
                ],
            )
        ),
        format="parquet",
        exclude_invalid_files=True,
    )

    logging.info(f"Found {len(dataset.files)} files")

    logging.info("Load profiles...")

    df = dataset.to_table(columns=columns)

    if output is not None:
        logging.info(f"Writing profiles to {output}")
        pq.write_table(df, output)
    else:
        return df.to_pandas()


import fire

if __name__ == "__main__":
    fire.Fire(load_profiles)

# Example usage

# python load_profiles.py \
#   cpg0016-jump \
#   source_4
#   --columns [Metadata_Source,Metadata_Plate,Metadata_Well,Cells_AreaShape_Eccentricity,Nuclei_AreaShape_Area] \
#   --batch 2021_06_14_Batch6 \
#   --output ~/Desktop/test.parquet
