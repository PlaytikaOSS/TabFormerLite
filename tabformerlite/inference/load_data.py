import os
from glob import glob

import numpy as np
import pandas as pd
from loguru import logger

from tabformerlite.misc.get_configuration import get_value


def load_data_for_encoding(inference_config_file, pretraining_config_file):
    """This function takes a config file containing a "data_directory"
    path and returns a dict with the train/valid/test inference data
    to encode using the TabDataset.

    Notes
    -----
    - The "data_directory" should contain the train/validation/test datasets
    in separate csv files.
    - The {set_type} should appear at the beginning of the csv file name,
    followed by an underscore: e.g.,: 'train_data_for_enconding_14feats.csv'

    Parameters
    ----------
    inference_config_file : dict
        A dictionary containing the "data_directory" path with the
        training/valid/test data
    pretraining_config_file : dict
        A dictionary containing the "user_col" and "date_col" names

    Returns
    -------
    data_dict : dict
        A dictionary with dataframes for each of the train/val/test sets

    """

    logger.info("Loading inference data - START")
    data_dict = {}

    # Directory contaning training/valid/test data
    data_directory = get_value(inference_config_file, "data_directory")
    data_directory = os.path.abspath(data_directory)

    # Load data for all sets
    data_dict["df_train"] = load_data_per_set_type(
        data_directory, pretraining_config_file, "train"
    )
    data_dict["df_valid"] = load_data_per_set_type(
        data_directory, pretraining_config_file, "valid"
    )
    data_dict["df_test"] = load_data_per_set_type(
        data_directory, pretraining_config_file, "test"
    )

    logger.info(f"Training set shape: {data_dict['df_train'].shape}")
    logger.info(f"Validation set shape: {data_dict['df_valid'].shape}")
    logger.info(f"Test set shape: {data_dict['df_test'].shape}")
    logger.info("Loading inference data - END\n")

    return data_dict


def load_data_per_set_type(data_dir, pretraining_config_file, set_type):
    """This function takes a path to a directory containing the
    data to encode and a "set_type" and loads it in a dataframe.

    Notes
    -----
    - The data for each set_type (train/val/test) should be in separate csv
    files.
    - The {set_type} should appear at the beginning of the
    csv file name, followed by an underscore:
        e.g.,: 'train_data_for_enconding_14feats.csv'

    Parameters
    ----------
    data_dir : str
       Directory with train/valid/test data to encode

    set_type : str, supported values: "train", "valid" and "test"
       The set type of the dataset.

    Returns
    ------
    df : dataframe
        A dataframe with the data to encode.
    """

    # Path to data file for provided set_type
    paths_list = glob(f"{data_dir}/{set_type}_*")

    assert len(paths_list) != 0, (
        f"The file '{set_type}_*.csv' wasn't found. "
        "Please respect the naming convention for the provided csv files. "
        "The set_type should appear at the beginning of the "
        "csv file name, followed by an underscore: e.g.,: "
        "'train_data_for_encoding_14feats.csv'."
    )

    file_path = paths_list[0]

    # check file extension
    _, file_extension = os.path.splitext(file_path)

    # verify it is a supported file extension
    assert file_extension in [".csv"], "Please enter a valid csv file."

    # Load data in dataframe
    df = pd.read_csv(file_path, low_memory=False)

    # Get user and date column names
    user_col = pretraining_config_file["user_col"]
    date_col = pretraining_config_file["date_col"]
    target_col = pretraining_config_file["target_col"]

    # Show imbalance percentage
    counts, imbalance_perc = get_imbalance_percentage(df[target_col].values)
    logger.info(f"Imbalance percentage for {set_type} set: {counts}, {imbalance_perc}")

    # Sort data by user and date
    df.sort_values(by=[user_col, date_col], inplace=True)

    # Hack to reduce data size (for dev)
    # df = df[0:120].copy()

    return df


def get_imbalance_percentage(y):
    """
    Get imbalance percentage of a dataset.
    Args:
        y (np.array): Dataset labels.
    Returns:
        imbalance_percentage (list): Imbalance percentage per class.
    """
    counts = np.array([np.count_nonzero(y == 0), np.count_nonzero(y == 1)])
    data_len = len(y)
    imbalance_perc = counts / data_len

    imbalance_perc = [round(x * 100, 2) for x in imbalance_perc.tolist()]
    return counts.tolist(), imbalance_perc
