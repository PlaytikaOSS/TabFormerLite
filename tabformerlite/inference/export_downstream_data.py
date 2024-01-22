import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def export_data(exp_config, pretraining_data_config, output_path, *outputs):
    """
    This function exports embeddings with labels for downstream tasks
    in a parquet file format with the following columns:
    - user_id
    - date
    - embeddings
    - label(s)

    Parameters:
    -----------
    exp_config : dict
        A dictionary containing the experiment configuration
    pretraining_data_config : dict
        A dictionary containing the pretraining data configuration
    output_path : str
        A string containing the path to the output file
    outputs : tuple
        A tuple containing the train/valid/test outputs from the embeddings
        extractor. Each output is a tuple of 3 elements:
        - user_id
        - date
        - list of extracted embeddings for each user_id and date pair.

    Returns:
    --------
    None
    """

    logger.info("Exporting embeddings with labels for downstream tasks - START")

    # Collect user_id, date and embeddings for train/valid/test in a DataFrame
    df_embeddings = collect_embeddings(pretraining_data_config, *outputs)

    # Add labels for downstream tasks
    df_embeddings_with_label = add_label_to_embeddings(
        df_embeddings, exp_config, pretraining_data_config
    )

    # Export labeled data
    logger.info(f"Data shape: {df_embeddings_with_label.shape}")

    if (
        exp_config["inference_config"]["pooling_on_time_axis"]["strategy"]
        != "no_pooling"
    ):
        # Export to parquet file
        logger.info(f"Output file name: {output_path}")
        df_embeddings_with_label.to_parquet(f"{output_path}", index=False)

    else:
        # Export to npz file
        # Get user and date column names
        user_col = pretraining_data_config["user_col"]
        date_col = pretraining_data_config["date_col"]
        target_cols = exp_config["downstream_task_config"]["target_cols_to_include"]

        output_path_npz = output_path.replace(".parquet", ".npz")
        logger.info(f"Output file name: {output_path_npz}")

        kwargs_arrays = {
            "embeddings": np.array(
                df_embeddings_with_label["embeddings"].values.tolist()
            ),
            user_col: df_embeddings_with_label[user_col].values,
            date_col: df_embeddings_with_label[date_col].values,
            target_cols[0]: df_embeddings_with_label[target_cols[0]].values,
        }

        np.savez(file=f"{output_path_npz}", **kwargs_arrays)

    logger.info("Exporting embeddings with labels for downstream tasks - END\n")


def collect_embeddings(pretraining_data_config, *outputs):
    """
    This function takes the train/valid/test outputs from the embeddings
    extractor and returns a DataFrame with the following columns:
    - user_id
    - date
    - embeddings

    Parameters:
    -----------
    pretraining_data_config : dict
        A dictionary containing the pretraining data configuration
    outputs : tuple
        A tuple containing the train/valid/test outputs from the embeddings
        extractor. Each output is a tuple of 3 elements:
        - user_id
        - date
        - list of extracted embeddings for each user_id and date pair.

    Returns:
    --------
    df_embeddings : pd.DataFrame
        A DataFrame containing the user_id, date and embeddings for
        train/valid/test
        Note: There are no labels in df_embeddings. They will be added in the
        next step with the function "add_label_to_embeddings"
    """

    # Get user and date column names
    user_col = pretraining_data_config["user_col"]
    date_col = pretraining_data_config["date_col"]

    # Create empty DataFrame
    df_embeddings = pd.DataFrame()

    # Concatenate user_col, date_col and embeddings
    # for train/valid/test in a DataFrame
    for output in outputs:
        df_embeddings = pd.concat(
            [
                df_embeddings,
                pd.DataFrame(
                    {user_col: output[0], date_col: output[1], "embeddings": output[2]}
                ),
            ],
            axis=0,
        )

    df_embeddings[date_col] = pd.to_datetime(df_embeddings[date_col])
    df_embeddings[date_col] = df_embeddings[date_col].astype(str)

    return df_embeddings


def add_label_to_embeddings(df_embeddings, exp_config, pretraining_data_config):
    """
    This function merges embeddings with the requested label columns.
    The labels can be in a parquet/csv file or in a directory containing
    multiple parquet/csv files.

    Parameters:
    -----------
    df_embeddings : pd.DataFrame
        A DataFrame containing the user_id, date and embeddings for
        train/valid/test
    exp_config : dict
        A dictionary containing the experiment configuration
    pretraining_data_config : dict
        A dictionary containing the pretraining data configuration

    Returns:
    --------
    df_embeddings_with_label : pd.DataFrame
        It returns a DataFrame with the following columns:
        - user_id
        - date
        - embeddings
        - label(s)
    """

    # Create lists with user, date and label column names
    target_cols = exp_config["downstream_task_config"]["target_cols_to_include"]

    # Get user and date column names in embeddings DataFrame
    user_col = pretraining_data_config["user_col"]
    date_col = pretraining_data_config["date_col"]

    # Get user and date column names in labels DataFrame
    user_col_labels = exp_config["downstream_task_config"]["user_col"]
    date_col_labels = exp_config["downstream_task_config"]["date_col"]

    cols_to_load = [user_col_labels, date_col_labels] + target_cols

    # Verify if path_to_data_with_labels is a file or a directory
    path_to_data_with_labels = exp_config["downstream_task_config"][
        "path_to_data_with_labels"
    ]

    # If path_to_data_with_labels is a file, read it
    if os.path.isfile(path_to_data_with_labels):
        # Get file extension
        _, file_extension = os.path.splitext(path_to_data_with_labels)

        # Verify it is a supported file extension
        if file_extension not in [".parquet", ".csv"]:
            raise ValueError("Please provide a valid parquet/csv file.")

        # Read labels with the appropriate pandas function
        if file_extension == ".parquet":
            df_labels = pd.read_parquet(path_to_data_with_labels)[cols_to_load]

        elif file_extension == ".csv":
            df_labels = pd.read_csv(path_to_data_with_labels)[cols_to_load]

    # If path_to_data_with_labels is a directory, create list
    # with all filepaths and load data from each file
    elif os.path.isdir(path_to_data_with_labels):
        folder_path = Path(path_to_data_with_labels)
        file_paths = [str(file) for file in folder_path.glob("*") if file.is_file()]

        df_labels = pd.DataFrame()

        for file_path in file_paths:
            # Get file extension
            _, file_extension = os.path.splitext(file_path)

            # Verify it is a supported file extension
            if file_extension not in [".parquet", ".csv"]:
                raise ValueError("Please provide a valid parquet/csv file.")

            # Read labels with the appropriate pandas function
            if file_extension == ".parquet":
                df_labels_temp = pd.read_parquet(file_path)[cols_to_load]

            elif file_extension == ".csv":
                df_labels_temp = pd.read_csv(file_path)[cols_to_load]

            df_labels = pd.concat([df_labels_temp, df_labels], axis=0)

        del df_labels_temp

    else:
        raise ValueError(
            f"{path_to_data_with_labels} is neither a file nor a directory."
        )

    # Fix dtypes in user and date columns
    df_labels[user_col_labels] = df_labels[user_col_labels].astype("int")
    df_labels[date_col_labels] = pd.to_datetime(df_labels[date_col_labels])
    df_labels[date_col_labels] = df_labels[date_col_labels].astype(str)

    # Merge embeddings with labels
    df_embeddings_with_label = pd.merge(
        df_embeddings,
        df_labels,
        left_on=[user_col, date_col],
        right_on=[user_col_labels, date_col_labels],
        how="outer",
    )

    df_embeddings_with_label = df_embeddings_with_label.dropna()
    df_embeddings_with_label.reset_index(drop=True, inplace=True)

    df_embeddings_with_label = df_embeddings_with_label[
        [user_col, date_col, "embeddings"] + target_cols
    ]

    if (
        exp_config["inference_config"]["pooling_on_time_axis"]["strategy"]
        != "no_pooling"
    ):
        # "Explode" embeddings column to one column per embedding dimension
        # to be able to export it to parquet file
        df_embeds_explode = pd.DataFrame(
            np.array(df_embeddings_with_label["embeddings"].values.tolist())
        )
        # Name embedding columns
        df_embeds_explode.rename(columns=lambda x: f"f{str(x)}", inplace=True)
        df_embeddings_with_label = pd.concat(
            [df_embeddings_with_label, df_embeds_explode], axis=1
        )
        df_embeddings_with_label.drop("embeddings", axis=1, inplace=True)

        del df_embeds_explode

    del (df_labels, df_embeddings)

    return df_embeddings_with_label
