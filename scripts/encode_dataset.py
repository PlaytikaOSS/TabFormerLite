import json
import os
from argparse import ArgumentParser

import pandas as pd
from loguru import logger

from tabformerlite.dataset.dataset import TabDataset
from tabformerlite.misc.tools_for_data_encoding import select_discretization_strategy


def main():
    """
    Main function to execute from command line.

    $ python3 scripts/encode_dataset.py -cfg --path-to-config-files

    Example usage
    -------------
    >>> $ python3 scripts/encode_dataset.py \
    -cfg ./configs/example/data_encoding/config_card_dataset_encoding.json

    The data pre-processing step prepares tabular data to be compatible with
    the TabFormerLite model by implementing the following:
    - transforms tabular data into discrete units of information
    (like NLP-style tokens)
    - shapes tabular data into sequences suitable for TabFormerLite using
    a sliding window approach
    - exports processed data in .h5 files for easy reloading in training phase
    """

    # Parse the config file
    parser = ArgumentParser()
    parser.add_argument("-cfg", "--config-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf8") as f:
        config = json.load(f)

    #####################################################################
    # Load the raw tabular data for encoding
    logger.info("Loading the data")

    # Create absolute path to data file
    data_path = os.path.join(config["data_dir"], config["data_name"])
    absolute_data_path = os.path.abspath(data_path)

    df = pd.read_csv(absolute_data_path, low_memory=False)
    logger.info(f"Data shape: {df.shape}\n")

    # Convert date column to datetime and then to int
    date_col = config["date_col"]
    df[date_col] = pd.to_datetime(df[date_col])
    df[date_col] = df[date_col].astype(int)

    #####################################################################
    # Create the TabDatataset

    # Select discretization strategy for each column:
    # Label Encoder: categorical columns
    # Quantile Binning:
    #   * continuous columns with > n_max unique values
    #   * discrete columns with > n_max unique values
    # No transformation: discrete columns with <= n_max unique values

    discretization_dict = select_discretization_strategy(
        df, n_max=config["n_max"], config=config, verbose=True
    )

    # Create an instance of the TabDataset class
    logger.info("Encoding the dataset")

    dataset = TabDataset(
        df,
        discretization_dict["categorical_columns"],
        discretization_dict["to_quantize_columns"],
        label_col=config["target_col"],
        user_col=config["user_col"],
        date_col=date_col,
        seq_len=config["seq_len"],
        num_bins=config["num_bins"],
        stride=config["stride"],
        return_labels=False,
        vocab_from_file=None,
        binning_from_file=None,
    )

    # Extract vocab and special tokens
    vocab = dataset.vocab
    logger.info(f"Vocab size: {len(vocab)}\n")

    #####################################################################
    # Export encoded data
    # Encoded data will be saved in .h5 files for easy reloading
    # in training phase

    data_dir = config["encoded_data_dir"]
    data_folder = config["encoded_data_folder"]
    encoded_data_path = os.path.join(data_dir, data_folder)
    encoded_data_path = os.path.abspath(encoded_data_path)
    logger.info(f"Encoded data will be exported in: {encoded_data_path}")

    # If unavailable, create directory in specified path
    os.makedirs(encoded_data_path, mode=0o777, exist_ok=True)

    # Save vocabulary and encoders
    logger.info("Exporting vocabulary and encoders")
    dataset.save_vocab_and_binning(encoded_data_path)

    # Save processed data
    logger.info("Exporting processed data")
    dataset.save_processed_data_and_labels(encoded_data_path)

    # Save columns used to create the player Dataset in json file
    logger.info("Exporting config file")
    column_lists_by_dtype = {
        "categorical_columns": discretization_dict["categorical_columns"],
        "to_quantize_columns": discretization_dict["to_quantize_columns"],
        "cols_no_transform": discretization_dict["cols_no_transform"],
    }
    json_string = json.dumps(column_lists_by_dtype)

    with open(
        os.path.join(encoded_data_path, "column_lists_by_dtype.json"),
        "w",
        encoding="utf8",
    ) as json_file:
        json_file.write(json_string)

    # Save the config file
    with open(
        os.path.join(encoded_data_path, "config.json"), "w", encoding="utf8"
    ) as fh:
        json.dump(config, fh)


if __name__ == "__main__":
    main()
