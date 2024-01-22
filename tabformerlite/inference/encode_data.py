import json
import os

import pandas as pd
from loguru import logger

from tabformerlite.dataset.dataset import TabDataset  # noqa: E402


class EncoderForPlayerData:
    """
    This class encodes player data using the TabDataset class.

    Attributes
    ----------
    config_file : dict
        A dictionary with info about the data used to pretrain TabFormer

    Methods
    -------
    load_column_dtypes_dict(path_to_encoded_data)
        This method loads a dictionary with the column names used during
        pre-training to create the vocabulary of the TabDataset.

    generage_kwargs_for_dataset:
        This method creates a dictionary with kwargs for TabDataset.

    encode(data_to_encode):
        This method encodes the "data_to_encode" using the TabDataset
        class.

    load_vocab_and_special_tokens(player_dataset)
        This method loads the vocabulary, special tokens and ncols created
        by the TabDataset class in a dictionary (player_dataset_dict).


    Example Usage
    -------------
    >>> encoder = EncoderForPlayerData(pretraining_data_config)
    >>> train_dataset = encoder.encode(data_dict["df_train"])
    >>> dataset_dict = encoder.load_vocab_and_special_tokens(train_dataset)

    """

    def __init__(self, config_file):
        # config file with info about pretraining data
        self.config_file = config_file

        self.pretrain_data_path = os.path.join(
            self.config_file["encoded_data_dir"],
            self.config_file["encoded_data_folder"],
        )

    #########################################################################
    def load_cols_dict(self, pretraining_data_path):
        """This method loads a dictionary with the column names used during
        pre-training to create the vocabulary in the TabDataset.
        """
        # Define path to column_dtypes_dict
        column_dtypes_filename = f"{pretraining_data_path}/column_lists_by_dtype.json"

        # Raise ValueError if dict is missing
        if not os.path.isfile(column_dtypes_filename):
            raise ValueError(
                "'column_lists_by_dtype.json' file doesn't exist"
                "in the provided path."
            )

        # Load json file
        with open(column_dtypes_filename, "r", encoding="utf8") as f:
            column_dtypes_dict = json.load(f)

        return column_dtypes_dict

    #########################################################################
    def generage_kwargs_for_dataset(self):
        """This method creates a config to use with the dataset class."""
        column_dtypes_dict = self.load_cols_dict(self.pretrain_data_path)

        dataset_kwargs = {
            "categorical_columns": column_dtypes_dict["categorical_columns"],
            "to_quantize_columns": column_dtypes_dict["to_quantize_columns"],
            "seq_len": self.config_file["seq_len"],
            "num_bins": self.config_file["num_bins"],
            "user_col": self.config_file["user_col"],
            "date_col": self.config_file["date_col"],
            "label_col": self.config_file["target_col"],
            "stride": 1,  # inference
            "return_labels": False,
            "vocab_from_file": f"{self.pretrain_data_path}/vocab.pickle",
            "binning_from_file": f"{self.pretrain_data_path}/binning.pickle",
            "processed_data_from_file": None,
        }

        return dataset_kwargs

    ##########################################################################
    def encode(self, data_to_encode: pd.DataFrame):
        """
        Sends the pandas DataFrame with the data to the dataset class.
        """
        # Convert date column
        data_to_encode[self.config_file["date_col"]] = pd.to_datetime(
            data_to_encode[self.config_file["date_col"]]
        )
        data_to_encode[self.config_file["date_col"]] = data_to_encode[
            self.config_file["date_col"]
        ].astype(int)

        # Define kwargs for player dataset
        player_dataset_kwargs = self.generage_kwargs_for_dataset()

        # create TabDataset
        player_dataset = TabDataset(data_to_encode, **player_dataset_kwargs)
        logger.info(f"length: {len(player_dataset)}")

        # Release memory
        del data_to_encode

        return player_dataset

    ##########################################################################
    def load_vocab_and_special_tokens(self, player_dataset):
        """Extracts vocab, ncols and special tokens in a dictionary"""
        logger.info("Extracting vocabulary and special tokens - START")

        player_dataset_dict = {}

        player_dataset_dict["vocab"] = player_dataset.vocab
        player_dataset_dict["special_tokens"] = player_dataset_dict[
            "vocab"
        ].get_special_tokens()
        player_dataset_dict["ncols"] = player_dataset.ncols

        logger.info(f"ncols: {player_dataset_dict['ncols']}")
        logger.info(f"custom_special_tokens: {player_dataset_dict['special_tokens']}")
        logger.info(f"Vocab size: {len(player_dataset_dict['vocab'])}")
        logger.info("Extracting vocabulary and special tokens - END\n")

        return player_dataset_dict
