import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from tabformerlite.dataset.vocab import Vocabulary
from tabformerlite.misc.tools_for_data_encoding import optimize_memory


class TabDataset(Dataset):
    """
    This class serves as a link between normal tabular data, in the form of
    pandas Dataframe, and the inputs that can be read by the TabFormer model,
    which are windows of "seq_len" rows for a single user. Those windows are
    taken every "stride" rows.

    According to what is specified, it will either categorize, leave untouched
    or quantize the input columns. The number of bins for this quantization can
    be changed.

    The data is only stored once, the windows are obtained by reading
    specified indexes, which are generated at the creation of the class.


    :param: df: Pandas Dataframe containing the rows to be encoded.
    :param: categorical_columns: list of columns to be categorized.
    :param: to_quantize_columns: list of columns to be quantized (quantile binning).
    :param: label_col: name of the columns containing the labels, defaults to "label".
    :param: user_col: name of the columns containing the user ids, defaults to "user".
    :param: date_col: name of the columns containing the dates, defaults to "date".
    :param: seq_len: size of the window to consider for a single data sample, defaults to 30.
    :param: num_bins: number of quantile bins to use for the quantization phase, defaults to 10.
    :param: stride: how many rows to pass before creating a new window (which can overlap), defaults to 5.
    :param: adap_thres: threshold for the number of unique elements in a column vocabulary.
            If reached, a custom criterion will be applied to compute the loss of this column, defaults to 10**8.
    :param: return_labels: flag whether to return the labels of not (e.g. False for MLM), defaults to False.
    :param: vocab_from_file: path to a saved vocabulary to reload (useful for inference), defaults to None.
    :param: binning_from_file: path to a saved binning to reload (useful for inference), defaults to None.
    :param: processed_data_from_file: path to already encoded data, defaults to "label".
            The vocab and binning files must also be reloaded from files when using this, defaults to None.
    :param: device: the output tensors can already be created on the specified device (e.g. 'cuda'),defaults to None.
    :param: labels_dtype: the labels will be converted to the following type, defaults to torch.long.
            Change to torch.float if doing regression.
    """

    def __init__(
        self,
        df,
        categorical_columns,
        to_quantize_columns,
        label_col="label",
        user_col="user",
        date_col="date",
        seq_len=30,
        num_bins=10,
        stride=5,
        adap_thres=10**8,
        return_labels=False,
        vocab_from_file=None,
        binning_from_file=None,
        processed_data_from_file=None,
        device=None,
        labels_dtype=torch.long,
    ):
        """
        Constructor for the TabDataset class.
        """
        self.df = df

        self.categorical_columns = categorical_columns
        self.to_quantize_columns = to_quantize_columns

        self.label_col = label_col
        self.user_col = user_col
        self.date_col = date_col

        self.return_labels = return_labels

        self.trans_stride = stride
        self.seq_len = seq_len

        self.encoder_fit = {}
        self.trans_table = None
        self.data_indices = []
        self.data = []
        self.labels = []
        self.labels_dtype = labels_dtype

        self.ncols = None
        self.num_bins = num_bins
        self.binning_from_file = binning_from_file
        self.device = device

        # Sanity checks
        if processed_data_from_file is not None:
            assert vocab_from_file is not None, (
                "If the processed data is loaded from a file, so must be the "
                "vocab file to ensure we use the right "
                "tokens and can save it again."
            )

            assert binning_from_file is not None, (
                "If the processed data is loaded from a file, so must be the "
                "binning file to ensure we can save them "
                "again."
            )

        # Reading encoders
        if binning_from_file is not None:
            logger.info("Loading the encoders")
            with open(binning_from_file, "rb") as fh:
                self.encoder_fit = pickle.load(fh)

        # Reading vocab
        if vocab_from_file is not None:
            with open(vocab_from_file, "rb") as fh:
                self.vocab = pickle.load(fh)
                self.vocab.filename = (
                    vocab_from_file[: vocab_from_file.rfind("/")]
                    + "/vocab_tokenizer.nb"
                )

        # If we reload the processed data, no need to do the rest
        if processed_data_from_file is not None:
            logger.info("Loading the encoded data")
            self.load_processed_data_and_labels(processed_data_from_file)

        # This is the case where we encode the data from scratch
        else:
            self.encode_data()

            # If the vocab is not reloaded, only then do we need to init it.
            # Must be done after the "encode_data" call.
            if vocab_from_file is None:
                self.vocab = Vocabulary(adap_thres, target_column_name=label_col)
                self.init_vocab()

            self.generate_data_indices()
            self.prepare_samples()

    def __getitem__(self, index):
        """
        Override the behaviour of accessing elements with square brackets ('[...]').

        :param index: index to access, can be a slice.
        :return: a tensor containing either the inputs alone, or a tuple of (inputs, labels)
                 if 'return_labels' has been set to True when creating the class.
        """

        idxs = self.data_indices[index]

        if isinstance(index, slice):
            raw_data = np.stack([self.data[idx[0] : idx[1]] for idx in idxs])
        else:
            raw_data = self.data[idxs[0] : idxs[1]]

        return_data = torch.tensor(raw_data, dtype=torch.long, device=self.device)

        if self.return_labels:
            if isinstance(index, slice):
                raw_label_data = np.stack(
                    [self.labels[idx[0] : idx[1]] for idx in idxs]
                )
            else:
                raw_label_data = self.labels[idxs[0] : idxs[1]]

            return_data = (
                return_data,
                torch.tensor(
                    raw_label_data, dtype=self.labels_dtype, device=self.device
                ),
            )

        return return_data

    def __len__(self):
        """
        Override the result of calling len(object).
        :return: The number of inputs contained in the TabDataset instance.
        """
        return len(self.data_indices)

    def save_vocab_and_binning(self, save_dir):
        """
        This function saves the vocabulary bins used for the quantization and
        the LabelEncoder categories assigned during the encoding process. They
        can be reloaded to create an instance of the TabDataset during training
        or inference.

        :param save_dir: folder path where the vocabulary and encoder files
        are saved.
        """
        file_name = os.path.join(save_dir, "vocab.pickle")
        with open(file_name, "wb") as fh:
            pickle.dump(self.vocab, fh)

        file_name = os.path.join(save_dir, "binning.pickle")
        with open(file_name, "wb") as fh:
            pickle.dump(self.encoder_fit, fh)

        # Also use the original save_vocab function
        # to set up the tokenizer vocabulary
        file_name = os.path.join(save_dir, "vocab_tokenizer.nb")
        self.vocab.save_vocab(file_name)

    def save_processed_data_and_labels(self, save_dir):
        """
        Save the already processed data and labels to avoid having to redo it
        everytime.

        :param save_dir: path of the folder where to save the already
        processed data.
        """
        file_name = os.path.join(save_dir, "processed_data_and_labels.h5")
        hf = h5py.File(f"{file_name}", "w", libver="latest")

        hf.create_dataset("data", data=self.data)
        hf.create_dataset("labels", data=self.labels)
        hf.create_dataset("data_indices", data=self.data_indices)

        hf.close()

    def load_processed_data_and_labels(self, filename):
        """
        Reloads the already processed data from disk.

        :param filename: path of the data file to reload. Should be in .h5 format.
        """
        hf = h5py.File(filename, "r", libver="latest")

        self.data = hf.get("data")[:]
        self.labels = hf.get("labels")[:]
        self.data_indices = hf.get("data_indices")[:]
        self.ncols = self.data.shape[1] - 2

        hf.close()

    def generate_data_indices(self):
        """
        Generate the indices that will be used to access the right input
        windows from the data.
        """

        # Obtain the number of lines per unique user.
        lengths = self.trans_table[self.user_col].value_counts(sort=False).values

        self.data_indices = []
        base_idx = 0
        # Create the indices of the data to take for each sample, avoiding duplicating the data.
        for ln in tqdm(lengths, "Generating data sample indices"):
            # Starting from the beginning index of a specific users, we create all the
            # indices of all the sample that can fit before the start of the next user.
            user_indices = [
                [base_idx + x, base_idx + x + self.seq_len]
                for x in range(0, ln, self.trans_stride)
                if x + self.seq_len <= ln
            ]
            self.data_indices.extend(user_indices)
            # Augment the base index to start generating indices for the next user.
            base_idx += ln

    def prepare_samples(self):
        """
        Converts the binned/categorized data values to tokens from the Vocabulary.
        Also adds the [SEP] token at the end of every row.
        """

        # Don't encode the user id, date columns and label.
        forbidden_cols = [self.label_col, self.user_col, self.date_col]
        to_encode_cols = [
            x for x in self.trans_table.columns if x not in forbidden_cols
        ]

        # For performance purposes, it is much faster to use the map function
        # of the pandas series to get from token to global id. This dict transformation
        # is necessary as the original one returns a tuple of (global_id, local_id).
        token2id_global = {}
        for field in self.vocab.token2id.keys():
            token2id_global[field] = {}
            for token in self.vocab.token2id[field].keys():
                token2id_global[field][token] = self.vocab.token2id[field][token][0]

        # If a token is not in the vocabulary, the [UNK] token will be used instead.
        unk_global_id = self.vocab.get_id(self.vocab.unk_token, special_token=True)

        for col in tqdm(
            to_encode_cols, "Converting the columns tokens to ids using the vocabulary"
        ):
            self.trans_table[col] = (
                self.trans_table[col].map(token2id_global[col]).fillna(unk_global_id)
            )

        self.data = self.trans_table.drop(columns=[self.label_col]).values
        self.labels = self.trans_table[self.label_col].values

        # Add the [SEP] special token as a extra column in the numpy array.
        sep_global_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)
        sep_special_column = np.ones((self.data.shape[0], 1), dtype=int) * sep_global_id
        self.data = np.hstack([self.data, sep_special_column])

        # Remove the date and the user_id columns from the counting
        self.ncols = self.data.shape[1] - 2

        del self.trans_table

    def init_vocab(self):
        """
        Initialize the vocabulary by assigning each binned/categorized value
        to a token id. Each column will have its own vocabulary, but a global
        one is generated and that is the one that will be used by the model.
        """

        forbidden_cols = [self.label_col, self.user_col, self.date_col]
        column_names = [x for x in self.trans_table.columns if x not in forbidden_cols]

        self.vocab.set_field_keys(column_names)

        for column in tqdm(column_names, "Vocabulary creation for each column"):
            unique_values = (
                self.trans_table[column].value_counts(sort=True).to_dict()
            )  # returns sorted
            for val in unique_values:
                self.vocab.set_id(val, column)

        for column in self.vocab.field_keys:
            vocab_size = len(self.vocab.token2id[column])

            if vocab_size > self.vocab.adap_thres:
                self.vocab.adap_sm_cols.add(column)

    def encode_data(self):
        """
        This function either ignores, categorizes or bins the raw data contained
        in the DataFrame, according to what is in "categorical_columns" and
        "to_quantize_columns".

        The bins for the quantization and the categories assigned can be saved
        using "save_vocab_and_binning". They can be reloaded later on the same
        data, or a different one (e.g. in the case of inference).
        """

        data = self.df

        # Put the user_dim and date_dim columns at the beginning of the dataframe.
        data.insert(0, self.user_col, data.pop(self.user_col))
        data.insert(1, self.date_col, data.pop(self.date_col))

        data = data.sort_values([self.user_col, self.date_col])

        for col_name in tqdm(self.categorical_columns, "Encoding categorical columns"):
            col_data = data[col_name]
            col_fit = self._get_categorical_encoder(col_data, col_name)

            # LabelEncoder cannot handle unknown values, so this is a workaround.
            # It is also faster to do it this way since the map method for pandas Series is optimized.
            col_fit_dict = dict(
                zip(col_fit.classes_, col_fit.transform(col_fit.classes_))
            )
            # Returns dict value; else Nan, which is filled with len(col_fit.classes_)
            col_data = col_data.map(col_fit_dict).fillna(len(col_fit.classes_))
            data[col_name] = col_data

        # Transform to_quantize_columns using the encoder_fit
        # The QuantileTransformer is applied only to the non-zero values
        # Zero values are assigned their own separate bin
        for col_name in tqdm(self.to_quantize_columns, "Encoding quantizable columns"):
            # Create mask to separate zero from non-zero values
            # Quantile binning is applied only to non-zero values
            mask = data[col_name] == 0

            # Get bin edges (for non-zero values)
            bin_edges = self._get_bin_edges(data[col_name], col_name, mask)

            # Initialize quant_data: Create array of zeros that will be filled with non-zero encoded data
            quant_data = np.zeros(len(mask))  # len(mask) = len(coldata)

            # Transform quant_data by applying the quantile transformer to non-zero values
            quant_data[~mask] = self._quantize_data(data[col_name][~mask], bin_edges)
            data[col_name] = quant_data

        # Apply memory optimization step to data
        data = optimize_memory(data)

        self.trans_table = data
        del data
        del self.df

    def _get_categorical_encoder(self, col_data, col_name):
        """
        Either create a new categorical encoded, or reload one saved on disk if specified.

        :param col_data: the pandas Series containing the column to encode.
        :param col_name: the name of the column to be transformed.
        :return: an instance of LabelEncoder to use for transforming the data.
        """
        if self.binning_from_file is not None:
            col_fit = self.encoder_fit[col_name]
        else:
            col_fit = LabelEncoder()
            col_fit.fit(col_data)
            self.encoder_fit[col_name] = col_fit

        return col_fit

    def _get_bin_edges(self, col_data, col_name, mask):
        """
        Either create a new set of bins edges for the quantization, or reload one saved on disk if specified.

        :param col_data: the pandas Series containing the column to encode.
        :param col_name: the name of the column to be transformed.
        :param mask: the mask to only select non-zero values in the column, as those are left as their own bin.
        :return: bin edges for quantizing the in the specified column.
        """
        if self.binning_from_file is not None:
            bin_edges = self.encoder_fit[col_name]
        else:
            # Creates quantiles array of [num_bins + 1] qtls of width (1/num_bins)
            qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
            # Get bin edges (for non-zero values)
            bin_edges = np.quantile(col_data[~mask], qtls, axis=0)
            # Store the bin edges in a list
            self.encoder_fit[col_name] = bin_edges

        return bin_edges

    def _quantize_data(self, data_to_bin, bin_edges):
        """
        Using the bin_edges specified, quantize the given data into bins using the pandas cut method.

        :param data_to_bin: the pandas Series containing the column to quantize.
        :param bin_edges: the bin edges to use for the pandas cut method.
        :return: the data quantized into bins.
        """

        # [num of unique bin_edges - 1] : Starts at 1, as label 0 is assigned to zero values.
        labels = [*range(1, len(np.unique(bin_edges)))]

        # Discretize variable into equal-sized buckets based on quantiles
        quant_data = pd.cut(
            x=data_to_bin,
            bins=bin_edges,
            labels=labels,
            retbins=False,
            precision=5,
            duplicates="drop",
            right=True,  # default value: True (right edges are inclusive)
            include_lowest=True,  # lowest edge of first bin is inclusive
        )

        return quant_data
