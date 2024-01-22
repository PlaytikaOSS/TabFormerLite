import numpy as np
from loguru import logger


def select_discretization_strategy(df, n_max, config, verbose=False):
    """
    This function is responsible for determining the discretization
    scheme to be applied to each column in the dataset.

    Arguments:
    - df: Pandas DataFrame containing the raw data that needs to be encoded
    before feeding to the TabFormer model.
    - n_max: integer, threshold of unique values in a column. If the column
    is not of "object" dtype and has more than "nmax" unique values, then it
    will be discretized using quantile binning (see below for more
    explanations).
    - config: json, configuration file.
    - verbose: If True, print the number of columns in each list.

    Returns:
    A dictionary containing:
    - categorical_columns: List of categorical columns.
    - to_quantize_columns: List of columns to be discretized using quantile
    binning.
    - cols_no_transform: List of columns that will not be transformed.

    A dataset may contain a mix of categorical, discrete continuous variables.
    The TabDataset implements quantization of continuous variables, enabling
    each variable to possess a well-defined and finite vocabulary. This process
    bears similarities to language modeling tasks in which each word is drawn
    from a finite vocabulary.

    The data pre-processing step offers three options for data discretization:
    - LabelEncoder: This is applied to categorical columns. It encodes
    categorical values into numerical labels.
    - Quantile Binning: This is used for continuous columns. It discretizes
    continuous data into quantiles or bins, which can be thought of as buckets
    or categories.
    - No Transformation: For columns that are already "discrete", such as
    columns with integer values, no additional transformation is applied.

    However, real-world datasets often defy conventional data type expecta-
    tions. For instance, discrete variables may be labeled with "float64"
    rather than "int64" data types. Consequently, in such cases, we cannot
    solely rely on default data types to determine which transformation to
    apply.

    To address this, we adopt the following approach:
    - If the data type is 'object', the column is considered categorical and
    will be transformed using the Label Encoder. We group these columns in a
    list denoted as "categorical_columns".
    - If the data type is not 'object' (indicating that the column could be
    either discrete or continuous), we employ the following criteria:
        * If the column contains less unique values than a specified threshold
        of unique values, denoted as "n_max", then the column is left unalte-
        red. We group these columns in a list denoted as "cols_no_transform".
        * If the column contains more unique values than the specified thres-
        hold "n_max", it is discretized using quantile binning. We
        group these columns in a list denoted as "to_quantize_columns".

    Users are encouraged to customize the "n_max" parameter to establish a
    threshold that determines which columns should undergo transformation
    with quantile binning. For example, if "n_max" is set to 50,
    columns whose dtype is not "object" and have less than 50 unique values
    will remain unchanged, while those with more than 50 unique values will
    be discretized using the quantile binning.
    """

    # Define useful variables
    user_col = config["user_col"]
    date_col = config["date_col"]
    target_col = config["target_col"]

    # Create list of categorical columns (-> Label Encoder)
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Create list of non-categorical columns (int or float)
    other_columns = df.select_dtypes(exclude=["object"]).columns.tolist()

    # Split other_columns into cols_no_transform and to_quantize_columns

    # columns with <= n_max unique values -> no transformation (int or float)
    cols_no_transform = [col for col in other_columns if df[col].nunique() <= n_max]

    # columns with > n_max unique values -> quantile binning
    to_quantize_columns = [col for col in other_columns if df[col].nunique() > n_max]

    # Remove target_col, user_col and date_col from the lists
    for col_name in [target_col, user_col, date_col]:
        if col_name in cols_no_transform:
            cols_no_transform.remove(col_name)
        if col_name in to_quantize_columns:
            to_quantize_columns.remove(col_name)

    if verbose:
        logger.info(f"Nbr of categorical cols: {len(categorical_columns)}")
        logger.info(f"Nbr of to_quantize_columns: {len(to_quantize_columns)}")
        logger.info(f"Nbr of cols_no_transform: {len(cols_no_transform)}\n")

    # Sanity check:
    # Total nbr of cols: categorical + to_quantize + cols_no_transform + 3,
    # where 3 corresponds to: (target_col, user_col, date_col)
    assert len(to_quantize_columns) + len(categorical_columns) + len(
        cols_no_transform
    ) + 3 == len(df.columns), (
        "Please check the provided data for missing colums. The total nbr of "
        "columns should be: categorical + to_quantize + cols_no_transform + 3,"
        " where 3 corresponds to: (target_col, user_col, date_col)"
    )

    return {
        "categorical_columns": categorical_columns,
        "to_quantize_columns": to_quantize_columns,
        "cols_no_transform": cols_no_transform,
    }


def optimize_memory(df, verbose=False):
    """
    Function to optimize memory usage based on machine limits for integer,
    float and object types.

    Source: stackoverflow.com
    """

    if verbose:
        initial = df.memory_usage(index=False, deep=True).sum() / 1024**2
        logger.info(f"* Initial memory footprint: {initial} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            if col_type.name != "category":
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df[col] = df[col].astype(np.int16)
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df[col] = df[col].astype(np.int32)
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df[col] = df[col].astype(np.int64)
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df[col] = df[col].astype(np.float16)
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    if verbose:
        final = df.memory_usage(index=False, deep=True).sum() / 1024**2
        logger.info(f"* Final memory footprint: {final} MB")

        logger.info(
            "* Memory footprint decreased by {:.1f}%".format(
                100 * (initial - final) / initial
            )
        )

    return df
