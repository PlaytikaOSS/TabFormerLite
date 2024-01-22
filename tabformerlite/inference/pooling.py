import torch

# This dict maps the 'selected_layers' specified by the
# user in the config file to the corresponding list of BERT
# layer indices. The 'selected_layers' is used with the
# following layer pooling strategies:
# * concat_pooling
# * weighted_layer_pooling
# * max_layer_pooling
SELECTED_LAYERS_MAPPING = {
    "last_two": [-1, -2],
    "last_three": [-1, -2, -3],
    "last_four": [-1, -2, -3, -4],
}


def concat_pooling_builder(selected_layers, time_pooling_strategy, nbr_days):
    """Higher-order function used for passing `selected_layers`,
    `time_pooling_strategy` and `nbr_days` parameters to the `concat_pooling`
    function below.

    Parameters
    ----------
    selected_layers : str
        The user can specify which layers to concatenate using
        the "select_layers" param in the config file. The options
        currently available are: "last_two" (concatenates the
        last two hidden layers), "last_three", and "last_four".

    time_pooling_strategy : str
        The strategy for pooling activations on the time axis. The
        available options are: "mean_pooling", "max_pooling" and
        "no_pooling"

    nbr_days : int
        The number of time dimensions over which pooling is
        performed, starting from the most recent day and up
        to -nbr_days days in the sequence.

    NOTE
    ----
    `nbr_days` should less or equal to dataset.seq_len

    Returns
    -------
    concat_pooling : function
        The concat_pooling function.

    """

    def concat_pooling(tensors):
        """The concat_pooling function:
        - concatenates the activations from the selected BERT layers
        - applies the specified "time_pooling_strategy" to the concatenated
        tensors across `nbr_days` time dimensions.

        Inputs
        ------
        tensors : tensors of shape: (n_layers, batch_size, seq_len,
            hidden_size)
            All BERT hidden states of the batch.

        NOTE
        ----
        In "concat_pooling", the 'pooling_layer' parameter is disregarded.
        This pooling method concatenates the "selected" BERT layers.

        * Shape of concatenated tensors (layer axis): (batch_size,
          seq_len, nbr_of_selected_layers x hidden_size)
        * Output shape (after pooling on the time axis): (batch_size,
          nbr_of_selected_layers x hidden_size)

        Returns
        -------
        Tensors of shape: (
        batch_size, nbr_of_selected_layers x hidden_size
        )

        """
        if selected_layers not in SELECTED_LAYERS_MAPPING:
            raise ValueError(
                f"Option {selected_layers} is not \
            supported. Please check the README file for available \
            options."
            )

        # Map "selected_layers" to list of BERT indices
        layer_range = SELECTED_LAYERS_MAPPING[selected_layers]

        # Pooling on layer axis
        # -- concatenate embeddings from BERT's last four layers
        tensors_concat = torch.cat(
            tensors=[tensors[i] for i in layer_range],
            dim=-1,
        )  # the dimension over which the tensors are concatenated

        # Pooling on time-axis
        # -- mean pooling across nbr_days dims.
        if time_pooling_strategy == "mean_pooling":
            return torch.mean(
                tensors_concat[:, -nbr_days:, :], dim=1
            )  # mean across time axis

        # -- max pooling across nbr_days dims.
        elif time_pooling_strategy == "max_pooling":
            # torch.max() returns a namedtuple: (values, indices)
            # Below, we return the values
            return torch.max(tensors_concat[:, -nbr_days:, :], dim=1)[
                0
            ]  # max across time axis
        # -- no pooling along time axis
        elif time_pooling_strategy == "no_pooling":
            return tensors_concat[:, :, :]

        else:
            raise ValueError(f"Option {time_pooling_strategy} is not supported.")

    return concat_pooling


#####################################################################


def max_layer_pooling_builder(selected_layers, time_pooling_strategy, nbr_days):
    """Higher-order function used for passing `selected_layers`,
    `time_pooling_strategy` and `nbr_days` parameters to the
    `max_layer_pooling` function below.

    Parameters
    ----------
    selected_layers : str
        The user can specify which layers to consider for applying
        "max_layer_pooling" using the "select_layers" param in the
        config file. The options currently available are:
        "last_two" (concatenates the last two hidden layers),
        "last_three", and "last_four".

    time_pooling_strategy : str
        The strategy for pooling activations on the time axis. The
        available options are: "mean_pooling", "max_pooling" and
        "no_pooling"

    nbr_days : int
        The number of time dimensions over which pooling is
        performed, starting from the most recent day and up
        to -nbr_days days in the sequence.

    NOTE
    ----
    `nbr_days` should less or equal to dataset.seq_len

    Returns
    -------
    concat_pooling : function
        The concat_pooling function.

    """

    def max_layer_pooling(tensors):
        """The max_layer_pooling function:
        - concatenates the activations from the selected BERT layers
        - applies the specified "time_pooling_strategy" to the concatenated
        tensors across `nbr_days` time dimensions.

        Inputs
        ------
        tensors : tensors of shape: (n_layers, batch_size, seq_len,
            hidden_size)
            All BERT hidden states of the batch.

        NOTE
        ----
        In "concat_pooling", the 'pooling_layer' parameter is disregarded.
        The user should specify which BERT layers to concatenate using
        the "selected_layers" parameter in the configuration file.

        * Shape of concatenated tensors (layer axis): (batch_size,
          seq_len, hidden_size)
        * Output shape (after pooling on the time axis): (batch_size,
          hidden_size)

        Returns
        -------
        Tensors of shape: (batch_size, hidden_size)

        """
        if selected_layers not in SELECTED_LAYERS_MAPPING:
            raise ValueError(
                f"Option {selected_layers} is not \
            supported. Please check the README file for available \
            options."
            )

        # Map "selected_layers" to list of BERT indices
        layer_range = SELECTED_LAYERS_MAPPING[selected_layers]

        # Pooling on layer axis
        # -- max pooling across selected layers
        tensors_max = torch.max(
            tensors[layer_range, :, :, :],
            dim=0,
        )[0]

        # Pooling on time-axis
        # -- mean pooling across nbr_days dims
        if time_pooling_strategy == "mean_pooling":
            return torch.mean(
                tensors_max[:, -nbr_days:, :], dim=1
            )  # mean across time axis

        # -- max pooling across nbr_days dims
        elif time_pooling_strategy == "max_pooling":
            # torch.max() returns a namedtuple: (values, indices)
            # Below, we return the values
            return torch.max(tensors_max[:, -nbr_days:, :], dim=1)[
                0
            ]  # max across time axis
        # -- no pooling along time axis
        elif time_pooling_strategy == "no_pooling":
            return tensors_max[:, :, :]

        else:
            raise ValueError(f"Option {time_pooling_strategy} is not supported.")

    return max_layer_pooling


#####################################################################
def single_layer_pooling_builder(pooling_layer, time_pooling_strategy, nbr_days):
    """Higher-order function used for passing `pooling_layer`,
    `time_pooling_strategy` and `nbr_days` parameters
    to the `single_layer_pooling_builder` function below.

    Parameters
    ----------
    pooling_layer : int
        The BERT layer that time-pooling will be operated on.
        Possible values are -1 to -(model.num_hidden_layers+1),
        e.g., -(12+1), where -1 means the last layer (closest
        to the output), -2 means the second-to-last, etc.

    time_pooling_strategy : str
        The strategy for pooling activations on the time axis. The
        available options are: "mean_pooling", "max_pooling" and
        "no_pooling"

    nbr_days : int
        The number of time dimensions over which pooling is
        performed, starting from the most recent day and up
        to -nbr_days days in the sequence.

    Returns
    -------
    single_layer_pooling : function
        The single_layer_pooling function.

    """

    def single_layer_pooling(tensors):
        """The single_layer_pooling function:
        - takes the activations from the specified BERT "pooling_layer"
        - applies the specified "time_pooling_strategy" to the concatenated
        tensors across `nbr_days` time dimensions.

        Inputs
        ------
        tensors : tensors of shape: (n_layers, batch_size, seq_len,
            hidden_size)
            All BERT hidden states of the batch.

        NOTE
        ----
        * Tensors' shape (layer axis): (batch_size, seq_len, hidden_size)
        * Output shape (after pooling on the time axis): (batch_size,
          hidden_size)

        Returns
        -------
        Tensors of shape: (batch_size, hidden_size)

        """

        # Pooling on time-axis
        # -- mean pooling across nbr_days dims.
        if time_pooling_strategy == "mean_pooling":
            return torch.mean(
                tensors[pooling_layer, :, -nbr_days:, :], dim=1
            )  # mean across time axis
        # -- max pooling across nbr_days dims.
        elif time_pooling_strategy == "max_pooling":
            # torch.max() returns a namedtuple: (values, indices)
            # Below, we return the values
            return torch.max(tensors[pooling_layer, :, -nbr_days:, :], dim=1)[
                0
            ]  # max across time axis
        # -- no pooling along time axis
        elif time_pooling_strategy == "no_pooling":
            return tensors[pooling_layer, :, :, :]

        else:
            raise ValueError(f"Option {time_pooling_strategy} is not supported.")

    return single_layer_pooling


#####################################################################
class WeightedLayerPooling(torch.nn.Module):
    """
    Adapted from here:
    https://www.kaggle.com/code/rhtsingh/\
    utilizing-transformer-representations-efficiently

    This class takes the weighted average of activations from
    the selected BERT layers, and applies the specified
    "time_pooling_strategy" across `nbr_days` time dimensions.

    Note
    ----
    - Unless specified, layer_weights default to 1.

    """

    def __init__(
        self,
        bert_layers_nbr,
        time_pooling_strategy,
        nbr_days,
        device,
        selected_layers: str,
        layer_weights=None,
    ):
        super(WeightedLayerPooling, self).__init__()

        self.layer_range = SELECTED_LAYERS_MAPPING[selected_layers]
        self.num_hidden_layers = bert_layers_nbr
        self.nbr_days = nbr_days
        self.device = device
        self.time_pooling_strategy = time_pooling_strategy

        # layer_weights default to 1
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else torch.nn.Parameter(
                torch.tensor(
                    [1] * (len(self.layer_range)),
                    dtype=torch.float,
                    device=self.device,
                )
            )
        )

    def forward(self, all_hidden_states):
        """The forward function of the model"""
        all_layer_embedding = all_hidden_states[self.layer_range, :, :, :]  # noqa: E402
        weight_factor = (
            self.layer_weights.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(all_layer_embedding.size())
        )

        # Compute weighted sum of selected layers
        # Shape: (batch_size, seq_len, hidden_size)
        weighted_average = (weight_factor * all_layer_embedding).sum(
            dim=0
        ) / self.layer_weights.sum()

        # Pooling on time-axis
        # -- mean pooling across nbr_days dims.
        if self.time_pooling_strategy == "mean_pooling":
            return torch.mean(
                weighted_average[:, -self.nbr_days :, :], dim=1  # noqa: E402
            )  # mean across time-axis
        # -- max pooling across nbr_days dims.
        elif self.time_pooling_strategy == "max_pooling":
            # torch.max() returns a namedtuple: (values, indices)
            # Below, we return the values
            return torch.max(
                weighted_average[:, -self.nbr_days :, :], dim=1  # noqa: E402
            )[
                0
            ]  # max across time axis
        elif self.time_pooling_strategy == "no_pooling":
            return weighted_average[:, :, :]

        else:
            raise ValueError(f"Option {self.time_pooling_strategy} is not supported.")
