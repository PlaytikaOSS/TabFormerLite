import warnings

import torch
from loguru import logger
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from tabformerlite.dataset.datacollator import TabDataCollatorForLanguageModeling
from tabformerlite.inference.pooling import (
    WeightedLayerPooling,
    concat_pooling_builder,
    max_layer_pooling_builder,
    single_layer_pooling_builder,
)


class EmbeddingsExtractor:
    """
    This class can be used to extract and aggregate activations
    from one or more BERT layers using different pooling strategies,
    both on the time and the layer axis.

    Inputs
    ------
    model : class
        The model class with loaded weights.

    config_file : dict
        A dictionary with information on the pooling strategies
        (across time and layer axes) and the batch_size to use.

    Notes
    -----
    Pooling across the time axis
        The supported pooling strategies are:
        * "mean_pooling",
        * "max_pooling",
        * "no_pooling".

        The "nbr_days" parameter is the number of
        time steps over which pooling is performed, starting
        from the most recent day and up to -nbr_days
        days in the sequence. Note that "nbr_days" should be
        less or equal to "seq_len".

        - In "mean_pooling", the activations from the selected
        layer(s) are averaged on the time axis over `nbr_days`,
        starting from the most recent one.

        - In "max_pooling", we return the maximum activation on
        the time axis from the selected layer(s) across
        `nbr_days`.

        - In "no_pooling", we return the activations from the
        selected layer(s) for all time steps in the sequence
        (no time pooling performed).


    Pooling across Bert layers
        The supported pooling strategies are:
        * "concat_pooling",
        * "single_layer_pooling"
        * "max_layer_pooling"
        * "weighed_layer_pooling".

        - In "concat_pooling", we concatenate activations from
        several hidden layers of the BERT model. In this case,
        the user can specify which layers to concatenate using
        the "select_layers" parameter in the config file. The
        available options are: "last_two" (concatenates the
        last two hidden layers), "last_three", and "last_four".
        In this case, the 'pooling_layer' parameter is
        disregarded.

        - In "max_pooling", we return the maximum activation
        from the selected BERT layers. The user can specify
        which layers to pool from using the "select_layers"
        param in the config file. The available options are:
        "last_two" (concatenates the last two hidden layers),
        "last_three", and "last_four". In this case, the
        'pooling_layer' parameter is disregarded.

        - In "single_layer_pooling", we return the activations
        from the specified BERT layer. The user specifies which
        BERT layer to return using the "pooling_layer" parameter
        in the config file. Possible values for the the
        "pooling_layer" parameter are from: -1 up to
        -(model.num_hidden_layers+1), e.g., -(12+1), where -1
        means the last layer (closest to the output), -2 means
        the second-to-last, etc.

        - In "weighed_layer_pooling", we average activations
        from several BERT layers. In this case, the
        'pooling_layer' parameter is disregarded. The user
        can specify which layers to concatenate using the
        "select_layers" param in the config file. The available
        options are: "last_two", "last_three", and "last_four".

    Methods
    -------
    set_dataloader(dataset)
        This method returns the data loader.

    build_pooling_fn:
        This method builds the pooling function using
        the specified pooling parameters from the config file.

    extract_embeddings(dataset)
        This method extracts activations from BERT layers in
        batches using the built pooling function. This function
        returns a tuple with the extracted users_ids, dates, and
        embeddings as lists.

    retrieve_usecase_filename(ckpt_dir)
        This method takes the "ckpt_dir" that we used to load
        model weights from and returns the "usecase_filename". The
        "usecase_filename" will be used later to name the output
        parquet file where the labeled embeddings will be exported.

    Below we show the naming convention for the "usecase_filename":
    (
        "checkpoint-xxxx_{layer_used}_"
        "{self.layer_pooling_strategy}_"
        "{self.time_pooling_nb_days}_days_"
        "{self.time_pooling_strategy}.parquet"
    )


    Example usage
    -------------
    >>> loader = TabNetLoader(dataset_config, model_config, exp_config, model_path)
    >>> extractor = EmbeddingsExtractor(tabnet_model, exp_config["inference_config"])
    >>> outputs_train = extractor.extract_embeddings(train_dataset)
    >>> usecase_filename = extractor.retrieve_usecase_filename(loader.ckpt_dir)
    """

    def __init__(self, model, config_file):
        self.model = model
        self.config_file = config_file

        # Define pooling params
        # -- pooling on time axis
        self.time_pooling_strategy = self.config_file["pooling_on_time_axis"].get(
            "strategy", "mean_pooling"
        )
        if self.time_pooling_strategy != "no_pooling":
            self.time_pooling_nb_days = self.config_file["pooling_on_time_axis"].get(
                "nbr_days", 3
            )
        else:
            self.time_pooling_nb_days = None

        # -- pooling on layer axis
        self.layer_pooling_strategy = self.config_file["pooling_on_layer_axis"].get(
            "strategy", "concat_pooling"
        )
        self.pooling_layer = self.config_file["pooling_on_layer_axis"].get(
            "pooling_layer", None
        )

        self.selected_layers = self.config_file["pooling_on_layer_axis"].get(
            "selected_layers", None
        )

        # Some testing
        # ------------
        assert self.time_pooling_strategy in (
            "mean_pooling",
            "max_pooling",
            "no_pooling",
        ), (
            f"Pooling strategy on time axis: {self.time_pooling_strategy} "
            "is not supported. "
            "Please check the README file for available options.\n"
        )
        assert self.layer_pooling_strategy in (
            "single_layer_pooling",
            "concat_pooling",
            "weighed_layer_pooling",
            "max_layer_pooling",
        ), (
            f"Pooling strategy on layer axis: {self.layer_pooling_strategy} "
            "is not supported. "
            "Please check the README file for available options.\n"
        )

        if (
            self.layer_pooling_strategy != "single_layer_pooling"
            and self.pooling_layer is not None
        ):
            warnings.warn(
                f"In {self.layer_pooling_strategy}, the 'pooling_layer' "
                "parameter is disregarded. Please use the 'selected_layers' "
                "parameter in the configuration file to specify which BERT "
                "layers you want to use.\n"
            )

        if (self.layer_pooling_strategy != "single_layer_pooling") and (
            self.selected_layers is None
        ):
            raise AssertionError(
                f"In {self.layer_pooling_strategy}, please use the "
                "'selected_layers' parameter in the configuration "
                "file to specify which BERT layers you want to use.\n"
            )

        if (self.layer_pooling_strategy == "single_layer_pooling") and (
            self.pooling_layer is None
        ):
            raise AssertionError(
                "In single_layer_pooling, please specify a BERT layer for "
                "extracting activations, using the 'pooling_layer' parameter "
                "in the configuration file.\n"
            )

        if (self.layer_pooling_strategy == "single_layer_pooling") and (
            abs(self.pooling_layer) > self.model.config.num_hidden_layers + 1
        ):
            raise AssertionError(
                f"Please specify a pooling_layer ({self.pooling_layer}) that "
                "is smaller than the total BERT layers + 1 "
                f"({self.model.config.num_hidden_layers+1}).\n"
            )

        # Logging
        # -------
        logger.info(
            f"Strategy for pooling on layer axis: {self.layer_pooling_strategy}"
        )
        if self.layer_pooling_strategy == "single_layer_pooling":
            logger.info(f"Pooling layer used: {self.pooling_layer}")
        elif self.layer_pooling_strategy != "single_layer_pooling":
            logger.info(f"Selected layers: {self.selected_layers}")

        logger.info(f"Strategy for pooling on time axis: {self.time_pooling_strategy}")
        if self.time_pooling_strategy != "no_pooling":
            logger.info(f"time_pooling_nb_days: {self.time_pooling_nb_days}\n")

    ############################################################
    def set_dataloader(self, dataset):
        """
        This method returns the data loader to use when extracting embeddings.
        """

        # Create data collator
        data_collator = TabDataCollatorForLanguageModeling(
            tokenizer=self.model.tokenizer,
            mlm=False,
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=self.config_file["batch_size"],
            collate_fn=data_collator,
            drop_last=False,
        )

        return dataloader

    #############################################################
    def build_pooling_fn(self):
        """Builds the pooling function"""

        if self.layer_pooling_strategy == "concat_pooling":
            pooling_fn = concat_pooling_builder(
                self.selected_layers,
                self.time_pooling_strategy,
                self.time_pooling_nb_days,
            )
        elif self.layer_pooling_strategy == "max_layer_pooling":
            pooling_fn = max_layer_pooling_builder(
                self.selected_layers,
                self.time_pooling_strategy,
                self.time_pooling_nb_days,
            )
        elif self.layer_pooling_strategy == "single_layer_pooling":
            pooling_fn = single_layer_pooling_builder(
                self.pooling_layer,
                self.time_pooling_strategy,
                self.time_pooling_nb_days,
            )
        elif self.layer_pooling_strategy == "weighed_layer_pooling":
            pooling_fn = WeightedLayerPooling(
                bert_layers_nbr=self.model.config.num_hidden_layers,
                time_pooling_strategy=self.time_pooling_strategy,
                nbr_days=self.time_pooling_nb_days,
                device=self.model.model.device,
                selected_layers=self.selected_layers,
                layer_weights=None,  # None: All weights are equal to 1.0
            )
        else:
            raise ValueError(
                f"{self.layer_pooling_strategy} strategy is not supported."
            )

        return pooling_fn

    #############################################################
    def extract_embeddings(self, dataset):
        """Extracts and aggregates activations from one or more
        BERT layers in batches using the specified pooling
        strategies.

        Notes
        -----
        batch : dict
            A dict with the following keys:
            ['input_ids', 'labels', 'user_ids', 'date']

        * input_ids shape: (batch_size, seq_len, ncols)
        * labels shape: (batch_size, seq_len, ncols) WHY??
        * user_ids shape: (batch_size)
        * date shape: (batch_size)

        hidden_states : Tensor
            Bert hidden states of shape:
            (n_layers, batch_size, seq_len, hidden_size)

        Returns
        -------
        A tuple with the extracted (users_ids, dates, embeddings)
        in lists.

        """

        if self.time_pooling_strategy != "no_pooling":
            # Check that nb_days <= seq_len
            assert (
                self.time_pooling_nb_days <= dataset.seq_len
            ), f"nb_days should be less or equal to: {dataset.seq_len}"

            # Check that nb_days != 0
            assert self.time_pooling_nb_days != 0, (
                "Please provide a non-zero value for aggregating "
                "activations on the time axis"
            )

        # Create data_loader
        dataloader = self.set_dataloader(dataset)

        # Create instance of the class removing MLM head
        prediction_m = PredictionModel(self.model.model)
        prediction_m.eval()

        # Create pooling function
        pooling_fn = self.build_pooling_fn()

        # Empty lists
        embeddings = []
        users_ids = []
        dates = []

        for batch in tqdm(dataloader):
            # Place input tensors on the same device as the model
            batch = {k: v.to(self.model.model.device) for k, v in batch.items()}

            # load Bert hidden states
            hidden_states = torch.stack(
                prediction_m(input_ids=batch["input_ids"], output_hidden_states=True)[
                    "hidden_states"
                ]
            )

            # embed_batch shape: (batch_size, hidden_size)
            embed_batch = pooling_fn(hidden_states)

            # Collect batch data in lists
            embeddings.extend(embed_batch.detach().cpu().numpy())
            users_ids.extend(batch["user_ids"].cpu().numpy())
            dates.extend(batch["date"].cpu().numpy())

        # Delete data from memory
        del dataset.data
        del dataset.labels

        return (users_ids, dates, embeddings)

    ################################################################
    def retrieve_usecase_filename(self, ckpt_dir):
        """This method takes the "ckpt_dir" that we used to load
        model weights from and returns the "usecase_filename". The
        "usecase_filename" will be later used to name the output
        parquet file where the labelled embeddings will be exported.

        Below we show the naming convention for the
        "usecase_filename":

        (
            "checkpoint-xxxx_{layer_used}_"
            "{self.layer_pooling_strategy}_"
            "{self.time_pooling_nb_days}_days_"
            "{self.time_pooling_strategy}.parquet"
        )

        """
        # Get ckpt id (e.g.: checkpoint-xxxx)
        _ckpt_id = ckpt_dir.split("/")[-1]

        # Create `layer_used` name
        # --- layer axis
        if self.layer_pooling_strategy == "single_layer_pooling":
            if self.pooling_layer == -1:
                layer_used = "last_hidden"
            elif self.pooling_layer == -2:
                layer_used = "second_to_last_hidden"
            else:
                layer_used = "layer_" + str(self.pooling_layer)
        elif self.layer_pooling_strategy == "concat_pooling":
            layer_used = self.selected_layers + "_hidden"
        elif self.layer_pooling_strategy == "weighed_layer_pooling":
            layer_used = self.selected_layers + "_hidden"
        elif self.layer_pooling_strategy == "max_layer_pooling":
            layer_used = self.selected_layers + "_hidden"
        else:
            layer_used = "layer_" + str(self.layer_pooling_strategy)

        usecase_filename = (
            f"{_ckpt_id}_{layer_used}_{self.layer_pooling_strategy}_"
            f"{self.time_pooling_nb_days}_days_"
            f"{self.time_pooling_strategy}.parquet"
        )

        return usecase_filename


###############################################################################
class PredictionModel(torch.nn.Module):
    """Simple class to remove the classifier head of the TabFormer model
    and extract hidden states from pre-trained model.

    Use output_hidden_states = True, to export all hidden states (n_layers + 1)
    """

    def __init__(self, tabnet_model):
        super().__init__()

        self.tab_embeddings = tabnet_model.tab_embeddings
        self.tabnet_bert = tabnet_model.tb_model.bert

    def forward(
        self,
        input_ids,
        output_hidden_states=False,
        **input_args,
    ):
        """Forward pass through the model."""
        inputs_embeds = self.tab_embeddings(input_ids)

        return self.tabnet_bert(
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            **input_args,
        )
