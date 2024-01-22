import os
import re
from glob import glob

import torch
from loguru import logger

from tabformerlite.models.modules import TabFormerBertLM


class TabNetLoader:
    """
    This class can be used to load a pre-trained TabNet model from
    a checkpoint.

    Inputs
    ------
    dataset_config : dict
        A dictionary with info about the data used to pretrain TabFormer

    model_config : dict
        A dictionary with info about the pretrained TabFormer model

    exp_config : dict
        A dictionary with info about the experiment which can contain
        the "ckpt_dir" (optional)

    model_path : str
        The path to the pre-trained model directory, where all model
        checkpoints are available.

    model_class: class
        If the user wants to instanciate a different model, the class
        to use can be specified here.

    Note
    ----
    - The model weights are loaded from the most recent checkpoint
    available in the provided "model_path" directory. Alternatively,
    the user can specify the desired checkpoint-id to use, utilizing
    the "ckpt_dir" argument in the "experiment_configuration.json"
    file, e.g.,: ckpt_dir : "checkpoint-9696".

    - The checkpoint directory should contain a "pytorch_model.bin";
    otherwise, an AssertionError will be raised.

    Attributes
    ----------
    ckpt_dir : str
        The path to the checkpoint directory from which we want to
        load model weights. It should contain a "pytorch_model.bin"
        file from which the model weights will be loaded.

    config : dict
        The configuration of the pre-trained model.

    Example Usage
    -------------
    >>> loader = TabNetLoader(
        dataset_config,
        model_config,
        exp_config,
        model_path
        )
    >>> tab_net = loader.from_pretrained()

    """

    def __init__(
        self, dataset_config, model_config, exp_config, model_path, model_class=None
    ):
        # Create model_config dictionary
        self.model_config = {
            "special_tokens": dataset_config["special_tokens"],
            "vocab": dataset_config["vocab"],
            "ncols": dataset_config["ncols"],
            "field_hidden_size": model_config["field_hidden_size"],
            "tab_embeddings_num_attention_heads": model_config[
                "tab_embeddings_num_attention_heads"
            ],
            "tab_embedding_num_encoder_layers": model_config[
                "tab_embedding_num_encoder_layers"
            ],
            "tab_embedding_dropout": model_config["tab_embedding_dropout"],
            "num_attention_heads": model_config["num_attention_heads"],
            "num_hidden_layers": model_config["num_hidden_layers"],
            "hidden_size": model_config["hidden_size"],
            "mlm_average_loss": model_config["mlm_average_loss"],
        }

        self.model = None
        # If specified, will use this class for the model.
        self.model_class = model_class

        # If the model class is specified, let the model weights
        # reloading not be strict since some parameters will be
        # different.
        self.strict_weights_loading = model_class is None

        # Fetch checkpoint directory
        self.ckpt_dir = self.retrieve_checkpoint_path(exp_config, model_path)

        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    ##########################################################################
    def retrieve_checkpoint_path(self, exp_config, model_path):
        """
        Retrieve the specified checkpoint in the experiment config.
        If no checkpoint is specified, retrieve the most recent one.
        """
        # If available, read ckpt_dir from config_file
        _ckpt_dir = exp_config["pretrained_model_config"].get("ckpt_dir", None)

        # If ckpt_dir is an empty str, set it to None
        if isinstance(_ckpt_dir, str) and len(_ckpt_dir) == 0:
            _ckpt_dir = None

        # Join model_path with _ckpt_dir if necessary
        if _ckpt_dir is not None and (model_path not in _ckpt_dir):
            _ckpt_dir = os.path.join(model_path, _ckpt_dir)

        # If _ckpt_dir is None,
        # retrieve path of most recent ckpt in model directory
        if _ckpt_dir is None:
            all_ckpts_dirs = glob(f"{model_path}/checkpoint-*")
            assert (
                len(all_ckpts_dirs) != 0
            ), f"No checkpoints were found in {model_path}."
            last_ckpt = sorted(
                all_ckpts_dirs, key=lambda f: [int(n) for n in re.findall(r"\d+", f)]
            )[-1]

            _ckpt_dir = last_ckpt

        # Raise AssertionError if _ckpt_dir is empty (or doesn't exist)
        assert os.path.isfile(f"{_ckpt_dir}/pytorch_model.bin"), (
            "We didn't find any 'pytorch_model.bin' file in the "
            "provided ckpt path. "
            "Please provide a valid checkpoint path."
        )

        return _ckpt_dir

    ##########################################################################
    def load_weights_from_ckpt(self):
        """
        Load model weights from checkpoint.
        """
        logger.info(f"Loading weights from checkpoint: {self.ckpt_dir}")

        if torch.cuda.is_available():
            return self.model.model.load_state_dict(
                torch.load(f"{self.ckpt_dir}/pytorch_model.bin"),
                strict=self.strict_weights_loading,
            )
        else:
            return self.model.model.load_state_dict(
                torch.load(
                    f"{self.ckpt_dir}/pytorch_model.bin",
                    map_location=torch.device("cpu"),
                ),
                strict=self.strict_weights_loading,
            )

    ##########################################################################
    def to_device(self, device):
        """
        Move the model to the specified pytorch device, CPU or GPU.
        """
        logger.info(f"Sending model to {self.device} device.")

        # Set device
        self.device = device

        if torch.cuda.is_available():
            self.model.model.to(self.device)

    ##########################################################################
    def from_pretrained(self):
        """
        Instanciate a model and load weights from checkpoint.
        """
        logger.info("Loading TabFormer model - START")

        # Create an instance of the model
        if self.model_class is None:
            # Pretraining with MLM masking
            self.model = TabFormerBertLM(**self.model_config)
        else:
            # Fine-tuning using TabFormerBertClassification
            self.model = self.model_class(**self.model_config)

        # Load weights from checkpoint
        # -- Should default to True
        load_weights_from_pretraining = self.model_config.get(
            "load_weights_from_pretraining", True
        )

        if load_weights_from_pretraining:
            logger.info("Loading weights from pretrained model\n")
            self.load_weights_from_ckpt()

        # Send model to specified device
        self.to_device(self.device)

        logger.info("Loading TabFormer model - END\n")

        return self.model
