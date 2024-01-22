import datetime
import json
import os
from argparse import ArgumentParser
from pprint import pprint

import torch
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import random_split
from transformers import (
    Trainer,
    TrainingArguments,
    get_polynomial_decay_schedule_with_warmup,
    set_seed,
)

from tabformerlite.dataset.datacollator import TabDataCollatorForLanguageModeling
from tabformerlite.dataset.dataset import TabDataset
from tabformerlite.misc.training_helpers import (
    FileLoggingCallback,
    compute_metrics,
    get_model_size,
    preprocess_logits_for_mlm_metrics_builder,
)
from tabformerlite.models.modules import TabFormerBertLM


def main():
    """
    Main function to execute from command line.

    $ python3 scripts/run_mlm_pretraining.py -cfg --path-to-config-files

    Example usage
    -------------
    >>> $ python3 scripts/run_mlm_pretraining.py \
    -cfg ./configs/example/pretraining/config_card_dataset_size_300.json

    This script is used to pretrain the TabFormerLite model using
    masked language modeling (MLM). It will save checkpoints of
    the model in the output folder specified in the configuration
    file.
    """

    # Parse config file
    parser = ArgumentParser()
    parser.add_argument("-cfg", "--config-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf8") as f:
        config = json.load(f)

    # Define device and other parameters
    device = "cuda" if torch.cuda.is_available() else torch.device("cpu")
    nbr_gpu = max(1, torch.cuda.device_count())

    # Set random seeds in  `random`, `numpy` and `torch`
    set_seed(config["seed"])

    # Define path to encoders, vocabulary and encoded dataset
    encoded_data_dir = config["encoded_data_dir"]
    encoded_data_folder = config["encoded_data_folder"]
    encoded_data_path = os.path.join(encoded_data_dir, encoded_data_folder)
    encoded_data_path = os.path.abspath(encoded_data_path)

    # Define path for storing pre-training outputs
    output_dir = config["output_dir"]
    output_folder = config["output_folder"]

    # Add date suffix to output filenames
    if config["add_date_suffix_to_output"]:
        suffix = datetime.datetime.now().strftime("%d_%m_%Y")
        output_path = os.path.join(output_dir, output_folder + f"_{suffix}")
    else:
        output_path = os.path.join(output_dir, output_folder)
    logger.info(f"Output path: {output_path}\n")

    output_path = os.path.abspath(output_path)

    # Create output directory in specified path
    os.makedirs(output_path, mode=0o777, exist_ok=True)

    # Save the configuration file in the output folder
    with open(os.path.join(output_path, "config.json"), "w", encoding="utf8") as fh:
        json.dump(config, fh)

    #######################################################################
    # Load encoded dataset
    logger.info("Loading encoded dataset\n")

    # Read json file with lists for discretization
    with open(
        os.path.join(encoded_data_path, "column_lists_by_dtype.json"),
        "r",
        encoding="utf8",
    ) as f:
        discretization_dict = json.load(f)

    # Load encoded data config file
    with open(
        os.path.join(encoded_data_path, "config.json"), "r", encoding="utf8"
    ) as f:
        encoded_data_config = json.load(f)

    # Define kwargs for dataset
    dataset_kwargs = {
        "categorical_columns": discretization_dict["categorical_columns"],
        "to_quantize_columns": discretization_dict["to_quantize_columns"],
        "seq_len": encoded_data_config["seq_len"],
        "num_bins": encoded_data_config["num_bins"],
        "label_col": encoded_data_config["target_col"],
        "user_col": encoded_data_config["user_col"],
        "date_col": encoded_data_config["date_col"],
        "stride": encoded_data_config["stride"],
        "return_labels": False,
        "vocab_from_file": os.path.join(encoded_data_path, "vocab.pickle"),
        "binning_from_file": os.path.join(encoded_data_path, "binning.pickle"),
        "processed_data_from_file": os.path.join(
            encoded_data_path, "processed_data_and_labels.h5"
        ),
        "device": device,
    }

    # Creating an instance of the TabDataset
    dataset = TabDataset(df=None, **dataset_kwargs)

    # Extract vocab, ncols and special tokens
    vocab = dataset.vocab
    custom_special_tokens = vocab.get_special_tokens()

    #######################################################################
    # Split dataset in train/val datasets

    # Train, val split
    total_n = len(dataset)
    train_n = int(0.99 * total_n)  # (99/1 split)
    val_n = total_n - train_n
    assert total_n == train_n + val_n

    lengths = [train_n, val_n]

    train_dataset, eval_dataset = random_split(dataset, lengths)

    #######################################################################
    # Initialize TabFormer model
    logger.info("Creating an instance of the TabFormer model")

    tab_net = TabFormerBertLM(
        special_tokens=custom_special_tokens,
        vocab=vocab,
        ncols=dataset.ncols,
        field_hidden_size=config["field_hidden_size"],
        tab_embeddings_num_attention_heads=config["tab_embeddings_num_attention_heads"],
        tab_embedding_num_encoder_layers=config["tab_embedding_num_encoder_layers"],
        tab_embedding_dropout=config["tab_embedding_dropout"],
        num_attention_heads=config["num_attention_heads"],
        num_hidden_layers=config["num_hidden_layers"],
        hidden_size=config[
            "hidden_size"
        ],  # must be a multiple of "num_attention_heads"
        mlm_average_loss=config["mlm_average_loss"],
    )

    # Extract number of model parameters
    model_params = tab_net.model.num_parameters()

    # Extract model size (and convert to GB)
    model_size_torch = get_model_size(tab_net.model) / 1024  # GB

    # Collect information about TabFormer model
    tabformer_info_dict = [
        {
            "ncols": dataset.ncols,
            "model_params": model_params,
            "model_size (GB)": model_size_torch,
            "field_hidden_size": tab_net.config.field_hidden_size,
            "tab_embeddings_num_attention_heads": tab_net.config.tab_embeddings_num_attention_heads,
            "tab_embedding_num_encoder_layers": tab_net.config.tab_embedding_num_encoder_layers,
            "tab_embedding_dropout": tab_net.config.tab_embedding_dropout,
            "num_attention_heads": tab_net.config.num_attention_heads,
            "num_hidden_layers": tab_net.config.num_hidden_layers,
            "hidden_size": tab_net.config.hidden_size,
            "mlm_average_loss": tab_net.config.mlm_average_loss,
            "vocab_size": tab_net.config.vocab_size,
        }
    ]

    # Show info about TabFormer model
    logger.info("Model parameters:\n")
    pprint(tabformer_info_dict, indent=4)
    print()

    # Extract field_names (i.e. the column names)
    # Ignore the label and include special tokens to match the samples' encoding
    field_names = vocab.get_field_keys(remove_target=True, ignore_special=False)

    ##############################################################################
    # Define training arguments

    # Data collator
    data_collator = TabDataCollatorForLanguageModeling(
        tokenizer=tab_net.tokenizer, mlm=True, mlm_probability=config["mlm_probability"]
    )
    # Batch size
    batch_size = config["batch_size"]

    # Number of gradient accumulation steps
    grad_acc_steps = int(config["grad_acc_steps"] / nbr_gpu)

    # Get number of steps per epoch
    steps_per_epoch = int(train_n / batch_size / grad_acc_steps)

    # Log N times per epoch
    logging_per_epoch = config["logging_per_epoch"]
    nbr_logging_steps = int(steps_per_epoch / logging_per_epoch)

    # Total number of epochs
    num_epochs = config["num_epochs"]

    # Resume pre-training from checkpoint
    if config["resume_from_checkpoint"]:
        assert "checkpoint_dir" in config, "Please provide a checkpoint directory"
        ckpt_dir = config["checkpoint_dir"]

        # Check that ckpt_dir isn't an empty str
        assert (
            isinstance(ckpt_dir, str) and len(ckpt_dir) != 0
        ), "Please provide a valid checkpoint directory"

    # Create checkpoint every N epochs
    ckpt_every_n = config["checkpoint_every_N_epochs"]
    checkpoint_steps = int(steps_per_epoch * ckpt_every_n / nbr_gpu)

    # Maximum learning rate
    lr_max = config["lr_max"]

    # Warm-up steps
    warm_up_n = config["warmup_steps_in_epochs"]
    warm_up_steps = int(steps_per_epoch * warm_up_n)

    # Collect training arguments
    training_args_user = [
        {
            "batch_size": batch_size,
            "grad_acc_steps": grad_acc_steps,
            "effective batch_size": batch_size * grad_acc_steps * nbr_gpu,
            "num_epochs": num_epochs,
            "steps per epoch": steps_per_epoch,
            "total_steps": num_epochs * steps_per_epoch,
            "logging_steps": nbr_logging_steps,
            "save_steps": checkpoint_steps * nbr_gpu,
            "lr_max": lr_max,
            "warm_up_steps": warm_up_steps * nbr_gpu,
            "nbr_gpu": nbr_gpu,
            "scheduler": "inverse_sqrt_with_warm_up",
        }
    ]

    # Combine training arguments with tabformer_info_dict
    training_args_user.append(tabformer_info_dict)
    json_string = json.dumps(training_args_user)

    # Export training with tabformer_info_dict
    with open(
        os.path.join(output_path, "training_args.json"), "w", encoding="utf8"
    ) as json_file:
        json_file.write(json_string)

    logger.info("Training arguments and model parameters:\n")
    pprint(training_args_user, indent=4)
    print()

    ##############################################################################
    # Create an instance of the TrainingArguments class
    training_args = TrainingArguments(
        output_dir=f"{output_path}",  # output directory
        logging_dir=f"{output_path}/logging/",  # TensorBoard log directory
        num_train_epochs=num_epochs,  # number of training epochs
        learning_rate=lr_max,  # max learning rate
        warmup_steps=warm_up_steps,  # nbr of steps used for a linear warmup from 0 to lr_max
        save_steps=checkpoint_steps,  # nbr of steps for creating model checkpoints
        save_total_limit=config["save_total_limit"],
        logging_steps=nbr_logging_steps,  # TFboard logging steps
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=nbr_logging_steps,
        prediction_loss_only=False,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        dataloader_pin_memory=True,
        dataloader_num_workers=4 * nbr_gpu,
        fp16=False,
        log_level="error",
        optim="adamw_torch",
        label_names=["masked_lm_labels"],
    )

    preprocess_logits_for_metrics_func = preprocess_logits_for_mlm_metrics_builder(
        field_names=field_names, vocab=vocab, device=device
    )

    num_training_steps = int(
        num_epochs * len(train_dataset) / (batch_size * grad_acc_steps)
    )
    optimizer = AdamW(tab_net.model.parameters(), lr=lr_max)

    poly_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warm_up_steps,
        num_training_steps=num_training_steps,
        lr_end=0,
        power=2,
    )

    #############################################################################################
    # Launch pre-training

    trainer = Trainer(
        model=tab_net.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, poly_scheduler),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics_func,
    )

    trainer.add_callback(FileLoggingCallback(f"{output_path}/callback_log.json"))

    if config["resume_from_checkpoint"]:
        trainer.train(resume_from_checkpoint=ckpt_dir)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
