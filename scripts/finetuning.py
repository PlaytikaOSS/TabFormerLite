import json
import os
from datetime import datetime
from pprint import pprint

import torch
from loguru import logger
from sklearn.metrics import confusion_matrix
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    get_polynomial_decay_schedule_with_warmup,
    set_seed,
)

from tabformerlite.dataset.datacollator import TabDataCollatorForClassification
from tabformerlite.inference.encode_data import EncoderForPlayerData
from tabformerlite.inference.load_data import load_data_for_encoding
from tabformerlite.inference.load_model import TabNetLoader
from tabformerlite.misc.get_configuration import (
    load_json,
    parse_command_line_args,
    save_json,
)
from tabformerlite.misc.training_helpers import (
    FileLoggingCallback,
    compute_metrics_classification_builder,
)
from tabformerlite.models.modules import TabFormerBertClassification


def main():
    """Main function to execute from command line.

    $ python3 scripts/finetuning.py -cfg --path-to-config-file

    Example usage
    -------------
    >>> python3 scripts/finetuning.py -cfg configs/example/ \
        finetuning/config_card_finetuning.json
    """

    # Collect experiment information
    path_config = parse_command_line_args()
    finetuning_config = load_json(f"{path_config}")

    # Set random seed in `random`, `numpy` and `torch`
    set_seed(finetuning_config["training_config"]["seed"])

    # Define the label of the minority class for which to optimize the F1-score
    pos_label = finetuning_config["training_config"]["pos_label"]

    # Count available GPUs
    if torch.cuda.is_available():
        nbr_gpu = torch.cuda.device_count()
    else:
        nbr_gpu = 1

    # Define path for storing fine-tuning outputs
    output_dir = finetuning_config["training_config"]["output_dir"]
    output_folder = finetuning_config["training_config"]["output_folder"]

    # Add date suffix to output filenames
    if finetuning_config["training_config"]["add_date_suffix_to_output"]:
        suffix = datetime.now().strftime("%d_%m_%Y_%H%M%S")
        output_path = os.path.join(output_dir, output_folder + f"_{suffix}")
    else:
        output_path = os.path.join(output_dir, output_folder)
    output_path = os.path.abspath(output_path)

    # Create directory in specified path
    os.makedirs(output_path, mode=0o777, exist_ok=True)
    logger.info(f"Output path: {output_path}\n")

    # Export finetuning_config file
    with open(f"{output_path}/config.json", "w", encoding="utf8") as fh:
        json.dump(finetuning_config, fh)

    # Collect info about pre-trained model
    logger.info("Collecting information about pre-trained TabFormerLite model")
    model_path = os.path.join(
        finetuning_config["pretrained_model_config"]["model_directory"],
        finetuning_config["pretrained_model_config"]["model_name"],
    )
    model_path = os.path.abspath(model_path)
    model_config = load_json(f"{model_path}/config.json")

    # Collect info about pretraining data
    logger.info("Collecting information about pre-training data\n")
    pretraining_data_config = load_json(
        os.path.abspath(
            finetuning_config["pretraining_data_config"]["path_to_config_file"]
        )
    )

    ####################################################################
    # Load & encode inference data
    # --- load data
    data_dict = load_data_for_encoding(
        finetuning_config["finetuning_data_config"],
        pretraining_data_config,
    )

    # --- encode data with TabDataset class
    encoder = EncoderForPlayerData(pretraining_data_config)

    logger.info("Creating TabDataset for the training data - START")
    train_dataset = encoder.encode(data_dict["df_train"])
    logger.info("Creating TabDataset for the training data - END\n")

    logger.info("Creating TabDataset for the validation data - START")
    valid_dataset = encoder.encode(data_dict["df_valid"])
    logger.info("Creating TabDataset for the validation data - END\n")

    logger.info("Creating TabDataset for the test data - START")
    test_dataset = encoder.encode(data_dict["df_test"])
    logger.info("Creating TabDataset for the test data - END\n")

    # Finetuning requires the labels
    train_dataset.return_labels = True
    valid_dataset.return_labels = True
    test_dataset.return_labels = True

    # --- collect vocab, special tokens and ncols in a dictionary
    dataset_config = encoder.load_vocab_and_special_tokens(train_dataset)

    ####################################################################
    # Load model using TabFormerBertClassification class
    model_class = TabFormerBertClassification
    loader = TabNetLoader(
        dataset_config, model_config, finetuning_config, model_path, model_class
    )
    # Adapt model_config to fine-tuning task
    # -- Add problem_type
    problem_type = finetuning_config["training_config"]["problem_type"]

    loader.model_config["problem_type"] = problem_type
    # -- Add option to load weights from pretrained model (default: True)
    loader.model_config["load_weights_from_pretraining"] = finetuning_config[
        "training_config"
    ].get("load_weights_from_pretraining", True)

    # -- Remove MLM loss
    del loader.model_config["mlm_average_loss"]

    # -- Compute pos_weight (for classification tasks only)
    if problem_type == "classification":
        if finetuning_config["training_config"]["compute_pos_weight"]:
            # Load pos_weight from config file
            assert (
                "pos_weight" in finetuning_config["training_config"]
            ), "pos_weight not found in config file\n"
            pos_weight = finetuning_config["training_config"]["pos_weight"]
            logger.info(f"pos_weight: {pos_weight}\n")
        else:
            pos_weight = None

        # Update model_config with pos_weight
        loader.model_config["pos_weight"] = pos_weight

    # Load model and weights from pretraining (optional)
    tabnet_model = loader.from_pretrained()

    ####################################################################
    # Training parameters
    data_collator = TabDataCollatorForClassification(
        tokenizer=tabnet_model.tokenizer, mlm=False
    )

    # Set batch size
    batch_size = finetuning_config["training_config"]["batch_size"]
    grad_acc_steps = int(
        finetuning_config["training_config"]["grad_acc_steps"] / nbr_gpu
    )  # default = 1

    # Get number of steps per epoch
    steps_per_epoch = int(len(train_dataset) / batch_size / grad_acc_steps)

    # Log N times per epoch
    logging_per_epoch = finetuning_config["training_config"]["logs_per_epoch"]
    nbr_logging_steps = int(steps_per_epoch / logging_per_epoch)
    # print(logging_per_epoch, steps_per_epoch)

    # Epochs
    num_epochs = finetuning_config["training_config"]["num_epochs"]

    # Create checkpoint every N epochs
    ckpt_every_n = finetuning_config["training_config"]["checkpoint_every_N_epochs"]
    checkpoint_steps = int(steps_per_epoch * ckpt_every_n / nbr_gpu)

    # Maximum learning rate
    lr_max = finetuning_config["training_config"]["lr_max"]

    # Warm-up steps
    warm_up_n = finetuning_config["training_config"]["warmup_steps_in_epochs"]
    warm_up_steps = int(steps_per_epoch * warm_up_n)

    # Metric used for early stopping
    early_stopping_metric = finetuning_config["training_config"].get(
        "metric_for_best_model", "loss"
    )

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

    json_string = json.dumps(training_args_user)

    with open(f"{output_path}/training_args.json", "w", encoding="utf8") as json_file:
        json_file.write(json_string)

    logger.info("Training arguments and model parameters:\n")
    pprint(training_args_user[0], indent=4)
    print()

    ##########################################################################
    # Create an instance of the TrainingArguments class
    training_args = TrainingArguments(
        output_dir=f"{output_path}",  # output directory
        logging_dir=f"{output_path}/logging/",  # TensorBoard log directory
        num_train_epochs=num_epochs,  # number of training epochs
        learning_rate=lr_max,  # max learning rate
        warmup_steps=warm_up_steps,  # nbr of steps used for linear warmup
        save_steps=checkpoint_steps,  # nbr of steps for creating checkpts
        save_total_limit=finetuning_config["training_config"]["save_tot_lim"],
        logging_steps=nbr_logging_steps,  # TFboard logging steps
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=nbr_logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model=early_stopping_metric,  # Metric for EarlyStopping
        greater_is_better=True,
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
        label_names=["labels"]
        # https://discuss.huggingface.co/t/huggingface-transformers-longformer-optimizer-warning-adamw/14711
    )

    num_training_steps = int(
        num_epochs * len(train_dataset) / (batch_size * grad_acc_steps)
    )
    optimizer = torch.optim.AdamW(tabnet_model.model.parameters(), lr=lr_max)

    poly_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warm_up_steps,
        num_training_steps=num_training_steps,
        lr_end=0,
        power=2,
    )

    ####################################
    # START TRAINING
    logger.info("Model training\n")

    if problem_type == "classification":
        compute_metrics_fct = compute_metrics_classification_builder(pos_label)
    else:
        raise NotImplementedError(
            "Only classification tasks are supported at the moment"
        )

    # Use the EarlyStoppingCallback
    patience = finetuning_config["training_config"].get("early_stopping_patience", 3)
    threshold = finetuning_config["training_config"].get(
        "early_stopping_threshold", 0.05
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=patience, early_stopping_threshold=threshold
    )

    # Create an instance of the Trainer class
    trainer = Trainer(
        model=tabnet_model.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        optimizers=(optimizer, poly_scheduler),
        compute_metrics=compute_metrics_fct,
    )
    # Add callbacks for logging and early stopping
    trainer.add_callback(FileLoggingCallback(f"{output_path}/callback_log.json"))
    trainer.add_callback(early_stopping)

    # Start training
    trainer.train()
    print()

    ####################################
    # Inference
    logger.info("Inference\n")

    # Predictions
    # Outputs are tuple with: 'predictions', 'label_ids', "metrics"
    # "Predictions": logits
    train_outputs = trainer.predict(train_dataset)
    valid_outputs = trainer.predict(valid_dataset)
    test_outputs = trainer.predict(test_dataset)
    print()

    ######################################################

    # Compute the confusion matrix
    logger.info("Confusion matrix\n")
    predicted_classes = (train_outputs[0] > 0.0).astype(int)
    print(predicted_classes)
    print(train_outputs[1])
    cm = confusion_matrix(train_outputs[1], predicted_classes)
    print(f"Train data: \n{cm}")

    # Compute the confusion matrix
    predicted_classes = (test_outputs[0] > 0.0).astype(int)
    cm = confusion_matrix(test_outputs[1], predicted_classes)
    print(f"Test data: \n{cm}")

    # Collect metrics for all datasets
    metrics = []
    metrics.append(
        {"train": train_outputs[2], "valid": valid_outputs[2], "test": test_outputs[2]}
    )

    # Export metrics to json
    save_json(
        metrics,
        f"{output_path}/results.json",
    )
    print()


if __name__ == "__main__":
    main()
