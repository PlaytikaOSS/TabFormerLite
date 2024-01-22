import os

from loguru import logger
from transformers import set_seed

from tabformerlite.inference.encode_data import EncoderForPlayerData
from tabformerlite.inference.export_downstream_data import export_data
from tabformerlite.inference.extract_embeddings import EmbeddingsExtractor
from tabformerlite.inference.load_data import load_data_for_encoding
from tabformerlite.inference.load_model import TabNetLoader
from tabformerlite.misc.get_configuration import load_json, parse_command_line_args


def main():
    """Main function to execute from command line.

    $ python3 scripts/inference_main.py -cfg --path-to-config-files

    Example usage
    -------------
    >>> python3 scripts/inference_main.py -cfg \
        configs/example/inference/config_card_dataset_inference.json
    """

    # Collect experiment information
    path_config = parse_command_line_args()
    exp_config = load_json(f"{path_config}")

    # Set random seed in `random`, `numpy` and `torch`
    set_seed(exp_config["seed"])

    # Collect info about pre-trained model
    logger.info("Collecting information about pre-trained Tabformer model")
    model_path = os.path.join(
        exp_config["pretrained_model_config"]["model_directory"],
        exp_config["pretrained_model_config"]["model_name"],
    )
    model_path = os.path.abspath(model_path)

    model_config = load_json(f"{model_path}/config.json")

    # Collect info about pretraining data
    logger.info("Collecting information about pre-training data\n")
    pretraining_data_config = load_json(
        exp_config["pretraining_data_config"]["path_to_config_file"]
    )

    # Load & encode inference data
    # --- load data
    data_dict = load_data_for_encoding(
        exp_config["inference_data_config"], pretraining_data_config
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

    # --- collect vocab, special tokens and ncols in a dictionary
    dataset_config = encoder.load_vocab_and_special_tokens(train_dataset)

    # Load pretrained model
    loader = TabNetLoader(dataset_config, model_config, exp_config, model_path)
    tabnet_model = loader.from_pretrained()  # tabnet_model is a class

    # Extract embeddings
    logger.info("Extracting embeddings - START")
    extractor = EmbeddingsExtractor(tabnet_model, exp_config["inference_config"])

    # --- training data
    logger.info("Extracting embeddings for the training data - START")
    outputs_train = extractor.extract_embeddings(train_dataset)
    logger.info(f"Embeddings for training data shape: {len(outputs_train[2])}")
    logger.info("Extracting embeddings for the training data - END\n")

    # --- validation data
    logger.info("Extracting embeddings for the validation data - START")
    outputs_valid = extractor.extract_embeddings(valid_dataset)
    logger.info(f"Embeddings for validation data shape: {len(outputs_valid[2])}")
    logger.info("Extracting embeddings for the validation data - END\n")

    # --- test data
    logger.info("Extracting embeddings for the test data - START")
    outputs_test = extractor.extract_embeddings(test_dataset)
    logger.info(f"Embeddings for test data shape: {len(outputs_test[2])}")
    logger.info("Extracting embeddings for the test data - END\n")

    # --- extract usecase_filename
    usecase_filename = extractor.retrieve_usecase_filename(loader.ckpt_dir)
    logger.info("Extracting embeddings - END\n")

    # Export embeddings with labels for downstream tasks

    # --- Create experiment_folder_path
    output_dir = exp_config["downstream_task_config"]["output_dir"]
    model_dir_name = exp_config["pretrained_model_config"]["model_name"]
    output_path = os.path.join(output_dir, model_dir_name, usecase_filename)
    output_path = os.path.abspath(output_path)

    # If unavailable, create directory in specified experiment_path
    os.makedirs(
        os.path.abspath(os.path.join(output_dir, model_dir_name)),
        mode=0o777,
        exist_ok=True,
    )

    # --- Export user_id, date, embeddings and labels
    export_data(
        exp_config,
        pretraining_data_config,
        output_path,
        outputs_train,
        outputs_valid,
        outputs_test,
    )


if __name__ == "__main__":
    main()
