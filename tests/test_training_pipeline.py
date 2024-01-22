import os


def test_training_pipeline():
    """
    Test the three main functions of the library:
    - encode the training data
    - train the MLM model
    - infer the responses on the test data
    """

    # Encode the training data
    os.system(
        "python ../scripts/encode_dataset.py -cfg ./test_configs/test_dataset_encoding.json"
    )

    # Launch the pretraining script
    os.system(
        "python ../scripts/run_mlm_pretraining.py -cfg ./test_configs/test_pretraining_config.json"
    )

    # Launch the script for data inference
    os.system(
        "poetry run python ../scripts/inference_main.py -cfg ./test_configs/test_inference_config.json"
    )
