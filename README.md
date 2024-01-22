# TabFormerLite: More Efficient Tabular Transformers

In this repository, we introduce TabFormerLite, a fork of IBM's [TabFormer library](https://github.com/IBM/TabFormer).
The original TabFormer library implements a transformer-based model for tabular time-series data, described in the paper [Tabular Transformers for modeling multivariate time series](https://arxiv.org/abs/2011.01843).

In TabFormerLite, we address some of the limitations encountered in the original TabFormer library. Our changes have resulted in an updated model that is lighter, more flexible, and faster to train.

The key modifications we have introduced are detailed below.
1. **Adaptive embedding size:** In the original library, the embedding size could not be adjusted by the user and was unnecessarily growing with the dataset's dimensionality. To rectify this, we modified the architecture to enable users to adjust embedding size independently of the number of input features. This change helps to reduce the number of model parameters and overall model size, resulting in leaner models and faster training times without affecting model performance.
2. **Efficient data processing**: We have implemented an efficient data processing step that significantly reduces memory and time resources required during data pre-processing, substantially enhancing the overall efficiency of the library.
3. **Extended Functionality**: We implemented new functionalities such as model fine-tuning and embedding extraction from pre-trained models, improving the usefulness of the library in downstream tasks. These functionalities were not available in the original library.
4. **Dependency Resolution**: We resolved code dependencies that relied on outdated library versions.

We use [HuggingFace's](https://huggingface.co/) Trainer to facilitate the training process for both pre-training and fine-tuning of the TabFormerLite model.

## Available functionalities

TabFormerLite can be used to accomplish the following tasks:
1. Pre-processing input data to be compatible with the model.
2. Pre-training the model through masked language modeling (MLM) task.
3. Extracting embeddings from pre-trained models using various pooling strategies accross both the layer and time dimensions. These embeddings can be used as input to simpler machine-learning models to perform downstream tasks.
4. Fine-tuning the model on regression and classification tasks.

## Requirements

TabFormerLite requires the following packages to be installed:

* Python (3.10)
* Pytorch (2.1.0)
* HuggingFace/Transformers (4.34.1)
* Accelerate (0.24.0)
* Numpy (1.26.1)
* Pandas (1.4.2)
* Scikit-learn (1.1.2)
* Pynvml (11.4.1)
* Loguru (0.6)
* Chardet (5.2.0)
* Tqdm (4.64.0)
* H5py (3.7.0)

(X.Y.Z) represents the versions on which we tested the code.

### Installation

The required packages can be installed with pip:
```
$ pip install -r requirements.txt
```

Or with Poetry:
```
$ poetry install
```

## 1. Data pre-processing

Running the following command line will pre-process your data to be suitable for the model.

```
$ python3 scripts/encode_dataset.py -cfg --path-to-config-files
```

An example of a configuration file is available in: `./configs/example/data_encoding/` and can be used like this:

```
$ python3 scripts/encode_dataset.py -cfg ./configs/example/data_encoding/config_card_dataset_encoding.json
```

## 2. Pre-training

Once the data pre-processing step is complete, use the following command to pre-train the model.
```
$ python3 scripts/run_mlm_pretraining.py -cfg --path-to-config-files
```

An example of a configuration file is available in: `./configs/example/pre-training/` and can be used like this:

```
$ python3 scripts/run_mlm_pretraining.py -cfg ./configs/example/pretraining/config_card_dataset_size_300.json
```

## 3. Embedding extraction from pre-trained models

Running the following command line extracts embeddings from a pre-trained model.

```command line
$ python3 scripts/inference_main.py -cfg --path-to-config-files
```
An example of a configuration file is available in the folder: `./configs/example/inference/` and can be used like this:
e.g.
```command line
$ python3 scripts/inference_main.py -cfg configs/example/inference/config_card_dataset_inference.json
```

## 4. Fine-tuning

Running the following command line will fine-tune the model on a downstream task.

```command line
$ python3 scripts/finetuning.py -cfg --path-to-config-files
```

An example of a configuration file is available in the folder: `./configs/example/finetuning/` and can be used like this:
e.g.
```command line
$ python3 scripts/finetuning.py -cfg configs/example/finetuning/config_card_finetuning.json
```

## Citation

@inproceedings{padhi2021tabular,
  title={Tabular transformers for modeling multivariate time series},
  author={Padhi, Inkit and Schiff, Yair and Melnyk, Igor and Rigotti, Mattia and Mroueh, Youssef and Dognin, Pierre and Ross, Jerret and Nair, Ravi and Altman, Erik},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3565--3569},
  year={2021},
  organization={IEEE},
  url={https://ieeexplore.ieee.org/document/9414142}
}
