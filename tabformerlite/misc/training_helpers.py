import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import TrainerCallback


def get_model_size(model):
    """
    Helper function to compute the size of a PyTorch model.
    Input:
        model
    Output:
        size in MB

    Source: https://discuss.pytorch.org/t/finding-model-size/130275
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def preprocess_logits_for_mlm_metrics_builder(field_names, vocab, device):
    """
    Create a function to convert logits to predicted labels before computing
    MLM metrics during training with the field names specific to the dataset.
    """

    def preprocess_logits_for_metrics(logits, labels):
        """
        Custom function to convert logits to predicted labels before computing
        MLM metrics during training.

        This function ensures that, for a given column, only tokens belonging
        to this column are predicted.

        INPUTS:

        - logits: tensor of shape: (bzs, seq_len x num_cols, voc_len)
        - labels: tensor, optional

        OUTPUTS:

        - preds_batch: tensor of shape: (bzs, seq_len x num_cols) with
                       the predicted labels of the batch
        """

        # Convert logits to predictions

        # y_dim: Size of the 2-nd dimension in logits (= seq_len x n_cols)
        y_dim = logits.shape[1]

        # First create a tensor to hold batch predictions (predicted labels)
        # At the beginning, this is a tensor of dummy -200s of shape:
        # (bzs, seq_len x n_cols). The values in this tensor will be eventually
        # replaced by the correct batch labels.
        preds_batch = -200 * torch.ones(size=(logits.shape[0], y_dim)).long().to(device)

        # field_names: col names
        for field_idx, field_name in enumerate(field_names):
            # Get indexes corresponding to a column
            col_ids = list(range(field_idx, y_dim, len(field_names)))

            # Get vocabulary ids (global) corresponding to a column
            global_ids_field = vocab.get_field_ids(field_name)

            # Get the logits corresponding to a given column and vocab positions
            # bzs, col_ids (=seq_len), len(global_ids_field)
            logits_field = logits[:, col_ids, :][:, :, global_ids_field]

            # Get predictions for a given column -> project to global (vocab) ids (otherwise local)
            predictions_field = (
                torch.argmax(logits_field, dim=-1).to(device) + global_ids_field[0]
            )

            # Update preds_batch with predicted labels for a given column
            preds_batch[:, col_ids] = predictions_field

        # Also return the CE loss.
        cross_entropy_loss = compute_ce_loss(labels, logits, len(vocab))

        return cross_entropy_loss, preds_batch

    return preprocess_logits_for_metrics


def compute_ce_loss(labels, logits, vocab_len):
    """
    This method is called if self.vocab_len is not None to compute
    the Cross Entropy loss during training.
    """

    # Compute CrossEntropyLoss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, vocab_len), labels.view(-1))

    return loss


def compute_metrics(eval_preds):
    """
    Custom function to compute MLM metrics on validation data during
    pre-training

    Notes:
    1. "Macro" averaging computes the average value of a metric across all
       samples without considering the number of samples in each class. This
       gives equal weight to all classes, no matter their size.
    2. Recall can be ill-defined when some classes have no true labels (i.e.,
       samples). This can happen since the recall is computed on the validation
       data, where some classes may not have samples. Using "zero_division=0"
       helps to mitigate this issue.
    3. Precision can be ill-defined when some classes have no predictions. For
       example, this can happen in imbalanced datasets of small sizes,
       especially in the minority classes. Using "zero_division=0" helps to
       mitigate this issue.
    """

    preds_tuple, labels = eval_preds
    (ce_losses, preds) = preds_tuple

    # Average the Cross Entropy loss across batches
    cross_entropy_loss = ce_losses.mean().item()

    # y_dim: Size of the 2-nd dimension in predictions (= seq_len x n_cols)
    y_dim = preds.shape[1]

    labels = labels.reshape(-1, y_dim)  # flattening

    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    # Compute MLM metrics
    acc = accuracy_score(labels, preds)

    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)

    return {
        "cross_entropy_loss": cross_entropy_loss,
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "recall": recall,
        "precision": precision,
    }


def compute_metrics_classification_builder(pos_label):
    """
    Create a function to compute classification metrics during
    fine-tuning using the specified pos_label.
    """

    def compute_metrics_classification(eval_preds):
        """
        Custom function to compute classification metrics on
        validation data during fine-tuning.

        Args:
        ----
        - eval_preds: EvalPrediction object containing predicted
        logits and labels

        Returns:
        -------
        - Dictionary containing the computed classification metrics
        """

        labels = eval_preds.label_ids
        logits = eval_preds.predictions

        # Compute probabilities
        assert logits.shape[1] == 1, "Only binary classification is supported"
        probs = torch.sigmoid(torch.as_tensor(logits))

        # Compute class labels
        predictions = (probs.cpu().numpy() > 0.5).astype(int)

        # Optimize probability threshold
        _, optimized_threshold = optimize_threshold_classification(
            probs, labels, pos_label=pos_label
        )
        predictions_optimized = (probs.cpu().numpy() > optimized_threshold).astype(int)

        return {
            "f1_score_macro": f1_score(
                labels, predictions, pos_label=pos_label, average="macro"
            ),
            "f1_score_minority": f1_score(labels, predictions, pos_label=pos_label),
            "f1_score_minority_optimized": f1_score(
                labels, predictions_optimized, pos_label=pos_label
            ),
            "optimized_threshold": optimized_threshold,
            "precision_minority": precision_score(
                labels, predictions, pos_label=pos_label, zero_division=0
            ),
            "recall_minority": recall_score(
                labels, predictions, pos_label=pos_label, zero_division=0
            ),
            "accuracy_score": accuracy_score(labels, predictions),
        }

    return compute_metrics_classification


def optimize_threshold_classification(y_probs, y_true, pos_label):
    """
    Computes the optimized threshold to obtain the
    best F1-score according to the minority class.

    :param y_probs: raw probabilities
    :param y_true: true label
    :param pos_label: which is the minority class
    :return: a tuple containing the predictions and the best threshold.
    """

    f1 = []
    thresholds = np.arange(0, 1, 0.05)
    for t in thresholds:
        y_pred = y_probs > t
        f1.append(f1_score(y_true, y_pred, pos_label=pos_label))

    # calculate predictions with the threshold that maximize the f1 score
    y_pred = y_probs > thresholds[f1.index(max(f1))]

    return y_pred, thresholds[f1.index(max(f1))]


class FileLoggingCallback(TrainerCallback):
    """
    Callback for the HuggingFace Trainer to save
    the logs to a file after each evaluation step.
    """

    def __init__(self, file_save_path):
        self.file_save_path = file_save_path

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            with open(self.file_save_path, "wt", encoding="utf8") as fh:
                json.dump(state.log_history, fh)


class CustomSchedulerCallback(TrainerCallback):
    """
    A callback that enables the use of a LR scheduler that uses the
    train/eval metrics as input.
    """

    def __init__(self, lr_scheduler, metric_name):
        self.lr_scheduler = lr_scheduler
        self.metric_name = metric_name

    # Necessary to be able to continue the training from a checkpoint
    # Source: https://github.com/pytorch/pytorch/issues/80809
    # def on_train_begin(self, args, state, control, logs=None, **kwargs):
    #    if state.is_local_process_zero:
    #        for p_group in self.lr_scheduler.optimizer.param_groups:
    #            p_group['capturable'] = True

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            metric = None
            if len(state.log_history) > 0:
                metric = state.log_history[-1].get(self.metric_name, None)
            # The global step is extracted from the training state
            # to be able to continue a previous train.
            self.lr_scheduler.custom_step(metric, epoch=state.global_step)
