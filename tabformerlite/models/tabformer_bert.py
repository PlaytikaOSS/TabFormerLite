# A lot of the code in this file is originally from the HuggingFace implementation.
# pylint: skip-file
import torch
from loguru import logger
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput

from tabformerlite.models.custom_criterion import CustomAdaptiveLogSoftmax


class TabFormerBertConfig(BertConfig):
    def __init__(
        self,
        ncols=12,
        vocab_size=30522,
        field_hidden_size=64,
        tab_embeddings_num_attention_heads=12,
        tab_embedding_num_encoder_layers=1,
        tab_embedding_dropout=0.1,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        pad_token_id=0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.ncols = ncols
        self.field_hidden_size = field_hidden_size
        self.tab_embedding_num_encoder_layers = (
            tab_embedding_num_encoder_layers  # Field Transformer - nbr layers
        )
        self.tab_embeddings_num_attention_heads = (
            tab_embeddings_num_attention_heads  # Field Transformer - nbr heads
        )
        self.tab_embedding_dropout = tab_embedding_dropout
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads  # BERT - nbr heads
        self.num_hidden_layers = num_hidden_layers  # BERT - nbr layers


class TabFormerBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.field_hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class TabFormerBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = TabFormerBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # The first dimension here must be the hidden_size used for the BERT
        # model. Apparently the self.init_weights() in TabFormerBertForMaskedLM
        # requires the decoder to have this first dimension, rewriting it if necessary.

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        """
        Need a link between the two variables so that the bias is correctly
        resized with `resize_token_embeddings`
        """
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class TabFormerBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = TabFormerBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class TabFormerBertForMaskedLM(BertForMaskedLM):
    """
    This module contains a Bert model followed by a custom MLM head.
    Most of the code is originally from the HuggingFace implementation.
    """

    def __init__(self, config, vocab):
        super().__init__(config)

        self.vocab = vocab
        # This reshapes the hidden size into something that is compatible with the "MLM" that follows
        # and remove the forced link with the number of columns.
        self.reshape_linear = nn.Linear(
            config.hidden_size, config.ncols * config.field_hidden_size
        )
        self.cls = TabFormerBertOnlyMLMHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, ncols*field_hidden_size]
        sequence_output = self.reshape_linear(sequence_output)
        output_sz = list(sequence_output.size())
        expected_sz = [output_sz[0], output_sz[1] * self.config.ncols, -1]
        # [batch_size, seq_len, ncols*field_hidden_size] -> [batch_size, seq_len*ncols, field_hidden_size]
        sequence_output = sequence_output.view(expected_sz)
        masked_lm_labels = masked_lm_labels.view(expected_sz[0], -1)

        # [batch_size, seq_len*ncols, field_hidden_size] -> [batch_size, seq_len*ncols, vocab_sz]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]

        # prediction_scores : [batch_size, seq_len*ncols, vocab_sz]
        # masked_lm_labels  : [batch_size, seq_len*ncols]

        total_masked_lm_loss = 0

        seq_len = prediction_scores.size(1)
        field_names = self.vocab.get_field_keys(
            remove_target=True, ignore_special=False
        )
        for field_idx, field_name in enumerate(field_names):
            col_ids = list(range(field_idx, seq_len, len(field_names)))

            global_ids_field = self.vocab.get_field_ids(field_name)

            prediction_scores_field = prediction_scores[:, col_ids, :][
                :, :, global_ids_field
            ]  # batch_size * 10 * K
            masked_lm_labels_field = masked_lm_labels[:, col_ids]
            masked_lm_labels_field_local = self.vocab.get_from_global_ids(
                global_ids=masked_lm_labels_field, what_to_get="local_ids"
            )

            nfeas = len(global_ids_field)
            loss_fct = self.get_criterion(field_name, nfeas, prediction_scores.device)

            masked_lm_loss_field = loss_fct(
                prediction_scores_field.view(-1, len(global_ids_field)),
                masked_lm_labels_field_local.view(-1),
            )

            # Due to some change in the Pytorch implementation of CrossEntropyLoss
            # if the target consist only of ignored_index, it now returns NaN
            # instead of 0. This is to circumvent this behavior.
            if not masked_lm_loss_field.isnan():
                total_masked_lm_loss += masked_lm_loss_field

        # Normalize loss by number of columns if specified.
        if self.config.mlm_average_loss:
            if len(field_names) > 0:
                total_masked_lm_loss /= len(field_names)

        return (total_masked_lm_loss,) + outputs

    def get_criterion(self, fname, vs, device, cutoffs=False, div_value=4.0):
        if fname in self.vocab.adap_sm_cols:
            if not cutoffs:
                cutoffs = [int(vs / 15), 3 * int(vs / 15), 6 * int(vs / 15)]

            criteria = CustomAdaptiveLogSoftmax(
                in_features=vs, n_classes=vs, cutoffs=cutoffs, div_value=div_value
            )

            return criteria.to(device)
        else:
            return CrossEntropyLoss()


class TabFormerBertForSequenceClassification(BertForSequenceClassification):
    """
    This module contains a Bert model followed by a custom classification head.
    Most of the code is originally from the HuggingFace implementation.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Binary classification task
        if self.config.problem_type == "classification":
            assert (
                self.num_labels == 1
            ), "num_labels must be 1 for binary classification"

            pos_weight = self.config.pos_weight

            if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight)

                logger.info(f"Using pos_weight: {pos_weight}\n")

                self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Regression task
        elif self.config.problem_type == "regression":
            self.loss_fct = nn.MSELoss()
        else:
            raise ValueError(
                f"Unknown problem type: {self.config.problem_type}. "
                "Supported problem types are: classification, regression."
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Only Binary classification and regression tasks
        are supported. In both cases, num_labels is 1 (default).

        Notes
        -----
        In BertForSequenceClassification:
        - The final hidden state is further processed by a
        pooler, composed of a linear layer and a hyperbolic
        tangent activation function, before being fed to the
        classifier.
        - self.dropout: Dropout(p=0.1, inplace=False)
        - BertPooler(
        (dense): Linear(in_features=300, out_features=300, bias=True)
        (activation): Tanh()
        )
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Bert Outputs
        # ------------
        # outputs[0]: hidden_states from last BERT layer
        # outputs[1]: pooler_output -> [batch_size, hidden_size]

        last_hidden_state = outputs[0]  # [batch_size, seq_len, hidden_size]

        # Change here to select which sequence token to use
        # -- default: first token in sequence
        pooled_output = last_hidden_state[:, self.config.pooled_output_index, :]

        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Compute logits
        logits = self.classifier(pooled_output)

        # Compute batch loss
        if self.config.problem_type == "classification":
            loss = self.loss_fct(logits.squeeze(), labels.float())
        elif self.config.problem_type == "regression":
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
