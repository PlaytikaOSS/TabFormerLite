from transformers import BertTokenizer
from transformers.modeling_utils import PreTrainedModel

from tabformerlite.models.hierarchical import TabFormerEmbeddings
from tabformerlite.models.tabformer_bert import (
    TabFormerBertConfig,
    TabFormerBertForMaskedLM,
    TabFormerBertForSequenceClassification,
)


class TabFormerHierarchicalLM(PreTrainedModel):
    """Redefine class from HuggingFace for use in this package for MLM."""

    base_model_prefix = "bert"

    def __init__(self, config, vocab):
        super().__init__(config)

        self.config = config

        self.tab_embeddings = TabFormerEmbeddings(self.config)
        self.tb_model = TabFormerBertForMaskedLM(self.config, vocab)

    def forward(self, input_ids, **input_args):
        """Forward pass of the model."""
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.tb_model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerHierarchicalClassification(PreTrainedModel):
    """Redefine class from HuggingFace for use in this package for classification."""

    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.tab_embeddings = TabFormerEmbeddings(self.config)
        self.tb_model = TabFormerBertForSequenceClassification(self.config)

    def forward(self, input_ids, **input_args):
        """Forward pass of the model."""
        inputs_embeds = self.tab_embeddings(input_ids)
        return self.tb_model(inputs_embeds=inputs_embeds, **input_args)


class TabFormerBertLM:
    """Main class of this package for MLM models."""

    def __init__(
        self,
        special_tokens,
        vocab,
        ncols=None,
        field_hidden_size=64,
        tab_embeddings_num_attention_heads=8,  # Field transformer
        tab_embedding_num_encoder_layers=1,  # Field transformer
        tab_embedding_dropout=0.1,
        num_attention_heads=12,  # Bert
        num_hidden_layers=12,  # Bert
        hidden_size=768,
        mlm_average_loss=False,
    ):
        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename

        assert (
            field_hidden_size % tab_embeddings_num_attention_heads == 0
        ), '"field_hidden_size" must be divisible by "tab_embeddings_num_attention_heads"'

        assert (
            hidden_size % num_attention_heads == 0
        ), '"hidden_size" must be divisible by "num_attention_heads"'

        self.config = TabFormerBertConfig(
            vocab_size=len(self.vocab),
            ncols=self.ncols,
            hidden_size=hidden_size,
            field_hidden_size=field_hidden_size,
            tab_embeddings_num_attention_heads=tab_embeddings_num_attention_heads,
            tab_embedding_num_encoder_layers=tab_embedding_num_encoder_layers,
            tab_embedding_dropout=tab_embedding_dropout,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
        )

        self.tokenizer = BertTokenizer(
            vocab_file, do_lower_case=False, **special_tokens
        )

        self.config.mlm_average_loss = mlm_average_loss
        self.model = TabFormerHierarchicalLM(self.config, self.vocab)


class TabFormerBertClassification:
    """Main class of this package for classification models."""

    def __init__(
        self,
        special_tokens,
        vocab,
        problem_type,
        ncols=None,
        field_hidden_size=64,
        tab_embeddings_num_attention_heads=8,
        tab_embedding_num_encoder_layers=1,
        tab_embedding_dropout=0.1,
        num_attention_heads=12,
        num_hidden_layers=12,
        hidden_size=768,
        num_labels=1,
        pooled_output_index=0,
        load_weights_from_pretraining=True,  # pylint: disable=unused-argument
        pos_weight=None,
    ):
        self.ncols = ncols
        self.vocab = vocab
        vocab_file = self.vocab.filename

        assert (
            field_hidden_size % tab_embeddings_num_attention_heads == 0
        ), '"field_hidden_size" must be divisible by "tab_embeddings_num_attention_heads"'

        assert (
            hidden_size % num_attention_heads == 0
        ), '"hidden_size" must be divisible by "num_attention_heads"'

        self.config = TabFormerBertConfig(
            vocab_size=len(self.vocab),
            ncols=self.ncols,
            hidden_size=hidden_size,
            field_hidden_size=field_hidden_size,
            tab_embeddings_num_attention_heads=tab_embeddings_num_attention_heads,
            tab_embedding_num_encoder_layers=tab_embedding_num_encoder_layers,
            tab_embedding_dropout=tab_embedding_dropout,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
        )

        self.tokenizer = BertTokenizer(
            vocab_file, do_lower_case=False, **special_tokens
        )

        self.config.num_labels = num_labels
        self.config.pooled_output_index = pooled_output_index
        self.config.problem_type = problem_type
        self.config.pos_weight = pos_weight

        self.model = TabFormerHierarchicalClassification(self.config)
