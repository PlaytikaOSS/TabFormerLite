from torch import nn


class TabFormerConcatEmbeddings(nn.Module):
    """TabFormerConcatEmbeddings: Embeds tabular data of categorical variables

    Notes: - All column entries must be integer indices in a vocabolary that is common across columns
           - `sparse=True` in `nn.Embedding` speeds up gradient computation for large vocabs

    Args:
        config.ncols
        config.vocab_size
        config.hidden_size

    Inputs:
        - **input_ids** (batch, seq_len, ncols): tensor of batch of sequences of rows

    Outputs:
        - **output'**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.field_hidden_size,
            padding_idx=getattr(config, "pad_token_id", 0),
            sparse=False,
        )
        self.lin_proj = nn.Linear(
            config.field_hidden_size * config.ncols, config.hidden_size
        )

        self.hidden_size = config.hidden_size
        self.field_hidden_size = config.field_hidden_size

    def forward(self, input_ids):
        """
        Forward pass of the module.
        """
        input_shape = input_ids.size()

        embeds_sz = list(input_shape[:-1]) + [input_shape[-1] * self.field_hidden_size]
        inputs_embeds = self.lin_proj(self.word_embeddings(input_ids).view(embeds_sz))

        return inputs_embeds


class TabFormerEmbeddings(nn.Module):
    """TabFormerEmbeddings: Embeds tabular data of categorical variables

    Notes: - All column entries must be integer indices in a vocabolary that is common across columns

    Args:
        config.ncols
        config.num_layers (int): Number of transformer layers
        config.vocab_size
        config.hidden_size
        config.field_hidden_size

    Inputs:
        - **input** (batch, seq_len, ncols): tensor of batch of sequences of rows

    Outputs:
        - **output**: (batch, seq_len, hidden_size): tensor of embedded rows
    """

    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.field_hidden_size,
            padding_idx=getattr(config, "pad_token_id", 0),
            sparse=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.field_hidden_size,
            nhead=config.tab_embeddings_num_attention_heads,
            dim_feedforward=config.field_hidden_size,
            dropout=config.tab_embedding_dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.tab_embedding_num_encoder_layers
        )

        self.lin_proj = nn.Linear(
            config.field_hidden_size * config.ncols, config.hidden_size
        )

    def forward(self, input_ids):
        """
        Forward pass of the module.
        """
        inputs_embeds = self.word_embeddings(input_ids)
        embeds_shape = list(inputs_embeds.size())

        # [batch_size, seq_len, ncols, field_hidden_size] -> [batch_size*seq_len, ncols, field_hidden_size]
        inputs_embeds = inputs_embeds.view([-1] + embeds_shape[-2:])
        # [batch_size*seq_len, ncols, field_hidden_size] -> same
        inputs_embeds = self.transformer_encoder(inputs_embeds)
        # [batch_size*seq_len, ncols, field_hidden_size] -> [batch_size, seq_len, ncols*field_hidden_size]
        # This operation aligns all the columns embedding in one "line" in their correct order.
        inputs_embeds = inputs_embeds.contiguous().view(embeds_shape[0:2] + [-1])

        inputs_embeds = self.lin_proj(inputs_embeds)

        return inputs_embeds
