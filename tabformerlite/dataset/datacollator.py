# A lot of the code in this file is originally from the IBM TabFormer implementation.
# pylint: skip-file
from typing import Dict, List, Tuple, Union

import torch
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import _torch_collate_batch


class TabDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    This class is a custom-made DataCollator for the MLM task of TabFormer.
    If the "mlm" flag is not set at the creation of the class, it will return the
    unmasked labels as well as the user_ids and dates of the data.

    IMPORTANT: In the dataset used, the first two columns (features) should be, in order,
    the user_dim_col and the date_dim_col, as specified in the config of the encoding.
    The TabDataset class automatically reorder the columns to make sure of this.
    """

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch_all = _torch_collate_batch(
            examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )

        # Remove userid and date
        batch = batch_all[:, :, 2:].clone()
        # Keep them separately
        user_ids = batch_all[:, 0, 0]
        date = batch_all[:, -1, 1]  # Return last day of the window

        sz = batch.shape

        # Only return the user_id and date if we are not doing MLM.
        if self.mlm:
            batch = batch.view(sz[0], -1)
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs.view(sz), "masked_lm_labels": labels.view(sz)}

        else:
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                "input_ids": batch,
                "labels": labels,
                "user_ids": user_ids,
                "date": date,
            }

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling:
        - 80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is"
                " necessary for masked language modeling."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training
        # (with probability args.mlm_probability
        # defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with
        # tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input
        # tokens unchanged
        return inputs, labels


class TabDataCollatorForClassification(TabDataCollatorForLanguageModeling):
    """
    This class inherits TabDataCollatorForLanguageModeling and adapts it for use in classification.
    Regression can also be done, if the convert_labels_to_long is set to False.

    IMPORTANT: In the dataset used, the first two columns (features) should be,
    in order, the user_dim_col and the date_dim_col, as specified in the config
    of the encoding. The TabDataset class automatically reorder the columns to
    make sure of this.
    """

    return_id_and_date: bool = False
    convert_labels_to_long: bool = True

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [x[0] for x in examples]

        if self.convert_labels_to_long:
            labels = torch.Tensor([int(x[1][0]) for x in examples]).long()
        else:
            labels = torch.Tensor([float(x[1][0]) for x in examples])

        batch_all = _torch_collate_batch(
            input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )

        # Remove userid and date
        batch = batch_all[:, :, 2:].clone()
        # Keep them separately
        user_ids = batch_all[:, 0, 0]
        date = batch_all[:, -1, 1]  # Return last day of the window

        if self.return_id_and_date:
            return {
                "input_ids": batch,
                "labels": labels,
                "user_ids": user_ids,
                "date": date,
            }
        else:
            return {"input_ids": batch, "labels": labels}
