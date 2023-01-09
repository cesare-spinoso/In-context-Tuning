from typing import Literal, Union
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, PreTrainedTokenizer


class VerbalizedModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        task_format: Literal["mlm", "clm"],
        tokenizer: PreTrainedTokenizer,
    ) -> None:
        super(VerbalizedModel, self).__init__()
        assert task_format in ["mlm", "clm"]
        if task_format == "mlm":
            self.lm_model = AutoModelForMaskedLM.from_pretrained(model_name)
        elif task_format == "clm":
            self.lm_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.task_format = task_format
        self.loss_fct = nn.CrossEntropyLoss()  # classification loss
        self.tokenizer = tokenizer

    def forward(
        self, input_dict: dict, verbalizer_word_ids: list[int], labels: list[int] = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        :param input_dict: {'input_ids': ..., 'attention_mask': ..., 'token_type_ids': ...} expected
        to be in a batch format. Expect each key to point to a torch tensor of shape (batch_size, seq_len)
        padded for the longest sequence (this is regardless of whether you use "mlm" or "clm").
        :param verbalizer_word_ids: list of integers (length equal to the number of classification classes),
        where the i-th number is the word id of the i-th verbalizer (index 0).
        :param labels: example labels as a list of token ids where the i-th token corresponds to the answer to
        the i-th input in the batch.

        :return: if labels is None, then return the logits for each example in the batch,
        otherwise return the logits and the loss for each example in the batch. The logits
        will only be for the tokens with ids in verbalizer_word_ids.
        """
        if self.task_format == "mlm":
            # Pass input to language model
            output = self.lm_model(**input_dict)
            # For MLM the output logits will be of shape (batch_size, seq_len, vocab_size)
            output_logits = output.logits
            # Check that there is one unique [MASK] token in each example
            # Locate the position of the [MASK] token in each example
            mask_pos = []
            for input_ids in input_dict["input_ids"]:
                assert (
                    sum(
                        [
                            input_id == self.tokenizer.mask_token_id
                            for input_id in input_ids
                        ]
                    )
                    == 1
                ), "There should be a unique [MASK] token in each example."
                mask_pos.append(input_ids.tolist().index(self.tokenizer.mask_token_id))
            # Get the output logits for the [MASK] token
            logits_verbalizers = []
            # Do this for each example in the batch
            for example_idx in range(len(output_logits)):
                logits_verbalizers.append(
                    torch.index_select(
                        output_logits[example_idx][mask_pos[example_idx]],
                        dim=0,
                        index=verbalizer_word_ids,
                    )
                )
            # This will have shape (batch_size, len(verbalizer_word_ids))
            output_logits = torch.vstack(logits_verbalizers)
        elif self.task_format == "clm":
            # Pass input to the language model
            output = self.model(**input_dict)
            # For CLM the output logits will be of shape (batch_size, seq_len, vocab_size)
            # seq_len will be the same as the input which is why the last token is selected
            output_logits = output.logits[:, -1, verbalizer_word_ids]
        else:
            raise NotImplementedError('self.task_format must be either "clm" or "mlm".')

        if labels is None:
            return output_logits  # (batch size, len(verbalizer_word_ids)), both mlm and clm
        else:
            loss = [
                self.loss_fct(
                    output_logits[example_idx].unsqueeze(dim=0),
                    labels[example_idx : example_idx + 1],
                )
                for example_idx in range(len(output_logits))
            ]
            loss = torch.mean(torch.hstack(loss))
            return loss, output_logits
