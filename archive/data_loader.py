import random
import warnings
from typing import Literal, Union

from transformers import PreTrainedTokenizer

from custom_types import (
    Example,
    TemplateExample,
    ModelInput,
    Task,
    Task2Verbalizers,
    Template,
    Verbalizer,
)


class DataLoader:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        task_format: Literal["mlm", "clm"],
        task2verbalizers: Task2Verbalizers,
        example_delimiter: str,
        device: str,
    ):
        assert task_format in ["mlm", "clm"]

        self.tokenizer = tokenizer
        self.task_format = task_format
        self.task2verbalizers = task2verbalizers
        self.example_delimiter = example_delimiter
        self.device = device

        # Get the token ids of each verbalizer for each task
        # This ensures that the indexing within the verbalized model
        # output is correct
        self.task2verbalizer_worids: dict[Task, list[int]] = {}
        for task in task2verbalizers:
            verbalizer_wordids = []
            for verbalizer in task2verbalizers[task]:
                wordids = self.tokenizer(verbalizer, add_special_tokens=False)[
                    "input_ids"
                ]
                assert (
                    len(wordids) == 1
                )  # current code assumes that each verbalizer is one token.
                verbalizer_wordids.append(wordids[0])
            self.task2verbalizer_worids[task] = verbalizer_wordids

    def _check_input_same(
        self,
        example1: Union[Example, TemplateExample],
        example2: Union[Example, TemplateExample],
    ) -> bool:
        """Checks that two examples have the same value for each of its keys.
        Skips this check for the <label> key."""
        assert (
            example1.keys() == example2.keys()
        ), "Cannot compare examples with different keys."
        for key in example1:
            if key != "<label>" and example1[key] != example2[key]:
                return False
        return True

    def _sample_demonstrations(
        self,
        query_example: Union[Example, TemplateExample],
        support_examples: Union[list[Example], list[TemplateExample]],
        num_demonstrations: int,
        allow_label_overlap: bool,
    ) -> list[int]:
        """Randomly samples num_demonstrations from support_examples and returns them
        as a list of indices into the support_examples list. The selected examples
        will never have the same input as the query_example. If allow_label_overlap is
        True then the selected examples can have the same label as the query_example.
        Otherwise, the selected examples will have a different label than the query_example."""
        assert isinstance(allow_label_overlap, bool)
        # List of example indices to sample from for demonstration examples
        selectable_example_idx = []
        if allow_label_overlap:
            # Only keep examples with a different input than the query example
            selectable_example_idx = [
                example_idx
                for example_idx, example in enumerate(support_examples)
                if not self._check_input_same(example, query_example)
            ]
        else:
            # Only keep examples with a different input and label than the query example
            selectable_example_idx = [
                example_idx
                for example_idx, example in enumerate(support_examples)
                if not self._check_input_same(example, query_example)
                and example["<label>"] != query_example["<label>"]
            ]
        assert len(selectable_example_idx) < len(support_examples)
        # Randomly sample demonstration examples
        # If num_demonstrations = 5 then this is 5-shot classification
        effective_num_demonstrations = (
            len(selectable_example_idx)
            if len(selectable_example_idx) < num_demonstrations
            else num_demonstrations
        )
        prefix_example_idxs = random.sample(selectable_example_idx, effective_num_demonstrations)
        return prefix_example_idxs

    def _encode_example_with_template(
        self, template: Template, example: Example, verbalizers: list[Verbalizer]
    ) -> tuple[str, str]:
        """Replace the <input> and <label> keys in the template with their values from
        the example. Two replacement versions are returned: one with the <label> replacement
        replaced with the mask token (for MLM) or removed (for CLM), the other with the
        <label> replaced with its corresponding value from the verbalizer."""
        templated_example = template
        for key in example:
            if key != "<label>":  # all input keys
                templated_example = templated_example.replace(key, example[key])
        if self.task_format == "mlm":
            templated_example_no_label = templated_example.replace(
                "<label>", self.tokenizer.mask_token
            )
        elif self.task_format == "clm":
            assert template.endswith(
                "<label>"
            ), "Template for CLM decoding must have labels at the end of the prompt."
            templated_example_no_label = templated_example.replace("<label>", "")
        templated_example_with_label = templated_example.replace(
            "<label>", verbalizers[example["<label>"]]
        )
        return templated_example_no_label, templated_example_with_label

    def _encode_input_str(
        self,
        prefix_examples: Union[list[Example], list[TemplateExample]],
        query_example: Union[Example, TemplateExample],
        template: Template,
        verbalizers: list[Verbalizer],
    ) -> str:
        """Assemble the string that will be passed to the model. This string will be the
        concatenation of the task instructions, prefix examples, and the query example.
        The way this is done is actually slightly different to how it's illustrated in
        Figure 1."""
        # Create the model input string by fully replacing the prefix examples
        # and replacing the <label> in the query example either with the mask
        # token (for MLM) or removing it (for CLM).
        input_texts = []
        for example in prefix_examples:
            if template is not None:
                _, templated_example_with_label = self._encode_example_with_template(
                    template, example, verbalizers
                )
            else:
                # Then this is a TemplateExample
                extracted_example = {
                    "<input>": example["<input>"],
                    "<label>": example["<label>"],
                }
                extracted_template = example["template"]
                _, templated_example_with_label = self._encode_example_with_template(
                    extracted_template, extracted_example, verbalizers
                )
            input_texts.append(templated_example_with_label)
        if template is None:
            extracted_query_example = {
                "<input>": query_example["<input>"],
                "<label>": query_example["<label>"],
            }
            extracted_template = query_example["template"]
            query_example_masked, _ = self._encode_example_with_template(
                extracted_template, extracted_query_example, verbalizers
            )
        else:
            query_example_masked, _ = self._encode_example_with_template(
                template, query_example, verbalizers
            )
        # Convert to string and then to model's token ids
        input_text = self.example_delimiter.join(input_texts + [query_example_masked])
        input_ids = self.tokenizer.encode(input_text)
        # # Warn the user
        # if (len_input_ids := len(input_ids)) > (
        #     model_max_length := self.tokenizer.model_max_length
        # ):
        #     warnings.warn(
        #         f"MODEL LENGTH EXCEEDED. Length of input text is {len_input_ids}. "
        #         + f"This exceeds the model's max length of {model_max_length}."
        #     )
        return input_text

    def prepare_input(
        self,
        task: Task,
        query_example: Union[Example, TemplateExample],
        support_examples: Union[list[Example], list[TemplateExample]],
        num_demonstrations: int,
        allow_label_overlap: bool,
        template: Template = None,
    ) -> ModelInput:
        """Sample the prefix examples (i.e. the few shot examples) and then assemble
        them into a string that will be passed to the model."""
        if len(support_examples) < num_demonstrations:
            # Minus 1 because don't include the query itself
            # NOTE: I'm assuming that this is ok because even if there are 4
            # examples in the support set, can find 4 different variations of
            # cycling through them
            num_demonstrations = len(support_examples) - 1
        prefix_example_idxs = self._sample_demonstrations(
            query_example, support_examples, num_demonstrations, allow_label_overlap
        )
        prefix_examples = [support_examples[idx] for idx in prefix_example_idxs]
        input_text = self._encode_input_str(
            prefix_examples, query_example, template, self.task2verbalizers[task]
        )
        return input_text
