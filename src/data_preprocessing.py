import random
import warnings
from typing import Literal, Union

from transformers import AutoTokenizer

from custom_types import (
    Prompt,
    Task,
    Example,
    ClassLabel,
    Task2Examples,
    Task2Prompts,
)


class ICTPreprocessor:
    """Data preprocessing class for the ICT dataset. This class is used
    to prepare the input to be fed into the ICTDataset class."""

    def __init__(
        self,
        k: int,
        model_name: str,
        task_format: Literal["mlm", "clm"],
        delimiter: str,
    ):
        assert k > 0 and isinstance(k, int)
        assert task_format in ["mlm", "clm"]
        assert isinstance(delimiter, str)

        self.k = k
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if task_format == "clm":
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Note that this means the padding will begin from the left!
            self.tokenizer.padding_side = "left"
        self.task_format = task_format
        self.delimiter = delimiter

    def get_tokenizer(self) -> AutoTokenizer:
        return self.tokenizer

    def get_class_label_token_ids(self, class_labels: list[ClassLabel]) -> list[int]:
        # Get the token ids of each class label
        # This ensures that the indexing within the model
        # output is correct
        class_label_token_ids = []
        for class_label in class_labels:
            token_id = self.tokenizer(class_label, add_special_tokens=False)[
                "input_ids"
            ]
            assert len(token_id) == 1
            class_label_token_ids.append(token_id[0])
        return class_label_token_ids

    def get_fold_data(
        self, task2examples: Task2Examples, task_folds: list[Task]
    ) -> Task2Examples:
        """Returns the examples for the given fold."""
        return {task: task2examples[task] for task in task_folds}

    def _sample_demonstrations(
        self,
        query_example: Example,
        support_examples: list[Example],
        num_demonstrations: int,
        allow_label_overlap: bool,
    ) -> list[int]:
        """Randomly samples num_demonstrations from support_examples and returns them
        as a list of indices into the support_examples list. The selected examples
        will never have the same input as the query_example because assume that it has been removed
        from support_examples. If allow_label_overlap is True then the selected examples
        can have the same label as the query_example.
        Otherwise, the selected examples will have a different label than the query_example."""
        assert isinstance(allow_label_overlap, bool)
        # List of example indices to sample from for demonstration examples
        selectable_example_idx = []
        if allow_label_overlap:
            # Only keep examples with a different input than the query example
            selectable_example_idx = list(range(len(support_examples)))
        else:
            # Only keep examples with a different input and label than the query example
            selectable_example_idx = [
                example_idx
                for example_idx, example in enumerate(support_examples)
                if example["<label>"] != query_example["<label>"]
            ]
        assert len(selectable_example_idx) <= len(support_examples)
        # Randomly sample demonstration examples
        # If num_demonstrations = 5 then this is 5-shot classification
        effective_num_demonstrations = (
            len(selectable_example_idx)
            if len(selectable_example_idx) < num_demonstrations
            else num_demonstrations
        )
        prefix_example_idxs = random.sample(
            selectable_example_idx, effective_num_demonstrations
        )
        return prefix_example_idxs

    def _replace_template_with_example(
        self,
        example: Example,
        replace_label: bool = False,
        class_labels: list[ClassLabel] = None,
    ) -> tuple[str, str]:
        """Replace the <input> and <label> keys in the template with their values from
        the example. Two replacement versions are returned: one with the <label> replacement
        replaced with the mask token (for MLM) or removed (for CLM), the other with the
        <label> replaced with its corresponding value from the verbalizer."""
        replaced_example = ""
        template = example["template"]
        replaced_example = template.replace("<input>", example["<input>"])
        if replace_label:
            assert (
                class_labels is not None
            ), "Must provide class labels to replace label."
            replaced_example = replaced_example.replace(
                "<label>", class_labels[example["<label>"]]
            )
        else:
            if self.task_format == "mlm":
                replaced_example = replaced_example.replace(
                    "<label>", self.tokenizer.mask_token
                )
            elif self.task_format == "clm":
                assert template.endswith(
                    "<label>"
                ), "Template for CLM decoding must have labels at the end of the prompt."
                replaced_example = replaced_example.replace("<label>", "")
        return replaced_example

    def _merge_examples_into_prompt(
        self,
        prefix_examples: list[Example],
        query_example: Example,
        class_labels: list[ClassLabel],
    ) -> str:
        """Assemble the string that will be passed to the model. This string will be the
        concatenation of the task instructions, prefix examples, and the query example.
        The way this is done is actually slightly different to how it's illustrated in
        Figure 1."""
        # Create the model input string by fully replacing the prefix examples
        # and replacing the <label> in the query example either with the mask
        # token (for MLM) or removing it (for CLM).
        replaced_prefix_templates = []
        for example in prefix_examples:
            replaced_template = self._replace_template_with_example(
                example, replace_label=True, class_labels=class_labels
            )
            replaced_prefix_templates.append(replaced_template)
        query_template_replaced = self._replace_template_with_example(
            query_example, replace_label=False
        )
        prompt = self.delimiter.join(
            replaced_prefix_templates + [query_template_replaced]
        )
        return prompt

    def _make_prompt(
        self,
        query_example: Example,
        support_examples: list[Example],
        allow_label_overlap: bool,
    ) -> str:
        """Sample the prefix examples (i.e. the few shot examples) and then assemble
        them into a string that will be passed to the model."""
        prefix_example_idxs = self._sample_demonstrations(
            query_example,
            support_examples,
            num_demonstrations=self.k,
            allow_label_overlap=allow_label_overlap,
        )
        prefix_examples = [support_examples[idx] for idx in prefix_example_idxs]
        prompt = self._merge_examples_into_prompt(
            prefix_examples, query_example, class_labels=self.class_labels
        )
        return prompt

    def _convert_examples_to_prompts(
        self,
        examples: list[Example],
        allow_label_overlap: bool,
    ) -> list[Prompt]:  # not sure what type this should be
        prompts = []
        for i, example in enumerate(examples):
            query_example = example
            support_examples = examples[:i] + examples[i + 1 :]
            prompt = self._make_prompt(
                query_example, support_examples, allow_label_overlap
            )
            prompts.append({"prompt": prompt, "label": example["<label>"]})
        return prompts

    def convert_examples_to_prompts(
        self, task2examples: Task2Examples, allow_label_overlap: bool
    ) -> Task2Prompts:  # not sure what the exact type should be
        task2prompts = {}
        for task in task2examples:
            task2prompts[task] = self._convert_examples_to_prompts(
                task2examples[task], allow_label_overlap
            )
        return task2prompts


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
task_folds = [
    "group0_2016SemEval6TweetEvalStanceHillary",
    "group2_KaggleCovidTweetSentiment",
]
task_format = "clm"
k = 2
allow_label_overlap = True
class_labels = ["no", "yes"]
delimiter = " "

pickled_data: Task2Examples = {
    "group0_2016SemEval6TweetEvalStanceHillary": [
        {
            "<input>": "Does #hillaryclinton lie and engage in cover-ups to do damage to the country or does she do it out of habit?  #tcot #uniteblue #SemST ",
            "template": "<input> Does the tweet take an opposing stance on Hillary? <label>",
            "<label>": 1,
        },
        {
            "<input>": "#Democrats and #Obama reduce OUR Army by 40,000   MEANWHILE   Has increased IRS to 100,000  #HillaryClinton #Hillary #SemST ",
            "template": "<input> Is there a supporting stance taken on Hillary in the tweet? <label>",
            "<label>": 1,
        },
        {
            "<input>": "@user I won't use the c-word, so I'll just say that everything Clinton does is a \"cunning stunt\". #HillaryonCNN #SemST ",
            "template": "<input> Does the tweet take a supporting stance on Hillary? <label>",
            "<label>": 1,
        },
        {
            "<input>": "@user made me proud today!  Nothing like reinforcing what I already knew!  #SemST ",
            "template": "<input> Does the tweet take a stance against Hillary? <label>",
            "<label>": 1,
        },
        {
            "<input>": "@user condemns anti-Israel boycotts as 'counter-productive' #SemST ",
            "template": "<input> Does the tweet take a stance in favor of Hillary? <label>",
            "<label>": 0,
        },
    ],
    "group2_KaggleCovidTweetSentiment": [
        {
            "<input>": "Avoid the panic and empty shelves at the stores and stock up now with Amazon Pantry items! \r\r\nhttps://t.co/1gWRwNuc0x   \r\r\n\r\r\n#ad #coronavirus #prepper #BePrepared \r\r\n#Corona #CoronavirusOutbreak #Soldout\r\r\n#Coronavid19  #wuhanvirus #CoronavirusUSA \r\r\n#COVID19 #WuhanCoronavirus #toiletpaper",
            "template": "<input> Does this tweet have a positive sentiment? <label>",
            "<label>": 1,
        },
        {
            "<input>": "With businesses closing, workers are going to be suffering big because of this pandemic. We NEED a stay on debt collection to prevent people from being thrown out of their homes and being financially ruined. \r\r\n\r\r\nRETWEET AND SIGN!!!\r\r\n\r\r\n#Coronadebtrelief\r\r\n\r\r\nhttps://t.co/Vn3l74Z1Qp",
            "template": "<input> Is the user feeling positive about the situation? <label>",
            "<label>": 1,
        },
        {
            "<input>": "italy has more cases of the covid 19 than china but everyone stock piled on pasta and no ones ordered chinese food. i smell racism",
            "template": "<input> Is the user feeling negative about the situation? <label>",
            "<label>": 0,
        },
        {
            "<input>": "Please RT Check on your elderly friends amp neighbors many of them are on a fixed income and can t afford to stock up on food tp They can t have groceries delivered but should be avoiding crowds Reach out amp offer a hand",
            "template": "<input> Does this tweet have a negative sentiment? <label>",
            "<label>": 1,
        },
    ],
}

ict_preprocessor = ICTPreprocessor(
    k=k,
    task_folds=task_folds,
    tokenizer=tokenizer,
    delimiter=delimiter,
)

fold_data = ict_preprocessor.get_fold_data(pickled_data)

class_label_token_ids = ict_preprocessor.get_class_label_token_ids()

task2prompts = ict_preprocessor.convert_examples_to_prompts(
    task2examples=fold_data, allow_label_overlap=allow_label_overlap
)
