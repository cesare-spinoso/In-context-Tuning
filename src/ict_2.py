import os
import pickle as pkl
import random
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
from tqdm.notebook import tqdm, trange
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

from custom_types import (
    ModelInput,
    Task2TemplateExamples,
    Task2Preds,
    Task2Scores,
    Task2Templates,
    Task2Verbalizers,
    TrainingBatch,
    TrainingExample,
)
from data_loader import DataLoader
from verbalized_model import VerbalizedModel
import wandb

class ICT:
    """
    Class implementing the In-Context Tuning Fine-Tuning paradigm developed by Chen et al. (2021).
    Fixes made to code on github.
    """

    def __init__(
        self,
        model_name: str,
        task_format: Literal["clm", "mlm"],
        device: str,
        identifier: str,
        load_model_path: Union[str, Path] = None,
    ) -> None:
        """
        Args:
                model_name (str): Name of the model you want to fine-tune e.g. "gpt2", "bert-base-uncased".
                task_format: Whether the task format is a masked language modeling task or a causal language modeling task.
                This depends on what the model you are fine-tuning is.
                device (str): Which device to use for training. Recommended to use "cuda" if available.
                identifier (str): Model name identifier. Should be unique.
                load_model_path (Union[str, Path], optional): Path to previously trained model. Defaults to None.
        """
        assert task_format in ["clm", "mlm"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        if "gpt2" in self.model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Note that this means the padding will begin from the left!
            self.tokenizer.padding_side = "left"
        self.task_format = task_format
        self.model = VerbalizedModel(
            model_name=model_name, task_format=task_format, tokenizer=self.tokenizer
        )
        self.device = device
        self.identifier = identifier
        if load_model_path is not None:
            self.model.load_state_dict(torch.load(load_model_path, map_location="cpu"))
        self.model.to(device)

    def meta_train(
        self,
        task2template_examples: Task2TemplateExamples,
        task2verbalizers: Task2Verbalizers,
        num_demonstrations: int,
        example_delimiter: str,
        allow_label_overlap: bool,
        lr: float,
        num_warmup_steps: int,
        num_epochs: int,
        bsz: int,
        output_dir: Union[str, Path],
    ) -> None:
        """Run the meta-training described in the paper."""
        wandb.init(
            project="ict",
            entity="cesare_spinoso",
            config={
                "model_name": self.model_name,
                "k": num_demonstrations,
                "lr": lr,
                "epochs": num_epochs,
            },
        )
        # Verify that passed arguments are valid
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # Create dataloader
        data_loader = DataLoader(
            tokenizer=self.tokenizer,
            task_format=self.task_format,
            task2verbalizers=task2verbalizers,
            example_delimiter=example_delimiter,
            device=self.device,
        )
        # Training setup
        trainable_parameters = []
        for param in self.model.named_parameters():
            assert param[1].requires_grad  # fine-tune all LM parameters
            trainable_parameters.append(param[1])
        # Initialize AdamW optimizer
        optimizer = torch.optim.AdamW(params=trainable_parameters, lr=lr)
        optimizer.zero_grad()
        # Compute the total number of training steps (i.e. number of times call .step)
        # to calibrate the learning rate scheduler (which uses a linear schedule with warmup)
        # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
        task_num_examples = [
            len(task2template_examples[task]) for task in task2template_examples
        ]
        num_steps = sum(task_num_examples) * num_epochs // bsz
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_steps
        )
        # Tries to automatically set the type of floating point precision
        scaler = torch.cuda.amp.GradScaler()  # fp16 training
        # Prepare the training examples
        epoch_train_examples: list[TrainingBatch] = []  # List of training batches
        for task in tqdm(task2template_examples, desc="Preparing training examples."):
            # Loop by task so batches only contain examples of the same task
            training_examples: list[TrainingExample] = []
            examples = task2template_examples[task]
            for query_example in examples:
                # Remove the random selection of templates
                input_text = data_loader.prepare_input(
                    task,
                    query_example,
                    examples,
                    num_demonstrations,
                    allow_label_overlap,
                    template=None,
                )
                # NOTE: Current implementation filters out completely the
                # examples that exceed model's maximum token length, will
                # probably need to change this so that we truncate rather than
                # filter out the example entirely
                if (
                    len(self.tokenizer(input_text)["input_ids"])
                    <= self.tokenizer.model_max_length
                ):
                    training_examples.append(
                        TrainingExample(task, input_text, query_example["<label>"])
                    )
            # Divide model_examples into batches, so that each batch has all examples from the same task
            # Shuffle within batch examples
            random.shuffle(training_examples)
            for idx in range(0, len(training_examples), bsz):
                epoch_train_examples.append(training_examples[idx : idx + bsz])
        # Begin training
        self.model.train()
        for _ in trange(num_epochs, desc="Epoch training."):
            # Shuffle the batches (so that e.g. don't see all the same task in a row)
            random.shuffle(epoch_train_examples)
            # Language model fine-tuning
            for batch_example_idx in trange(
                len(epoch_train_examples), desc="Batch training."
            ):
                optimizer.zero_grad()
                batch_train_examples = epoch_train_examples[batch_example_idx]
                # NOTE: Interesting technical problem is the truncation length
                input_dict = self.tokenizer(
                    [example.model_input for example in batch_train_examples],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                labels = [example.label for example in batch_train_examples]
                task = batch_train_examples[0].task
                with torch.cuda.amp.autocast():
                    # Loss for k-way classification
                    loss, _ = self.model.forward(
                        input_dict,
                        torch.LongTensor(data_loader.task2verbalizer_worids[task]).to(
                            self.device
                        ),
                        torch.LongTensor(labels).to(self.device),
                    )
                wandb.log({"loss": loss.item()})
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

        if output_dir is not None:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pkl"))

    def meta_test(
        self,
        task2template_examples: Task2TemplateExamples,
        task2verbalizers: Task2Verbalizers,
        num_demonstrations: int,
        example_delimiter: str,
        allow_label_overlap: bool,
        num_prefix_selections: int,
        bsz: int,
    ) -> tuple[Task2Preds, Task2Scores]:
        """Run the meta-testing described in the paper. Note that num_prefix_selections
        is the number of few-shot choices to evaluate. It is much smaller than the total
        number of few-shot choices due to the computational cost of evaluating each choice.
        See the "Sampling" subsection in 3.2 of the paper."""
        # Create dataloader
        data_loader = DataLoader(
            tokenizer=self.tokenizer,
            task_format=self.task_format,
            task2verbalizers=task2verbalizers,
            example_delimiter=example_delimiter,
            device=self.device,
        )
        # Record the predictions (logits for each task) and the scores
        task2preds, task2scores = {}, {}
        for task in tqdm(task2template_examples, desc="Meta-testing for each task."):
            # Prepare the input texts and labels for the task
            examples = task2template_examples[task]
            input_texts: list[ModelInput] = []
            labels: list[int] = []
            current_number_prefix_selections = num_prefix_selections
            if num_prefix_selections > len(examples):
                current_number_prefix_selections = len(examples)
            current_number_demonstrations = (
                len(examples)
                if num_demonstrations > len(examples)
                else num_demonstrations
            )
            for query_example in examples:
                for _ in range(current_number_prefix_selections):
                    input_text = data_loader.prepare_input(
                        task,
                        query_example,
                        examples,
                        current_number_demonstrations,
                        allow_label_overlap,
                        template=None,
                    )
                    input_texts.append(input_text)
                    labels.append(query_example["<label>"])
            # Predict on the input, in batches
            self.model.eval()
            output_logits = []
            for example_idx in tqdm(
                np.arange(0, len(input_texts), bsz), desc="Predicting."
            ):
                input_dict = self.tokenizer(
                    input_texts[example_idx : example_idx + bsz],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        batch_output_logits = self.model.forward(
                            input_dict,
                            torch.LongTensor(
                                data_loader.task2verbalizer_worids[task]
                            ).to(self.device),
                        )
                output_logits += [
                    logits.cpu().numpy() for logits in batch_output_logits
                ]
            task2preds[task] = output_logits
            # Compute score based on logits
            # Compute mean reciprocal rank (MRR), precision@1, and precision@10
            predicted_logits = []  # Use this to calculate the AUC
            precision1, precision, mrr = [], [], []
            for logits, gt_label in tqdm(zip(output_logits, labels), "Evaluating."):
                # Find the rank of the ground-truth label
                # w.r.t. all other labels based on logits
                # Best rank is 1, worst rank is num_labels
                predicted_logits.append(logits[gt_label])
                rank = np.count_nonzero(logits > logits[gt_label]) + 1
                assert rank >= 1
                mrr.append(1 / rank)
                precision1.append(1 if rank <= 1 else 0)
                precision.append(1 if rank <= 10 else 0)
            task2scores[task] = {
                "precision1": np.mean(
                    precision1
                ),  # This becomes the average accuracy for binary classification
                "precision10": np.mean(precision),
                "mrr": np.mean(mrr),
            }
        return task2preds, task2scores
