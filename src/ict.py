import os
from pathlib import Path
import pickle as pkl
import random
from typing import Literal, NamedTuple, TypedDict, Union

import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from data_loader import DataLoader
from verbalized_model import VerbalizedModel

Task = str
"""Name of the task. For LAMA, tasks are given special codenames like P530."""

ModelInput = str
"""String that will be passed to the model. Consists of K examples with their
answers and a query example with its answer masked out. For LAMA, this looks
like: 'France has diplomatic relations with Germany. Canada works with the
U.S. The U.K. is allied with [MASK].'"""

class TrainingExample(NamedTuple):
	"""A single training example that will be passed to the language model.
	Consists of the task name, the model input and the label."""
	task: Task
	model_input: ModelInput
	label: int

TrainingBatch = list[TrainingExample]
"""A single training batch should contain training examples all from the same task."""

class Example(TypedDict):
	"""Single example for a task, consists of input and corresponding label.
	e.g. For LAMA, the input is the subject and the label is the object."""
	"<input>": str
	"""The input to the task e.g. For LAMA, the name of a country."""
	"<label>": str
	"""The label of the task for the given input.
	e.g. For LAMA, the name of a country that maintains diplomatic relations with the input country."""

Task2Examples = dict[Task, list[Example]]
"""List of all the examples (input, label pairs) for a given task.
e.g. For LAMA, the task might be the relations between two countries
and the examples might be pairs of countries."""

Template = str
"""Template for the task, used to fill in the input and label for a given example.
e.g. For LAMA, a template might be <input> maintains diplomatic relations with <label>."""

Task2Templates = dict[Task, list[Template]]
"""List of all the templates for a given task.
e.g. For LAMA, the task might be the diplomatic relations and the templates
might be <input> maintains diplomatic relations with <label> as well as other
variations."""

Verbalizer = str
"""There is a one-to-one mapping between verbalizers and class labels.
For LAMA, because each task is a 21K-way classification task, there are
21K verbalizers."""

Task2Verbalizers = dict[Task, list[Verbalizer]]
"""List of all the verbalizers for a given task.
e.g. For LAMA, each task will have the same list of verbalizers."""

# TODO
Task2Preds = None
Task2Scores = None

class ICT:
	"""
	Class implementing the In-Context Tuning Fine-Tuning paradigm developed by Chen et al. (2021).
	"""

	def __init__(
		self,
		model_name: str,
		task_format: Literal["clm", "mlm"],
		device: str,
		load_model_path: Union[str, Path] = None,
	) -> None:
		"""
		Args:
			model_name (str): Name of the model you want to fine-tune e.g. "gpt2", "bert-base-uncased".
			task_format: Whether the task format is a masked language modeling task or a causal language modeling task.
			This depends on what the model you are fine-tuning is.
			device (str): Which device to use for training. Recommended to use "cuda" if available.
			load_model_path (Union[str, Path], optional): Path to previously trained model. Defaults to None.
		"""
		assert task_format in ["clm", "mlm"]

		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		if "gpt2" in model_name:
			self.tokenizer.pad_token = self.tokenizer.eos_token
			# Note that this means the padding will begin from the left!
			self.tokenizer.padding_side = "left"
		self.task_format = task_format

		self.model = VerbalizedModel(
			model_name=model_name, task_format=task_format, tokenizer=self.tokenizer
		)
		self.device = device

		if load_model_path is not None:
			self.model.load_state_dict(torch.load(load_model_path, map_location="cpu"))
		self.model.to(device)

	def meta_train(
		self,
		task2examples: Task2Examples,
		task2templates: Task2Templates,
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
		# Verify that passed arguments are valid
		assert task2examples.keys() == task2templates.keys()
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
		task_num_examples = [len(task2examples[task]) for task in task2examples]
		num_steps = sum(task_num_examples) * num_epochs // bsz
		lr_scheduler = get_linear_schedule_with_warmup(
			optimizer, num_warmup_steps, num_steps
		)
		# Tries to automatically set the type of floating point precision
		scaler = torch.cuda.amp.GradScaler()  # fp16 training
		# Begin training
		self.model.train()
		for _ in trange(num_epochs, desc="Epoch training."):
			# Prepare the training examples
			epoch_train_examples: list[TrainingBatch] = []
			for task in task2examples:
				training_examples: list[TrainingExample] = []
				examples = task2examples[task]
				for query_example in examples:
					# TODO: Why is the template chosen randomly?
					# Associate each example with a randomly sampled template
					template = random.sample(task2templates[task], 1)[0]
					input_text = data_loader.prepare_input(
						task,
						query_example,
						examples,
						num_demonstrations,
						template,
						allow_label_overlap,
					)
					# TODO: Current implementation filters out completely the
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
			# Shuffle the batches (so that e.g. don't see all the same task in a row)
			random.shuffle(epoch_train_examples)
			train_loss = []
			# Language model fine-tuning
			for batch_example_idx in trange(len(epoch_train_examples), desc="Batch training."):
				optimizer.zero_grad()
				batch_train_examples = epoch_train_examples[batch_example_idx]
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
				train_loss.append(loss.item())
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				lr_scheduler.step()

			if output_dir is not None:
				with open(os.path.join(output_dir, "train.log"), "a") as f:
					f.write(f"Epoch epoch_idx - train loss: {np.average(train_loss):.4f}\n")
					f.flush()

		if output_dir is not None:
			torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pkl"))

	def meta_test(
		self,
		task2examples: Task2Examples,
		task2templates: Task2Templates,
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
		# Validate the inputs
		assert task2examples.keys() == task2templates.keys()
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
		for task in tqdm(task2examples, desc="Meta-testing for each task."):
			# Prepare the input texts and labels for the task
			examples, templates = task2examples[task], task2templates[task]
			input_texts: list[ModelInput] = []
			labels: list[int] = []
			for template in templates:
				for query_example in examples:
					for _ in range(num_prefix_selections):
						input_text = data_loader.prepare_input(
							task,
							query_example,
							examples,
							num_demonstrations,
							template,
							allow_label_overlap,
						)
						input_texts.append(input_text)
						labels.append(query_example["<label>"])
			# Predict on the input, in batches
			self.model.eval()
			output_logits = []
			for example_idx in np.arange(0, len(input_texts), bsz):
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
			precision1, precision, mrr = [], [], []
			for logits, gt_label in zip(output_logits, labels):
				# Find the rank of the ground-truth label
				# w.r.t. all other labels based on logits
				# Best rank is 1, worst rank is num_labels
				rank = np.count_nonzero(logits > logits[gt_label]) + 1
				assert rank >= 1
				mrr.append(1 / rank)
				precision1.append(1 if rank <= 1 else 0)
				precision.append(1 if rank <= 10 else 0)
			task2scores[task] = {
				"precision1": np.mean(precision1),
				"precision10": np.mean(precision),
				"mrr": np.mean(mrr),
			}
		return task2preds, task2scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Meta-training / meta-testing of in-context tuning."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["meta-train", "meta-test"],
        help="whether you want to meta-train a model or meta-test a meta-trained model.",
    )

    parser.add_argument("--model_name", type=str, help="name of the model.")
    parser.add_argument(
        "--task_format",
        type=str,
        choices=["mlm", "clm"],
        help="frame verbalizer classifcation as masked language modeling or causal language modeling.",
    )
    parser.add_argument("--device", type=str, help="device used for this experiment..")
    parser.add_argument(
        "--task2examples",
        type=str,
        help="path to the meta-training/meta-testing data file (a dictionary of task2examples).",
    )
    parser.add_argument(
        "--task2templates",
        type=str,
        help="path to the meta-training/meta-testing templates file (a dictionary of task2templates).",
    )
    parser.add_argument(
        "--task2verbalizers",
        type=str,
        help="path to the meta-training/meta-testing verbalizers file (a dictionary of task2verbalizers).",
    )
    parser.add_argument(
        "--num_demonstrations", type=int, help="number of few-shot demonstrations."
    )
    parser.add_argument(
        "--example_delimiter",
        type=int,
        help="delimiter used to separate consecutive examples in the input text.",
    )
    parser.add_argument(
        "--allow_label_overlap",
        action="store_true",
        help="whether few-shot support examples are allowed to have overlapping labels with the query example.",
    )
    parser.add_argument(
        "--bsz", type=int, help="batch size for meta-training/meta-testing."
    )

    # arguments only used for meta-training
    parser.add_argument("--lr", type=float, help="learning rate for meta-training.")
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        help="number of warmup steps of the learning rate scheduler.",
    )
    parser.add_argument(
        "--num_epochs", type=int, help="number of meta-training epochs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory to store the meta-trained model and the meta-training log file.",
    )

    # arguments only used for meta-testing
    parser.add_argument(
        "--num_prefix_selections",
        type=int,
        help="number of demonstration sampling for each query example (result averaged across different sampled demonstrations).",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        help="path to the meta-trained model to evaluate with.",
    )

    args = parser.parse_args()
    assert args.mode in ["meta-train", "meta-test"]
    if args.mode == "meta-train":
        assert None not in [
            args.model_name,
            args.task_format,
            args.device,
            args.task2examples,
            args.task2templates,
            args.task2verbalizers,
            args.num_demonstrations,
            args.example_delimiter,
            args.allow_label_overlap,
            args.lr,
            args.num_warmup_steps,
            args.num_epochs,
            args.bsz,
            args.output_dir,
        ]
        ict = ICT(args.model_name, args.task_format, args.device)
        task2examples = pkl.load(open(args.task2examples, "rb"))
        task2templates = pkl.load(open(args.task2templates, "rb"))
        task2verbalizers = pkl.load(open(args.task2verbalizers, "rb"))
        ict.meta_train(
            task2examples,
            task2templates,
            task2verbalizers,
            args.num_demonstrations,
            args.example_delimiter,
            args.allow_label_overlap,
            args.lr,
            args.num_warmup_steps,
            args.num_epochs,
            args.bsz,
            args.output_dir,
        )

    elif args.mode == "meta-test":
        assert None not in [
            args.model_name,
            args.task_format,
            args.device,
            args.load_model_path,
            args.task2examples,
            args.task2templates,
            args.task2verbalizers,
            args.num_demonstrations,
            args.example_delimiter,
            args.allow_label_overlap,
            args.num_prefix_selections,
            args.bsz,
        ]
        ict = ICT(args.model_name, args.task_format, args.device, args.load_model_path)
        task2examples = pkl.load(open(args.task2examples, "rb"))
        task2templates = pkl.load(open(args.task2templates, "rb"))
        task2verbalizers = pkl.load(open(args.task2verbalizers, "rb"))
        ict.meta_test(
            task2examples,
            task2templates,
            task2verbalizers,
            args.num_demonstrations,
            args.example_delimiter,
            args.allow_label_overlap,
            args.num_prefix_selections,
            args.bsz,
        )
