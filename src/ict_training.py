from typing import Literal
import pickle as pkl
from custom_dataset import ICTDataset, TaskSampler
from src.data_preprocessing import ICTPreprocessor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    AdamW,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


class ICTData(LightningDataModule):
    def __init__(
        self,
        model_name: str,
        task_format: str,
        k: int,
        delimiter: str,
        allow_label_overlap: bool,
        data_path: str,
        cv_split_path: str,
        class_label_path: str,
        batch_size: int,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.task_format = task_format
        self.k = k
        self.delimiter = delimiter
        self.allow_label_overlap = allow_label_overlap
        self.data_path = data_path
        self.cv_split_path = cv_split_path
        self.class_label_path = class_label_path
        self.batch_size = batch_size

        self.ict_preprocessor = None

    def setup(self, stage: str):
        if self.ict_preprocessor is None:
            raise ValueError("You need to call `prepare_data` first.")
        self.datasets = {"train": [], "val": [], "test": []}
        self.samplers = {"train": [], "val": [], "test": []}
        self.class_label_token_ids = self.ict_preprocessor.get_class_label_token_ids()
        for split in self.cv_split:
            for fold_name in self.datasets.keys():
                fold_data = self.ict_preprocessor.get_fold_data(
                    self.data, split[fold_name]
                )
                task2prompts = self.ict_preprocessor.convert_examples_to_prompts(
                    task2examples=fold_data,
                    allow_label_overlap=self.allow_label_overlap,
                    delimiter=self.delimiter,
                )
                dataset = ICTDataset(task2prompts)
                self.datasets[fold_name].append(dataset)
                sampler = TaskSampler(
                    dataset, self.batch_size, inner_shuffle=True, outer_shuffle=True
                )
                self.samplers[fold_name].append(sampler)

    def prepare_data(self):
        # Load the data, cv_split and class labels
        with open(self.data_path, "rb") as f:
            self.data = pkl.load(f)
        with open(self.cv_split_path, "rb") as f:
            self.cv_split = pkl.load(f)
        with open(self.class_label_path, "r") as f:
            self.class_labels = [line for line in f.readlines() if len(line) > 0]
        # Create preprocessor
        self.ict_preprocessor = ICTPreprocessor(
            model_name=self.model_name,
            task_format=self.task_format,
            k=self.k,
            delimiter=self.delimiter,
        )

    def _custom_collate_fn(self, batch):
        collator = DataCollatorWithPadding(self.ict_preprocessor.get_tokenizer())
        return {**collator(batch["prompt"]), "labels": batch["label"]}

    def _create_dataloader(self, dataset, sampler):
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self._custom_collate_fn,
        )

    def _create_dataloaders(self, datasets, samplers):
        return [
            self._create_dataloader(dataset, sampler)
            for dataset, sampler in zip(datasets, samplers)
        ]

    def train_dataloader(self):
        train_datasets = self.datasets["train"]
        train_samplers = self.samplers["train"]
        return self._create_dataloaders(train_datasets, train_samplers)

    def val_dataloader(self):
        val_datasets = self.datasets["val"]
        val_samplers = self.samplers["val"]
        return self._create_dataloaders(val_datasets, val_samplers)

    def test_dataloader(self):
        test_datasets = self.datasets["test"]
        test_samplers = self.samplers["test"]
        return self._create_dataloaders(test_datasets, test_samplers)


class ICTModel(LightningModule):
    def __init__(
        self,
        model_name: str,
        task_format: Literal["mlm", "clm"],
        tokenizer: AutoTokenizer,
        class_label_token_ids: list[int],
        learning_rate: float,
        num_warmup_steps: int,
        num_epochs: int,
        bsz: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            "learning_rate", "num_warmup_steps", "num_epochs", "bsz"
        )
        assert task_format in ["mlm", "clm"], "Unsupported task format."
        self.model_name = model_name
        self.task_format = task_format
        self.tokenizer = tokenizer
        self.class_label_token_ids = torch.LongTensor(class_label_token_ids)
        if self.task_format == "mlm":
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        elif self.task_format == "clm":
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_dict: dict) -> torch.Tensor:
        return self.model(**input_dict)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        output_logits = outputs.logits
        if self.task_format == "mlm":
            masked_token_ids = torch.nonzero(
                batch["input_ids"] == self.tokenizer.mask_token_id, as_tuple=True
            )[1]
            output_logits = torch.index_select(output_logits, 1, masked_token_ids)
            class_label_logits = output_logits[:, self.class_label_token_ids]
        elif self.task_format == "clm":
            class_label_logits = output_logits[:, -1, self.class_label_token_ids]
        return self.loss_fct(class_label_logits, batch["labels"])

    def configure_optimizers(self):
        """Prepare optimizer and scheduler (linear warmup)"""
        model = self.model
        optimizer = AdamW(
            model.named_parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
