from typing import Literal
import pickle as pkl

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)


class ICTData(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        cv_split_path: str,
        class_label_path: str,
        model_name: str,
        task_format: str,
        k: int,
        batch_size: int,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.cv_split_path = cv_split_path
        self.class_label_path = class_label_path
        self.model_name = model_name
        self.task_format = task_format
        self.k = k
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ict_preprocessor = None

    def setup(self, stage: str):
        pass

    def prepare_data(self):
        # Load the data, cv_split and class labels
        data, cv_split, class_labels = None, None, None
        with open(self.data_path, "rb") as f:
            data = pkl.load(f)
        with open(self.cv_split_path, "rb") as f:
            cv_split = pkl.load(f)
        with open(self.class_label_path, "r") as f:
            class_labels = [line for line in f.readlines() if len(line) > 0]
        


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
