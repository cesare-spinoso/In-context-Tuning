from typing import Literal

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
