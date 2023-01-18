from typing import Literal
import pickle as pkl
from custom_dataset import ICTDataset, TaskSampler
from data_preprocessing import ICTPreprocessor
from copy import deepcopy
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
    pickled_data = {
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
        "group0_KaggleTwitterPolitics": [
            {
                "<input>": "RT @Surgeon_General: We lose someone from an opioid overdoes every 12.5 minutes, with more than half dying at home. This impacts every commâ\x80¦",
                "template": "<input> Is this a tweet from Republican Party? <label>",
                "<label>": 0,
            },
            {
                "<input>": "In the event of a military attack, @PacificCommand should be the source of the information and the lead in notificaâ\x80¦ https://t.co/o9A3iICqYq",
                "template": "<input> Is this a tweet from Democratic Party? <label>",
                "<label>": 1,
            },
            {
                "<input>": "â\x80\x9cPart of Congressâ\x80\x99 recent abdication is letting the Executive Branch make all the tough decisions...we should haveâ\x80¦ https://t.co/0MIurgB0sX",
                "template": "<input> Does this lean toward Democratic Party? <label>",
                "<label>": 0,
            },
            {
                "<input>": "GOP @HouseCommerce leaders are attempting to remedy their slow response to the #opioid epidemic by hastily markingâ\x80¦ https://t.co/StAmHRLzYb",
                "template": "<input> Is this a Democratic post? <label>",
                "<label>": 0,
            },
            {
                "<input>": "Appreciated the opportunity to meet with @97AMW Commander, Col. Todd Hohn, to discuss the latest at Altus AFB https://t.co/zm2MapIbiU",
                "template": "<input> Is this a Republican post? <label>",
                "<label>": 1,
            },
        ],
    }
    cv_splits = [
        {
            "train": ["group0_2016SemEval6TweetEvalStanceHillary"],
            "val": ["group2_KaggleCovidTweetSentiment"],
            "test": ["group0_KaggleTwitterPolitics"],
        },
        {
            "train": ["group2_KaggleCovidTweetSentiment"],
            "val": ["group0_KaggleTwitterPolitics"],
            "test": ["group0_2016SemEval6TweetEvalStanceHillary"],
        },
        {
            "train": ["group0_KaggleTwitterPolitics"],
            "val": ["group0_KaggleTwitterPolitics"],
            "test": ["group2_KaggleCovidTweetSentiment"],
        },
    ]
    class_labels = ["no", "yes"]

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
        self._fold_number = None

    @property
    def fold_number(self):
        return self._fold_number

    @fold_number.setter
    def fold_number(self, value):
        if not (hasattr(self, "datasets") and hasattr(self, "samplers")):
            raise ValueError("You need to call `setup` first.")
        else:
            self._fold_number = value

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
                )
                dataset = ICTDataset(task2prompts)
                self.datasets[fold_name].append(dataset)
                sampler = TaskSampler(
                    dataset, self.batch_size, inner_shuffle=True, outer_shuffle=True
                )
                self.samplers[fold_name].append(sampler)

    def prepare_data(self):
        # Load the data, cv_split and class labels
        # with open(self.data_path, "rb") as f:
        #     self.data = pkl.load(f)
        # with open(self.cv_split_path, "rb") as f:
        #     self.cv_split = pkl.load(f)
        # with open(self.class_label_path, "r") as f:
        #     self.class_labels = [line for line in f.readlines() if len(line) > 0]
        self.data = ICTData.pickled_data
        self.cv_split = ICTData.cv_splits
        self.class_labels = ICTData.class_labels
        # Create preprocessor
        self.ict_preprocessor = ICTPreprocessor(
            model_name=self.model_name,
            task_format=self.task_format,
            k=self.k,
            class_labels=self.class_labels,
            allow_label_overlap=self.allow_label_overlap,
            delimiter=self.delimiter,
        )

    def _custom_collate_fn(self, batch):
        tokenizer = self.ict_preprocessor.get_tokenizer()
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        labels = torch.LongTensor([item["label"] for item in batch])
        prompts = [item["prompt"] for item in batch]
        return {**collator(tokenizer(prompts)), "labels": labels}

    def _create_dataloader(self, dataset, sampler):
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self._custom_collate_fn,
        )

    def train_dataloader(self):
        if self._fold_number is None:
            raise ValueError("You need to set `fold_number` first.")
        train_dataset = self.datasets["train"][self._fold_number]
        train_sampler = self.samplers["train"][self._fold_number]
        return self._create_dataloader(train_dataset, train_sampler)

    def val_dataloader(self):
        if self._fold_number is None:
            raise ValueError("You need to set `fold_number` first.")
        val_dataset = self.datasets["val"][self._fold_number]
        val_sampler = self.samplers["val"][self._fold_number]
        return self._create_dataloader(val_dataset, val_sampler)

    def test_dataloader(self):
        if self._fold_number is None:
            raise ValueError("You need to set `fold_number` first.")
        test_dataset = self.datasets["test"][self._fold_number]
        test_sampler = self.samplers["test"][self._fold_number]
        return self._create_dataloader(test_dataset, test_sampler)


class ICTModel(LightningModule):
    def __init__(
        self,
        model_name: str,
        task_format: Literal["mlm", "clm"],
        tokenizer: AutoTokenizer,
        class_label_token_ids: list[int],
        num_folds: int,
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
        self.models = [deepcopy(self.model) for _ in range(num_folds)]
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_dict: dict) -> torch.Tensor:
        return self.model(**input_dict)

    def training_step(self, batch, batch_idx):
        outputs = self(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
        )
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
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = AdamW(
            params,
            lr=self.hparams.learning_rate,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


from pathlib import Path

parent_dir = Path(__file__).parent
model_name = "gpt2"
task_format = "clm"
k = 2
delimiter = " "
allow_label_overlap = True
batch_size = 3
ict_data_module = ICTData(
    model_name,
    task_format,
    k,
    delimiter,
    allow_label_overlap,
    "mock_data_path",
    "mock_cv_split_path",
    "mock_class_label_path",
    batch_size,
)
ict_data_module.prepare_data()
ict_data_module.setup("mock_stage")
ict_data_module.fold_number = 0
tokenizer = ict_data_module.ict_preprocessor.get_tokenizer()
class_label_token_ids = ict_data_module.ict_preprocessor.get_class_label_token_ids()
model = ICTModel(
    model_name=model_name,
    task_format=task_format,
    tokenizer=tokenizer,
    class_label_token_ids=class_label_token_ids,
    num_folds=len(ict_data_module.cv_split),
    learning_rate=2e-5,
    num_warmup_steps=100,
    num_epochs=1,
    bsz=batch_size,
)
trainer = Trainer(
    max_epochs=1,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
)
trainer.fit(model, datamodule=ict_data_module)
