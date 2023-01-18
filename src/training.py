import argparse
import itertools
import torch
from pytorch_lightning import Trainer
from pathlib import Path
from ict import ICTData, ICTModel

parent_dir = Path(__file__).parent


def parameters(dataset_name: str) -> dict:
    params = {}
    if dataset_name == "lama":
        params["model_names"] = [
            "bert-base-cased",
            "bert-large-cased",
            "microsoft/deberta-v2-xlarge",
        ]
        params["number_of_demonstrations"] = [0, 1, 2, 5]
        params["task_format"] = "mlm"
        params["number_of_epochs"] = [10, 15, 30]
        params["learning_rates"] = [1e-7, 3e-7, 1e-6, 3e-6]
        params["example_delimiter"] = " "
        params["batch_size"] = 48
        params["num_warmup_steps"] = 100
        params["allow_label_overlap"] = False
        params["device"] = "cuda"
        params["num_prefix_selections"] = 5
        params["metrics"] = ["mrr", "precision1", "precision10"]
    elif dataset_name == "biclfs":
        params["model_names"] = ["gpt2", "gpt2-medium", "gpt2-large"]
        params["number_of_demonstrations"] = [0, 5]
        params["task_format"] = "clm"
        params["number_of_epochs"] = [10, 15, 30]
        params["learning_rates"] = [1e-7, 3e-7, 1e-6, 3e-6]
        params["example_delimiter"] = " "
        params["batch_size"] = 16
        params["num_warmup_steps"] = 100
        params["allow_label_overlap"] = True
        params["device"] = "cuda"
        params["num_prefix_selections"] = 20
        params["metrics"] = ["mrr", "precision1", "precision10"]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return params


def train(dataset_name, training_params):
    model_names = training_params["model_names"]
    number_of_demonstrations = training_params["number_of_demonstrations"]
    task_format = training_params["task_format"]
    table_level_combinations = list(
        itertools.product(model_names, number_of_demonstrations)
    )
    for model_name, k in table_level_combinations:
        # Create the data module for this k-shot model
        delimiter = training_params["example_delimiter"]
        allow_label_overlap = training_params["allow_label_overlap"]
        data_path = parent_dir / f"data_{dataset_name}" / "data.pkl"
        cv_split_path = parent_dir / f"data_{dataset_name}" / "data.pkl"
        class_label_path = parent_dir / f"data_{dataset_name}" / "data.pkl"
        batch_size = training_params["batch_size"]
        ict_data_module = ICTData(
            model_name=model_name,
            task_format=task_format,
            k=k,
            delimiter=delimiter,
            allow_label_overlap=allow_label_overlap,
            data_path=data_path,
            cv_split_path=cv_split_path,
            class_label_path=class_label_path,
            batch_size=batch_size,
        )
        # Load the datasets, initialize preprocessor
        ict_data_module.prepare_data()
        # Preprocess data for each fold
        ict_data_module.setup()
        tokenizer = ict_data_module.ict_preprocessor.get_tokenizer()
        class_label_token_ids = (
            ict_data_module.ict_preprocessor.get_class_label_token_ids()
        )
        for fold_number in range(len(ict_data_module.cv_split)):
            ict_data_module.fold_number = fold_number
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
                devices=1
                if torch.cuda.is_available()
                else None,  # limiting got iPython runs
            )
            trainer.fit(model, datamodule=ict_data_module)
            test_results = trainer.test(datamodule=ict_data_module)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="biclfs", required=True)
    args = parser.parse_args()
    return args.dataset_name


def main():
    dataset_name = parse_args()
    training_params = parameters(dataset_name)
    train(dataset_name, training_params)


if __name__ == "__main__":
    main()
