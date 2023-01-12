"""Training script using the modified ict version."""

import itertools
import pickle as pkl
import sys
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple, OrderedDict

import numpy as np
from tqdm import tqdm

from ict_2 import ICT

parent_dir = Path(__file__).parent.parent


class TableKey(NamedTuple):
    model_name: str
    num_demonstrations: int


class HPKey(NamedTuple):
    number_of_epochs: int
    learning_rate: float


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
        params["model_names"] = ["gpt2-medium", "gpt2-large"]
        params["number_of_demonstrations"] = [5, 0]
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


def main(dataset_name: str):
    # Load the parameters for the dataset
    params = parameters(dataset_name)
    # Table setup
    model_names = params["model_names"]
    number_of_demonstrations = params["number_of_demonstrations"]
    task_format = params["task_format"]
    table_level_combinations = list(
        itertools.product(model_names, number_of_demonstrations)
    )
    # Training hyper-parameters
    number_of_epochs = params["number_of_epochs"]
    learning_rates = params["learning_rates"]
    example_delimiter = params["example_delimiter"]
    batch_size = params["batch_size"]
    num_warmup_steps = params["num_warmup_steps"]
    allow_label_overlap = params["allow_label_overlap"]
    device = params["device"]
    num_prefix_selections = params["num_prefix_selections"]
    metrics = params["metrics"]
    hp_level_combinations = list(itertools.product(number_of_epochs, learning_rates))

    # Prepare data
    cv_split = pkl.load(
        open(parent_dir / f"data_{dataset_name}" / "cross_validation_splits.pkl", "rb")
    )
    data = pkl.load(open(parent_dir / f"data_{dataset_name}" / "data.pkl", "rb"))

    # Load verbalizers
    verbalizers = []
    with open(parent_dir / "data_biclfs" / "class_verbalizers.txt", "r") as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) != 0:
                verbalizers.append(word)

    # Prepare data structures to hold results for the table
    # Trying to replicate
    table_level_results = {}
    fold_level_results = {}
    selected_model_names = {}

    for model_name, num_demonstrations in tqdm(
        table_level_combinations,
        desc=f"Table Level Loop",
    ):
        table_level_results[(model_name, num_demonstrations)] = []
        fold_level_results[(model_name, num_demonstrations)] = []
        selected_model_names[(model_name, num_demonstrations)] = []
        for fold_idx, fold in tqdm(enumerate(cv_split), desc=f"Fold Loop"):
            # Each fold is like a model with a training, validation and testing set
            # Find optimal HP based on validation set and then get the test set results
            # Get the tasks for this fold
            train_tasks, val_tasks, test_tasks = (
                fold["train"],
                fold["val"],
                fold["test"],
            )
            # Training
            train_task2examples = {task: data[task] for task in train_tasks}
            train_task2verbalizers = {task: verbalizers for task in train_tasks}
            # Validation
            val_task2examples = {task: data[task] for task in val_tasks}
            val_task2verbalizers = {task: verbalizers for task in val_tasks}
            # Testing
            test_task2examples = {task: data[task] for task in test_tasks}
            test_task2verbalizers = {task: verbalizers for task in test_tasks}

            # Best val score tracker
            best_val_score = 0
            ict_max = None
            for epochs, lr in tqdm(hp_level_combinations, desc=f"HP Level Loop"):
                ict_identifier = f"model_{model_name}_k_{num_demonstrations}_fold_{fold_idx}_epochs_{epochs}_lr_{lr}"
                ict = ICT(
                    model_name=model_name,
                    task_format=task_format,
                    device=device,
                    identifier=ict_identifier,
                )
                # Meta-train
                ict.meta_train(
                    task2template_examples=train_task2examples,
                    task2verbalizers=train_task2verbalizers,
                    num_demonstrations=num_demonstrations,
                    example_delimiter=example_delimiter,
                    allow_label_overlap=allow_label_overlap,
                    lr=lr,
                    num_warmup_steps=num_warmup_steps,
                    num_epochs=epochs,
                    bsz=batch_size,
                    output_dir=parent_dir / f"output_{dataset_name}" / ict_identifier,
                )
                # Meta-test on val
                _, val_task2scores = ict.meta_test(
                    val_task2examples,
                    task2verbalizers=val_task2verbalizers,
                    num_demonstrations=num_demonstrations,
                    example_delimiter=example_delimiter,
                    allow_label_overlap=allow_label_overlap,
                    num_prefix_selections=num_prefix_selections,
                    bsz=batch_size,
                )
                # Average across tasks
                metric2scores = OrderedDict(
                    {
                        metric: [
                            task_score[metric]
                            for task_score in val_task2scores.values()
                        ]
                        for metric in metrics
                    }
                )
                val_metric2avg_scores = [
                    np.mean(scores) for scores in metric2scores.values()
                ]
                # Track best val score, only meta-test best ict
                # Use precision1 as the metric to track
                # Both for LAMA and BiCLFS (where p@1 is the accuracy)
                if val_metric2avg_scores[1] > best_val_score:
                    best_val_score = val_metric2avg_scores[1]
                    ict_max = deepcopy(ict)
            # Meta-test with ict_max on test
            _, test_task2scores = ict_max.meta_test(
                test_task2examples,
                task2verbalizers=test_task2verbalizers,
                num_demonstrations=num_demonstrations,
                example_delimiter=example_delimiter,
                allow_label_overlap=False,
                num_prefix_selections=num_prefix_selections,
                bsz=batch_size,
            )
            # Save the best models for each fold (to be used afterwards)
            selected_model_names[(model_name, num_demonstrations)].append(
                ict_max.identifier
            )
            # Average across tasks
            test_metric = [
                np.mean(
                    [task_score[metric] for task_score in test_task2scores.values()]
                )
                for metric in metrics
            ]
            fold_level_results[(model_name, num_demonstrations)].append(test_metric)
        # Average across folds
        table_level_results[(model_name, num_demonstrations)] = np.mean(
            fold_level_results[(model_name, num_demonstrations)], axis=0
        )
        # Pickle the table results as they come in
        with open(
            parent_dir / "results" / f"fold_level_results_{dataset_name}.pkl", "wb"
        ) as f:
            pkl.dump(fold_level_results, f)
        with open(
            parent_dir / "results" / f"selected_model_names_{dataset_name}.pkl", "wb"
        ) as f:
            pkl.dump(selected_model_names, f)
        with open(
            parent_dir / "results" / f"table_level_results_{dataset_name}.pkl", "wb"
        ) as f:
            pkl.dump(table_level_results, f)


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    main(dataset_name)
