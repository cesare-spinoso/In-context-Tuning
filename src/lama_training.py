import pickle as pkl
import itertools
from pathlib import Path
from typing import NamedTuple
import numpy as np
from ict import ICT
from tqdm import tqdm

parent_dir = Path(__file__).parent.parent


class TableKey(NamedTuple):
    model_name: str
    num_demonstrations: int


class HPKey(NamedTuple):
    number_of_epochs: int
    learning_rate: float


def main():
    # Table setup
    # model_names = ["bert-base-cased", "bert-large-cased", "deberta-v2-xlarge"]
    # number_of_demonstrations = [0, 1, 2, 5]
    model_names = ["bert-base-cased"]
    number_of_demonstrations = [0]
    task_format = "mlm"
    table_level_combinations = list(
        itertools.product(model_names, number_of_demonstrations)
    )
    # Training hyper-parameters
    # number_of_epochs = [10, 15, 30]
    # learning_rates = [1e-7, 3e-7, 1e-6, 3e-6]
    number_of_epochs = [1]
    learning_rates = [1e-7]
    example_delimiter = " "
    batch_size = 48
    num_warmup_steps = 100
    allow_label_overlap = False
    device = "cuda"
    num_prefix_selections = 5
    metrics = ["mrr", "precision1", "precision10"]
    hp_level_combinations = list(itertools.product(number_of_epochs, learning_rates))

    # Prepare data
    cv_split = pkl.load(
        open(parent_dir / "example_data" / "cross_validation_splits.pkl", "rb")
    )
    data = pkl.load(open(parent_dir / "example_data" / "data.pkl", "rb"))
    templates = pkl.load(open(parent_dir / "example_data" / "templates.pkl", "rb"))

    # Load verbalizers
    verbalizers = []
    with open(parent_dir / "example_data" / "class_verbalizers.txt", "r") as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) != 0:
                verbalizers.append(word)

    # Prepare data structures to hold results for the table
    # Trying to replicate
    table_level_results = {}

    for model_name, num_demonstrations in tqdm(
        table_level_combinations, desc="Table Level Loop"
    ):
        table_level_results[TableKey(model_name, num_demonstrations)] = []
        for fold_idx, fold in tqdm(enumerate(cv_split), desc="Fold Loop"):
            # Each fold is like a mode with a training, validation and testing set
            # Find optimal HP based on validation set and then get the test set results
            val_hp_level_results = []
            test_hp_level_results = []
            # Get the tasks for this fold
            train_tasks, val_tasks, test_tasks = (
                fold["train"],
                fold["val"],
                fold["test"],
            )
            # Training
            train_task2examples = {task: data[task] for task in train_tasks}
            train_task2templates = {task: templates[task] for task in train_tasks}
            train_task2verbalizers = {task: verbalizers for task in train_tasks}
            # Validation
            val_task2examples = {task: data[task] for task in val_tasks}
            val_task2templates = {task: templates[task] for task in val_tasks}
            val_task2verbalizers = {task: verbalizers for task in val_tasks}
            # Testing
            test_task2examples = {task: data[task] for task in test_tasks}
            test_task2templates = {task: templates[task] for task in test_tasks}
            test_task2verbalizers = {task: verbalizers for task in test_tasks}
            for epochs, lr in tqdm(hp_level_combinations, desc="HP Level Loop"):
                ict = ICT(model_name=model_name, task_format=task_format, device=device)
                # Meta-train
                ict.meta_train(
                    task2examples=train_task2examples,
                    task2templates=train_task2templates,
                    task2verbalizers=train_task2verbalizers,
                    num_demonstrations=num_demonstrations,
                    example_delimiter=example_delimiter,
                    allow_label_overlap=allow_label_overlap,
                    lr=lr,
                    num_warmup_steps=num_warmup_steps,
                    num_epochs=epochs,
                    bsz=batch_size,
                    output_dir=parent_dir
                    / "output"
                    / f"model_{model_name}_k_{num_demonstrations}_fold_{fold_idx}_epochs_{epochs}_lr_{lr}",
                )
                # Meta-test on val
                _, val_task2scores = ict.meta_test(
                    val_task2examples,
                    val_task2templates,
                    task2verbalizers=val_task2verbalizers,
                    num_demonstrations=num_demonstrations,
                    example_delimiter=example_delimiter,
                    allow_label_overlap=False,
                    num_prefix_selections=num_prefix_selections,
                    bsz=batch_size,
                )
                # Average across tasks
                metric2scores = {
                    metric: [
                        task_score[metric] for task_score in val_task2scores.values()
                    ]
                    for metric in metrics
                }
                val_metric2avg_scores = (
                    np.mean(scores) for scores in metric2scores.values()
                )
                # Meta-test on test
                _, test_task2scores = ict.meta_test(
                    test_task2examples,
                    test_task2templates,
                    task2verbalizers=test_task2verbalizers,
                    num_demonstrations=num_demonstrations,
                    example_delimiter=example_delimiter,
                    allow_label_overlap=False,
                    num_prefix_selections=num_prefix_selections,
                    bsz=batch_size,
                )
                # Average across tasks
                metric2scores = {
                    metric: [
                        task_score[metric] for task_score in test_task2scores.values()
                    ]
                    for metric in metrics
                }
                test_metric2avg_scores = (
                    np.mean(scores) for scores in metric2scores.values()
                )
                # Save results for this HP configuration
                val_hp_level_results.append(val_metric2avg_scores)
                test_hp_level_results.append(test_metric2avg_scores)
            # Get the max val score and save the corresponding test score into the table results
            argmax_val_score = np.argmax(val_hp_level_results)
            table_level_results[TableKey(model_name, num_demonstrations)].append(
                test_hp_level_results[argmax_val_score]
            )
        # Average across folds
        table_level_results[TableKey(model_name, num_demonstrations)] = np.mean(
            table_level_results[TableKey(model_name, num_demonstrations)]
        )
        # Pickle the table results as they come in
        with open(parent_dir / "table_level_results.pkl", "wb") as f:
            pkl.dump(table_level_results, f)


if __name__ == "__main__":
    main()
