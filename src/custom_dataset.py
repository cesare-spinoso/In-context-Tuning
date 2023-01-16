import random
from custom_types import Task2Prompts
from torch.utils.data import Dataset, Sampler, DataLoader


class ICTDataset(Dataset):
    def __init__(self, task2prompts: Task2Prompts) -> None:
        # Flatten the dictionary
        # Assumption 1: The template is also in the example
        # Assumption 2: The fold is already taken care of outside the dataset
        self.data = [
            {"task": task, **prompt}
            for task, prompts in task2prompts.items()
            for prompt in prompts
        ]
        self.unflattened_data = [
            {"task": task, "data": prompts} for task, prompts in task2prompts.items()
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        return self.data[idx]


class TaskSampler(Sampler):
    def __init__(
        self,
        dataset: ICTDataset,
        batch_size: int,
        inner_shuffle: bool,
        outer_shuffle: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.inner_shuffle = inner_shuffle
        self.outer_shuffle = outer_shuffle

    def __iter__(self):
        batch_indices = []
        index_shift = 0
        for task_data in self.dataset.unflattened_data:
            for i in range(0, len(task_data["data"]), self.batch_size):
                start_index = i
                end_index = i + self.batch_size
                if end_index > len(task_data["data"]):
                    end_index = len(task_data["data"])
                batch_indices.append(
                    list(range(start_index + index_shift, end_index + index_shift))
                )
            index_shift += len(task_data["data"])
        if self.inner_shuffle:
            for i, batch in enumerate(batch_indices):
                batch_indices[i] = random.sample(batch, len(batch))
        if self.outer_shuffle:
            random.shuffle(batch_indices)
        for batch in batch_indices:
            yield batch

    def __len__(self):
        len(self.dataset)


from data_preprocessing import task2prompts

dataset = ICTDataset(task2prompts)

dataloader = DataLoader(
    dataset=dataset, batch_sampler=TaskSampler(dataset, 3, True, True)
)

for batch in dataloader:
    print(batch)
