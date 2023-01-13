import pickle as pkl
from torch.utils.data import Dataset, DataLoader

# Create simple dataset to use with a dataloader to test training

class ICTDataset(Dataset):
    def __init__(self, dataset_name: str, pickle_path: str) -> None:
        self.dataset_name = dataset_name
        pickle_data = {}
        with open(pickle_path, "rb") as f:
            pickle_data = pkl.load(f)
        # Flatten the dictionary
        # Assumption: Template also in the example
        self.data = [{"task": task, **example} for task, examples in pickle_data for example in examples]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> dict:
        return self.data[idx]
        
        
        

