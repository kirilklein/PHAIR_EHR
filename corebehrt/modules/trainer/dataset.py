import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = (
            X.to(torch.float32)
            if isinstance(X, torch.Tensor)
            else torch.tensor(X, dtype=torch.float32)
        )
        self.y = (
            y.to(torch.float32)
            if isinstance(y, torch.Tensor)
            else torch.tensor(y, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return (features, label)
        return self.X[idx], self.y[idx]