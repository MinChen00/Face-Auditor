from torch.utils.data.dataset import Dataset
import torch


class PyTorchDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        data = self.x[index]
        if self.transform is not None and not isinstance(data, torch.Tensor):
            data = self.transform(data)
        return data, self.y[index]
