from __future__ import annotations
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        self.ids = token_ids
        self.block = block_size

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.block - 1)

    def __getitem__(self, i: int):
        x = torch.tensor(self.ids[i:i+self.block], dtype=torch.long)
        y = torch.tensor(self.ids[i+1:i+self.block+1], dtype=torch.long)
        return x, y
