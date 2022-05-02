import torch
from torch.utils.data import Dataset


class ArxivDataset(Dataset):
    def __init__(
        self,
        dataset,
    ) -> None:
        super().__init__()

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        encodings = self.dataset[idx]["encodings"]

        label = self.dataset[idx]["label"]

        return encodings, label
