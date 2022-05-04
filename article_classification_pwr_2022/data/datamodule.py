import pickle as pkl

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .dataset import ArxivDataset


def custom_collate(batch):
    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    encodings, labels = zip(*sorted_batch)

    return (
        [torch.tensor(e) for e in encodings],
        torch.tensor(labels),
    )


class ArxivDataModule(pl.LightningDataModule):
    def __init__(
        self,
        segment_size: int,
        segment_overlap: int,
        batch_size: int,
        train_data_path: str,
        val_data_path: str,
        test_data_path: str,
    ) -> None:
        super().__init__()

        self.segment_size = segment_size
        self.segment_overlap = segment_overlap
        self.batch_size = batch_size

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path

        self.workers = 4

    def prepare_data(self) -> None:
        with open(self.train_data_path, "rb") as f:
            self.train_dataset = ArxivDataset(pkl.load(f))

        with open(self.val_data_path, "rb") as f:
            self.val_dataset = ArxivDataset(pkl.load(f))

        with open(self.test_data_path, "rb") as f:
            self.test_dataset = ArxivDataset(pkl.load(f))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=custom_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=custom_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=custom_collate,
        )
