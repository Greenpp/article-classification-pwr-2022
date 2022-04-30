import pytorch_lightning as pl
import torch
from datasets.load import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..config import TrainingConfig
from .dataset import ArxivDataset


def custom_collate(batch):
    input_ids, attention_masks, token_type_ids, labels = zip(*batch)

    lengths = [len(x) for x in input_ids]

    return (
        torch.cat(input_ids, dim=0),
        torch.cat(attention_masks, dim=0),
        torch.cat(token_type_ids, dim=0),
        torch.tensor(lengths),
        torch.tensor(labels),
    )


class ArxivDataModule(pl.LightningDataModule):
    def __init__(self, segment_size: int, segment_overlap: int) -> None:
        super().__init__()

        self.segment_size = segment_size
        self.segment_overlap = segment_overlap

    def prepare_data(self) -> None:
        dataset = load_dataset("ccdv/arxiv-classification")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.train_dataset = ArxivDataset(dataset["train"], tokenizer, self.segment_size, self.segment_overlap)  # type: ignore
        self.val_dataset = ArxivDataset(dataset["validation"], tokenizer, self.segment_size, self.segment_overlap)  # type: ignore
        self.test_dataset = ArxivDataset(dataset["test"], tokenizer, self.segment_size, self.segment_overlap)  # type: ignore

        self.workers = 4

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=TrainingConfig.batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=custom_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=TrainingConfig.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=custom_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=TrainingConfig.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=custom_collate,
        )
