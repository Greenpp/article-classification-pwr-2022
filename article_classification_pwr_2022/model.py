import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import wandb
from torch.nn.utils.rnn import pack_sequence
from transformers import AutoModelForSequenceClassification
from wandb.plot import confusion_matrix

from .config import TrainingConfig
from .data.label_translator import LABEL_MAP


class ArxivModel(pl.LightningModule):
    def __init__(
        self, lr: float, encoding_dim: int, aggregation_dim: int, classes: int
    ) -> None:
        super().__init__()
        self.lr = lr

        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            TrainingConfig.model, num_labels=encoding_dim
        )
        self.encoder.trainable = False
        self.aggregator = nn.GRU(input_size=encoding_dim, hidden_size=aggregation_dim)
        self.classifier = nn.Sequential(
            nn.Linear(aggregation_dim, 32),
            nn.ReLU(),
            nn.Linear(32, classes),
        )

        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(num_classes=classes),
                torchmetrics.F1Score(num_classes=classes),
            ]
        )
        self.train_metrics = metrics.clone("train_")
        self.val_metrics = metrics.clone("val_")
        self.test_metrics = metrics.clone("test_")

    def forward(self, X) -> torch.Tensor:
        input_ids, att_mask, lengths = X
        with torch.no_grad():
            encoder_out = self.encoder(input_ids, attention_mask=att_mask)
        encoding = encoder_out.logits

        encodings = []
        encoding_end = 0
        for i in range(len(lengths)):
            encoding_start = encoding_end
            encoding_end = encoding_start + lengths[i]
            encodings.append(encoding[encoding_start:encoding_end])
        packed_encoding = pack_sequence(encodings, enforce_sorted=False)

        _, hidden = self.aggregator(packed_encoding)
        logits = self.classifier(hidden.squeeze(0))

        return logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        *X, y = batch
        y_hat = self.forward(X)

        loss = F.cross_entropy(y_hat, y)
        metrics = self.train_metrics(y_hat, y)

        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx) -> dict:
        *X, y = batch
        y_hat = self.forward(X)

        loss = F.cross_entropy(y_hat, y)
        metrics = self.val_metrics(y_hat, y)

        self.log_dict({"val_loss": loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "y_hat": y_hat.detach().cpu().numpy(),
            "y": y.detach().cpu().numpy(),
        }

    def validation_epoch_end(self, outputs: dict) -> None:
        y_hats, ys = [], []
        for o in outputs:
            y_hat, y = o["y_hat"], o["y"]
            y_hats.append(y_hat)
            ys.append(y)

        y_hats_cat = np.concatenate(y_hats)
        ys_cat = np.concatenate(ys)

        wandb.log(
            {
                "val_confusion_matrix": confusion_matrix(
                    probs=y_hats_cat,
                    y_true=ys_cat,
                    class_names=list(LABEL_MAP.values()),
                )
            }
        )

    def test_step(self, batch, batch_idx) -> dict:
        *X, y = batch
        y_hat = self.forward(X)

        loss = F.cross_entropy(y_hat, y)
        metrics = self.test_metrics(y_hat, y)

        self.log_dict({"test_loss": loss}, on_step=False, on_epoch=True, prog_bar=False)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)

        return {
            "y_hat": y_hat.detach().cpu().numpy(),
            "y": y.detach().cpu().numpy(),
        }

    def test_epoch_end(self, outputs: list) -> None:
        y_hats, ys = [], []
        for o in outputs:
            y_hat, y = o["y_hat"], o["y"]
            y_hats.append(y_hat)
            ys.append(y)

        y_hats_cat = np.concatenate(y_hats)
        ys_cat = np.concatenate(ys)

        wandb.log(
            {
                "test_confusion_matrix": confusion_matrix(
                    probs=y_hats_cat,
                    y_true=ys_cat,
                    class_names=list(LABEL_MAP.values()),
                )
            }
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
