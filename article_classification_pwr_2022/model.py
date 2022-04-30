import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.nn.utils.rnn import pack_sequence
from transformers import AutoModelForSequenceClassification


class ArxivModel(pl.LightningModule):
    def __init__(
        self, lr: float, encoding_dim: int, aggregation_dim: int, classes: int
    ) -> None:
        super().__init__()
        self.lr = lr

        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=encoding_dim
        )
        self.aggregator = nn.GRU(input_size=encoding_dim, hidden_size=aggregation_dim)
        self.classifier = nn.Sequential(
            nn.Linear(aggregation_dim, 32),
            nn.ReLU(),
            nn.Linear(32, classes),
        )

        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(num_classes=classes),
                torchmetrics.Precision(num_classes=classes),
                torchmetrics.Recall(num_classes=classes),
                torchmetrics.F1Score(num_classes=classes),
                torchmetrics.ConfusionMatrix(num_classes=classes),
            ]
        )
        self.train_metrics = metrics.clone("train_")
        self.val_metrics = metrics.clone("val_")
        self.test_metrics = metrics.clone("test_")

    def forward(self, X) -> torch.Tensor:
        input_ids, att_mask, token_type_ids, lengths = X
        with torch.no_grad():
            encoder_out = self.encoder(
                input_ids, attention_mask=att_mask, token_type_ids=token_type_ids
            )
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

    def training_step(self, batch, batch_idx) -> dict:
        *X, y = batch
        y_hat = self.forward(X)

        loss = F.cross_entropy(y_hat, y)
        metrics = self.train_metrics(y_hat, y)

        self.log_dict({"loss": loss}, on_step=True, on_epoch=False, prog_bar=True)
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> None:
        *X, y = batch
        y_hat = self.forward(X)

        loss = F.cross_entropy(y_hat, y)
        metrics = self.val_metrics(y_hat, y)

        self.log_dict({"val_loss": loss}, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        *X, y = batch
        y_hat = self.forward(X)

        loss = F.cross_entropy(y_hat, y)
        metrics = self.test_metrics(y_hat, y)

        self.log_dict({"test_loss": loss}, on_step=True, on_epoch=True, prog_bar=False)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
