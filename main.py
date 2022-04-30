# %%
import logging

import pytorch_lightning as pl

from article_classification_pwr_2022.config import TrainingConfig
from article_classification_pwr_2022.data.datamodule import ArxivDataModule
from article_classification_pwr_2022.model import ArxivModel

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
# %%
dm = ArxivDataModule(TrainingConfig.segment_size, TrainingConfig.segment_overlap)
model = ArxivModel(
    TrainingConfig.learning_rate,
    TrainingConfig.encoding_dim,
    TrainingConfig.aggregation_dim,
    TrainingConfig.classes,
)
# %%
trainer = pl.Trainer(
    gpus=1,
    max_epochs=TrainingConfig.epochs,
    fast_dev_run=True,
)

# %%
trainer.fit(model, dm)
# %%
