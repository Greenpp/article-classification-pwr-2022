# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from article_classification_pwr_2022.config import RANDOM_SEED, TrainingConfig
from article_classification_pwr_2022.data.datamodule import ArxivDataModule
from article_classification_pwr_2022.model import ArxivModel

# %%
pl.seed_everything(RANDOM_SEED)

dm = ArxivDataModule(
    TrainingConfig.segment_size,
    TrainingConfig.segment_overlap,
    TrainingConfig.batch_size,
    TrainingConfig.processed_train,
    TrainingConfig.processed_val,
    TrainingConfig.processed_test,
)
model = ArxivModel(
    TrainingConfig.learning_rate,
    TrainingConfig.encoding_dim,
    TrainingConfig.aggregation_dim,
    TrainingConfig.classes,
)

logger = WandbLogger(project=TrainingConfig.project_name, name=TrainingConfig.run_name)
trainer = pl.Trainer(
    gpus=1,
    max_epochs=TrainingConfig.epochs,
    logger=logger,
)

# %%
trainer.fit(model, dm)
