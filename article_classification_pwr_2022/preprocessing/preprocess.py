import pickle as pkl
from pathlib import Path
from typing import Iterable

import click
from datasets.load import load_dataset
from tqdm import tqdm

from ..config import TrainingConfig
from .preprocessor import BERTPreprocessor


def generate_batches(data, batch_size: int) -> Iterable[dict]:
    batches_num = len(data) // batch_size
    if not len(data) % batch_size:
        batches_num += 1

    for i in tqdm(range(batches_num), total=batches_num, smoothing=0):
        yield data[i * batch_size : (i + 1) * batch_size]


def save_data(data: dict, path: str) -> None:
    with open(path, "wb") as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


def process_data(data: list, preprocessor: BERTPreprocessor, batch_size: int) -> list:
    processed_data = []
    for batch in generate_batches(data, batch_size):
        processed_data.extend(preprocessor.preprocess(batch))

    return processed_data


@click.command()
@click.option(
    "--segment_size",
    type=int,
    default=TrainingConfig.segment_size,
    help="The size of the segments",
)
@click.option(
    "--segment_overlap",
    type=int,
    default=TrainingConfig.segment_overlap,
    help="The overlap of the segments",
)
@click.option(
    "--encoding_dim",
    type=int,
    default=TrainingConfig.encoding_dim,
    help="The dimension of the encoding",
)
@click.option(
    "--processing_batch_size",
    type=int,
    default=TrainingConfig.processing_batch_size,
    help="The processing batch size",
)
@click.option(
    "--tokenization_batch_size",
    type=int,
    default=TrainingConfig.tokenization_batch_size,
    help="The tokenization batch size",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="The device to use",
)
@click.option(
    "--output_file",
    type=str,
    default=TrainingConfig.processed_data,
    help="The output file",
)
def main(
    segment_size: int,
    segment_overlap: int,
    encoding_dim: int,
    processing_batch_size: int,
    tokenization_batch_size: int,
    device: str,
    output_file: str,
) -> None:
    print(f"Preprocessing with parameters {locals()}")
    dataset = load_dataset("ccdv/arxiv-classification")

    train_dataset = dataset["train"]  # type: ignore
    val_dataset = dataset["validation"]  # type: ignore
    test_dataset = dataset["test"]  # type: ignore

    preprocessor = BERTPreprocessor(
        segment_size=segment_size,
        segment_overlap=segment_overlap,
        encoding_dim=encoding_dim,
        batch_size=processing_batch_size,
        device=device,
    )

    print("Preprocessing train data...")
    train_data = process_data(train_dataset, preprocessor, tokenization_batch_size)
    print("Preprocessing val data...")
    val_data = process_data(val_dataset, preprocessor, tokenization_batch_size)
    print("Preprocessing test data...")
    test_data = process_data(test_dataset, preprocessor, tokenization_batch_size)

    data = {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    }

    print("Saving data...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    save_data(data, output_file)


if __name__ == "__main__":
    main()
