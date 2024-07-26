import pathlib as pl
import typer as ty
from loguru import logger as log
import kaggle as kg
from tqdm import tqdm
from learn_convnet.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = ty.Typer()


@app.command()
def main():
    try:
        download_dataset(RAW_DATA_DIR)
    except Exception as e:
        log.error(f"Error downloading dataset: {e}")


def download_dataset(raw_data_dir: pl.Path):
    log.info("Downloading data...")
    raw_data_dir.resolve().mkdir(parents=True, exist_ok=True)
    kg.api.dataset_download_files(
        dataset="ryanholbrook/car-or-truck", path=raw_data_dir, unzip=True
    )
    log.success("Data downloaded.")


def process_dataset(raw_data_dir: pl.Path, processed_data_dir: pl.Path):
    raw_data_train_dir = raw_data_dir / "train"
    raw_data_valid_dir = raw_data_dir / "valid"
    pass


if __name__ == "__main__":
    app()
