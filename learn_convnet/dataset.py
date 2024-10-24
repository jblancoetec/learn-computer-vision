import pathlib as pl
import typer as ty
import learn_convnet.config as cfg
import numpy as np
import tensorflow as tf
import keras as kr
import loguru as lg
import kaggle as kg
import os


app = ty.Typer()


@app.command()
def main():
    try:
        download_dataset(raw_data_dir=cfg.RAW_DATA_DIR)
        set_seed_of_rand_numbers()
        # dataset_train, dataset_valid = process_dataset(raw_data_dir=cfg.RAW_DATA_DIR)
        # dataset_train.save(cfg.PROCESSED_DATA_DIR / "dataset_train")
        # dataset_valid.save(cfg.PROCESSED_DATA_DIR / "dataset_valid")

    except Exception as e:
        lg.logger.error(f"Error downloading dataset: {e}")


def download_dataset(raw_data_dir: pl.Path):
    lg.logger.info("Downloading data...")
    raw_data_dir.resolve()
    if raw_data_dir.exists() and list(raw_data_dir.iterdir()):
        lg.logger.info(f"Data already downloaded in {raw_data_dir}")
        return

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    kg.api.dataset_download_files(
        dataset="ryanholbrook/car-or-truck", path=raw_data_dir, unzip=True
    )
    lg.logger.success("Data downloaded.")


def set_seed_of_rand_numbers():
    seed = int(os.environ.get("PYTHONHASHSEED") or cfg.DEFAULT_PYTHONHASHSEED)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def process_dataset(raw_data_dir: pl.Path) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset_train = get_images_and_labels(raw_data_dir=raw_data_dir / "train")
    dataset_valid = get_images_and_labels(raw_data_dir=raw_data_dir / "valid")
    return (dataset_train, dataset_valid)


def get_images_and_labels(raw_data_dir: pl.Path) -> tf.data.Dataset:
    raw_data_dir.resolve()
    raw_images = kr.preprocessing.image_dataset_from_directory(
        directory=raw_data_dir,
        labels="inferred",
        label_mode="binary",
        image_size=[128, 128],
        interpolation="nearest",
        batch_size=64,
        shuffle=False,
    )

    convert_image = tf.image.convert_image_dtype
    images_dataset = (
        raw_images.map(lambda image, label: (convert_image(image, tf.float32), label))  # type: ignore
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return images_dataset


if __name__ == "__main__":
    app()
