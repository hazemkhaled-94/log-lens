# mypy: disable-error-code=import-untyped

"""Utility script to train and test Drain3 masking on local datasets."""

import logging

from data_manager.masker.config import Drain3Config
from data_manager.masker.pipeline import Drain3Pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    """Run Drain3 training and testing over configured directories."""
    config = Drain3Config()
    pipeline = Drain3Pipeline(config)
    pipeline.train_from_directory()
    pipeline.test_from_directory()


if __name__ == "__main__":
    main()
