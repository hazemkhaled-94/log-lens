"""Configuration dataclass for model training hyperparameters."""

import logging
from dataclasses import dataclass
import math

from transformers import TrainingArguments

from configs import RANDOM_STATE

logger = logging.getLogger(__name__)

SUPPORTED_LOSS_FUNCTIONS: tuple[str, ...] = ("weighted_ce", "gce")


@dataclass
class TrainerArgs:
    """Configuration parameters for the model training loop."""

    model_path: str
    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    logging_steps: int = 10
    gradient_accumulation_steps: int = 4
    metric_for_best: str = "eval_f1"
    greater_is_better: bool = True
    warmup_ratio: float = 0.1
    loss_function: str = "weighted_ce"
    gce_q: float = 0.7
    seed: int = RANDOM_STATE

    def __init__(
        self, output_dir: str, logging_dir: str, model_path: str, **kwds
    ) -> None:
        """Initialize trainer arguments, applying any keyword overrides.

        Raises:
            ValueError: If ``loss_function`` or ``gce_q`` is invalid.
        """
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.model_path = model_path
        for key, value in kwds.items():
            setattr(self, key, value)

        if self.loss_function not in SUPPORTED_LOSS_FUNCTIONS:
            raise ValueError(
                "Unsupported loss_function: "
                f"{self.loss_function}. Supported values are "
                f"{SUPPORTED_LOSS_FUNCTIONS}."
            )

        if not 0.0 < float(self.gce_q) <= 1.0:
            raise ValueError(
                f"gce_q must be in range (0, 1], got {self.gce_q}."
            )

    def get_training_args(
        self, train_dataset_length: int
    ) -> TrainingArguments:
        """Build a ``TrainingArguments`` instance from this config."""
        logger.info("Generating Hugging Face TrainingArguments...")

        effective_batch_size = (
            self.batch_size * self.gradient_accumulation_steps
        )
        total_training_steps = (
            math.ceil(train_dataset_length / effective_batch_size)
            * self.epochs
        )
        dynamic_warmup_steps = int(total_training_steps * self.warmup_ratio)
        logger.info(
            f"Dynamic Warmup Calculated: {dynamic_warmup_steps} steps "
            "("
            f"{self.warmup_ratio * 100}% of "
            f"{total_training_steps} total steps"
            ")"
        )

        return TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            report_to="tensorboard",
            metric_for_best_model=self.metric_for_best,
            greater_is_better=self.greater_is_better,
            logging_steps=self.logging_steps,
            load_best_model_at_end=True,
            max_grad_norm=1.0,
            warmup_steps=dynamic_warmup_steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=False,
            bf16=False,
            seed=self.seed,
            data_seed=self.seed,
        )
