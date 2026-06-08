"""Training orchestration for log classification."""

import logging
from datetime import datetime

import torch
import torch.nn.functional as F
import evaluate  # type: ignore
import numpy as np
from datasets import DatasetDict  # type: ignore
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import EvalPrediction, TrainOutput

from data_manager.logs.log_labels import LogLevel
from .trainer_args import TrainerArgs

logger = logging.getLogger(__name__)


class UnfreezeCallback(TrainerCallback):
    """Unfreezes the base model backbone at a specific epoch."""

    def __init__(self, unfreeze_epoch: int):
        """Store the epoch at which backbone unfreezing starts."""
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Unfreeze backbone parameters when target epoch is reached."""

        model = kwargs.get("model")
        if model is None:
            logger.error("UnfreezeCallback: No model found in kwargs.")
            raise RuntimeError("Model not found in callback kwargs.")

        if (
            model is not None
            and state.epoch is not None
            and int(state.epoch) == self.unfreeze_epoch
        ):
            logger.info(
                f"\nEPOCH {self.unfreeze_epoch}: "
                "Unfreezing the base model backbone!"
            )

            for param in model.base_model.parameters():
                param.requires_grad = True

            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()


class WeightedLossTrainer(Trainer):
    """Trainer supporting weighted and generalized cross-entropy."""

    def __init__(
        self,
        class_weights: list[float] | None = None,
        loss_function: str = "weighted_ce",
        gce_q: float = 0.7,
        *args,
        **kwargs,
    ):
        """Initialize weighted loss trainer behavior.

        Args:
            class_weights: Optional per-class training weights.
            loss_function: Either ``weighted_ce`` or ``gce``.
            gce_q: Generalized cross-entropy parameter in range (0, 1].

        Raises:
            ValueError: If ``loss_function`` or ``gce_q`` is invalid.
        """
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
        self.gce_q = float(gce_q)
        self._class_weights: torch.Tensor | None
        if class_weights is not None:
            self._class_weights = torch.tensor(
                class_weights, dtype=torch.float32
            )
        else:
            self._class_weights = None

        if self.loss_function not in ("weighted_ce", "gce"):
            raise ValueError(
                "Unsupported loss_function: "
                f"{self.loss_function}. Supported values are "
                "('weighted_ce', 'gce')."
            )
        if not 0.0 < self.gce_q <= 1.0:
            raise ValueError(
                f"gce_q must be in range (0, 1], got {self.gce_q}."
            )

    def _get_class_weights(self, device: torch.device) -> torch.Tensor | None:
        """Return class weights moved to the given device, or None if unset."""
        if self._class_weights is None:
            return None
        if self._class_weights.device != device:
            self._class_weights = self._class_weights.to(device)
        return self._class_weights

    def _compute_weighted_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class-weighted cross-entropy loss."""
        class_weights = self._get_class_weights(logits.device)
        return F.cross_entropy(logits, labels, weight=class_weights)

    def _compute_gce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute generalized cross-entropy with optional class weights."""
        probs = torch.softmax(logits, dim=-1)
        target_probs = probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)
        target_probs = target_probs.clamp_min(1e-8)
        per_example_loss = (1.0 - target_probs.pow(self.gce_q)) / self.gce_q

        class_weights = self._get_class_weights(logits.device)
        if class_weights is None:
            return per_example_loss.mean()

        sample_weights = class_weights.gather(dim=0, index=labels)
        weighted_sum = (per_example_loss * sample_weights).sum()
        weight_norm = sample_weights.sum().clamp_min(1e-8)
        return weighted_sum / weight_norm

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Override default Trainer loss with weighted CE or GCE loss."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_labels = int(logits.size(-1))
        flat_logits = logits.view(-1, num_labels)
        flat_labels = labels.view(-1).long()

        if self.loss_function == "gce":
            loss = self._compute_gce_loss(flat_logits, flat_labels)
        else:
            loss = self._compute_weighted_ce_loss(flat_logits, flat_labels)

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self, model=None):
        """Build AdamW with separate LRs for the head and backbone."""

        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model if model is None else model
        if opt_model is None:
            raise RuntimeError("Optimizer creation requires a model instance.")
        no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")

        head_named_params = [
            (name, param)
            for name, param in opt_model.named_parameters()
            if "classifier" in name
        ]

        base_named_params = [
            (name, param)
            for name, param in opt_model.named_parameters()
            if "classifier" not in name
        ]

        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for name, param in head_named_params
                    if not any(term in name for term in no_decay_terms)
                ],
                "lr": 1e-4,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in head_named_params
                    if any(term in name for term in no_decay_terms)
                ],
                "lr": 1e-4,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    param
                    for name, param in base_named_params
                    if not any(term in name for term in no_decay_terms)
                ],
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in base_named_params
                    if any(term in name for term in no_decay_terms)
                ],
                "lr": self.args.learning_rate,
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

        return self.optimizer


class LogModelTrainer:
    """Orchestrates model initialization and the training loop."""

    def __init__(self, config: TrainerArgs) -> None:
        """Initialize model architecture and evaluation metrics."""
        logger.info(f"Loading model from: {config.model_path}")
        self.config = config

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_path,
            num_labels=len(LogLevel),
            id2label=LogLevel.id2label(),
            label2id=LogLevel.label2id(),
            cache_dir=config.output_dir,
            trust_remote_code=True,
        )

        logger.info("Freezing base model backbone for Epoch 0...")
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        logger.info("Loading evaluation metrics...")
        self.accuracy = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")

    def _compute_metrics(self, eval_pred: EvalPrediction) -> dict[str, float]:
        """Return accuracy and macro F1 for an evaluation batch."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc_result = self.accuracy.compute(
            predictions=predictions,
            references=labels,
        )
        acc = acc_result["accuracy"] if acc_result is not None else 0.0

        f1_result = self.f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="macro",
        )
        f1_score = f1_result["f1"] if f1_result is not None else 0.0

        return {"accuracy": float(acc), "f1": float(f1_score)}

    def _sync_model_tokenizer_vocab(self, tokenizer) -> None:
        """Resize model token embeddings if the tokenizer vocab has grown."""
        model_vocab_size = self.model.get_input_embeddings().num_embeddings
        tokenizer_vocab_size = len(tokenizer)

        if tokenizer_vocab_size == model_vocab_size:
            logger.info(
                "Tokenizer and model vocab already aligned at %d tokens.",
                model_vocab_size,
            )
            return

        logger.info(
            "Resizing model token embeddings from %d to %d.",
            model_vocab_size,
            tokenizer_vocab_size,
        )
        self.model.resize_token_embeddings(tokenizer_vocab_size)

    def _compute_inverse_frequency_class_weights(
        self,
        train_labels: list[int],
    ) -> list[float]:
        """Compute balanced inverse-frequency class weights from labels."""
        num_labels = int(self.model.config.num_labels)
        label_ids = np.asarray(train_labels, dtype=np.int64)
        label_counts = np.bincount(
            label_ids,
            minlength=num_labels,
        )[:num_labels]

        non_zero_mask = label_counts > 0
        if not np.any(non_zero_mask):
            logger.warning(
                "No training labels found; using uniform class weights."
            )
            return [1.0] * num_labels

        weights = np.zeros(num_labels, dtype=np.float32)
        weights[non_zero_mask] = float(label_ids.size) / (
            float(num_labels) * label_counts[non_zero_mask]
        )

        id2label = self.model.config.id2label
        class_distribution = {
            str(id2label.get(i, i)): int(label_counts[i])
            for i in range(num_labels)
        }
        class_weights = {
            str(id2label.get(i, i)): float(weights[i])
            for i in range(num_labels)
        }
        logger.info(
            "Training split class distribution: %s",
            class_distribution,
        )
        logger.info("Inverse-frequency class weights: %s", class_weights)

        return weights.tolist()

    def train(
        self, datasets: DatasetDict, tokenizer
    ) -> tuple[TrainOutput, str]:
        """Configure hyperparameters and execute model training.

        Args:
            datasets: DatasetDict with train/validation/test splits.
            tokenizer: Tokenizer used for collation and persistence.

        Returns:
            Tuple of ``(TrainOutput, save_dir)`` where ``save_dir`` is the
            path of the saved checkpoint.
        """

        self._sync_model_tokenizer_vocab(tokenizer)

        train_len = len(datasets["train"])
        training_args = self.config.get_training_args(
            train_dataset_length=train_len
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        unfreeze_callback = UnfreezeCallback(unfreeze_epoch=1)
        train_labels = datasets["train"]["label"]
        weights = self._compute_inverse_frequency_class_weights(train_labels)

        logger.info(
            f"Starting training on {len(datasets['train'])} instances..."
        )
        logger.info(
            f"Starting training on {len(datasets['validation'])} instances..."
        )
        logger.info(
            "Using loss function: %s (gce_q=%.3f)",
            self.config.loss_function,
            self.config.gce_q,
        )

        trainer = WeightedLossTrainer(
            model=self.model,
            class_weights=weights,
            loss_function=self.config.loss_function,
            gce_q=self.config.gce_q,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            compute_metrics=self._compute_metrics,
            data_collator=data_collator,
            callbacks=[unfreeze_callback],
        )

        model_name: str = (
            "log_classifier_model_"
            + self.config.loss_function
            + "_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        logger.info("Starting training loop...")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        training_output = trainer.train()
        save_dir = self.config.output_dir + "/" + model_name
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        tokenizer.save_pretrained(self.config.output_dir)

        logger.info(
            "Training complete! Saved model and tokenizer to %s",
            save_dir,
        )

        return training_output, save_dir
