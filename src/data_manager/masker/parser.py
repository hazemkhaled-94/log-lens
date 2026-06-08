# mypy: disable-error-code=import-untyped

"""Low-level Drain3 parser wrapper used by the masking pipeline."""

from drain3 import TemplateMiner  # type: ignore
from drain3.file_persistence import FilePersistence  # type: ignore

from configs import Drain3Config
from data_manager.masker.template_results import TestResult, TrainResult


class Drain3Parser:
    """Low-level wrapper around Drain3 TemplateMiner with typed results."""

    def __init__(self, config: Drain3Config) -> None:
        """Create a TemplateMiner from the given config."""
        persistence = FilePersistence(config.state_file)
        self._miner = TemplateMiner(persistence, config.build_miner_config())

    def train(self, message: str) -> TrainResult:
        """Update the parse tree with a new log message."""
        raw = self._miner.add_log_message(message)
        return TrainResult(
            template=raw["template_mined"],
            cluster_id=raw["cluster_id"],
            is_new_cluster=raw["change_type"] == "cluster_created",
        )

    def match(self, message: str) -> TestResult:
        """Match a message against the frozen parse tree (read-only)."""
        masked = self._miner.masker.mask(message)
        cluster = self._miner.match(message)
        if cluster:
            return TestResult(
                template=cluster.get_template(),
                matched=True,
                masked_message=masked,
            )
        return TestResult(
            template="<UNKNOWN_FORMAT>",
            matched=False,
            masked_message=masked,
        )

    @property
    def cluster_count(self) -> int:
        """Total number of clusters in the parse tree."""
        return len(self._miner.drain.clusters)
