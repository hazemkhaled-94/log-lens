"""Build a balanced training corpus from a raw log collection.

Offline, run-once preprocessing: turns a large, uneven log corpus into
a level-balanced set of ``.log`` files to point ``DATA_DIR`` at. Not
part of the live training run or inference.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from configs import RANDOM_STATE
from data_manager.logs.log_file import LogFile

logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    """Sample a representative subset of a raw log corpus.

    Each first-level directory under the source counts as one dataset
    and gets an equal share of the budget; within a dataset, lines are
    stratified by log level (largest-remainder rounding) and drawn with
    reservoir sampling so memory stays bounded by the budget. Sampled
    lines are shuffled before being written across ``<stem>_NNN.log``
    chunks.
    """

    def __init__(
        self,
        total: int = 200_000,
        lines_per_file: int = 50_000,
        seed: int = 42,
    ) -> None:
        if lines_per_file <= 0:
            raise ValueError("lines_per_file must be positive")
        self._total = total
        self._lines_per_file = lines_per_file
        self._rng = random.Random(seed)

    def build(self, src: Path, out: Path) -> list[Path]:
        """Sample entries from ``src`` and write chunked .log files at ``out``.

        Returns:
            The chunk files written.

        Raises:
            FileNotFoundError: If ``src`` has no usable .json/.log files.
        """
        # Quiet per-file loader logs so sampler progress stays readable.
        logging.getLogger("data_manager.logs.log_file").setLevel(
            logging.WARNING
        )

        files = list(LogFile.iter_files(src))
        if not files:
            raise FileNotFoundError(
                f"No .json or .log files found under {src}"
            )

        # Pass 1: group files into datasets and count their log levels.
        active: list[tuple[str, list[Path], Counter]] = []
        for name, group in self._group_by_dataset(src, files):
            counts = Counter(lvl for _, lvl in self._iter_entries(group))
            if counts:
                active.append((name, group, counts))
            else:
                logger.warning("  %s: no usable entries; skipping", name)
        if not active:
            raise FileNotFoundError("No datasets contributed usable entries.")

        # Give each dataset a near-equal share of the total budget.
        base, extra = divmod(self._total, len(active))
        budgets = [base + (i < extra) for i in range(len(active))]
        logger.info(
            "Sampling %d entries across %d dataset(s)",
            self._total,
            len(active),
        )

        # Pass 2: sample each dataset, then shuffle globally.
        lines: list[str] = []
        for (name, group, counts), budget in zip(active, budgets):
            sampled = self._sample_dataset(group, counts, budget)
            logger.info(
                "  %s: sampled %d / budget %d", name, len(sampled), budget
            )
            lines += sampled
        self._rng.shuffle(lines)

        paths = self._write_chunks(lines, out)
        if len(lines) < self._total:
            logger.warning(
                "Short by %d entries — some datasets had fewer "
                "than their budget.",
                self._total - len(lines),
            )
        return paths

    def _sample_dataset(
        self, files: list[Path], level_counts: Counter, budget: int
    ) -> list[str]:
        """Reservoir-sample up to ``budget`` lines, stratified by level.

        Reservoir sampling keeps memory bounded by the per-level quota
        rather than the (possibly huge) dataset size.
        """
        if budget <= 0:
            return []

        quotas = self._allocate_quotas(level_counts, budget)
        reservoirs: dict[str, list[str]] = {
            lvl: [] for lvl, k in quotas.items() if k > 0
        }
        seen: Counter = Counter()
        for raw, level in self._iter_entries(files):
            k = quotas.get(level, 0)
            if k <= 0:
                continue
            i = seen[level]
            seen[level] += 1
            if i < k:
                reservoirs[level].append(raw)
            elif (j := self._rng.randint(0, i)) < k:
                reservoirs[level][j] = raw
        return [line for slot in reservoirs.values() for line in slot]

    def _write_chunks(self, lines: list[str], out: Path) -> list[Path]:
        """Write ``lines`` across ``<stem>_NNN.log`` chunks.

        A trailing ``.log`` on ``out`` is treated as a stem. Embedded
        newlines are collapsed so each record stays one output line, and
        the index width auto-scales so filenames sort correctly.
        """
        stem = out.with_suffix("") if out.suffix.lower() == ".log" else out
        stem.parent.mkdir(parents=True, exist_ok=True)

        n_chunks = max(1, -(-len(lines) // self._lines_per_file))
        width = max(3, len(str(n_chunks)))

        written: list[Path] = []
        for i in range(n_chunks):
            start = i * self._lines_per_file
            chunk = lines[start:start + self._lines_per_file]
            if not chunk:
                break
            text = "".join(
                raw.replace("\r\n", " ").replace("\n", " ").rstrip() + "\n"
                for raw in chunk
            )
            path = stem.with_name(f"{stem.name}_{i + 1:0{width}d}.log")
            path.write_text(text, encoding="utf-8")
            written.append(path)
            logger.info("  wrote %d lines -> %s", len(chunk), path.name)
        return written

    @staticmethod
    def _iter_entries(files: list[Path]):
        """Yield ``(raw_line, upper_level)`` for every non-blank entry."""
        for path in files:
            for entry in LogFile.from_file(path).entries:
                if entry.raw_line:
                    yield entry.raw_line, entry.line_level.upper()

    @staticmethod
    def _allocate_quotas(level_counts: Counter, budget: int) -> dict[str, int]:
        """Split ``budget`` across levels in proportion to their counts.

        Largest-remainder rounding makes the per-level integers sum
        exactly to ``budget``, capped by what each level can supply.
        """
        total = sum(level_counts.values())
        if not total or not budget:
            return {}

        budget = min(budget, total)
        raw = {lvl: budget * c / total for lvl, c in level_counts.items()}
        quota = {lvl: int(v) for lvl, v in raw.items()}
        ranked = sorted(
            ((raw[lvl] - quota[lvl], lvl) for lvl in quota), reverse=True
        )
        for _, lvl in ranked[: budget - sum(quota.values())]:
            quota[lvl] += 1
        return {lvl: min(k, level_counts[lvl]) for lvl, k in quota.items()}

    @staticmethod
    def _group_by_dataset(
        src: Path, files: list[Path]
    ) -> list[tuple[str, list[Path]]]:
        """Bucket files by their first path component under ``src``.

        A file directly under ``src`` is its own one-file dataset, keyed
        by its stem. Sorted for deterministic ordering.
        """
        groups: dict[str, list[Path]] = defaultdict(list)
        for f in files:
            rel = f.relative_to(src)
            key = rel.parts[0] if len(rel.parts) > 1 else rel.stem
            groups[key].append(f)
        return sorted(groups.items())


def main() -> None:
    """CLI entry point for building a balanced training corpus."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cli = argparse.ArgumentParser(
        description=(
            "Sample a balanced training corpus from a raw log collection."
        )
    )
    cli.add_argument(
        "--src", type=Path, required=True,
        help="Source directory, walked recursively.",
    )
    cli.add_argument(
        "--out", type=Path, required=True,
        help="Output stem; chunks are written as '<stem>_001.log', ...",
    )
    cli.add_argument(
        "--total", type=int, default=200_000,
        help="Total entries to sample (default: 200000).",
    )
    cli.add_argument(
        "--lines-per-file", type=int, default=50_000,
        help="Max lines per output chunk (default: 50000).",
    )
    cli.add_argument(
        "--seed", type=int, default=RANDOM_STATE,
        help="Random seed (default: RANDOM_STATE from env).",
    )
    args = cli.parse_args()

    preprocessor = DatasetPreprocessor(
        args.total, args.lines_per_file, args.seed
    )
    try:
        preprocessor.build(args.src, args.out)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
