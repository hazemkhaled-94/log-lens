"""Build a representative subset of a log corpus into chunked .log files.

Walks a source directory recursively (consuming both .json and .log
files via the framework's LogFile loader) and writes up to N total
entries out as one-line-per-record, split across multiple .log files
of a configurable size.

Sampling unit: dataset
  A "dataset" is the first path component under ``--src``. A directory
  immediately under ``--src`` is one dataset, regardless of how many
  files it fans out into; a single ``.log`` or ``.json`` file directly
  under ``--src`` is its own one-file dataset. This prevents corpora
  that ship a single source as thousands of per-application files
  (Spark, Hadoop) from drowning out single-file corpora (Apache,
  Thunderbird, ...).

Sampling rules:
  * Each dataset receives an equal share of the total budget.
  * Within each dataset, samples are stratified by that dataset's
    aggregate log-level distribution. The per-level quota uses
    largest-remainder rounding so the integer counts sum exactly to
    the dataset budget — every level appears in the output at the
    same proportion (modulo at most a one-record rounding error per
    level) as in the original dataset.
  * Per-level sampling within a dataset uses reservoir sampling, so
    memory is bounded by the dataset budget rather than the dataset
    size, even for huge sources like Spark.
  * All sampled entries are globally shuffled before being written so
    each output chunk is itself a representative slice of the whole
    sample (rather than a contiguous block from one dataset).

Output naming:
  ``--out`` is treated as a stem. A trailing ``.log`` is stripped, and
  chunks are written as ``<stem>_001.log``, ``<stem>_002.log``, ...

Usage::

    python -m tests.sample_logs \\
        --src resources/datasets/loghub \\
        --out resources/datasets/sample.log \\
        --total 200000 \\
        --lines-per-file 50000
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from data_manager.logs.log_file import LogFile

logger = logging.getLogger(__name__)


def _allocate_quotas(
    level_counts: Counter, budget: int
) -> Dict[str, int]:
    """Allocate ``budget`` across levels in proportion to their counts.

    Uses largest-remainder rounding so the per-level integers sum
    exactly to ``budget`` (capped by what's available per level).
    """
    total = sum(level_counts.values())
    if total == 0 or budget == 0:
        return {}

    budget = min(budget, total)
    raw = {lvl: budget * c / total for lvl, c in level_counts.items()}
    floored = {lvl: int(v) for lvl, v in raw.items()}
    remainder = budget - sum(floored.values())

    fractions = sorted(
        ((raw[lvl] - floored[lvl], lvl) for lvl in floored),
        reverse=True,
    )
    for _, lvl in fractions[:remainder]:
        floored[lvl] += 1

    for lvl, count in level_counts.items():
        if floored[lvl] > count:
            floored[lvl] = count
    return floored


def _group_by_dataset(
    src: Path, files: List[Path]
) -> List[Tuple[str, List[Path]]]:
    """Bucket files by the first path component under ``src``.

    Returns a deterministic alphabetically-sorted list of
    ``(dataset_name, files)`` tuples. A file directly under ``src``
    forms its own one-file dataset keyed by its stem.
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        rel = f.relative_to(src)
        key = rel.parts[0] if len(rel.parts) > 1 else rel.stem
        groups[key].append(f)
    return sorted(groups.items())


def _iter_dataset_entries(files: List[Path]):
    """Yield ``(raw_line, level_upper)`` for every non-blank entry."""
    for path in files:
        log_file = LogFile.from_file(path)
        for entry in log_file.entries:
            if entry.raw_line:
                yield entry.raw_line, entry.line_level.upper()


def _count_dataset_levels(files: List[Path]) -> Counter:
    """Stream every file once and tally log levels."""
    counts: Counter = Counter()
    for _, level in _iter_dataset_entries(files):
        counts[level] += 1
    return counts


def _reservoir_sample_per_level(
    files: List[Path],
    quotas: Dict[str, int],
    rng: random.Random,
) -> List[str]:
    """Pass 2: stream every file once and reservoir-sample per level.

    Memory is bounded by ``sum(quotas.values())``: at most one
    reservoir slot per ultimately-emitted line.
    """
    reservoirs: Dict[str, List[str]] = {
        lvl: [] for lvl, k in quotas.items() if k > 0
    }
    seen: Counter = Counter()
    for raw, level in _iter_dataset_entries(files):
        k = quotas.get(level, 0)
        if k <= 0:
            continue
        i = seen[level]
        seen[level] += 1
        if i < k:
            reservoirs[level].append(raw)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoirs[level][j] = raw

    out: List[str] = []
    for lines in reservoirs.values():
        out.extend(lines)
    return out


def _sample_dataset(
    name: str,
    files: List[Path],
    level_counts: Counter,
    budget: int,
    rng: random.Random,
) -> List[str]:
    """Reservoir-sample ``budget`` raw lines from a dataset."""
    if budget <= 0 or not level_counts:
        return []

    quotas = _allocate_quotas(level_counts, budget)
    sampled = _reservoir_sample_per_level(files, quotas, rng)

    total_in = sum(level_counts.values())
    logger.info(
        f"  [{name}] files={len(files)} total={total_in} "
        f"budget={budget} sampled={len(sampled)}"
    )
    for lvl in sorted(level_counts):
        in_pct = 100.0 * level_counts[lvl] / total_in
        out_pct = (
            100.0 * quotas.get(lvl, 0) / max(len(sampled), 1)
        )
        logger.info(
            f"      {lvl:>10}: in={level_counts[lvl]:>9} "
            f"({in_pct:5.2f}%) -> out={quotas.get(lvl, 0):>6} "
            f"({out_pct:5.2f}%)"
        )
    return sampled


def _normalize(line: str) -> str:
    """Collapse embedded newlines so each record stays one output line."""
    return line.replace("\r\n", " ").replace("\n", " ").rstrip()


def _resolve_chunk_stem(out: Path) -> Path:
    """Strip a trailing ``.log`` from ``out`` so it can be used as a stem."""
    if out.suffix.lower() == ".log":
        return out.with_suffix("")
    return out


def _write_chunks(
    lines: List[str], stem: Path, lines_per_file: int
) -> List[Path]:
    """Write ``lines`` across ``<stem>_NNN.log`` chunks.

    The chunk index width auto-scales to the total number of chunks so
    filenames sort correctly in directory listings.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = max(1, (len(lines) + lines_per_file - 1) // lines_per_file)
    width = max(3, len(str(n_chunks)))

    written_paths: List[Path] = []
    for chunk_idx in range(n_chunks):
        start = chunk_idx * lines_per_file
        end = start + lines_per_file
        chunk = lines[start:end]
        if not chunk:
            break
        path = stem.with_name(
            f"{stem.name}_{chunk_idx + 1:0{width}d}.log"
        )
        with path.open("w", encoding="utf-8") as fh:
            fh.writelines(chunk)
        written_paths.append(path)
        logger.info(
            f"  wrote {len(chunk):>7} lines -> {path.name}"
        )
    return written_paths


def sample_corpus(
    src: Path,
    out: Path,
    total: int,
    seed: int,
    lines_per_file: int,
) -> None:
    """Walk ``src``, sample ``total`` entries, write to chunked files."""
    if lines_per_file <= 0:
        logger.error("--lines-per-file must be positive")
        sys.exit(2)

    rng = random.Random(seed)

    # Silence per-file INFO logs from the framework so sampler progress stays readable.
    logging.getLogger("data_manager.logs.log_file").setLevel(
        logging.WARNING
    )

    files = list(LogFile.iter_files(src))
    if not files:
        logger.error(f"No .json or .log files found under {src}")
        sys.exit(1)

    datasets = _group_by_dataset(src, files)
    stem = _resolve_chunk_stem(out)
    logger.info(
        f"Discovered {len(datasets)} dataset(s) under {src}; "
        f"counting log entries (pass 1) ..."
    )

    # Pass 1: count levels per dataset; filter empty ones before budget allocation.
    active: List[Tuple[str, List[Path], Counter]] = []
    for name, group_files in datasets:
        counts = _count_dataset_levels(group_files)
        if counts:
            active.append((name, group_files, counts))
            logger.info(
                f"  {name}: {len(group_files)} file(s), "
                f"{sum(counts.values())} entries"
            )
        else:
            logger.warning(
                f"  {name}: no usable entries; skipping"
            )

    if not active:
        logger.error("No datasets contributed any usable log entries.")
        sys.exit(1)

    n_active = len(active)
    per_dataset = total // n_active
    leftover = total - per_dataset * n_active
    dataset_budgets = [
        per_dataset + (1 if i < leftover else 0)
        for i in range(n_active)
    ]

    logger.info(
        f"Sampling {total} entries from {len(files)} files "
        f"across {n_active} dataset(s) (~{per_dataset} per dataset) "
        f"into {stem.parent}/{stem.name}_*.log"
    )

    all_lines: List[str] = []
    for (name, group_files, counts), budget in zip(active, dataset_budgets):
        sampled = _sample_dataset(name, group_files, counts, budget, rng)
        for line in sampled:
            all_lines.append(_normalize(line) + "\n")

    rng.shuffle(all_lines)
    logger.info(
        f"Globally shuffled {len(all_lines)} entries; "
        f"writing chunks of {lines_per_file} lines"
    )

    paths = _write_chunks(all_lines, stem, lines_per_file)

    logger.info(
        f"Done. Wrote {len(all_lines)} lines across {len(paths)} chunk(s)."
    )
    if len(all_lines) < total:
        shortfall = total - len(all_lines)
        logger.warning(
            f"Output is short by {shortfall} entries — some datasets had "
            "fewer entries than their per-dataset budget."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Sample a representative subset of a log corpus into "
            "chunked .log files."
        )
    )
    parser.add_argument(
        "--src", type=Path, required=True,
        help="Source directory (walked recursively).",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help=(
            "Output stem. A trailing '.log' is stripped; chunks are "
            "written as '<stem>_001.log', '<stem>_002.log', ..."
        ),
    )
    parser.add_argument(
        "--total", type=int, default=200000,
        help="Total number of log entries to sample (default: 200000).",
    )
    parser.add_argument(
        "--lines-per-file", type=int, default=50000,
        help="Maximum lines per output chunk (default: 50000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()
    sample_corpus(
        args.src, args.out, args.total, args.seed, args.lines_per_file
    )


if __name__ == "__main__":
    main()
