"""Log file management and parsing utilities.

Handles both Loki/Grafana JSON dumps (``.json``) and plain text log
files (``.log``) where each line is a single log record. Directory
walks recurse into every subfolder and dispatch by file suffix.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from .log_entry import LogEntry


logger = logging.getLogger(__name__)


_JSON_SUFFIX = ".json"
_LOG_SUFFIX = ".log"
_SUPPORTED_SUFFIXES = (_JSON_SUFFIX, _LOG_SUFFIX)


@dataclass
class LogFile:
    """Collection of LogEntry objects loaded from one or more log files."""
    entries: List[LogEntry] = field(default_factory=list)

    @classmethod
    def from_json_string(cls, json_data: str) -> "LogFile":
        """Create a LogFile from a JSON string containing an array of records.

        Non-list payloads (e.g. index files sharing the .json suffix) are
        skipped with a warning.

        Raises:
            json.JSONDecodeError: If the JSON data is malformed.
        """
        data = json.loads(json_data)
        if not isinstance(data, list):
            logger.warning(
                "Expected a JSON array of log records, got %s; skipping.",
                type(data).__name__,
            )
            return cls()

        entries = [
            LogEntry(raw_json_dict=item)
            for item in data
            if isinstance(item, dict)
        ]
        if not entries and data:
            logger.warning(
                "JSON payload contained %d items but none were log "
                "record objects; skipping.",
                len(data),
            )
        elif len(entries) < len(data):
            logger.warning(
                "Dropped %d non-object items from JSON payload.",
                len(data) - len(entries),
            )
        return cls(entries=entries)

    @classmethod
    def from_log_string(cls, text: str) -> "LogFile":
        """Create a LogFile from plain-text content (one entry per line)."""
        entries = [
            LogEntry.from_line(line)
            for line in text.splitlines()
            if line.strip()
        ]
        return cls(entries=entries)

    @classmethod
    def from_file(cls, file_path: Path) -> "LogFile":
        """Read a single log file and return a LogFile.

        Dispatches by suffix: .json as a JSON array, .log line-by-line.
        Returns an empty LogFile on read errors or unsupported suffixes.
        """
        logger.info(f"Reading {file_path.name}")
        suffix = file_path.suffix.lower()

        if suffix == _JSON_SUFFIX:
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = cls.from_json_string(json.dumps(data)).entries
                    return cls(entries=entries)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {file_path.name}: {e}")
                return cls()

        if suffix == _LOG_SUFFIX:
            try:
                with file_path.open(
                    "r", encoding="utf-8", errors="replace"
                ) as f:
                    return cls.from_log_string(f.read())
            except OSError as e:
                logger.error(f"Could not read {file_path.name}: {e}")
                return cls()

        logger.warning(f"Unsupported file type: {file_path.name}")
        return cls()

    @classmethod
    def iter_files(
        cls,
        data_dir: Path,
        pattern: Optional[re.Pattern] = None,
    ) -> Iterator[Path]:
        """Recursively yield .json and .log files under data_dir.

        Args:
            data_dir: Root directory to walk.
            pattern: Optional compiled regex filtered against filenames.

        Raises:
            FileNotFoundError: If data_dir does not exist.
        """
        if not data_dir.exists() or not data_dir.is_dir():
            logger.error(f"Directory not found at {data_dir}")
            raise FileNotFoundError(f"Directory not found at {data_dir}")

        for file_path in sorted(data_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            if pattern is not None and not pattern.search(file_path.name):
                continue
            yield file_path

    @classmethod
    def from_directory(
        cls,
        data_dir: Path,
        pattern: Optional[re.Pattern] = None,
    ) -> "LogFile":
        """Aggregate all entries from every supported log file under data_dir.

        Raises:
            FileNotFoundError: If data_dir does not exist.
        """
        if not data_dir.exists() or not data_dir.is_dir():
            logger.error(f"Directory not found at {data_dir}")
            raise FileNotFoundError(f"Directory not found at {data_dir}")

        logger.info(
            f"Recursively scanning {data_dir} for .json and .log files..."
        )

        all_entries: List[LogEntry] = []
        files_processed = 0

        for file_path in cls.iter_files(data_dir, pattern):
            log_file = cls.from_file(file_path)
            all_entries.extend(log_file.entries)
            if log_file.entries:
                files_processed += 1

        logger.info(
            f"Successfully loaded {len(all_entries)} "
            f"log entries from {files_processed} files."
        )

        return cls(entries=all_entries)

    @classmethod
    def yield_from_directory(
        cls,
        data_dir: Path,
        pattern: Optional[re.Pattern] = None,
    ) -> Iterator[LogEntry]:
        """Yield entries one at a time to avoid loading everything into memory.

        Raises:
            FileNotFoundError: If data_dir does not exist.
        """
        if not data_dir.exists() or not data_dir.is_dir():
            logger.error(f"Directory not found at {data_dir}")
            raise FileNotFoundError(f"Directory not found at {data_dir}")

        logger.info(
            f"Recursively scanning {data_dir} for .json and .log files..."
        )

        files_processed = 0
        for file_path in cls.iter_files(data_dir, pattern):
            log_file = cls.from_file(file_path)
            yield from log_file.entries
            if log_file.entries:
                files_processed += 1

        logger.info(f"Successfully processed {files_processed} files.")

    def get_filtered_data(
        self,
        level: str = "",
        include_unknown: bool = True
    ) -> tuple[list[str], list[str]]:
        """Return (messages, levels) lists filtered by level.

        Args:
            level: Level to filter by; empty string returns all levels.
            include_unknown: If False, drops entries with no level.

        Returns:
            A tuple of (messages_list, levels_list) of equal length.
        """
        target = level.upper() if level else None
        messages = []
        levels = []

        for entry in self.entries:
            if not entry.message:
                continue

            lvl = entry.line_level
            lvl_upper = lvl.upper() if lvl else "UNKNOWN"

            if not include_unknown and lvl_upper == "UNKNOWN":
                continue

            if target and lvl_upper != target:
                continue

            messages.append(entry.message)
            levels.append(lvl_upper)

        return messages, levels
