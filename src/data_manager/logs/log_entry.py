"""Parsing and structuring of log entries.

Supports JSON dicts (Loki/Grafana shape with 'line', 'timestamp', 'fields')
and plain ``.log`` files where each line is a single record. Level/message
extraction is via regex; value masking is left to Drain3.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


LOG_PATTERN = re.compile(
    r"^"
    r"(?:.*?(?P<line_timestamp>\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d+)?\s*(?:[+-]\d{4}|Z)?))?"
    r".*?"
    # Covers Java/Logback/Log4j, Python, java.util.logging, syslog,
    # Go zap, and Android logcat single-letter levels.
    r"\b(?P<line_level>"
    r"TRACE|VERBOSE|VERB|"
    r"DEBUG|DBG|FINEST|FINER|FINE|"
    r"WARNING|WARN|WRN|"
    r"ERROR|SEVERE|ERR|"
    r"CRITICAL|CRIT|EMERGENCY|EMERG|DPANIC|PANIC|FATAL|ALERT|"
    r"NOTICE|INFO|"
    r"[TVDIWEF]"
    r")\b"
    r".*?"
    r"(?:---|:|\s-\s|\])\s*"
    r"(?P<message>.*)$",
    re.IGNORECASE
)


# Fallback for formats that put the level inside a header without a
# trailing delimiter (e.g. BGL "... RAS KERNEL INFO message text").
# Uppercase-only, no single-letter levels, and preceded by a
# header-shaped token to avoid false positives in prose.
LOG_PATTERN_LOOSE = re.compile(
    r"^"
    r"(?:.*?(?P<line_timestamp>\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d+)?\s*(?:[+-]\d{4}|Z)?))?"
    r".*?"
    r"(?:^|\s)[A-Z0-9][A-Z0-9_:\-./]+\s+"
    r"(?P<line_level>"
    r"TRACE|VERBOSE|VERB|"
    r"DEBUG|DBG|FINEST|FINER|FINE|"
    r"WARNING|WARN|WRN|"
    r"ERROR|SEVERE|ERR|"
    r"CRITICAL|CRIT|EMERGENCY|EMERG|DPANIC|PANIC|FATAL|ALERT|"
    r"NOTICE|INFO"
    r")"
    r"\s+(?P<message>.*)$"
)


@dataclass
class LogMetadata:
    """Infrastructure metadata from JSON 'fields'."""
    app: str = ""
    container: str = ""
    detected_level: str = ""
    filename: str = ""
    job: str = ""
    namespace: str = ""
    node_name: str = ""
    pod: str = ""
    service_name: str = ""
    stream: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogMetadata":
        """Create a LogMetadata from a dict, ignoring unknown keys."""
        if not data:
            return cls()
        valid_data = {
            k: v for k, v in data.items()
            if k in cls.__annotations__
        }
        return cls(**valid_data)


@dataclass
class LogEntry:
    """Fully parsed log entry with metadata and application data.

    Accepts either a JSON dict (Loki/Grafana shape) via ``raw_json_dict``
    or a plain log line via ``raw_line`` / :meth:`from_line`. When built
    from a raw line, ``log_timestamp`` and ``metadata`` are left empty.
    """
    raw_json_dict: Optional[Dict[str, Any]] = field(default=None, repr=False)
    raw_line: str = ""

    log_timestamp: str = field(init=False, default="")
    metadata: LogMetadata = field(init=False, default_factory=LogMetadata)

    line_timestamp: str = field(init=False, default="")
    line_level: str = field(init=False, default="")
    message: str = field(init=False, default="")

    def __post_init__(self) -> None:
        """Build from a JSON dict: extract fields, then parse the line."""
        if self.raw_json_dict is not None:
            self.raw_line = self.raw_json_dict.get("line", "")
            self.log_timestamp = self.raw_json_dict.get("timestamp", "")

            fields_dict = self.raw_json_dict.get("fields", {})
            self.metadata = LogMetadata.from_dict(fields_dict)

        self._parse_line()

    @classmethod
    def from_line(cls, raw_line: str) -> "LogEntry":
        """Build a LogEntry from a single plain-text log line."""
        return cls(raw_line=raw_line)

    def _parse_line(self) -> None:
        """Extract level, timestamp, and message from raw_line via regex."""
        match = LOG_PATTERN.search(self.raw_line)
        if not match:
            match = LOG_PATTERN_LOOSE.search(self.raw_line)

        if match:
            groups = match.groupdict()
            self.line_timestamp = groups.get("line_timestamp") or ""
            self.line_level = groups.get("line_level", "")
            self.message = (groups.get("message") or "").strip()
        else:
            self.message = self.raw_line.strip()

        if not self.line_level and self.metadata.detected_level:
            self.line_level = self.metadata.detected_level.upper()

        if not self.line_level:
            self.line_level = "UNKNOWN"

        if not match and self.line_level != "UNKNOWN":
            leak_pattern = re.compile(
                rf"^.*?\[?{self.line_level}\]?\s*(?:-|:|\])?\s*",
                re.IGNORECASE,
            )
            self.message = leak_pattern.sub("", self.message).strip()
