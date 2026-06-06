"""Log level enumerations.

This module defines the standard log levels and integer mappings
used across the sequence classification pipeline.
"""

from enum import IntEnum


class LogLevel(IntEnum):
    """Integer enumeration of supported log levels.

    Six canonical levels (TRACE/DEBUG/INFO/WARN/ERROR/FATAL) plus aliases
    covering Android logcat letters, Python WARNING/CRITICAL,
    java.util.logging FINE/SEVERE, syslog, and Go zap PANIC/DPANIC.
    ``id2label`` skips aliases so the label space stays at six classes.
    """

    TRACE = 0
    T = 0
    VERBOSE = 0
    VERB = 0
    V = 0

    DEBUG = 1
    D = 1
    DBG = 1
    FINE = 1
    FINER = 1
    FINEST = 1

    INFO = 2
    I = 2
    NOTICE = 2

    WARN = 3
    WARNING = 3
    W = 3
    WRN = 3

    ERROR = 4
    ERR = 4
    E = 4
    SEVERE = 4

    FATAL = 5
    F = 5
    CRITICAL = 5
    CRIT = 5
    EMERG = 5
    EMERGENCY = 5
    ALERT = 5
    PANIC = 5
    DPANIC = 5

    @classmethod
    def id2label(cls) -> dict[int, str]:
        """Return {int: canonical_name} for Hugging Face label config."""
        return {e.value: e.name for e in cls}

    @classmethod
    def label2id(cls) -> dict[str, int]:
        """Return {name: int} for all members including aliases."""
        return {name: member.value for name, member in cls.__members__.items()}

    @classmethod
    def canonicalize(cls, label: str) -> str:
        """Resolve an alias or short-form to its canonical level name.

        Unknown labels are returned upper-cased so sentinels like
        ``UNKNOWN`` pass through untouched.
        """
        normalized = label.upper().strip()
        member = cls.__members__.get(normalized)
        if member is None:
            return normalized
        return member.name
