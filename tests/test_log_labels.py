"""Tests for log-level enumeration and canonicalization."""

from data_manager.logs.log_labels import LogLevel


def test_canonicalize_resolves_aliases():
    assert LogLevel.canonicalize("I") == "INFO"
    assert LogLevel.canonicalize("warning") == "WARN"
    assert LogLevel.canonicalize("dbg") == "DEBUG"
    assert LogLevel.canonicalize("CRITICAL") == "FATAL"


def test_canonicalize_passes_through_unknown_uppercased():
    assert LogLevel.canonicalize("weird") == "WEIRD"
    assert LogLevel.canonicalize("unknown") == "UNKNOWN"


def test_id2label_is_the_six_canonical_levels():
    id2label = LogLevel.id2label()
    assert id2label == {
        0: "TRACE",
        1: "DEBUG",
        2: "INFO",
        3: "WARN",
        4: "ERROR",
        5: "FATAL",
    }


def test_label2id_includes_aliases():
    label2id = LogLevel.label2id()
    assert label2id["INFO"] == 2
    assert label2id["I"] == 2
    assert label2id["WARNING"] == 3
