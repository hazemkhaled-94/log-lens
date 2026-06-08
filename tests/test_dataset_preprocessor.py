"""Tests for the pure sampling helpers of DatasetPreprocessor."""

from collections import Counter
from pathlib import Path

from data_manager.logs.dataset_preprocessor import DatasetPreprocessor


def test_allocate_quotas_preserves_proportions_exactly():
    counts = Counter({"INFO": 70, "ERROR": 30})
    quotas = DatasetPreprocessor._allocate_quotas(counts, budget=100)
    assert quotas == {"INFO": 70, "ERROR": 30}


def test_allocate_quotas_sums_to_budget_with_rounding():
    counts = Counter({"INFO": 7, "ERROR": 3})
    quotas = DatasetPreprocessor._allocate_quotas(counts, budget=5)
    # Largest-remainder rounding must sum exactly to the budget.
    assert sum(quotas.values()) == 5


def test_allocate_quotas_never_exceeds_available_per_level():
    counts = Counter({"RARE": 1, "COMMON": 100})
    quotas = DatasetPreprocessor._allocate_quotas(counts, budget=50)
    for level, count in counts.items():
        assert quotas[level] <= count


def test_allocate_quotas_empty_inputs():
    assert DatasetPreprocessor._allocate_quotas(Counter(), budget=10) == {}
    assert (
        DatasetPreprocessor._allocate_quotas(Counter({"INFO": 5}), budget=0)
        == {}
    )


def test_group_by_dataset_buckets_by_first_component():
    src = Path("/corpus")
    files = [
        Path("/corpus/Hadoop/a.log"),
        Path("/corpus/Hadoop/b.log"),
        Path("/corpus/app.log"),
    ]
    groups = DatasetPreprocessor._group_by_dataset(src, files)
    names = [name for name, _ in groups]
    assert names == ["Hadoop", "app"]  # sorted, single file keyed by stem
    hadoop_files = dict(groups)["Hadoop"]
    assert len(hadoop_files) == 2
