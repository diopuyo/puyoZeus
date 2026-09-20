"""採用台帳の追記と物理定数の変更を区別し、旧来歴の偽装を拒否する。"""
from pathlib import Path
import hashlib

import pytest

from scripts.production_dependency_contract import (
    HISTORICAL_SHA256, dependency_receipt, production_compatible, saved_dependency_compatible,
)


@pytest.mark.parametrize("source, expected", [
    ("GHOST_CHAIN_RULE_ENABLED = True\n", True),
    ("GHOST_CHAIN_RULE_ENABLED = True\nNEW_FLAG = False\n", True),
    ("GHOST_CHAIN_RULE_ENABLED = False\n", False),
    ("GHOST_CHAIN_RULE_ENABLED = 1\n", False),
    ("OTHER = True\n", False),
    ("GHOST_CHAIN_RULE_ENABLED = True\nGHOST_CHAIN_RULE_ENABLED = False\n", False),
    ("GHOST_CHAIN_RULE_ENABLED = True\nif True:\n GHOST_CHAIN_RULE_ENABLED = False\n", False),
])
def test_dependency_values_not_whole_ledger(tmp_path: Path, source: str, expected: bool) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/production_config.py").write_text(source, encoding="utf-8")
    assert production_compatible(tmp_path) is expected
    receipt = dependency_receipt(tmp_path)
    assert receipt["unchanged"] is False
    assert saved_dependency_compatible(receipt["actual_sha256"], receipt) is expected


def test_unknown_saved_hash_requires_bound_semantic_receipt() -> None:
    assert saved_dependency_compatible(HISTORICAL_SHA256)
    assert not saved_dependency_compatible("a" * 64)
    receipt = dependency_receipt()
    assert not saved_dependency_compatible("a" * 64, receipt)
    receipt["actual_sha256"] = "a" * 64
    assert not saved_dependency_compatible("a" * 64, receipt)


def test_saved_integer_one_cannot_impersonate_boolean_true() -> None:
    receipt = dependency_receipt()
    source = "GHOST_CHAIN_RULE_ENABLED = 1\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    receipt.update(source_text=source, actual_sha256=digest)
    assert not saved_dependency_compatible(digest, receipt)
