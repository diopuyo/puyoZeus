"""M2純増sourceのquarantine再検証テスト。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts import revalidate_advantage_m2_increment_sources_v1 as target
from scripts import run_advantage_m2_increment_sources_v1 as campaign


def test_retry_paths_reuse_existing_attempt_without_creating(tmp_path: Path) -> None:
    row = campaign.load_manifest(campaign.DEFAULT_MANIFEST)[0]

    paths = target._retry_paths(tmp_path, row)

    assert paths.retry_index == 0
    assert paths.attempt_id.endswith("-c82")
    assert paths.run_result.name == "run.json"
    assert not paths.root.exists()


def test_revalidation_rejects_existing_output_before_reading_source(
    tmp_path: Path,
) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    args = argparse.Namespace(
        source_root=tmp_path / "missing", output_root=output,
        manifest=campaign.DEFAULT_MANIFEST,
        parent_manifest=campaign.DEFAULT_PARENT_MANIFEST,
    )

    with pytest.raises(target.IncrementRevalidationError, match="新規必須"):
        target.run_revalidation(args)


def test_safe_revalidate_records_failure_without_hiding_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    row = campaign.load_manifest(campaign.DEFAULT_MANIFEST)[1]

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise target.IncrementRevalidationError("人工FAIL")

    monkeypatch.setattr(target, "_revalidate_one", fail)
    result = target._safe_revalidate(tmp_path, row, lambda *_args: 0)

    assert result["state"] == "failed"
    assert result["target_id"] == "c83"
    assert "人工FAIL" in str(result["error"])
