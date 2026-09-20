"""M2純増S級source収集ランナーの契約テスト。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts import run_advantage_m2_increment_sources_v1 as target
from scripts import run_event_pilot_48_v1 as event_campaign


def test_fixed_manifest_collapses_aliases_and_balances_s_tier_folds() -> None:
    rows = target.load_manifest(target.DEFAULT_MANIFEST)

    assert tuple(row.target_id for row in rows) == ("c82", "c83")
    assert tuple(row.video_alias for row in rows) == ("video_s1", "video_s2")
    assert tuple(row.historical_alias for row in rows) == ("video_c82", "video_c83")
    assert tuple(row.fold for row in rows) == (1, 3)
    assert len({row.youtube_id for row in rows}) == 2


def test_runner_command_is_full_length_with_fixed_calibration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    row = target.load_manifest(target.DEFAULT_MANIFEST)[0]
    paths = event_campaign.RetryPaths(
        0, "attempt", tmp_path, tmp_path / "work.npz", tmp_path / "collection.log",
        tmp_path / "run.json", tmp_path / "runs", tmp_path / "validation",
    )
    metadata = event_campaign.VideoMetadata(frame_count=9000, fps=30.0)
    monkeypatch.setattr(event_campaign, "probe_video", lambda _: metadata)

    command, actual = target._runner_command(row, paths)

    assert actual == metadata
    assert "--score-region-calibration" in command
    calibration = Path(command[command.index("--score-region-calibration") + 1])
    assert calibration.name == "video_s1_score_region_v1.json"
    assert float(command[command.index("--max-sec") + 1]) > 300.0
    assert command[command.index("--start-sec") + 1] == "0"


def test_campaign_rejects_existing_output_before_any_processing(tmp_path: Path) -> None:
    output = tmp_path / "exists"
    output.mkdir()
    args = argparse.Namespace(
        manifest=target.DEFAULT_MANIFEST,
        parent_manifest=target.DEFAULT_PARENT_MANIFEST,
        output_root=output,
        preflight_only=False,
    )

    with pytest.raises(target.AdvantageM2IncrementCampaignError, match="新規必須"):
        target.run_campaign(args)
