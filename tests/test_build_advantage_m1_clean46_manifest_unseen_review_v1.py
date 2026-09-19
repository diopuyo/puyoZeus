from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import build_advantage_m1_clean46_manifest_unseen_review_v1 as review


def test_log_loss_rewards_correct_probability() -> None:
    labels = np.asarray([1.0, 0.0])
    assert review._log_loss(labels, np.asarray([0.9, 0.1])) < review._log_loss(
        labels, np.asarray([0.6, 0.4]),
    )


def test_prior_review_manifest_matches_only_source_id(tmp_path: Path) -> None:
    (tmp_path / "a.manifest.json").write_text(
        json.dumps({"source_video_id": review.VIDEO_ID}), encoding="utf-8",
    )
    (tmp_path / "b.manifest.json").write_text(
        json.dumps({"source_video_id": "other"}), encoding="utf-8",
    )
    assert review._prior_review_manifests(tmp_path) == [
        str((tmp_path / "a.manifest.json").resolve()),
    ]


def test_m0_statistics_is_clean46_fold4(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = SimpleNamespace(
        source_groups=np.asarray([review.SOURCE_GROUP_ID, "other"]),
        game_keys=np.asarray([review.GAME_KEY, "other:game-0001"]),
        folds=np.asarray([4, 1]), labels=np.asarray([1.0, 0.0]),
    )
    predictions = {
        "m0__seed_ensemble__calibrated": np.asarray([0.6, 0.4]),
        "m1_zero_values_and_masks__seed_ensemble__calibrated": np.asarray([0.7, 0.3]),
    }
    artifact = SimpleNamespace(receipt={"training_root": "fixed"})
    monkeypatch.setattr(review.single, "_load_analysis_dataset", lambda _root: (samples, {}))
    monkeypatch.setattr(review.comparison, "_ensemble", lambda *_args: ((artifact,), predictions))
    result = review._clean46_m0_statistics()
    assert result["state_count"] == 1
    assert result["eval_fold"] == 4
    assert result["m1_minus_m0_log_loss"] < 0.0
    assert result["label_use"] == "offline_report_only_not_display_path"


def test_render_command_uses_complete_game_and_resize() -> None:
    command = review._render_command()
    assert command[command.index("--start-game-position") + 1] == "2"
    assert command[command.index("--end-game-position") + 1] == "2"
    assert "--resize-1080p" in command
    assert "--event-source-run" in command
    assert command[command.index("--frame-stride") + 1] == "1"


def test_builder_uses_new_v3_roots_and_keeps_prior_attempts() -> None:
    assert str(review.DELIVERY).endswith("_v3")
    assert str(review.VERIFY).endswith("_v3")
    assert [str(path)[-3:] for path in review.SUPERSEDED_DELIVERIES] == ["_v1", "_v2"]


def test_validate_render_fails_on_wrong_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        review.media, "_video_properties",
        lambda _path: {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": 891},
    )
    monkeypatch.setattr(review.media, "_audio_has_packets", lambda _path: True)
    manifest = {
        "audio_muxed": True,
        "game_segment": {
            "start_game_key": review.GAME_KEY, "start_position": 2, "end_position": 2,
            "resolved_start_sec": 171.3, "resolved_end_sec": 200.0,
            "exact_start_frame_inclusive": review.EXPECTED_START_FRAME,
            "exact_end_frame_exclusive": review.EXPECTED_END_FRAME_EXCLUSIVE,
            "exact_source_frame_count": review.EXPECTED_SOURCE_FRAME_COUNT,
            "exact_frame_source": "event_source_formal_exact/v1",
        },
        "stats": {
            "terminal_outcomes": [{"terminal_loser": "p2"}],
            "source_start_frame": review.EXPECTED_START_FRAME,
            "source_end_frame_exclusive": review.EXPECTED_END_FRAME_EXCLUSIVE,
            "source_frame_count": review.EXPECTED_SOURCE_FRAME_COUNT,
        },
        "true_oof_prediction_source": {"true_oof": True, "model_count": 3},
    }
    with pytest.raises(review.ManifestUnseenReviewError, match="全試合検収失敗"):
        review._validate_render(manifest)


def test_validate_render_accepts_exact_891_source_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        review.media, "_video_properties",
        lambda _path: {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": 891},
    )
    monkeypatch.setattr(review.media, "_audio_has_packets", lambda _path: True)
    monkeypatch.setattr(review.media, "_decode_entire_media", lambda _path: {"decode": "ok"})
    manifest = {
        "audio_muxed": True,
        "game_segment": {
            "start_game_key": review.GAME_KEY, "start_position": 2, "end_position": 2,
            "resolved_start_sec": review.EXPECTED_START_SEC,
            "resolved_end_sec": review.EXPECTED_END_SEC,
            "exact_start_frame_inclusive": review.EXPECTED_START_FRAME,
            "exact_end_frame_exclusive": review.EXPECTED_END_FRAME_EXCLUSIVE,
            "exact_source_frame_count": review.EXPECTED_SOURCE_FRAME_COUNT,
            "exact_frame_source": "event_source_formal_exact/v1",
        },
        "stats": {
            "terminal_outcomes": [{"terminal_loser": "p2"}],
            "source_start_frame": review.EXPECTED_START_FRAME,
            "source_end_frame_exclusive": review.EXPECTED_END_FRAME_EXCLUSIVE,
            "source_frame_count": review.EXPECTED_SOURCE_FRAME_COUNT,
        },
        "true_oof_prediction_source": {"true_oof": True, "model_count": 3},
    }

    result = review._validate_render(manifest)

    assert all(result["checks"].values())
    assert result["decode"] == "ok"
