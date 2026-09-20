"""clean46 championをmanifest未登場のマスター級1試合で通し検収する。"""

from __future__ import annotations

from scripts.production_dependency_contract import (
    dependency_receipt, production_compatible, saved_dependency_compatible,
)

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import analyze_advantage_m1_fixed_oof_v1 as single
from scripts import compare_advantage_m1_46v_48v_common_v1 as comparison
from scripts import run_yamada_norasuke_first20_old_v11_vs_m1_v3_v1 as media
from scripts import train_advantage_m0_current_cnn_v1 as base


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "venv/bin/python"
VIDEO_ID = "video_c80"
SOURCE_GROUP_ID = "6asLcTglLQI"
GAME_KEY = "video_c80:game-0003"
SOURCE = Path("data/frames/video_c80.mp4")
SOURCE_SHA256 = "97b6671bb0ea4e34eabb31ac8ff94ea30a7a9cbc153d4e4447333d90ffb0399c"
PRODUCTION_CONFIG_SHA256 = "3fe3c2578b3196a60cffffcafa594c1f64d2a913bb0487932ac5cd8f6f86d376"
DATASET = Path("data/verify/advantage_m1_canonical_dataset_46v_2026-09-06_v4_quarantine_clean")
TRAINING_ROOTS = (
    Path("data/verify/advantage_m1_zero_counterfactual_46v_clean_2026-09-06_v1_allfolds_seed20260904_merged"),
    Path("data/verify/advantage_m1_zero_counterfactual_46v_clean_2026-09-06_v1_allfolds_seeds20260905_20260906_merged"),
)
PREDICTION_ROOT = Path("data/verify/advantage_m1_clean46_true_oof_fullmatch_2026-09-07_v1/video_c80")
EVENT_RUN = Path(
    "data/verify/event_source_v1_remaining18_reexport_v3_2026-08-30/runs/schema=v1/"
    "video=video_c80/build=build-23b626609b70740db6b59980fe92f081c247c408e58daec8012ea7f78b4efbc7/"
    "attempt=remaining18-order-ambiguity-v3-c80"
)
DELIVERY = Path("/mnt/d/puyo_analyzer/videos/review/m1_clean46_manifest_unseen_c80_game3_2026-09-07_v3")
SUPERSEDED_DELIVERIES = (
    Path("/mnt/d/puyo_analyzer/videos/review/m1_clean46_manifest_unseen_c80_game3_2026-09-07_v1"),
    Path("/mnt/d/puyo_analyzer/videos/review/m1_clean46_manifest_unseen_c80_game3_2026-09-07_v2"),
)
VERIFY = Path("data/verify/advantage_m1_clean46_manifest_unseen_c80_game3_review_2026-09-07_v3")
OUTPUT = DELIVERY / "video_c80_game03_clean46_true_oof_full_match.mp4"
GAME_POSITION = 2
EXPECTED_START_SEC = 171.3
EXPECTED_END_SEC = 201.0
EXPECTED_START_FRAME = 5139
EXPECTED_END_FRAME_EXCLUSIVE = 6030
EXPECTED_SOURCE_FRAME_COUNT = 891


class ManifestUnseenReviewError(RuntimeError):
    """未見試合レビューの入力receiptまたは完成検査が不正。"""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ManifestUnseenReviewError(f"JSON objectではありません: {path}")
    return value


def _prior_review_manifests(review_root: Path) -> list[str]:
    matches: list[str] = []
    for path in review_root.rglob("*.manifest.json"):
        try:
            value = _load_json(path)
        except (OSError, json.JSONDecodeError, ManifestUnseenReviewError):
            continue
        if value.get("source_video_id") == VIDEO_ID:
            matches.append(str(path.resolve()))
    return sorted(matches)


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return float(np.mean(-(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))))


def _clean46_m0_statistics() -> dict[str, Any]:
    samples, manifest = single._load_analysis_dataset(DATASET)
    artifacts, predictions = comparison._ensemble(samples, manifest, TRAINING_ROOTS)
    mask = (
        (np.asarray(samples.source_groups) == SOURCE_GROUP_ID)
        & (np.asarray(samples.game_keys) == GAME_KEY)
    )
    if int(mask.sum()) <= 0 or set(map(int, np.asarray(samples.folds)[mask])) != {4}:
        raise ManifestUnseenReviewError("c80 game3のclean46 fold4 stateがありません")
    labels = np.asarray(samples.labels, dtype=np.float64)[mask]
    m0 = comparison._probability(predictions, "m0")[mask]
    m1 = comparison._probability(predictions, comparison.PRIMARY_VARIANT)[mask]
    delta = m1 - m0
    return {
        "label_use": "offline_report_only_not_display_path", "state_count": len(labels),
        "winner_side": "1P", "eval_fold": 4,
        "m0_log_loss": _log_loss(labels, m0), "m1_log_loss": _log_loss(labels, m1),
        "m1_minus_m0_log_loss": _log_loss(labels, m1) - _log_loss(labels, m0),
        "mean_absolute_probability_delta": float(np.mean(np.abs(delta))),
        "max_absolute_probability_delta": float(np.max(np.abs(delta))),
        "side_flip_count": int(np.count_nonzero((m0 >= 0.5) != (m1 >= 0.5))),
        "training_artifacts": [dict(item.receipt) for item in artifacts],
    }


def _render_command() -> tuple[str, ...]:
    return (
        str(PYTHON), "-u", "-m", "scripts.render_provisional_oof_review_v1",
        "--heldout-prediction-root", str(PREDICTION_ROOT),
        "--source-video", str(SOURCE), "--source-video-id", VIDEO_ID,
        "--event-source-run", str(EVENT_RUN), "--output", str(OUTPUT),
        "--probability", "calibrated", "--allow-calibrated-review",
        "--show-probability-graph", "--review-dashboard-layout", "--resize-1080p",
        "--freeze-stable-state-between-events", "--frame-stride", "1",
        "--start-game-position", str(GAME_POSITION),
        "--end-game-position", str(GAME_POSITION),
    )


def _render() -> dict[str, Any]:
    DELIVERY.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(
        _render_command(), cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode or not OUTPUT.is_file():
        raise ManifestUnseenReviewError(f"全試合render失敗: {result.stderr[-3000:]}")
    return _load_json(OUTPUT.with_suffix(".manifest.json"))


def _validate_render(manifest: Mapping[str, Any]) -> dict[str, Any]:
    segment, prediction = manifest.get("game_segment", {}), manifest.get("true_oof_prediction_source", {})
    stats = manifest.get("stats", {})
    terminal = stats.get("terminal_outcomes", [])
    properties = media._video_properties(OUTPUT)
    checks = {
        "manifest_unseen_before_render": True,
        "source_sha256": base.file_sha256(SOURCE) == SOURCE_SHA256,
        "game_identity": segment.get("start_game_key") == GAME_KEY,
        "game_position": (segment.get("start_position"), segment.get("end_position")) == (2, 2),
        "full_boundary_range": math.isclose(segment.get("resolved_start_sec", -1), EXPECTED_START_SEC, abs_tol=0.001)
        and math.isclose(segment.get("resolved_end_sec", -1), EXPECTED_END_SEC, abs_tol=0.001),
        "exact_boundary_frames": (
            segment.get("exact_start_frame_inclusive"),
            segment.get("exact_end_frame_exclusive"),
            segment.get("exact_source_frame_count"),
        ) == (EXPECTED_START_FRAME, EXPECTED_END_FRAME_EXCLUSIVE, EXPECTED_SOURCE_FRAME_COUNT),
        "exact_boundary_source": segment.get("exact_frame_source") == "event_source_formal_exact/v1",
        "stats_source_frames": (
            stats.get("source_start_frame"), stats.get("source_end_frame_exclusive"),
            stats.get("source_frame_count"),
        ) == (EXPECTED_START_FRAME, EXPECTED_END_FRAME_EXCLUSIVE, EXPECTED_SOURCE_FRAME_COUNT),
        "true_oof": prediction.get("true_oof") is True,
        "three_seed_fold_excluded": prediction.get("model_count") == 3,
        "resolution_1920x1080": (properties["width"], properties["height"]) == (1920, 1080),
        "fps_30": abs(properties["fps"] - 30.0) < 0.1,
        "output_frame_count_891": int(properties.get("frame_count", -1)) == EXPECTED_SOURCE_FRAME_COUNT,
        "audio_packets": media._audio_has_packets(OUTPUT),
        "audio_muxed": manifest.get("audio_muxed") is True,
        "terminal_1p_win": len(terminal) == 1 and terminal[0].get("terminal_loser") == "p2",
        "production_dependencies_compatible": production_compatible(ROOT),
    }
    if not all(checks.values()):
        raise ManifestUnseenReviewError(f"全試合検収失敗: {checks}")
    return {"checks": checks, "properties": properties, **media._decode_entire_media(OUTPUT)}


def _write_report(
    manifest: Mapping[str, Any], validation: Mapping[str, Any],
    prior: Sequence[str], superseded_attempts: Sequence[str],
    m0_statistics: Mapping[str, Any],
) -> dict[str, Any]:
    report = {
        "format_version": "advantage-m1-clean46-manifest-unseen-full-match-review/v2",
        "not_production": True, "review_role": "unselected-continuous-time-development-review",
        "manifest_unseen_prior_matches": list(prior),
        "manifest_unseen_scope": "no-review-manifest-before-initial-v1-attempt",
        "superseded_attempt_roots": [str(path.resolve()) for path in SUPERSEDED_DELIVERIES],
        "superseded_attempt_manifests": list(superseded_attempts),
        "superseded_failed_attempts": list(superseded_attempts),
        "source": {"video_id": VIDEO_ID, "tier": "マスター", "sha256": SOURCE_SHA256},
        "game": {
            "game_key": GAME_KEY, "winner_side": "1P",
            "start_sec": EXPECTED_START_SEC, "end_sec": EXPECTED_END_SEC,
            "start_frame_inclusive": EXPECTED_START_FRAME,
            "end_frame_exclusive": EXPECTED_END_FRAME_EXCLUSIVE,
            "source_frame_count": EXPECTED_SOURCE_FRAME_COUNT,
        },
        "prediction_manifest_sha256": base.file_sha256(PREDICTION_ROOT / "manifest.json"),
        "render_manifest_sha256": base.file_sha256(OUTPUT.with_suffix(".manifest.json")),
        "output": {"path": str(OUTPUT.resolve()), "sha256": base.file_sha256(OUTPUT)},
        "validation": dict(validation), "clean46_matched_m0_offline_comparison": dict(m0_statistics),
        "production_config_sha256": dependency_receipt(ROOT)["actual_sha256"],
        "production_dependency_contract": dependency_receipt(ROOT),
        "renderer_receipt": dict(manifest.get("true_oof_prediction_source", {})),
    }
    VERIFY.mkdir(parents=True, exist_ok=False)
    report_path = base._write_json_exclusive(VERIFY / "REPORT.json", report)
    base._write_json_exclusive(VERIFY / "COMPLETE", {"report_sha256": base.file_sha256(report_path)})
    return report


def main() -> int:
    if DELIVERY.exists() or VERIFY.exists():
        raise ManifestUnseenReviewError("出力先は新規必須です")
    found = _prior_review_manifests(Path("/mnt/d/puyo_analyzer/videos/review"))
    prefixes = tuple(str(path.resolve()) for path in SUPERSEDED_DELIVERIES)
    superseded = [path for path in found if path.startswith(prefixes)]
    prior = [path for path in found if path not in superseded]
    if prior:
        raise ManifestUnseenReviewError(f"c80は既存review manifestに登場済みです: {prior}")
    m0_statistics = _clean46_m0_statistics()
    manifest = _render()
    report = _write_report(
        manifest, _validate_render(manifest), prior, superseded, m0_statistics,
    )
    print(json.dumps({"output": report["output"], "checks": report["validation"]["checks"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
