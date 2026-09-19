"""M0 50動画成果物とレビュー動画を一括でfail-closed検収する。"""

from __future__ import annotations

from scripts.production_dependency_contract import (
    dependency_receipt, production_compatible, saved_dependency_compatible,
)

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.build_advantage_m0_review_predictions_v1 import file_sha256


AUDIT_VERSION = "advantage-m0-50v-delivery-audit/v1"
PRODUCTION_CONFIG_SHA256 = "3fe3c2578b3196a60cffffcafa594c1f64d2a913bb0487932ac5cd8f6f86d376"


def verify(args: argparse.Namespace) -> dict[str, Any]:
    training = _verified_training(args.training_root)
    repeat = _verified_training(args.repeat_training_root)
    prediction = _verified_artifact(args.prediction_root, "predictions.parquet")
    learned = _verified_artifact(args.learned_indicator_root, "learned_indicators.parquet")
    analysis = _verified_json_artifact(args.analysis_root, "analysis.json", "analysis_sha256")
    stability = _verified_json_artifact(args.stability_root, "stability.json", "stability_sha256")
    review = _verified_json_artifact(args.review_analysis_root, "review_analysis.json", "analysis_sha256")
    video = _verified_video(args.video)
    checks = _checks(args, training, repeat, prediction, learned, analysis, stability, review, video)
    report = {"format_version": AUDIT_VERSION, "checks": checks,
              "all_pass": all(checks.values()), "production_dependency_contract": dependency_receipt()}
    if not report["all_pass"]:
        raise ValueError(f"M0 delivery audit FAIL: {[key for key, ok in checks.items() if not ok]}")
    return _write(args.output_root, report)


def _verified_training(root: Path) -> dict[str, Any]:
    manifest = _load(root / "manifest.json")
    complete = _load(root / "COMPLETE")
    files = (root / "model.pt", root / "selection_model.pt")
    valid = all((
        complete.get("manifest_sha256") == file_sha256(root / "manifest.json"),
        complete.get("model_sha256") == file_sha256(files[0]),
        complete.get("selection_model_sha256") == file_sha256(files[1]),
        manifest.get("model", {}).get("sha256") == file_sha256(files[0]),
        manifest.get("selection_model", {}).get("sha256") == file_sha256(files[1]),
    ))
    if not valid:
        raise ValueError(f"training成果物が不正です: {root}")
    return manifest


def _verified_artifact(root: Path, data_name: str) -> dict[str, Any]:
    manifest = _load(root / "manifest.json")
    complete = _load(root / "COMPLETE")
    data_path = root / data_name
    if complete.get("manifest_sha256") != file_sha256(root / "manifest.json"):
        raise ValueError(f"manifest完了hashが不正です: {root}")
    data_hashes = [value for key, value in complete.items() if key.endswith("_sha256")]
    if file_sha256(data_path) not in data_hashes:
        raise ValueError(f"data完了hashが不正です: {data_path}")
    return manifest


def _verified_json_artifact(root: Path, name: str, hash_key: str) -> dict[str, Any]:
    path = root / name
    complete = _load(root / "COMPLETE")
    if complete.get(hash_key) != file_sha256(path):
        raise ValueError(f"JSON完了hashが不正です: {path}")
    return _load(path)


def _verified_video(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    manifest = _load(manifest_path)
    if manifest.get("output", {}).get("sha256") != file_sha256(path):
        raise ValueError("レビュー動画hashが不正です")
    return manifest


def _checks(
    args: argparse.Namespace, training: Mapping[str, Any], repeat: Mapping[str, Any],
    prediction: Mapping[str, Any], learned: Mapping[str, Any], analysis: Mapping[str, Any],
    stability: Mapping[str, Any], review: Mapping[str, Any], video: Mapping[str, Any],
) -> dict[str, bool]:
    training_hash = file_sha256(args.training_root / "manifest.json")
    return {
        "training_exact_50": training.get("source_count") == 50,
        "review_source_excluded": training.get("review_source_excluded") is True,
        "not_production": all(item.get("not_production") is True for item in (
            training, prediction, learned, analysis, stability, review,
        )),
        "deterministic_repeat_manifest": training_hash == file_sha256(args.repeat_training_root / "manifest.json"),
        "deterministic_repeat_model": training["model"]["sha256"] == repeat["model"]["sha256"],
        "deterministic_repeat_selection": training["selection_model"]["sha256"] == repeat["selection_model"]["sha256"],
        "prediction_training_match": prediction.get("training_manifest_sha256") == training_hash,
        "learned_indicator_training_match": learned.get("training_manifest_sha256") == training_hash,
        "analysis_training_match": analysis.get("training_manifest_sha256") == training_hash,
        "stability_training_match": stability.get("training_manifest_sha256") == training_hash,
        "review_first_20": review.get("scope", {}).get("game_count") == 20,
        "video_first_20": video.get("game_segment", {}).get("end_position") == 20,
        "video_audio_muxed": video.get("audio_muxed") is True,
        "video_30fps_stride": video.get("frame_stride") == 2,
        "video_heldout_prediction": video.get("heldout_prediction_source", {}).get("training_excluded") is True,
        "production_dependencies_compatible": production_compatible(),
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON objectではありません: {path}")
    return value


def _write(root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    path = root / "audit.json"
    _write_json_exclusive(path, report)
    _write_json_exclusive(root / "COMPLETE", {
        "format_version": "advantage-m0-50v-delivery-audit-complete/v1",
        "audit_sha256": file_sha256(path),
    })
    print(json.dumps(report, sort_keys=True))
    return dict(report)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--repeat-training-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--learned-indicator-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--stability-root", type=Path, required=True)
    parser.add_argument("--review-analysis-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    verify(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
