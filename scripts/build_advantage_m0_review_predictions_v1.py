"""M0 50本モデルで完成event runの学習外レビュー予測を作る。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.advantage_m0_current_cnn_v1 import (
    AdvantageM0CurrentCNNV1,
    AdvantageM0CurrentCNNV2,
    M0_MODEL_VERSION,
    board_categories_from_raw,
    queue_categories_from_raw,
)
from src.event_learning_tables_v1 import build_learning_tables_from_run


PREDICTION_VERSION = "heldout-review-predictions/m0-v1"
QUEUE_KEYS = ("next_first", "next_second", "double_next_first", "double_next_second")
DEATH_ROW = 1
DEATH_COLUMN = 2


class AdvantageM0ReviewPredictionError(RuntimeError):
    """M0学習外レビュー予測の由来または出力が不正。"""


def build_predictions(args: argparse.Namespace) -> dict[str, Any]:
    training = _load_training(args.training_root)
    event_manifest = _load_json(args.event_source_run / "manifest.json")
    source = _validate_source(args, training, event_manifest)
    tables = build_learning_tables_from_run(
        args.event_source_run, fold=0, tier=args.tier,
        source_group_id=args.source_group_id,
    )
    if len(tables.states) != len(tables.labels):
        raise AdvantageM0ReviewPredictionError("状態表と正解対応表の行数が一致しません")
    model = _load_model(args.training_root, training, args.device)
    probability = _predict_states(model, tables.states, args.batch_size, args.device)
    rows = [_prediction_row(
        state, label, float(value), args.source_group_id,
        args.symmetric_calibration_slope,
    ) for state, label, value in zip(
        tables.states, tables.labels, probability, strict=True,
    )]
    return _write_outputs(args, training, event_manifest, source, rows)


def _load_training(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path)
    complete = _load_json(root / "COMPLETE")
    supported = {
        "advantage-m0-training/v1": M0_MODEL_VERSION,
        "advantage-m0-training/v2-color-invariant": AdvantageM0CurrentCNNV2.model_version,
    }
    valid = all((
        complete.get("manifest_sha256") == file_sha256(manifest_path),
        complete.get("model_sha256") == file_sha256(root / "model.pt"),
        supported.get(str(manifest.get("format_version"))) == manifest.get("model_version"),
        manifest.get("not_production") is True,
        manifest.get("source_count") == 50,
        manifest.get("review_source_excluded") is True,
    ))
    if not valid:
        raise AdvantageM0ReviewPredictionError("M0学習成果物の完成条件が一致しません")
    return manifest


def _validate_source(
    args: argparse.Namespace, training: Mapping[str, Any], event_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    source = event_manifest.get("source", {})
    if not isinstance(source, Mapping):
        raise AdvantageM0ReviewPredictionError("event runのsource説明が不正です")
    groups = set(str(value) for value in training.get("source_group_ids", []))
    valid = all((
        args.source_group_id not in groups,
        source.get("source_video_id") == args.source_video_id,
        source.get("source_video_sha256") == file_sha256(args.source_video),
    ))
    if not valid:
        raise AdvantageM0ReviewPredictionError("レビュー元の除外またはhashが不正です")
    return source


def _load_model(
    root: Path, manifest: Mapping[str, Any], device_name: str,
) -> nn.Module:
    device = _device(device_name)
    checkpoint = torch.load(root / "model.pt", map_location=device, weights_only=False)
    if checkpoint.get("model_version") != manifest.get("model_version"):
        raise AdvantageM0ReviewPredictionError("checkpointのモデル版が一致しません")
    model = _new_model(str(checkpoint["model_version"])).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    if manifest.get("model", {}).get("sha256") != file_sha256(root / "model.pt"):
        raise AdvantageM0ReviewPredictionError("manifestのmodel hashが一致しません")
    return model


def _new_model(model_version: str) -> nn.Module:
    if model_version == AdvantageM0CurrentCNNV2.model_version:
        return AdvantageM0CurrentCNNV2()
    if model_version == M0_MODEL_VERSION:
        return AdvantageM0CurrentCNNV1()
    raise AdvantageM0ReviewPredictionError("未対応のM0モデル版です")


def _predict_states(
    model: nn.Module, states: Sequence[Mapping[str, Any]],
    batch_size: int, device_name: str,
) -> np.ndarray:
    boards, queues = _state_tensors(states)
    device = _device(device_name)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            end = min(len(states), start + batch_size)
            result = model(
                torch.from_numpy(boards[start:end]).to(device=device, dtype=torch.int64),
                torch.from_numpy(queues[start:end]).to(device=device, dtype=torch.int64),
            )
            outputs.append(result.raw_probability.cpu().numpy())
    return np.concatenate(outputs)


def _state_tensors(
    states: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    board_rows: list[np.ndarray] = []
    queue_rows: list[np.ndarray] = []
    for row in states:
        side_boards, side_queues = [], []
        for side in ("p1", "p2"):
            raw = np.asarray(row[f"a_{side}_grid"], dtype=np.int8).reshape(13, 6)
            side_boards.append(board_categories_from_raw(raw))
            values = [row.get(f"a_{side}_{key}") for key in QUEUE_KEYS]
            side_queues.append(queue_categories_from_raw(values))
        board_rows.append(np.stack(side_boards))
        queue_rows.append(np.stack(side_queues))
    return np.stack(board_rows).astype(np.int8), np.stack(queue_rows).astype(np.int8)


def _prediction_row(
    state: Mapping[str, Any], label: Mapping[str, Any], probability: float,
    source_group_id: str, calibration_slope: float,
) -> dict[str, Any]:
    if state["state_id"] != label["state_id"]:
        raise AdvantageM0ReviewPredictionError("state_id対応が一致しません")
    grids = {side: np.asarray(state[f"a_{side}_grid"]).reshape(13, 6)
             for side in ("p1", "p2")}
    return {
        "schema_version": PREDICTION_VERSION, "state_id": state["state_id"],
        "source_video_id": state["source_video_id"], "source_group_id": source_group_id,
        "available_frame": int(state["available_frame"]),
        "available_ms": int(state["available_ms"]),
        "online_segment_index": int(state["online_segment_index"]),
        "game_key": label.get("game_key"), "official_game_key": label.get("game_key"),
        "game_key_source": "official_game_assignment" if label.get("game_key") else "unassigned",
        "input_usable": bool(state.get("a_input_usable")),
        "raw_probability": probability,
        "calibrated_probability": _symmetric_calibrate(probability, calibration_slope),
        "candidate_used": True,
        "model_count": 1, "training_excluded": True, "heldout_review": True,
        "m0_current_cnn_50": True,
        "chain_active_p1": bool(state.get("b_p1_chain_active")),
        "chain_active_p2": bool(state.get("b_p2_chain_active")),
        "death_candidate_p1": _death_candidate(grids["p1"]),
        "death_candidate_p2": _death_candidate(grids["p2"]),
    }


def _symmetric_calibrate(probability: float, slope: float) -> float:
    clipped = float(np.clip(probability, 1e-7, 1.0 - 1e-7))
    logit = np.log(clipped / (1.0 - clipped))
    return float(1.0 / (1.0 + np.exp(-float(slope) * logit)))


def _death_candidate(grid: np.ndarray) -> bool:
    value = int(grid[DEATH_ROW, DEATH_COLUMN])
    return value not in {0, 10}


def _write_outputs(
    args: argparse.Namespace, training: Mapping[str, Any],
    event_manifest: Mapping[str, Any], source: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=False)
    prediction_path = args.output_root / "predictions.parquet"
    _write_parquet(prediction_path, rows)
    manifest = _manifest(args, training, event_manifest, source, rows, prediction_path)
    manifest_path = _write_json_exclusive(args.output_root / "manifest.json", manifest)
    _write_json_exclusive(args.output_root / "COMPLETE", {
        "format_version": "heldout-review-predictions-m0-complete/v1",
        "manifest_sha256": file_sha256(manifest_path),
        "predictions_sha256": file_sha256(prediction_path),
    })
    print(json.dumps({"rows": len(rows), "games": manifest["review_game_count"]},
                     ensure_ascii=False, sort_keys=True), flush=True)
    return manifest


def _manifest(
    args: argparse.Namespace, training: Mapping[str, Any],
    event_manifest: Mapping[str, Any], source: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]], prediction_path: Path,
) -> dict[str, Any]:
    game_keys = {row["game_key"] for row in rows if row.get("game_key")}
    available = [int(row["available_ms"]) for row in rows]
    return {
        "format": PREDICTION_VERSION, "model_version": training["model_version"],
        "heldout_review": True, "training_excluded": True,
        "m0_current_cnn_50": True, "not_production": True,
        "prediction_method": "single_full_fit_50_source_m0",
        "probability": "raw_and_symmetric_calibrated", "model_count": 1,
        "symmetric_calibration": {
            "slope": args.symmetric_calibration_slope,
            "source": "fixed_training_validation_source_groups",
            "preserves_side_swap_complement": True,
        },
        "source_group_id": args.source_group_id,
        "source": {"source_video_id": args.source_video_id,
                   "source_video_sha256": source["source_video_sha256"],
                   "source_was_in_training": False,
                   "training_source_count": training["source_count"]},
        "training_root": str(args.training_root.resolve()),
        "training_manifest_sha256": file_sha256(args.training_root / "manifest.json"),
        "model_sha256": file_sha256(args.training_root / "model.pt"),
        "event_source_run": str(args.event_source_run.resolve()),
        "event_manifest_sha256": file_sha256(args.event_source_run / "manifest.json"),
        "event_semantic_sha256": event_manifest.get("semantic_sha256"),
        "review_start_ms": min(available), "review_end_ms": max(available),
        "review_game_count": len(game_keys), "row_count": len(rows),
        "usable_prediction_count": sum(bool(row["input_usable"]) for row in rows),
        "predictions": {"name": prediction_path.name, "row_count": len(rows),
                        "sha256": file_sha256(prediction_path)},
    }


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet
    table = pa.Table.from_pylist(list(rows))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        parquet.write_table(table, handle, compression="zstd")


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdvantageM0ReviewPredictionError(f"JSON objectではありません: {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, indent=2) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return path


def _device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available()
                        else "cpu" if value == "auto" else value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--event-source-run", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--source-video-id", required=True)
    parser.add_argument("--source-group-id", required=True)
    parser.add_argument("--tier", default="S級")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--symmetric-calibration-slope", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    build_predictions(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
