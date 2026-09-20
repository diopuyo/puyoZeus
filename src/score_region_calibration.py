"""複数フレームから得点表示の左右別座標補正を決定する。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.score_ocr import ScoreOcr, Side


SCHEMA_VERSION = "score_region_calibration_v1"
MIN_VALID_FRAMES = 3
MIN_VALID_GAIN = 2
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class SearchGrid:
    """探索する相対座標の格子。"""

    dx_min: int = -16
    dx_max: int = 16
    dy_min: int = -12
    dy_max: int = 12
    step: int = 2

    def candidates(self) -> tuple[tuple[int, int], ...]:
        if self.step <= 0:
            raise ValueError("探索刻みは0より大きくしてください")
        if self.dx_min > self.dx_max or self.dy_min > self.dy_max:
            raise ValueError("探索範囲の最小値と最大値が逆です")
        return tuple(
            (dx, dy)
            for dy in range(self.dy_min, self.dy_max + 1, self.step)
            for dx in range(self.dx_min, self.dx_max + 1, self.step)
        )


@dataclass(frozen=True, slots=True)
class OffsetEvaluation:
    """一つの相対座標における読み取り成績。"""

    dx: int
    dy: int
    valid_frame_count: int
    recognized_digit_count: int
    confidence_sum: float


@dataclass(frozen=True, slots=True)
class SideCalibration:
    """片側の基準・最良候補と最終採用値。"""

    side: Side
    adopted: bool
    selected_dx: int
    selected_dy: int
    base: OffsetEvaluation
    best: OffsetEvaluation
    candidate_count: int


@dataclass(frozen=True, slots=True)
class ScoreRegionCalibration:
    """一映像に固定した得点表示位置の設定。"""

    source_video_id: str
    source_video_sha256: str
    requested_sample_count: int
    sampled_timestamps_sec: tuple[float, ...]
    sample_window_sec: float
    search_grid: SearchGrid
    p1: SideCalibration
    p2: SideCalibration

    def offsets(self) -> dict[Side, tuple[int, int]]:
        return {
            "1P": (self.p1.selected_dx, self.p1.selected_dy),
            "2P": (self.p2.selected_dx, self.p2.selected_dy),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_video_id": self.source_video_id,
            "source_video_sha256": self.source_video_sha256,
            "requested_sample_count": self.requested_sample_count,
            "sampled_timestamps_sec": list(self.sampled_timestamps_sec),
            "sample_window_sec": self.sample_window_sec,
            "search_grid": asdict(self.search_grid),
            "sides": {"1P": asdict(self.p1), "2P": asdict(self.p2)},
        }


def calibrate_score_regions(
    frames: Sequence[np.ndarray],
    ocr: ScoreOcr,
    source_video_id: str,
    source_video_sha256: str,
    sampled_timestamps_sec: Sequence[float],
    requested_sample_count: int,
    sample_window_sec: float,
    search_grid: SearchGrid = SearchGrid(),
) -> ScoreRegionCalibration:
    """左右の探索結果を、採用条件込みの再現可能な設定へまとめる。"""
    if not frames or len(frames) != len(sampled_timestamps_sec):
        raise ValueError("フレームと時刻は同じ非ゼロ件数で指定してください")
    candidates = search_grid.candidates()
    p1 = _calibrate_side(frames, ocr, "1P", candidates)
    p2 = _calibrate_side(frames, ocr, "2P", candidates)
    return ScoreRegionCalibration(
        source_video_id=source_video_id,
        source_video_sha256=source_video_sha256,
        requested_sample_count=requested_sample_count,
        sampled_timestamps_sec=tuple(float(value) for value in sampled_timestamps_sec),
        sample_window_sec=float(sample_window_sec),
        search_grid=search_grid,
        p1=p1,
        p2=p2,
    )


def _calibrate_side(
    frames: Sequence[np.ndarray],
    ocr: ScoreOcr,
    side: Side,
    candidates: Sequence[tuple[int, int]],
) -> SideCalibration:
    evaluations = tuple(
        _evaluate_offset(frames, ocr, side, dx, dy) for dx, dy in candidates
    )
    base = next((item for item in evaluations if (item.dx, item.dy) == (0, 0)), None)
    if base is None:
        raise ValueError("探索候補には基準座標 (0, 0) が必要です")
    best = max(evaluations, key=_evaluation_rank)
    adopted = (
        best.valid_frame_count >= MIN_VALID_FRAMES
        and best.valid_frame_count - base.valid_frame_count >= MIN_VALID_GAIN
    )
    selected = best if adopted else base
    return SideCalibration(
        side, adopted, selected.dx, selected.dy, base, best, len(evaluations)
    )


def _evaluate_offset(
    frames: Sequence[np.ndarray],
    ocr: ScoreOcr,
    side: Side,
    dx: int,
    dy: int,
) -> OffsetEvaluation:
    valid = 0
    recognized = 0
    confidence_sum = 0.0
    for frame in frames:
        score, _confidence, labels, confidences = ocr.read_side_detail_at_offset(
            frame, side, dx, dy,
        )
        valid += int(score is not None)
        recognized += sum(label is not None for label in labels)
        confidence_sum += sum(confidences)
    return OffsetEvaluation(dx, dy, valid, recognized, confidence_sum)


def _evaluation_rank(item: OffsetEvaluation) -> tuple[float, ...]:
    """同率なら原点に近く、縦・横の移動が小さい候補を選ぶ。"""
    return (
        float(item.valid_frame_count),
        float(item.recognized_digit_count),
        item.confidence_sum,
        float(-abs(item.dx) - abs(item.dy)),
        float(-abs(item.dy)),
        float(-abs(item.dx)),
    )


def write_score_region_calibration(
    path: Path, calibration: ScoreRegionCalibration,
) -> None:
    """同名上書きを禁止して設定をJSONへ保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(calibration.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_score_region_calibration(
    path: Path,
    expected_video_id: str | None = None,
    expected_source_sha256: str | None = None,
) -> ScoreRegionCalibration:
    """形式と元映像の一致を確認して設定を読む。"""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    calibration = _calibration_from_dict(value)
    _validate_loaded_calibration(calibration)
    if expected_video_id is not None and calibration.source_video_id != expected_video_id:
        raise ValueError("位置補正設定の動画IDが入力映像と一致しません")
    if (expected_source_sha256 is not None
            and calibration.source_video_sha256 != expected_source_sha256):
        raise ValueError("位置補正設定の映像SHA-256が入力映像と一致しません")
    return calibration


def load_score_region_offsets_for_video(
    calibration_path: Path, video_path: Path,
) -> dict[Side, tuple[int, int]]:
    """映像IDと内容ハッシュを照合して、読み取り器向け座標を返す。"""
    calibration = load_score_region_calibration(
        calibration_path,
        expected_video_id=video_path.stem,
        expected_source_sha256=source_file_sha256(video_path),
    )
    return calibration.offsets()


def source_file_sha256(path: Path) -> str:
    """較正対象の元映像内容を識別する。"""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def _calibration_from_dict(value: object) -> ScoreRegionCalibration:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("位置補正設定の形式が不正です")
    try:
        grid = SearchGrid(**value["search_grid"])
        sides = value["sides"]
        p1 = _side_from_dict(sides["1P"])
        p2 = _side_from_dict(sides["2P"])
        return ScoreRegionCalibration(
            str(value["source_video_id"]), str(value["source_video_sha256"]),
            int(value["requested_sample_count"]),
            tuple(float(item) for item in value["sampled_timestamps_sec"]),
            float(value["sample_window_sec"]), grid, p1, p2,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("位置補正設定の内容が不正です") from exc


def _side_from_dict(value: dict[str, Any]) -> SideCalibration:
    side = str(value["side"])
    if side not in ("1P", "2P"):
        raise ValueError("位置補正設定のsideが不正です")
    base = _evaluation_from_dict(value["base"])
    best = _evaluation_from_dict(value["best"])
    return SideCalibration(
        side=side,  # type: ignore[arg-type]
        adopted=bool(value["adopted"]),
        selected_dx=int(value["selected_dx"]),
        selected_dy=int(value["selected_dy"]),
        base=base,
        best=best,
        candidate_count=int(value["candidate_count"]),
    )


def _evaluation_from_dict(value: dict[str, Any]) -> OffsetEvaluation:
    return OffsetEvaluation(
        dx=int(value["dx"]),
        dy=int(value["dy"]),
        valid_frame_count=int(value["valid_frame_count"]),
        recognized_digit_count=int(value["recognized_digit_count"]),
        confidence_sum=float(value["confidence_sum"]),
    )


def _validate_loaded_calibration(value: ScoreRegionCalibration) -> None:
    """手編集されたJSONでも採用規則と元映像識別子を破れないようにする。"""
    if not value.source_video_id or not SHA256_PATTERN.fullmatch(value.source_video_sha256):
        raise ValueError("位置補正設定の元映像識別子が不正です")
    sample_count = len(value.sampled_timestamps_sec)
    if sample_count < 3 or value.requested_sample_count < sample_count:
        raise ValueError("位置補正設定の標本数が不足しています")
    if value.sample_window_sec <= 0:
        raise ValueError("位置補正設定の標本時間範囲が不正です")
    candidates = value.search_grid.candidates()
    for expected_side, side in (("1P", value.p1), ("2P", value.p2)):
        if side.side != expected_side or side.candidate_count != len(candidates):
            raise ValueError("位置補正設定のsideまたは候補数が不正です")
        if (side.base.dx, side.base.dy) != (0, 0):
            raise ValueError("位置補正設定の基準座標が不正です")
        if (side.best.dx, side.best.dy) not in candidates:
            raise ValueError("位置補正設定の最良座標が探索範囲外です")
        _validate_evaluation_counts(side.base, sample_count)
        _validate_evaluation_counts(side.best, sample_count)
        expected_adopted = (
            side.best.valid_frame_count >= MIN_VALID_FRAMES
            and side.best.valid_frame_count - side.base.valid_frame_count >= MIN_VALID_GAIN
        )
        selected = side.best if expected_adopted else side.base
        if side.adopted != expected_adopted:
            raise ValueError("位置補正設定の採用判定が規則と一致しません")
        if (side.selected_dx, side.selected_dy) != (selected.dx, selected.dy):
            raise ValueError("位置補正設定の採用座標が成績と一致しません")


def _validate_evaluation_counts(value: OffsetEvaluation, sample_count: int) -> None:
    if not 0 <= value.valid_frame_count <= sample_count:
        raise ValueError("位置補正設定の成功フレーム数が不正です")
    if not 0 <= value.recognized_digit_count <= sample_count * 8:
        raise ValueError("位置補正設定の認識桁数が不正です")
    if not np.isfinite(value.confidence_sum) or value.confidence_sum < 0:
        raise ValueError("位置補正設定の信頼度合計が不正です")
