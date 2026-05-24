"""seed dataset の品質を自動評価する metric 集.

ユーザー目視 10 動画 22 PNG レビュー (= 2026-05-21) で判明した汚染パターン
(yellow に red 混入 60%、 red に green/effect 混入 30% 等) を自動 catch する。
強化アナリスト (recognition_evaluator) は cell 出力評価で seed 入力は対象外
だったため、 別 module として実装。

主 metric:
    S1: cross_color_purity = 各 cell の HSV-only 判定と patch HSV 主色の
        一致率 (= ラベル「yellow」 だが H 中央値が red 域 = 不純)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_YELLOW,
)
from src.self_supervised.pseudo_label import PseudoLabelSample

# S1: 各色の H 中央値「core 域」 (= extract_hsv_seed_dataset.py と同じ定数)
COLOR_H_CENTER: dict[int, int] = {
    COLOR_YELLOW: 25,
    COLOR_GREEN: 60,
    COLOR_BLUE: 110,
    COLOR_PURPLE: 140,
}
H_CORE_DELTA: int = 12  # 評価時は採取時 (=8) より緩く判定
RED_H_LOW_MAX: int = 12
RED_H_HIGH_MIN: int = 168

COLOR_NAMES: dict[int, str] = {
    COLOR_RED: "red",
    COLOR_BLUE: "blue",
    COLOR_GREEN: "green",
    COLOR_YELLOW: "yellow",
    COLOR_PURPLE: "purple",
    COLOR_OJAMA: "ojama",
    COLOR_EMPTY: "empty",
}


@dataclass
class SeedQualityReport:
    video_id: str
    per_color_counts: dict[int, int]
    per_color_pure: dict[int, int]
    per_color_purity: dict[int, float]
    overall_purity: float
    contamination_examples: list[dict[str, Any]]  # 不純 sample の (color, h_med, expected)

    def to_json(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "per_color_counts": {COLOR_NAMES.get(c, str(c)): n for c, n in self.per_color_counts.items()},
            "per_color_pure": {COLOR_NAMES.get(c, str(c)): n for c, n in self.per_color_pure.items()},
            "per_color_purity": {COLOR_NAMES.get(c, str(c)): round(p, 4) for c, p in self.per_color_purity.items()},
            "overall_purity": round(self.overall_purity, 4),
            "contamination_examples": self.contamination_examples[:50],
        }


def _h_in_core(h_med: int, color: int) -> bool:
    """H 中央値が color の core 域に入るか (= 採取時と同じ判定、 但し delta 緩め)."""
    if color == COLOR_RED:
        return h_med < RED_H_LOW_MAX or h_med > RED_H_HIGH_MIN
    center = COLOR_H_CENTER.get(color)
    if center is None:
        return True  # OJAMA / EMPTY は判定対象外
    return abs(h_med - center) <= H_CORE_DELTA


def evaluate_seed_dir(seed_root: Path, video_id: str | None = None) -> SeedQualityReport:
    """1 動画分の seed dir を評価."""
    if video_id is None:
        video_id = seed_root.name
    jsonl = seed_root / "cell.jsonl"
    if not jsonl.exists():
        return SeedQualityReport(
            video_id=video_id,
            per_color_counts={}, per_color_pure={}, per_color_purity={},
            overall_purity=0.0, contamination_examples=[],
        )
    counts: dict[int, int] = {}
    pure: dict[int, int] = {}
    contamination: list[dict[str, Any]] = []
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sample = PseudoLabelSample.from_jsonable(obj)
            except Exception:
                continue
            try:
                color = int(sample.label)
            except (TypeError, ValueError):
                continue
            patch = sample.input_data
            if isinstance(patch, dict):
                patch = patch.get("patch")
            if not isinstance(patch, np.ndarray):
                continue
            patch_arr = patch
            counts[color] = counts.get(color, 0) + 1
            if color in (COLOR_OJAMA, COLOR_EMPTY):
                pure[color] = pure.get(color, 0) + 1
                continue
            hsv = cv2.cvtColor(patch_arr, cv2.COLOR_BGR2HSV)
            h_med = int(np.median(hsv[:, :, 0]))
            if _h_in_core(h_med, color):
                pure[color] = pure.get(color, 0) + 1
            else:
                if len(contamination) < 200:
                    contamination.append({
                        "color": COLOR_NAMES.get(color, str(color)),
                        "h_med": h_med,
                        "expected_center": COLOR_H_CENTER.get(color, "wrap" if color == COLOR_RED else None),
                    })
    purity: dict[int, float] = {}
    for c, n in counts.items():
        p = pure.get(c, 0) / n if n > 0 else 1.0
        purity[c] = p
    total = sum(counts.values())
    total_pure = sum(pure.values())
    overall = total_pure / total if total > 0 else 1.0
    return SeedQualityReport(
        video_id=video_id,
        per_color_counts=counts,
        per_color_pure=pure,
        per_color_purity=purity,
        overall_purity=overall,
        contamination_examples=contamination,
    )


def _decode_patch(patch_data: Any) -> np.ndarray | None:
    """JSON 化された patch を numpy array に戻す."""
    if patch_data is None:
        return None
    if isinstance(patch_data, np.ndarray):
        return patch_data
    if isinstance(patch_data, list):
        try:
            arr = np.array(patch_data, dtype=np.uint8)
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                return arr
        except (ValueError, TypeError):
            return None
    return None


def aggregate_reports(reports: list[SeedQualityReport]) -> dict[str, Any]:
    """複数 video の report を集約 (= 全体傾向把握用)."""
    total_counts: dict[int, int] = {}
    total_pure: dict[int, int] = {}
    for r in reports:
        for c, n in r.per_color_counts.items():
            total_counts[c] = total_counts.get(c, 0) + n
        for c, n in r.per_color_pure.items():
            total_pure[c] = total_pure.get(c, 0) + n
    overall_purity = {}
    for c, n in total_counts.items():
        overall_purity[COLOR_NAMES.get(c, str(c))] = round(
            total_pure.get(c, 0) / n if n > 0 else 1.0, 4,
        )
    total = sum(total_counts.values())
    total_p = sum(total_pure.values())
    return {
        "videos_evaluated": len(reports),
        "total_samples": total,
        "total_pure": total_p,
        "overall_purity": round(total_p / total if total > 0 else 1.0, 4),
        "per_color_purity": overall_purity,
        "per_color_counts": {COLOR_NAMES.get(c, str(c)): n for c, n in total_counts.items()},
    }
