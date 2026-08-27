"""Gate 4 の実表示区間から WIN★パネル由来の勝者根拠を抽出する。"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.match_winner import MatchWinnerDetector  # noqa: E402


@dataclass(frozen=True)
class GameStart:
    """1区間内の表示gameと、その表示開始時刻。"""

    segment: str
    game_idx: int
    start_sec: float


def _load_segment_starts(path: Path) -> list[GameStart]:
    """密displayのgame_idx遷移を時系列の開始点へ変換する。"""
    with np.load(path, allow_pickle=False) as data:
        times = data["t_sec"].astype(float)
        games = data["game_idx"].astype(int)
    starts: list[GameStart] = []
    for index in np.flatnonzero(np.r_[True, games[1:] != games[:-1]]):
        starts.append(GameStart(path.stem[:5], int(games[index]), float(times[index])))
    return starts


def _extract_rows(
    video: Path, display_dir: Path,
) -> list[dict[str, object]]:
    """隣接game開始時のWIN数値差から勝者を抽出する。"""
    detector = MatchWinnerDetector.load_default()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"動画を開けない: {video}")
    rows: list[dict[str, object]] = []
    try:
        for path in sorted(display_dir.glob("seg*_display.npz")):
            starts = _load_segment_starts(path)
            rows.extend(_extract_segment(detector, cap, starts))
    finally:
        cap.release()
    if not rows:
        raise ValueError(f"勝者抽出対象がない: {display_dir}")
    return rows


def _extract_segment(
    detector: MatchWinnerDetector, cap: cv2.VideoCapture,
    starts: list[GameStart],
) -> list[dict[str, object]]:
    """区間末尾の未完gameを除き、隣接開始点を比較する。"""
    rows: list[dict[str, object]] = []
    for current, following in zip(starts, starts[1:]):
        result = detector.detect_winner(
            cap, current.start_sec, following.start_sec)
        rows.append({
            "segment": current.segment, "game_idx": current.game_idx,
            "start_sec": f"{current.start_sec:.3f}",
            "next_start_sec": f"{following.start_sec:.3f}",
            "winner": result.winner or "UNKNOWN",
            "left_hamming": result.left_hamming,
            "right_hamming": result.right_hamming,
            "left_changed": int(result.left_changed),
            "right_changed": int(result.right_changed),
            "source": "win_panel_digit_delta",
        })
    return rows


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """既存成果物を上書きせずTSVへ保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _apply_overrides(
    rows: list[dict[str, object]], path: Path | None,
) -> None:
    """画面を人手確認した例外だけを、証跡つきで反映する。"""
    if path is None:
        return
    with path.open(encoding="utf-8", newline="") as handle:
        overrides = {
            (row["segment"], int(row["game_idx"])): row
            for row in csv.DictReader(handle, delimiter="\t")}
    for row in rows:
        override = overrides.get((str(row["segment"]), int(row["game_idx"])))
        if override is None:
            continue
        row["winner"] = override["winner"]
        row["source"] = "win_panel_manual_review"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--display-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--override-tsv", type=Path)
    args = parser.parse_args()
    rows = _extract_rows(args.video, args.display_dir)
    _apply_overrides(rows, args.override_tsv)
    _write_tsv(args.out, rows)
    known = sum(row["winner"] != "UNKNOWN" for row in rows)
    print(f"WIN★勝者根拠 {known}/{len(rows)} -> {args.out}")
    return int(known == 0)


if __name__ == "__main__":
    raise SystemExit(main())
