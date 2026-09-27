"""固定判定の勝敗照合用に、既存WINパネル比較器で未収録ラベルを補う。"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.run_e3_exchange_eval_20260926 import SOURCES, VIDEO_ROOT, save_json
from src.match_winner import MatchWinnerDetector

OUT = Path("logs/e10")
OFFSETS = (1.0, 2.0, 3.0, 4.0)
MIN_AGREEMENT = 2


def labels(source: str, video: Path, display: Path) -> list:
    """認識試合境界の前後のWIN数値差を複数時刻で照合する。"""
    data = np.load(display)
    times, games = data["t_sec"], data["game_idx"]
    starts = np.flatnonzero(np.r_[True, games[1:] != games[:-1]])
    rows = []
    detector = MatchWinnerDetector.load_default()
    cap = cv2.VideoCapture(str(video))
    for pos, index in enumerate(starts):
        end = float(times[starts[pos+1]]) if pos+1 < len(starts) else float(times[-1])
        start = float(times[index])
        observations = []
        for offset in OFFSETS:
            comparison = min(end + offset, cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)-1)
            result = detector.detect_winner(cap, start, comparison, offset_before=offset, offset_after=0)
            observations.append(dict(t_before=start+offset, t_after=comparison, **asdict(result)))
        counts = Counter(r["winner"] for r in observations if r["winner"] is not None)
        winner = next(iter(counts)) if len(counts) == 1 and max(counts.values()) >= MIN_AGREEMENT else None
        rows.append(dict(source=source, game_idx=int(games[index]), start=start, end=end,
                         winner=winner, observations=observations, method="WIN_panel_difference"))
    cap.release()
    return rows


def main() -> None:
    """既存動画を読み取り、盤面勝率とは独立のWIN表示だけを保存する。"""
    rows = []
    for source in SOURCES:
        rows.extend(labels(source, VIDEO_ROOT / f"{source}_first_0_900_20260925_v1.mp4",
                           Path("logs/e9/renders") / source / "on/display.npz"))
    save_json(OUT / "panel_outcomes.json", rows)
    print([(r["source"], r["game_idx"], r["winner"]) for r in rows], flush=True)


def warmup() -> None:
    """記録に含まれる第40試合のwarmupも、公式41試合開始後のWIN差で照合する。"""
    rows = json.loads((OUT / "panel_outcomes.json").read_text())
    video = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/video_zenchi_c0BQoMJwwQU.mp4")
    cap = cv2.VideoCapture(str(video))
    detector = MatchWinnerDetector.load_default()
    start, end = 2550.566, 2579.066
    observations = [asdict(detector.detect_winner(cap, start, end, offset, offset)) for offset in OFFSETS]
    counts = Counter(r["winner"] for r in observations if r["winner"] is not None)
    winner = next(iter(counts)) if len(counts) == 1 and max(counts.values()) >= MIN_AGREEMENT else None
    rows = [r for r in rows if r["source"] != "zenchi"]
    rows.append(dict(source="zenchi", game_idx=0, start=start, end=end,
                     winner=winner, observations=observations, method="WIN_panel_difference"))
    save_json(OUT / "panel_outcomes.json", rows)
    cap.release()
    print(rows[-1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true")
    options = parser.parse_args()
    warmup() if options.warmup else main()
