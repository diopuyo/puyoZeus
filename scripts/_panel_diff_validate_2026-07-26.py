"""#43 対応: --panel-diff-mode の検証スクリプト (使い捨て、本実装ではない)。

1. c系4本 (c1/c4/c34/c82) を --panel-diff-mode で再抽出し、
   data/verify/step0_winstar_cseries_2026-07-26/panel_diff_video_cN.json に保存。
2. v系2本 (v29/v33) をデフォルトモードで再抽出し、既存
   data/indicators_v2/winners/video_NN.json と完全一致するか確認 (回帰確認)。
3. c系4本について、増分検出時刻のパネルクロップを各動画3枚
   (先頭/中間/末尾の増分イベント) 出力する (userレビュー用)。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from scripts.extract_match_winners import (
    extract_game_records,
    records_to_json,
)
from src.match_winner import MatchWinnerDetector
from src.score_zero import ScoreZeroDetector
from src.win_panel import PANEL_X_RANGE, PANEL_Y_RANGE, WinPanelDetector

FRAMES_DIR = Path("data/frames")
OUT_DIR = Path("data/verify/step0_winstar_cseries_2026-07-26")
WINNERS_DIR = Path("data/indicators_v2/winners")

C_VIDEOS: list[str] = ["c1", "c4", "c34", "c82"]
V_VIDEOS: list[str] = ["29", "33"]

CROP_SCALE: int = 3


def _crop_panel(cap: cv2.VideoCapture, t_sec: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    y1, y2 = PANEL_Y_RANGE
    x1, x2 = PANEL_X_RANGE
    roi = frame[y1:y2, x1:x2].copy()
    h, w = roi.shape[:2]
    return cv2.resize(roi, (w * CROP_SCALE, h * CROP_SCALE), interpolation=cv2.INTER_NEAREST)


def run_c_series() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_det = WinPanelDetector.load_default()
    zero_det = ScoreZeroDetector.load_default()
    winner_det = MatchWinnerDetector.load_default()

    print("=== c系4本: --panel-diff-mode 検証 ===")
    for vid in C_VIDEOS:
        video_path = FRAMES_DIR / f"video_{vid}.mp4"
        if not video_path.exists():
            print(f"[SKIP] 動画なし: {video_path}")
            continue
        records = extract_game_records(
            video_path, panel_det, zero_det, winner_det, panel_diff_mode=True,
        )
        video_id = f"video_{vid}"
        n_total = len(records)
        n_unknown = sum(1 for r in records if r.winner is None)
        unknown_rate = (n_unknown / n_total) if n_total else float("nan")
        print(f"  {video_id}: 検出試合数={n_total}  不明={n_unknown}  不明率={unknown_rate:.1%}")

        out_json = OUT_DIR / f"panel_diff_{video_id}.json"
        with out_json.open("w", encoding="utf-8") as fp:
            json.dump(records_to_json(records, video_id), fp, ensure_ascii=False, indent=2)
        print(f"    JSON保存: {out_json}")

        # 目視用クロップ: 先頭/中間/末尾の増分イベント時刻
        if n_total == 0:
            continue
        idxs = sorted({0, n_total // 2, n_total - 1})
        cap = cv2.VideoCapture(str(video_path))
        for i in idxs:
            r = records[i]
            crop = _crop_panel(cap, r.end_sec)
            if crop is None:
                continue
            fname = (
                f"panel_diff_{video_id}_evt{i:02d}_t{int(r.end_sec):05d}"
                f"_winner_{r.winner or 'None'}_{r.confidence}.png"
            )
            cv2.imwrite(str(OUT_DIR / fname), crop)
            print(f"    クロップ保存: {fname}")
        cap.release()
    print()


def run_v_regression() -> None:
    panel_det = WinPanelDetector.load_default()
    zero_det = ScoreZeroDetector.load_default()
    winner_det = MatchWinnerDetector.load_default()

    print("=== v系2本: デフォルトモード回帰確認 ===")
    for vid in V_VIDEOS:
        video_path = FRAMES_DIR / f"video_{vid}.mp4"
        baseline_path = WINNERS_DIR / f"video_{vid}.json"
        if not video_path.exists() or not baseline_path.exists():
            print(f"[SKIP] 動画または baseline JSON なし: video_{vid}")
            continue
        records = extract_game_records(
            video_path, panel_det, zero_det, winner_det, panel_diff_mode=False,
        )
        video_id = f"video_{vid}"
        new_json = records_to_json(records, video_id)
        with baseline_path.open("r", encoding="utf-8") as fp:
            old_json = json.load(fp)
        same = new_json == old_json
        print(f"  {video_id}: 旧方式と完全一致={same}  "
              f"(新={len(new_json['games'])}試合 / 旧={len(old_json['games'])}試合)")
        if not same:
            print(f"    [WARNING] 差分あり。詳細は手動確認要。")
    print()


def main() -> int:
    run_c_series()
    run_v_regression()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
