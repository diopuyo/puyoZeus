"""
試合境界を自動検出し、各試合の勝者 (1P/2P/None) を抽出する。

## 手法
1. ScoreZeroDetector + WinPanelDetector で playing→zero 遷移を試合境界として検出
   (count_match_by_score_reset.py と同ロジック)
2. MatchWinnerDetector.detect_all_winners で各ゲームの勝者を判定

## 出力
- stdout: 各試合の (start_t, end_t, winner) リスト
- --out-json: 動画別勝者系列 JSON (label_all_winners.py で利用)

## 使い方
    python -m scripts.extract_match_winners --video data/frames/video_29.mp4
    python -m scripts.extract_match_winners --video data/frames/video_29.mp4 --out-json data/indicators_v2/winners/video_29.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from src.match_winner import MatchWinnerDetector
from src.score_zero import ScoreZeroDetector
from src.win_panel import WinPanelDetector

# 試合境界検出パラメータ
SCAN_INTERVAL_SEC: float = 2.0      # スキャン間隔 (秒) - 55分動画なので2秒間隔
CONFIRM_COUNT: int = 2              # 状態確定に必要な連続フレーム数
# 勝者判定で試合開始直後の安定時刻オフセット (秒)
WINNER_OFFSET_SEC: float = 2.0
# 最終試合判定用: 動画末尾からパネル可視範囲 (秒)
LAST_PANEL_SCAN_BACK: float = 60.0


@dataclass
class GameRecord:
    """1 試合分の情報。"""
    game_abs_idx: int          # 動画内での絶対試合インデックス (0-based)
    start_sec: float           # 試合開始時刻
    end_sec: float             # 試合終了時刻 (next_start - offset、概算)
    winner: Optional[str]      # "1P" / "2P" / None
    left_hamming: int
    right_hamming: int
    confidence: str            # "strict" / "asymmetric" / "none"


def _detect_state(
    frame: cv2.Mat,
    panel_det: WinPanelDetector,
    zero_det: ScoreZeroDetector,
) -> str:
    """フレームから "none" / "zero" / "playing" を返す。"""
    panel = panel_det.detect(frame)
    if not panel.present:
        return "none"
    z = zero_det.detect(frame)
    return "zero" if z.both_zero else "playing"


def _resize_if_needed(frame: cv2.Mat) -> cv2.Mat:
    """1920x1080 以外はリサイズして返す。"""
    if frame.shape[:2] != (1080, 1920):
        return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def detect_match_starts(
    cap: cv2.VideoCapture,
    duration_sec: float,
    panel_det: WinPanelDetector,
    zero_det: ScoreZeroDetector,
) -> list[float]:
    """
    playing 状態の開始時刻リストを返す。

    遷移パターン:
      none/zero -> playing : 試合開始
    """
    confirmed: str = "none"
    pending: Optional[str] = None
    pending_count: int = 0
    match_starts: list[float] = []
    match_start_t: Optional[float] = None

    t: float = 0.0
    while t < duration_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += SCAN_INTERVAL_SEC
            continue
        frame = _resize_if_needed(frame)
        raw = _detect_state(frame, panel_det, zero_det)

        if raw == confirmed:
            pending = None
            pending_count = 0
        else:
            if pending == raw:
                pending_count += 1
            else:
                pending = raw
                pending_count = 1

            if pending_count >= CONFIRM_COUNT:
                old = confirmed
                confirmed = raw
                pending = None
                pending_count = 0

                # 試合開始: zero/none -> playing
                if confirmed == "playing" and old in ("zero", "none"):
                    match_start_t = t
                # 試合終了: playing -> zero/none
                elif old == "playing" and confirmed in ("zero", "none"):
                    if match_start_t is not None:
                        match_starts.append(match_start_t)
                        match_start_t = None

        t += SCAN_INTERVAL_SEC

    # 動画末尾でまだ playing 中の場合も収録
    if match_start_t is not None:
        match_starts.append(match_start_t)

    return match_starts


def extract_game_records(
    video_path: Path,
    panel_det: WinPanelDetector,
    zero_det: ScoreZeroDetector,
    winner_det: MatchWinnerDetector,
) -> list[GameRecord]:
    """動画全体の試合記録を抽出して返す。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けない: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    print(f"[extract_match_winners] {video_path.name}  "
          f"duration={duration_sec:.0f}s  fps={fps:.1f}")

    # 1. 試合開始時刻リストを検出
    print("  試合境界スキャン中 ...")
    match_starts = detect_match_starts(cap, duration_sec, panel_det, zero_det)
    print(f"  試合開始検出: {len(match_starts)} 件  "
          f"-> {match_starts}")

    if not match_starts:
        cap.release()
        print("  [WARNING] 試合境界が検出できなかった")
        return []

    # 2. 最終試合の end_sec 推定用 (パネルが見える末尾時刻)
    last_t = min(duration_sec, match_starts[-1] + LAST_PANEL_SCAN_BACK)

    # 3. 各試合の勝者判定
    results = winner_det.detect_all_winners(
        cap=cap,
        match_starts=match_starts,
        last_observable_sec=last_t,
        offset_before=WINNER_OFFSET_SEC,
    )
    cap.release()

    # 4. GameRecord 化
    records: list[GameRecord] = []
    for i, (start_t, res) in enumerate(zip(match_starts, results)):
        # end_sec: 次の試合開始 or 動画末尾
        end_t = match_starts[i + 1] if i + 1 < len(match_starts) else duration_sec

        # 確信度を分類
        dl = res.left_hamming
        dr = res.right_hamming
        from src.match_winner import (
            DIGIT_ASYMMETRY_MIN, DIGIT_ASYMMETRY_RATIO,
            DIGIT_DIFF_HAMMING, DIGIT_SAME_HAMMING,
        )
        if res.winner is not None:
            left_strict = dl >= DIGIT_DIFF_HAMMING and dr <= DIGIT_SAME_HAMMING
            right_strict = dr >= DIGIT_DIFF_HAMMING and dl <= DIGIT_SAME_HAMMING
            confidence = "strict" if (left_strict or right_strict) else "asymmetric"
        else:
            confidence = "none"

        records.append(GameRecord(
            game_abs_idx=i,
            start_sec=start_t,
            end_sec=end_t,
            winner=res.winner,
            left_hamming=dl,
            right_hamming=dr,
            confidence=confidence,
        ))

    return records


def print_records(records: list[GameRecord], video_id: str) -> None:
    """試合記録をコンソール出力する。"""
    print()
    print(f"=== {video_id} 勝者系列 ===")
    print(f"  {'idx':>3}  {'start':>7}  {'end':>7}  {'winner':<6}  {'L_ham':>5}  {'R_ham':>5}  {'conf'}")
    print("  " + "-" * 55)
    won_1p = sum(1 for r in records if r.winner == "1P")
    won_2p = sum(1 for r in records if r.winner == "2P")
    won_none = sum(1 for r in records if r.winner is None)
    for r in records:
        print(f"  {r.game_abs_idx:>3}  {r.start_sec:>7.1f}  {r.end_sec:>7.1f}  "
              f"{str(r.winner) or 'None':<6}  {r.left_hamming:>5}  {r.right_hamming:>5}  "
              f"{r.confidence}")
    print()
    print(f"  合計 {len(records)} 試合: 1P={won_1p}  2P={won_2p}  不明={won_none}")


def records_to_json(records: list[GameRecord], video_id: str) -> dict:
    """GameRecord リストを JSON シリアライズ可能な dict に変換する。"""
    return {
        "video_id": video_id,
        "games": [
            {
                "game_abs_idx": r.game_abs_idx,
                "start_sec": r.start_sec,
                "end_sec": r.end_sec,
                "winner": r.winner,
                "left_hamming": r.left_hamming,
                "right_hamming": r.right_hamming,
                "confidence": r.confidence,
            }
            for r in records
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="動画から試合境界と勝者を抽出する"
    )
    parser.add_argument("--video", required=True, help="対象動画パス")
    parser.add_argument(
        "--out-json", default=None,
        help="出力 JSON パス (省略時は stdout のみ)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] 動画が存在しない: {video_path}", file=sys.stderr)
        return 1

    panel_det = WinPanelDetector.load_default()
    zero_det = ScoreZeroDetector.load_default()
    winner_det = MatchWinnerDetector.load_default()

    records = extract_game_records(video_path, panel_det, zero_det, winner_det)
    video_id = video_path.stem  # "video_29"
    print_records(records, video_id)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(records_to_json(records, video_id), fp, ensure_ascii=False, indent=2)
        print(f"  JSON 保存: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
