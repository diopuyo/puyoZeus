"""
試合境界を自動検出し、各試合の勝者 (1P/2P/None) を抽出する。

## 手法 (デフォルト: v系「1動画=通算リセットあり」向け)
1. ScoreZeroDetector + WinPanelDetector で playing→zero 遷移を試合境界として検出
   (count_match_by_score_reset.py と同ロジック)
2. MatchWinnerDetector.detect_all_winners で各ゲームの勝者を判定

## 手法 (--panel-diff-mode: c系「通算成績型パネル」向け)
WIN★パネルの左右カウント値が試合をまたいで累積し続け (0-0 に戻らない) 動画では、
上記のスコア0リセット前提が成立しない (playing→zero遷移が発生しない)。
その場合は次の方式に切り替える:
1. detect_panel_increments でパネル数値を定期サンプリングし、値の増分
   (=直前に1試合終了) を左右非対称ハミング距離で検出する。
2. 増分が確定した時刻を試合終了、増えた側をその試合の勝者とする。
3. 増分未確定のまま動画が終わる「進行中の試合」は勝者が決まらないため出力に含めない
   (通算カウンタの性質上、確定していない試合を数えないのが正しい)。

## 出力
- stdout: 各試合の (start_t, end_t, winner) リスト
- --out-json: 動画別勝者系列 JSON (label_all_winners.py で利用)
  スキーマは --panel-diff-mode の有無によらず同一 (GameRecord 互換)。

## 使い方
    python -m scripts.extract_match_winners --video data/frames/video_29.mp4
    python -m scripts.extract_match_winners --video data/frames/video_29.mp4 --out-json data/indicators_v2/winners/video_29.json
    python -m scripts.extract_match_winners --video data/frames/video_c34.mp4 --panel-diff-mode --out-json data/indicators_v2/winners/video_c34.json
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
import numpy as np

from src.match_winner import (
    DIGIT_DIFF_HAMMING,
    DIGIT_SAME_HAMMING,
    MatchWinnerDetector,
    compare_digit_pairs,
    extract_digit_patches,
)
from src.score_zero import ScoreZeroDetector
from src.win_panel import WinPanelDetector

# 試合境界検出パラメータ
SCAN_INTERVAL_SEC: float = 2.0      # スキャン間隔 (秒) - 55分動画なので2秒間隔
CONFIRM_COUNT: int = 2              # 状態確定に必要な連続フレーム数
# 勝者判定で試合開始直後の安定時刻オフセット (秒)
WINNER_OFFSET_SEC: float = 2.0
# 最終試合判定用: 動画末尾からパネル可視範囲 (秒)
LAST_PANEL_SCAN_BACK: float = 60.0

# --panel-diff-mode: パネル値の増分確定に必要な連続サンプル数
# (CONFIRM_COUNT と同値だが、状態遷移デバウンスとは別概念のため独立定数として持つ)
PANEL_DIFF_CONFIRM_COUNT: int = 2


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


@dataclass(frozen=True)
class PanelIncrementEvent:
    """WIN★パネル値の増分検出イベント (= 1 試合が確定した瞬間)。--panel-diff-mode 用。"""
    event_sec: float           # 増分が確定したサンプル時刻
    winner: Optional[str]      # 増えた側 ("1P" / "2P")
    left_hamming: int
    right_hamming: int
    confidence: str            # "strict" / "asymmetric"


def _resolve_confidence(dl: int, dr: int, winner: Optional[str]) -> str:
    """左右ハミング距離と勝者から確信度ラベルを返す (strict / asymmetric / none)。"""
    if winner is None:
        return "none"
    left_strict = dl >= DIGIT_DIFF_HAMMING and dr <= DIGIT_SAME_HAMMING
    right_strict = dr >= DIGIT_DIFF_HAMMING and dl <= DIGIT_SAME_HAMMING
    return "strict" if (left_strict or right_strict) else "asymmetric"


def _panel_diff_step(
    winner: Optional[str],
    pending_winner: Optional[str],
    pending_count: int,
    confirm_count: int = PANEL_DIFF_CONFIRM_COUNT,
) -> tuple[bool, Optional[str], int]:
    """
    パネル値変化候補のデバウンス判定 (純粋関数)。

    直近サンプルの比較結果 (winner) を受け取り、同じ勝者が confirm_count 回
    連続したら「増分確定」として (True, None, 0) を返す。1 フレームだけの
    ノイズ (glow アニメーション等) で誤確定しないためのガード。

    Returns:
        (confirmed, new_pending_winner, new_pending_count)
    """
    if winner is None:
        return False, None, 0
    if winner == pending_winner:
        pending_count += 1
    else:
        pending_winner = winner
        pending_count = 1
    if pending_count >= confirm_count:
        return True, None, 0
    return False, pending_winner, pending_count


def detect_panel_increments(
    cap: cv2.VideoCapture,
    panel_det: WinPanelDetector,
    duration_sec: float,
    interval_sec: float = SCAN_INTERVAL_SEC,
    confirm_count: int = PANEL_DIFF_CONFIRM_COUNT,
) -> tuple[Optional[float], list[PanelIncrementEvent]]:
    """
    WIN★パネルの左右数値を定期サンプリングし、増分イベント列を返す。

    通算成績型パネル (c系) 向け: 値は試合終了まで持続し、終了時に勝者側が
    +1 される。増分を検出した時刻を試合終了、増えた側をその試合の勝者とする。
    OCR (数字自体の読み取り) は行わず、既存の digit_signature ハミング距離
    比較 (compare_digit_pairs) を流用する。

    Args:
        cap: 動画キャプチャ
        panel_det: パネル存在検出器
        duration_sec: 走査終了時刻 (動画長)
        interval_sec: サンプリング間隔 (秒)
        confirm_count: 増分確定に必要な連続サンプル数

    Returns:
        (first_visible_sec, events)
        first_visible_sec: パネルが最初に検出された時刻 (= 最初の試合の開始とみなす)。
        パネルが一度も検出できなければ None。
    """
    first_visible: Optional[float] = None
    ref_left: np.ndarray | None = None
    ref_right: np.ndarray | None = None
    pending_winner: Optional[str] = None
    pending_count = 0
    events: list[PanelIncrementEvent] = []

    t = 0.0
    while t < duration_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += interval_sec
            continue
        frame = _resize_if_needed(frame)
        panel = panel_det.detect(frame)
        if not panel.present:
            t += interval_sec
            continue
        if first_visible is None:
            first_visible = t
        cur_left, cur_right = extract_digit_patches(frame)

        if ref_left is None:
            ref_left, ref_right = cur_left, cur_right
            t += interval_sec
            continue

        cmp_result = compare_digit_pairs(ref_left, ref_right, cur_left, cur_right)
        confirmed, pending_winner, pending_count = _panel_diff_step(
            cmp_result.winner, pending_winner, pending_count, confirm_count,
        )
        if confirmed:
            events.append(PanelIncrementEvent(
                event_sec=t,
                winner=cmp_result.winner,
                left_hamming=cmp_result.left_hamming,
                right_hamming=cmp_result.right_hamming,
                confidence=_resolve_confidence(
                    cmp_result.left_hamming, cmp_result.right_hamming, cmp_result.winner,
                ),
            ))
            ref_left, ref_right = cur_left, cur_right
        t += interval_sec

    return first_visible, events


def game_records_from_panel_diff(
    events: list[PanelIncrementEvent],
    first_visible_sec: Optional[float],
) -> list[GameRecord]:
    """
    パネル増分イベント列から GameRecord (start/end/winner) を作る。

    試合 i の開始 = 直前試合の終了時刻 (最初の試合は first_visible_sec)。
    試合 i の終了 = イベント i の検出時刻。
    増分未確定のまま動画が終わる末尾の「進行中の試合」は勝者が決まらないため
    出力に含めない (通算カウンタの性質上、未確定試合を数えないのが正しい)。
    """
    if not events or first_visible_sec is None:
        return []
    starts = [first_visible_sec] + [e.event_sec for e in events[:-1]]
    records: list[GameRecord] = []
    for i, (start_t, ev) in enumerate(zip(starts, events)):
        records.append(GameRecord(
            game_abs_idx=i,
            start_sec=start_t,
            end_sec=ev.event_sec,
            winner=ev.winner,
            left_hamming=ev.left_hamming,
            right_hamming=ev.right_hamming,
            confidence=ev.confidence,
        ))
    return records


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
    panel_diff_mode: bool = False,
) -> list[GameRecord]:
    """
    動画全体の試合記録を抽出して返す。

    Args:
        panel_diff_mode: True の場合、通算成績型 WIN★パネル (c系) 向けの
            パネル値増分検出方式に切り替える (後方互換: デフォルト False で
            既存のスコア0リセット方式のまま)。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けない: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    print(f"[extract_match_winners] {video_path.name}  "
          f"duration={duration_sec:.0f}s  fps={fps:.1f}")

    if panel_diff_mode:
        print("  [panel-diff-mode] パネル値増分スキャン中 ...")
        first_visible, events = detect_panel_increments(cap, panel_det, duration_sec)
        cap.release()
        print(f"  増分検出: {len(events)} 件  初回パネル可視: {first_visible}")
        return game_records_from_panel_diff(events, first_visible)

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
    parser.add_argument(
        "--panel-diff-mode", action="store_true",
        help="通算成績型 WIN★パネル (c系) 向け: パネル値増分検出で試合境界+勝者を判定する",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] 動画が存在しない: {video_path}", file=sys.stderr)
        return 1

    panel_det = WinPanelDetector.load_default()
    zero_det = ScoreZeroDetector.load_default()
    winner_det = MatchWinnerDetector.load_default()

    records = extract_game_records(
        video_path, panel_det, zero_det, winner_det,
        panel_diff_mode=args.panel_diff_mode,
    )
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
