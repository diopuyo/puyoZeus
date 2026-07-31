"""掛け算式(=score None区間)の間隔を実測 (scratch).

score OCR が None の区間 = 掛け算式表示中。
非None(読める)区間 = 掛け算式が消えている間。
「掛け算式が消えてから次が出るまで」= 非None区間の長さ を集計する。
連鎖内のステップ間ギャップ(短) と 連鎖間ギャップ(長) を分けて平均を出す。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, ".")

from src.board_state_machine import BoardState
from src.recognition_pipeline import RecognitionPipeline

VIDEO = "data/frames/video_124_4min.mp4"
TARGET_W, TARGET_H = 1920, 1080
CHAIN_STATES = {BoardState.CHAIN, BoardState.GRAVITY_SETTLE, BoardState.EFFECT}


def _runs(flags: list[bool]) -> list[int]:
    """連続 True の run 長 (フレーム数) のリスト。"""
    out, cur = [], 0
    for x in flags:
        if x:
            cur += 1
        elif cur:
            out.append(cur); cur = 0
    if cur:
        out.append(cur)
    return out


def _analyze(none_flags: list[bool], near_chain: list[bool], fps: float, label: str) -> None:
    # None区間 (掛け算式表示中)
    none_runs = [r / fps for r in _runs(none_flags)]
    # 非None区間 = 掛け算式が消えている間。連鎖近傍のみ対象で「連鎖内ステップ間」を見る
    readable = [not x for x in none_flags]
    # 連鎖近傍(前後に掛け算式がある)の readable run を抽出
    gaps_in_chain, gaps_all = [], []
    cur, start = 0, 0
    for i, x in enumerate(readable):
        if x:
            if cur == 0:
                start = i
            cur += 1
        elif cur:
            dur = cur / fps
            gaps_all.append(dur)
            # 直前・直後が None(掛け算式) かつ 近傍が連鎖 → 連鎖内ギャップ
            if start > 0 and none_flags[start - 1] and i < len(none_flags) and none_flags[i]:
                if any(near_chain[max(0, start - 3): i + 3]):
                    gaps_in_chain.append(dur)
            cur = 0

    def stat(xs: list[float]) -> str:
        if not xs:
            return "なし"
        xs2 = sorted(xs)
        return (f"n={len(xs)} 平均={sum(xs)/len(xs):.2f}s 中央={xs2[len(xs2)//2]:.2f}s "
                f"min={xs2[0]:.2f}s max={xs2[-1]:.2f}s")

    print(f"\n=== {label} ===")
    print(f"掛け算式(None)区間      : {stat(none_runs)}")
    print(f"掛け算式が消えている区間(全): {stat(gaps_all)}")
    print(f"連鎖内ステップ間ギャップ   : {stat(gaps_in_chain)}")


def main() -> None:
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    if hasattr(pipe, "set_video_id"):
        pipe.set_video_id("video_124")
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(180 * fps)
    none1, none2, ch1, ch2 = [], [], [], []
    last1: int | None = None
    last2: int | None = None
    for fi in range(n):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        r = pipe.update(fi, fi / fps, frame)
        # 掛け算式検知 = 生OCRがNone (機能D の判定を直接使う)
        f1 = RecognitionPipeline._check_formula_detected(frame, pipe._score_ocr, "1P", last1)
        f2 = RecognitionPipeline._check_formula_detected(frame, pipe._score_ocr, "2P", last2)
        none1.append(f1)
        none2.append(f2)
        ch1.append(r.p1.state in CHAIN_STATES)
        ch2.append(r.p2.state in CHAIN_STATES)
        if r.p1.score is not None:
            last1 = r.p1.score
        if r.p2.score is not None:
            last2 = r.p2.score
    cap.release()
    print(f"fps={fps:.2f} frames={len(none1)}")
    _analyze(none1, ch1, fps, "1P")
    _analyze(none2, ch2, fps, "2P")


if __name__ == "__main__":
    main()
