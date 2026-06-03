"""
調査スクリプト: 連鎖掛け算式 score ROI の実フレーム確認。

v70m2_buf15s.mp4 を 0.05s 刻みでスキャンし、
ScoreOcr が読めなくなる区間 (= 連鎖式表示の可能性) を特定、
その前後の 1P score ROI を PNG で保存する。

Usage:
    PYTHONPATH=. venv/bin/python3.12 scripts/investigate_chain_formula.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# プロジェクトルートを想定 (PYTHONPATH=. で実行)
from src.score_ocr import (
    ScoreOcr,
    SCORE_1P_REGION,
    SCORE_2P_REGION,
    NCC_MIN_CONFIDENCE,
)

# ============================
# 定数
# ============================
VIDEO_PATH: str = "data/evaluation_videos_v2/v70m2_buf15s.mp4"
# 別候補
VIDEO_PATH_ALT: str = "data/evaluation_videos/v70_match2_113s.mp4"

# スキャン間隔 (秒)
SCAN_STEP_SEC: float = 0.05

# スコアROI が「読めない」と判定する confidence 閾値
UNREADABLE_CONF_THRESHOLD: float = NCC_MIN_CONFIDENCE

# 切り出し保存ディレクトリ
OUT_DIR: Path = Path("data/investigation/chain_formula_frames")

# 連続 unreadable が何 frame 以上続いたら「式表示区間」とみなすか
MIN_UNREADABLE_RUN: int = 3


class FrameRecord(NamedTuple):
    """1 フレームの読取り結果。"""
    t_sec: float
    frame_idx: int
    score_1p: int | None
    conf_1p: float
    score_2p: int | None
    conf_2p: float
    readable_1p: bool
    readable_2p: bool


def crop_roi(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    """y1, y2, x1, x2 の順で ROI を切り出す。"""
    y1, y2, x1, x2 = region
    return frame[y1:y2, x1:x2].copy()


def scan_video(
    path: str,
    ocr: ScoreOcr,
    step_sec: float,
) -> list[FrameRecord]:
    """動画を step_sec 刻みでスキャンし、スコア読取り結果を返す。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けない: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_sec = total_frames / fps
    print(f"[INFO] fps={fps:.2f} total_frames={total_frames} dur={dur_sec:.2f}s")

    step_frames = max(1, int(fps * step_sec))
    records: list[FrameRecord] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step_frames == 0:
            t_sec = frame_idx / fps
            # 1080p へリサイズ
            h, w = frame.shape[:2]
            if (h, w) != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            res = ocr.read(frame)
            readable_1p = res.score_1p is not None
            readable_2p = res.score_2p is not None
            records.append(FrameRecord(
                t_sec=t_sec,
                frame_idx=frame_idx,
                score_1p=res.score_1p,
                conf_1p=res.confidence_1p,
                score_2p=res.score_2p,
                conf_2p=res.confidence_2p,
                readable_1p=readable_1p,
                readable_2p=readable_2p,
            ))
        frame_idx += 1

    cap.release()
    print(f"[INFO] スキャン完了: {len(records)} レコード")
    return records


def find_unreadable_runs(
    records: list[FrameRecord],
    side: str,
    min_run: int,
) -> list[tuple[int, int]]:
    """
    side='1P' or '2P' で連続 unreadable 区間の (開始 index, 終了 index) を返す。
    """
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for i, r in enumerate(records):
        is_unreadable = not r.readable_1p if side == "1P" else not r.readable_2p
        if is_unreadable:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                length = i - run_start
                if length >= min_run:
                    runs.append((run_start, i - 1))
                run_start = None
    if run_start is not None:
        length = len(records) - run_start
        if length >= min_run:
            runs.append((run_start, len(records) - 1))
    return runs


def save_roi_png(
    cap: cv2.VideoCapture,
    frame_idx: int,
    fps: float,
    out_path: Path,
    side: str = "1P",
) -> None:
    """指定フレームの score ROI を PNG で保存。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ret, frame = cap.read()
    if not ret:
        print(f"[WARN] frame {frame_idx} 読込失敗")
        return
    h, w = frame.shape[:2]
    if (h, w) != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    region = SCORE_1P_REGION if side == "1P" else SCORE_2P_REGION
    roi = crop_roi(frame, region)
    cv2.imwrite(str(out_path), roi)
    t_sec = frame_idx / fps
    print(f"  saved {out_path.name}  (frame={frame_idx}, t={t_sec:.3f}s)")


def save_full_frame_png(
    cap: cv2.VideoCapture,
    frame_idx: int,
    fps: float,
    out_path: Path,
) -> None:
    """指定フレームのフルフレームを PNG で保存 (スコアエリアを赤枠付きで)。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ret, frame = cap.read()
    if not ret:
        print(f"[WARN] frame {frame_idx} 読込失敗")
        return
    h, w = frame.shape[:2]
    if (h, w) != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    # 1P 赤枠
    y1, y2, x1, x2 = SCORE_1P_REGION
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # 2P 青枠
    y1b, y2b, x1b, x2b = SCORE_2P_REGION
    cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (255, 0, 0), 2)
    # 縮小して保存 (960x540)
    small = cv2.resize(frame, (960, 540))
    cv2.imwrite(str(out_path), small)
    t_sec = frame_idx / fps
    print(f"  saved {out_path.name}  (frame={frame_idx}, t={t_sec:.3f}s)")


def print_score_trace(
    records: list[FrameRecord],
    start_idx: int,
    end_idx: int,
    context: int = 5,
) -> None:
    """区間周辺の score トレースを表示。"""
    lo = max(0, start_idx - context)
    hi = min(len(records) - 1, end_idx + context)
    print(f"\n  [score trace t={records[lo].t_sec:.2f}s .. {records[hi].t_sec:.2f}s]")
    for r in records[lo:hi + 1]:
        marker = ""
        if start_idx <= records.index(r) <= end_idx:
            marker = "*** UNREADABLE"
        print(
            f"    t={r.t_sec:7.3f}s  "
            f"1P={str(r.score_1p):>10} (conf={r.conf_1p:.3f})  "
            f"2P={str(r.score_2p):>10} (conf={r.conf_2p:.3f})  "
            f"{marker}"
        )


def main() -> None:
    """メイン調査ロジック。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 動画パスを決定
    video_path = VIDEO_PATH
    if not Path(video_path).exists():
        video_path = VIDEO_PATH_ALT
    if not Path(video_path).exists():
        raise FileNotFoundError(f"動画が見つかない: {VIDEO_PATH}, {VIDEO_PATH_ALT}")

    print(f"[INFO] 調査対象: {video_path}")
    ocr = ScoreOcr.load_default()

    # 1. 全フレームスキャン
    records = scan_video(video_path, ocr, SCAN_STEP_SEC)

    # 2. 1P 側の unreadable 区間を特定
    runs_1p = find_unreadable_runs(records, "1P", MIN_UNREADABLE_RUN)
    runs_2p = find_unreadable_runs(records, "2P", MIN_UNREADABLE_RUN)

    print(f"\n[RESULT] 1P unreadable 区間: {len(runs_1p)} 件")
    for s, e in runs_1p:
        t_s = records[s].t_sec
        t_e = records[e].t_sec
        n_frames = e - s + 1
        print(f"  t={t_s:.3f}s .. {t_e:.3f}s ({n_frames} scan-steps = {(t_e - t_s):.3f}s)")

    print(f"\n[RESULT] 2P unreadable 区間: {len(runs_2p)} 件")
    for s, e in runs_2p:
        t_s = records[s].t_sec
        t_e = records[e].t_sec
        n_frames = e - s + 1
        print(f"  t={t_s:.3f}s .. {t_e:.3f}s ({n_frames} scan-steps = {(t_e - t_s):.3f}s)")

    # 3. 代表的な区間について ROI と full frame を保存
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    saved_count = 0
    # 最大 3 区間を保存
    for run_idx, (s, e) in enumerate(runs_1p[:3]):
        print(f"\n--- 1P unreadable 区間 #{run_idx + 1} ---")
        print_score_trace(records, s, e)

        # 区間直前 (= 正常スコア表示の最後)
        if s > 0:
            r_before = records[s - 1]
            fi = r_before.frame_idx
            save_roi_png(
                cap, fi, fps,
                OUT_DIR / f"run{run_idx + 1}_before_roi1p.png", "1P",
            )
            save_full_frame_png(
                cap, fi, fps,
                OUT_DIR / f"run{run_idx + 1}_before_full.png",
            )
        # 区間中央 (= 式表示の可能性が最も高い)
        mid = (s + e) // 2
        r_mid = records[mid]
        save_roi_png(
            cap, r_mid.frame_idx, fps,
            OUT_DIR / f"run{run_idx + 1}_mid_roi1p.png", "1P",
        )
        save_roi_png(
            cap, r_mid.frame_idx, fps,
            OUT_DIR / f"run{run_idx + 1}_mid_roi2p.png", "2P",
        )
        save_full_frame_png(
            cap, r_mid.frame_idx, fps,
            OUT_DIR / f"run{run_idx + 1}_mid_full.png",
        )
        # 区間直後 (= 連鎖後スコアが確定した最初)
        if e + 1 < len(records):
            r_after = records[e + 1]
            fi = r_after.frame_idx
            save_roi_png(
                cap, fi, fps,
                OUT_DIR / f"run{run_idx + 1}_after_roi1p.png", "1P",
            )
            save_full_frame_png(
                cap, fi, fps,
                OUT_DIR / f"run{run_idx + 1}_after_full.png",
            )
        saved_count += 1

    cap.release()

    # 4. score 変化のサマリ表示 (最初の 20 秒分)
    print("\n[SCORE TRACE 全体 (0-30秒)]")
    prev_1p: int | None = None
    for r in records:
        if r.t_sec > 30.0:
            break
        delta = ""
        if r.score_1p is not None and prev_1p is not None:
            d = r.score_1p - prev_1p
            if d != 0:
                delta = f" delta={d:+d}"
        if r.score_1p != prev_1p:
            print(
                f"  t={r.t_sec:7.3f}s  "
                f"1P={str(r.score_1p):>10} (conf={r.conf_1p:.3f})"
                f"  2P={str(r.score_2p):>10} (conf={r.conf_2p:.3f})"
                f"{delta}"
            )
        prev_1p = r.score_1p

    print(f"\n[DONE] PNG 保存先: {OUT_DIR.absolute()}")
    print(f"  保存区間数: {saved_count}")


if __name__ == "__main__":
    main()
