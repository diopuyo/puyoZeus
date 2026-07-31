"""大ROI走査の間引き (enable_large_roi_throttle) の採否検証 (2026-07-31)。

間引きは **走査を飛ばすので原理的に bit-identical にならない**唯一の施策。
リスクは「試合終了演出の検出が最大 LARGE_ROI_THROTTLE_FRAMES(=4) 遅れ、
lockdown (盤面凍結) の開始が遅れる」こと。

そこで **試合終了時刻をまたぐ窓**を勝利数パネルのデータから選び、
OFF/ON で以下を測る:
  1. hard_match_off / lockdown の立ち上がりフレームのずれ
  2. is_match_active の立ち下がりのずれ
  3. 確定盤面の差分 (どのフレームでどれだけ違うか)
  4. 速度

試合中だけの窓では match_end/telop が発火しないので差分ゼロになり、
「検証した」ことにならない (2026-07-30 の初回測定がまさにそれだった)。

窓の出典: data/verify/winners_panel_diff_2026-07-26/video_*.json
(勝利数パネル基準の試合境界。93動画分が存在)

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_large_roi_throttle_ab_2026-07-31 \
        --videos video_c56 video_c60 video_c65
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

TARGET_W, TARGET_H = 1920, 1080
WARMUP_FRAMES: int = 5
# 試合終了時刻の前後に取るフレーム数 (30fps 換算で前 5 秒 / 後 5 秒)
PRE_END_FRAMES: int = 150
POST_END_FRAMES: int = 150
# 検証する試合終了イベントの数 (動画あたり)
N_END_EVENTS: int = 2


def _load_match_ends(video_name: str, panel_dir: Path) -> list[float]:
    """勝利数パネルのデータから試合終了時刻 (秒) を取り出す。"""
    path = panel_dir / f"{video_name}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [float(g["end_sec"]) for g in data.get("games", [])]


def _read_window(
    video: Path, end_sec: float,
) -> tuple[list[np.ndarray], int]:
    """試合終了時刻をまたぐ窓を読み出す。

    Returns:
        (フレーム列, 窓内で終了時刻に対応する index)。
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けない: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    end_frame = int(end_sec * fps)
    start_frame = max(0, end_frame - PRE_END_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    for _ in range(PRE_END_FRAMES + POST_END_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != TARGET_W or frame.shape[0] != TARGET_H:
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        frames.append(frame)
    cap.release()
    return frames, end_frame - start_frame


def _snapshot(result: object) -> tuple:
    """1 フレームの (試合中フラグ, 1P/2P 盤面) を比較可能な形にする。"""
    parts: list = [getattr(result, "is_match_active", None)]
    for attr in ("side_1p", "side_2p"):
        sr = getattr(result, attr, None)
        board = getattr(sr, "confirmed_board", None) if sr is not None else None
        grid = getattr(board, "grid", None) if board is not None else None
        parts.append(None if grid is None else np.asarray(grid).tobytes())
    return tuple(parts)


def _run(frames: list[np.ndarray], throttle: bool) -> tuple[list[tuple], float]:
    """パイプラインを 1 本走らせ、(フレーム毎スナップショット, 定常中央 ms)。"""
    from src.recognition_pipeline import RecognitionPipeline

    kwargs = {"enable_large_roi_throttle": True} if throttle else {}
    pipe = RecognitionPipeline.load_default(**kwargs)
    snaps: list[tuple] = []
    times: list[float] = []
    for idx, frame in enumerate(frames):
        t0 = time.perf_counter()
        res = pipe.update(idx, idx / 30.0, frame)
        times.append((time.perf_counter() - t0) * 1000.0)
        snaps.append(_snapshot(res))
    arr = np.asarray(times)
    steady = arr[WARMUP_FRAMES:] if arr.size > WARMUP_FRAMES else arr
    return snaps, float(np.median(steady))


def _first_active_falling_edge(snaps: list[tuple]) -> int | None:
    """is_match_active が True → False になった最初の index。"""
    for i in range(1, len(snaps)):
        if snaps[i - 1][0] is True and snaps[i][0] is False:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--videos", nargs="+", default=["video_c56", "video_c60", "video_c65"],
    )
    ap.add_argument("--video-dir", type=Path, default=Path("data/frames"))
    ap.add_argument(
        "--panel-dir", type=Path,
        default=Path("data/verify/winners_panel_diff_2026-07-26"),
    )
    args = ap.parse_args()

    cv2.setNumThreads(1)
    print(
        f"窓: 各試合終了時刻の前 {PRE_END_FRAMES}f / 後 {POST_END_FRAMES}f、"
        f"動画あたり {N_END_EVENTS} イベント / cv_threads=1\n"
    )

    tot_events = 0
    tot_board_diff = 0
    tot_frames = 0
    edge_shifts: list[int] = []
    for name in args.videos:
        path = args.video_dir / f"{name}.mp4"
        ends = _load_match_ends(name, args.panel_dir)
        if not path.exists() or not ends:
            print(f"[skip] 動画または境界データ不在: {name}")
            continue
        # 動画の後半から選ぶ (前半は試合が始まっていない可能性がある)
        picks = ends[len(ends) // 2: len(ends) // 2 + N_END_EVENTS]
        for end_sec in picks:
            frames, end_idx = _read_window(path, end_sec)
            if len(frames) < 60:
                print(f"[skip] {name} end={end_sec}s: フレーム不足 {len(frames)}")
                continue
            off, ms_off = _run(frames, throttle=False)
            on, ms_on = _run(frames, throttle=True)
            n = min(len(off), len(on))
            board_diff = sum(
                1 for i in range(n)
                if off[i][1] != on[i][1] or off[i][2] != on[i][2]
            )
            e_off = _first_active_falling_edge(off)
            e_on = _first_active_falling_edge(on)
            shift = (
                (e_on - e_off) if (e_off is not None and e_on is not None) else None
            )
            if shift is not None:
                edge_shifts.append(shift)
            tot_events += 1
            tot_frames += n
            tot_board_diff += board_diff
            gain = 100.0 * (ms_off - ms_on) / ms_off if ms_off else 0.0
            print(
                f"{name} end={end_sec}s (窓内 index {end_idx}): {n}フレーム  "
                f"速度 {ms_off:.1f} → {ms_on:.1f}ms ({gain:+.1f}%)"
            )
            print(
                f"  試合中フラグの立ち下がり: OFF={e_off} / ON={e_on} "
                f"→ ずれ {shift if shift is not None else '検出なし'} フレーム"
            )
            print(
                f"  盤面差分: {board_diff}/{n} フレーム "
                f"({100.0 * board_diff / max(1, n):.2f}%)"
            )

    print(f"\n=== 合計 ({tot_events} イベント) ===")
    print(
        f"盤面差分 {tot_board_diff}/{tot_frames} フレーム "
        f"({100.0 * tot_board_diff / max(1, tot_frames):.2f}%)"
    )
    if edge_shifts:
        arr = np.asarray(edge_shifts)
        print(
            f"試合終了検出のずれ: 中央 {int(np.median(arr))}f / "
            f"最大 {int(arr.max())}f / 最小 {int(arr.min())}f "
            f"(n={arr.size})"
        )
        print(
            "→ 判定基準: ずれが LARGE_ROI_THROTTLE_FRAMES(=4) 以内なら想定内。"
            "5f 以上なら別経路の遅延が乗っているので調査が必要。"
        )
    else:
        print(
            "試合終了検出のずれ: 立ち下がりを検出できず。"
            "**窓の選び方が誤っている可能性 (演出をまたいでいない)。**"
        )


if __name__ == "__main__":
    main()
