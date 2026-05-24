"""Phase E-5: 新動画 (v20+) の認識精度メトリクス集計.

各動画の冒頭 N 秒 (or 試合 1 区間) を pipeline 通し:
    - 処理 frame 数
    - 両側 STABLE 率
    - 1P / 2P 別の state 分布
    - confirmed_board 取得回数
    - 連鎖イベント件数
    - 浮きぷよ検出件数 (gravity_filter で除去された数)

出力:
    - data/verify/phase_e_recognition_review.tsv (メトリクス集計)
    - 標準出力に簡易レポート

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_recognition_review \
        --videos 20-40 --duration 90 --fps 10
"""
from __future__ import annotations

import argparse
import csv
import sys
import time as time_mod
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2  # noqa: E402

from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402


def parse_videos(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def find_match1(video_id: int) -> tuple[float, float] | None:
    for ver in ("v5", "v4"):
        p = (
            _ROOT
            / f"data/verify/match_boundaries_{ver}/video_{video_id:02d}/"
            f"matches.tsv"
        )
        if not p.exists():
            continue
        with p.open() as f:
            rows = list(csv.reader(f, delimiter="\t"))
        if len(rows) > 1:
            row = rows[1]
            try:
                return float(row[1]), float(row[2])
            except (IndexError, ValueError):
                continue
    return None


def count_floating_puyo(board) -> int:
    """盤面の浮きぷよ件数 (空 cell の上の puyo)."""
    if board is None:
        return 0
    grid = board.grid if hasattr(board, "grid") else None
    if grid is None:
        return 0
    floating = 0
    n_rows, n_cols = grid.shape
    for c in range(n_cols):
        seen_empty = False
        for r in range(n_rows - 1, -1, -1):
            cell = int(grid[r, c])
            if cell == 0:
                seen_empty = True
            elif seen_empty and cell != 10:  # COLOR_UNKNOWN は除外
                floating += 1
    return floating


def run_video(
    video_id: int, duration: float, fps_sample: float,
    stable_n: int = 2, fallback_start: float = 0.0,
) -> dict | None:
    video_path = _ROOT / "data/frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        return None

    # 試合 1 が分かる場合はそこから、なければ冒頭から
    bnd = find_match1(video_id)
    if bnd is not None:
        start_sec, end_sec = bnd
        end_sec = min(end_sec, start_sec + duration)
        source = "match1"
    else:
        start_sec = fallback_start
        end_sec = fallback_start + duration
        source = "fallback"

    # Per-video CNN モデル (selector あれば適用、なければデフォルト)
    try:
        from src.per_video_model_selector import select_phase_b_model
        m = select_phase_b_model(video_id)
        cnn_model = Path(m) if m else None
    except Exception:
        cnn_model = None

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=stable_n,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=cnn_model,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    interval = 1.0 / fps_sample
    t = start_sec
    frame_idx = 0
    state_counter: Counter[tuple[BoardState, BoardState]] = Counter()
    side_p1: Counter[BoardState] = Counter()
    side_p2: Counter[BoardState] = Counter()
    chain_events_p1 = 0
    chain_events_p2 = 0
    confirmed_p1 = 0
    confirmed_p2 = 0
    floating_p1_total = 0
    floating_p2_total = 0
    cnn_floating_p1_total = 0
    cnn_floating_p2_total = 0
    last_chain_trig_p1: float | None = None
    last_chain_trig_p2: float | None = None
    t0 = time_mod.time()
    while t < end_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        result = pipe.update(frame_idx, t, frame)
        s = (result.p1.state, result.p2.state)
        state_counter[s] += 1
        side_p1[result.p1.state] += 1
        side_p2[result.p2.state] += 1
        if result.p1.confirmed_board is not None:
            confirmed_p1 += 1
            floating_p1_total += count_floating_puyo(
                result.p1.confirmed_board,
            )
        if result.p2.confirmed_board is not None:
            confirmed_p2 += 1
            floating_p2_total += count_floating_puyo(
                result.p2.confirmed_board,
            )
        # 生 CNN 出力での浮きぷよ (gravity_filter 適用前)
        cnn_floating_p1_total += count_floating_puyo(result.p1.cnn_board)
        cnn_floating_p2_total += count_floating_puyo(result.p2.cnn_board)
        # ChainEvent の trigger_sec で重複排除して件数カウント
        ce1 = result.p1.chain_event
        if ce1 is not None and ce1.trigger_sec != last_chain_trig_p1:
            chain_events_p1 += 1
            last_chain_trig_p1 = ce1.trigger_sec
        ce2 = result.p2.chain_event
        if ce2 is not None and ce2.trigger_sec != last_chain_trig_p2:
            chain_events_p2 += 1
            last_chain_trig_p2 = ce2.trigger_sec
        frame_idx += 1
        t += interval
    cap.release()
    elapsed = time_mod.time() - t0

    n_frames = frame_idx
    if n_frames == 0:
        return None
    both_stable = state_counter[(BoardState.STABLE, BoardState.STABLE)]
    p1_stable_rate = side_p1[BoardState.STABLE] / n_frames
    p2_stable_rate = side_p2[BoardState.STABLE] / n_frames
    return {
        "video_id": video_id,
        "source": source,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "n_frames": n_frames,
        "elapsed_sec": elapsed,
        "p1_stable_rate": p1_stable_rate,
        "p2_stable_rate": p2_stable_rate,
        "both_stable_rate": both_stable / n_frames,
        "confirmed_p1": confirmed_p1,
        "confirmed_p2": confirmed_p2,
        "chain_events_p1": chain_events_p1,
        "chain_events_p2": chain_events_p2,
        "floating_per_confirmed_p1": (
            floating_p1_total / confirmed_p1 if confirmed_p1 else 0.0
        ),
        "floating_per_confirmed_p2": (
            floating_p2_total / confirmed_p2 if confirmed_p2 else 0.0
        ),
        "cnn_floating_per_frame_p1": cnn_floating_p1_total / n_frames,
        "cnn_floating_per_frame_p2": cnn_floating_p2_total / n_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos", type=str, default="20-40",
        help="評価対象動画 (例: 20-40 / 1,5,20)",
    )
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--out-tsv", type=Path,
        default=_ROOT / "data/verify/phase_e_recognition_review.tsv",
    )
    args = parser.parse_args()
    target_ids = parse_videos(args.videos)
    print(f"[target] {target_ids}")

    rows: list[dict] = []
    for vid in target_ids:
        print(f"\n[run] v{vid:02d} ...")
        r = run_video(vid, args.duration, args.fps)
        if r is None:
            print(f"  [skip] v{vid:02d}: video not found or empty")
            continue
        print(
            f"  [done] frames={r['n_frames']} both_stable="
            f"{r['both_stable_rate']:.1%} chains_p1={r['chain_events_p1']} "
            f"chains_p2={r['chain_events_p2']} "
            f"floats_p1={r['floating_per_confirmed_p1']:.2f} "
            f"floats_p2={r['floating_per_confirmed_p2']:.2f} "
            f"({r['elapsed_sec']:.1f}s)"
        )
        rows.append(r)

    if not rows:
        print("[empty] no rows")
        return 0
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\n[saved] {to_windows_path(args.out_tsv)}")

    print("\n=== summary (sorted by both_stable_rate) ===")
    rows.sort(key=lambda r: -r["both_stable_rate"])
    for r in rows:
        print(
            f"  v{r['video_id']:02d}  "
            f"both_stable={r['both_stable_rate']:.1%}  "
            f"chain_p1={r['chain_events_p1']:>2d}  "
            f"chain_p2={r['chain_events_p2']:>2d}  "
            f"cnn_float_p1={r['cnn_floating_per_frame_p1']:.2f}  "
            f"cnn_float_p2={r['cnn_floating_per_frame_p2']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
