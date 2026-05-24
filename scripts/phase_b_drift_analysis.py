"""Phase B-9: drift 詳細解析 — STABLE 中の CNN 誤認識を診断.

BoardStateMachine が STABLE 確定中は「直近 STABLE 盤面 = 真値」と
みなせる。その frame で観測される CNN cell drift = CNN の単 frame 誤認識。

集計:
    1. 動画別 単 frame 精度 (= 1 - drift cell 数 / 全 cell 数)
    2. cell 位置 (row, col) ごとの drift 頻度ヒートマップ
    3. confusion matrix (true_color → cnn_color)
    4. side (1P/2P) 別統計

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_b_drift_analysis \
        --duration 30 --fps 10 --per-video-model

出力:
    data/phase_b_drift_analysis.tsv  (動画別 1 行サマリ)
    標準出力: ヒートマップ + confusion matrix
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN, HIDDEN_ROWS,
    VISIBLE_ROWS,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# 集計対象 cell (可視 12 行 + 全 6 列)
_TARGET_ROWS: list[int] = list(range(HIDDEN_ROWS, HIDDEN_ROWS + VISIBLE_ROWS))
_TARGET_COLS: list[int] = list(range(BOARD_COLS))

_COLOR_NAME = {
    0: "EM", 1: "RD", 2: "BL", 3: "GR",
    4: "YE", 5: "PU", 9: "OJ", 10: "??",
}


def get_match1(video_id: int) -> tuple[float, float] | None:
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
            try:
                return float(rows[1][1]), float(rows[1][2])
            except (IndexError, ValueError):
                continue
    return None


def select_cnn_model(
    video_id: int, per_video: bool, single_model: Path | None,
) -> Path | None:
    if per_video:
        from src.per_video_model_selector import select_phase_b_model
        m = select_phase_b_model(video_id)
        return Path(m) if m else None
    return single_model


def analyze_video(
    video_id: int, start_sec: float, end_sec: float,
    fps_sample: float, stable_n: int,
    cnn_model: Path | None,
) -> dict | None:
    video_path = _ROOT / "data" / "frames" / f"video_{video_id:02d}.mp4"
    if not video_path.exists():
        return None
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=stable_n,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=cnn_model,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cell_drift: Counter[tuple[int, int]] = Counter()
    cell_total: Counter[tuple[int, int]] = Counter()
    confusion: Counter[tuple[int, int]] = Counter()  # (true, cnn)
    side_drift = {"1P": 0, "2P": 0}
    side_total = {"1P": 0, "2P": 0}

    interval = 1.0 / fps_sample
    t = start_sec
    frame_idx = 0
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

        for side_label, side_res in (
            ("1P", result.p1), ("2P", result.p2),
        ):
            if side_res.state != BoardState.STABLE:
                continue
            if side_res.confirmed_board is None:
                continue
            true_b = side_res.confirmed_board
            cnn_b = side_res.cnn_board
            for r in _TARGET_ROWS:
                for c in _TARGET_COLS:
                    true_v = int(true_b.get(r, c))
                    cnn_v = int(cnn_b.get(r, c))
                    if true_v == COLOR_UNKNOWN or cnn_v == COLOR_UNKNOWN:
                        continue
                    cell_total[(r, c)] += 1
                    side_total[side_label] += 1
                    if true_v != cnn_v:
                        cell_drift[(r, c)] += 1
                        side_drift[side_label] += 1
                        confusion[(true_v, cnn_v)] += 1
        frame_idx += 1
        t += interval
    cap.release()

    total = sum(side_total.values())
    drift = sum(side_drift.values())
    return {
        "video_id": video_id,
        "frames": frame_idx,
        "total_compared": total,
        "total_drift": drift,
        "accuracy": (total - drift) / total if total else 0.0,
        "side_drift": dict(side_drift),
        "side_total": dict(side_total),
        "cell_drift": dict(cell_drift),
        "cell_total": dict(cell_total),
        "confusion": dict(confusion),
    }


def print_heatmap(cell_total: dict, cell_drift: dict) -> None:
    """cell 位置別 drift 率ヒートマップ (1 列 col 0..5、row 1..12)."""
    print()
    print("[heatmap] drift rate by cell position (% of compared frames)")
    print("        " + "  ".join(f"c{c}" for c in range(BOARD_COLS)))
    for r in _TARGET_ROWS:
        rates: list[str] = []
        for c in _TARGET_COLS:
            tot = cell_total.get((r, c), 0)
            drf = cell_drift.get((r, c), 0)
            if tot == 0:
                rates.append(" - ")
            else:
                rate = 100.0 * drf / tot
                rates.append(f"{rate:4.1f}")
        print(f"  r{r:2d}  " + " ".join(rates))


def print_confusion(confusion: dict, top_k: int = 15) -> None:
    print()
    print(f"[confusion] top {top_k} (true_color → cnn_color, count)")
    items = sorted(confusion.items(), key=lambda kv: -kv[1])
    for (t, p), n in items[:top_k]:
        print(f"  {_COLOR_NAME.get(t, '??'):>3} -> {_COLOR_NAME.get(p, '??'):<3}: {n}")


def print_summary_table(per_video: list[dict]) -> None:
    print()
    print("[summary] per-video accuracy")
    print(
        f"  {'video':<5}  {'frames':>6}  {'compared':>8}  "
        f"{'drift':>6}  {'accuracy':>8}  {'1P':>6}  {'2P':>6}"
    )
    for r in per_video:
        if r is None:
            continue
        s1 = r["side_total"]["1P"]
        d1 = r["side_drift"]["1P"]
        s2 = r["side_total"]["2P"]
        d2 = r["side_drift"]["2P"]
        a1 = (s1 - d1) / s1 if s1 else 0.0
        a2 = (s2 - d2) / s2 if s2 else 0.0
        print(
            f"  v{r['video_id']:02d}    {r['frames']:>6d}  "
            f"{r['total_compared']:>8d}  {r['total_drift']:>6d}  "
            f"{r['accuracy']*100:>7.3f}%  "
            f"{a1*100:>5.2f}%  {a2*100:>5.2f}%"
        )


def aggregate(rows: list[dict]) -> tuple[dict, dict, dict]:
    cell_total: Counter[tuple[int, int]] = Counter()
    cell_drift: Counter[tuple[int, int]] = Counter()
    confusion: Counter[tuple[int, int]] = Counter()
    for r in rows:
        for k, v in r["cell_total"].items():
            cell_total[k] += v
        for k, v in r["cell_drift"].items():
            cell_drift[k] += v
        for k, v in r["confusion"].items():
            confusion[k] += v
    return dict(cell_total), dict(cell_drift), dict(confusion)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--stable-n", type=int, default=6)
    parser.add_argument("--videos", type=str, default="")
    parser.add_argument(
        "--cnn-model", type=Path, default=None,
        help="一律 CNN model path",
    )
    parser.add_argument(
        "--per-video-model", action="store_true",
        help="動画別 selector で HSV/CNN を切替",
    )
    parser.add_argument(
        "--out-tsv", type=Path,
        default=_ROOT / "data" / "phase_b_drift_analysis.tsv",
    )
    args = parser.parse_args()

    target_ids = (
        [int(s) for s in args.videos.split(",") if s.strip()]
        if args.videos else list(range(1, 20))
    )

    rows: list[dict] = []
    for vid in target_ids:
        m = get_match1(vid)
        if m is None:
            continue
        start = m[0]
        end = min(m[1], start + args.duration)
        cnn_model = select_cnn_model(
            vid, args.per_video_model, args.cnn_model,
        )
        tag = f"CNN={cnn_model.name}" if cnn_model else "HSV"
        print(f"[run] v{vid:02d} ({tag}): [{start:.1f}, {end:.1f}]", flush=True)
        r = analyze_video(
            video_id=vid, start_sec=start, end_sec=end,
            fps_sample=args.fps, stable_n=args.stable_n,
            cnn_model=cnn_model,
        )
        if r is None:
            print(f"[fail] v{vid:02d}")
            continue
        print(
            f"[done] v{vid:02d}: compared={r['total_compared']} "
            f"drift={r['total_drift']} accuracy={r['accuracy']*100:.3f}%"
        )
        rows.append(r)

    if not rows:
        print("[empty] no data")
        return 0

    print_summary_table(rows)
    cell_total, cell_drift, confusion = aggregate(rows)
    print_heatmap(cell_total, cell_drift)
    print_confusion(confusion)

    # tsv 保存
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video", "frames", "compared", "drift", "accuracy",
        "1P_compared", "1P_drift", "1P_acc",
        "2P_compared", "2P_drift", "2P_acc",
    ]
    with args.out_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            s1, d1 = r["side_total"]["1P"], r["side_drift"]["1P"]
            s2, d2 = r["side_total"]["2P"], r["side_drift"]["2P"]
            w.writerow({
                "video": f"v{r['video_id']:02d}",
                "frames": r["frames"],
                "compared": r["total_compared"],
                "drift": r["total_drift"],
                "accuracy": f"{r['accuracy']*100:.3f}",
                "1P_compared": s1, "1P_drift": d1,
                "1P_acc": f"{(s1-d1)/s1*100:.3f}" if s1 else "0",
                "2P_compared": s2, "2P_drift": d2,
                "2P_acc": f"{(s2-d2)/s2*100:.3f}" if s2 else "0",
            })
    print(f"\n[saved] {to_windows_path(args.out_tsv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
