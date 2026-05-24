"""W7: 試合区間を StatePipeline で再認識し、補正後でも残る誤認を自動レビュー。

処理:
    各 detect_interval 秒ごとに StatePipeline.extract
    - oscillation_corrected (B2 発動数)
    - sanity_corrected (W6 4+ 連結補正数)
    - 補正後 board に残る 4+ 連結 (補正失敗)
    - 補正後 board に残る浮遊ぷよ
    - next P1/P2 不一致 (stable 取れず None)
    - score conf 低下フレーム
    を時刻別に tsv 出力。違反多発時刻 TOP10 のフレーム画像も保存。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_auto_review \
        --video data/frames/video_04.mp4 --start 9052 --end 9129 \
        --bg-fp-time 9050 --tag v04_m81
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, COLOR_UNKNOWN,
    HIDDEN_ROWS, Board,
)
from src.chain import ChainSimulator, MIN_ERASE_COUNT
from src.physics_sanity import PhysicsSanityChecker, ViolationKind
from src.state_pipeline import StatePipeline


def count_unresolved_4links(board: Board, simulator: ChainSimulator) -> int:
    """補正後 board に残る 4+ 同色連結セル数。"""
    n = 0
    for g in simulator.find_groups(board):
        if g.color in (COLOR_OJAMA,):
            continue
        if g.size >= MIN_ERASE_COUNT:
            n += g.size
    return n


def count_airborne(board: Board) -> int:
    """空中浮遊セル数 (HIDDEN_ROWS 以下、下が空)。"""
    from src.board import COLOR_EMPTY
    n = 0
    for r in range(HIDDEN_ROWS, BOARD_ROWS - 1):
        for c in range(BOARD_COLS):
            cv = int(board.get(r, c))
            if cv == COLOR_EMPTY or cv == COLOR_UNKNOWN:
                continue
            below = int(board.get(r + 1, c))
            if below == COLOR_EMPTY:
                n += 1
    return n


def count_unknown(board: Board) -> int:
    n = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if int(board.get(r, c)) == COLOR_UNKNOWN:
                n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument(
        "--bg-fp-time", type=float, default=-1.0,
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--detect-interval", type=float, default=0.5,
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="violations 多発時刻のフレーム画像保存数",
    )
    parser.add_argument(
        "--out-dir", default="data/verify/phase_w_results/auto_review",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"video open failed: {args.video}")
        return 1

    pipeline = StatePipeline()
    if args.bg_fp_time >= 0:
        ok = pipeline.set_background_fingerprints_from_video(
            cap, args.bg_fp_time,
        )
        print(f"BG FP: {'OK' if ok else 'FAIL'}")
    pipeline.reset(match_start_sec=args.start)
    simulator = ChainSimulator()
    # AnimationFilter で連鎖中を除外
    from src.animation_filter import AnimationFilter
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
    af_p1 = AnimationFilter()
    af_p2 = AnimationFilter()
    p1_roi = (
        DEFAULT_P1_REGION.x, DEFAULT_P1_REGION.y,
        DEFAULT_P1_REGION.width, DEFAULT_P1_REGION.height,
    )
    p2_roi = (
        DEFAULT_P2_REGION.x, DEFAULT_P2_REGION.y,
        DEFAULT_P2_REGION.width, DEFAULT_P2_REGION.height,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / f"{args.tag}_top_violations"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t = args.start
    while t <= args.end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok or fr is None:
            t += args.detect_interval
            continue
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)
        try:
            state = pipeline.extract(fr, t_sec=t)
        except Exception as e:
            print(f"  err t={t}: {e}")
            t += args.detect_interval
            continue

        # アニメ中判定 (連鎖中は違反扱いしない)
        anim_p1 = af_p1.is_animation(fr, p1_roi).is_animation
        anim_p2 = af_p2.is_animation(fr, p2_roi).is_animation

        # 補正後 board に残る違反
        b1 = state.board_p1
        b2 = state.board_p2
        # アニメ中はカウントしない (正常な一時状態)
        unresolved_4link_p1 = (
            0 if anim_p1 else count_unresolved_4links(b1, simulator)
        )
        unresolved_4link_p2 = (
            0 if anim_p2 else count_unresolved_4links(b2, simulator)
        )
        airborne_p1 = 0 if anim_p1 else count_airborne(b1)
        airborne_p2 = 0 if anim_p2 else count_airborne(b2)
        unknown_p1 = count_unknown(b1)
        unknown_p2 = count_unknown(b2)

        # next stable 採用率 (None なら認識ガバ)
        next_stable_count = sum([
            state.next_p1 is not None,
            state.next_p2 is not None,
            state.dnext_p1 is not None,
            state.dnext_p2 is not None,
        ])

        violations_total = (
            unresolved_4link_p1 + unresolved_4link_p2
            + airborne_p1 + airborne_p2
        )

        rows.append({
            "t_sec": t,
            "violations_total": violations_total,
            "unresolved_4link_p1": unresolved_4link_p1,
            "unresolved_4link_p2": unresolved_4link_p2,
            "airborne_p1": airborne_p1,
            "airborne_p2": airborne_p2,
            "unknown_p1": unknown_p1,
            "unknown_p2": unknown_p2,
            "next_stable_count": next_stable_count,
            "is_telop": int(state.is_telop_visible),
            "is_locked": int(state.is_match_end_locked),
            "score_conf_p1": state.score_confidence_p1,
            "score_conf_p2": state.score_confidence_p2,
        })
        t += args.detect_interval
    cap.release()

    if not rows:
        print("no rows")
        return 1

    # tsv 出力
    tsv_path = out_dir / f"{args.tag}.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")
    print(f"tsv: {to_windows_path(tsv_path)}")

    # TOP-K 違反時刻のフレーム画像保存
    sorted_rows = sorted(
        rows, key=lambda r: r["violations_total"], reverse=True,
    )
    top_rows = sorted_rows[: args.top_k]
    cap = cv2.VideoCapture(args.video)
    for i, r in enumerate(top_rows, 1):
        if r["violations_total"] == 0:
            break
        t = r["t_sec"]
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080), interpolation=cv2.INTER_AREA)
        out_path = frames_dir / (
            f"top{i:02d}_t{t:.1f}_v{r['violations_total']}.png"
        )
        cv2.imwrite(str(out_path), fr)
    cap.release()
    print(f"top-{args.top_k} frames: {to_windows_path(frames_dir)}")

    # 統計サマリ
    n = len(rows)
    total_v = sum(r["violations_total"] for r in rows)
    n_violations_frames = sum(1 for r in rows if r["violations_total"] > 0)
    avg_unknown = sum(
        r["unknown_p1"] + r["unknown_p2"] for r in rows
    ) / max(1, n)
    next_stable_rate = sum(
        r["next_stable_count"] for r in rows
    ) / (4 * max(1, n))
    print(
        f"\nsummary: {n} frames, "
        f"{n_violations_frames} with violations ({n_violations_frames / n:.1%}), "
        f"total_violations={total_v}, "
        f"avg_unknown={avg_unknown:.1f}, "
        f"next_stable_rate={next_stable_rate:.1%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
