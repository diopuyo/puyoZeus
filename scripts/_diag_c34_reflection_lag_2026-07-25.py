"""真因診断: video_c34 (30fps) の反映遅延・初期盤面汚染・empty率 実測 (2026-07-25)。

完全 read-only 診断スクリプト。src/ および既存 scripts/ は一切変更しない。
新規ファイルのみ、labeled_win/boards への書込みもしない。

## 依頼背景 (user 実画面レビュー、video_c34 game1 465.6-511.8s)
1. 置いてから盤面反映までが遅すぎる (特に 2P)
2. 試合開始直後に青2個が既に盤面にある (コールドスタート汚染疑い)
3. 反映できる瞬間もあるが、かなりの時間 empty や誤認が多い

c34 は 30fps (既存診断4動画 c62/30/35/38 は 60fps)。stable_frame_count=6 /
empty_to_color 3票 / P5復旧8フレーム 等の確定ロジックは全てフレーム数ベース
のため、30fps では実時間 2 倍かかる仮説を検証する。

## 計測
1. 反映遅延分布: TSUMO_FALL 状態を抜けたフレーム (=着地) を起点に、
   その 2 セルが confirmed_board に (色を問わず) 非空で現れるまでの
   実時間・フレーム数を 1P/2P 別に集計する。
2. 初期盤面汚染: game_start_sec 前後の confirmed_board をダンプする
   (書き込み元 route_id の特定は既存 _diag_confirmed_write_trace_2026-07-25.py
   を別途 --video c34 --start-sec 460 --max-sec 55 で実行して突合する)。
3. empty 率時系列: confirmed_board のぷよ数と cnn_board (生観測) のぷよ数の
   差分を時系列バケットで集計する (「かなりの時間 empty」の定量化)。
4. 30fps 影響の直接確認: 計測1 の遅延をフレーム数でも出し、60fps 動画
   (video_30, idx1 ゲーム開始付近) の遅延フレーム数と比較する。

## 制約
- read-only 診断 (src/ は一切改変しない)。
- 熱対策: cv2.setNumThreads(1)、並列しない。
- --smoke で短窓のみ処理する動作確認モードを用意。

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/_diag_c34_reflection_lag_2026-07-25.py
    PYTHONPATH=. ./venv/bin/python scripts/_diag_c34_reflection_lag_2026-07-25.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# 熱対策 (feedback_thermal_safety_mandatory 準拠)。並列しない。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 定数
# ============================
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "c34_reflection_lag_2026-07-25"

# レビュー動画設定に合わせる (enable_landing_observed_color=True + 現行既定)。
ENABLE_LANDING_OBSERVED_COLOR: bool = True

# 対象窓: user指摘の video_c34 game1 (465.6-511.8s)。前後マージンで捕捉する。
DEFAULT_VIDEO: str = "c34"
DEFAULT_START_SEC: float = 460.0
DEFAULT_MAX_SEC: float = 55.0
DEFAULT_GAME_START_SEC: float = 465.6
DEFAULT_GAME_END_SEC: float = 511.8

# 30fps影響比較対象: video_30 (60fps、良好AUC動画、既存coldstart診断で
# idx1 game-start=153.0 実測済み、対照として最も文脈が揃っている)。
DEFAULT_COMPARE_VIDEO: str = "30"
DEFAULT_COMPARE_START_SEC: float = 150.0
DEFAULT_COMPARE_MAX_SEC: float = 45.0

# 反映探索の打ち切り秒数 (これを超えても反映されなければ「未反映」扱い)。
REFLECT_SEARCH_MAX_SEC: float = 20.0

# 着地セルとみなす色 (通常ツモは 1-5 の2色ペア、おじゃまはツモ落下由来ではない)。
_VALID_LANDING_COLORS: frozenset[int] = frozenset({1, 2, 3, 4, 5})
_EMPTY_LIKE: frozenset[int] = frozenset({COLOR_EMPTY, COLOR_UNKNOWN})

# empty率時系列のバケット幅 (秒)。
EMPTY_TS_BUCKET_SEC: float = 5.0

SMOKE_MAX_SEC: float = 12.0
SMOKE_COMPARE_MAX_SEC: float = 12.0


def _print_progress(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================
# データ構造
# ============================


@dataclass
class _FrameRec:
    """1 (video, side, frame) 分の観測値。"""

    frame_idx: int
    t: float
    state: str
    is_match_active: bool
    cnn_grid: np.ndarray
    confirmed_grid: "np.ndarray | None"


@dataclass
class _PlacementEvent:
    """1 回の着地イベント (TSUMO_FALL 終了) の反映遅延記録。"""

    video: str
    side: str
    frame_place: int
    t_place: float
    n_landed_cells: int
    cells: list[tuple[int, int, int]]
    frame_reflect: "int | None"
    t_reflect: "float | None"
    delay_frames: "int | None"
    delay_sec: "float | None"
    reflected_within_window: bool


# ============================
# パス1: 走査
# ============================


def _video_path(video_stem: str) -> Path:
    return VIDEO_DIR / f"video_{video_stem}.mp4"


def _collect_records(
    video_stem: str, start_sec: float, max_sec: float,
) -> tuple[list[_FrameRec], list[_FrameRec], float]:
    """video を走査し、1P/2P それぞれの frame 記録を返す (現行既定構成)。"""
    cv2.setNumThreads(1)
    video_path = _video_path(video_stem)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + max_sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    # 現行既定 (force_in_match=False = 自然な試合検出、stable_frame_count=6
    # 既定値) + レビュー動画設定 (enable_landing_observed_color=True)。
    pipe = RecognitionPipeline.load_default(
        enable_landing_observed_color=ENABLE_LANDING_OBSERVED_COLOR,
    )
    pipe.set_video_id(video_stem)

    recs_1p: list[_FrameRec] = []
    recs_2p: list[_FrameRec] = []
    fi = start_frame
    n_read = 0
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        for side_recs, side_res in ((recs_1p, r.p1), (recs_2p, r.p2)):
            side_recs.append(_FrameRec(
                frame_idx=fi, t=t, state=side_res.state.name,
                is_match_active=r.is_match_active,
                cnn_grid=side_res.cnn_board._grid.copy(),
                confirmed_grid=(
                    side_res.confirmed_board._grid.copy()
                    if side_res.confirmed_board is not None else None
                ),
            ))
        fi += 1
        n_read += 1
        if n_read % 900 == 0:
            _print_progress(f"[{video_stem}] t={t:.1f}s まで処理済み ({n_read} frames)")
    cap.release()
    return recs_1p, recs_2p, fps


# ============================
# 計測1: 反映遅延分布
# ============================


def _diff_landed_cells(
    before_grid: np.ndarray, after_grid: np.ndarray,
) -> list[tuple[int, int, int]]:
    """before=空/UNKNOWN → after=有効色 のセル一覧 (着地セル候補)。"""
    out: list[tuple[int, int, int]] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            bv, av = int(before_grid[r, c]), int(after_grid[r, c])
            if bv in _EMPTY_LIKE and av in _VALID_LANDING_COLORS:
                out.append((r, c, av))
    return out


def _find_reflect_frame(
    records: list[_FrameRec], exit_idx: int, cells: list[tuple[int, int, int]],
    fps: float,
) -> "int | None":
    """exit_idx 以降で cells 全てが confirmed_board 上で非空になる最初の index。"""
    search_limit = exit_idx + int(REFLECT_SEARCH_MAX_SEC * fps)
    end = min(search_limit, len(records))
    for k in range(exit_idx, end):
        confirmed = records[k].confirmed_grid
        if confirmed is None:
            continue
        if all(int(confirmed[r, c]) != COLOR_EMPTY for (r, c, _) in cells):
            return k
    return None


def _find_placement_events(
    records: list[_FrameRec], video: str, side: str, fps: float,
) -> list[_PlacementEvent]:
    """TSUMO_FALL 終了イベントごとに反映遅延を計測する。"""
    events: list[_PlacementEvent] = []
    for i in range(1, len(records)):
        if records[i - 1].state != "TSUMO_FALL" or records[i].state == "TSUMO_FALL":
            continue
        seg_start = i - 1
        while seg_start > 0 and records[seg_start - 1].state == "TSUMO_FALL":
            seg_start -= 1
        before_grid = records[max(0, seg_start - 1)].cnn_grid
        cells = _diff_landed_cells(before_grid, records[i].cnn_grid)
        reflect_idx = _find_reflect_frame(records, i, cells, fps) if cells else None
        events.append(_PlacementEvent(
            video=video, side=side,
            frame_place=records[i].frame_idx, t_place=records[i].t,
            n_landed_cells=len(cells), cells=cells,
            frame_reflect=(records[reflect_idx].frame_idx if reflect_idx is not None else None),
            t_reflect=(records[reflect_idx].t if reflect_idx is not None else None),
            delay_frames=(
                records[reflect_idx].frame_idx - records[i].frame_idx
                if reflect_idx is not None else None
            ),
            delay_sec=(
                records[reflect_idx].t - records[i].t if reflect_idx is not None else None
            ),
            reflected_within_window=(reflect_idx is not None),
        ))
    return events


def _delay_stats(events: list[_PlacementEvent]) -> dict:
    """n_landed_cells==2 のクリーンな着地のみを対象にした遅延分布統計。"""
    clean = [e for e in events if e.n_landed_cells == 2]
    delays_sec = [e.delay_sec for e in clean if e.delay_sec is not None]
    delays_frm = [e.delay_frames for e in clean if e.delay_frames is not None]
    n_landed_hist: dict[int, int] = {}
    for e in events:
        n_landed_hist[e.n_landed_cells] = n_landed_hist.get(e.n_landed_cells, 0) + 1
    return {
        "n_events_total": len(events),
        "n_events_clean_2cell": len(clean),
        "n_landed_cells_hist": n_landed_hist,
        "n_never_reflected_within_20s": sum(1 for e in clean if not e.reflected_within_window),
        "delay_sec_median": (float(np.median(delays_sec)) if delays_sec else None),
        "delay_sec_mean": (float(np.mean(delays_sec)) if delays_sec else None),
        "delay_sec_max": (float(np.max(delays_sec)) if delays_sec else None),
        "delay_sec_p90": (float(np.percentile(delays_sec, 90)) if delays_sec else None),
        "delay_frames_median": (float(np.median(delays_frm)) if delays_frm else None),
        "delay_frames_mean": (float(np.mean(delays_frm)) if delays_frm else None),
        "delay_frames_max": (float(np.max(delays_frm)) if delays_frm else None),
    }


# ============================
# 計測3: empty 率時系列
# ============================


def _puyo_count(grid: "np.ndarray | None") -> "int | None":
    if grid is None:
        return None
    return int(np.sum((grid != COLOR_EMPTY) & (grid != COLOR_UNKNOWN)))


def _empty_rate_timeseries(
    records: list[_FrameRec], start_sec: float,
) -> list[dict]:
    """confirmed_board のぷよ数 vs cnn_board のぷよ数の差を時系列バケットで集計する。"""
    buckets: dict[int, dict] = {}
    for rec in records:
        idx = int((rec.t - start_sec) // EMPTY_TS_BUCKET_SEC)
        b = buckets.setdefault(idx, {
            "n_frames": 0, "n_confirmed_none": 0, "gaps": [], "cnn_counts": [],
        })
        b["n_frames"] += 1
        cnn_n = _puyo_count(rec.cnn_grid)
        b["cnn_counts"].append(cnn_n)
        confirmed_n = _puyo_count(rec.confirmed_grid)
        if confirmed_n is None:
            b["n_confirmed_none"] += 1
        else:
            b["gaps"].append(cnn_n - confirmed_n)
    out: list[dict] = []
    for idx in sorted(buckets):
        b = buckets[idx]
        out.append({
            "t_bucket_start": start_sec + idx * EMPTY_TS_BUCKET_SEC,
            "n_frames": b["n_frames"],
            "pct_confirmed_none": 100.0 * b["n_confirmed_none"] / b["n_frames"],
            "mean_gap_cnn_minus_confirmed": (float(np.mean(b["gaps"])) if b["gaps"] else None),
            "mean_cnn_puyo_count": float(np.mean(b["cnn_counts"])),
        })
    return out


# ============================
# 計測2: 初期盤面ダンプ
# ============================


def _grid_to_rows(grid: "np.ndarray | None") -> "list[list[int]] | None":
    return grid.tolist() if grid is not None else None


def _dump_initial_board(
    records: list[_FrameRec], game_start_sec: float,
) -> dict:
    """game_start_sec 直前・直後の confirmed_board をダンプする。"""
    before = [r for r in records if r.t < game_start_sec]
    after = [r for r in records if r.t >= game_start_sec]
    rec_before = before[-1] if before else None
    rec_after0 = after[0] if after else None
    # is_match_active が最初に True になる時刻。
    t_active: "float | None" = None
    for r in after:
        if r.is_match_active:
            t_active = r.t
            break
    return {
        "t_before": (rec_before.t if rec_before else None),
        "confirmed_before": _grid_to_rows(rec_before.confirmed_grid if rec_before else None),
        "cnn_before": _grid_to_rows(rec_before.cnn_grid if rec_before else None),
        "t_after0": (rec_after0.t if rec_after0 else None),
        "confirmed_after0": _grid_to_rows(rec_after0.confirmed_grid if rec_after0 else None),
        "cnn_after0": _grid_to_rows(rec_after0.cnn_grid if rec_after0 else None),
        "t_match_active_true": t_active,
    }


# ============================
# 出力整形
# ============================


def _format_delay_table(label: str, stats_1p: dict, stats_2p: dict) -> str:
    lines = [f"--- {label}: 反映遅延分布 (n_landed_cells==2 のクリーン着地のみ) ---"]
    for side, s in (("1P", stats_1p), ("2P", stats_2p)):
        lines.append(
            f"  {side}: n_clean={s['n_events_clean_2cell']}/{s['n_events_total']} "
            f"未反映(20s超)={s['n_never_reflected_within_20s']} "
            f"delay_sec(中央値/平均/p90/最大)="
            f"{s['delay_sec_median']}/{s['delay_sec_mean']}/{s['delay_sec_p90']}/{s['delay_sec_max']} "
            f"delay_frames(中央値/平均/最大)="
            f"{s['delay_frames_median']}/{s['delay_frames_mean']}/{s['delay_frames_max']}",
        )
        lines.append(f"    n_landed_cells内訳: {s['n_landed_cells_hist']}")
    return "\n".join(lines)


def _write_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_events_csv(events: list[_PlacementEvent], path: Path) -> None:
    cols = [
        "video", "side", "frame_place", "t_place", "n_landed_cells",
        "frame_reflect", "t_reflect", "delay_frames", "delay_sec",
        "reflected_within_window",
    ]
    lines = [",".join(cols)]
    for e in events:
        lines.append(
            f"{e.video},{e.side},{e.frame_place},{e.t_place:.3f},{e.n_landed_cells},"
            f"{e.frame_reflect},{e.t_reflect},{e.delay_frames},{e.delay_sec},"
            f"{e.reflected_within_window}",
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_timeseries_csv(ts: list[dict], path: Path) -> None:
    cols = [
        "t_bucket_start", "n_frames", "pct_confirmed_none",
        "mean_gap_cnn_minus_confirmed", "mean_cnn_puyo_count",
    ]
    lines = [",".join(cols)]
    for row in ts:
        lines.append(",".join(str(row[c]) for c in cols))
    path.write_text("\n".join(lines), encoding="utf-8")


# ============================
# メイン
# ============================


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="短窓のみ処理する動作確認モード")
    ap.add_argument("--video", type=str, default=DEFAULT_VIDEO)
    ap.add_argument("--start-sec", type=float, default=DEFAULT_START_SEC)
    ap.add_argument("--max-sec", type=float, default=DEFAULT_MAX_SEC)
    ap.add_argument("--game-start-sec", type=float, default=DEFAULT_GAME_START_SEC)
    ap.add_argument("--compare-video", type=str, default=DEFAULT_COMPARE_VIDEO)
    ap.add_argument("--compare-start-sec", type=float, default=DEFAULT_COMPARE_START_SEC)
    ap.add_argument("--compare-max-sec", type=float, default=DEFAULT_COMPARE_MAX_SEC)
    return ap.parse_args()


def _process_one(video: str, start_sec: float, max_sec: float) -> dict:
    """1 動画分の走査 + 計測1(遅延)+3(empty率) をまとめて実行する。"""
    t0 = time.time()
    _print_progress(f"[{video}] 走査開始 start={start_sec:.1f}s dur={max_sec:.1f}s")
    recs_1p, recs_2p, fps = _collect_records(video, start_sec, max_sec)
    _print_progress(f"[{video}] 走査完了 ({time.time() - t0:.1f}s) fps={fps:.2f}")

    events_1p = _find_placement_events(recs_1p, video, "1P", fps)
    events_2p = _find_placement_events(recs_2p, video, "2P", fps)
    stats_1p = _delay_stats(events_1p)
    stats_2p = _delay_stats(events_2p)
    ts_1p = _empty_rate_timeseries(recs_1p, start_sec)
    ts_2p = _empty_rate_timeseries(recs_2p, start_sec)
    return {
        "video": video, "fps": fps, "start_sec": start_sec, "max_sec": max_sec,
        "recs_1p": recs_1p, "recs_2p": recs_2p,
        "events_1p": events_1p, "events_2p": events_2p,
        "stats_1p": stats_1p, "stats_2p": stats_2p,
        "ts_1p": ts_1p, "ts_2p": ts_2p,
    }


def main() -> None:
    cv2.setNumThreads(1)
    args = _parse_args()
    max_sec = SMOKE_MAX_SEC if args.smoke else args.max_sec
    compare_max_sec = SMOKE_COMPARE_MAX_SEC if args.smoke else args.compare_max_sec
    if args.smoke:
        _print_progress("[SMOKE MODE] 短窓のみ処理します (本走行ではありません)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main_result = _process_one(args.video, args.start_sec, max_sec)
    compare_result = _process_one(args.compare_video, args.compare_start_sec, compare_max_sec)

    initial_dump = _dump_initial_board(main_result["recs_1p"], args.game_start_sec)
    initial_dump_2p = _dump_initial_board(main_result["recs_2p"], args.game_start_sec)

    for tag, result in (("main", main_result), ("compare", compare_result)):
        _write_events_csv(
            result["events_1p"] + result["events_2p"],
            OUTPUT_DIR / f"events_{tag}_{result['video']}.csv",
        )
        _write_timeseries_csv(result["ts_1p"], OUTPUT_DIR / f"ts_1p_{tag}_{result['video']}.csv")
        _write_timeseries_csv(result["ts_2p"], OUTPUT_DIR / f"ts_2p_{tag}_{result['video']}.csv")

    summary = {
        "main": {
            "video": main_result["video"], "fps": main_result["fps"],
            "stats_1p": main_result["stats_1p"], "stats_2p": main_result["stats_2p"],
        },
        "compare": {
            "video": compare_result["video"], "fps": compare_result["fps"],
            "stats_1p": compare_result["stats_1p"], "stats_2p": compare_result["stats_2p"],
        },
        "initial_board_dump_1p": initial_dump,
        "initial_board_dump_2p": initial_dump_2p,
    }
    _write_json(summary, OUTPUT_DIR / "summary.json")

    text_lines = [
        "==== video_c34 反映遅延・初期盤面汚染・empty率 診断 (2026-07-25) ====",
        _format_delay_table(
            f"video_{main_result['video']} (fps={main_result['fps']:.1f})",
            main_result["stats_1p"], main_result["stats_2p"],
        ),
        _format_delay_table(
            f"video_{compare_result['video']} (fps={compare_result['fps']:.1f}, 比較対照)",
            compare_result["stats_1p"], compare_result["stats_2p"],
        ),
        "--- 初期盤面ダンプ (1P) ---",
        f"  game_start_sec={args.game_start_sec} 直前 t={initial_dump['t_before']} "
        f"confirmed={initial_dump['confirmed_before']}",
        f"  直後 t={initial_dump['t_after0']} confirmed={initial_dump['confirmed_after0']}",
        f"  is_match_active初True: {initial_dump['t_match_active_true']}",
        "--- 初期盤面ダンプ (2P) ---",
        f"  直前 t={initial_dump_2p['t_before']} confirmed={initial_dump_2p['confirmed_before']}",
        f"  直後 t={initial_dump_2p['t_after0']} confirmed={initial_dump_2p['confirmed_after0']}",
        f"  is_match_active初True: {initial_dump_2p['t_match_active_true']}",
    ]
    text = "\n".join(text_lines)
    (OUTPUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    _print_progress(f"[DONE] 出力先: {OUTPUT_DIR}")
    print(text)


if __name__ == "__main__":
    main()
