"""色→空 破壊 (postprocess_corruption) 犯人探し 診断スクリプト (2026-07-26)。

背景: measure_stable_cell_acc.py の postprocess_corruption 集計は
raw_cnn==raw_hsv (合意) なのに confirmed が異なるセルを検知するが、
ログは 100 件上限 (CORRUPTION_LOG_LIMIT) で捕捉時刻・書込経路までは分からない。
本スクリプトは 2 パス構成で「合意色を空に上書きした書込経路」を特定する:

  Pass A (時間帯スキャン): 5 秒バケット別 corruption 密度を測り、
      最悪時間帯 (犯人探しの標的窓) を特定する。read-only、hook 無し。
  Pass B (write_trace 相関): HSV-only pipeline で標的窓の raw_hsv 盤面を
      先に全量収集し (hook 無し)、その後 CNN pipeline を
      _diag_confirmed_write_trace_2026-07-25._install_write_trace_hooks で
      計装しながら再生し、corruption セル (raw_cnn==raw_hsv==色 なのに
      confirmed==EMPTY) を検知した瞬間、直近の write_trace 書込経路を
      逆引きする。

read-only 原則: src/ は一切変更しない。measure_stable_cell_acc.py /
_diag_confirmed_write_trace_2026-07-25.py の既存関数を import して再利用する
(重複実装を避ける、本体既定構成と bit-identical を保証)。

Usage:
    PYTHONPATH=. python -m scripts._diag_color_to_empty_hunt_2026-07-26 \\
        --scan --video-path data/evaluation_videos/v29_match2_156s.mp4 \\
        --video-id v29_match2_156s --bucket-sec 5.0

    PYTHONPATH=. python -m scripts._diag_color_to_empty_hunt_2026-07-26 \\
        --hunt --video-path data/evaluation_videos/v29_match2_156s.mp4 \\
        --video-id v29_match2_156s --start-sec 40.0 --max-sec 10.0
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import cv2

# 熱対策 (feedback_thermal_safety_mandatory 準拠): 単発診断、並列しない。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from scripts.measure_stable_cell_acc import (  # noqa: E402
    COLOR_NAMES, _make_pipeline_cnn, _make_pipeline_hsv_only,
)

# _diag_confirmed_write_trace_2026-07-25 はファイル名にハイフンを含み
# `from ... import` 構文 (識別子解析) では読み込めないため importlib で動的 import する
# (read-only 原則: src/ は無改修、既存モジュールの再利用のみ)。
_wt_mod = importlib.import_module("scripts._diag_confirmed_write_trace_2026-07-25")
WriteTraceRecord = _wt_mod.WriteTraceRecord
_install_write_trace_hooks = _wt_mod._install_write_trace_hooks
ROUTE_UNATTRIBUTED = _wt_mod.ROUTE_UNATTRIBUTED

OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "color_to_empty_hunt_2026-07-26"

# Step1: 時間帯スキャンのバケット幅 (秒)。マジックナンバー回避のため定数化。
DEFAULT_BUCKET_SEC: float = 5.0
# corruption 判定と write_trace 相関: 遡って書込経路を探す最大フレーム差
# (P5 復旧ゲート等は複数フレーム後に発火するため、直近 N フレームまで遡る)。
ATTRIBUTION_LOOKBACK_FRAMES: int = 30

# Pass A (時間帯スキャン) のサンプリング間隔 (秒)。with51_flags 測定と揃える
# (data/verify/cell_accuracy_recheck_2026-07-26/*.json の meta.sample_interval_sec)。
SCAN_SAMPLE_INTERVAL_SEC: float = 0.06666666

# 深掘り実例の出力上限件数。
HUNT_EXAMPLE_LIMIT: int = 8


def _corruption_cells_for_frame(
    raw_cnn_board: object, raw_hsv_board: object, confirmed_board: object,
) -> list[tuple[int, int, int, int]]:
    """1 frame・1 side の corruption セル一覧を返す (row, col, raw_val, confirmed_val)。

    measure_stable_cell_acc._check_postprocess_corruption と同一条件
    (raw_cnn==raw_hsv かつ confirmed が異なり、両者非 UNKNOWN)。
    confirmed_board が None (非 STABLE) の場合は空リストを返す。
    """
    if confirmed_board is None:
        return []
    cells: list[tuple[int, int, int, int]] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            cnn_v = int(raw_cnn_board.get(r, c))
            hsv_v = int(raw_hsv_board.get(r, c))
            conf_v = int(confirmed_board.get(r, c))
            if cnn_v != hsv_v or cnn_v == COLOR_UNKNOWN or conf_v == COLOR_UNKNOWN:
                continue
            if conf_v == cnn_v:
                continue
            cells.append((r, c, cnn_v, conf_v))
    return cells


def _scan_time_buckets(
    video_path: Path, video_id: str, bucket_sec: float,
    start_sec: float, max_sec: float,
) -> dict[float, dict[str, int]]:
    """Pass A: 時間帯 (bucket_sec 幅) 別に corruption セル数と color_to_empty 数を数える。

    hook 無し・src 無改修。measure_stable_cell_acc の pipeline 構築関数を再利用し、
    本体既定構成 (per-video HSV 手調整は除外、disable_per_video_hsv=True 相当) と
    同一条件で走らせる。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    n_frames = int(max_sec * fps)
    interval_frames = max(1, int(round(SCAN_SAMPLE_INTERVAL_SEC * fps)))

    pipe_cnn = _make_pipeline_cnn(video_id, disable_per_video_hsv=True)
    pipe_hsv = _make_pipeline_hsv_only(video_id, disable_per_video_hsv=True)

    buckets: dict[float, dict[str, int]] = {}
    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fi = start_frame + local_i
        if fi % interval_frames != 0:
            continue
        t_sec = fi / fps
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        res_cnn = pipe_cnn.update(fi, t_sec, frame)
        res_hsv = pipe_hsv.update(fi, t_sec, frame)
        bkey = (t_sec // bucket_sec) * bucket_sec
        bucket = buckets.setdefault(bkey, {"total": 0, "color_to_empty": 0, "n_stable_side": 0})
        for sr_cnn, sr_hsv in ((res_cnn.p1, res_hsv.p1), (res_cnn.p2, res_hsv.p2)):
            if sr_cnn.state != BoardState.STABLE or sr_cnn.confirmed_board is None:
                continue
            bucket["n_stable_side"] += 1
            for _r, _c, raw_v, conf_v in _corruption_cells_for_frame(
                sr_cnn.cnn_board, sr_hsv.cnn_board, sr_cnn.confirmed_board,
            ):
                bucket["total"] += 1
                if raw_v != COLOR_EMPTY and conf_v == COLOR_EMPTY:
                    bucket["color_to_empty"] += 1
    cap.release()
    return buckets


def _print_bucket_table(buckets: dict[float, dict[str, int]], out_path: Path) -> None:
    """バケット別 corruption 密度を降順で出力する (総数・色→空数)。"""
    rows = sorted(buckets.items(), key=lambda kv: -kv[1]["color_to_empty"])
    lines = ["==== 時間帯別 corruption 密度 (color_to_empty 降順) ===="]
    for bkey, v in rows:
        lines.append(
            f"  t=[{bkey:6.1f}s-{bkey + DEFAULT_BUCKET_SEC:6.1f}s) "
            f"total={v['total']:5d} color_to_empty={v['color_to_empty']:5d} "
            f"n_stable_side={v['n_stable_side']:4d}"
        )
    text = "\n".join(lines)
    print(text)
    out_path.write_text(text, encoding="utf-8")
    (out_path.with_suffix(".json")).write_text(
        json.dumps({str(k): v for k, v in buckets.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================
# Pass B: write_trace 相関 (犯人経路特定)
# ============================


def _collect_raw_hsv_boards(
    video_path: Path, video_id: str, start_sec: float, max_sec: float,
) -> dict[tuple[int, str], list[list[int]]]:
    """HSV-only pipeline を hook 無しで走らせ、frame 毎の raw_hsv 盤面を全量収集する。

    write_trace hook 計装区間 (Pass B 本体) より前に完了させることで、
    グローバル monkeypatch との interleave 汚染を避ける (設計上の必須分離)。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    n_frames = int(max_sec * fps)
    pipe_hsv = _make_pipeline_hsv_only(video_id, disable_per_video_hsv=True)

    boards: dict[tuple[int, str], list[list[int]]] = {}
    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fi = start_frame + local_i
        t_sec = fi / fps
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        res = pipe_hsv.update(fi, t_sec, frame)
        for side, sr in (("1P", res.p1), ("2P", res.p2)):
            grid = [[int(sr.cnn_board.get(r, c)) for c in range(BOARD_COLS)] for r in range(BOARD_ROWS)]
            boards[(fi, side)] = grid
    cap.release()
    return boards


def _attribute_corruption_route(
    row: int, col: int, target_val: int, frame_idx: int, side: str,
    records: list["WriteTraceRecord"],
) -> tuple[str, dict | None]:
    """corruption セル (row,col)=target_val を書いた直近の write_trace 経路を探す。

    ATTRIBUTION_LOOKBACK_FRAMES 以内・同一 side・cells に (row,col,*,target_val)
    を含む record のうち frame_idx が最大 (直近) のものを採用する。
    見つからなければ ROUTE_UNATTRIBUTED (= それ以前から持続する corruption、
    今回の走査窓より前の書込が原因の可能性)。
    """
    lo = frame_idx - ATTRIBUTION_LOOKBACK_FRAMES
    candidates = [
        rec for rec in records
        if rec.side == side and lo <= rec.frame_idx <= frame_idx
        and any(cr == row and cc == col and after == target_val for cr, cc, _before, after in rec.cells)
    ]
    if not candidates:
        return ROUTE_UNATTRIBUTED, None
    candidates.sort(key=lambda r: r.frame_idx)
    best = candidates[-1]
    return best.route_id, {"route_frame_idx": best.frame_idx, "route_meta": best.meta}


def _run_hunt(
    video_path: Path, video_id: str, start_sec: float, max_sec: float, out_stem: str,
) -> None:
    """Pass B 本体: raw_hsv 事前収集 → write_trace 計装付き CNN 再生 → 経路相関。"""
    print(f"[{time.strftime('%H:%M:%S')}] Pass A(raw_hsv収集) 開始 window={start_sec:.1f}-{start_sec + max_sec:.1f}s")
    raw_hsv_boards = _collect_raw_hsv_boards(video_path, video_id, start_sec, max_sec)
    print(f"[{time.strftime('%H:%M:%S')}] raw_hsv 収集完了 {len(raw_hsv_boards)} (frame,side) 件")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    n_frames = int(max_sec * fps)

    route_counts: dict[str, int] = {}
    examples: list[dict] = []
    with _install_write_trace_hooks(video_id) as (recorder, _matchstart_diag):
        pipe_cnn = _make_pipeline_cnn(video_id, disable_per_video_hsv=True)
        for local_i in range(n_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            fi = start_frame + local_i
            t_sec = fi / fps
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            res_cnn = pipe_cnn.update(fi, t_sec, frame)
            for side, sr_cnn in (("1P", res_cnn.p1), ("2P", res_cnn.p2)):
                hsv_grid = raw_hsv_boards.get((fi, side))
                if hsv_grid is None or sr_cnn.state != BoardState.STABLE or sr_cnn.confirmed_board is None:
                    continue
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        cnn_v = int(sr_cnn.cnn_board.get(r, c))
                        hsv_v = hsv_grid[r][c]
                        conf_v = int(sr_cnn.confirmed_board.get(r, c))
                        if cnn_v != hsv_v or cnn_v == COLOR_UNKNOWN or conf_v == COLOR_UNKNOWN:
                            continue
                        if conf_v == cnn_v:
                            continue  # 合意色と confirmed が一致 = corruption なし
                        if not (cnn_v != COLOR_EMPTY and conf_v == COLOR_EMPTY):
                            continue  # 本スクリプトは「色→空」方向のみを対象にする (task 指定)
                        route, route_info = _attribute_corruption_route(
                            r, c, conf_v, fi, side, recorder.records,
                        )
                        route_counts[route] = route_counts.get(route, 0) + 1
                        if len(examples) < HUNT_EXAMPLE_LIMIT:
                            examples.append({
                                "video": video_id, "frame_idx": fi, "t_sec": round(t_sec, 2),
                                "side": side, "row": r, "col": c,
                                "raw_cnn": COLOR_NAMES.get(cnn_v, str(cnn_v)),
                                "confirmed": COLOR_NAMES.get(conf_v, str(conf_v)),
                                "attributed_route": route, "route_info": route_info,
                            })
    cap.release()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "video": video_id, "window": {"start_sec": start_sec, "max_sec": max_sec},
        "route_counts": route_counts, "examples": examples,
        "n_write_trace_records": len(recorder.records),
    }
    out_path = OUTPUT_DIR / f"{out_stem}_hunt.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] hunt 完了 route_counts={route_counts}")
    print(f"  出力: {out_path}")


# ============================
# CLI / main
# ============================


def _parse_args() -> argparse.Namespace:
    """CLI引数をパースする。"""
    ap = argparse.ArgumentParser(description="色→空 破壊 犯人探し診断 (Pass A/B)")
    ap.add_argument("--scan", action="store_true", help="Pass A (時間帯スキャン) を実行する。")
    ap.add_argument("--hunt", action="store_true", help="Pass B (write_trace 相関) を実行する。")
    ap.add_argument("--video-path", type=str, required=True, help="動画ファイルパス。")
    ap.add_argument("--video-id", type=str, required=True, help="動画ID (per-video HSV解決用)。")
    ap.add_argument("--bucket-sec", type=float, default=DEFAULT_BUCKET_SEC)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--max-sec", type=float, default=60.0)
    ap.add_argument("--output-stem", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    """メイン処理: --scan または --hunt を実行する。"""
    cv2.setNumThreads(1)  # 熱対策・並列しない
    args = _parse_args()
    video_path = PROJ_ROOT / args.video_path
    out_stem = args.output_stem if args.output_stem is not None else args.video_id
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.scan:
        buckets = _scan_time_buckets(
            video_path, args.video_id, args.bucket_sec, args.start_sec, args.max_sec,
        )
        _print_bucket_table(buckets, OUTPUT_DIR / f"{out_stem}_scan.txt")
    if args.hunt:
        _run_hunt(video_path, args.video_id, args.start_sec, args.max_sec, out_stem)
    if not args.scan and not args.hunt:
        print("[ERROR] --scan または --hunt のいずれかを指定してください。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
