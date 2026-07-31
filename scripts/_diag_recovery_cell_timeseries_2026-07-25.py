"""復旧セル時系列診断 (2026-07-25, read-only)。

## 背景・目的
列ゲート緩和 (enable_column_partial_support, src/board_state_machine.py) の
設計・実装前提として、「設計C 事後復旧ゲート」(_apply_stable_recovery_gate) の
stable_recovery_counters が実際どう振る舞っているかを video_c34 の実データで
確認する。判定したい仮説は2つ:

    (a) 「7まで行って毎回リセット」型: counter が STABLE_RECOVERY_MIN_FRAMES
        (=8) 直前まで積み上がるが、CNN≠HSV の瞬間的な不一致で 0 にリセット
        され続け、発火に至らない。→ 列ゲート緩和 (counter>=2 で支持) が
        効果を持ちうる。
    (b) 「一度も合意しない」型: CNN==HSV の合意自体がほとんど成立せず、
        counter がそもそも低い値からリセットされ続ける (蓄積が起きない)。
        → 列ゲート緩和では救えない (合意自体の頻度を上げる別対策が必要)。

## 対象
- video_c34: 既存 2026-07-25 系診断 (_diag_placement_confirm_frames_2026-07-25.py)
  と同一の game1 境界 (470.0-516.0s、472-512s を完全に包含するマージン込み窓)。
- 本番候補構成: enable_landing_observed_color=True +
  DriftDetector再同期ループ暴走ガード2種 (enable_drift_resync_match_start_guard,
  enable_drift_resync_hsv_gate) = True (他は RecognitionPipeline.load_default 既定)。
- 対象セル: 列0-1・row8-12 (計10セル) + 2Pの真の失敗2件 (user指定,
  t=492.0 (5,1)/(6,1)、t=492.9 (3,1)/(3,0)) を追加観測。

## 出力
- data/verify/recovery_cell_timeseries_2026-07-25/{video}_cell_timeseries.jsonl:
  毎STABLEフレーム (_apply_stable_recovery_gate 発火フレーム) の対象セルごとに
  {frame_idx, t_sec, side, r, c, cnn_v, hsv_v, confirmed_v_before/after,
  counter_before/after} を記録。
- {video}_reset_summary.json/.txt: counter が (>0 → 0) にリセットされた
  イベント一覧 (直前到達値 peak_counter, 実書込み成功か否か fired_write 込み)
  と型判定 (a)/(b) の集計。
- recovery_cell_timeseries_2026-07-25/*.png: cnn_v != hsv_v の代表フレーム
  (4-6件) のセルパッチ画像 (生 HSV 値を注記)。

## 制約
- read-only 診断。src/ は一切変更しない (monkeypatch のみ、with を抜けると復元)。
- 熱対策: cv2.setNumThreads(1)、並列しない (feedback_thermal_safety_mandatory)。
- --smoke で短窓の動作確認モードを用意する。

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/_diag_recovery_cell_timeseries_2026-07-25.py
    PYTHONPATH=. ./venv/bin/python scripts/_diag_recovery_cell_timeseries_2026-07-25.py --smoke
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# 熱対策 (feedback_thermal_safety_mandatory 準拠)。並列しない。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import src.board_state_machine as bsm  # noqa: E402
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN  # noqa: E402
from src.board_state_machine import STABLE_RECOVERY_MIN_FRAMES  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 定数
# ============================
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "recovery_cell_timeseries_2026-07-25"

# 対象窓: _diag_placement_confirm_frames_2026-07-25.py の
# C34_START_SEC/C34_MAX_SEC と同一 (game1 472-512s + 前後マージン)。
VIDEO_STEM: str = "c34"
START_SEC: float = 470.0
MAX_SEC: float = 46.0
SMOKE_MAX_SEC: float = 12.0

# 本番候補構成 (アーキ指定): 着地色補正 + Drift再同期ループ暴走ガード2種。
CANDIDATE_KWARGS: dict = {
    "enable_landing_observed_color": True,
    "enable_drift_resync_match_start_guard": True,
    "enable_drift_resync_hsv_gate": True,
}

# 対象セル: 列0-1・row8-12 (両側共通の観測域)。
DEFAULT_TARGET_ROWS: tuple[int, ...] = tuple(range(8, 13))
DEFAULT_TARGET_COLS: tuple[int, ...] = (0, 1)

# 2Pの真の失敗2件 (user指定) を追加観測。
EXTRA_TARGET_CELLS_2P: tuple[tuple[int, int], ...] = (
    (5, 1), (6, 1), (3, 1), (3, 0),
)

# 代表フレームPNG抽出の目標件数。
MIN_PATCH_FRAMES: int = 4
MAX_PATCH_FRAMES: int = 6

# 型判定 (a): peak_counter がこの値以上でのリセットを「8f直前リセット」とみなす。
NEAR_THRESHOLD_MIN: int = STABLE_RECOVERY_MIN_FRAMES - 2  # =6


def _print_progress(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _target_cells(side: str) -> set[tuple[int, int]]:
    """観測対象セル集合を返す (2P は user指定の失敗セルを追加する)。"""
    cells = {(r, c) for r in DEFAULT_TARGET_ROWS for c in DEFAULT_TARGET_COLS}
    if side == "2P":
        cells |= set(EXTRA_TARGET_CELLS_2P)
    return cells


# ============================
# データ構造
# ============================


@dataclass
class _CellFrameRecord:
    """1 (frame, side, cell) 分の観測値。"""

    frame_idx: int
    t_sec: float
    side: str
    r: int
    c: int
    cnn_v: int
    hsv_v: "int | None"
    confirmed_v_before: int
    confirmed_v_after: int
    counter_before: int
    counter_after: int


@dataclass
class _TimeseriesRecorder:
    """全フックが共有する記録先 (read-only、状態は本クラスのみ保持)。"""

    records: list[_CellFrameRecord] = field(default_factory=list)

    def record(self, rec: _CellFrameRecord) -> None:
        self.records.append(rec)


# ============================
# フック本体 (monkeypatch, read-only)
# ============================


def _make_gate_wrapper(orig, side_holder: dict, recorder: _TimeseriesRecorder):
    """board_state_machine._apply_stable_recovery_gate をラップする。

    呼び出し前後の confirmed_board / stable_recovery_counters を対象セルのみ
    比較記録する (in-place mutation を壊さないよう copy してから比較)。
    """

    @functools.wraps(orig)
    def wrapped(ctx, signals, min_frames, **kwargs):
        side = side_holder["side"]
        cells = _target_cells(side)
        confirmed_before = (
            ctx.confirmed_board.copy() if ctx.confirmed_board is not None else None
        )
        counters_before = dict(ctx.stable_recovery_counters)
        orig(ctx, signals, min_frames, **kwargs)
        confirmed_after = ctx.confirmed_board
        counters_after = ctx.stable_recovery_counters
        if confirmed_before is None or confirmed_after is None:
            return
        cnn_board = signals.cnn_board
        hsv_board = signals.hsv_board
        for r, c in cells:
            recorder.record(_CellFrameRecord(
                frame_idx=ctx.frame_idx, t_sec=ctx.time_sec, side=side, r=r, c=c,
                cnn_v=int(cnn_board.get(r, c)),
                hsv_v=(int(hsv_board.get(r, c)) if hsv_board is not None else None),
                confirmed_v_before=int(confirmed_before.get(r, c)),
                confirmed_v_after=int(confirmed_after.get(r, c)),
                counter_before=counters_before.get((r, c), 0),
                counter_after=counters_after.get((r, c), 0),
            ))

    return wrapped


def _make_step_side_wrapper(orig, side_holder: dict):
    """RecognitionPipeline._step_side をラップし、現在処理中の side を記録する。

    1P→2P の順に同期呼び出しされる前提 (feedback: 並列化しない) のため、
    共有 dict 経由でも競合しない。
    """

    @functools.wraps(orig)
    def wrapped(self, side, *args, **kwargs):
        side_holder["side"] = side
        return orig(self, side, *args, **kwargs)

    return wrapped


@contextmanager
def _install_hooks():
    """計装を一時的に有効化する (with を抜けると必ず元実装へ復元)。"""
    side_holder: dict = {"side": ""}
    recorder = _TimeseriesRecorder()
    orig_gate = bsm._apply_stable_recovery_gate
    orig_step_side = RecognitionPipeline._step_side
    bsm._apply_stable_recovery_gate = _make_gate_wrapper(orig_gate, side_holder, recorder)
    RecognitionPipeline._step_side = _make_step_side_wrapper(orig_step_side, side_holder)
    try:
        yield recorder
    finally:
        bsm._apply_stable_recovery_gate = orig_gate
        RecognitionPipeline._step_side = orig_step_side


# ============================
# 走査本体
# ============================


def _video_path(video_stem: str) -> Path:
    return VIDEO_DIR / f"video_{video_stem}.mp4"


def _scan(
    video_stem: str, start_sec: float, max_sec: float,
    *, enable_column_partial_support: bool = False,
) -> list[_CellFrameRecord]:
    """1 動画・1 窓分を計装付きで走査し、対象セルの時系列記録を返す。"""
    cv2.setNumThreads(1)
    video_path = _video_path(video_stem)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + max_sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipe = RecognitionPipeline.load_default(
        enable_column_partial_support=enable_column_partial_support,
        **CANDIDATE_KWARGS,
    )
    pipe.set_video_id(video_stem)

    t0 = time.time()
    with _install_hooks() as recorder:
        fi = start_frame
        n_read = 0
        while fi < end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            t = fi / fps
            pipe.update(fi, t, frame)
            fi += 1
            n_read += 1
            if n_read % 300 == 0:
                _print_progress(
                    f"[{video_stem}] t={t:.1f}s まで処理済み "
                    f"({n_read} frames, {time.time() - t0:.1f}s)",
                )
        cap.release()
    _print_progress(
        f"[{video_stem}] 走査完了 ({time.time() - t0:.1f}s) "
        f"records={len(recorder.records)}",
    )
    return recorder.records


# ============================
# 集計: リセットイベント検出 + 型判定
# ============================


def _detect_reset_events(records: list[_CellFrameRecord]) -> list[dict]:
    """counter が (>0 → 0) にリセットされたイベント一覧を抽出する。

    counter_before/after は同一フックコール内で比較しているため、
    フレーム間の突合処理なしに直接判定できる。
    fired_write: このフレームで confirmed_v が実際に変化したか
    (= 復旧成功によるリセットか、単なる不一致によるリセットかの区別)。
    """
    events: list[dict] = []
    for rec in records:
        if rec.counter_before > 0 and rec.counter_after == 0:
            events.append({
                "side": rec.side, "r": rec.r, "c": rec.c,
                "frame_idx": rec.frame_idx, "t_sec": rec.t_sec,
                "peak_counter": rec.counter_before,
                "fired_write": rec.confirmed_v_before != rec.confirmed_v_after,
            })
    return events


def _judge_pattern(records: list[_CellFrameRecord], reset_events: list[dict]) -> dict:
    """型判定 (a)「8f直前で毎回リセット」 vs (b)「一度も合意しない」を集計する。"""
    non_fire_resets = [e for e in reset_events if not e["fired_write"]]
    near_threshold_resets = [
        e for e in non_fire_resets if e["peak_counter"] >= NEAR_THRESHOLD_MIN
    ]
    n_cnn_hsv_agree = sum(
        1 for r in records if r.hsv_v is not None and r.cnn_v == r.hsv_v
    )
    n_total = len(records)
    agree_rate = (100.0 * n_cnn_hsv_agree / n_total) if n_total else None
    verdict = (
        "(a) 8f直前で毎回リセット型"
        if len(near_threshold_resets) >= max(1, len(non_fire_resets) // 2)
        else "(b) 一度も合意しない型"
    )
    return {
        "n_records": n_total,
        "n_cnn_hsv_agree_frames": n_cnn_hsv_agree,
        "cnn_hsv_agree_rate_pct": agree_rate,
        "n_reset_events_total": len(reset_events),
        "n_reset_events_non_fire": len(non_fire_resets),
        "n_reset_events_near_threshold": len(near_threshold_resets),
        "near_threshold_min": NEAR_THRESHOLD_MIN,
        "verdict": verdict,
    }


# ============================
# 出力
# ============================


def _write_jsonl(records: list[_CellFrameRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({
                "frame_idx": rec.frame_idx, "t_sec": rec.t_sec, "side": rec.side,
                "r": rec.r, "c": rec.c, "cnn_v": rec.cnn_v, "hsv_v": rec.hsv_v,
                "confirmed_v_before": rec.confirmed_v_before,
                "confirmed_v_after": rec.confirmed_v_after,
                "counter_before": rec.counter_before, "counter_after": rec.counter_after,
            }, ensure_ascii=False) + "\n")


def _write_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _format_summary_text(judge: dict, reset_events: list[dict]) -> str:
    lines = [
        "==== 復旧セル時系列診断 (2026-07-25) ====",
        f"対象記録数: {judge['n_records']}",
        f"CNN==HSV 合意率: {judge['cnn_hsv_agree_rate_pct']}",
        f"リセットイベント計: {judge['n_reset_events_total']} "
        f"(非発火リセット: {judge['n_reset_events_non_fire']}, "
        f"8f直前({judge['near_threshold_min']}以上)リセット: "
        f"{judge['n_reset_events_near_threshold']})",
        f"判定: {judge['verdict']}",
        "--- リセットイベント詳細 (先頭20件) ---",
    ]
    for e in reset_events[:20]:
        lines.append(
            f"  side={e['side']} cell=({e['r']},{e['c']}) t={e['t_sec']:.2f}s "
            f"peak_counter={e['peak_counter']} fired_write={e['fired_write']}",
        )
    return "\n".join(lines)


# ============================
# 代表フレームPNG出力 (cnn_v != hsv_v)
# ============================


# 盤面グリッドのセルサイズ推定用 (1920x1080 前提、既存 viz スクリプトの
# 盤面ROIと同一の大まかな目安値。診断用の粗いクロップで十分なため、
# 厳密な ROI 座標算出ロジックへの依存は避ける)。
_BOARD_ROI_1P: tuple[int, int, int, int] = (60, 90, 420, 900)  # x0,y0,x1,y1 目安
_BOARD_ROI_2P: tuple[int, int, int, int] = (1500, 90, 1860, 900)


def _cell_patch_bbox(side: str, r: int, c: int) -> tuple[int, int, int, int]:
    """診断用の粗い セル bbox を推定する (可視化目的、認識ロジックには使わない)。"""
    x0, y0, x1, y1 = _BOARD_ROI_1P if side == "1P" else _BOARD_ROI_2P
    cell_w = (x1 - x0) / BOARD_COLS
    cell_h = (y1 - y0) / BOARD_ROWS
    cx0 = int(x0 + c * cell_w)
    cy0 = int(y0 + r * cell_h)
    return cx0, cy0, int(cx0 + cell_w), int(cy0 + cell_h)


def _extract_patch_frames(
    video_stem: str, records: list[_CellFrameRecord],
) -> None:
    """cnn_v != hsv_v の代表フレーム (4-6件) をセルパッチPNGとして出力する。"""
    mismatches = [r for r in records if r.hsv_v is not None and r.cnn_v != r.hsv_v]
    if not mismatches:
        _print_progress("[patch] cnn_v!=hsv_v フレームなし (PNG出力スキップ)")
        return
    # frame_idx でソートし、時系列的に分散した代表例を選ぶ (先頭/中間/末尾寄り)。
    mismatches.sort(key=lambda r: r.frame_idx)
    n_pick = min(MAX_PATCH_FRAMES, max(MIN_PATCH_FRAMES, min(len(mismatches), MIN_PATCH_FRAMES)))
    n_pick = min(n_pick, len(mismatches))
    idxs = sorted(set(
        int(round(i * (len(mismatches) - 1) / max(1, n_pick - 1))) for i in range(n_pick)
    ))
    picks = [mismatches[i] for i in idxs]

    video_path = _video_path(video_stem)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for rec in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(rec.frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        x0, y0, x1, y1 = _cell_patch_bbox(rec.side, rec.r, rec.c)
        patch = frame[max(0, y0):y1, max(0, x0):x1].copy()
        if patch.size == 0:
            continue
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_med = float(np.median(hsv_patch[:, :, 0]))
        s_med = float(np.median(hsv_patch[:, :, 1]))
        v_med = float(np.median(hsv_patch[:, :, 2]))
        annotated = cv2.resize(patch, (200, 200), interpolation=cv2.INTER_NEAREST)
        label = f"cnn={rec.cnn_v} hsv={rec.hsv_v} H={h_med:.0f} S={s_med:.0f} V={v_med:.0f}"
        cv2.putText(
            annotated, label, (4, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
            (0, 255, 0), 1, cv2.LINE_AA,
        )
        out_name = (
            f"{video_stem}_{rec.side}_r{rec.r}c{rec.c}_f{rec.frame_idx}_"
            f"t{rec.t_sec:.2f}.png"
        )
        cv2.imwrite(str(OUTPUT_DIR / out_name), annotated)
        _print_progress(f"[patch] 出力: {out_name} ({label})")
    cap.release()


# ============================
# CLI / main
# ============================


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="復旧セル時系列診断 (2026-07-25)")
    ap.add_argument("--smoke", action="store_true", help="短窓のみ処理する動作確認モード")
    ap.add_argument(
        "--enable-column-partial-support", dest="enable_column_partial_support",
        action="store_true", default=False,
        help="既定False。指定時は enable_column_partial_support (列ゲート緩和) を"
             "有効化した構成で走査する (型判定diagは通常False運用)。",
    )
    return ap.parse_args()


def main() -> None:
    cv2.setNumThreads(1)
    args = _parse_args()
    max_sec = SMOKE_MAX_SEC if args.smoke else MAX_SEC
    if args.smoke:
        _print_progress("[SMOKE MODE] 短窓のみ処理します (本走行ではありません)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _scan(
        VIDEO_STEM, START_SEC, max_sec,
        enable_column_partial_support=args.enable_column_partial_support,
    )
    _write_jsonl(records, OUTPUT_DIR / f"{VIDEO_STEM}_cell_timeseries.jsonl")

    reset_events = _detect_reset_events(records)
    judge = _judge_pattern(records, reset_events)
    _write_json(
        {"judge": judge, "reset_events": reset_events},
        OUTPUT_DIR / f"{VIDEO_STEM}_reset_summary.json",
    )
    text = _format_summary_text(judge, reset_events)
    (OUTPUT_DIR / f"{VIDEO_STEM}_reset_summary.txt").write_text(text, encoding="utf-8")
    print(text)

    _extract_patch_frames(VIDEO_STEM, records)
    _print_progress(f"[DONE] 出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
