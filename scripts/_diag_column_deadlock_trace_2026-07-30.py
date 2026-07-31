"""列デッドロック根因確定トレース (c34 1P col=1, 2026-07-30)。

完全 read-only 診断スクリプト。src/ は一切変更しない (モンキーパッチは
with ブロックで必ず元へ復元する)。前例
scripts/_diag_confirmed_write_trace_2026-07-25.py の with 復元パターンを流用。

## 背景 (アーキ読解ベースの仮説・要実測確認)
c34 1P col=1 で、frame 14340 を境に r8/r9/r10 が「confirmed=空 / cnn=色」の
不一致に一斉に落ち、以後 14756 まで 16 エピソード全 run で頭から尻まで
デッドロックする (episodes_c34.csv 実測)。r11c1 は 14330 まで
color_to_empty (conf=4黄, cnn=0) だったが 14340 以降は conf==cnn==0 で
差分が消える (= r11 の confirmed が 4→0 に変化した証拠)。

読解仮説:
  (甲) 初発 14340: NON-STABLE→STABLE 復帰 merge (_merge_diff_only) で
       r11 の puyo→空 が allow_puyo_to_empty=True (呼び出し元既定) のため
       無投票で即消去され、直後の _apply_gravity_filter が r9/r10 を
       浮き扱いで連鎖消去する。
  (乙) 継続: 事後復旧ゲート (_apply_stable_recovery_gate) で r9/r10 は
       add 候補になるが、下の r11 が空かつ非候補のため列却下される
       (安全弁C, _check_recovery_column)。ただし HSV が光沢で空誤読なら
       そもそも cnn≠hsv でカウンタが 8 に届かない可能性もある (質問 a)。

## 本スクリプトが確定する 3 点
  (a) P5 復旧ゲートの却下型: column_rejected (カウンタ≥8 維持で列却下) か
      cnn_hsv_disagree / unknown_reset (8 に届かず) か。
  (b) r11c1 の HSV 値: HSV も光沢で空誤読か (CNN×HSV 独立二重合意が
      光沢で崩れる証拠か)。
  (c) 初発 frame 14340 の再現: merge で r11 4→0 の無投票消去 +
      r9/r10 の gravity filter 消去が実際に観測されるか。

## 構成 (perframe 測定 _diag_perframe_cell_accuracy_2026-07-30.py と同一)
  load_default(stable_frame_count=3, temporal_smoothing=1, force_in_match=True,
               load_score_ocr=True, enable_chain_tracker=True)
  → enable_stable_recovery_gate=True(既定) / recovery_min_frames=8 /
    enable_column_partial_support=False(既定) /
    enable_cnn_flicker_hsv_fallback=True(既定)。
  キャプチャは perframe の _scan_video を複製 (cv2 直読み + INTER_AREA
  リサイズ + pipe.update ループ) し、同一トラジェクトリを保証する。

Usage (WSL 経由):
    PYTHONPATH=. nice -n 19 ./venv/bin/python \
        scripts/_diag_column_deadlock_trace_2026-07-30.py
    (--smoke で先頭 3 秒のみ動作確認)
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

# CPU 競合対策: 基準データ収集が 8 並列稼働中のため 1 スレッドに固定する。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "1")

import cv2  # noqa: E402

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import src.board_state_machine as bsm  # noqa: E402
import src.recognition_pipeline as rp  # noqa: E402
from src.board import (  # noqa: E402
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_UNKNOWN,
)
from src.board_state_machine import RECOVERY_EXCLUDED_COLORS  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 定数
# ============================
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "column_deadlock_2026-07-30"

VIDEO_STEM: str = "c34"
START_SEC: float = 463.0
DUR_SEC: float = 31.0        # 463-494s (frame 14340=478s を内包)
SMOKE_DUR_SEC: float = 3.0

# 本番同値設定 (perframe 測定・依頼指定と一致)
STABLE_FRAME_COUNT: int = 3
TEMPORAL_SMOOTHING: int = 1
FORCE_IN_MATCH: bool = True

# 追跡対象セル (col=1 の r8-r11) と対象列。
TARGET_CELLS: tuple[tuple[int, int], ...] = ((8, 1), (9, 1), (10, 1), (11, 1))
TARGET_COL: int = 1
# merge / col_check で記録する col=1 の行範囲 (可視上部)。
COL_ROW_RANGE: tuple[int, ...] = (7, 8, 9, 10, 11, 12)
# 追跡対象 side (デッドロックは 1P で発生)。
TARGET_SIDE: str = "1P"
# 復旧ゲート発火閾値 (STABLE_RECOVERY_MIN_FRAMES=8、分類の基準)。
RECOVERY_MIN_FRAMES: int = 8

# 初発フレーム周辺 (merge/infer を詳細記録する窓)。
INITIAL_FRAME: int = 14340
WINDOW_LO: int = 14325
WINDOW_HI: int = 14360

PROGRESS_EVERY_FRAMES: int = 600


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================
# 共有トレースコンテキスト (単一スレッド同期実行前提)
# ============================


@dataclass
class _Ctx:
    """全フックが共有する可変コンテキスト。

    RecognitionPipeline.update は 1P→2P を同期で順に処理するため、
    共有可変状態でも競合しない (並列化しない前提)。
    """

    frame_idx: int = -1
    side: str = ""
    t_sec: float = 0.0
    # 現フレームの gate orig 内で _check_recovery_column(col=1) が返した
    # 通過候補セル集合 (column_rejected と gravity_erased の判別に使う)。
    # None = この frame では col=1 の列チェックが呼ばれていない。
    col1_passed: "set[tuple[int, int]] | None" = None


@dataclass
class _Recorders:
    """全記録を保持する (read-only、状態は本クラスのみ)。"""

    gate: list[dict] = field(default_factory=list)
    col_check: list[dict] = field(default_factory=list)
    merge: list[dict] = field(default_factory=list)
    infer: list[dict] = field(default_factory=list)


# ============================
# 分類ロジック (観測値ベース、推測しない)
# ============================


def _classify_gate_cell(
    conf_before: int, conf_after: int, cnn_v: int, hsv_v: "int | None",
    cnt_before: int, cnt_after: int, in_col1_passed: bool,
    hsv_is_none: bool,
) -> str:
    """1 target セルの復旧ゲート挙動を観測値のみから分類する (推測しない)。

    観測 = confirmed の前後・カウンタの前後 (ctx.stable_recovery_counters の
    実値)・cnn/hsv 値。src の分岐を再構成せず、状態変化の事実で判定する。
    """
    if hsv_is_none:
        return "no_hsv"
    if conf_after != conf_before:
        return "fired"  # confirmed が実際に書き換わった
    # confirmed 不変。カウンタ挙動で理由を切り分ける。
    if cnt_after >= RECOVERY_MIN_FRAMES and conf_before == COLOR_EMPTY:
        # 発火閾値到達済みの add 候補だが confirmed が空のまま
        # = 列却下 (カウンタ非 pop) か、通過後 gravity 再消去 (稀)。
        return "fired_then_gravity_erased" if in_col1_passed else "column_rejected"
    if cnt_after == cnt_before + 1:
        return "counting"
    if cnt_after == 0:
        if conf_before == cnn_v:
            return "no_diff_reset"
        if cnn_v in RECOVERY_EXCLUDED_COLORS or (
            hsv_v is not None and hsv_v in RECOVERY_EXCLUDED_COLORS
        ):
            return "unknown_reset"
        if hsv_v is not None and cnn_v != hsv_v:
            return "cnn_hsv_disagree"
        return "reset_other"
    return "other"


# ============================
# フック生成
# ============================


def _make_gate_wrapper(orig, ctx: _Ctx, rec: _Recorders):
    """P5: board_state_machine._apply_stable_recovery_gate をラップする。"""

    @functools.wraps(orig)
    def wrapped(gate_ctx, signals, min_frames, **kwargs):
        if ctx.side != TARGET_SIDE:
            return orig(gate_ctx, signals, min_frames, **kwargs)
        conf = gate_ctx.confirmed_board
        hsv = signals.hsv_board
        cnn = signals.cnn_board
        hsv_is_none = hsv is None
        snap_before: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        for (r, c) in TARGET_CELLS:
            conf_b = int(conf.get(r, c)) if conf is not None else COLOR_EMPTY
            cnn_v = int(cnn.get(r, c))
            hsv_v = None if hsv_is_none else int(hsv.get(r, c))
            cnt_b = int(gate_ctx.stable_recovery_counters.get((r, c), 0))
            snap_before[(r, c)] = (conf_b, cnn_v, hsv_v, cnt_b)
        ctx.col1_passed = None  # 今 frame の col_check 結果を初期化
        orig(gate_ctx, signals, min_frames, **kwargs)
        _record_gate_cells(gate_ctx, snap_before, hsv_is_none, ctx, rec)

    return wrapped


def _record_gate_cells(
    gate_ctx, snap_before: dict, hsv_is_none: bool, ctx: _Ctx, rec: _Recorders,
) -> None:
    """gate orig 実行後の target セル状態を分類して記録する。"""
    conf = gate_ctx.confirmed_board
    passed = ctx.col1_passed or set()
    for (r, c) in TARGET_CELLS:
        conf_b, cnn_v, hsv_v, cnt_b = snap_before[(r, c)]
        conf_a = int(conf.get(r, c)) if conf is not None else COLOR_EMPTY
        cnt_a = int(gate_ctx.stable_recovery_counters.get((r, c), 0))
        label = _classify_gate_cell(
            conf_b, conf_a, cnn_v, hsv_v, cnt_b, cnt_a,
            (r, c) in passed, hsv_is_none,
        )
        rec.gate.append({
            "frame_idx": ctx.frame_idx, "t_sec": round(ctx.t_sec, 3),
            "cell": [r, c], "conf_before": conf_b, "conf_after": conf_a,
            "cnn": cnn_v, "hsv": hsv_v,
            "cnt_before": cnt_b, "cnt_after": cnt_a, "label": label,
        })


def _make_col_check_wrapper(orig, ctx: _Ctx, rec: _Recorders):
    """安全弁C: board_state_machine._check_recovery_column をラップする。"""

    @functools.wraps(orig)
    def wrapped(confirmed, col, candidates, **kwargs):
        result = orig(confirmed, col, candidates, **kwargs)
        if ctx.side == TARGET_SIDE and col == TARGET_COL:
            passed = {(r, c) for (r, c, _) in result}
            ctx.col1_passed = passed
            in_cands = sorted(
                [[r, c, color] for (r, c, color) in candidates if c == col],
                key=lambda x: -x[0],
            )
            col_confirmed = {
                r: int(confirmed.get(r, col)) for r in COL_ROW_RANGE
            }
            rec.col_check.append({
                "frame_idx": ctx.frame_idx, "t_sec": round(ctx.t_sec, 3),
                "candidates_col1": in_cands,
                "passed_col1": sorted([list(p) for p in passed]),
                "col1_confirmed_r7_12": col_confirmed,
            })
        return result

    return wrapped


def _make_merge_wrapper(orig, ctx: _Ctx, rec: _Recorders):
    """P1: board_state_machine._merge_diff_only をラップする (col=1 記録)。"""

    @functools.wraps(orig)
    def wrapped(baseline, new_cnn, *args, **kwargs):
        merged = orig(baseline, new_cnn, *args, **kwargs)
        if (
            ctx.side == TARGET_SIDE
            and WINDOW_LO <= ctx.frame_idx <= WINDOW_HI
            and baseline is not None
        ):
            guard = kwargs.get("empty_to_color_guard")
            allow_p2e = kwargs.get("allow_puyo_to_empty", True)
            hsv = kwargs.get("hsv_board")
            hsv_guard_flag = kwargs.get("enable_puyo_to_empty_hsv_guard", False)
            rows = []
            for r in COL_ROW_RANGE:
                rows.append({
                    "row": r,
                    "base": int(baseline.get(r, TARGET_COL)),
                    "cnn": int(new_cnn.get(r, TARGET_COL)),
                    "hsv": (int(hsv.get(r, TARGET_COL))
                            if hsv is not None else None),
                    "guard": (int(guard.get(r, TARGET_COL))
                              if guard is not None else None),
                    "merged": int(merged.get(r, TARGET_COL)),
                })
            rec.merge.append({
                "frame_idx": ctx.frame_idx, "t_sec": round(ctx.t_sec, 3),
                "allow_puyo_to_empty": allow_p2e,
                "guard_present": guard is not None,
                "hsv_board_present": hsv is not None,
                "enable_puyo_to_empty_hsv_guard": hsv_guard_flag,
                "col1_rows": rows,
            })
        return merged

    return wrapped


def _make_infer_wrapper(orig, ctx: _Ctx, rec: _Recorders):
    """P2: recognition_pipeline.infer_placement をラップする (col=1 diff 記録)。"""

    @functools.wraps(orig)
    def wrapped(prev_confirmed, cnn_board, falling_pair, *args, **kwargs):
        result = orig(prev_confirmed, cnn_board, falling_pair, *args, **kwargs)
        if (
            ctx.side == TARGET_SIDE
            and WINDOW_LO <= ctx.frame_idx <= WINDOW_HI
            and result is not None and prev_confirmed is not None
        ):
            col1_changes = []
            for r in range(BOARD_ROWS):
                bv = int(prev_confirmed.get(r, TARGET_COL))
                av = int(result.get(r, TARGET_COL))
                if bv != av:
                    col1_changes.append([r, bv, av])
            rec.infer.append({
                "frame_idx": ctx.frame_idx, "t_sec": round(ctx.t_sec, 3),
                "falling_pair": list(falling_pair) if falling_pair else None,
                "col1_changes": col1_changes,
            })
        return result

    return wrapped


def _make_step_side_wrapper(orig, ctx: _Ctx):
    """RecognitionPipeline._step_side をラップし、frame/side/t_sec を ctx へ設定。"""

    @functools.wraps(orig)
    def wrapped(self, side, frame_idx, time_sec, *args, **kwargs):
        ctx.side = side
        ctx.frame_idx = frame_idx
        ctx.t_sec = time_sec
        return orig(self, side, frame_idx, time_sec, *args, **kwargs)

    return wrapped


@contextmanager
def _install_hooks(ctx: _Ctx, rec: _Recorders):
    """計装を一時有効化する (with を抜けると必ず元実装へ復元)。"""
    orig_gate = bsm._apply_stable_recovery_gate
    orig_col = bsm._check_recovery_column
    orig_merge = bsm._merge_diff_only
    orig_infer = rp.infer_placement
    orig_step = rp.RecognitionPipeline._step_side
    bsm._apply_stable_recovery_gate = _make_gate_wrapper(orig_gate, ctx, rec)
    bsm._check_recovery_column = _make_col_check_wrapper(orig_col, ctx, rec)
    bsm._merge_diff_only = _make_merge_wrapper(orig_merge, ctx, rec)
    rp.infer_placement = _make_infer_wrapper(orig_infer, ctx, rec)
    rp.RecognitionPipeline._step_side = _make_step_side_wrapper(orig_step, ctx)
    try:
        yield
    finally:
        bsm._apply_stable_recovery_gate = orig_gate
        bsm._check_recovery_column = orig_col
        bsm._merge_diff_only = orig_merge
        rp.infer_placement = orig_infer
        rp.RecognitionPipeline._step_side = orig_step


# ============================
# キャプチャ (perframe _scan_video を複製し同一トラジェクトリを保証)
# ============================


def _scan(
    stem: str, start_sec: float, dur_sec: float, ctx: _Ctx, rec: _Recorders,
    *, enable_hsv_guard: bool = False,
) -> float:
    """本番同値 pipeline で走査する (計装フック有効)。fps を返す。

    enable_hsv_guard: 案A (enable_puyo_to_empty_hsv_guard) の A/B 計測用。
        default False = perframe 測定と bit-identical。
    """
    path = VIDEO_DIR / f"video_{stem}.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fi = int(start_sec * fps)
    end_frame = int((start_sec + dur_sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=STABLE_FRAME_COUNT,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=TEMPORAL_SMOOTHING,
        force_in_match=FORCE_IN_MATCH,
        enable_puyo_to_empty_hsv_guard=enable_hsv_guard,
    )
    pipe.set_video_id(stem)
    n, t0 = 0, time.time()
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        pipe.update(fi, fi / fps, frame)
        fi += 1
        n += 1
        if n % PROGRESS_EVERY_FRAMES == 0:
            rate = n / max(time.time() - t0, 1e-6)
            _log(f"[{stem}] {n}f 処理済 ({rate:.1f}f/s) gate={len(rec.gate)}")
    cap.release()
    return fps


# ============================
# 集計・出力
# ============================


def _summarize_gate(rec: _Recorders) -> dict:
    """target セル別に gate ラベルの件数分布を集計する。"""
    by_cell: dict[str, dict[str, int]] = {}
    for g in rec.gate:
        key = f"r{g['cell'][0]}c{g['cell'][1]}"
        d = by_cell.setdefault(key, {})
        d[g["label"]] = d.get(g["label"], 0) + 1
    return by_cell


def _deadlock_window_gate(rec: _Recorders) -> dict:
    """デッドロック区間 (14340-14756) の r9c1 ラベル分布 (質問 a の核心)。"""
    out: dict[str, dict[str, int]] = {}
    for g in rec.gate:
        if not (INITIAL_FRAME <= g["frame_idx"] <= 14756):
            continue
        key = f"r{g['cell'][0]}c{g['cell'][1]}"
        d = out.setdefault(key, {})
        d[g["label"]] = d.get(g["label"], 0) + 1
    return out


def _r11_hsv_stats(rec: _Recorders) -> dict:
    """r11c1 の HSV/CNN 値分布 (質問 b)。"""
    hsv_counts: dict[str, int] = {}
    cnn_counts: dict[str, int] = {}
    for g in rec.gate:
        if g["cell"] != [11, 1]:
            continue
        hsv_counts[str(g["hsv"])] = hsv_counts.get(str(g["hsv"]), 0) + 1
        cnn_counts[str(g["cnn"])] = cnn_counts.get(str(g["cnn"]), 0) + 1
    return {"hsv_value_counts": hsv_counts, "cnn_value_counts": cnn_counts}


def _write_outputs(rec: _Recorders, fps: float, *, out_suffix: str = "") -> None:
    """全記録と集計を jsonl / json / txt で出力する。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{VIDEO_STEM}{out_suffix}"
    for name, records in (
        ("gate", rec.gate), ("col_check", rec.col_check),
        ("merge", rec.merge), ("infer", rec.infer),
    ):
        with open(OUT_DIR / f"{stem}_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "video": VIDEO_STEM, "fps": fps,
        "config": {
            "stable_frame_count": STABLE_FRAME_COUNT,
            "temporal_smoothing": TEMPORAL_SMOOTHING,
            "force_in_match": FORCE_IN_MATCH,
            "recovery_min_frames": RECOVERY_MIN_FRAMES,
        },
        "gate_label_by_cell_full": _summarize_gate(rec),
        "gate_label_by_cell_deadlock_14340_14756": _deadlock_window_gate(rec),
        "r11c1_hsv_cnn_stats": _r11_hsv_stats(rec),
        "n_records": {
            "gate": len(rec.gate), "col_check": len(rec.col_check),
            "merge": len(rec.merge), "infer": len(rec.infer),
        },
    }
    (OUT_DIR / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"出力完了 → {OUT_DIR}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    cv2.setNumThreads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="先頭3秒のみの動作確認")
    ap.add_argument(
        "--enable-hsv-guard", dest="enable_hsv_guard", action="store_true",
        default=False,
        help="案A (enable_puyo_to_empty_hsv_guard) を有効化する。"
             "既定 False = perframe 測定と bit-identical。",
    )
    ap.add_argument(
        "--out-suffix", dest="out_suffix", type=str, default="",
        help="出力ファイル stem の接尾辞 (ON/OFF 比較の上書き防止用)。",
    )
    ap.add_argument(
        "--dur-sec", dest="dur_sec", type=float, default=None,
        help="走査秒数の上書き (開始秒は 463 固定でトラジェクトリ再現)。"
             "merge frame 14332 のみ確認する短時間走査用。",
    )
    args = ap.parse_args()
    dur = SMOKE_DUR_SEC if args.smoke else (
        args.dur_sec if args.dur_sec is not None else DUR_SEC
    )
    ctx, rec = _Ctx(), _Recorders()
    _log(
        f"[{VIDEO_STEM}] 走査開始 {START_SEC:.1f}s + {dur:.1f}s "
        f"(初発 frame {INITIAL_FRAME}) hsv_guard={args.enable_hsv_guard}"
    )
    t0 = time.time()
    with _install_hooks(ctx, rec):
        fps = _scan(
            VIDEO_STEM, START_SEC, dur, ctx, rec,
            enable_hsv_guard=args.enable_hsv_guard,
        )
    _log(f"[{VIDEO_STEM}] 走査完了 {time.time() - t0:.0f}s fps={fps:.1f}")
    _write_outputs(rec, fps, out_suffix=args.out_suffix)


if __name__ == "__main__":
    main()
