"""W13修正 (highlight_override) 副作用 (13セル悪化) の機構特定 (2026-08-17、計装専用)。

物差しv2で確認された悪化2盤面 (000_c109_1P_f652064 col=0 9セル、
002_c11_2P_f54124 col=2 4セル) について、実際に採点で使われたのと同一の
チャンク切り出し (video, start_sec, max_sec, collect_flags()) を
c1p(OFF)/w13(ON) 両方で再現し、以下を計装で確定する:

  1. 対象列で OFF/ON の confirmed_board が最初に分岐するフレーム (t, frame_idx)
  2. その分岐フレームで、対象セルの tier1 (patch-NCC) 結果・
     highlight blob 検出結果 (面積比込み)・tier2 (is_empty_by_fp) 結果・
     hsv_only フォールバック結果 を OFF/ON 両方で独立計算 (src非変更、
     read_board と同一の計算式を診断スクリプト側で再現するのみ)
  3. tier1 warmup (_tier1_warmup_remaining_1p/2p, _ojama_tier1_warmup_remaining_1p/2p)
     の残余フレーム数を毎フレーム記録し、 OFF/ON で warmup 窓の
     開始/終了タイミングが分岐しているかを確認

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_w13_sideeffect_mechanism_2026-08-17
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.background_fingerprint import (
    CellFingerprint,
    HIGHLIGHT_MIN_PIXEL_RATIO,
    HIGHLIGHT_REGION_Y_RATIO,
    HIGHLIGHT_REGION_Y_RATIO_LOW,
    HIGHLIGHT_S_MAX,
    HIGHLIGHT_V_MIN,
    PatchBackgroundFingerprint,
    detect_highlight_blob,
    is_empty_by_fp,
)
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, HIDDEN_ROWS
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.fps_normalize import resolve_normalize_fps_30_stride
from src.production_config import collect_flags
from src.recognition_pipeline import RecognitionPipeline

_bc = importlib.import_module("scripts._collect_yardstick_v2_bc_2026-08-15")

VIDEO_DIR = Path.home() / "frames"
OUT_DIR = Path("data/verify/diag_w13_sideeffect_2026-08-17")
LOG_DIR = Path("logs")

# 対象2盤面 (video_id, chunk_idx, side, target_col, target_frame_idx)
# extra_detail_frames: 1回目の実行で判明した first_divergence の frame_idx
# (2回目実行でその周辺の tier1/blob/tier2 詳細も追加取得するため)
TARGETS = [
    {"sid": "000_c109_1P_f652064", "video_id": "c109", "chunk_idx": 2,
     "side": "1P", "col": 0, "target_frame_idx": 652064,
     "extra_detail_frames": {651998}},
    {"sid": "002_c11_2P_f54124", "video_id": "c11", "chunk_idx": 1,
     "side": "2P", "col": 2, "target_frame_idx": 54124,
     "extra_detail_frames": {54018}},
]

CHUNK_SEC = _bc.CHUNK_SEC


def _parse_collect_flags() -> dict:
    """collect_flags() の文字列を RecognitionPipeline.load_default kwargs に変換する。

    collect_boards_lean.collect_lean の CLI 引数パーサと同じマッピングを
    最小限だけ手動で再現する (フラグ追加や argparse 定義変更に依存しない
    ようベタ書き、本診断専用)。
    """
    flags = collect_flags().split()
    kwargs = {}
    mapping_bool = {
        "--enable-effect-gate": "enable_effect_gate",
        "--enable-burst-guard-v2": "enable_burst_guard_v2",
        "--enable-transition-merge-guard": "enable_transition_merge_guard",
        "--enable-hidden-row-burst-guard": "enable_hidden_row_burst_guard",
        "--enable-match-transition-debounce": "enable_match_transition_debounce",
        "--enable-ojama-fall-placement-override": "enable_ojama_fall_placement_override",
        "--enable-chain-tracker": "enable_chain_tracker",
    }
    i = 0
    while i < len(flags):
        f = flags[i]
        if f == "--burst-gate-open-threshold":
            kwargs["burst_gate_open_threshold"] = float(flags[i + 1])
            i += 2
            continue
        if f in mapping_bool:
            kwargs[mapping_bool[f]] = True
            i += 1
            continue
        raise ValueError(f"未知のフラグ (診断スクリプトのマッピング更新が必要): {f}")
    return kwargs


def build_pipeline(enable_highlight_override: bool) -> RecognitionPipeline:
    kwargs = _parse_collect_flags()
    kwargs["enable_highlight_override"] = enable_highlight_override
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=kwargs.pop("enable_chain_tracker", False),
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        **kwargs,
    )
    return pipe


def probe_duration_sec(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if fps > 0 else 0.0


def _median_hsv(patch_hsv: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.median(patch_hsv[:, :, 0])),
        float(np.median(patch_hsv[:, :, 1])),
        float(np.median(patch_hsv[:, :, 2])),
    )


def _white_ratio(patch_hsv: np.ndarray) -> float:
    """detect_highlight_blob と同一式で white pixel 比率を計算 (bool でなく実数で見る用)。"""
    h_size, w_size = patch_hsv.shape[:2]
    total = h_size * w_size
    if total == 0:
        return 0.0
    y0 = int(h_size * HIGHLIGHT_REGION_Y_RATIO)
    y1 = int(h_size * HIGHLIGHT_REGION_Y_RATIO_LOW)
    if y0 >= y1:
        return 0.0
    band = patch_hsv[y0:y1, :, :]
    s_ch = band[:, :, 1].astype(np.float64)
    v_ch = band[:, :, 2].astype(np.float64)
    mask = (v_ch >= HIGHLIGHT_V_MIN) & (s_ch <= HIGHLIGHT_S_MAX)
    return float(np.count_nonzero(mask)) / total


def analyze_cell(
    pipe: RecognitionPipeline, frame_1920: np.ndarray, side: str, col: int, row: int,
) -> dict:
    """read_board と同一計算式でセル1つの tier1/blob/tier2/hsv_only を再現する。"""
    reader = pipe._reader
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    bg_fp = reader._bg_fp_p1 if side == "1P" else reader._bg_fp_p2
    out = {"bg_fp_present": bg_fp is not None}
    if bg_fp is None:
        return out
    visible_row = row - HIDDEN_ROWS
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    h, w = frame_1920.shape[:2]
    x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
    hsv_full = cv2.cvtColor(frame_1920, cv2.COLOR_BGR2HSV)
    hsv_patch = hsv_full[y1:y2, x1:x2]
    if hsv_patch.size == 0:
        out["hsv_patch_empty"] = True
        return out
    h_med, s_med, v_med = _median_hsv(hsv_patch)
    cur_fp = CellFingerprint(h_med, s_med, v_med)
    bg_cell_median = bg_fp.cell_at(visible_row, col)
    if isinstance(bg_fp, PatchBackgroundFingerprint):
        bg_cell_for_tier1 = bg_fp.cell_at_patch(visible_row, col)
    else:
        bg_cell_for_tier1 = bg_cell_median
    tier1_result = reader._is_empty_tier1(
        bg_cell_for_tier1, hsv_patch, cur_fp, visible_row, col,
    )
    blob_result = detect_highlight_blob(hsv_patch)
    white_ratio = _white_ratio(hsv_patch)
    tier2_result = is_empty_by_fp(cur_fp, bg_cell_median, threshold=reader._bg_threshold)
    hsv_target = getattr(reader._classifier, "_hsv", reader._classifier)
    patch_bgr = frame_1920[y1:y2, x1:x2]
    try:
        hsv_only_color = int(hsv_target.classify(patch_bgr))
    except Exception:
        hsv_only_color = None
    out.update({
        "cur_fp_hsv": [h_med, s_med, v_med],
        "tier1_result": bool(tier1_result),
        "blob_result": bool(blob_result),
        "white_ratio": white_ratio,
        "min_pixel_ratio_threshold": HIGHLIGHT_MIN_PIXEL_RATIO,
        "tier2_result": bool(tier2_result),
        "hsv_only_color": hsv_only_color,
        "bg_v_med_patch": (
            float(np.median(bg_cell_for_tier1.patch_hsv[:, :, 2]))
            if isinstance(bg_fp, PatchBackgroundFingerprint) else None
        ),
    })
    return out


def run_pass(
    target: dict, enable_highlight_override: bool, start_sec: float, end_sec: float,
    extra_detail_frames: "set[int] | None" = None,
) -> list[dict]:
    pipe = build_pipeline(enable_highlight_override)
    video_path = VIDEO_DIR / _bc.video_filename_of(target["video_id"])
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    # 本番 collect_lean と同一の fps 正規化 (normalize_fps_30=True 既定、
    # 60fps→stride=2)。 これを怠ると STABLE/warmup タイミングが本番と
    # 一致せず別条件を調べたことになる (再現条件不一致の教訓)。
    stride = resolve_normalize_fps_30_stride(fps)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    side = target["side"]
    col = target["col"]
    target_frame_idx = target["target_frame_idx"]
    rows = list(range(HIDDEN_ROWS, BOARD_ROWS))

    records: list[dict] = []
    for fi in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if (fi - start_frame) % stride != 0:
            continue
        t = fi / fps
        recog_frame = (
            frame if frame.shape[:2] == (1080, 1920)
            else cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        )
        # warmup 残余 (この frame の update() 呼び出し「前」の値 = 実際にこの
        # frame の skip_tier1 判定に使われた値)
        warm_1p = getattr(pipe, "_tier1_warmup_remaining_1p", None)
        warm_2p = getattr(pipe, "_tier1_warmup_remaining_2p", None)
        ojama_warm_1p = getattr(pipe, "_ojama_tier1_warmup_remaining_1p", None)
        ojama_warm_2p = getattr(pipe, "_ojama_tier1_warmup_remaining_2p", None)

        r = pipe.update(fi, t, recog_frame)

        rec = {
            "fi": fi, "t": t,
            "p1_state": str(r.p1.state), "p2_state": str(r.p2.state),
            "warm_1p_pre": warm_1p, "warm_2p_pre": warm_2p,
            "ojama_warm_1p_pre": ojama_warm_1p, "ojama_warm_2p_pre": ojama_warm_2p,
        }
        board = r.p1.confirmed_board if side == "1P" else r.p2.confirmed_board
        if board is not None:
            rec["col_vals"] = [int(board.get(row, col)) for row in rows]
        else:
            rec["col_vals"] = None

        # 目標フレーム付近 (±3 frame) + 追加指定フレーム付近は
        # 全行 tier1/blob/tier2 詳細を保存
        _near_target = abs(fi - target_frame_idx) <= 3
        _near_extra = extra_detail_frames is not None and any(
            abs(fi - ef) <= 3 for ef in extra_detail_frames
        )
        if _near_target or _near_extra:
            rec["cell_detail"] = {
                row: analyze_cell(pipe, recog_frame, side, col, row) for row in rows
            }
        records.append(rec)
    cap.release()
    return records


def find_first_divergence(off_records: list[dict], on_records: list[dict]) -> dict | None:
    by_fi_on = {r["fi"]: r for r in on_records}
    for r_off in off_records:
        r_on = by_fi_on.get(r_off["fi"])
        if r_on is None:
            continue
        if r_off["col_vals"] is None or r_on["col_vals"] is None:
            continue
        if r_off["col_vals"] != r_on["col_vals"]:
            return {"fi": r_off["fi"], "t": r_off["t"],
                    "off_col_vals": r_off["col_vals"], "on_col_vals": r_on["col_vals"]}
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    for target in TARGETS:
        sid = target["sid"]
        vid = target["video_id"]
        video_path = VIDEO_DIR / _bc.video_filename_of(vid)
        duration = probe_duration_sec(video_path)
        frac = _bc.CHUNK_OFFSET_FRACTIONS[target["chunk_idx"]]
        start_sec = max(0.0, frac * duration)
        end_sec = start_sec + CHUNK_SEC

        print(f"[{sid}] video={vid} start_sec={start_sec:.2f} end_sec={end_sec:.2f} "
              f"target_frame_idx={target['target_frame_idx']}")

        edf = target.get("extra_detail_frames")
        print("  OFF パス実行中...")
        off_records = run_pass(target, False, start_sec, end_sec, extra_detail_frames=edf)
        print(f"  {len(off_records)} フレーム処理")
        print("  ON パス実行中...")
        on_records = run_pass(target, True, start_sec, end_sec, extra_detail_frames=edf)
        print(f"  {len(on_records)} フレーム処理")

        divergence = find_first_divergence(off_records, on_records)

        # 目標フレームでの詳細 (両パス)
        off_at_target = next(
            (r for r in off_records if r["fi"] == target["target_frame_idx"]), None,
        )
        on_at_target = next(
            (r for r in on_records if r["fi"] == target["target_frame_idx"]), None,
        )

        result = {
            "target": target,
            "start_sec": start_sec, "end_sec": end_sec,
            "first_divergence": divergence,
            "at_target_frame": {
                "off": off_at_target, "on": on_at_target,
            },
        }
        summary[sid] = result

        # 生タイムライン (対象列のみ、warmup込み) を保存
        out_path = OUT_DIR / f"{sid}_timeline.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "off": [
                    {k: v for k, v in r.items() if k != "cell_detail"}
                    for r in off_records
                ],
                "on": [
                    {k: v for k, v in r.items() if k != "cell_detail"}
                    for r in on_records
                ],
                "off_cell_detail": [
                    r for r in off_records if "cell_detail" in r
                ],
                "on_cell_detail": [
                    r for r in on_records if "cell_detail" in r
                ],
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"  タイムライン保存: {out_path}")

    out_summary = OUT_DIR / "summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nサマリ保存: {out_summary}")


if __name__ == "__main__":
    main()
