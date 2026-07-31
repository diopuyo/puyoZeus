"""video_c34 序盤「Bが2つ出る」誤認の根因を確定する診断スクリプト (診断専用、修正なし)。

背景: userレビュー (v1/v4) で一貫して「序盤に青(B)が2つある」と指摘されている。
既存 write_trace 計装 (scripts/_diag_confirmed_write_trace_2026-07-25.py) は
どの経路 (P1〜P9) が confirmed_board のセルを書いたかはタグ付けするが、
「そのセルの生CNN色・HSV色・NEXTキュー色」までは記録しないため、
本スクリプトで最小限の追加計装を行う (read-only、src/ は一切変更しない)。

計装対象 (すべて既存関数への monkeypatch、with を抜けると必ず復元):
    1. write_trace hooks (P1/P2/P3/P5/inline) — 既存スクリプトをそのまま再利用。
    2. src.recognition_pipeline.infer_placement — 着地セルの
       (書き込み色, cnn_board融合色, 生CNN色+確信度, HSV-only色, falling_pair)
       を記録する (frame_bgr/region は infer_placement の既存 kwargs から取得、
       再分類は reader._classifier の既存オブジェクトを呼ぶだけで副作用なし)。
    3. src.placement_inferrer.correct_landing_cells_by_observed_color —
       着地色補正 (真因A対処) が実際に発火したか・結果色を記録する。

出力: data/verify/write_trace/<out_stem>_earlygame_color_table.txt/.json
      + data/verify/write_trace/<out_stem>_frames/ (各手の実フレームPNG)

Usage:
    PYTHONPATH=. python scripts/_diag_c34_earlygame_color_2026-07-25.py \
        --video c34 --start-sec 465.6 --max-sec 40.0
"""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import sys
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# スレッド制限 (熱暴走防止、feedback_thermal_safety_mandatory 準拠)。並列しない。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import src.recognition_pipeline as rp  # noqa: E402
import src.placement_inferrer as pi  # noqa: E402
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.patch_classifier import CLASS_INDEX_TO_COLOR  # noqa: E402
from scripts.recognition_physics_review import _capture_frames, VIDEO_DIR  # noqa: E402

# write_trace 計装本体 (ハイフン付きファイル名のため importlib 経由、
# import 文では読めない。importlib.import_module は識別子検証をしないため可能)。
_wt = importlib.import_module("scripts._diag_confirmed_write_trace_2026-07-25")

OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "write_trace"
COLOR_NAME_JP: dict[int, str] = {
    COLOR_EMPTY: "空", 1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫",
    COLOR_OJAMA: "おじゃま", COLOR_UNKNOWN: "UNKNOWN",
}
# 最初に詳細レポートする手数 (ユーザー指示)。
N_EARLY_MOVES: int = 5


def _color_jp(v: int | None) -> str:
    """色コードを日本語ラベルに変換する (None は '-')。"""
    if v is None:
        return "-"
    return f"{v}:{COLOR_NAME_JP.get(v, '?')}"


def _side_from_region(region: object) -> str:
    """BoardRegion の identity から 1P/2P を判定する (module singleton前提)。"""
    if region is DEFAULT_P1_REGION:
        return "1P"
    if region is DEFAULT_P2_REGION:
        return "2P"
    return "?"


@dataclass
class ColorDiagRecord:
    """1 着地セル分の色診断レコード (infer_placement 時点 + 着地補正後)。"""

    side: str
    frame_idx: int
    t_sec: float
    r: int
    c: int
    before_color: int
    infer_written_color: int
    cnn_fused_color: int | None
    cnn_raw_color: int | None
    cnn_raw_prob: float | None
    hsv_only_color: int | None
    falling_pair: list[int] | None
    landing_corrected: bool | None = None
    final_color_after_correction: int | None = None


def _reclassify_cell(
    reader: object, frame_bgr: np.ndarray | None, region: object, r: int, c: int,
) -> tuple[int | None, float | None, int | None]:
    """着地セルの patch を再抽出し、生CNN色+確信度・HSV-only色を独立に求める。

    reader._classifier (HybridClassifier) が保持する既存の _cnn / _hsv
    オブジェクトを呼ぶだけ (副作用なし、production の分類結果は変更しない)。
    """
    if reader is None or frame_bgr is None or region is None:
        return None, None, None
    patch = pi._extract_cell_patch_from_frame(frame_bgr, region, r, c)
    if patch is None or patch.size == 0:
        return None, None, None
    classifier = getattr(reader, "_classifier", None)
    cnn_model = getattr(classifier, "_cnn", None)
    hsv_clf = getattr(classifier, "_hsv", None)
    cnn_color: int | None = None
    cnn_prob: float | None = None
    hsv_color: int | None = None
    if cnn_model is not None:
        try:
            probs = cnn_model.predict_proba(patch)
            best_idx = int(np.argmax(probs))
            cnn_color = int(CLASS_INDEX_TO_COLOR[best_idx])
            cnn_prob = float(probs[best_idx])
        except Exception:
            pass
    if hsv_clf is not None:
        try:
            hsv_color = int(hsv_clf.classify(patch))
        except Exception:
            pass
    return cnn_color, cnn_prob, hsv_color


def _record_infer_placement(
    result: object, prev_confirmed: object, cnn_board: object,
    falling_pair: object, kwargs: dict, frame_ctx: dict, pipeline_out: dict,
    color_records: list[ColorDiagRecord], key_index: dict,
) -> None:
    """infer_placement 呼び出し1回分の着地セル差分を色診断レコードとして記録する。"""
    if result is None or prev_confirmed is None:
        return
    diff = _wt._diff_cells(prev_confirmed, result)
    if not diff:
        return
    region = kwargs.get("region")
    side = _side_from_region(region)
    frame_bgr = kwargs.get("frame_bgr")
    reader = getattr(pipeline_out.get("pipeline"), "_reader", None)
    for (r, c, bv, av) in diff:
        cnn_fused = int(cnn_board.get(r, c)) if cnn_board is not None else None
        cnn_raw, cnn_prob, hsv_only = _reclassify_cell(reader, frame_bgr, region, r, c)
        rec = ColorDiagRecord(
            side=side, frame_idx=frame_ctx["frame_idx"], t_sec=frame_ctx["t_sec"],
            r=r, c=c, before_color=bv, infer_written_color=av,
            cnn_fused_color=cnn_fused, cnn_raw_color=cnn_raw, cnn_raw_prob=cnn_prob,
            hsv_only_color=hsv_only,
            falling_pair=list(falling_pair) if falling_pair else None,
        )
        color_records.append(rec)
        key_index[(side, frame_ctx["frame_idx"], r, c)] = rec


def _record_landing_correction(
    inferred: object, result: object, pattern: object, region: object,
    frame_ctx: dict, key_index: dict,
) -> None:
    """着地色補正 (真因A対処) の発火有無・結果色を対応レコードに書き足す。"""
    side = _side_from_region(region)
    fi = frame_ctx["frame_idx"]
    for (r, c) in pattern.cells:
        rec = key_index.get((side, fi, r, c))
        if rec is None:
            continue
        before_v = int(inferred.get(r, c))
        after_v = int(result.get(r, c))
        rec.landing_corrected = (before_v != after_v)
        rec.final_color_after_correction = after_v


@contextmanager
def _install_color_diag_hooks(color_records: list[ColorDiagRecord], pipeline_out: dict):
    """infer_placement / correct_landing_cells_by_observed_color を追加計装する。

    write_trace hooks が既にインストール済の rp.infer_placement を
    さらにラップする (合成、写像順は write_trace → 本フックの順で実行される)。
    """
    frame_ctx: dict = {"frame_idx": -1, "t_sec": 0.0}
    key_index: dict = {}

    orig_update = rp.RecognitionPipeline.update
    orig_infer = rp.infer_placement
    orig_correct = pi.correct_landing_cells_by_observed_color

    @functools.wraps(orig_update)
    def wrapped_update(self, frame_idx, time_sec, frame):
        frame_ctx["frame_idx"] = frame_idx
        frame_ctx["t_sec"] = time_sec
        return orig_update(self, frame_idx, time_sec, frame)

    @functools.wraps(orig_infer)
    def wrapped_infer(prev_confirmed, cnn_board, falling_pair, *args, **kwargs):
        result = orig_infer(prev_confirmed, cnn_board, falling_pair, *args, **kwargs)
        _record_infer_placement(
            result, prev_confirmed, cnn_board, falling_pair, kwargs,
            frame_ctx, pipeline_out, color_records, key_index,
        )
        return result

    @functools.wraps(orig_correct)
    def wrapped_correct(inferred, pattern, cnn_board, hsv_classifier, frame_bgr, region, *args, **kwargs):
        result = orig_correct(
            inferred, pattern, cnn_board, hsv_classifier, frame_bgr, region, *args, **kwargs,
        )
        _record_landing_correction(inferred, result, pattern, region, frame_ctx, key_index)
        return result

    rp.RecognitionPipeline.update = wrapped_update
    rp.infer_placement = wrapped_infer
    pi.correct_landing_cells_by_observed_color = wrapped_correct
    try:
        yield
    finally:
        rp.RecognitionPipeline.update = orig_update
        rp.infer_placement = orig_infer
        pi.correct_landing_cells_by_observed_color = orig_correct


def _group_into_moves(records: list[ColorDiagRecord]) -> list[list[ColorDiagRecord]]:
    """side別に (frame_idx) が同じレコードを1手とみなしてグルーピングする。"""
    by_side: dict[str, list[ColorDiagRecord]] = {}
    for rec in records:
        by_side.setdefault(rec.side, []).append(rec)
    moves: list[list[ColorDiagRecord]] = []
    for side, recs in by_side.items():
        recs.sort(key=lambda r: (r.frame_idx, r.r, r.c))
        cur_key = None
        cur_group: list[ColorDiagRecord] = []
        for rec in recs:
            if cur_key != (rec.side, rec.frame_idx):
                if cur_group:
                    moves.append(cur_group)
                cur_group = [rec]
                cur_key = (rec.side, rec.frame_idx)
            else:
                cur_group.append(rec)
        if cur_group:
            moves.append(cur_group)
    moves.sort(key=lambda g: (g[0].side, g[0].frame_idx))
    return moves


def _format_move_row(move_no: int, group: list[ColorDiagRecord]) -> str:
    """1手分のテーブル行を整形する。"""
    r0 = group[0]
    next_str = (
        f"{_color_jp(r0.falling_pair[0])}+{_color_jp(r0.falling_pair[1])}"
        if r0.falling_pair else "-"
    )
    lines = [
        f"  手#{move_no} side={r0.side} frame={r0.frame_idx} t={r0.t_sec:.2f}s "
        f"NEXT記録色={next_str}",
    ]
    for rec in group:
        corr = (
            "-" if rec.landing_corrected is None
            else ("補正あり→" + _color_jp(rec.final_color_after_correction))
            if rec.landing_corrected else "補正なし(条件不成立or不一致)"
        )
        cnn_raw = (
            f"{_color_jp(rec.cnn_raw_color)}(p={rec.cnn_raw_prob:.2f})"
            if rec.cnn_raw_color is not None else "-"
        )
        lines.append(
            f"    cell=({rec.r},{rec.c}) before={_color_jp(rec.before_color)} "
            f"P2書込色={_color_jp(rec.infer_written_color)} "
            f"cnn_board(融合)={_color_jp(rec.cnn_fused_color)} "
            f"生CNN={cnn_raw} HSVのみ={_color_jp(rec.hsv_only_color)} "
            f"着地色補正={corr}",
        )
    return "\n".join(lines)


def _extract_frame_png(video_stem: str, frame_idx: int, out_path: Path) -> bool:
    """指定 frame_idx の実フレームを PNG 保存する (視覚照合用)。"""
    video_path = VIDEO_DIR / f"video_{video_stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return True


def _cell_write_history(recorder, side: str, r: int, c: int) -> list[str]:
    """write_trace 全体から指定セルへの書き込み履歴 (経路つき) を時系列で返す。"""
    rows: list[str] = []
    for rec in recorder.records:
        if rec.side != side:
            continue
        for cell in rec.cells:
            if cell[0] == r and cell[1] == c:
                rows.append(
                    f"    frame={rec.frame_idx} t={rec.t_sec:.2f}s route={rec.route_id} "
                    f"before={_color_jp(cell[2])} after={_color_jp(cell[3])}",
                )
    return rows


def _collect_palette(by_side: dict) -> dict[str, set[int]]:
    """走査窓内で観測された有効puyo色 (1-5) を side別に集計する (4色推定の材料)。

    短時間窓のためこの窓だけでは4色全ては確定できない可能性がある点に注意
    (診断メモに limitation として明記する)。
    """
    palette: dict[str, set[int]] = {"1P": set(), "2P": set()}
    for side, recs in by_side.items():
        for rec in recs:
            if rec.grid is None:
                continue
            for v in np.unique(rec.grid):
                vi = int(v)
                if 1 <= vi <= 5:
                    palette[side].add(vi)
    return palette


def run(video_stem: str, start_sec: float, max_sec: float, out_stem: str) -> None:
    """診断本体: write_trace + 色診断計装を有効化して1動画1窓分を処理する。"""
    print(
        f"[診断開始] video={video_stem} start={start_sec:.1f}s max={max_sec:.1f}s "
        f"out_stem={out_stem} 構成=着地色補正ON+Driftガード2種ON+--no-force-in-match",
    )
    pipeline_out: dict = {}
    color_records: list[ColorDiagRecord] = []
    with ExitStack() as stack:
        recorder, matchstart_diag = stack.enter_context(_wt._install_write_trace_hooks(video_stem))
        stack.enter_context(_install_color_diag_hooks(color_records, pipeline_out))
        by_side = _capture_frames(
            video_stem, start_sec, max_sec,
            enable_landing_observed_color=True,
            force_in_match=False,
            enable_drift_resync_match_start_guard=True,
            enable_drift_resync_hsv_gate=True,
            pipeline_out=pipeline_out,
        )
    print(f"[処理完了] write_trace記録 {len(recorder.records)} 件、着地色診断 {len(color_records)} 件")
    # is_active反転/reset呼び出しの実測 (試合開始誤判定の裏付け用、既存ヘルパー流用)。
    _wt._write_matchstart_diag_outputs(matchstart_diag, out_stem, start_sec, max_sec)
    _report(video_stem, out_stem, recorder, color_records, by_side)


def _report(
    video_stem: str, out_stem: str, recorder, color_records: list[ColorDiagRecord], by_side: dict,
) -> None:
    """最初のN手のテーブル・実フレーム抽出・4色推定・セル書込履歴を出力する。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    moves = _group_into_moves(color_records)
    lines = ["==== c34 序盤 着地色診断 (最初の各side {}手) ====".format(N_EARLY_MOVES)]
    by_side_moves: dict[str, list] = {}
    for g in moves:
        by_side_moves.setdefault(g[0].side, []).append(g)
    frame_dir = OUTPUT_DIR / f"{out_stem}_frames"
    for side in ("1P", "2P"):
        lines.append(f"--- {side} (全{len(by_side_moves.get(side, []))}手検出) ---")
        for i, g in enumerate(by_side_moves.get(side, [])[:N_EARLY_MOVES], start=1):
            lines.append(_format_move_row(i, g))
            fi = g[0].frame_idx
            png_path = frame_dir / f"{side}_move{i}_frame{fi}_t{g[0].t_sec:.2f}s.png"
            ok = _extract_frame_png(video_stem, fi, png_path)
            lines.append(f"    実フレーム: {png_path if ok else '抽出失敗'}")
            for (r, c) in {(rec.r, rec.c) for rec in g}:
                hist = _cell_write_history(recorder, side, r, c)
                if len(hist) > 1:
                    lines.append(f"    cell=({r},{c}) 書込履歴(窓内全件):")
                    lines.extend(hist)
    palette = _collect_palette(by_side)
    lines.append(f"--- 窓内で観測された有効puyo色 (4色推定の材料、短時間窓のため参考値) ---")
    for side, colors in palette.items():
        lines.append(f"  {side}: {sorted(_color_jp(v) for v in colors)}")
    text = "\n".join(lines)
    print(text)
    (OUTPUT_DIR / f"{out_stem}_color_table.txt").write_text(text, encoding="utf-8")
    _write_json(out_stem, moves, palette)


def _write_json(out_stem: str, moves: list, palette: dict[str, set[int]]) -> None:
    """テーブルをJSONでも保存する (機械可読、後続分析用)。"""
    payload = {
        "moves": [
            [
                {
                    "side": rec.side, "frame_idx": rec.frame_idx, "t_sec": rec.t_sec,
                    "r": rec.r, "c": rec.c, "before_color": rec.before_color,
                    "infer_written_color": rec.infer_written_color,
                    "cnn_fused_color": rec.cnn_fused_color,
                    "cnn_raw_color": rec.cnn_raw_color, "cnn_raw_prob": rec.cnn_raw_prob,
                    "hsv_only_color": rec.hsv_only_color,
                    "falling_pair": rec.falling_pair,
                    "landing_corrected": rec.landing_corrected,
                    "final_color_after_correction": rec.final_color_after_correction,
                }
                for rec in g
            ]
            for g in moves
        ],
        "palette_window_sample": {k: sorted(v) for k, v in palette.items()},
    }
    (OUTPUT_DIR / f"{out_stem}_color_table.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    """CLI引数をパースする。"""
    ap = argparse.ArgumentParser(description="c34 序盤 Bが2つ出る誤認 根因診断")
    ap.add_argument("--video", type=str, default="c34")
    ap.add_argument("--start-sec", type=float, default=465.6)
    ap.add_argument("--max-sec", type=float, default=40.0)
    ap.add_argument("--output-stem", type=str, default="c34_earlygame")
    return ap.parse_args()


def main() -> None:
    """メイン処理。"""
    cv2.setNumThreads(1)
    args = _parse_args()
    run(args.video, args.start_sec, args.max_sec, args.output_stem)


if __name__ == "__main__":
    main()
