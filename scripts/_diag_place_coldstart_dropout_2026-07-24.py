"""真因診断: 設置直後の取りこぼし (失敗1、2026-07-24 user目視確定分)。

完全 read-only 診断スクリプト。src/ および既存 scripts/ は一切変更しない。

## 依頼背景
user が目視で確定した頻発失敗1:
    「ぷよを置いた直後、その設置ぷよが確定盤面(confirmed_board)に取り込まれ
    ないまま連鎖が始まるパターン。連鎖の起点盤面が『置いたぷよが無い』誤った
    状態で始まる」

## コード読解で特定した機構仮説 (実測前の仮説、本スクリプトで検証)
recognition_pipeline.py / board_state_machine.py / state_detectors.py を
読解した結果、以下の 2 経路がどちらも「_prev_confirmed_1p / _prev_confirmed_2p」
(= 直近 STABLE 確定盤面。STABLE 以外では前回値を保持し続ける) を
連鎖 before_board として使っており、この値が TSUMO_FALL → STABLE の
多段フレーム合意 (consec_threshold=2 + landed_consec=2、
state_detectors.py:176-252) を待つ前に、独立した連鎖検知が先に発火すると
「置いたばかりのぷよを含まない」 stale な盤面のまま before_board が確定して
しまう構造的リスクがある:

  経路A (機能D 掛け算式検知、デフォルト ON、recognition_pipeline.py:3101-3153):
      score ROI が掛け算式表示になった時点 (CHAIN_FORMULA_CONSEC_FRAMES=2
      フレームのみ、recognition_pipeline.py:85) で即座に
      `before = prev_confirmed.copy()` (= self._prev_confirmed_1p/2p)
      を before_board として ChainEvent を生成する。

  経路B (VideoChainTracker、デフォルト ON、chain_detector.py:186-250):
      board_for_tracker = self._prev_confirmed_1p (recognition_pipeline.py:
      2036-2043、cycle71d 案D8のコメントで明記) を入力とし、非空セル数が
      4 以上減った時点で「1 frame 前に tracker に渡した board」を
      before_board として ChainEvent を生成する。

  一方、TSUMO_FALL→STABLE の物理推論経路 (infer_placement、
  recognition_pipeline.py:3577-3818) は `prev_state==TSUMO_FALL and
  ctx.state==STABLE` の場合のみ動作し、正しく `inferred_landing`
  (= 設置反映済み盤面) を before_board に使う (line 3798)。
  つまり「配置→即消去」の実況タイミングが速く、ChainPhaseDetector の
  優先度 (state_detectors.py:89-111、CHAIN 判定が TsumoPhaseDetector より
  優先) により ctx.state が STABLE を経由せず TSUMO_FALL→CHAIN に
  直行した場合、上記の正しい経路が一度も実行されないまま stale な
  before_board が使われ続ける。

  本スクリプトはこの「TSUMO_FALL→CHAIN 直行 (STABLE を経由しない state
  遷移)」を客観的フィンガープリントとして検出し、実際に before_board が
  設置直前フレームの CNN 観測と比べて何セル分「足りない」かを定量する。

## 診断区間 (real is_match_active 検出、force_in_match は使わない)
    - c62 game9: 872.4-949.5s (実測済 score0 境界、_run_full_game9_c62 参照)
      前後マージン込みで 862.0-955.0s。
    - video_30 idx3 (233.0-451.0s, 最長ゲーム, match_boundaries_v5):
      225.0-380.0s (計算コスト抑制のため前半 155s に制限)。
    - video_35 idx46 (3118.0-3247.0s, match_boundaries_v5): 3110.0-3255.0s。
    - video_38 idx37 (2593.0-2780.0s, match_boundaries_v5): 2585.0-2735.0s
      (計算コスト抑制のため前半 150s に制限)。

video_30/35/38 は project_indicator_win_eval_2026-07-05 (2026-07-23 追記)
で「序盤 AUC 最低 (0.36-0.37)、video_35/38 は中盤終盤も不振 (0.37-0.44)」
と評価された動画そのもの (仮説: 認識コールドスタート/取りこぼしが真因では
ないか、を突き合わせる対象)。video_30 は同じ評価で中盤終盤とも良好
(AUC 0.8 超) だった対照サンプル。

## 出力
data/verify/recognition_diag_placement_dropout_2026-07-24/
    - summary.json / summary.txt: 動画別・全体の頻度サマリ
    - triggers_<video>.csv: 検出した連鎖トリガー毎の生データ
    - viz_<video>_t<trigger>.png: 代表例の実画面 (設置直前 vs トリガー時)

Usage (WSL 経由、CLAUDE.md プロセス管理ルール準拠):
    wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
      PYTHONPATH=. ./venv/bin/python scripts/_diag_place_coldstart_dropout_2026-07-24.py"
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# 熱対策 (feedback_thermal_safety_mandatory 準拠)。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, Board, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.visualize_recognition import (  # noqa: E402
    P1_ROI_X, P1_ROI_Y, P2_ROI_X, P2_ROI_Y, ROI_H, ROI_W,
    draw_cell_overlay,
)

# ============================
# 定数
# ============================

# 診断対象ウィンドウ: (video_stem, start_sec, end_sec, note)
# c62 は実測 score0 境界 (_run_full_game9_c62_2026-07-23.sh)。
# video_30/35/38 は match_boundaries_v5/*/matches.tsv の最長ゲームを採用
# (計算コスト抑制のため一部は前半のみに制限)。
TARGET_WINDOWS: tuple[tuple[str, float, float, str], ...] = (
    ("c62", 862.0, 955.0, "game9 (872.4-949.5) 実測境界 + warmup margin"),
    ("30", 225.0, 315.0, "idx3 (233-451) 最長ゲーム冒頭90s、良好AUC動画(対照)"),
    ("35", 3110.0, 3200.0, "idx46 (3118-3247) 冒頭90s、序盤/中盤/終盤AUC不振動画"),
    ("38", 2585.0, 2675.0, "idx37 (2593-2780) 冒頭90s、序盤/中盤/終盤AUC不振動画"),
)
# 実測スループット (2026-07-24 smoke test): RecognitionPipeline.load_default() の
# 本番デフォルト設定で約 3.1 fps相当 (=180 frame/58秒)。全 window 合計
# 93+90+90+90=363s (60fps→約21780 frame) で概算 117分 (約2時間) の見込み。

# 連鎖検知が「配置直後 stale before_board」フィンガープリントかどうかの判定に
# 使う、トリガー直前フレームからの遡り数 (chain_detector.SNAPSHOT_LOOKBACK と
# 同じ考え方: 消去アニメが始まる直前の真の盤面を見るため)。
PRE_TRIGGER_LOOKBACK_FRAMES: int = 2

# 有効な puyo 色 (空/UNKNOWN/おじゃま以外)。
_INVALID_COLORS = (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA)

CROP_MARGIN_PX: int = 24
TOTAL_CELLS: int = BOARD_ROWS * BOARD_COLS

OUTPUT_DIR: Path = (
    PROJ_ROOT / "data" / "verify" / "recognition_diag_placement_dropout_2026-07-24"
)

# viz を出す代表例の上限数 (動画毎)。
MAX_VIZ_PER_VIDEO: int = 4


def _video_path(video_stem: str) -> Path:
    return PROJ_ROOT / "data" / "frames" / f"video_{video_stem}.mp4"


# ============================
# データ構造
# ============================


@dataclass
class _FrameRec:
    """1 frame・1 side 分の最小記録 (メモリ節約のため必要最小限のみ保持)。"""

    frame_idx: int
    t: float
    state: str
    confirmed_present: bool
    cnn_grid: np.ndarray  # HxW int8 (raw CNN board、隠し段含む)
    chain_trigger_sec: float | None
    chain_before_grid: np.ndarray | None
    chain_count_event: int | None


@dataclass
class _TriggerRec:
    """1 連鎖トリガー分の診断結果。"""

    video: str
    side: str
    t_trigger: float
    frame_idx: int
    prev_state: str  # トリガー直前フレームの state
    is_skip_event: bool  # prev_state == TSUMO_FALL (STABLE を経由せず直行)
    before_puyo_count: int
    lookback_puyo_count: int
    missing_cells: int  # before=EMPTY かつ lookback=有効色 のセル数
    missing_cells_upper: int  # 上記のうち可視盤面 (隠し段除く) のみ
    predicted_chain_count_before: int
    predicted_chain_count_corrected: int
    severity_confirmed: bool  # missing_cells>=1 かつ chain_count が変化
    before_grid: np.ndarray = field(repr=False, compare=False, default=None)  # viz 専用


# ============================
# パス1: pipeline 走査
# ============================


def _collect_records(
    video_stem: str, start_sec: float, end_sec: float,
) -> tuple[list[_FrameRec], list[_FrameRec], float]:
    """video を走査し、1P/2P それぞれの frame 記録を返す。

    Returns:
        (records_1p, records_2p, fps)
    """
    cv2.setNumThreads(1)
    video_path = _video_path(video_stem)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipe = RecognitionPipeline.load_default()  # 本番デフォルト設定そのまま
    pipe.set_video_id(video_stem)

    recs_1p: list[_FrameRec] = []
    recs_2p: list[_FrameRec] = []
    fi = start_frame
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        for side_recs, side_res in ((recs_1p, r.p1), (recs_2p, r.p2)):
            ce = side_res.chain_event
            side_recs.append(_FrameRec(
                frame_idx=fi, t=t, state=side_res.state.name,
                confirmed_present=side_res.confirmed_board is not None,
                cnn_grid=side_res.cnn_board._grid.copy(),
                chain_trigger_sec=(float(ce.trigger_sec) if ce is not None else None),
                chain_before_grid=(ce.before_board._grid.copy() if ce is not None else None),
                chain_count_event=(int(ce.chain_count) if ce is not None else None),
            ))
        fi += 1
    cap.release()
    return recs_1p, recs_2p, fps


# ============================
# トリガー抽出 + 診断
# ============================


def _new_chain_trigger_idxs(records: list[_FrameRec]) -> list[int]:
    """chain_event が新規出現した frame index 一覧。"""
    idxs: list[int] = []
    last: float | None = None
    for i, rec in enumerate(records):
        if rec.chain_trigger_sec is not None and rec.chain_trigger_sec != last:
            idxs.append(i)
            last = rec.chain_trigger_sec
        elif rec.chain_trigger_sec is None:
            last = None
    return idxs


def _count_puyos(grid: np.ndarray) -> int:
    return int((grid != COLOR_EMPTY).sum())


def _missing_cells(
    before_grid: np.ndarray, lookback_grid: np.ndarray,
) -> tuple[int, int]:
    """before=EMPTY かつ lookback=有効色 のセル数 (全体 / 可視のみ)。"""
    from src.board import HIDDEN_ROWS
    total = 0
    upper = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            bv = int(before_grid[r, c])
            lv = int(lookback_grid[r, c])
            if bv == COLOR_EMPTY and lv not in _INVALID_COLORS:
                total += 1
                if r >= HIDDEN_ROWS:
                    upper += 1
    return total, upper


def _analyze_triggers(
    video_stem: str, side: str, records: list[_FrameRec], sim: ChainSimulator,
) -> list[_TriggerRec]:
    trigger_idxs = _new_chain_trigger_idxs(records)
    out: list[_TriggerRec] = []
    for idx in trigger_idxs:
        rec = records[idx]
        prev_idx = idx - 1
        prev_state = records[prev_idx].state if prev_idx >= 0 else "UNKNOWN"
        lookback_idx = max(0, idx - PRE_TRIGGER_LOOKBACK_FRAMES)
        lookback_grid = records[lookback_idx].cnn_grid
        before_grid = rec.chain_before_grid
        if before_grid is None:
            continue
        missing_total, missing_upper = _missing_cells(before_grid, lookback_grid)
        before_count = _count_puyos(before_grid)
        lookback_count = _count_puyos(lookback_grid)

        pred_before = sim.simulate(Board.from_list(before_grid.tolist()))
        corrected_grid = before_grid.copy()
        # 「取りこぼしセル」を lookback 観測色で埋めた場合の再シミュレート。
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if (
                    int(before_grid[r, c]) == COLOR_EMPTY
                    and int(lookback_grid[r, c]) not in _INVALID_COLORS
                ):
                    corrected_grid[r, c] = lookback_grid[r, c]
        pred_corrected = sim.simulate(Board.from_list(corrected_grid.tolist()))

        out.append(_TriggerRec(
            video=video_stem, side=side, t_trigger=rec.chain_trigger_sec,
            frame_idx=rec.frame_idx, prev_state=prev_state,
            is_skip_event=(prev_state == BoardState.TSUMO_FALL.name),
            before_puyo_count=before_count, lookback_puyo_count=lookback_count,
            missing_cells=missing_total, missing_cells_upper=missing_upper,
            predicted_chain_count_before=pred_before.chain_count,
            predicted_chain_count_corrected=pred_corrected.chain_count,
            severity_confirmed=(
                missing_total > 0
                and pred_before.chain_count != pred_corrected.chain_count
            ),
            before_grid=before_grid,
        ))
    return out


# ============================
# viz
# ============================


def _roi_for_side(side: str) -> tuple[int, int]:
    return (P1_ROI_X, P1_ROI_Y) if side == "1P" else (P2_ROI_X, P2_ROI_Y)


def _crop_roi(frame: np.ndarray, roi_x: int, roi_y: int) -> np.ndarray:
    x1 = max(0, roi_x - CROP_MARGIN_PX)
    y1 = max(0, roi_y - CROP_MARGIN_PX)
    x2 = min(frame.shape[1], roi_x + ROI_W + CROP_MARGIN_PX)
    y2 = min(frame.shape[0], roi_y + ROI_H + CROP_MARGIN_PX)
    return frame[y1:y2, x1:x2].copy()


def _seek_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def _render_viz(
    video_stem: str, tr: _TriggerRec, fps: float,
) -> None:
    """設置直前 (lookback) の実画面 vs トリガー時 before_board overlay を並べる。"""
    cap = cv2.VideoCapture(str(_video_path(video_stem)))
    roi_x, roi_y = _roi_for_side(tr.side)
    lookback_fi = tr.frame_idx - PRE_TRIGGER_LOOKBACK_FRAMES
    frame_lookback = _seek_frame(cap, lookback_fi)
    frame_trigger = _seek_frame(cap, tr.frame_idx)
    cap.release()
    if frame_lookback is None or frame_trigger is None:
        return
    left = _crop_roi(frame_lookback, roi_x, roi_y)
    cv2.putText(
        left, f"実画面(生) t={tr.t_trigger - PRE_TRIGGER_LOOKBACK_FRAMES / fps:.2f}s "
        f"(設置直前/lookback)", (8, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55,
        (0, 255, 255), 2, cv2.LINE_AA,
    )
    # before_board (診断対象の連鎖起点盤面、取りこぼし疑いのある盤面) を
    # トリガー時の実画面上に overlay して目視比較できるようにする。
    right = frame_trigger.copy()
    before_board_obj = Board.from_list(tr.before_grid.tolist())
    draw_cell_overlay(right, before_board_obj, roi_x, roi_y)
    right_c = _crop_roi(right, roi_x, roi_y)
    cv2.putText(
        right_c, f"before_board(起点盤面) t={tr.t_trigger:.2f}s missing={tr.missing_cells}"
        f" skip={tr.is_skip_event}", (8, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55,
        (0, 0, 255) if tr.missing_cells > 0 else (0, 255, 255), 2, cv2.LINE_AA,
    )
    sep = np.full((left.shape[0], 6, 3), (255, 255, 255), dtype=np.uint8)
    out = np.hstack([left, sep, right_c])
    label = f"{tr.t_trigger:.2f}".replace(".", "_")
    cv2.imwrite(str(OUTPUT_DIR / f"viz_{video_stem}_{tr.side}_t{label}.png"), out)


# ============================
# メイン
# ============================


def _print_progress(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim = ChainSimulator()
    all_triggers: list[_TriggerRec] = []
    per_video_summary: dict[str, dict] = {}

    for video_stem, start_sec, end_sec, note in TARGET_WINDOWS:
        _print_progress(
            f"[{video_stem}] 開始 window={start_sec:.1f}-{end_sec:.1f}s ({note})",
        )
        t0 = time.time()
        recs_1p, recs_2p, fps = _collect_records(video_stem, start_sec, end_sec)
        elapsed = time.time() - t0
        _print_progress(
            f"[{video_stem}] pass1 完了 ({len(recs_1p)} frame, {elapsed:.1f}s, "
            f"{len(recs_1p) / max(elapsed, 1e-6):.2f} fps相当)",
        )
        triggers_1p = _analyze_triggers(video_stem, "1P", recs_1p, sim)
        triggers_2p = _analyze_triggers(video_stem, "2P", recs_2p, sim)
        video_triggers = triggers_1p + triggers_2p
        all_triggers.extend(video_triggers)

        # CSV 出力
        csv_path = OUTPUT_DIR / f"triggers_{video_stem}.csv"
        _write_triggers_csv(video_triggers, csv_path)

        n = len(video_triggers)
        n_skip = sum(1 for t in video_triggers if t.is_skip_event)
        n_missing = sum(1 for t in video_triggers if t.missing_cells > 0)
        n_skip_and_missing = sum(
            1 for t in video_triggers if t.is_skip_event and t.missing_cells > 0
        )
        n_severity = sum(1 for t in video_triggers if t.severity_confirmed)
        per_video_summary[video_stem] = {
            "note": note, "window": [start_sec, end_sec],
            "n_chain_triggers": n,
            "n_skip_events_tsumo_to_chain": n_skip,
            "skip_rate": (n_skip / n if n else None),
            "n_triggers_with_missing_cells": n_missing,
            "missing_cell_rate": (n_missing / n if n else None),
            "n_skip_and_missing_both": n_skip_and_missing,
            "n_severity_confirmed_chain_count_changed": n_severity,
        }
        _print_progress(
            f"[{video_stem}] トリガー{n}件 / skip直行{n_skip}件 "
            f"/ missing_cells>0 {n_missing}件 / 両方{n_skip_and_missing}件 "
            f"/ severity確定{n_severity}件",
        )

        # viz: skip_event かつ missing_cells>0 の代表例を最大 MAX_VIZ_PER_VIDEO 件
        viz_targets = [
            t for t in video_triggers if t.is_skip_event and t.missing_cells > 0
        ][:MAX_VIZ_PER_VIDEO]
        for tr in viz_targets:
            _render_viz(video_stem, tr, fps)
        _print_progress(f"[{video_stem}] viz {len(viz_targets)}件 出力完了")

    summary = _build_overall_summary(all_triggers, per_video_summary)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.txt").write_text(_format_summary_text(summary), encoding="utf-8")
    _print_progress(f"[DONE] 出力先: {OUTPUT_DIR}")
    print(_format_summary_text(summary))


def _write_triggers_csv(triggers: list[_TriggerRec], out_path: Path) -> None:
    lines = [
        "video,side,t_trigger,frame_idx,prev_state,is_skip_event,"
        "before_puyo_count,lookback_puyo_count,missing_cells,missing_cells_upper,"
        "predicted_chain_count_before,predicted_chain_count_corrected,severity_confirmed",
    ]
    for t in triggers:
        lines.append(
            f"{t.video},{t.side},{t.t_trigger:.3f},{t.frame_idx},{t.prev_state},"
            f"{t.is_skip_event},{t.before_puyo_count},{t.lookback_puyo_count},"
            f"{t.missing_cells},{t.missing_cells_upper},"
            f"{t.predicted_chain_count_before},{t.predicted_chain_count_corrected},"
            f"{t.severity_confirmed}",
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _build_overall_summary(
    all_triggers: list[_TriggerRec], per_video: dict[str, dict],
) -> dict:
    n = len(all_triggers)
    n_skip = sum(1 for t in all_triggers if t.is_skip_event)
    n_missing = sum(1 for t in all_triggers if t.missing_cells > 0)
    n_skip_and_missing = sum(
        1 for t in all_triggers if t.is_skip_event and t.missing_cells > 0
    )
    n_severity = sum(1 for t in all_triggers if t.severity_confirmed)
    missing_given_skip = [t.missing_cells for t in all_triggers if t.is_skip_event]
    return {
        "n_videos": len(per_video),
        "n_chain_triggers_total": n,
        "n_skip_events_tsumo_to_chain_total": n_skip,
        "skip_rate_overall": (n_skip / n if n else None),
        "n_triggers_with_missing_cells_total": n_missing,
        "missing_cell_rate_overall": (n_missing / n if n else None),
        "n_skip_and_missing_both_total": n_skip_and_missing,
        "rate_of_skip_events_with_missing_cells": (
            n_skip_and_missing / n_skip if n_skip else None
        ),
        "n_severity_confirmed_total": n_severity,
        "missing_cells_mean_given_skip_event": (
            float(np.mean(missing_given_skip)) if missing_given_skip else None
        ),
        "per_video": per_video,
    }


def _format_summary_text(summary: dict) -> str:
    lines = [
        "==== 設置直後の取りこぼし (失敗1) 頻度診断サマリ (2026-07-24) ====",
        f"対象動画数: {summary['n_videos']}",
        f"連鎖トリガー総数: {summary['n_chain_triggers_total']}",
        f"TSUMO_FALL→CHAIN 直行 (STABLE 経由なし) skip イベント数: "
        f"{summary['n_skip_events_tsumo_to_chain_total']} "
        f"(全体比 {summary['skip_rate_overall']})",
        f"before_board に missing_cells>0 (取りこぼし疑い) のトリガー数: "
        f"{summary['n_triggers_with_missing_cells_total']} "
        f"(全体比 {summary['missing_cell_rate_overall']})",
        f"skip かつ missing_cells>0 (両方=直接証拠) 件数: "
        f"{summary['n_skip_and_missing_both_total']} "
        f"(skipイベント中の比率 {summary['rate_of_skip_events_with_missing_cells']})",
        f"severity確定 (missing_cells>0 かつ簡易補正で chain_count が変化): "
        f"{summary['n_severity_confirmed_total']} 件",
        f"skipイベント時の missing_cells 平均: "
        f"{summary['missing_cells_mean_given_skip_event']}",
        "--- 動画別 ---",
    ]
    for video, s in summary["per_video"].items():
        lines.append(
            f"  {video} ({s['note']}, window={s['window']}): "
            f"triggers={s['n_chain_triggers']} skip={s['n_skip_events_tsumo_to_chain']}"
            f"({s['skip_rate']}) missing={s['n_triggers_with_missing_cells']}"
            f"({s['missing_cell_rate']}) both={s['n_skip_and_missing_both']} "
            f"severity={s['n_severity_confirmed_chain_count_changed']}",
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
