"""真因診断: c62 game9 1P の estimated_board 崩壊 (2026-07-23)。

完全 read-only 診断スクリプト。src/ および既存 scripts/ は一切変更しない。

背景 (依頼元の要約):
    data/verify/recognition_physics_review/20260723_174208_after_iter6_c62.json
    で 1P の estimated_board coverage が 9.8% (2P は 65-99.8%) と崩壊しており、
    ghost_mismatch_events の t=902.8 (11 cell / 14%) と t=914.3 (35-37 cell /
    45-47%) が user 目視の「7秒付近の1P上部誤認」と時間帯・side が一致する。

切り分けたい仮説:
    H1: before_board (連鎖起点=直前確定盤面) 自体の CNN 誤読で simulate が
        破綻し、推定 (chain_count>0) が立ち上がらない。
    H2: 小連鎖 (chain_count=1) が短間隔連続する場面で state が瞬間的に
        CHAIN/GRAVITY_SETTLE から外れ、保持中の estimate が破棄され次
        trigger まで空白になる。
    H3 (計装調査で追加): VideoChainTracker への入力 board_for_tracker が
        hold 中に一時的に noisy raw CNN へフォールバックし、
        `_last_stable_board` (= 次回 before_board) を汚染していないか
        (recognition_pipeline.py:1980-1987 のコメント上は
        `_prev_confirmed_1p` は STABLE 以外で「前回値を維持」する設計だが、
        実データで本当にそう動いているかを計装で確認する)。

出力先: data/verify/recognition_diag_c62_1p_estimate_collapse/
    - trigger_<t>_before.png            : 連鎖起点 (before_board) 実画面+overlay
    - trigger_<t>_actual_vs_expected.png : 解消後 実画面の 実測 vs 物理予測 比較
    - mismatch_cell_dump_<t>.json        : 必須2点 (902.8 / 914.3) の cell別 HSV/CNN dump
    - state_trace.csv                    : frame毎 state/confirmed/estimated/provenance
    - coverage_timeline.png              : estimated_board 補完可否のタイムライン
    - score_ocr_crosscheck.json          : simulate由来score vs 実測score OCR差分
    - summary.json / summary.txt         : 数値サマリ + 結論

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/_diag_estimate_collapse_c62_1p.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 熱対策 (feedback_thermal_safety_mandatory 準拠、read-only診断でも一応制限)。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, Board, HIDDEN_ROWS,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.recognition_pipeline import (  # noqa: E402
    DEFAULT_P1_REGION, RecognitionPipeline,
)
from scripts.visualize_recognition import (  # noqa: E402
    CELL_H, CELL_W, COLOR_BGR, P1_ROI_X, P1_ROI_Y, ROI_H, ROI_W,
    draw_cell_overlay,
)

# ============================
# 定数
# ============================
VIDEO_PATH: str = "data/frames/video_c62.mp4"
PROC_START_SEC: float = 850.0   # state machine warmup 用マージン
DIAG_START_SEC: float = 895.0   # 診断区間 (physics_review と同一窓)
DIAG_END_SEC: float = 960.0
RECORD_MARGIN_SEC: float = 5.0  # trigger 前後余裕を持って記録開始

# 必須2点 (依頼本文の t≈902.8 / t≈914.3) を許容誤差内で判定するための tolerance。
REQUIRED_TRIGGER_TIMES: tuple[float, ...] = (902.8, 914.3)
REQUIRED_TRIGGER_TOL_SEC: float = 0.5

# viz 用: ROI crop に付与する余白 (実画面の隣接文脈を見えるようにする)。
CROP_MARGIN_PX: int = 24
# post-erasure グレア確認用にトリガーからずらす秒数。
POST_ERASURE_OFFSETS_SEC: tuple[float, ...] = (0.5, 1.0)
# 上部 row 定義 (user「上部」発言との整合確認用): 隠し段 row0 の次、
# 可視 row1-2 (盤面座標系、HIDDEN_ROWS=1 なので可視最上段は row1)。
UPPER_VISIBLE_ROWS: tuple[int, ...] = (1, 2)
# 光沢ぷよ誤読ヒューリスティック (project_specular_highlight_empty_misread):
# 彩度が低く明度が高い = 白ハイライト疑い。
SPECULAR_S_MAX: int = 60
SPECULAR_V_MIN: int = 200

TOTAL_CELLS: int = BOARD_ROWS * BOARD_COLS

OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "recognition_diag_c62_1p_estimate_collapse"


# ============================
# データ構造
# ============================


@dataclass
class _Rec:
    """1 frame・1P 分の観測値 (診断専用、recognition_physics_review._FrameRecord の拡張)。"""

    frame_idx: int
    t: float
    state: str
    confirmed_grid: np.ndarray | None
    estimated_grid: np.ndarray | None
    provenance: str
    cnn_grid: np.ndarray
    score: int | None
    chain_trigger_sec: float | None
    chain_before_grid: np.ndarray | None
    chain_count_event: int | None
    chain_total_score: int | None
    chain_ojama_sent: int | None
    prev_confirmed_snapshot: np.ndarray | None  # H3 調査用 (pipe._prev_confirmed_1p)


def _build_rec(fi: int, t: float, side: object, prev_confirmed: np.ndarray | None) -> _Rec:
    """SideResult (1P) から 1 frame 分の _Rec を組み立てる。"""
    ce = side.chain_event
    return _Rec(
        frame_idx=fi, t=t, state=side.state.name,
        confirmed_grid=(
            side.confirmed_board._grid.copy() if side.confirmed_board is not None else None
        ),
        estimated_grid=(
            side.estimated_board._grid.copy() if side.estimated_board is not None else None
        ),
        provenance=side.board_provenance,
        cnn_grid=side.cnn_board._grid.copy(),
        score=side.score,
        chain_trigger_sec=float(ce.trigger_sec) if ce is not None else None,
        chain_before_grid=ce.before_board._grid.copy() if ce is not None else None,
        chain_count_event=int(ce.chain_count) if ce is not None else None,
        chain_total_score=int(ce.total_score) if ce is not None else None,
        chain_ojama_sent=int(ce.ojama_sent) if ce is not None else None,
        prev_confirmed_snapshot=prev_confirmed.copy() if prev_confirmed is not None else None,
    )


# ============================
# パス1: pipeline 走査 (state machine warmup + 記録)
# ============================


def _collect_records() -> tuple[list[_Rec], float, object]:
    """video を warmup 付きで処理し、診断区間 (+マージン) の 1P 記録を返す。

    Returns:
        (records, fps, reader): reader は pipe._reader (HSV-only 分類器
        アクセス用に流用、パス2 の cell dump で新規 ImageReader を
        再構築せず同一 calibration 状態を使い回すため)。
    """
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(PROJ_ROOT / VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    start_frame = int(PROC_START_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    end_frame = int(DIAG_END_SEC * fps)
    keep_from_sec = DIAG_START_SEC - RECORD_MARGIN_SEC

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    pipe.set_video_id("c62")

    records: list[_Rec] = []
    fi = start_frame
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        if t >= keep_from_sec:
            prev_conf = getattr(pipe, "_prev_confirmed_1p", None)
            prev_conf_grid = prev_conf._grid if prev_conf is not None else None
            records.append(_build_rec(fi, t, r.p1, prev_conf_grid))
        fi += 1
    cap.release()
    print(f"[pass1] {len(records)} frame 記録 (fps={fps:.2f})")
    return records, fps, pipe._reader


# ============================
# トリガー抽出・答え合わせ
# ============================


def _new_chain_triggers(records: list[_Rec]) -> list[int]:
    """chain_event が新規出現した frame の index 一覧 (trigger_sec 変化で検出)。"""
    idxs: list[int] = []
    last: float | None = None
    for i, rec in enumerate(records):
        if rec.chain_trigger_sec is not None and rec.chain_trigger_sec != last:
            idxs.append(i)
            last = rec.chain_trigger_sec
        elif rec.chain_trigger_sec is None:
            last = None
    return idxs


def _find_first_stable_after(records: list[_Rec], start_idx: int) -> int | None:
    """start_idx 以降で最初に STABLE かつ confirmed_grid が有効な frame index。"""
    for i in range(start_idx, len(records)):
        if records[i].state == BoardState.STABLE.name and records[i].confirmed_grid is not None:
            return i
    return None


def _first_valid_score_from(records: list[_Rec], start_idx: int, forward: bool) -> int | None:
    """start_idx から前方 (forward=False) または後方 (forward=True) に score None でない値を探す。"""
    rng = range(start_idx, len(records)) if forward else range(start_idx, -1, -1)
    for i in rng:
        if records[i].score is not None:
            return records[i].score
    return None


@dataclass
class _TriggerAnalysis:
    """1 連鎖トリガー分の答え合わせ結果。"""

    t_trigger: float
    trigger_idx: int
    before_grid: np.ndarray
    predicted_chain_count: int
    predicted_final_grid: np.ndarray | None
    event_chain_count: int
    event_total_score: int
    first_stable_idx: int | None
    actual_grid: np.ndarray | None
    mismatch_cells: list[tuple[int, int]]
    ghost_duration_sec: float | None
    resolved: bool | None
    score_before_ocr: int | None
    score_after_ocr: int | None
    ocr_delta: int | None
    prev_confirmed_matches_before: bool  # H3 調査: before_grid == 直前保持値か
    # H2 調査 (main() 側で後付けする、初期値 False): この trigger の
    # first_stable_idx に到達する前に「次の trigger」が発火していたら True。
    # True の場合、この trigger 単体の ocr_delta/mismatch は複数連鎖が
    # 混ざった値になっている疑いがある (= 短間隔連続で state が
    # 一度も真に STABLE 復帰しないまま次の連鎖に突入している直接証拠)。
    contaminated_by_next_trigger: bool = False


def _analyze_trigger(records: list[_Rec], idx: int, sim: ChainSimulator) -> _TriggerAnalysis:
    """1 トリガー分の物理答え合わせ + score OCR 突合を計算する。"""
    rec = records[idx]
    before = Board.from_list(rec.chain_before_grid.tolist())
    sim_result = sim.simulate(before)
    first_stable_idx = _find_first_stable_after(records, idx)
    actual_grid = records[first_stable_idx].confirmed_grid if first_stable_idx is not None else None
    mismatch: list[tuple[int, int]] = []
    expected_grid = sim_result.final_board._grid if sim_result.chain_count > 0 else None
    if actual_grid is not None and expected_grid is not None:
        diff = actual_grid != expected_grid
        mismatch = [(int(r), int(c)) for r, c in zip(*np.where(diff))]
    ghost_dur, resolved = _ghost_resolution(records, first_stable_idx, expected_grid)
    score_before = _first_valid_score_from(records, idx, forward=False)
    score_after = (
        _first_valid_score_from(records, first_stable_idx, forward=True)
        if first_stable_idx is not None else None
    )
    ocr_delta = (
        score_after - score_before
        if score_before is not None and score_after is not None else None
    )
    prev_ok = (
        rec.prev_confirmed_snapshot is not None
        and np.array_equal(rec.prev_confirmed_snapshot, rec.chain_before_grid)
    )
    return _TriggerAnalysis(
        t_trigger=rec.chain_trigger_sec, trigger_idx=idx, before_grid=rec.chain_before_grid,
        predicted_chain_count=sim_result.chain_count,
        predicted_final_grid=expected_grid,
        event_chain_count=rec.chain_count_event, event_total_score=rec.chain_total_score,
        first_stable_idx=first_stable_idx, actual_grid=actual_grid, mismatch_cells=mismatch,
        ghost_duration_sec=ghost_dur, resolved=resolved,
        score_before_ocr=score_before, score_after_ocr=score_after, ocr_delta=ocr_delta,
        prev_confirmed_matches_before=prev_ok,
    )


def _mark_contaminated_triggers(analyses: list[_TriggerAnalysis]) -> None:
    """H2 調査: 次 trigger が first_stable 到達前に発火したケースに印を付ける (in-place)。

    analyses は trigger_idx 昇順 (時系列順) であることを前提とする
    (`_new_chain_triggers` の走査順そのままのため保証される)。
    """
    for i in range(len(analyses) - 1):
        cur, nxt = analyses[i], analyses[i + 1]
        boundary = cur.first_stable_idx if cur.first_stable_idx is not None else float("inf")
        if nxt.trigger_idx < boundary:
            cur.contaminated_by_next_trigger = True


def _ghost_resolution(
    records: list[_Rec], first_stable_idx: int | None, expected_grid: np.ndarray | None,
) -> tuple[float | None, bool | None]:
    """first_stable_idx 以降で expected_grid に一致する最初の frame までの遅延秒。"""
    if first_stable_idx is None or expected_grid is None:
        return None, None
    t0 = records[first_stable_idx].t
    max_sec = 15.0
    for i in range(first_stable_idx, len(records)):
        rec = records[i]
        if rec.t - t0 > max_sec:
            break
        if rec.confirmed_grid is not None and np.array_equal(rec.confirmed_grid, expected_grid):
            return rec.t - t0, True
    return max_sec, False


# ============================
# viz: 実画面 + 認識 overlay
# ============================


def _crop_roi(frame: np.ndarray) -> np.ndarray:
    """1P 盤面 ROI を余白付きで crop する。"""
    x1 = max(0, P1_ROI_X - CROP_MARGIN_PX)
    y1 = max(0, P1_ROI_Y - CROP_MARGIN_PX)
    x2 = min(frame.shape[1], P1_ROI_X + ROI_W + CROP_MARGIN_PX)
    y2 = min(frame.shape[0], P1_ROI_Y + ROI_H + CROP_MARGIN_PX)
    return frame[y1:y2, x1:x2].copy(), x1, y1


def _render_before_panel(frame: np.ndarray, before_grid: np.ndarray, t_trigger: float) -> np.ndarray:
    """連鎖起点 (before_board) を実画面上に overlay した画像を作る。"""
    canvas = frame.copy()
    draw_cell_overlay(canvas, Board.from_list(before_grid.tolist()), P1_ROI_X, P1_ROI_Y)
    crop, _, _ = _crop_roi(canvas)
    cv2.putText(
        crop, f"before_board t={t_trigger:.2f}s", (8, 24),
        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return crop


def _draw_mismatch_boxes(canvas: np.ndarray, mismatch: list[tuple[int, int]]) -> None:
    """mismatch cell を赤枠で強調する (可視 row のみ、隠し row は overlay 対象外のため skip)。"""
    for row, col in mismatch:
        if row < HIDDEN_ROWS:
            continue
        display_row = row - HIDDEN_ROWS
        x1 = P1_ROI_X + col * CELL_W
        y1 = P1_ROI_Y + display_row * CELL_H
        cv2.rectangle(canvas, (x1, y1), (x1 + CELL_W, y1 + CELL_H), (0, 0, 255), 3)


def _render_actual_vs_expected(
    frame: np.ndarray, actual_grid: np.ndarray, expected_grid: np.ndarray,
    mismatch: list[tuple[int, int]], t_label: float,
) -> np.ndarray:
    """解消後の実画面に「実測」と「物理予測」を並べて overlay し、mismatch を赤枠強調する。"""
    left = frame.copy()
    draw_cell_overlay(left, Board.from_list(actual_grid.tolist()), P1_ROI_X, P1_ROI_Y)
    _draw_mismatch_boxes(left, mismatch)
    right = frame.copy()
    draw_cell_overlay(right, Board.from_list(expected_grid.tolist()), P1_ROI_X, P1_ROI_Y)
    _draw_mismatch_boxes(right, mismatch)
    left_c, _, _ = _crop_roi(left)
    right_c, _, _ = _crop_roi(right)
    for img, label in ((left_c, "ACTUAL(認識確定)"), (right_c, "EXPECTED(物理予測)")):
        cv2.putText(
            img, f"{label} t={t_label:.2f}s", (8, 24),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
    sep = np.full((left_c.shape[0], 6, 3), (255, 255, 255), dtype=np.uint8)
    return np.hstack([left_c, sep, right_c])


# ============================
# HSV/CNN cell dump (必須2点専用)
# ============================


def _row_bucket(row: int) -> str:
    """行位置の分類 (H2/user発言整合確認用)。"""
    if row < HIDDEN_ROWS:
        return "hidden_row0"
    if row in UPPER_VISIBLE_ROWS:
        return "upper_visible_row1_2"
    return "rest"


def _cell_hsv_stats(frame: np.ndarray, row: int, col: int) -> dict:
    """1 cell の HSV 中央値 + 光沢(specular)ヒューリスティック判定。"""
    x1, y1, x2, y2 = DEFAULT_P1_REGION.cell_sample_rect(row, col)
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(x1 + 1, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(y1 + 1, min(y2, h))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return {"h": -1, "s": -1, "v": -1, "specular_suspect": False}
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_med, s_med, v_med = (int(np.median(hsv[:, :, i])) for i in range(3))
    specular = s_med <= SPECULAR_S_MAX and v_med >= SPECULAR_V_MIN
    return {"h": h_med, "s": s_med, "v": v_med, "specular_suspect": specular}


def _dump_mismatch_cells(
    reader: object, frame: np.ndarray, ana: _TriggerAnalysis,
) -> list[dict]:
    """必須2点用: mismatch cell 毎に expected/actual/cnn/hsv-only/生HSV を dump する。"""
    hsv_board = reader.read_board_hsv_only(frame, DEFAULT_P1_REGION)
    out: list[dict] = []
    for row, col in ana.mismatch_cells:
        expected = int(ana.predicted_final_grid[row, col])
        actual = int(ana.actual_grid[row, col])
        cnn_raw = int(-1)  # 呼出元 (main) が記録済み cnn_grid から埋める
        hsv_only = int(hsv_board.get(row, col))
        stats = _cell_hsv_stats(frame, row, col)
        out.append({
            "row": row, "col": col, "row_bucket": _row_bucket(row),
            "expected_color": expected, "actual_confirmed_color": actual,
            "cnn_raw_color": cnn_raw, "hsv_only_color": hsv_only, **stats,
        })
    return out


# ============================
# メイン
# ============================


def main() -> None:
    """診断本体: pass1 走査 → トリガー答え合わせ → viz/JSON 出力。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records, fps, reader = _collect_records()
    diag_records = [r for r in records if DIAG_START_SEC <= r.t <= DIAG_END_SEC]
    trigger_idxs_full = _new_chain_triggers(records)
    trigger_idxs = [
        i for i in trigger_idxs_full if DIAG_START_SEC <= records[i].t <= DIAG_END_SEC
    ]
    print(f"[pass1] 診断区間内 1P トリガー数: {len(trigger_idxs)}")

    sim = ChainSimulator()
    analyses = [_analyze_trigger(records, i, sim) for i in trigger_idxs]
    _mark_contaminated_triggers(analyses)
    _write_state_trace(diag_records, OUTPUT_DIR / "state_trace.csv")
    _write_coverage_timeline(diag_records, analyses, OUTPUT_DIR / "coverage_timeline.png")
    _render_and_dump_triggers(records, analyses, fps, reader)
    summary = _build_summary(analyses)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.txt").write_text(_format_summary_text(summary), encoding="utf-8")
    print(f"[DONE] 出力先: {OUTPUT_DIR}")
    print(_format_summary_text(summary))


def _write_state_trace(records: list[_Rec], out_path: Path) -> None:
    """frame毎 state/coverage 系列を CSV 出力する。"""
    lines = ["t_sec,frame_idx,state,confirmed_present,estimated_present,provenance,score"]
    for r in records:
        lines.append(
            f"{r.t:.3f},{r.frame_idx},{r.state},"
            f"{int(r.confirmed_grid is not None)},{int(r.estimated_grid is not None)},"
            f"{r.provenance},{r.score if r.score is not None else ''}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_coverage_timeline(
    records: list[_Rec], analyses: list[_TriggerAnalysis], out_path: Path,
) -> None:
    """estimated_board 補完可否のタイムライン画像を出力する (matplotlib)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r.t for r in records])
    # カテゴリ: 0=confirmed(観測), 1=estimated(chain_estimate), 2=estimated(low_confidence),
    # 3=空白(None/None) 。
    cat = np.zeros(len(records), dtype=np.int32)
    for i, r in enumerate(records):
        if r.confirmed_grid is not None:
            cat[i] = 0
        elif r.estimated_grid is not None and r.provenance == "chain_estimate":
            cat[i] = 1
        elif r.estimated_grid is not None and r.provenance == "chain_estimate_low_confidence":
            cat[i] = 2
        else:
            cat[i] = 3
    colors = {0: "#2ca02c", 1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}
    fig, ax = plt.subplots(figsize=(16, 3))
    for k, color in colors.items():
        mask = cat == k
        ax.scatter(t[mask], np.ones(mask.sum()), c=color, s=6, marker="s")
    for ana in analyses:
        ax.axvline(ana.t_trigger, color="black", linestyle="--", alpha=0.5)
        ax.text(ana.t_trigger, 1.05, f"{ana.t_trigger:.1f}", rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("time (sec)")
    ax.set_title(
        "1P coverage timeline: green=confirmed blue=estimate orange=low_conf red=NONE",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _render_raw_strip(cap: cv2.VideoCapture, trigger_frame_idx: int, fps: float) -> np.ndarray:
    """連鎖起点フレーム+POST_ERASURE_OFFSETS_SEC 経過後の生画像 (overlay無し) を横に並べる。

    「連鎖前後のフレームの生画像も並べて起点盤面が実画面と合っているか目視
    できる形に」という要件専用 (認識結果に依存しない純粋なスクリーンショット列)。
    """
    offsets = (0.0,) + POST_ERASURE_OFFSETS_SEC
    panels: list[np.ndarray] = []
    for off in offsets:
        fi = trigger_frame_idx + int(round(off * fps))
        frame = _seek_frame(cap, fi)
        if frame is None:
            continue
        crop, _, _ = _crop_roi(frame)
        cv2.putText(
            crop, f"+{off:.1f}s", (8, 24), cv2.FONT_HERSHEY_DUPLEX, 0.7,
            (0, 255, 255), 2, cv2.LINE_AA,
        )
        panels.append(crop)
    if not panels:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    sep = np.full((panels[0].shape[0], 6, 3), (255, 255, 255), dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, sep, p])
    return out


def _render_and_dump_triggers(
    records: list[_Rec], analyses: list[_TriggerAnalysis], fps: float, reader: object,
) -> None:
    """各トリガーの viz 画像を出力し、必須2点は cell dump も追加出力する。"""
    cap = cv2.VideoCapture(str(PROJ_ROOT / VIDEO_PATH))
    for ana in analyses:
        label = f"{ana.t_trigger:.2f}".replace(".", "_")
        trigger_frame_idx = records[ana.trigger_idx].frame_idx
        frame_before = _seek_frame(cap, trigger_frame_idx)
        if frame_before is not None:
            img = _render_before_panel(frame_before, ana.before_grid, ana.t_trigger)
            cv2.imwrite(str(OUTPUT_DIR / f"trigger_{label}_before.png"), img)
        strip = _render_raw_strip(cap, trigger_frame_idx, fps)
        cv2.imwrite(str(OUTPUT_DIR / f"trigger_{label}_raw_strip.png"), strip)
        if ana.first_stable_idx is not None and ana.predicted_final_grid is not None:
            fi_stable = records[ana.first_stable_idx].frame_idx
            frame_stable = _seek_frame(cap, fi_stable)
            if frame_stable is not None:
                cmp_img = _render_actual_vs_expected(
                    frame_stable, ana.actual_grid, ana.predicted_final_grid,
                    ana.mismatch_cells, records[ana.first_stable_idx].t,
                )
                cv2.imwrite(str(OUTPUT_DIR / f"trigger_{label}_actual_vs_expected.png"), cmp_img)
                if _is_required_trigger(ana.t_trigger):
                    _dump_required_trigger(records, ana, frame_stable, label, reader)
    cap.release()


def _is_required_trigger(t_trigger: float) -> bool:
    """依頼本文の必須2点 (t≈902.8 / t≈914.3) に該当するかを許容誤差付きで判定。"""
    return any(abs(t_trigger - req) <= REQUIRED_TRIGGER_TOL_SEC for req in REQUIRED_TRIGGER_TIMES)


def _dump_required_trigger(
    records: list[_Rec], ana: _TriggerAnalysis, frame_stable: np.ndarray, label: str,
    reader: object,
) -> None:
    """必須2点: mismatch cell 毎の HSV/CNN raw dump + row 分布を JSON 出力する。

    reader は pass1 で使った pipe._reader をそのまま流用する (per-video
    calibration 済みの HSV 分類器状態を再利用するため、新規 ImageReader は
    構築しない)。
    """
    cells = _dump_mismatch_cells(reader, frame_stable, ana)
    cnn_grid = records[ana.first_stable_idx].cnn_grid
    for cell in cells:
        cell["cnn_raw_color"] = int(cnn_grid[cell["row"], cell["col"]])
    row_hist: dict[str, int] = {}
    for cell in cells:
        row_hist[cell["row_bucket"]] = row_hist.get(cell["row_bucket"], 0) + 1
    out = {
        "t_trigger": ana.t_trigger, "n_mismatch": len(cells),
        "row_bucket_histogram": row_hist, "cells": cells,
        "predicted_chain_count": ana.predicted_chain_count,
        "event_chain_count": ana.event_chain_count,
        "ocr_delta": ana.ocr_delta, "event_total_score": ana.event_total_score,
        "prev_confirmed_matches_before": ana.prev_confirmed_matches_before,
    }
    (OUTPUT_DIR / f"mismatch_cell_dump_{label}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _seek_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    """指定 frame index の BGR フレームを直接 seek して取得する。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def _build_summary(analyses: list[_TriggerAnalysis]) -> dict:
    """H1/H2/H3 切り分け用の数値サマリを構築する。"""
    n = len(analyses)
    n_predicted_zero = sum(1 for a in analyses if a.predicted_chain_count == 0)
    n_count_mismatch = sum(
        1 for a in analyses
        if a.predicted_chain_count > 0 and a.predicted_chain_count != a.event_chain_count
    )
    ocr_diffs = [
        abs(a.event_total_score - a.ocr_delta)
        for a in analyses if a.ocr_delta is not None
    ]
    prev_confirmed_ok = sum(1 for a in analyses if a.prev_confirmed_matches_before)
    n_contaminated = sum(1 for a in analyses if a.contaminated_by_next_trigger)
    return {
        "n_triggers_1p_895_960": n,
        "n_predicted_chain_zero_by_before_board": n_predicted_zero,
        "n_predicted_chain_count_mismatch": n_count_mismatch,
        "score_ocr_abs_diff_mean": (float(np.mean(ocr_diffs)) if ocr_diffs else None),
        "score_ocr_abs_diff_max": (float(np.max(ocr_diffs)) if ocr_diffs else None),
        "n_score_ocr_pairs": len(ocr_diffs),
        "h3_prev_confirmed_matches_before_rate": (
            prev_confirmed_ok / n if n > 0 else None
        ),
        # H2 調査: 次 trigger が first_stable 到達前に発火した (= 短間隔連続で
        # 真の STABLE 復帰を挟まずに次連鎖へ突入した) trigger の割合。
        "h2_contaminated_by_next_trigger_rate": (n_contaminated / n if n > 0 else None),
        "h2_contaminated_by_next_trigger_count": n_contaminated,
        "triggers": [
            {
                "t_trigger": a.t_trigger, "predicted_chain_count": a.predicted_chain_count,
                "event_chain_count": a.event_chain_count,
                "mismatch_cells": len(a.mismatch_cells),
                "mismatch_rate": len(a.mismatch_cells) / TOTAL_CELLS,
                "ocr_delta": a.ocr_delta, "event_total_score": a.event_total_score,
                "score_diff": (
                    abs(a.event_total_score - a.ocr_delta)
                    if a.ocr_delta is not None else None
                ),
                "prev_confirmed_matches_before": a.prev_confirmed_matches_before,
                "contaminated_by_next_trigger": a.contaminated_by_next_trigger,
            }
            for a in analyses
        ],
    }


def _format_summary_text(summary: dict) -> str:
    """人間向けテキストサマリ (viz レビュー資料用)。"""
    lines = [
        "==== c62 game9 1P estimated_board 崩壊 真因診断 サマリ ====",
        f"診断区間内トリガー数: {summary['n_triggers_1p_895_960']}",
        f"before_board を simulate して chain_count=0 (推定が立ち上がらない): "
        f"{summary['n_predicted_chain_zero_by_before_board']} 件",
        f"chain_count 不一致 (立ち上がるが数が違う=low_confidence相当): "
        f"{summary['n_predicted_chain_count_mismatch']} 件",
        f"score OCR 絶対差 平均/最大: "
        f"{summary['score_ocr_abs_diff_mean']} / {summary['score_ocr_abs_diff_max']} "
        f"(n={summary['n_score_ocr_pairs']})",
        f"H3 (before_board==直前保持confirmed) 一致率: "
        f"{summary['h3_prev_confirmed_matches_before_rate']}",
        f"H2 (次trigger混入=真のSTABLE復帰を挟まず連続発火): "
        f"{summary['h2_contaminated_by_next_trigger_count']} 件 "
        f"({summary['h2_contaminated_by_next_trigger_rate']})",
        "--- トリガー別 ---",
    ]
    for tr in summary["triggers"]:
        lines.append(
            f"  t={tr['t_trigger']:.2f} pred_cc={tr['predicted_chain_count']} "
            f"event_cc={tr['event_chain_count']} mismatch={tr['mismatch_cells']}"
            f"({tr['mismatch_rate']:.2f}) ocr_delta={tr['ocr_delta']} "
            f"event_score={tr['event_total_score']} diff={tr['score_diff']} "
            f"prev_ok={tr['prev_confirmed_matches_before']} "
            f"contaminated={tr['contaminated_by_next_trigger']}",
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
