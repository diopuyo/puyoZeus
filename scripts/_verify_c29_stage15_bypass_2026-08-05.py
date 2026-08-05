"""残存33セル「要個別調査」事例 (video_c29 2P t=2408.6, 3セル) の経路確定 (2026-08-05)。

`scripts/_verify_burst_write_path_2026-08-05.py` の monkeypatch 計装方式を
流用し、v2本命構成 (`enable_burst_guard_v2` + `enable_transition_merge_guard`
+ `burst_gate_open_threshold=0.954`) で「窓が確実に開いていた (score=1.000)
のに OFF 時の誤り (5=紫) が v2 でさらに悪化 (9=おじゃま 等) した」3セルの
書き込み経路を確定する。

`data/verify/burst_guard_2026-08-05/residual33_cell_classification.csv` の
c29@2408.6 行 (row2, col1/2/3) が対象。

## 検証する4分類 (coordinator指示)
    (a) OJAMA_FALL許容スコープの正規通過: `_filter_transition_new_cnn_for_burst_guard`
        が from_state=OJAMA_FALL で正しく評価され、baseline=EMPTY かつ
        cnn_v=9 (許容色) だったため通過した (=設計限界、おじゃま実落下枠が
        広すぎてバースト誤読の9も通す)。
    (b) from_state誤判定: フィルタは評価されたが from_state が実際は
        TSUMO_FALL 等で、`_TRANSITION_MERGE_GUARD_SCOPE` の許容色集合を
        誤って適用した。
    (c) フィルタ非評価の別経路: `signals.effect_gate_window_active` が
        その遷移フレームでは実は False だった、または
        `enable_transition_merge_guard` 自体が効いていない等の理由で
        フィルタ関数が一度も呼ばれなかった。
    (d) その他: 上記いずれにも当てはまらない (recovery経路・infer_hidden_row
        等の別経路)。

## 計装方式 (src/ 本番コード変更禁止、読み取りとラップのみ)
    1. `BoardStateMachine.update` — side/time相関 + 遷移直前 state を記録。
    2. `_filter_transition_new_cnn_for_burst_guard` — 呼び出しごとに
       from_state/baseline_v/raw_cnn_v/filtered_v を記録 (分類(a)/(b)/(c)の
       直接証拠、フィルタが「呼ばれたか」自体も本記録の有無で判定できる)。
    3. `_merge_diff_only` — 対象 cell の baseline/cnn(=フィルタ後)/hsv/merged
       を記録 (最終的な書き込み値の直接証拠)。
    4. `_recovery_or_effect_gate_pass` — 対象 cell の is_gated/hard_freeze/
       発火結果を記録 (分類(d)のうち recovery経由の可能性を排除)。
    5. `_resolve_burst_gate_state` — 毎frameの視覚スコアとWindow開閉を記録。

## 範囲限定 (cold-start 対策、coordinator指示で長めに)
PRE_WINDOW_SEC=90秒 (前回 `_verify_burst_write_path_2026-08-05.py` と同一方針)。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._verify_c29_stage15_bypass_2026-08-05
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.board_state_machine as bsm  # noqa: E402
import src.recognition_pipeline as rp  # noqa: E402
from src.board_state_machine import BoardState, DetectorSignals, NON_STABLE_STATES  # noqa: E402
from src.effect_glow_detector import EFFECT_GATE_TOP_ROWS, compute_effect_glow_score  # noqa: E402
from src.fps_normalize import resolve_normalize_fps_30_stride  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# =============================================================================
# 定数
# =============================================================================

VIDEO_DIR: Path = Path("/home/ryouj/frames")
OUT_DIR: Path = Path("data/verify/c29_stage15_bypass_2026-08-05")

PRE_WINDOW_SEC: float = 90.0
POST_WINDOW_SEC: float = 8.0

TARGET_W: int = 1920
TARGET_H: int = 1080

# v2本命構成 (scripts/_jobs_burst_guard_full_2026-08-05.txt と同一)。
BURST_GATE_OPEN_THRESHOLD_RECALIBRATED: float = 0.954

# 分類ラベル。
CAT_OJAMA_SCOPE_LEGIT: str = "a_ojama_fall_scope_legit"
CAT_FROM_STATE_MISDETECT: str = "b_from_state_misdetect"
CAT_FILTER_NOT_EVALUATED: str = "c_filter_not_evaluated"
CAT_OTHER: str = "d_other"
CAT_UNRESOLVED: str = "unresolved"


@dataclass(frozen=True)
class TargetCell:
    """診断対象の1セル (residual33_cell_classification.csv の該当行から転記)。"""

    row: int
    col: int
    off_value: int
    correct_value: int
    v2_value: int


@dataclass(frozen=True)
class VerifyCase:
    """1動画分の検証ケース。"""

    video_stem: str
    side: str
    onset_t_sec: float
    cells: tuple[TargetCell, ...]


# data/verify/burst_guard_2026-08-05/residual33_cell_classification.csv
# (video,side,label_t_sec,row,col,off_value,correct_value,v2_match_mode,
#  v2_value,onset_t_sec,...) c29@2408.6 の3行より転記。
CASES: tuple[VerifyCase, ...] = (
    VerifyCase(
        video_stem="c29", side="2P", onset_t_sec=2408.599,
        cells=(
            TargetCell(2, 1, off_value=5, correct_value=4, v2_value=9),
            TargetCell(2, 2, off_value=5, correct_value=1, v2_value=5),
            TargetCell(2, 3, off_value=5, correct_value=0, v2_value=5),
        ),
    ),
)


# =============================================================================
# 記録用データ構造
# =============================================================================


@dataclass
class CellFrameRecord:
    """1 cell × 1 frame の観測 (対象 side のみ記録)。"""

    frame_idx: int
    t_sec: float
    prev_state: str
    state: str
    row: int
    col: int
    cnn_v: "int | None"
    hsv_v: "int | None"
    confirmed_v: "int | None"
    transitioned_to_stable: bool


@dataclass
class FilterEventRecord:
    """`_filter_transition_new_cnn_for_burst_guard` 1 呼び出し分の対象 cell 観測。"""

    frame_idx: int
    t_sec: float
    from_state: str
    row: int
    col: int
    baseline_v: "int | None"
    raw_cnn_v: int
    filtered_v: int
    allowed_colors: str


@dataclass
class MergeRecord:
    """`_merge_diff_only` 1 呼び出し分の対象 cell 観測 (フィルタ後の値が入る)。"""

    frame_idx: int
    t_sec: float
    row: int
    col: int
    baseline_v: "int | None"
    cnn_v: int
    hsv_v: "int | None"
    merged_v: int


@dataclass
class RecoveryPassRecord:
    """`_recovery_or_effect_gate_pass` 1 呼び出し分の対象 cell 観測。"""

    frame_idx: int
    t_sec: float
    row: int
    col: int
    is_gated: bool
    hard_freeze: bool
    confirmed_v: int
    agreed_v: int
    fired: bool


@dataclass
class BurstGateRecord:
    """`_resolve_burst_gate_state` 1 呼び出し分の観測 (対象 side のみ)。"""

    frame_idx: int
    t_sec: float
    score: "float | None"
    is_open_after: bool


@dataclass
class _ProbeState:
    """計装の可変状態 (case ごとにリセットする)。"""

    target_side: str = ""
    target_cells: tuple[TargetCell, ...] = ()
    target_cell_set: frozenset[tuple[int, int]] = frozenset()
    current_side: str = ""
    current_frame_idx: int = -1
    current_t_sec: float = -1.0
    sm_side_map: dict[int, str] = field(default_factory=dict)
    ctx_side_map: dict[int, str] = field(default_factory=dict)
    cell_records: list[CellFrameRecord] = field(default_factory=list)
    filter_records: list[FilterEventRecord] = field(default_factory=list)
    merge_records: list[MergeRecord] = field(default_factory=list)
    recovery_records: list[RecoveryPassRecord] = field(default_factory=list)
    burst_records: list[BurstGateRecord] = field(default_factory=list)


_STATE = _ProbeState()


# =============================================================================
# monkeypatch 1: BoardStateMachine.update (side/time相関 + 遷移直前state)
# =============================================================================

_ORIG_SM_UPDATE: Callable = bsm.BoardStateMachine.update


def _record_cell_frame(
    prev_state: BoardState, ctx: "bsm.StateContext", signals: DetectorSignals,
    cell: TargetCell,
) -> None:
    """1 cell 分の frame 観測を記録する (遷移直前state付き)。"""
    r, c = cell.row, cell.col
    cnn_v = int(signals.cnn_board.get(r, c)) if signals.cnn_board is not None else None
    hsv_v = int(signals.hsv_board.get(r, c)) if signals.hsv_board is not None else None
    confirmed_v = (
        int(ctx.confirmed_board.get(r, c)) if ctx.confirmed_board is not None else None
    )
    transitioned = prev_state in NON_STABLE_STATES and ctx.state == BoardState.STABLE
    _STATE.cell_records.append(CellFrameRecord(
        frame_idx=_STATE.current_frame_idx, t_sec=signals.time_sec,
        prev_state=prev_state.name, state=ctx.state.name, row=r, col=c,
        cnn_v=cnn_v, hsv_v=hsv_v, confirmed_v=confirmed_v,
        transitioned_to_stable=transitioned,
    ))


def _wrapped_sm_update(
    self: "bsm.BoardStateMachine", frame_idx: int, signals: DetectorSignals,
) -> "bsm.StateContext":
    """`BoardStateMachine.update` のラップ (対象 side のみ全 target cell を記録)。"""
    side = _STATE.sm_side_map.get(id(self))
    prev_state = self._ctx.state
    # 重要: target_side と不一致でも必ず更新する (前回スクリプトの罠と同型)。
    _STATE.current_side = side or ""
    ctx = _ORIG_SM_UPDATE(self, frame_idx, signals)
    if side == _STATE.target_side:
        for cell in _STATE.target_cells:
            _record_cell_frame(prev_state, ctx, signals, cell)
    return ctx


# =============================================================================
# monkeypatch 2: _filter_transition_new_cnn_for_burst_guard (分類(a)/(b)/(c))
# =============================================================================

_ORIG_FILTER: Callable = bsm._filter_transition_new_cnn_for_burst_guard


def _wrapped_filter(
    baseline: "bsm.Board | None", new_cnn: "bsm.Board", from_state: BoardState,
) -> "bsm.Board":
    """`_filter_transition_new_cnn_for_burst_guard` のラップ (呼ばれたか自体を記録)。"""
    filtered = _ORIG_FILTER(baseline, new_cnn, from_state)
    if _STATE.current_side == _STATE.target_side:
        allowed = bsm._TRANSITION_MERGE_GUARD_SCOPE.get(from_state)
        for cell in _STATE.target_cells:
            r, c = cell.row, cell.col
            _STATE.filter_records.append(FilterEventRecord(
                frame_idx=_STATE.current_frame_idx, t_sec=_STATE.current_t_sec,
                from_state=from_state.name, row=r, col=c,
                baseline_v=(int(baseline.get(r, c)) if baseline is not None else None),
                raw_cnn_v=int(new_cnn.get(r, c)), filtered_v=int(filtered.get(r, c)),
                allowed_colors=(sorted(allowed) if allowed is not None else "None"),
            ))
    return filtered


# =============================================================================
# monkeypatch 3: _merge_diff_only (最終書き込み値の直接証拠)
# =============================================================================

_ORIG_MERGE_DIFF_ONLY: Callable = bsm._merge_diff_only


def _wrapped_merge_diff_only(
    baseline: "bsm.Board | None", new_cnn: "bsm.Board", **kwargs: object,
) -> "bsm.Board":
    """`_merge_diff_only` のラップ (対象 cell の baseline/cnn/merged 値を記録)。"""
    merged = _ORIG_MERGE_DIFF_ONLY(baseline, new_cnn, **kwargs)
    if _STATE.current_side != _STATE.target_side:
        return merged
    hsv_board = kwargs.get("hsv_board")
    for cell in _STATE.target_cells:
        r, c = cell.row, cell.col
        _STATE.merge_records.append(MergeRecord(
            frame_idx=_STATE.current_frame_idx, t_sec=_STATE.current_t_sec,
            row=r, col=c,
            baseline_v=(int(baseline.get(r, c)) if baseline is not None else None),
            cnn_v=int(new_cnn.get(r, c)),
            hsv_v=(int(hsv_board.get(r, c)) if hsv_board is not None else None),
            merged_v=int(merged.get(r, c)),
        ))
    return merged


# =============================================================================
# monkeypatch 4: _recovery_or_effect_gate_pass (recovery経路の排除確認)
# =============================================================================

_ORIG_RECOVERY_PASS: Callable = bsm._recovery_or_effect_gate_pass


def _wrapped_recovery_pass(
    ctx: "bsm.StateContext", cell: "tuple[int, int]", confirmed_v: int,
    agreed_v: int, recovery_counters: "dict", min_frames: int,
    add_min_frames: "int | None", effect_gate_active_rows: "frozenset[int] | None",
    effect_gate_persist_sec: float, effect_gate_hard_freeze: bool = False,
) -> bool:
    """`_recovery_or_effect_gate_pass` のラップ (対象 cell の判定過程を記録)。"""
    fired = _ORIG_RECOVERY_PASS(
        ctx, cell, confirmed_v, agreed_v, recovery_counters, min_frames,
        add_min_frames, effect_gate_active_rows, effect_gate_persist_sec,
        effect_gate_hard_freeze,
    )
    side = _STATE.ctx_side_map.get(id(ctx))
    if side == _STATE.target_side and cell in _STATE.target_cell_set:
        is_gated = (
            effect_gate_active_rows is not None and cell[0] in effect_gate_active_rows
        )
        _STATE.recovery_records.append(RecoveryPassRecord(
            frame_idx=_STATE.current_frame_idx, t_sec=_STATE.current_t_sec,
            row=cell[0], col=cell[1], is_gated=is_gated,
            hard_freeze=effect_gate_hard_freeze, confirmed_v=confirmed_v,
            agreed_v=agreed_v, fired=fired,
        ))
    return fired


# =============================================================================
# monkeypatch 5: _resolve_burst_gate_state (窓開閉の直接証拠)
# =============================================================================

_ORIG_RESOLVE_BURST: Callable = rp._resolve_burst_gate_state


def _region_side(region: object) -> str:
    """region オブジェクトの identity から side を判定する。"""
    return "1P" if region is DEFAULT_P1_REGION else "2P"


def _wrapped_resolve_burst_gate_state(
    frame_bgr: "np.ndarray | None", region: object, rows: "frozenset[int]",
    prev_open: bool, prev_opened_at: "float | None", prev_quiet: "float | None",
    time_sec: float, force_close: bool,
    open_threshold: float = rp.BURST_GATE_OPEN_THRESHOLD,
    close_threshold: float = rp.BURST_GATE_CLOSE_THRESHOLD,
) -> "tuple[bool, float | None, float | None]":
    """`_resolve_burst_gate_state` のラップ (score + Window開閉を記録)。

    2026-08-05 閾値パラメータ化 (`burst_gate_open_threshold`) 追加分の
    `open_threshold`/`close_threshold` 引数もそのまま転送する
    (前回スクリプト作成時点にはまだ存在しなかった新引数)。
    """
    result = _ORIG_RESOLVE_BURST(
        frame_bgr, region, rows, prev_open, prev_opened_at, prev_quiet,
        time_sec, force_close, open_threshold, close_threshold,
    )
    if _region_side(region) == _STATE.target_side:
        score = (
            compute_effect_glow_score(frame_bgr, region, rows)
            if frame_bgr is not None else None
        )
        _STATE.burst_records.append(BurstGateRecord(
            frame_idx=_STATE.current_frame_idx, t_sec=_STATE.current_t_sec,
            score=score, is_open_after=result[0],
        ))
    return result


def install_probes() -> None:
    """全5 monkeypatch をインストールする (src/ 本番ファイルは書き換えない)。"""
    bsm.BoardStateMachine.update = _wrapped_sm_update
    bsm._filter_transition_new_cnn_for_burst_guard = _wrapped_filter
    bsm._merge_diff_only = _wrapped_merge_diff_only
    bsm._recovery_or_effect_gate_pass = _wrapped_recovery_pass
    rp._resolve_burst_gate_state = _wrapped_resolve_burst_gate_state


def uninstall_probes() -> None:
    """monkeypatch を復元する (他スクリプトへの汚染防止)。"""
    bsm.BoardStateMachine.update = _ORIG_SM_UPDATE
    bsm._filter_transition_new_cnn_for_burst_guard = _ORIG_FILTER
    bsm._merge_diff_only = _ORIG_MERGE_DIFF_ONLY
    bsm._recovery_or_effect_gate_pass = _ORIG_RECOVERY_PASS
    rp._resolve_burst_gate_state = _ORIG_RESOLVE_BURST


# =============================================================================
# 1ケース分の範囲限定再走行
# =============================================================================


def _build_pipeline() -> RecognitionPipeline:
    """v2本命構成 (scripts/_jobs_burst_guard_full_2026-08-05.txt と同一)。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=BURST_GATE_OPEN_THRESHOLD_RECALIBRATED,
    )


def run_case(case: VerifyCase) -> None:
    """1ケース分を onset 周辺だけ再走行し、5種の全frame記録を残す。"""
    _STATE.target_side = case.side
    _STATE.target_cells = case.cells
    _STATE.target_cell_set = frozenset((c.row, c.col) for c in case.cells)
    _STATE.cell_records = []
    _STATE.filter_records = []
    _STATE.merge_records = []
    _STATE.recovery_records = []
    _STATE.burst_records = []

    video_path = VIDEO_DIR / f"video_{case.video_stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_sec = max(0.0, case.onset_t_sec - PRE_WINDOW_SEC)
    start_frame = int(start_sec * fps)
    n_frames = int((PRE_WINDOW_SEC + POST_WINDOW_SEC) * fps) + int(fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipeline = _build_pipeline()
    pipeline.set_video_id(f"video_{case.video_stem}")
    _STATE.sm_side_map = {id(pipeline._sm_1p): "1P", id(pipeline._sm_2p): "2P"}
    _STATE.ctx_side_map = {
        id(pipeline._sm_1p.context): "1P", id(pipeline._sm_2p.context): "2P",
    }
    stride = resolve_normalize_fps_30_stride(fps)
    print(
        f"[{case.video_stem}] start_sec={start_sec:.2f} start_frame={start_frame} "
        f"fps={fps:.2f} stride={stride} n_frames={n_frames}"
    )

    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if local_i % stride != 0:
            continue
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        fi = start_frame + local_i
        t_sec = fi / fps
        _STATE.current_frame_idx = fi
        _STATE.current_t_sec = t_sec
        pipeline.update(fi, t_sec, frame)
    cap.release()
    _write_case_csv(case)
    _report_case(case)


# =============================================================================
# CSV 出力
# =============================================================================


def _write_records_csv(path: Path, records: list) -> None:
    """dataclass のリストを1枚のCSVに書き出す共通ヘルパー。"""
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].__dataclass_fields__))
        writer.writeheader()
        for r in records:
            writer.writerow(r.__dict__)


def _write_case_csv(case: VerifyCase) -> None:
    """5種の記録をそれぞれ CSV に出力する。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = case.video_stem
    _write_records_csv(OUT_DIR / f"{stem}_cell_timeline.csv", _STATE.cell_records)
    _write_records_csv(OUT_DIR / f"{stem}_filter_events.csv", _STATE.filter_records)
    _write_records_csv(OUT_DIR / f"{stem}_merge_events.csv", _STATE.merge_records)
    _write_records_csv(OUT_DIR / f"{stem}_recovery_events.csv", _STATE.recovery_records)
    _write_records_csv(OUT_DIR / f"{stem}_burst_gate_timeline.csv", _STATE.burst_records)
    print(f"[{stem}] CSV出力: 5ファイル ({OUT_DIR})")


# =============================================================================
# 経路分類レポート
# =============================================================================


def _find_confirm_record(
    records: list[CellFrameRecord], target_value: int, onset_t_sec: float,
) -> "CellFrameRecord | None":
    """confirmed_v が target_value になった記録のうち onset_t_sec に最も近い1件を返す。"""
    candidates = [r for r in records if r.confirmed_v == target_value]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(r.t_sec - onset_t_sec))


def _burst_open_at(t_sec: float) -> "bool | None":
    """指定 t_sec に最も近い (以下最新の) burst window 状態を返す。"""
    candidates = [r for r in _STATE.burst_records if r.t_sec <= t_sec]
    return candidates[-1].is_open_after if candidates else None


def _classify_confirm(cell: TargetCell, confirm: CellFrameRecord) -> str:
    """確定frameの証拠から経路 (a)/(b)/(c)/(d) を1つ確定する。"""
    filter_calls = [
        f for f in _STATE.filter_records
        if f.frame_idx == confirm.frame_idx and (f.row, f.col) == (cell.row, cell.col)
    ]
    if not filter_calls:
        return CAT_FILTER_NOT_EVALUATED
    fc = filter_calls[-1]
    if fc.from_state != "OJAMA_FALL":
        return CAT_FROM_STATE_MISDETECT
    if fc.filtered_v == cell.v2_value and fc.baseline_v == 0:
        return CAT_OJAMA_SCOPE_LEGIT
    return CAT_OTHER


def _report_cell(case: VerifyCase, cell: TargetCell) -> None:
    """1 cell の判定を stdout に出力する。"""
    recs = [r for r in _STATE.cell_records if (r.row, r.col) == (cell.row, cell.col)]
    confirm = _find_confirm_record(recs, cell.v2_value, case.onset_t_sec)
    print(
        f"  --- {case.video_stem} {case.side} row{cell.row}col{cell.col} "
        f"(off={cell.off_value}, v2={cell.v2_value}, correct={cell.correct_value}) ---"
    )
    if confirm is None:
        print(f"    [{CAT_UNRESOLVED}] 観測窓内で v2_value に確定した frame が見つからない")
        return
    category = _classify_confirm(cell, confirm)
    window_open = _burst_open_at(confirm.t_sec)
    filter_calls = [
        f for f in _STATE.filter_records
        if f.frame_idx == confirm.frame_idx and (f.row, f.col) == (cell.row, cell.col)
    ]
    print(
        f"    確定frame: frame_idx={confirm.frame_idx} t_sec={confirm.t_sec:.3f} "
        f"prev_state={confirm.prev_state} state={confirm.state} "
        f"cnn={confirm.cnn_v} hsv={confirm.hsv_v} window_open={window_open}"
    )
    if filter_calls:
        fc = filter_calls[-1]
        print(
            f"    filter: from_state={fc.from_state} allowed={fc.allowed_colors} "
            f"baseline={fc.baseline_v} raw_cnn={fc.raw_cnn_v} filtered={fc.filtered_v}"
        )
    else:
        print("    filter: 呼び出し記録なし (この frame では評価されていない)")
    print(f"    [分類] {category}")


def _report_case(case: VerifyCase) -> None:
    """1ケース全 cell の判定レポート + 経路別集計を出力する。"""
    print(f"\n=== {case.video_stem} ({case.side}) 判定 ===")
    categories: list[str] = []
    for cell in case.cells:
        _report_cell(case, cell)
        recs = [r for r in _STATE.cell_records if (r.row, r.col) == (cell.row, cell.col)]
        confirm = _find_confirm_record(recs, cell.v2_value, case.onset_t_sec)
        categories.append(
            _classify_confirm(cell, confirm) if confirm is not None else CAT_UNRESOLVED
        )
    print(f"\n  --- 経路別集計 ({len(categories)}セル) ---")
    for cat in (
        CAT_OJAMA_SCOPE_LEGIT, CAT_FROM_STATE_MISDETECT,
        CAT_FILTER_NOT_EVALUATED, CAT_OTHER, CAT_UNRESOLVED,
    ):
        n = categories.count(cat)
        if n:
            print(f"    {cat}: {n}件")


# =============================================================================
# メイン
# =============================================================================


def main() -> None:
    install_probes()
    try:
        for case in CASES:
            run_case(case)
    finally:
        uninstall_probes()


if __name__ == "__main__":
    main()
