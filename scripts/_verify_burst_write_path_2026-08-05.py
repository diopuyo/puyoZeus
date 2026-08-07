"""バーストガードv2 (enable_burst_guard_v2) 書き込み経路検証 (2026-08-05、使い捨て診断)。

`scripts/_verify_persist_inversion_2026-08-05.py` の monkeypatch 計装方式を
流用し、video_c18 2P t≈845.8 (10セル同時焼き付き、glow score 実測1.0=窓が
確実に開く条件) を対象に「窓は開いたのに誤りが防げない」機構を確定する。

## 検証する4分類 (docs/BURST_GUARD_DESIGN_2026-08-05.md §1.1 の仮説)
    (a) 遷移merge経由: NON-STABLE→STABLE 遷移時の `_merge_diff_only`
        (`effect_gate_*` 系引数を一切受け取らない設計上ゲート対象外、
        §1.1) が、その遷移フレームの CNN 誤読 (バースト重畳) をそのまま
        confirmed_board に書き込んだ。
    (b) 復旧経路なのに凍結が効いていない: `_recovery_or_effect_gate_pass`
        が is_gated=True かつ effect_gate_hard_freeze=True にも関わらず
        True (発火) を返した (実装バグ)。
    (c) 窓が実は開いていなかった: 誤値確定の瞬間、`_resolve_burst_gate_state`
        の Schmitt trigger 出力 (is_open) が False だった (閾値/タイミング
        問題、視覚スコアは高くても Window が閉じていた)。
    (d) その他: 上記いずれにも当てはまらない (fallback 経路等、
        設計書 §1.1 「残存リスク」参照)。

## 計装方式 (src/ 本番コード変更禁止、読み取りとラップのみ)
    1. `BoardStateMachine.update` — 対象 side/cell の毎frame値 + 遷移直前
       state を記録 (NON-STABLE→STABLE 遷移の検知に使う)。
    2. `_merge_diff_only` — 呼び出しごとに対象 cell の baseline/cnn/hsv/
       merged 値を記録 (分類(a)の直接証拠)。
    3. `_recovery_or_effect_gate_pass` — 対象 cell 呼び出しごとに
       is_gated/hard_freeze/発火結果を記録 (分類(b)の直接証拠)。
    4. `_resolve_burst_gate_state` — 毎frameの視覚スコア+Window開閉状態を
       記録 (分類(c)の直接証拠)。

## 範囲限定 (cold-start 限界への対処)
前回 (c12/c5, persist_inversion) は開始オフセット30秒で5セット中1セットしか
再現できなかった反省を踏まえ、PRE_WINDOW_SEC を60秒→90秒に拡大する
(HSVオンライン校正/VideoChainTracker のウォームアップ時間を確保)。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._verify_burst_write_path_2026-08-05
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
OUT_DIR: Path = Path("data/verify/burst_write_path_2026-08-05")

# cold-start 限界への対処 (coordinator指示、前回 c12/c5 の反省)。
PRE_WINDOW_SEC: float = 90.0
POST_WINDOW_SEC: float = 8.0

TARGET_W: int = 1920
TARGET_H: int = 1080

# 分類ラベル。
CAT_TRANSITION_MERGE: str = "a_transition_merge"
CAT_FREEZE_NOT_WORKING: str = "b_freeze_not_working"
CAT_WINDOW_NOT_OPEN: str = "c_window_not_open"
CAT_OTHER: str = "d_other"
CAT_UNRESOLVED: str = "unresolved"


@dataclass(frozen=True)
class TargetCell:
    """診断対象の1セル (index_refined.md の該当行から転記、wrong→correct)。"""

    row: int
    col: int
    wrong_value: int
    correct_value: int


@dataclass(frozen=True)
class VerifyCase:
    """1動画分の検証ケース。"""

    video_stem: str
    side: str
    onset_t_sec: float
    cells: tuple[TargetCell, ...]


# data/verify/error_onset_sheet_2026-08-04/index_refined.md (行23-32) より転記。
# onset_t_sec は data/verify/effect_gate_2026-08-04_c/diag_zero_effect.csv の
# c18 行 (onset_frame_idx=50746) から精密値を採用。
CASES: tuple[VerifyCase, ...] = (
    VerifyCase(
        video_stem="c18", side="2P", onset_t_sec=845.7670288085938,
        cells=(
            TargetCell(1, 1, 4, 0), TargetCell(1, 2, 4, 0), TargetCell(1, 3, 4, 0),
            TargetCell(2, 0, 4, 1), TargetCell(2, 1, 4, 1), TargetCell(2, 2, 4, 0),
            TargetCell(2, 3, 4, 0), TargetCell(2, 4, 9, 5), TargetCell(2, 5, 4, 1),
            TargetCell(3, 1, 4, 1),
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
class MergeRecord:
    """`_merge_diff_only` 1 呼び出し分の対象 cell 観測。"""

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
    # 重要: target_side と不一致でも必ず更新する (= 1P呼び出し中は "1P" に
    # しておかないと、直前フレームの2P呼び出しで残った値が nested な
    # `_merge_diff_only` 呼び出し (1P分) を誤って2Pとして記録してしまう)。
    _STATE.current_side = side or ""
    ctx = _ORIG_SM_UPDATE(self, frame_idx, signals)
    if side == _STATE.target_side:
        for cell in _STATE.target_cells:
            _record_cell_frame(prev_state, ctx, signals, cell)
    return ctx


# =============================================================================
# monkeypatch 2: _merge_diff_only (分類(a)の直接証拠)
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
# monkeypatch 3: _recovery_or_effect_gate_pass (分類(b)の直接証拠)
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
# monkeypatch 4: _resolve_burst_gate_state (分類(c)の直接証拠)
# =============================================================================

_ORIG_RESOLVE_BURST: Callable = rp._resolve_burst_gate_state


def _region_side(region: object) -> str:
    """region オブジェクトの identity から side を判定する。"""
    return "1P" if region is DEFAULT_P1_REGION else "2P"


def _wrapped_resolve_burst_gate_state(
    frame_bgr: "np.ndarray | None", region: object, rows: "frozenset[int]",
    prev_open: bool, prev_opened_at: "float | None", prev_quiet: "float | None",
    time_sec: float, force_close: bool,
) -> "tuple[bool, float | None, float | None]":
    """`_resolve_burst_gate_state` のラップ (score + Window開閉を記録)。"""
    result = _ORIG_RESOLVE_BURST(
        frame_bgr, region, rows, prev_open, prev_opened_at, prev_quiet,
        time_sec, force_close,
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
    """全4 monkeypatch をインストールする (src/ 本番ファイルは書き換えない)。"""
    bsm.BoardStateMachine.update = _wrapped_sm_update
    bsm._merge_diff_only = _wrapped_merge_diff_only
    bsm._recovery_or_effect_gate_pass = _wrapped_recovery_pass
    rp._resolve_burst_gate_state = _wrapped_resolve_burst_gate_state


def uninstall_probes() -> None:
    """monkeypatch を復元する (他スクリプトへの汚染防止)。"""
    bsm.BoardStateMachine.update = _ORIG_SM_UPDATE
    bsm._merge_diff_only = _ORIG_MERGE_DIFF_ONLY
    bsm._recovery_or_effect_gate_pass = _ORIG_RECOVERY_PASS
    rp._resolve_burst_gate_state = _ORIG_RESOLVE_BURST


# =============================================================================
# 1ケース分の範囲限定再走行
# =============================================================================


def _build_pipeline() -> RecognitionPipeline:
    """collect_boards_lean.collect_lean と同一構成 (--enable-burst-guard-v2 相当)。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
    )


def run_case(case: VerifyCase) -> None:
    """1ケース分を onset 周辺だけ再走行し、4種の全frame記録を残す。"""
    _STATE.target_side = case.side
    _STATE.target_cells = case.cells
    _STATE.target_cell_set = frozenset((c.row, c.col) for c in case.cells)
    _STATE.cell_records = []
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
    """4種の記録をそれぞれ CSV に出力する。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = case.video_stem
    _write_records_csv(OUT_DIR / f"{stem}_cell_timeline.csv", _STATE.cell_records)
    _write_records_csv(OUT_DIR / f"{stem}_merge_events.csv", _STATE.merge_records)
    _write_records_csv(OUT_DIR / f"{stem}_recovery_events.csv", _STATE.recovery_records)
    _write_records_csv(OUT_DIR / f"{stem}_burst_gate_timeline.csv", _STATE.burst_records)
    print(f"[{stem}] CSV出力: 4ファイル ({OUT_DIR})")


# =============================================================================
# 経路分類レポート
# =============================================================================


def _find_confirm_record(
    records: list[CellFrameRecord], wrong_value: int, onset_t_sec: float,
) -> "CellFrameRecord | None":
    """confirmed_v が wrong_value になった記録のうち onset_t_sec に最も近い1件を返す。

    PRE_WINDOW_SEC を90秒に拡大したため、対象色 (例: 4=黄) が観測窓の
    早い時間帯に正当な理由で出現している可能性がある (=誤値と同じ色値だが
    無関係な通常プレイ)。「最初の出現」でなく「実際の onset に最も近い
    出現」を採用することでこの偽陽性を避ける。
    """
    candidates = [r for r in records if r.confirmed_v == wrong_value]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(r.t_sec - onset_t_sec))


def _burst_open_at(t_sec: float) -> "bool | None":
    """指定 t_sec に最も近い (以下最新の) burst window 状態を返す。"""
    candidates = [r for r in _STATE.burst_records if r.t_sec <= t_sec]
    return candidates[-1].is_open_after if candidates else None


def _classify_confirm(
    cell: TargetCell, confirm: CellFrameRecord,
) -> str:
    """確定frameの証拠から経路 (a)/(b)/(c)/(d) を1つ確定する。"""
    if confirm.transitioned_to_stable:
        has_merge_evidence = any(
            m.frame_idx == confirm.frame_idx and (m.row, m.col) == (cell.row, cell.col)
            and m.merged_v == cell.wrong_value
            for m in _STATE.merge_records
        )
        if has_merge_evidence:
            return CAT_TRANSITION_MERGE
    bad_recovery = [
        r for r in _STATE.recovery_records
        if r.frame_idx == confirm.frame_idx and (r.row, r.col) == (cell.row, cell.col)
        and r.is_gated and r.hard_freeze and r.fired
    ]
    if bad_recovery:
        return CAT_FREEZE_NOT_WORKING
    window_open = _burst_open_at(confirm.t_sec)
    if window_open is False:
        return CAT_WINDOW_NOT_OPEN
    return CAT_OTHER


def _report_cell(case: VerifyCase, cell: TargetCell) -> None:
    """1 cell の判定を stdout に出力する。"""
    recs = [r for r in _STATE.cell_records if (r.row, r.col) == (cell.row, cell.col)]
    confirm = _find_confirm_record(recs, cell.wrong_value, case.onset_t_sec)
    print(f"  --- {case.video_stem} {case.side} row{cell.row}col{cell.col} (wrong={cell.wrong_value}) ---")
    if confirm is None:
        print(f"    [{CAT_UNRESOLVED}] 観測窓内で wrong_value に確定した frame が見つからない")
        return
    category = _classify_confirm(cell, confirm)
    window_open = _burst_open_at(confirm.t_sec)
    print(
        f"    確定frame: frame_idx={confirm.frame_idx} t_sec={confirm.t_sec:.3f} "
        f"prev_state={confirm.prev_state} state={confirm.state} "
        f"transitioned_to_stable={confirm.transitioned_to_stable} "
        f"cnn={confirm.cnn_v} hsv={confirm.hsv_v} window_open={window_open}"
    )
    print(f"    [分類] {category}")


def _report_case(case: VerifyCase) -> None:
    """1ケース全 cell の判定レポート + 経路別集計を出力する。"""
    print(f"\n=== {case.video_stem} ({case.side}) 判定 ===")
    categories: list[str] = []
    for cell in case.cells:
        _report_cell(case, cell)
        recs = [r for r in _STATE.cell_records if (r.row, r.col) == (cell.row, cell.col)]
        confirm = _find_confirm_record(recs, cell.wrong_value, case.onset_t_sec)
        categories.append(
            _classify_confirm(cell, confirm) if confirm is not None else CAT_UNRESOLVED
        )
    print(f"\n  --- 経路別集計 ({len(categories)}セル) ---")
    for cat in (CAT_TRANSITION_MERGE, CAT_FREEZE_NOT_WORKING, CAT_WINDOW_NOT_OPEN, CAT_OTHER, CAT_UNRESOLVED):
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
