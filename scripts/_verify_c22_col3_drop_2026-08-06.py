"""診断1: v4構成 c22 2P f188154 の12セル列ドロップ機構確定 (2026-08-06、使い捨て診断)。

物差し52盤面測定 (baseline16 vs v4=28 悪化) の主因。v4構成 (Stage1.5+0.954+
1.5b、§12系なし) で col3 rows3-11 (9セル) + col4 rows5-7 (3セル) の計12セル
が、correct (baseline npz) では色ぷよが積まれているのに v4 では全てEMPTYに
確定する。窓付き収集・全編収集の両方で同一の12セルが再現する決定論的な
挙動を、OFF対照 (バーストガード全部無し) と並走させて経路別に確定する。

## 対象セル (baseline npz vs v4_prod npz の直接diffで確定済み)
    col3: rows 3,4,5,6,7,8,9,10,11 (baseline値 [1,4,3,3,4,4,4,4,3] → v4は全0)
    col4: rows 5,6,7               (baseline値 [1,4,3]             → v4は全0)

## 検証する仮説 (coordinator指示)
    (i)  フィルタが正規設置を棄却し続けた蓄積 (c19雪だるまの亜種、
         §12 [close延長/クールダウン] が無くても起きるか)
    (ii) ×印/UI被り等の視覚障害由来 (バーストガードと無関係)
    (iii) バースト頻発区間で窓が高デューティ (raw score 自体が長時間 高い)

## 計装方式 (src/ 本番コード変更禁止、読み取りとラップのみ)
    1. `RecognitionPipeline._step_side` — side/time相関 + own_chain/opp_chain
       (v4 pipeline のみ、pipeline識別は id map で行う)。
    2. `_filter_transition_new_cnn_for_burst_guard` — 対象12セルの
       from_state/baseline_v/raw_cnn_v/filtered_v を毎呼び出し記録
       (仮説(i)の直接証拠)。
    3. `_recovery_or_effect_gate_pass` — 対象12セルの is_gated/hard_freeze/
       発火結果を記録 (recovery経路の関与を確認)。
    4. `_resolve_burst_gate_state` / `_resolve_effective_burst_gate_active`
       — 毎frameの視覚スコア・raw_open・実効active を記録 (仮説(iii)の
       直接証拠、デューティ比を算出する)。
    5. OFF/v4 二重pipelineで同一フレームを並走させ、confirmed_board を
       直接比較する (OFFが正しく積み上がることの対照確認)。

## 範囲限定 (coordinator指定)
onset_t_sec=3135.9 (f188154, 60fps) の前60秒〜後10秒。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._verify_c22_col3_drop_2026-08-06
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
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
from src.fps_normalize import resolve_normalize_fps_30_stride  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# =============================================================================
# 定数
# =============================================================================

VIDEO_DIR: Path = Path("/home/ryouj/frames")
OUT_DIR: Path = Path("data/verify/c22_col3_drop_2026-08-06")
VIDEO_STEM: str = "c22"
TARGET_SIDE: str = "2P"

ONSET_T_SEC: float = 3135.9  # f188154 @ 60fps
PRE_WINDOW_SEC: float = 60.0
POST_WINDOW_SEC: float = 10.0

# 2026-08-06 修正: coordinator指定の「前60秒」(t=3075.9開始) は baseline npz
# (c22_g43.npz, t_sec範囲 3110.7〜3157.8) より前の "前の試合" に踏み込んでいる
# ことが実測で判明 (OFF側も同一12セルで失敗 = cold-start自体が試合境界を
# 跨いで無効化されていた)。真の試合開始 t≈3110.72 (baseline npz先頭) の直後
# から再走行することで、対象試合の最初から列の積み上がりを追跡できるように
# する (シーン逆算ではなく、baseline npz自体の試合境界という物理的事実に
# 基づく補正)。
GAME_START_T_SEC: float = 3110.72
GAME_START_MARGIN_SEC: float = 1.0

# 追加検証用トグル (下記コメント参照)。既定Falseでcoordinator指定通りの
# fps正規化stride、Trueで全frame処理 (baseline npz "allframes_ref" 相当)。
FORCE_ALL_FRAMES: bool = True

# 追加検証2: 「試合開始直後リセット」が本当に十分か切り分けるため、前の試合
# (game_idx=42) の終盤から跨いで開始する (実際のreset()発火の瞬間を自分の
# runでも経験させる)。
CROSS_BOUNDARY_START_T_SEC: "float | None" = 3050.0

# 2026-08-06 機構確定実験 (docs/LONGRUN_DEGRADATION_INVESTIGATION_2026-08-06.md
# アーキ設計、コーディネータ指示): 実験A/Bの切替。
#   "A": OFF/v4 両方に enable_drift_resync_hsv_gate=False を明示指定
#        (③ drift resync 安全弁の永久停止が主因なら解消するはず)。
#   "B": RecognitionPipeline.reset() をラップし、末尾で
#        _online_hsv/_online_hsv_injected/_online_hsv_injected_colors を
#        追加クリアする診断monkeypatch (①②=凍結+reset未クリアが主因なら
#        解消するはず)。src/本体は変更しない (このラップのみ)。
EXPERIMENT: str = "A"

TARGET_W: int = 1920
TARGET_H: int = 1080

BURST_GATE_OPEN_THRESHOLD_V4: float = 0.954

# baseline npz (correct) との直接diffで確定した12セル (row, col, correct_value)。
TARGET_CELLS: "tuple[tuple[int, int, int], ...]" = (
    (3, 3, 1), (4, 3, 4), (5, 3, 3), (6, 3, 3), (7, 3, 4),
    (8, 3, 4), (9, 3, 4), (10, 3, 3), (11, 3, 3),  # 2026-08-06 修正: row10c3 correct=3 (誤記4→3)
    (5, 4, 1), (6, 4, 4), (7, 4, 3),
)


@dataclass
class FilterEvent:
    """`_filter_transition_new_cnn_for_burst_guard` 1呼び出し分の対象セル観測。"""

    frame_idx: int
    t_sec: float
    from_state: str
    row: int
    col: int
    baseline_v: "int | None"
    raw_cnn_v: int
    filtered_v: int


@dataclass
class RecoveryEvent:
    """`_recovery_or_effect_gate_pass` 1呼び出し分の対象セル観測。"""

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
class GateFrame:
    """burst gate 1frame分の観測。"""

    frame_idx: int
    t_sec: float
    score: "float | None"
    raw_open: bool
    effective_active: bool


@dataclass
class _ProbeState:
    """計装の可変状態。"""

    pipeline_label_map: dict[int, str] = field(default_factory=dict)
    ctx_label_map: dict[int, str] = field(default_factory=dict)
    current_side: str = ""
    current_frame_idx: int = -1
    current_t_sec: float = -1.0
    filter_events: list[FilterEvent] = field(default_factory=list)
    recovery_events: list[RecoveryEvent] = field(default_factory=list)
    gate_frames: list[GateFrame] = field(default_factory=list)


_STATE = _ProbeState()
_TARGET_RC = frozenset((r, c) for r, c, _ in TARGET_CELLS)


# =============================================================================
# monkeypatch 1: RecognitionPipeline._step_side (side/time相関、v4のみ)
# =============================================================================

_ORIG_STEP_SIDE: Callable = RecognitionPipeline._step_side


def _wrapped_step_side(self: RecognitionPipeline, side: str, frame_idx: int, *args: object, **kwargs: object):
    """`_step_side` のラップ (v4 pipeline 呼び出し時のみ context を更新)。"""
    label = _STATE.pipeline_label_map.get(id(self))
    if label == "V4" and side == TARGET_SIDE:
        _STATE.current_side = side
        _STATE.current_frame_idx = frame_idx
        _STATE.current_t_sec = float(args[0]) if args else _STATE.current_t_sec
    else:
        _STATE.current_side = ""
    return _ORIG_STEP_SIDE(self, side, frame_idx, *args, **kwargs)


# =============================================================================
# monkeypatch 2: _filter_transition_new_cnn_for_burst_guard (仮説(i)の直接証拠)
# =============================================================================

_ORIG_FILTER: Callable = bsm._filter_transition_new_cnn_for_burst_guard


def _wrapped_filter(
    baseline: "bsm.Board | None", new_cnn: "bsm.Board", from_state: "bsm.BoardState",
) -> "bsm.Board":
    """`_filter_transition_new_cnn_for_burst_guard` のラップ (対象12セルを記録)。"""
    filtered = _ORIG_FILTER(baseline, new_cnn, from_state)
    if _STATE.current_side != TARGET_SIDE or baseline is None:
        return filtered
    for r, c in _TARGET_RC:
        _STATE.filter_events.append(FilterEvent(
            frame_idx=_STATE.current_frame_idx, t_sec=_STATE.current_t_sec,
            from_state=from_state.name, row=r, col=c,
            baseline_v=int(baseline.get(r, c)), raw_cnn_v=int(new_cnn.get(r, c)),
            filtered_v=int(filtered.get(r, c)),
        ))
    return filtered


# =============================================================================
# monkeypatch 3: _recovery_or_effect_gate_pass (recovery経路の関与確認)
# =============================================================================

_ORIG_RECOVERY_PASS: Callable = bsm._recovery_or_effect_gate_pass


def _wrapped_recovery_pass(
    ctx: "bsm.StateContext", cell: "tuple[int, int]", confirmed_v: int,
    agreed_v: int, recovery_counters: "dict", min_frames: int,
    add_min_frames: "int | None", effect_gate_active_rows: "frozenset[int] | None",
    effect_gate_persist_sec: float, effect_gate_hard_freeze: bool = False,
) -> bool:
    """`_recovery_or_effect_gate_pass` のラップ (対象12セルのみ記録)。"""
    fired = _ORIG_RECOVERY_PASS(
        ctx, cell, confirmed_v, agreed_v, recovery_counters, min_frames,
        add_min_frames, effect_gate_active_rows, effect_gate_persist_sec,
        effect_gate_hard_freeze,
    )
    if _STATE.ctx_label_map.get(id(ctx)) == "V4" and cell in _TARGET_RC:
        is_gated = (
            effect_gate_active_rows is not None and cell[0] in effect_gate_active_rows
        )
        _STATE.recovery_events.append(RecoveryEvent(
            frame_idx=_STATE.current_frame_idx, t_sec=_STATE.current_t_sec,
            row=cell[0], col=cell[1], is_gated=is_gated,
            hard_freeze=effect_gate_hard_freeze, confirmed_v=confirmed_v,
            agreed_v=agreed_v, fired=fired,
        ))
    return fired


# =============================================================================
# monkeypatch 4/5: burst gate 状態 (仮説(iii)の直接証拠)
# =============================================================================

_ORIG_RESOLVE_BURST: Callable = rp._resolve_burst_gate_state
_ORIG_RESOLVE_EFFECTIVE: Callable = rp._resolve_effective_burst_gate_active
_LAST_SCORE: "list[float | None]" = [None]
_LAST_RAW_OPEN: "list[bool]" = [False]


def _wrapped_resolve_burst_gate_state(
    frame_bgr: "np.ndarray | None", region: object, rows: "frozenset[int]",
    prev_open: bool, prev_opened_at: "float | None", prev_quiet: "float | None",
    time_sec: float, force_close: bool,
    open_threshold: float = rp.BURST_GATE_OPEN_THRESHOLD,
    close_threshold: float = rp.BURST_GATE_CLOSE_THRESHOLD,
) -> "tuple[bool, float | None, float | None]":
    """`_resolve_burst_gate_state` のラップ (生スコア/raw_open を記録)。"""
    result = _ORIG_RESOLVE_BURST(
        frame_bgr, region, rows, prev_open, prev_opened_at, prev_quiet,
        time_sec, force_close, open_threshold, close_threshold,
    )
    if _STATE.current_side == TARGET_SIDE:
        _LAST_SCORE[0] = (
            rp.compute_effect_glow_score(frame_bgr, region, rows)
            if frame_bgr is not None else None
        )
        _LAST_RAW_OPEN[0] = result[0]
    return result


def _wrapped_resolve_effective(
    enable_extension: bool, raw_is_open: bool, force_close: bool,
    last_open_time: float, opponent_chain_active: bool, time_sec: float,
    cooldown_sec: float = rp.BURST_GATE_POST_CLOSE_COOLDOWN_SEC,
    chain_gap_max_sec: float = rp.BURST_GATE_OPPONENT_CHAIN_GAP_MAX_SEC,
) -> bool:
    """`_resolve_effective_burst_gate_active` のラップ (1frame分記録)。"""
    result = _ORIG_RESOLVE_EFFECTIVE(
        enable_extension, raw_is_open, force_close, last_open_time,
        opponent_chain_active, time_sec, cooldown_sec, chain_gap_max_sec,
    )
    if _STATE.current_side == TARGET_SIDE:
        _STATE.gate_frames.append(GateFrame(
            frame_idx=_STATE.current_frame_idx, t_sec=time_sec,
            score=_LAST_SCORE[0], raw_open=raw_is_open, effective_active=result,
        ))
    return result


_ORIG_PIPELINE_RESET: Callable = RecognitionPipeline.reset


def _wrapped_reset_clear_calibration(self: RecognitionPipeline) -> None:
    """実験B: `RecognitionPipeline.reset()` 診断ラップ (src/本体は変更しない)。

    末尾で OnlineHsvCalibrator の較正状態を追加クリアする
    (①凍結ガード+②reset未クリア、の2仮説を同時に無効化する診断パッチ)。
    `self._online_hsv` は再構築せず `reset()` を呼ぶだけに留める (較正器
    自体のAPIをそのまま使い、独自ロジックを再実装しないため)。
    """
    _ORIG_PIPELINE_RESET(self)
    if getattr(self, "_online_hsv", None) is not None:
        self._online_hsv.reset()
    self._online_hsv_injected = False
    self._online_hsv_injected_colors.clear()


def install_probes() -> None:
    """全5 monkeypatch + 実験Bの追加パッチ (EXPERIMENT=="B" 時のみ) をインストールする。"""
    RecognitionPipeline._step_side = _wrapped_step_side
    bsm._filter_transition_new_cnn_for_burst_guard = _wrapped_filter
    bsm._recovery_or_effect_gate_pass = _wrapped_recovery_pass
    rp._resolve_burst_gate_state = _wrapped_resolve_burst_gate_state
    rp._resolve_effective_burst_gate_active = _wrapped_resolve_effective
    if EXPERIMENT == "B":
        RecognitionPipeline.reset = _wrapped_reset_clear_calibration


def uninstall_probes() -> None:
    """monkeypatch を復元する。"""
    RecognitionPipeline._step_side = _ORIG_STEP_SIDE
    bsm._filter_transition_new_cnn_for_burst_guard = _ORIG_FILTER
    bsm._recovery_or_effect_gate_pass = _ORIG_RECOVERY_PASS
    rp._resolve_burst_gate_state = _ORIG_RESOLVE_BURST
    rp._resolve_effective_burst_gate_active = _ORIG_RESOLVE_EFFECTIVE
    RecognitionPipeline.reset = _ORIG_PIPELINE_RESET


# =============================================================================
# pipeline構築
# =============================================================================


def _build_off_pipeline() -> RecognitionPipeline:
    """OFF対照 (バーストガード系フラグ全て既定False)。

    実験A (EXPERIMENT=="A") では `enable_drift_resync_hsv_gate=False` を
    明示指定する (coordinator指示、③ドリフト再同期安全弁永久停止仮説の検証)。
    実験B ("B") ではこの引数は既定True (無指定) のままとし、
    `install_probes` がインストールする `reset()` 較正クリアパッチで
    ①②仮説を検証する。
    """
    kwargs: dict = {}
    if EXPERIMENT == "A":
        kwargs["enable_drift_resync_hsv_gate"] = False
    return RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        **kwargs,
    )


def _build_v4_pipeline() -> RecognitionPipeline:
    """v4確定構成 (scripts/_jobs_yardstick_v4prod_2026-08-05.txt と同一)。

    実験A/Bの分岐は `_build_off_pipeline` と同一方針。
    """
    kwargs: dict = {}
    if EXPERIMENT == "A":
        kwargs["enable_drift_resync_hsv_gate"] = False
    return RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        enable_effect_gate=True, enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=BURST_GATE_OPEN_THRESHOLD_V4,
        enable_hidden_row_burst_guard=True,
        **kwargs,
    )


# =============================================================================
# メイン走行
# =============================================================================


def run() -> None:
    """c22 を範囲限定 (前60秒〜後10秒) で OFF/v4 二重再走行する。"""
    video_path = VIDEO_DIR / f"video_{VIDEO_STEM}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 試合境界補正 (上記コメント参照): 素朴な60秒前は前の試合に踏み込むため、
    # 実際の試合開始 (GAME_START_T_SEC) 直後から開始する。
    # 追加検証2が指定されていれば、前試合の終盤から跨いで開始する。
    start_sec = (
        CROSS_BOUNDARY_START_T_SEC if CROSS_BOUNDARY_START_T_SEC is not None
        else GAME_START_T_SEC + GAME_START_MARGIN_SEC
    )
    start_frame = int(start_sec * fps)
    n_frames = int((ONSET_T_SEC + POST_WINDOW_SEC - start_sec) * fps) + int(fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipeline_off = _build_off_pipeline()
    pipeline_v4 = _build_v4_pipeline()
    pipeline_off.set_video_id(f"video_{VIDEO_STEM}")
    pipeline_v4.set_video_id(f"video_{VIDEO_STEM}")
    _STATE.pipeline_label_map = {id(pipeline_off): "OFF", id(pipeline_v4): "V4"}
    _STATE.ctx_label_map = {
        id(pipeline_v4._sm_1p.context): "V4", id(pipeline_v4._sm_2p.context): "V4",
    }
    # 2026-08-06 追加検証: baseline npz (allframes_ref) は全frame処理・fps正規化
    # 無しで収集されている疑い (ネーミングより)。OFF側の再現失敗がサンプリング
    # 密度差によるものかを切り分けるため、全frame処理を強制する。
    stride = 1 if FORCE_ALL_FRAMES else resolve_normalize_fps_30_stride(fps)
    print(
        f"[{VIDEO_STEM}] start_sec={start_sec:.2f} start_frame={start_frame} "
        f"fps={fps:.2f} stride={stride} n_frames={n_frames}"
    )

    cell_rows: list[dict] = []
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
        res_off = pipeline_off.update(fi, t_sec, frame)
        res_v4 = pipeline_v4.update(fi, t_sec, frame)
        off_board = res_off.p2.confirmed_board if TARGET_SIDE == "2P" else res_off.p1.confirmed_board
        v4_board = res_v4.p2.confirmed_board if TARGET_SIDE == "2P" else res_v4.p1.confirmed_board
        row = {"frame_idx": fi, "t_sec": t_sec, "state_v4": (
            res_v4.p2.state.name if TARGET_SIDE == "2P" else res_v4.p1.state.name
        )}
        for r, c, correct_v in TARGET_CELLS:
            off_v = int(off_board.get(r, c)) if off_board is not None else None
            v4_v = int(v4_board.get(r, c)) if v4_board is not None else None
            row[f"off_r{r}c{c}"] = off_v
            row[f"v4_r{r}c{c}"] = v4_v
        cell_rows.append(row)
    cap.release()
    _write_outputs(cell_rows)
    _report(cell_rows)


# =============================================================================
# 出力
# =============================================================================


def _write_outputs(cell_rows: list[dict]) -> None:
    """cell比較/filter/recovery/gateの4種CSVを出力する。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = VIDEO_STEM
    if cell_rows:
        with (OUT_DIR / f"{stem}_cell_compare.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cell_rows)
    for name, records in (
        ("filter_events", _STATE.filter_events),
        ("recovery_events", _STATE.recovery_events),
        ("gate_frames", _STATE.gate_frames),
    ):
        path = OUT_DIR / f"{stem}_{name}.csv"
        if records:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].__dataclass_fields__))
                writer.writeheader()
                for r in records:
                    writer.writerow(r.__dict__)
        else:
            path.write_text("", encoding="utf-8")
    print(f"[{stem}] CSV出力: 4ファイル ({OUT_DIR})")


# =============================================================================
# レポート
# =============================================================================


def _off_success_check(cell_rows: list[dict]) -> None:
    """OFF対照が最終的に正しく積み上がるか (最後の frame の値) を確認する。"""
    if not cell_rows:
        print("  記録なし")
        return
    last = cell_rows[-1]
    mismatches = [
        (r, c) for r, c, correct in TARGET_CELLS
        if last.get(f"off_r{r}c{c}") != correct
    ]
    print(f"  OFF最終frame ({last['t_sec']:.3f}秒) の12セル誤り: {len(mismatches)}件 {mismatches}")


def _v4_final_check(cell_rows: list[dict]) -> None:
    """v4対照が最終的にどうなっているか (最後の frame の値) を確認する。"""
    if not cell_rows:
        return
    last = cell_rows[-1]
    mismatches = [
        (r, c) for r, c, correct in TARGET_CELLS
        if last.get(f"v4_r{r}c{c}") != correct
    ]
    print(f"  v4最終frame ({last['t_sec']:.3f}秒) の12セル誤り: {len(mismatches)}件 {mismatches}")


def _at_onset_check(cell_rows: list[dict], prefix: str) -> None:
    """onset (f188154, t=3135.9) 瞬間ちょうどの値を確認する (最終frameでの

    確認は後続の正当なクリアと混同するため、2026-08-06調査で判明した罠を
    踏まないよう、ラベル瞬間そのものも別途チェックする)。
    """
    if not cell_rows:
        return
    nearest = min(cell_rows, key=lambda r: abs(r["t_sec"] - ONSET_T_SEC))
    mismatches = [
        (r, c) for r, c, correct in TARGET_CELLS
        if nearest.get(f"{prefix}_r{r}c{c}") != correct
    ]
    print(
        f"  {prefix} onset瞬間 (t={nearest['t_sec']:.3f}秒) の12セル誤り: "
        f"{len(mismatches)}件 {mismatches}"
    )


def _duty_cycle_report() -> None:
    """報告窓内 (onset-60〜+10秒) のデューティ比・最長連続activeを報告する。"""
    recs = _STATE.gate_frames
    if not recs:
        print("  記録なし")
        return
    active_n = sum(1 for r in recs if r.effective_active)
    duty = active_n / len(recs)
    longest = 0.0
    cur_start: "float | None" = None
    prev_t = recs[0].t_sec
    for r in recs:
        if r.effective_active:
            if cur_start is None:
                cur_start = r.t_sec
        else:
            if cur_start is not None:
                longest = max(longest, prev_t - cur_start)
                cur_start = None
        prev_t = r.t_sec
    if cur_start is not None:
        longest = max(longest, prev_t - cur_start)
    print(f"  デューティ比={duty:.1%} ({active_n}/{len(recs)} frame) 最長連続active={longest:.2f}秒")


def _filter_summary() -> None:
    """filter拒否イベントの集計 (from_state別・rejected率) を報告する。"""
    by_state: "dict[str, list[FilterEvent]]" = defaultdict(list)
    for e in _STATE.filter_events:
        by_state[e.from_state].append(e)
    for state, evs in by_state.items():
        n_rejected = sum(1 for e in evs if e.filtered_v == 10 and e.raw_cnn_v != e.baseline_v)
        n_total = len(evs)
        print(f"  from_state={state}: 呼び出し{n_total}件 (対象セル×frame数)")
    n_calls = len({(e.frame_idx,) for e in _STATE.filter_events})
    print(f"  filter呼び出しframe数(対象セル関与): {n_calls}")


def _report(cell_rows: list[dict]) -> None:
    """OFF対照・duty比・filter集計・最終誤りセル数を出力する。"""
    print(f"\n=== {VIDEO_STEM} {TARGET_SIDE} 判定 (onset={ONSET_T_SEC}, EXPERIMENT={EXPERIMENT}) ===")
    print("\n[1] OFF対照確認 (正しく積み上がるはず)")
    _off_success_check(cell_rows)
    _at_onset_check(cell_rows, "off")
    print("\n[2] v4結果確認")
    _v4_final_check(cell_rows)
    _at_onset_check(cell_rows, "v4")
    print("\n[3] burst gate デューティ比・最長連続active")
    _duty_cycle_report()
    print("\n[4] filter拒否イベント集計")
    _filter_summary()
    print(f"\n[5] recovery経路イベント総数 (対象12セル): {len(_STATE.recovery_events)}件")
    fired = [r for r in _STATE.recovery_events if r.fired]
    print(f"    発火(fired=True): {len(fired)}件")


def main() -> None:
    install_probes()
    try:
        run()
    finally:
        uninstall_probes()


if __name__ == "__main__":
    main()
