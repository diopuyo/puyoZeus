"""v3c構成の c19 t=332.5 stale (58セル) 実態確定 (2026-08-05、使い捨て診断)。

v3c構成 (`enable_effect_gate` + `enable_burst_guard_v2` +
`enable_transition_merge_guard` + `burst_gate_open_threshold=0.954` +
`enable_hidden_row_burst_guard` + `enable_burst_close_extension` +
`burst_chain_gap_max_sec=0.0`) で、video_c19 t=332.5 (okラベル盤面、OFF時は
誤り0) の confirmed_board が58セル相当 stale になる実態を、OFF基準との
二重pipeline比較で定量化する。

## 計測方法
OFF pipeline (バーストガード系フラグ全て既定False) と v3c pipeline に
同一フレームを毎frame投入し、両者の confirmed_board を cell 単位で比較する
(OFF側の値を ground truth とみなす、coordinator確認済みの前提
「OFF時は誤り0」に基づく)。これにより「58セル」を独立に再現・検証する。

## 計装方式 (src/ 本番コード変更禁止、読み取りとラップのみ)
    1. `RecognitionPipeline._step_side` — side/time相関 + own_chain_active/
       opponent_chain_active を記録 (v3c側のみ)。
    2. `_resolve_burst_gate_state` — 生視覚スコア + raw is_open (Schmitt trigger
       そのもの) を記録。
    3. `_resolve_effective_burst_gate_active` — 実効ゲート信号の合成過程
       (raw_open/force_close/cooldown中/連鎖延長) を記録。
    4. `_filter_transition_new_cnn_for_burst_guard` — 遷移merge1回あたり
       「何セルがEMPTY起点で許容色以外を拒否されたか」を集計 (58セルの
       直接的な発生源を特定する候補)。
    5. `_hidden_row_trust_gate_ok` — 呼び出しごとの許可/拒否を記録 (補助)。

## 範囲限定 (cold-start対策 + coordinator指定の報告窓)
coordinator指定の報告窓は t=290〜340 (50秒)。cold-start対策として
PRE_WARMUP_SEC=90秒分を前倒しで実行するが、集計・プロット・タイムライン
要約は t∈[290,340] のみに限定する (前回タスクの反省を踏襲)。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._verify_c19_stale_2026-08-05
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
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN  # noqa: E402
from src.fps_normalize import resolve_normalize_fps_30_stride  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# =============================================================================
# 定数
# =============================================================================

VIDEO_DIR: Path = Path("/home/ryouj/frames")
OUT_DIR: Path = Path("data/verify/c19_stale_2026-08-05")

VIDEO_STEM: str = "c19"
REPORT_START_SEC: float = 290.0
REPORT_END_SEC: float = 340.0
PRE_WARMUP_SEC: float = 90.0

TARGET_W: int = 1920
TARGET_H: int = 1080

BURST_GATE_OPEN_THRESHOLD_V3C: float = 0.954
BURST_CHAIN_GAP_MAX_SEC_V3C: float = 0.0

# 332.5周辺タイムライン要約の窓幅。
TIMELINE_HALF_WINDOW_SEC: float = 6.0
LABEL_T_SEC: float = 332.5


@dataclass
class SideFrameRecord:
    """1 side × 1 frame の観測 (v3c pipeline 側のみ記録)。"""

    frame_idx: int
    t_sec: float
    side: str
    state: str
    own_chain_active: bool
    opponent_chain_active: bool
    raw_score: "float | None"
    raw_is_open: bool
    effective_active: bool
    mismatch_count: int


@dataclass
class FilterAggEvent:
    """`_filter_transition_new_cnn_for_burst_guard` 1呼び出し分の集計 (全78セル)。"""

    frame_idx: int
    t_sec: float
    from_state: str
    allowed_colors: str
    n_diff_cells: int
    n_rejected_cells: int
    n_passed_cells: int


@dataclass
class _ProbeState:
    """計装の可変状態。"""

    pipeline_label_map: dict[int, str] = field(default_factory=dict)
    current_side: str = ""
    current_frame_idx: int = -1
    current_t_sec: float = -1.0
    current_own_chain: bool = False
    current_opp_chain: bool = False
    side_records: list[SideFrameRecord] = field(default_factory=list)
    filter_events: list[FilterAggEvent] = field(default_factory=list)
    hidden_row_calls: int = 0
    hidden_row_allowed: int = 0


_STATE = _ProbeState()


# =============================================================================
# monkeypatch 1: RecognitionPipeline._step_side (side/time/chain flag相関)
# =============================================================================

_ORIG_STEP_SIDE: Callable = RecognitionPipeline._step_side


def _wrapped_step_side(self: RecognitionPipeline, side: str, frame_idx: int, *args: object, **kwargs: object):
    """`_step_side` のラップ (v3c pipeline 呼び出し時のみ context を更新)。"""
    label = _STATE.pipeline_label_map.get(id(self))
    if label == "V3C":
        _STATE.current_side = side
        _STATE.current_frame_idx = frame_idx
        _STATE.current_t_sec = float(args[0]) if args else _STATE.current_t_sec
        _STATE.current_own_chain = bool(kwargs.get("own_chain_active", False))
        _STATE.current_opp_chain = bool(kwargs.get("opponent_chain_active", False))
    else:
        _STATE.current_side = ""
    return _ORIG_STEP_SIDE(self, side, frame_idx, *args, **kwargs)


# =============================================================================
# monkeypatch 2: _resolve_burst_gate_state (生スコア + raw open)
# =============================================================================

_ORIG_RESOLVE_BURST: Callable = rp._resolve_burst_gate_state
_LAST_RAW_SCORE: dict[str, "float | None"] = {"1P": None, "2P": None}
_LAST_RAW_OPEN: dict[str, bool] = {"1P": False, "2P": False}


def _wrapped_resolve_burst_gate_state(
    frame_bgr: "np.ndarray | None", region: object, rows: "frozenset[int]",
    prev_open: bool, prev_opened_at: "float | None", prev_quiet: "float | None",
    time_sec: float, force_close: bool,
    open_threshold: float = rp.BURST_GATE_OPEN_THRESHOLD,
    close_threshold: float = rp.BURST_GATE_CLOSE_THRESHOLD,
) -> "tuple[bool, float | None, float | None]":
    """`_resolve_burst_gate_state` のラップ (生スコア/raw_open を side別に記録)。"""
    result = _ORIG_RESOLVE_BURST(
        frame_bgr, region, rows, prev_open, prev_opened_at, prev_quiet,
        time_sec, force_close, open_threshold, close_threshold,
    )
    if _STATE.current_side in ("1P", "2P"):
        score = (
            rp.compute_effect_glow_score(frame_bgr, region, rows)
            if frame_bgr is not None else None
        )
        _LAST_RAW_SCORE[_STATE.current_side] = score
        _LAST_RAW_OPEN[_STATE.current_side] = result[0]
    return result


# =============================================================================
# monkeypatch 3: _resolve_effective_burst_gate_active (実効ゲート信号の合成)
# =============================================================================

_ORIG_RESOLVE_EFFECTIVE: Callable = rp._resolve_effective_burst_gate_active


def _wrapped_resolve_effective(
    enable_extension: bool, raw_is_open: bool, force_close: bool,
    last_open_time: float, opponent_chain_active: bool, time_sec: float,
    cooldown_sec: float = rp.BURST_GATE_POST_CLOSE_COOLDOWN_SEC,
    chain_gap_max_sec: float = rp.BURST_GATE_OPPONENT_CHAIN_GAP_MAX_SEC,
) -> bool:
    """`_resolve_effective_burst_gate_active` のラップ (side別に1frame記録)。"""
    result = _ORIG_RESOLVE_EFFECTIVE(
        enable_extension, raw_is_open, force_close, last_open_time,
        opponent_chain_active, time_sec, cooldown_sec, chain_gap_max_sec,
    )
    side = _STATE.current_side
    if side in ("1P", "2P"):
        _STATE.side_records.append(SideFrameRecord(
            frame_idx=_STATE.current_frame_idx, t_sec=time_sec, side=side,
            state="", own_chain_active=_STATE.current_own_chain,
            opponent_chain_active=opponent_chain_active,
            raw_score=_LAST_RAW_SCORE.get(side), raw_is_open=raw_is_open,
            effective_active=result, mismatch_count=-1,
        ))
    return result


# =============================================================================
# monkeypatch 4: _filter_transition_new_cnn_for_burst_guard (58セルの発生源候補)
# =============================================================================

_ORIG_FILTER: Callable = bsm._filter_transition_new_cnn_for_burst_guard


def _wrapped_filter(
    baseline: "bsm.Board | None", new_cnn: "bsm.Board", from_state: "bsm.BoardState",
) -> "bsm.Board":
    """`_filter_transition_new_cnn_for_burst_guard` のラップ (全78セルの拒否数を集計)。"""
    filtered = _ORIG_FILTER(baseline, new_cnn, from_state)
    if _STATE.current_side not in ("1P", "2P") or baseline is None:
        return filtered
    allowed = bsm._TRANSITION_MERGE_GUARD_SCOPE.get(from_state)
    n_diff = 0
    n_rejected = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if baseline.get(r, c) == new_cnn.get(r, c):
                continue
            n_diff += 1
            if filtered.get(r, c) == COLOR_UNKNOWN:
                n_rejected += 1
    if n_diff > 0:
        _STATE.filter_events.append(FilterAggEvent(
            frame_idx=_STATE.current_frame_idx, t_sec=_STATE.current_t_sec,
            from_state=from_state.name,
            allowed_colors=(sorted(allowed) if allowed is not None else "None"),
            n_diff_cells=n_diff, n_rejected_cells=n_rejected,
            n_passed_cells=n_diff - n_rejected,
        ))
    return filtered


# =============================================================================
# monkeypatch 5: _hidden_row_trust_gate_ok (補助記録)
# =============================================================================

_ORIG_HIDDEN_ROW_GATE: Callable = rp._hidden_row_trust_gate_ok


def _wrapped_hidden_row_gate(
    enable_guard: bool, window_active: bool, last_burst_open_time: float,
    time_sec: float, cooldown_sec: float = rp.HIDDEN_ROW_TRUST_COOLDOWN_SEC,
) -> bool:
    """`_hidden_row_trust_gate_ok` のラップ (呼び出し回数+許可回数のみ集計)。"""
    result = _ORIG_HIDDEN_ROW_GATE(
        enable_guard, window_active, last_burst_open_time, time_sec, cooldown_sec,
    )
    if _STATE.current_side in ("1P", "2P"):
        _STATE.hidden_row_calls += 1
        if result:
            _STATE.hidden_row_allowed += 1
    return result


def install_probes() -> None:
    """全5 monkeypatch をインストールする (src/ 本番ファイルは書き換えない)。"""
    RecognitionPipeline._step_side = _wrapped_step_side
    rp._resolve_burst_gate_state = _wrapped_resolve_burst_gate_state
    rp._resolve_effective_burst_gate_active = _wrapped_resolve_effective
    bsm._filter_transition_new_cnn_for_burst_guard = _wrapped_filter
    rp._hidden_row_trust_gate_ok = _wrapped_hidden_row_gate


def uninstall_probes() -> None:
    """monkeypatch を復元する (他スクリプトへの汚染防止)。"""
    RecognitionPipeline._step_side = _ORIG_STEP_SIDE
    rp._resolve_burst_gate_state = _ORIG_RESOLVE_BURST
    rp._resolve_effective_burst_gate_active = _ORIG_RESOLVE_EFFECTIVE
    bsm._filter_transition_new_cnn_for_burst_guard = _ORIG_FILTER
    rp._hidden_row_trust_gate_ok = _ORIG_HIDDEN_ROW_GATE


# =============================================================================
# pipeline構築
# =============================================================================


def _build_off_pipeline() -> RecognitionPipeline:
    """OFF基準 (バーストガード系フラグ全て既定False、ground truth代理)。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )


def _build_v3c_pipeline() -> RecognitionPipeline:
    """v3c本命構成 (coordinator指定のフラグ組)。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        enable_effect_gate=True, enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=BURST_GATE_OPEN_THRESHOLD_V3C,
        enable_hidden_row_burst_guard=True, enable_burst_close_extension=True,
        burst_chain_gap_max_sec=BURST_CHAIN_GAP_MAX_SEC_V3C,
    )


# =============================================================================
# 盤面比較 (OFF vs v3c、mismatch集計)
# =============================================================================


def _count_mismatch(off_board: "bsm.Board | None", v3c_board: "bsm.Board | None") -> int:
    """OFF確定盤面を ground truth とみなし、v3c確定盤面との不一致セル数を数える。"""
    if off_board is None or v3c_board is None:
        return -1
    n = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            off_v = off_board.get(r, c)
            if off_v == COLOR_UNKNOWN:
                continue
            if v3c_board.get(r, c) != off_v:
                n += 1
    return n


# =============================================================================
# メイン走行
# =============================================================================


def run() -> None:
    """c19 を範囲限定 (PRE_WARMUP込み) で OFF/v3c 二重再走行する。"""
    video_path = VIDEO_DIR / f"video_{VIDEO_STEM}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けません: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_sec = max(0.0, REPORT_START_SEC - PRE_WARMUP_SEC)
    start_frame = int(start_sec * fps)
    n_frames = int((REPORT_END_SEC - start_sec + 2.0) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipeline_off = _build_off_pipeline()
    pipeline_v3c = _build_v3c_pipeline()
    pipeline_off.set_video_id(f"video_{VIDEO_STEM}")
    pipeline_v3c.set_video_id(f"video_{VIDEO_STEM}")
    _STATE.pipeline_label_map = {
        id(pipeline_off): "OFF", id(pipeline_v3c): "V3C",
    }
    stride = resolve_normalize_fps_30_stride(fps)
    print(
        f"[{VIDEO_STEM}] start_sec={start_sec:.2f} start_frame={start_frame} "
        f"fps={fps:.2f} stride={stride} n_frames={n_frames}"
    )

    mismatch_rows: list[dict] = []
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
        res_v3c = pipeline_v3c.update(fi, t_sec, frame)
        m1p = _count_mismatch(res_off.p1.confirmed_board, res_v3c.p1.confirmed_board)
        m2p = _count_mismatch(res_off.p2.confirmed_board, res_v3c.p2.confirmed_board)
        mismatch_rows.append({
            "frame_idx": fi, "t_sec": t_sec,
            "state_1p": res_v3c.p1.state.name, "state_2p": res_v3c.p2.state.name,
            "mismatch_1p": m1p, "mismatch_2p": m2p,
        })
    cap.release()
    _write_outputs(mismatch_rows)
    _report(mismatch_rows)


# =============================================================================
# 出力 + レポート
# =============================================================================


def _write_outputs(mismatch_rows: list[dict]) -> None:
    """mismatch/gate/filterの3種CSVを出力する。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{VIDEO_STEM}_mismatch_timeline.csv").open(
        "w", encoding="utf-8", newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(mismatch_rows[0].keys()) if mismatch_rows else [])
        if mismatch_rows:
            writer.writeheader()
            writer.writerows(mismatch_rows)
    with (OUT_DIR / f"{VIDEO_STEM}_gate_timeline.csv").open(
        "w", encoding="utf-8", newline="",
    ) as f:
        recs = _STATE.side_records
        writer = csv.DictWriter(f, fieldnames=list(recs[0].__dataclass_fields__) if recs else [])
        if recs:
            writer.writeheader()
            for r in recs:
                writer.writerow(r.__dict__)
    with (OUT_DIR / f"{VIDEO_STEM}_filter_events.csv").open(
        "w", encoding="utf-8", newline="",
    ) as f:
        evs = _STATE.filter_events
        writer = csv.DictWriter(f, fieldnames=list(evs[0].__dataclass_fields__) if evs else [])
        if evs:
            writer.writeheader()
            for e in evs:
                writer.writerow(e.__dict__)
    print(f"[{VIDEO_STEM}] CSV出力: 3ファイル ({OUT_DIR})")


def _filter_report_window(rows: list[dict]) -> list[dict]:
    """t∈[REPORT_START_SEC, REPORT_END_SEC] のみ抜き出す。"""
    return [r for r in rows if REPORT_START_SEC <= r["t_sec"] <= REPORT_END_SEC]


def _duty_cycle_report(side: str) -> None:
    """側別のデューティ比・最長連続active時間を報告する。"""
    recs = [
        r for r in _STATE.side_records
        if r.side == side and REPORT_START_SEC <= r.t_sec <= REPORT_END_SEC
    ]
    if not recs:
        print(f"  {side}: 記録なし")
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
    print(
        f"  {side}: デューティ比={duty:.1%} ({active_n}/{len(recs)} frame) "
        f"最長連続active={longest:.2f}秒"
    )


def _stale_profile_report(rows_in_window: list[dict]) -> None:
    """1秒ビンでの mismatch 最大値をプロットし (a)/(b) を判定する。"""
    per_sec: dict[int, list[int]] = defaultdict(list)
    for r in rows_in_window:
        sec = int(r["t_sec"])
        per_sec[sec].append(max(r["mismatch_1p"], r["mismatch_2p"]))
    print("  秒 : mismatch最大値")
    for sec in sorted(per_sec):
        vals = per_sec[sec]
        print(f"  {sec:4d}: {max(vals):3d}  (frame内最大, n={len(vals)})")


def _timeline_summary() -> None:
    """LABEL_T_SEC 周辺のイベント要約を出力する。"""
    lo, hi = LABEL_T_SEC - TIMELINE_HALF_WINDOW_SEC, LABEL_T_SEC + TIMELINE_HALF_WINDOW_SEC
    print(f"\n  --- t={LABEL_T_SEC}±{TIMELINE_HALF_WINDOW_SEC}秒 イベント要約 ---")
    for side in ("1P", "2P"):
        recs = [r for r in _STATE.side_records if r.side == side and lo <= r.t_sec <= hi]
        prev_active = None
        prev_own = None
        prev_opp = None
        for r in recs:
            if (r.effective_active, r.own_chain_active, r.opponent_chain_active) != (
                prev_active, prev_own, prev_opp,
            ):
                print(
                    f"    [{side}] t={r.t_sec:.3f} effective_active={r.effective_active} "
                    f"raw_open={r.raw_is_open} score={r.raw_score} "
                    f"own_chain={r.own_chain_active} opp_chain={r.opponent_chain_active}"
                )
                prev_active, prev_own, prev_opp = (
                    r.effective_active, r.own_chain_active, r.opponent_chain_active,
                )
    for e in _STATE.filter_events:
        if lo <= e.t_sec <= hi:
            print(
                f"    [filter] t={e.t_sec:.3f} from_state={e.from_state} "
                f"allowed={e.allowed_colors} diff={e.n_diff_cells} "
                f"rejected={e.n_rejected_cells} passed={e.n_passed_cells}"
            )


def _report(mismatch_rows: list[dict]) -> None:
    """デューティ比・stale時間プロファイル・タイムライン要約を出力する。"""
    rows_in_window = _filter_report_window(mismatch_rows)
    print(f"\n=== {VIDEO_STEM} 判定 (報告窓 t={REPORT_START_SEC}〜{REPORT_END_SEC}) ===")
    print("\n[1] デューティ比・最長連続active")
    _duty_cycle_report("1P")
    _duty_cycle_report("2P")
    print("\n[2] staleプロファイル (1秒ビン、mismatch最大値)")
    _stale_profile_report(rows_in_window)
    print(
        f"\n[3] filter拒否イベント総数: {len(_STATE.filter_events)}件 "
        f"(報告窓内: {sum(1 for e in _STATE.filter_events if REPORT_START_SEC <= e.t_sec <= REPORT_END_SEC)}件)"
    )
    for e in _STATE.filter_events:
        if e.n_rejected_cells >= 5 and REPORT_START_SEC <= e.t_sec <= REPORT_END_SEC:
            print(
                f"    [大規模拒否] t={e.t_sec:.3f} from_state={e.from_state} "
                f"allowed={e.allowed_colors} diff={e.n_diff_cells} "
                f"rejected={e.n_rejected_cells}"
            )
    _timeline_summary()
    print(
        f"\n[4] hidden_row_trust_gate 呼び出し: {_STATE.hidden_row_calls}件 "
        f"許可: {_STATE.hidden_row_allowed}件"
    )


def main() -> None:
    install_probes()
    try:
        run()
    finally:
        uninstall_probes()


if __name__ == "__main__":
    main()
