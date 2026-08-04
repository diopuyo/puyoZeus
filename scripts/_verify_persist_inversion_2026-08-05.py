"""案B (enable_effect_visual_gate) 誤り焼き付き根因検証 (2026-08-05、使い捨て診断スクリプト)。

`data/verify/effect_gate_2026-08-04_c/diag_zero_effect.csv` の
`gate_should_have_fired` (4条件すべて成立していたのに焼き付いた) 6件のうち
video_c12 (5セル同時) と video_c5 (1セル) を対象に、onset 周辺のみを
RecognitionPipeline で再走行し、1 frame 単位で「誤値が確定した正確な瞬間」の
`effect_gate_window_active` の状態を突き止める。

## 検証する2仮説
    - 仮説A「persist逆転」: window_active=True が継続したまま、誤読が
      persist_sec (既定0.4秒) 以上持続して `_update_effect_gate_hold` 経由で
      正規に (しかし誤って) confirm された。
    - 仮説B「窓のフリッカ」: window_active が1frameでも False に落ちた瞬間、
      `_apply_stable_recovery_gate` はその cell を通常の frame カウント経路
      (`stable_recovery_counters`, 閾値 STABLE_RECOVERY_MIN_FRAMES=8frame≈
      0.27秒@30fps) に切り替える。この経路は persist_sec (0.4秒) より緩い
      ため、窓が数frame閉じるだけで「保護されていない経路」から即確定する。

## 計装方式 (src/ 本番コード変更禁止・読み取りとラップのみ)
    1. `src.board_state_machine.BoardStateMachine.update` をラップし、
       対象 side の対象 cell について毎 frame
       (cnn_v, hsv_v, confirmed_v, signals.effect_gate_window_active,
       ctx.effect_gate_hold[cell], ctx.stable_recovery_counters[cell]) を
       記録する。`signals.effect_gate_window_active` が対象 cell の
       `is_gated` 判定と等価であること (row∈EFFECT_GATE_TOP_ROWS={1,2,3} は
       対象セル全てで真) を利用し、「その frame でゲート経路/通常経路の
       どちらを通ったか」を直接判定する。
    2. `src.recognition_pipeline._compute_effect_gate_window_active` を
       ラップし、4条件の入力 (time_window_active/own_chain_active/
       all_clear_pending) と最終判定を毎 frame 記録する (視覚グロー比率は
       表示用に `scripts._diag_c_zero_effect_2026-08-04._max_bright_ratio_in_rows`
       を再利用して独立計算、閾値判定ロジック自体は再実装しない)。

## 範囲限定
onset_t_sec の前 PRE_WINDOW_SEC 秒〜後 POST_WINDOW_SEC 秒のみを
`--start-sec` 相当 (cv2.VideoCapture のフレーム seek) で処理する
(フル動画再走行は禁止)。この範囲限定のため HSV オンライン校正・
VideoChainTracker 等は数十秒分のウォームアップしか持たない
(限界として stdout に明記する)。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._verify_persist_inversion_2026-08-05
"""
from __future__ import annotations

import csv
import importlib
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
from src.board_state_machine import BoardState, DetectorSignals, EFFECT_GATE_TOP_ROWS  # noqa: E402
from src.fps_normalize import resolve_normalize_fps_30_stride  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# 既存診断スクリプトの視覚グロー再計算ヘルパーを再利用 (コピペ禁止指示)。
_diag_c = importlib.import_module("scripts._diag_c_zero_effect_2026-08-04")

# =============================================================================
# 定数
# =============================================================================

VIDEO_DIR: Path = Path("/home/ryouj/frames")
OUT_DIR: Path = Path("data/verify/persist_inversion_2026-08-05")

# onset 周辺の範囲限定 (フル動画再走行禁止)。
PRE_WINDOW_SEC: float = 30.0
POST_WINDOW_SEC: float = 5.0

# 「直前の推移」表示用の履歴長 (persist_sec=0.4秒より少し長く取る)。
HISTORY_DISPLAY_SEC: float = 0.6

TARGET_W: int = 1920
TARGET_H: int = 1080


@dataclass(frozen=True)
class TargetCell:
    """診断対象の1セル (diag_zero_effect.csv の該当行から転記)。"""

    row: int
    col: int
    wrong_value: int
    correct_value: int


@dataclass(frozen=True)
class VerifyCase:
    """1動画分の検証ケース (diag_zero_effect.csv の gate_should_have_fired 行群)。"""

    video_stem: str
    side: str
    game_idx: int
    onset_t_sec: float
    label_t_sec: float
    cells: tuple[TargetCell, ...]


# diag_zero_effect.csv より転記 (video,side,game_idx,row,col,wrong,correct,onset_t_sec)。
CASES: tuple[VerifyCase, ...] = (
    VerifyCase(
        video_stem="c12", side="2P", game_idx=19,
        onset_t_sec=1498.2330322265625, label_t_sec=1498.2330322265625,
        cells=(
            TargetCell(1, 0, 9, 4), TargetCell(1, 1, 5, 0), TargetCell(1, 2, 5, 0),
            TargetCell(1, 4, 9, 4), TargetCell(2, 2, 9, 4),
        ),
    ),
    VerifyCase(
        video_stem="c5", side="2P", game_idx=13,
        onset_t_sec=1011.3670043945312, label_t_sec=1014.5670166015625,
        cells=(TargetCell(3, 4, 9, 0),),
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
    state: str
    row: int
    col: int
    cnn_v: "int | None"
    hsv_v: "int | None"
    confirmed_v: "int | None"
    window_active: bool
    hold_color: "int | None"
    hold_since_sec: "float | None"
    counter: "int | None"


@dataclass
class GateFrameRecord:
    """`_compute_effect_gate_window_active` 呼び出し1回分 (対象 side のみ記録)。"""

    frame_idx: int
    t_sec: float
    time_window_active: bool
    own_chain_active: bool
    all_clear_pending: bool
    bright_ratio_max: "float | None"
    window_active_final: bool


@dataclass
class _ProbeState:
    """計装の可変状態 (case ごとにリセットする、パイプライン外部の記録先)。"""

    target_side: str = ""
    target_cells: tuple[TargetCell, ...] = ()
    current_frame_idx: int = -1
    current_t_sec: float = -1.0
    sm_side_map: dict[int, str] = field(default_factory=dict)
    cell_records: list[CellFrameRecord] = field(default_factory=list)
    gate_records: list[GateFrameRecord] = field(default_factory=list)


_STATE = _ProbeState()


# =============================================================================
# monkeypatch: BoardStateMachine.update ラップ (cell 単位の生値+ゲート状態記録)
# =============================================================================

_ORIG_SM_UPDATE: Callable = bsm.BoardStateMachine.update


def _record_cell_frame(
    ctx: "bsm.StateContext", signals: DetectorSignals, cell: TargetCell,
) -> None:
    """1 cell 分の frame 観測を `_STATE.cell_records` に追記する。"""
    r, c = cell.row, cell.col
    cnn_v = int(signals.cnn_board.get(r, c)) if signals.cnn_board is not None else None
    hsv_v = int(signals.hsv_board.get(r, c)) if signals.hsv_board is not None else None
    confirmed_v = (
        int(ctx.confirmed_board.get(r, c)) if ctx.confirmed_board is not None else None
    )
    hold = ctx.effect_gate_hold.get((r, c))
    counter = ctx.stable_recovery_counters.get((r, c))
    _STATE.cell_records.append(CellFrameRecord(
        frame_idx=_STATE.current_frame_idx, t_sec=signals.time_sec,
        state=ctx.state.name, row=r, col=c, cnn_v=cnn_v, hsv_v=hsv_v,
        confirmed_v=confirmed_v, window_active=bool(signals.effect_gate_window_active),
        hold_color=(hold[0] if hold is not None else None),
        hold_since_sec=(hold[1] if hold is not None else None),
        counter=counter,
    ))


def _wrapped_sm_update(
    self: "bsm.BoardStateMachine", frame_idx: int, signals: DetectorSignals,
) -> "bsm.StateContext":
    """`BoardStateMachine.update` のラップ (対象 side のみ全 target cell を記録)。"""
    ctx = _ORIG_SM_UPDATE(self, frame_idx, signals)
    side = _STATE.sm_side_map.get(id(self))
    if side == _STATE.target_side:
        for cell in _STATE.target_cells:
            _record_cell_frame(ctx, signals, cell)
    return ctx


# =============================================================================
# monkeypatch: _compute_effect_gate_window_active ラップ (4条件の入力を記録)
# =============================================================================

_ORIG_COMPUTE_GATE: Callable = rp._compute_effect_gate_window_active


def _region_side(region: object) -> str:
    """region オブジェクトの identity から side を判定する。"""
    return "1P" if region is DEFAULT_P1_REGION else "2P"


def _wrapped_compute_gate(
    *, time_window_active: bool, own_chain_active: bool, all_clear_pending: bool,
    frame_bgr: "np.ndarray | None", region: object, enable_visual_gate: bool,
) -> bool:
    """`_compute_effect_gate_window_active` のラップ (4条件の入力を記録、判定ロジックは再実装しない)。"""
    result = _ORIG_COMPUTE_GATE(
        time_window_active=time_window_active, own_chain_active=own_chain_active,
        all_clear_pending=all_clear_pending, frame_bgr=frame_bgr, region=region,
        enable_visual_gate=enable_visual_gate,
    )
    if _region_side(region) == _STATE.target_side:
        ratio = None
        if frame_bgr is not None:
            ratio = _diag_c._max_bright_ratio_in_rows(frame_bgr, region, EFFECT_GATE_TOP_ROWS)
        _STATE.gate_records.append(GateFrameRecord(
            frame_idx=_STATE.current_frame_idx, t_sec=_STATE.current_t_sec,
            time_window_active=time_window_active, own_chain_active=own_chain_active,
            all_clear_pending=all_clear_pending, bright_ratio_max=ratio,
            window_active_final=result,
        ))
    return result


def install_probes() -> None:
    """両 monkeypatch をインストールする (src/ 本番ファイルは書き換えない)。"""
    bsm.BoardStateMachine.update = _wrapped_sm_update
    rp._compute_effect_gate_window_active = _wrapped_compute_gate


def uninstall_probes() -> None:
    """monkeypatch を復元する (他スクリプトへの汚染防止)。"""
    bsm.BoardStateMachine.update = _ORIG_SM_UPDATE
    rp._compute_effect_gate_window_active = _ORIG_COMPUTE_GATE


# =============================================================================
# 1ケース分の範囲限定再走行
# =============================================================================


def _build_pipeline() -> RecognitionPipeline:
    """collect_boards_lean.collect_lean と同一構成の RecognitionPipeline を構築する。"""
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        enable_effect_gate=True,
        effect_gate_persist_sec=None,
        enable_effect_visual_gate=True,
    )


def run_case(case: VerifyCase) -> None:
    """1ケース分を onset 周辺だけ再走行し、cell/gate の全 frame 記録を残す。"""
    _STATE.target_side = case.side
    _STATE.target_cells = case.cells
    _STATE.cell_records = []
    _STATE.gate_records = []

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
    _STATE.sm_side_map = {
        id(pipeline._sm_1p): "1P",
        id(pipeline._sm_2p): "2P",
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


def _write_case_csv(case: VerifyCase) -> None:
    """cell 記録 + gate 記録をそれぞれ CSV に出力する。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cell_path = OUT_DIR / f"{case.video_stem}_cell_timeline.csv"
    with cell_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CellFrameRecord.__dataclass_fields__))
        writer.writeheader()
        for r in _STATE.cell_records:
            writer.writerow(r.__dict__)
    gate_path = OUT_DIR / f"{case.video_stem}_gate_timeline.csv"
    with gate_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(GateFrameRecord.__dataclass_fields__))
        writer.writeheader()
        for r in _STATE.gate_records:
            writer.writerow(r.__dict__)
    print(f"[{case.video_stem}] CSV出力: {cell_path.name} / {gate_path.name}")


# =============================================================================
# 仮説判定レポート
# =============================================================================


def _find_confirm_record(
    records: list[CellFrameRecord], wrong_value: int,
) -> "CellFrameRecord | None":
    """観測窓内で confirmed_v が初めて wrong_value になった記録を返す。"""
    for rec in records:
        if rec.confirmed_v == wrong_value:
            return rec
    return None


def _recent_gate_history(
    gate_records: list[GateFrameRecord], t_sec: float, span_sec: float,
) -> list[GateFrameRecord]:
    """confirm 直前 span_sec 秒分の gate 記録を時刻昇順で返す。"""
    return [r for r in gate_records if t_sec - span_sec <= r.t_sec <= t_sec]


def _report_cell(case: VerifyCase, cell: TargetCell) -> None:
    """1 cell の判定を stdout に出力する (仮説A/B/第三の経路)。"""
    recs = [r for r in _STATE.cell_records if (r.row, r.col) == (cell.row, cell.col)]
    confirm = _find_confirm_record(recs, cell.wrong_value)
    header = f"  --- {case.video_stem} 2P row{cell.row}col{cell.col} (wrong={cell.wrong_value}) ---"
    print(header)
    if confirm is None:
        print("    [判定不能] 観測窓内で wrong_value に確定した frame が見つからない")
        return
    print(
        f"    確定frame: frame_idx={confirm.frame_idx} t_sec={confirm.t_sec:.3f} "
        f"state={confirm.state} window_active={confirm.window_active} "
        f"cnn={confirm.cnn_v} hsv={confirm.hsv_v} "
        f"hold=({confirm.hold_color},{confirm.hold_since_sec}) counter={confirm.counter}"
    )
    recent = [r for r in recs if confirm.t_sec - HISTORY_DISPLAY_SEC <= r.t_sec <= confirm.t_sec]
    flips = sum(
        1 for i in range(1, len(recent))
        if recent[i].window_active != recent[i - 1].window_active
    )
    print(
        f"    直前{HISTORY_DISPLAY_SEC}秒の window_active 推移: "
        f"{[r.window_active for r in recent]} (反転回数={flips})"
    )
    gate_recent = _recent_gate_history(_STATE.gate_records, confirm.t_sec, HISTORY_DISPLAY_SEC)
    print(f"    直前{HISTORY_DISPLAY_SEC}秒の bright_ratio_max: "
          f"{[round(r.bright_ratio_max, 3) if r.bright_ratio_max is not None else None for r in gate_recent]}")
    if confirm.window_active:
        print("    [仮説A] 確定frameで window_active=True → persist経路 (effect_gate_hold) で確定")
    elif flips == 0:
        print("    [第三の経路] 確定frameで window_active=False だが直前も一貫してFalse "
              "(相手連鎖/全消しラッチ条件で最初から窓が開いていない可能性)")
    else:
        print("    [仮説B] 確定frameで window_active=False (直前にフリッカあり) → 通常frameカウント経路で確定")


def _report_case(case: VerifyCase) -> None:
    """1ケース全 cell の判定レポートを出力する。"""
    print(f"\n=== {case.video_stem} ({case.side} game_idx={case.game_idx}) 判定 ===")
    for cell in case.cells:
        _report_cell(case, cell)


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
