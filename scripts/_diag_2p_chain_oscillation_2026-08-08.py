"""2P 連鎖中 chain⇄他状態 振動の原因調査 (読み取り専用・使い捨て計装、2026-08-08)。

FINAL5 デモと同一構成 (--enable-asymmetric-recovery-min-frames
--recovery-add-min-frames 3、cnn-model=cnn_finetune_olRyxDGacbg_demo_v3)
で dio_vs_ts_m01_clip.mp4 の t=45〜75秒を処理し、2P (と 1P) の全 state 遷移を
(frame, 旧状態→新状態, 遷移根拠) で記録する。

src/ は一切変更しない。BoardStateMachine._apply_transition と各 detector
インスタンスの detect() をインスタンス単位でラップして計装する
(detect() の呼び出し回数・副作用は無変更、ラッパーはログ追記のみ)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402
init_console()

from src.board_state_machine import BoardStateMachine  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO = _ROOT / "data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4"
MODEL = _ROOT / "models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt"
T_START = 45.0
T_END = 75.0
OUT_JSONL = _ROOT / "data/verify/youtube_demo_2026-08-07/_diag_2p_osc_transitions_2026-08-08.jsonl"
OUT_DETECT_JSONL = _ROOT / "data/verify/youtube_demo_2026-08-07/_diag_2p_osc_detect_events_2026-08-08.jsonl"

# ============================
# 計装: _apply_transition ラップ (実際に state が切り替わった frame を記録)
# ============================
transitions: list[dict] = []
_orig_apply_transition = BoardStateMachine._apply_transition


def _signal_snapshot(signals: object) -> dict:
    """DetectorSignals から遷移根拠に使うフィールドのみ抜き出す."""
    ev = getattr(signals, "chain_event", None)
    ev_info = None
    if ev is not None:
        ev_info = {
            "trigger_sec": round(float(getattr(ev, "trigger_sec", -1.0)), 3),
            "end_sec": round(float(getattr(ev, "end_sec", -1.0)), 3),
            "chain_count": getattr(ev, "chain_count", None),
            "mechanism": getattr(ev, "mechanism", None),
        }
    cnn_board = getattr(signals, "cnn_board", None)
    return {
        "chain_event": ev_info,
        "score_delta": getattr(signals, "score_delta", None),
        "ojama_top_positive": getattr(signals, "ojama_top_positive", None),
        "slide_motion": getattr(signals, "slide_motion", None),
        "placement_validated": getattr(signals, "placement_validated", None),
        "effect_visible": getattr(signals, "effect_visible", None),
        "chain_max_hold_expired": getattr(signals, "chain_max_hold_expired", None),
        "effect_gate_window_active": getattr(signals, "effect_gate_window_active", None),
        "cnn_puyo_count": int(cnn_board.count_puyos()) if cnn_board is not None else None,
    }


_sm_side_map: dict[int, str] = {}


def _patched_apply_transition(self, new_state, signals):  # noqa: ANN001
    old_state = self._ctx.state
    side = _sm_side_map.get(id(self), "?")
    confirmed = self._ctx.confirmed_board
    entry = {
        "side": side,
        "frame_idx": self._ctx.frame_idx,
        "t_sec": round(float(signals.time_sec), 3),
        "old_state": old_state.value,
        "new_state": new_state.value,
        "confirmed_puyo_count": (
            int(confirmed.count_puyos()) if confirmed is not None else None
        ),
        "signals": _signal_snapshot(signals),
    }
    transitions.append(entry)
    return _orig_apply_transition(self, new_state, signals)


BoardStateMachine._apply_transition = _patched_apply_transition

# ============================
# 計装: 各 detector インスタンスの detect() を instance-level でラップ
# (クラス全体でなくインスタンス属性を上書き → 他インスタンスに影響なし、
#  detect() の呼び出し回数は不変なので TsumoPhaseDetector 等のステートフル
#  内部カウンタは無変更)
# ============================
detect_events: list[dict] = []


def _wrap_detector(det: object, side: str) -> None:
    orig_detect = det.detect
    det_name = type(det).__name__

    def wrapped(ctx, signals):  # noqa: ANN001
        old_state_value = ctx.state.value
        res = orig_detect(ctx, signals)
        if res is not None:
            detect_events.append({
                "side": side,
                "detector": det_name,
                "frame_idx": ctx.frame_idx,
                "t_sec": round(float(signals.time_sec), 3),
                "old_state": old_state_value,
                "proposed_state": res.value,
                "is_real_transition": res.value != old_state_value,
                "signals": _signal_snapshot(signals),
            })
        return res

    det.detect = wrapped


def main() -> int:
    if not VIDEO.exists():
        print(f"[ERROR] video not found: {VIDEO}", file=sys.stderr)
        return 1
    if not MODEL.exists():
        print(f"[ERROR] model not found: {MODEL}", file=sys.stderr)
        return 1

    # FINAL5 と同一構成 (BASEFLAGS): --enable-effect-gate --enable-burst-guard-v2
    # --enable-transition-merge-guard --burst-gate-open-threshold 0.954
    # --enable-hidden-row-burst-guard --enable-match-transition-debounce
    # --enable-asymmetric-recovery-min-frames --recovery-add-min-frames 3
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        cnn_model_path=str(MODEL),
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=0.954,
        enable_hidden_row_burst_guard=True,
        enable_match_transition_debounce=True,
        enable_asymmetric_recovery_min_frames=True,
        recovery_add_min_frames=3,
    )

    # side map + detector instance wrap (計装、src 側は無変更)
    _sm_side_map[id(pipeline._sm_1p)] = "1P"  # noqa: SLF001
    _sm_side_map[id(pipeline._sm_2p)] = "2P"  # noqa: SLF001
    for det in pipeline._sm_1p._detectors:  # noqa: SLF001
        _wrap_detector(det, "1P")
    for det in pipeline._sm_2p._detectors:  # noqa: SLF001
        _wrap_detector(det, "2P")

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"[ERROR] cannot open video: {VIDEO}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_start = int(T_START * fps)
    frame_end = int(T_END * fps)
    print(f"[info] fps={fps:.2f} frame_start={frame_start} frame_end={frame_end}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fi = 0
    n_processed = 0
    while fi <= frame_end:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t_sec = fi / fps
        # FINAL5 と同一サンプリング間隔 (DEFAULT_SAMPLE_INTERVAL=0.033s ≒ 全 frame)
        pipeline.update(fi, t_sec, frame)
        n_processed += 1
        if fi % 300 == 0:
            print(f"  [progress] frame={fi} t={t_sec:.2f}s")
        fi += 1
    cap.release()
    print(f"[done] processed {n_processed} frames (frame 0〜{fi-1}, ログは t={T_START}〜{T_END}秒のみ抜粋)")

    # t=45〜75 範囲のみ出力
    transitions_in_range = [e for e in transitions if T_START <= e["t_sec"] <= T_END]
    detect_events_in_range = [e for e in detect_events if T_START <= e["t_sec"] <= T_END]

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for e in transitions_in_range:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(OUT_DETECT_JSONL, "w", encoding="utf-8") as f:
        for e in detect_events_in_range:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[saved] transitions → {OUT_JSONL} ({len(transitions_in_range)} 件)")
    print(f"[saved] detect events → {OUT_DETECT_JSONL} ({len(detect_events_in_range)} 件)")

    # 2P の遷移を簡易表示 (振動確認用)
    p2_trans = [e for e in transitions_in_range if e["side"] == "2P"]
    print(f"\n[2P transitions] {len(p2_trans)} 件 (t={T_START}〜{T_END}s):")
    for e in p2_trans:
        print(
            f"  frame={e['frame_idx']:6d} t={e['t_sec']:6.2f}s "
            f"{e['old_state']:>14s} -> {e['new_state']:<14s} "
            f"signals={e['signals']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
