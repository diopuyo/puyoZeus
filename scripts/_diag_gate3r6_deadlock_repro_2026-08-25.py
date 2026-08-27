"""Gate 3R-6 対象1: STABLE 凍結デッドロックの現行コード再現確認 (2026-08-25)。

2026-08-24 の計装 (`scripts/_diag_chain_state_lag_2026-08-24.py`) と同一の
起動条件 (同動画・同窓 t=6600〜6720・warmup 90 秒・production flags) で、
**現在のコード + 現在の本番構成 (掛け算式3フラグ RECOGNITION_ADOPTED 済み)**
を再実行し、当時実測した「検証ゲートが 4.17 秒握りつぶす」現象が
現行でも再現するかを 1 フレーム単位で確定する。

8/24 との差分 (追加計装):
  - `_apply_chain_formula_early_fire` をラップし、read_fire (根治②の実読
    発火経路) を通ったか / simulate 検証で却下されたかを記録する。
  - `_resolve_formula_chain_count` をラップし、却下 (None 返し) を計数する。

本体コード (src/) は一切変更しない。monkeypatch のみ。

出力: logs/_diag_gate3r6_deadlock_repro_2026-08-25/trace_1p.jsonl
      logs/_diag_gate3r6_deadlock_repro_2026-08-25/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chain_detector import VideoChainTracker, count_non_empty  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
import scripts.visualize_advantage_overlay as vao  # noqa: E402

LOG_DIR = Path("logs/_diag_gate3r6_deadlock_repro_2026-08-25")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRACE_PATH = LOG_DIR / "trace_1p.jsonl"
SUMMARY_PATH = LOG_DIR / "summary.json"

WINDOW_LO = 6685.0
WINDOW_HI = 6715.0

# monkeypatch 越しに現フレームの time_sec を共有するための可変箱
_state: dict = {"time_sec": -1.0, "trace_f": None}
# 集計カウンタ (窓外も含めた全区間で数える)
_counters: dict = {
    "early_fire_calls_1p": 0,
    "early_fire_read_fire_1p": 0,
    "early_fire_simulate_reject_1p": 0,
    "early_fire_simulate_pass_1p": 0,
    "early_fire_skipped_active_1p": 0,
}


def _patched_formula(frame, score_ocr, side, last_score, cached_score_val=None):
    """_check_formula_detected をラップしログする (staticmethod なので self なし)。"""
    from src.recognition_pipeline import _SCORE_VAL_NOT_CACHED
    if cached_score_val is None:
        cached_score_val = _SCORE_VAL_NOT_CACHED
    result = _orig_formula(
        frame, score_ocr, side, last_score, cached_score_val,
    )
    t = _state["time_sec"]
    if side == "1P" and WINDOW_LO <= t <= WINDOW_HI and _state["trace_f"] is not None:
        _state["trace_f"].write(json.dumps({
            "t": round(t, 3), "kind": "formula_check",
            "last_score": last_score,
            "formula_detected": bool(result),
        }, ensure_ascii=False) + "\n")
    return result


_orig_formula = RecognitionPipeline._check_formula_detected


def _wrap_tracker(tracker: VideoChainTracker, side: str) -> None:
    """VideoChainTracker.update をインスタンス単位でラップしログする。"""
    orig_update = tracker.update

    def _patched(t_sec: float, board):
        last_stable_count_before = tracker._last_stable_count
        current_count = count_non_empty(board)
        ev = orig_update(t_sec, board)
        if side == "1P" and WINDOW_LO <= t_sec <= WINDOW_HI and _state["trace_f"] is not None:
            _state["trace_f"].write(json.dumps({
                "t": round(t_sec, 3), "kind": "tracker_update",
                "last_stable_count_before": last_stable_count_before,
                "current_count": current_count,
                "drop": (
                    None if last_stable_count_before is None
                    else last_stable_count_before - current_count
                ),
                "event_fired": ev is not None,
                "event_mechanism": (ev.mechanism if ev is not None else None),
                "event_chain_count": (ev.chain_count if ev is not None else None),
            }, ensure_ascii=False) + "\n")
        return ev

    tracker.update = _patched


_orig_load_default = RecognitionPipeline.load_default.__func__


def _patched_load_default(cls, *args, **kwargs):
    pipe = _orig_load_default(cls, *args, **kwargs)
    if pipe._chain_tracker_1p is not None:
        _wrap_tracker(pipe._chain_tracker_1p, "1P")
    if pipe._chain_tracker_2p is not None:
        _wrap_tracker(pipe._chain_tracker_2p, "2P")
    # 起動時の実効フラグを記録 (配線事故の検出: 3フラグが実際に届いたか)
    _counters["effective_read_verify"] = bool(
        getattr(pipe, "_enable_chain_formula_read_verify", False))
    _counters["effective_count_update"] = bool(
        getattr(pipe, "_enable_formula_chain_count_update", False))
    _counters["effective_step_interlude"] = bool(
        getattr(pipe, "_enable_formula_step_interlude", False))
    _counters["effective_simulate_verify"] = bool(
        getattr(pipe, "_enable_chain_formula_simulate_verify", False))
    return pipe


def _patched_early_fire(self, side: str, time_sec: float, prev_confirmed):
    """_apply_chain_formula_early_fire をラップし read_fire/却下を記録する。

    本体を呼ぶ前に read_fire 判定と active スキップ判定を自前で再現して分類し、
    呼び出し後に発火の有無 (before/after の _active_chain_*) を確認する。
    分類ロジックは recognition_pipeline.py:6221-6254 の複製 (読み取り専用)。
    """
    is_1p = (side == "1P")
    active_before = (
        self._active_chain_1p if is_1p else self._active_chain_2p
    ) is not None
    read_res = (
        self._formula_last_read_1p if is_1p else self._formula_last_read_2p
    )
    read_fire = (
        self._enable_chain_formula_read_verify
        and read_res is not None
        and bool(getattr(read_res, "valid", False))
    )
    _orig_early_fire(self, side, time_sec, prev_confirmed)
    active_after = (
        self._active_chain_1p if is_1p else self._active_chain_2p
    ) is not None
    fired = (not active_before) and active_after
    if is_1p:
        _counters["early_fire_calls_1p"] += 1
        if active_before:
            _counters["early_fire_skipped_active_1p"] += 1
        elif read_fire:
            _counters["early_fire_read_fire_1p"] += 1
        elif fired:
            _counters["early_fire_simulate_pass_1p"] += 1
        else:
            _counters["early_fire_simulate_reject_1p"] += 1
    if is_1p and WINDOW_LO <= time_sec <= WINDOW_HI and _state["trace_f"] is not None:
        _state["trace_f"].write(json.dumps({
            "t": round(time_sec, 3), "kind": "early_fire",
            "active_before": active_before,
            "read_res_valid": (
                bool(getattr(read_res, "valid", False))
                if read_res is not None else None
            ),
            "read_fire": read_fire,
            "fired": fired,
        }, ensure_ascii=False) + "\n")


_orig_early_fire = RecognitionPipeline._apply_chain_formula_early_fire


def _patched_update(self, frame_idx: int, time_sec: float, frame):
    _state["time_sec"] = time_sec
    r = _orig_update(self, frame_idx, time_sec, frame)
    if WINDOW_LO <= time_sec <= WINDOW_HI and _state["trace_f"] is not None:
        _state["trace_f"].write(json.dumps({
            "t": round(time_sec, 3), "kind": "state",
            "state1": r.p1.state.value, "state2": r.p2.state.value,
            "score1": (
                self._score_tracker_1p.last_score
                if self._score_tracker_1p is not None else None
            ),
            "score2": (
                self._score_tracker_2p.last_score
                if self._score_tracker_2p is not None else None
            ),
            "active_chain_1p": (self._active_chain_1p is not None),
            "chain_until_1p": round(self._chain_until_1p, 3),
        }, ensure_ascii=False) + "\n")
    return r


_orig_update = RecognitionPipeline.update


def main() -> None:
    RecognitionPipeline._check_formula_detected = staticmethod(_patched_formula)
    RecognitionPipeline.load_default = classmethod(_patched_load_default)
    RecognitionPipeline.update = _patched_update
    RecognitionPipeline._apply_chain_formula_early_fire = _patched_early_fire

    with TRACE_PATH.open("w", encoding="utf-8") as f:
        _state["trace_f"] = f
        argv_backup = sys.argv[:]
        try:
            # 8/24 計装 (_diag_chain_state_lag_2026-08-24.py) と同一の起動条件。
            # production 認識フラグは既定 ON (現在の RECOGNITION_ADOPTED =
            # 掛け算式3フラグを含む) が自動適用される。
            sys.argv = [
                "visualize_advantage_overlay.py",
                "--video", "data/frames/video_zenchi_c0BQoMJwwQU.mp4",
                "--start-sec", "6600",
                "--end-sec", "6720",
                "--layout", "panel", "--panel-subtitle-h", "0",
                "--no-force-in-match", "--no-render",
                "--model-dir", "data/verify/retrain_model62_2026-08-21",
                "--warmup-sec", "90",
                "--kill-override-chain-completion",
                "--enable-slide-exit-min-display-guard",
            ]
            # production_config.advantage_overlay_flags() を単一情報源として追記
            # (feedback_use_single_source_for_flags_2026-08-22 に従う)。
            import src.production_config as pc
            adopted = pc.advantage_overlay_flags()
            if adopted:
                sys.argv.extend(adopted.split())
            # 応手ビームサーチは認識に無関係の後段計算のため高速化目的で無効化
            # (8/24 計装と同条件)。
            sys.argv.append("--no-counter-reach")
            vao.main()
        finally:
            sys.argv = argv_backup
            _state["trace_f"] = None

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(_counters, f, ensure_ascii=False, indent=2)
    print(f"[保存] {TRACE_PATH}")
    print(f"[保存] {SUMMARY_PATH}")
    print(json.dumps(_counters, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
