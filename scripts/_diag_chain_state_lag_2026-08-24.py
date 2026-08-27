"""連鎖開始 検知の実態計装 (2026-08-24)。

「画面で連鎖しているのに state が STABLE と誤認される」現象の機構を
1フレーム単位で確定するための計装スクリプト。本体コード
(src/recognition_pipeline.py 等) は一切変更せず、monkeypatch でラップして
内部値をログするのみ (読み込み専用の診断)。

対象: data/frames/video_zenchi_c0BQoMJwwQU.mp4 (30先2セット動画)
      seg08 (t=6104.6〜) と全く同じ起動条件 (--warmup-sec 30、production
      flags) で t=6104.6〜6720 を再実行し、以下を1フレームずつ記録する:
        - _check_formula_detected の入出力 (score_val is None か / ink_ratio /
          判定結果) — 機能D (掛け算式検知) が実際に反応したか
        - VideoChainTracker.update の内部カウント
          (last_stable_count vs current_count、drop_detected) — 機能①
          (puyo数減少検知) が反応したか。この tracker には毎フレーム
          `_prev_confirmed_1p` (= confirmed_board のコピー) が渡される
          (recognition_pipeline.py:4487-4494)。
        - state1/state2 の遷移 (PipelineResult.p1.state 等)
        - score1/score2 の生値

出力: logs/_diag_chain_state_lag_2026-08-24/trace_1p.jsonl (1行1frame、
      6685<=t<=6715 に限定して記録、それ以外は集計のみ)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.chain_detector import VideoChainTracker, count_non_empty  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
import scripts.visualize_advantage_overlay as vao  # noqa: E402

LOG_DIR = Path("logs/_diag_chain_state_lag_2026-08-24")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRACE_PATH = LOG_DIR / "trace_1p.jsonl"

WINDOW_LO = 6685.0
WINDOW_HI = 6715.0

# monkeypatch 越しに現フレームの time_sec を共有するための可変箱
_state: dict = {"time_sec": -1.0, "trace_f": None}


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
            "cached_score_val": (
                None if cached_score_val is _SCORE_VAL_NOT_CACHED
                else cached_score_val
            ),
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
    return pipe


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

    with TRACE_PATH.open("w", encoding="utf-8") as f:
        _state["trace_f"] = f
        argv_backup = sys.argv[:]
        try:
            sys.argv = [
                "visualize_advantage_overlay.py",
                "--video", "data/frames/video_zenchi_c0BQoMJwwQU.mp4",
                "--start-sec", "6600",
                "--end-sec", "6720",
                "--layout", "panel", "--panel-subtitle-h", "0",
                "--no-force-in-match", "--no-render",
                "--model-dir", "data/verify/retrain_model62_2026-08-21",
                "--warmup-sec", "90",
                "--resolved-exchange-eval", "--resolved-decisive-amplify",
                "--resolved-live-defender",
                "--kill-override-chain-completion",
                "--enable-slide-exit-min-display-guard",
            ]
            # production_config.advantage_overlay_flags() を単一情報源として追記
            # (feedback_use_single_source_for_flags_2026-08-22 に従う)。
            import src.production_config as pc
            adopted = pc.advantage_overlay_flags()
            if adopted:
                sys.argv.extend(adopted.split())
            # 高速化: 応手ビームサーチ (counter-reach) は RecognitionPipeline
            # (認識本体) には一切フィードバックしない有利不利側の後段計算
            # (別モジュール) であり、本計装が見るのは state1/score1/chain_event
            # のみのため無効化して高速化する (認識結果には無関係、
            # --no-counter-reach は $ADOPTED の後に置いて確実に上書きする)。
            sys.argv.append("--no-counter-reach")
            vao.main()
        finally:
            sys.argv = argv_backup
            _state["trace_f"] = None

    print(f"[保存] {TRACE_PATH}")


if __name__ == "__main__":
    main()
