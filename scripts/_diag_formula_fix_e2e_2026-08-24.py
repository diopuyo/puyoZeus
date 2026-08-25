"""STABLE凍結デッドロック根治の E2E 検証 (2026-08-24 コーダ)。

_diag_chain_state_lag_2026-08-24.py と同一の起動条件・同一区間で、
新フラグ ON/OFF の 1 フレームトレースを取る。

  OFF (陽性対照): t=6697.5〜6701.7 の 4.17 秒の CHAIN 突入遅れが再現すること
  ON  (根治確認): 掛け算式検知の連続成立直後 (t≈6697.6) に CHAIN 突入すること

使い方:
  python scripts/_diag_formula_fix_e2e_2026-08-24.py off
  python scripts/_diag_formula_fix_e2e_2026-08-24.py on

出力: logs/_diag_formula_fix_e2e_2026-08-24/trace_{mode}.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
import scripts.visualize_advantage_overlay as vao  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "on"
assert MODE in ("off", "on"), MODE

LOG_DIR = Path("logs/_diag_formula_fix_e2e_2026-08-24")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRACE_PATH = LOG_DIR / f"trace_{MODE}.jsonl"

WINDOW_LO = 6685.0
WINDOW_HI = 6715.0

# 新フラグ (MODE=on のときだけ inject)
NEW_FLAGS = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
)

_state: dict = {"trace_f": None}

_orig_load_default = RecognitionPipeline.load_default.__func__


def _patched_load_default(cls, *args, **kwargs):
    if MODE == "on":
        kwargs.update(NEW_FLAGS)
    return _orig_load_default(cls, *args, **kwargs)


_orig_update = RecognitionPipeline.update


def _patched_update(self, frame_idx: int, time_sec: float, frame):
    r = _orig_update(self, frame_idx, time_sec, frame)
    if WINDOW_LO <= time_sec <= WINDOW_HI and _state["trace_f"] is not None:
        ev = self._active_chain_1p
        fr = self._formula_last_read_1p
        _state["trace_f"].write(json.dumps({
            "t": round(time_sec, 3), "state1": r.p1.state.value,
            "state2": r.p2.state.value,
            "score1": (
                self._score_tracker_1p.last_score
                if self._score_tracker_1p is not None else None
            ),
            "active_1p": None if ev is None else {
                "cc": ev.chain_count, "mech": ev.mechanism,
                "score": ev.total_score,
                "trig": round(ev.trigger_sec, 2),
            },
            "formula_read_1p": None if fr is None else {
                "valid": bool(getattr(fr, "valid", False)),
                "left": getattr(fr, "left", None),
                "right": getattr(fr, "right", None),
            },
            "accum_1p": (
                None if self._formula_accum_1p is None else {
                    "steps": self._formula_accum_1p.step_count,
                    "power": self._formula_accum_1p.total_power,
                }
            ),
        }, ensure_ascii=False) + "\n")
    return r


def main() -> None:
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
            import src.production_config as pc
            adopted = pc.advantage_overlay_flags()
            if adopted:
                sys.argv.extend(adopted.split())
            sys.argv.append("--no-counter-reach")
            vao.main()
        finally:
            sys.argv = argv_backup
            _state["trace_f"] = None
    print(f"[保存] {TRACE_PATH}")


if __name__ == "__main__":
    main()
