"""連鎖中の全フレームで ChainEvent.chain_count がどう動くかを追う (診断専用).

FINAL7/8 で「画面は 3れんさ! なのに 1P のラベルが 1renza のまま」だった原因を
特定するための計装。 既存の遷移ログ (_diag_2p_osc_transitions_*) は **state が
変わったフレームだけ**を記録しているため、 連鎖中の大半のフレームで
chain_event が何を返しているかが見えていなかった。

出力: data/verify/youtube_demo_2026-08-07/_diag_chain_count_stream_t26_2026-08-08.tsv
      (frame, t_sec, side, state, chain_count, trigger_sec, end_sec, mechanism)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402

init_console()

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO = _ROOT / "data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4"
MODEL = _ROOT / "models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt"
OUT_TSV = (
    _ROOT / "data/verify/youtube_demo_2026-08-07"
    / "_diag_chain_count_stream_t26_2026-08-08.tsv"
)
# 1P が 9 連鎖を撃っている区間を含む窓。
T_START: float = 20.0
T_END: float = 35.0


def main() -> int:
    pipeline = RecognitionPipeline.load_default(
        cnn_model_path=MODEL,
        force_in_match=True,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=0.954,
        enable_hidden_row_burst_guard=True,
        enable_match_transition_debounce=True,
        enable_asymmetric_recovery_min_frames=True,
        recovery_add_min_frames=3,
        enable_ojama_entry_gravity_settle_guard=True,
        enable_gravity_settle_reset_on_exit=True,
    )
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lines = ["frame\tt_sec\tside\tstate\tchain_count\ttrigger_sec\tend_sec\tmechanism"]
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)
        if T_START <= t_sec <= T_END:
            for side, sr in (("1P", result.p1), ("2P", result.p2)):
                ev = getattr(sr, "chain_event", None)
                cnt = getattr(ev, "chain_count", None) if ev is not None else None
                trig = getattr(ev, "trigger_sec", None) if ev is not None else None
                end = getattr(ev, "end_sec", None) if ev is not None else None
                mech = getattr(ev, "mechanism", None) if ev is not None else None
                lines.append(
                    f"{fi}\t{t_sec:.3f}\t{side}\t{sr.state.value}\t"
                    f"{cnt if cnt is not None else ''}\t"
                    f"{trig if trig is not None else ''}\t"
                    f"{end if end is not None else ''}\t{mech or ''}"
                )
        fi += 1
    cap.release()
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"出力: {OUT_TSV} ({len(lines) - 1} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
