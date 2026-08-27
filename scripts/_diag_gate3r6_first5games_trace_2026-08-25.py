"""Gate 3R-6: zenchi 先頭5試合 (t=0〜340) のフレーム単位計装 + timeline dump 生成。

現在のコード + 現在の本番構成 (RECOGNITION_ADOPTED = 掛け算式3フラグON) で
先頭5試合を処理し、以下を記録する。本体コードは無変更 (monkeypatch のみ)。

  1. timeline dump npz (旧 8/22 OFF 構成 seg01 と同一走査器で比較するため)
     -> data/verify/gate3r6_diag_2026-08-25/first5games_on.npz
  2. フレームトレース jsonl (毎フレーム、両side):
     state1/2, score1/2, 掛け算式実読 valid (formula_last_read), active_chain
     -> logs/_diag_gate3r6_first5games_2026-08-25/trace.jsonl

対象1 (STABLE凍結デッドロック) の判定材料:
  「掛け算式が実読できている (valid) のに state が STABLE」の連続時間。
対象2 (W32 is_dead) の判定材料:
  dump の is_dead run を `_diag_gate3r6_isdead_runs_2026-08-25.py` で走査。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
import scripts.visualize_advantage_overlay as vao  # noqa: E402

LOG_DIR = Path("logs/_diag_gate3r6_first5games_2026-08-25")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRACE_PATH = LOG_DIR / "trace.jsonl"
OUT_DIR = Path("data/verify/gate3r6_diag_2026-08-25")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DUMP_PATH = OUT_DIR / "first5games_on.npz"

_state: dict = {"trace_f": None}


def _read_valid(pipe: RecognitionPipeline, side: str) -> bool:
    r = (pipe._formula_last_read_1p if side == "1P"
         else pipe._formula_last_read_2p)
    return bool(getattr(r, "valid", False)) if r is not None else False


def _patched_update(self, frame_idx: int, time_sec: float, frame):
    r = _orig_update(self, frame_idx, time_sec, frame)
    if _state["trace_f"] is not None:
        _state["trace_f"].write(json.dumps({
            "t": round(time_sec, 3),
            "s1": r.p1.state.value, "s2": r.p2.state.value,
            "sc1": (self._score_tracker_1p.last_score
                    if self._score_tracker_1p is not None else None),
            "sc2": (self._score_tracker_2p.last_score
                    if self._score_tracker_2p is not None else None),
            "fv1": _read_valid(self, "1P"), "fv2": _read_valid(self, "2P"),
            "ac1": (self._active_chain_1p is not None),
            "ac2": (self._active_chain_2p is not None),
        }, ensure_ascii=False) + "\n")
    return r


_orig_update = RecognitionPipeline.update


def main() -> None:
    RecognitionPipeline.update = _patched_update

    with TRACE_PATH.open("w", encoding="utf-8") as f:
        _state["trace_f"] = f
        argv_backup = sys.argv[:]
        try:
            sys.argv = [
                "visualize_advantage_overlay.py",
                "--video", "data/frames/video_zenchi_c0BQoMJwwQU.mp4",
                "--start-sec", "0",
                # 実試合1〜5 = game_idx 1〜5 (t=86.9〜416.5、実画面で確認済み:
                # game_idx 0 はまちうけ画面。t=336 の実画面は WIN 4★-0 で
                # 試合5序盤)。余裕を見て 420 まで。
                "--end-sec", "420",
                "--layout", "panel", "--panel-subtitle-h", "0",
                "--no-force-in-match", "--no-render",
                "--dump-timeline", str(DUMP_PATH),
                "--model-dir", "data/verify/retrain_model62_2026-08-21",
                "--warmup-sec", "0",
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
    print(f"[保存] {DUMP_PATH}")


if __name__ == "__main__":
    main()
