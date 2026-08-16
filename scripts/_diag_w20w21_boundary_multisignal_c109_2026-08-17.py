"""W20/W21根治 (試合境界マルチシグナル) の実動画A/B検証 (2026-08-17)。

docs/KNOWN_WEAKNESSES.md W21 の実例 (c109 2P game_idx=24 / 43 に複数試合の
得点が混入) の周辺だけを切り出して、旧 (score-reset 単独) vs 新
(--enable-boundary-multisignal) で game_idx の割れ方・異常イベント検出を
比較する。フル動画 (60fps, 3.77時間) は処理コストが大きいため、既知の
混入区間を含む短い窓のみを対象にする (診断スクリプト、本番収集フローには
組み込まない)。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.collect_boards_lean import collect_lean  # noqa: E402

VIDEO = Path("data/frames/video_c109.mp4")
OUT_DIR = Path("data/verify/w20w21_boundary_multisignal_2026-08-17")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# W21実例: game_idx=24 (2P, t=2403.8-2477.6) 周辺、game_idx=43 (2P,
# t=3635.5-3692.7) 周辺。前後にマージンを取って切り出す。
WINDOWS = [
    ("g24", 2370.0, 180.0),
    ("g43", 3590.0, 220.0),
]


def _run_one(label: str, start_sec: float, max_sec: float, multisignal: bool) -> None:
    tag = f"{label}_{'new' if multisignal else 'old'}"
    out_npz = OUT_DIR / f"c109_{tag}.npz"
    t0 = time.time()
    n = collect_lean(
        VIDEO, out_npz,
        start_sec=start_sec, max_sec=max_sec,
        enable_boundary_multisignal=multisignal,
    )
    dt = time.time() - t0
    print(f"[{tag}] snapshots={n} elapsed={dt:.1f}s -> {out_npz}", flush=True)


def main() -> int:
    for label, start_sec, max_sec in WINDOWS:
        for multisignal in (False, True):
            _run_one(label, start_sec, max_sec, multisignal)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
