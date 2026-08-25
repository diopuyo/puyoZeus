"""Gate 3R-6 案A (enable_nonstable_hold_is_dead) の効果測定用 dump 生成。

`scripts/_diag_gate3r6_first5games_trace_2026-08-25.py` と**同一の動画・区間・
フラグ構成** (zenchi 先頭5試合、t=0〜420、本番構成 model62) で timeline dump を
生成する。違いは以下のみ:

  1. トレース jsonl の monkeypatch なし (dump 生成には無関係のログ計装)
  2. 出力先が data/verify/gate3r6_planA_2026-08-25/ (既存 diag は上書き禁止)
  3. --mode on のときだけ --nonstable-hold-is-dead を追加

使い方:
  python scripts/_diag_gate3r6_planA_dump_2026-08-25.py --mode off
    -> first5games_planA_off.npz (フラグOFF。既存 first5games_on.npz との
       配列一致 = 「本実装が既定OFFで bit-identical」の実測証明)
  python scripts/_diag_gate3r6_planA_dump_2026-08-25.py --mode on
    -> first5games_planA_on.npz (フラグON。効果測定用)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402

OUT_DIR = Path("data/verify/gate3r6_planA_2026-08-25")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "on"], required=True)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump_path = OUT_DIR / f"first5games_planA_{a.mode}.npz"

    argv_backup = sys.argv[:]
    try:
        # 既存 diag (first5games_trace) と完全同一の引数列 (dump先のみ変更)。
        sys.argv = [
            "visualize_advantage_overlay.py",
            "--video", "data/frames/video_zenchi_c0BQoMJwwQU.mp4",
            "--start-sec", "0",
            "--end-sec", "420",
            "--layout", "panel", "--panel-subtitle-h", "0",
            "--no-force-in-match", "--no-render",
            "--dump-timeline", str(dump_path),
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
        if a.mode == "on":
            sys.argv.append("--nonstable-hold-is-dead")
        vao.main()
    finally:
        sys.argv = argv_backup

    print(f"[保存] {dump_path}")


if __name__ == "__main__":
    main()
