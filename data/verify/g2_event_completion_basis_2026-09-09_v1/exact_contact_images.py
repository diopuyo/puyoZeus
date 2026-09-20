"""復帰候補34924と直前接地境界だけを既存原画像抽出器で追加取得する。"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'g2_normal_chain_settlement_2026-09-09_v1'
FRAMES = (34912, 34916, 34920, 34924, 34928)


def main() -> int:
    sys.path.insert(0, str(ORIGINAL))
    import images as original
    started, arguments = time.perf_counter(), list(sys.argv)
    original.ROOT, original.FRAMES = ROOT, FRAMES
    try:
        sys.argv = [arguments[0], 'exact_contact_v1']
        status = original.main()
    finally:
        sys.argv[:] = arguments
    print({'elapsed_sec': time.perf_counter() - started, 'new_video': False, 'model_rerun': False})
    return status


if __name__ == '__main__':
    raise SystemExit(main())
