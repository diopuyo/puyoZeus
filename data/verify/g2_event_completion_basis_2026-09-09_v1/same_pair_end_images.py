"""同色NEXTの連続区間末尾を既存無加工抽出器で確認する。"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'g2_normal_chain_settlement_2026-09-09_v1'
FIRST, LAST, STRIDE = 33790, 33808, 2


def main() -> int:
    sys.path.insert(0, str(ORIGINAL))
    import images as original
    started, arguments = time.perf_counter(), list(sys.argv)
    original.ROOT, original.FRAMES = ROOT, tuple(range(FIRST, LAST + STRIDE, STRIDE))
    try:
        sys.argv = [arguments[0], 'same_pair_end_33794_v1']
        status = original.main()
    finally:
        sys.argv[:] = arguments
    print({'elapsed_sec': time.perf_counter() - started, 'new_video': False, 'model_rerun': False})
    return status


if __name__ == '__main__':
    raise SystemExit(main())
