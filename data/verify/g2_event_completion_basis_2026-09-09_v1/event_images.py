"""実NEXT進行と偽slideの元映像を既存無加工抽出器で確認する。"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'g2_normal_chain_settlement_2026-09-09_v1'
WINDOWS = {
    'normal_motion_v1': (34876, 34884, 34888, 34892, 34896, 34898, 34902, 34906,
                         34910, 34914, 34918, 34922, 34926, 34930, 34934, 34938, 34940, 34944, 34948),
    'chain_exit_motion_v1': (35772, 35776, 35780, 35784, 35786, 35790, 35794, 35798, 35802),
}


def main() -> int:
    sys.path.insert(0, str(ORIGINAL))
    import images as original
    started, arguments = time.perf_counter(), list(sys.argv)
    original.ROOT = ROOT
    try:
        for name, frames in WINDOWS.items():
            original.FRAMES = frames
            sys.argv = [arguments[0], name]
            status = original.main()
            if status != 0:
                return status
    finally:
        sys.argv[:] = arguments
    print({'elapsed_sec': time.perf_counter() - started, 'new_video': False, 'model_rerun': False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
