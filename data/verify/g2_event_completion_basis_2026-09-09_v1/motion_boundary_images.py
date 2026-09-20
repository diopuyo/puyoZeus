"""固定2frame刻みの移動境界と同色NEXT候補を無加工抽出する。"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'g2_normal_chain_settlement_2026-09-09_v1'
WINDOWS = {
    'normal_motion_gap_v1': (34890, 34894, 34932, 34936),
    'same_pair_motion_v1': (35492, 35494, 35496, 35498, 35500, 35502, 35504, 35506, 35508),
    'chain_motion_gap_v1': (35788,),
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
            if status:
                return status
    finally:
        sys.argv[:] = arguments
    print({'elapsed_sec': time.perf_counter() - started, 'new_video': False, 'model_rerun': False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
