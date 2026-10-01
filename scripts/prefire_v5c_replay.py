"""5Cの実測P95からLを固定し、5Bと同じ再生器で採点する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import prefire_v5b_replay as replay

OUT = Path('logs/prefire_prediction/v5c')


def main() -> None:
    """5件の通知測定が完了するまではLを作らず、再生も開始しない。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('freeze', *replay.BASELINE_DIRS, 'short',
                                        *(f'c{i}' for i in range(1, 7))))
    source = parser.parse_args().source
    replay.OUT = OUT
    result = replay.freeze_latency() if source == 'freeze' else replay.run(source)
    print(json.dumps(result, default=str), flush=True)


if __name__ == '__main__':
    main()
