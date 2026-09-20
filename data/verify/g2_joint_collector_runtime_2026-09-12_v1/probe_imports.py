"""採録入口の初期importだけを調べる。動画・update・モデルを実行しない。"""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_belief_publication_runtime_2026-09-11_v1'))
import run_v8 as R


def main() -> None:
    names = ('integrated_connection', 'raw_bound', 'collector_connection', 'tail_auth', 'continuous_guard')
    value = {n: None if n not in sys.modules else str(Path(sys.modules[n].__file__).resolve()) for n in names}
    print(json.dumps(value))


if __name__ == '__main__':
    main()
