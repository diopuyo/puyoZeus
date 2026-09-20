"""既存の全prefix consumerをA22原保存へ再用。元tail未完了拒否は保持する。"""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'g2_split_tail_runtime_2026-09-13_v22'
ORIGINAL = ROOT.parent / 'g2_m1_second_postrun_2026-09-13_v1/probe_prefix_replay.py'


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def lines(path: Path) -> Iterator[dict]:
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def main() -> None:
    sys.path.insert(0, str(RUNTIME))
    import target_entry as T
    support = ModuleType('_async_original_prefix_support')
    support.ROOT, support.T, support.R = ROOT, T, T.A
    support.read, support.lines = read, lines
    previous = sys.modules.get('review_arrivals')
    try:
        sys.modules['review_arrivals'] = support
        spec = importlib.util.spec_from_file_location('_async_original_prefix_replay', ORIGINAL)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
    finally:
        if previous is None:
            sys.modules.pop('review_arrivals', None)
        else:
            sys.modules['review_arrivals'] = previous
    assert original.RUN.name == T.OUTPUT_NAME == 'video38_split_tail_candidate_v22'
    original.main()


if __name__ == '__main__':
    main()
