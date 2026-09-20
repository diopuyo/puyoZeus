"""旧runの実contextを原coverageで読む。run自体のFAILは変更しない。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v45'
SOURCE = ROOT.parent / 'g2_provisional_context_capture_2026-09-09_v1/observer.py'
FIRST, LAST, STRIDE = 34772, 35410, 2


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(reason for row in rows for side in ('1P', '2P')
                      for reason in row['sides'][side]['hold_reasons'])
    usable = [row['frame_idx'] for row in rows if not row['hold_reasons']
              and all(set(row['sides'][side]['hold_reasons']) <= {'next_missing'}
                      for side in ('1P', '2P'))]
    last = rows[-1]
    return dict(side_hold_counts=dict(reasons), legacy_next_only_frames=usable,
        last_queues={side: {name: last['sides'][side]['before_hold'].get(name)
                          for name in ('next_pair', 'dnext_pair')}
                     for side in ('1P', '2P')}, eligibility_not_proved=True)


def main() -> None:
    paths = [SOURCE, RUN/'provisional_context.jsonl', RUN/'PROVISIONAL_CONTEXT_STATUS.json']
    before = {str(path): sha(path) for path in paths}
    spec = importlib.util.spec_from_file_location('_saved_actual_context', SOURCE)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    observer.guards()
    rows = [json.loads(line) for line in paths[1].read_text().splitlines()]
    assert json.loads(paths[2].read_text()) == dict(closed=True, installed=True, errors=[])
    frames = list(range(FIRST, LAST+STRIDE, STRIDE))
    bounded = observer.coverage(rows, frames)
    rejected = []
    for name, sample, expected in [('full_video_scope', rows, observer.expected()),
                                  ('missing_update', rows[:-1], frames),
                                  ('duplicate_update', rows+[rows[-1]], frames)]:
        try:
            observer.coverage(sample, expected)
        except ValueError as error:
            assert str(error) == 'context_scope_coverage'
            rejected.append(name)
        else:
            raise AssertionError(name)
    after = {str(path): sha(path) for path in paths}
    assert before == after
    result: dict[str, Any] = dict(bounded=bounded, rejected=rejected, guards=after, holds=holds(rows),
        source_unchanged=True, original_run_pass=False, actual_model_evaluated=False,
        quality_gate_clear=False, conclusion='bounded_context_present_full_video_not_proved')
    with (ROOT/'SAVED_CONTEXT_RESULT_v2.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != 'guards'}))


if __name__ == '__main__':
    main()
