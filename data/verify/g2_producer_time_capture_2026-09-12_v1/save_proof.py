"""既存実cold fixtureと検収経路を排他保存し、原ソース不変を確認する。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import pytest
import test_capture as T

ROOT = Path(__file__).resolve().parent


def sources() -> dict[str, str]:
    paths = [ROOT / n for n in ('capture.py', 'test_capture.py', 'save_proof.py')]
    paths += [Path(T.U.__file__), Path(T.U.U.__file__), T.U.U.SNAPSHOT / 'scripts/collect_boards_lean.py',
              T.U.U.REPO / 'src/event_accounting_observer_v1.py', Path(T.L.__file__), T.L.SOURCE]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main() -> None:
    before, start = sources(), time.monotonic()
    real = T.real.__wrapped__()
    outputs = []
    for first in (0, T.T.FIRST):
        output = ROOT / ('actual_' + str(first) + '_v1')
        output.mkdir()
        with pytest.MonkeyPatch.context() as patch:
            T.test_original_updates(real, output, patch, first)
        outputs.append(str(output / 'CAPTURE.json'))
    after = sources()
    assert before == after
    result = {'actual_original_updates': 4, 'source_unchanged': True, 'sources': before,
              'captures': outputs, 'seconds': time.monotonic() - start,
              'artificial_pipeline': True, 'actual_video': False, 'game_anchor_qualified': False,
              'quality_gate_clear': False, 'restored_and_saved_verified': True}
    with (ROOT / 'SAVED_PROOF_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print('原採録4更新・2開始点・実設定・保存/復元/原ソース不変: PASS限定')


if __name__ == '__main__':
    main()
