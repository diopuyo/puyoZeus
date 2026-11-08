"""正式版の同一入力・全70列と、アダプタの非撃ち合い経路を検査する。"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase_j import live_ex_engine as E

GOLDEN = json.loads(Path(__file__).with_name('live_ex_golden.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case', GOLDEN['rows'], ids=lambda c: c['mode'])
def test_same_input_matches_frozen_extractor(case: dict) -> None:
    result = E.extra_from(case['input'])
    assert tuple(result) == E.COLUMNS == tuple(GOLDEN['columns'])
    for column in E.COLUMNS:
        expected, actual = case['expected'][column], result[column]
        assert actual == expected or (math.isnan(actual) and math.isnan(expected)), column


def test_outside_exchange_does_not_search(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = object.__new__(E.LiveExEngine)
    wrapper.monitor = E.EventMonitor()
    wrapper.engine = SimpleNamespace(probe=lambda: SimpleNamespace(fight=False, key=(1, 2, (None, None))))
    monkeypatch.setattr(E, 'compute', lambda _: pytest.fail('撃ち合い外の探索'))
    result = wrapper.calculate()
    assert result['result'] is None and not result['fight']
    assert len(result['extra']) == len(E.COLUMNS)


def test_frozen_dependency_receipt() -> None:
    root = Path(E.__file__).parent / '_live_ex_frozen'
    manifest = json.loads((root / 'SOURCE.json').read_text(encoding='utf-8'))
    assert manifest['commit'] == '5c02afed204ffca2d7aee5d54008c91754178fab'
    assert manifest['retr_source_commit'] == '0a7a7f63'   # retR v2の受領元 (SHA256SUMS照合済み)
    assert len(manifest['files']) == 119
    for entry in manifest['files']:
        data = (root / entry['source']).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(data).hexdigest() == entry['adapter_sha256'], entry['source']


def test_frozen_all_clear_credit_with_empty_scores() -> None:
    # 全消しの分が残り、得点の列と時刻がまだ空の発火 (216 本の版の段 1 で IndexError、学習側 5e63861b と同じ直し)
    from src.phase_j._live_ex_frozen.src.prefire_context_v410 import chain_events
    from src.phase_j._live_ex_frozen.src.prefire_world_v410 import GENERATE
    from src.phase_j._live_ex_frozen.src.exchange_count_ledger import FORMULA_SCORE_DELAY_SEC
    events, carry, end = chain_events(1, 'Y', 1., [], (0,)*78, 0, 0., 0.5, credit=30, times=[])
    generated = [e for e in events if e.kind == GENERATE]
    assert [e.amount for e in generated] == [30]
    assert generated[0].t == pytest.approx(1. + FORMULA_SCORE_DELAY_SEC)
    assert carry == 0 and end >= 1.
