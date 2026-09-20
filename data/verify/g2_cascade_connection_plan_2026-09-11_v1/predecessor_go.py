"""既に親検収済みv39の保存票を固定し、再走せず次CPU入口の前提にする。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v39'
PINS = {
    'PROBABILISTIC_CPU_FINISH.json': '2f7e77f4ea90d5069d1ce5d01e8275b56a1cdb49ec93e6a8a9d588ba994c9ad9',
    'SIDE_RETIREMENT_SOURCE.json': '0b6bca18ae39487e0de534f5050e268c00fa566cbfa05fd3d7263c159275ebc0',
    'SIDE_OCCURRENCE_RETIREMENT.json': '5f089a4b2815ba2aa2240a5eb05e427ec9e4b4c6a186409a94786c4e82b0de7a',
    'FUSION_RESULT.json': '631eb5b7470593d093a2f0509399104765c77eb74c3a89deb162741398a39d2b',
    'ARCHIVE_LIFETIME_SOURCE.json': 'cbc093bc705ff1b9ce1532e29280a1bc3f4081d85306a76101f7e45d6d8dc836',
    'HIDDEN_PRIOR_SOURCE.json': 'd99d0150faa894bbf0197b85cabddc30be154ab5dfa4f272aff9a436a25aae51',
    'SIDE_ACTUAL_CONNECTION.json': 'de2a70e946e537faa435225fac2c7bc32ee4b9d6f0b392f451e3e1fbde3f1ec0',
}


def verify() -> dict[str, Any]:
    docs = {}
    for name, expected in PINS.items():
        raw = (ROOT / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, 'cascade_predecessor_changed:' + name
        docs[name] = json.loads(raw)
    outer = docs['FUSION_RESULT.json']
    assert outer['exit_code'] == 0 and outer['error'] is None and outer['source_unchanged']
    for name in ('SIDE_RETIREMENT_SOURCE.json', 'ARCHIVE_LIFETIME_SOURCE.json', 'HIDDEN_PRIOR_SOURCE.json'):
        assert docs[name]['original_exit'] == 0 and docs[name]['unchanged']
    assert docs['PROBABILISTIC_CPU_FINISH.json']['live_factory_verified']
    assert docs['PROBABILISTIC_CPU_FINISH.json']['cpu_basis_only_finalization_verified']
    assert docs['SIDE_OCCURRENCE_RETIREMENT.json']['stage'] == 'side_occurrence_retired'
    assert docs['SIDE_ACTUAL_CONNECTION.json']['full_reset_calls'] == 0
    return dict(run=str(ROOT), sha256=PINS, parent_side_cpu_pass=True, quality_gate_clear=False)
