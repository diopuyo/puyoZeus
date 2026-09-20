"""親が検収後に一度だけ発行する固定区間GO。動画処理は起動しない。"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import target_entry as T

TESTS = {'target_entry_v3.xml': 8, 'review125_clock_v2.xml': 26,
         'review125_after_v1.xml': 27, 'start_failure_trace_v1.xml': 15}


def test_receipts() -> dict:
    result = {}
    for name, count in TESTS.items():
        path = T.ROOT / name
        suite = ET.parse(path).getroot().find('testsuite')
        T.require(suite is not None and int(suite.get('tests')) == count
                  and int(suite.get('failures')) == 0 and int(suite.get('errors')) == 0, 'test_receipt:' + name)
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> None:
    T.require(not T.GO.exists(), 'exclusive_target_go')
    output = T.ROOT.parent / T.OUTPUT_NAME
    T.require(not output.exists(), 'exclusive_target_output')
    tests = test_receipts()
    raw = {p: p.read_bytes() for p in T.runtime_sources()}
    extra = {str(p): hashlib.sha256(value).hexdigest() for p, value in raw.items()}
    with ExitStack() as stack:
        selected = T.A.configured(stack)
        common = selected.__globals__['K']
        T.A.protect(stack, common, extra)
        session_common = selected.__globals__['S'].K
        if session_common is not common:
            T.A.protect(stack, session_common, extra)
        pins = common.guards() | session_common.guards()
    snapshot = T.ROOT / 'target_source_snapshot_v1'
    snapshot.mkdir(exist_ok=False)
    for path, value in raw.items():
        with (snapshot / path.name).open('xb') as stream:
            stream.write(value)
    value = dict(decision='GO_fixed_whole_target_observation', output=str(output), frozen_files=pins,
        quality_gate_clear=False, production_permission=False, known_connection_failures=[],
        stop_conditions=['原例外/clock/原票矛盾は即停止', '原RSS8GiB/残RAM2GiB resource guard',
                         '原source変更を拒否', '未適用M1/欠測/未出現条件を合格へ昇格しない'],
        reviews=['Opus125:原持続と非MENUを修復', 'Opus126:raw finalize/確率層pin/最適化拒否を修復'],
        test_sha256=tests, source_snapshot=str(snapshot), bounds=[29052, 36298],
        purpose='固定実区間のproducer/M1/原終了観測。全G2品質や本番採用のGOではない')
    with T.GO.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(decision=value['decision'], source_count=len(pins), output=str(output),
                          quality_gate_clear=False)), flush=True)


if __name__ == '__main__':
    main()
