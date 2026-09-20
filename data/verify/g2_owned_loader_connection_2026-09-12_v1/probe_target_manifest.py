"""親が検収後に一度だけ発行する固定区間GO。動画処理は起動しない。"""
from __future__ import annotations
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import target_entry as T

TESTS = {'entry_v2.xml': 8, 'combined_v1.xml': 8, 'source_coverage_v1.xml': 1,
         '../g2_historical_chain_next_reset_2026-09-12_v1/combined_v1.xml': 12,
         '../g2_menu_exit_start_qualification_2026-09-12_v1/qualification_v1.xml': 28,
         '../g2_menu_exit_start_qualification_2026-09-12_v1/join_v2.xml': 7}


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
    terminal = json.loads((T.ROOT / 'seal_parent_v2/RESULT.json').read_bytes())
    T.require(terminal['child_exit'] == terminal['parent_exit'] == 0
              and terminal['original_seal_to_separate_parent'] is True, 'separate_parent_seal')
    constructor = json.loads((T.ROOT / 'constructor_v1/CONSTRUCTOR_CLOSED.json').read_bytes())
    T.require(constructor['actual_constructor'] is True and constructor['new_side_selected'] is True
              and constructor['owned_builds'] == 1 and constructor['updates'] == 0, 'actual_owned_constructor')
    model = json.loads((T.A.START / 'engine_v1/RESULT.json').read_bytes())
    T.require(model['supported'] is True and model['child_exit'] == 0
              and model['actual_video'] is False and model['quality_gate_clear'] is False, 'model_cpu_receipt')
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
        destination = snapshot / path.relative_to(T.ROOT.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(value)
    value = dict(decision='GO_fixed_whole_target_observation', output=str(output), frozen_files=pins,
        quality_gate_clear=False, production_permission=False, known_connection_failures=[],
        stop_conditions=['原例外/clock/原票矛盾は即停止', '原RSS8GiB/残RAM2GiB resource guard',
                         '原source変更を拒否', '未適用M1/欠測/未出現条件を合格へ昇格しない'],
        reviews=['Opus127:過去NEXTの寿命', 'Opus128:MENU離脱の正規追随reset',
                 'Opus131:owned loader設計', 'Opus132:owned接続と環境終了の最終差分'],
        planned_runtime_selection=dict(side=str(T.A.SIDE / 'side_actual_connection.py'),
            capture=str(T.A.START / 'start_capture.py'), worker=str(T.A.START / 'start_worker.py'),
            protected_old_sources='旧版pinは非変更保護集合。実選択/本番採用一覧ではない'),
        test_sha256=tests, source_snapshot=str(snapshot), bounds=[29052, 36298],
        purpose='固定実区間のproducer/M1/原終了観測。全G2品質や本番採用のGOではない')
    with T.GO.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(decision=value['decision'], source_count=len(pins), output=str(output),
                          quality_gate_clear=False)), flush=True)


if __name__ == '__main__':
    main()
