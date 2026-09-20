"""新開始資格から原3seedモデル/保存/closeまでの人工CPU結合。実動画ではない。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT.parent / 'g2_joint_ledger_engine_2026-09-12_v1'
JOINT = ROOT.parent / 'g2_joint_collector_runtime_2026-09-12_v1'
sys.path[:0] = [str(LEDGER), str(JOINT)]
import test_engine as OLD
import test_start_join as FIXTURE
import start_client as CLIENT


def main() -> None:
    output = ROOT / 'start_engine_v3'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    base, identity = OLD.producer()
    producer = FIXTURE.artificial(base)
    request = OLD.observation(producer, identity)
    with CLIENT.Client(output / 'events.jsonl', identity) as client:
        result = client.join(producer, request)
        assert result['supported'] is True and result['integrity_valid'] is True
        assert result['quality_gate_clear'] is False and result['actual_video'] is False
    completion = json.loads((output / 'events.jsonl.complete.json').read_bytes())
    assert completion['closed'] and completion['child_exit_code'] == 0 and completion['accepted_count'] == 1
    receipt = dict(status='ARTIFICIAL_START_ENGINE_VERIFIED', actual_video=False, quality_gate_clear=False,
        seconds=time.perf_counter() - started, supported=result['supported'], child_exit=client.child.returncode,
        frame=result['frame'], start_decision=producer['start_decision'],
        probability_p1=result['details']['calibrated']['probability_p1'])
    with (output / 'RESULT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
