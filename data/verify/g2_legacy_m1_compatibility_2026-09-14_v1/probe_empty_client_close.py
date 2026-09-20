"""旧M1未要求の実子が正常closeし、終了票を保存することだけ検収。"""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent/'g2_start_client_session_contract_2026-09-12_v1'),
    str(ROOT.parent/'g2_joint_collector_runtime_2026-09-12_v1')]
import start_client as C


def main() -> None:
    path = ROOT / 'empty_client_v1.jsonl'
    client = C.Client(path, ('cpu-empty-source', 'cpu-empty-run', 'cpu-empty-attempt'))
    child = client.child.pid
    client.close()
    completion = json.loads(path.with_suffix('.jsonl.complete.json').read_bytes())
    assert client.closed and client.child.poll() == 0 and client.seq == 0
    assert completion['accepted_count'] == 0 and completion['child_exit_code'] == 0
    result = dict(real_start_worker=True, child_pid=child, child_exit=client.child.returncode,
        accepted_count=client.seq, closed=client.closed, completion=completion,
        no_actual_video_or_model_evaluation=True, quality_gate_clear=False)
    with (ROOT/'EMPTY_CLIENT_CLOSE_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
