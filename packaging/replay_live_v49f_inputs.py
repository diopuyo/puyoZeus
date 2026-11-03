"""保存した実入力を順番どおり再生し、旧採用世代と修正後の入力世代を比較する。"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPLY_TIMEOUT_SEC = 120.


def replay(args: argparse.Namespace) -> None:
    from src.phase_j.live_v49f_async import AsyncV49F
    args.out.mkdir(exist_ok=False)
    assets = (args.bundle/'app/models/v49f/b3_v6.joblib',
              args.bundle/'app/models/b3/timing_table.json',
              args.bundle/'app/models/v49f/policy.json', args.bundle/'app/models/v49f/pre_model.json')
    worker = AsyncV49F(assets, args.out)
    count, failure = 0, None
    try:
        with args.inputs.open(encoding='utf-8') as stream, (args.out/'results.jsonl').open('w', encoding='utf-8') as out:
            for line in stream:
                packet = json.loads(line)
                if args.legacy:
                    packet.pop('input_epoch', None)
                elif 'input_epoch' not in packet:
                    raise ValueError('入力世代の実記録が必要。省略して推定しない')
                worker.submit(packet)
                if not worker.connection.poll(REPLY_TIMEOUT_SEC):
                    raise TimeoutError('v4.9f replay reply')
                result = worker.connection.recv()
                out.write(json.dumps(result, ensure_ascii=False)+'\n')
                count += 1
                if result.get('fatal'):
                    failure = dict(packet=packet, result=result, index=count-1)
                    break
    finally:
        worker.close()
    receipt = dict(inputs=str(args.inputs), processed=count, legacy=args.legacy, failure=failure)
    (args.out/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(processed=count, fatal=(failure or {}).get('result', {}).get('fatal'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('inputs', 'bundle', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--legacy', action='store_true')
    replay(parser.parse_args())
