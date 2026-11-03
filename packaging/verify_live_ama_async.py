"""保存観測によるama専用processの起動・最新要求・結果公開・終了の結合検査。速度試験ではない。"""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main() -> None:
    from src.phase_j.live_ama import LiveAma
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    owner = LiveAma(args.out)
    count, valid = 0, 0
    try:
        path = Path('D:/puyo_analyzer/packaging/build_20261019a/packets/80_7.jsonl.gz')
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                packet = json.loads(line)
                if packet['game'] != 7:
                    continue
                while owner.worker.inputs.qsize() > 10:
                    time.sleep(.01)
                frame = dict(game_idx=7,frame=packet['id'],_ama=packet)
                owner.update(frame,1)
                row = dict(t_sec=packet['t'],match_active=True)
                owner.decorate(row)
                valid += row['ama_plain']['p1'] is not None
                assert owner.failure is None, owner.failure
                count += 1
                if count >= 500:
                    break
        for _ in range(200):
            time.sleep(.05)
            owner.update(frame,1)
            owner.decorate(row)
            valid += row['ama_plain']['p1'] is not None
            assert owner.failure is None, owner.failure
            if valid:
                break
    finally:
        owner.close()
    receipt = dict(inputs=count,valid=valid,alive_after_close=owner.worker.process.is_alive())
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    assert valid and not receipt['alive_after_close'], receipt
    print(receipt,flush=True)


if __name__ == '__main__':
    main()
