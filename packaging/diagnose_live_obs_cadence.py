"""実動画の通知間隔と台帳未確定を再現する。時刻・得点を補完しない。"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PUYO_B3_M3'] = '0'
from src.b3_score_ledger import SCORE_PERSIST_SEC, ScoreLedger
from src.phase_j.live_ama_inputs import AmaInputs


def diagnose(path: Path) -> dict:
    groups, identity, owner, previous = {}, None, None, None
    for line in path.read_text(encoding='utf-8').splitlines():
        packet = json.loads(line)
        key = (packet['generation'], packet['game'])
        if key != identity:
            identity, previous = key, None
            owner = AmaInputs.__new__(AmaInputs)
            owner.ledger = ScoreLedger(newest_first_offset=True, enable_inflight=False, observed_landing_rule=False)
        value = groups.setdefault(str(key), dict(rows=0, gaps_within_window=0, min_gap_sec=None,
                                                first_sec=packet['t'], first_scores=[r['score'] for r in packet['raw']],
                                                valid_pending=0, reasons=Counter()))
        if previous is not None:
            gap = packet['t']-previous
            value['gaps_within_window'] += gap <= SCORE_PERSIST_SEC
            value['min_gap_sec'] = min(value['min_gap_sec'] or gap, gap)
        previous = packet['t']
        book = owner.book(packet)
        value['rows'] += 1
        value['valid_pending'] += book is not None and all(p is not None for p in book['pending'])
        value['reasons'][str(owner.missing_reason)] += 1
    return dict(input=str(path), score_persist_sec=SCORE_PERSIST_SEC, groups=groups,
                scope='実際のama配送原票のみ。新規得点観測・時刻の補間なし。')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = diagnose(args.input)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
