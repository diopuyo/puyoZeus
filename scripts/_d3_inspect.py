"""D3合図の関係を確認する読み取り専用補助。"""
from __future__ import annotations
import json
import os
from collections import Counter
from scripts._d3_candidates import load
from scripts._d3_inventory import OUT, SOURCES


def main() -> None:
    """式の出現が連鎖途中通知か初段かを確認する。"""
    os.nice(19)
    for source in SOURCES:
        rows = load(source)
        counts: Counter = Counter()
        data = json.loads((OUT/f'{source}_candidates.json').read_text())
        for side, group in enumerate(data):
            for event in group['signals']:
                if event['type'] != 'formula':
                    continue
                chain = rows[event['i']]['sides'][side]['chain_event']
                counts[(event['during_chain'], str(chain))] += 1
            print(source, side, Counter((e['type'],e['during_chain']) for e in group['signals']))
        print(list(counts.items())[:12])


if __name__ == '__main__':
    main()
