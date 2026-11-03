"""H1の開始観測を固定入力だけで監査する。評価結果・合否には触れない。"""
from __future__ import annotations

import json

from scripts.falsedeath_replay_20261001 import OUT, input_record
from scripts.falsedeath_v2_ladder_20261002 import STOP
from src.exchange_event_record import read_records
from src.exchange_history_integrity import MatchHistoryIntegrity


def main() -> None:
    """短区間の開始観測値を保存し、実再生のauditとの照合に使う。"""
    rows = {}
    for source, end in STOP.items():
        guard = MatchHistoryIntegrity(retain_audit=True)
        for item in read_records(input_record('v2d', source)):
            if item['kind'] != 'update':
                continue
            result, _, _, stamp, game, *_ = item['args']
            if stamp > end:
                break
            guard.observe((result.p1, result.p2), game, stamp)
        rows[source] = guard.summary()['games']
    with (OUT/'v2d/COVERAGE.json').open('x', encoding='utf-8') as stream:
        json.dump(rows, stream, ensure_ascii=False, indent=1)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    main()
