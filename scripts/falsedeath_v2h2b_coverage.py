"""H1′の受理瞬間の証拠を固定入力から取り直し、最終盤面で上書きされた証拠と区別する。"""
from __future__ import annotations

import json

from scripts.falsedeath_replay_20261001 import OUT, input_record, records
from src.exchange_event_record import read_records
from src.exchange_history_integrity import MatchHistoryIntegrity

DEST = OUT/'v2h2b'


def observe_record(source: str) -> dict:
    """採点に使った全体履歴との同一性を確認し、開始受理の瞬間だけ複製する。"""
    guard = MatchHistoryIntegrity(retain_audit=True, boundary_score_start=True, strict_integrity=True)
    accepted = {}
    for item in read_records(input_record('v2h2b', source)):
        if item['kind'] != 'update':
            continue
        result, _, _, stamp, game, *_ = item['args']
        guard.observe((result.p1, result.p2), game, stamp)
        evidence = guard.evidence()
        if evidence['boundary_score_evidence']['accepted'] and game not in accepted:
            accepted[game] = dict(t_sec=stamp, **evidence)
    path = OUT/'replay/v2h2b'/source/'history_integrity_audit.json'
    original = json.loads(path.read_text())['games']
    assert guard.summary()['games'] == original, source
    return dict(accepted_count=len(accepted), accepted=accepted, full_history_identical=True)


def main() -> None:
    """18記録の受理証拠を新規ファイルへ保存する。"""
    result = {source: observe_record(source) for source in records()}
    with (DEST/'START_ACCEPTANCE.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=1)
    print(json.dumps({s: r['accepted_count'] for s, r in result.items()}))


if __name__ == '__main__':
    main()
