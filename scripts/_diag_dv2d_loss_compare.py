"""D4再収集前後の入力を時刻で照合し、診断再生の同一性も点検する。"""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
from typing import Iterator

from scripts._diag_dv2d_loss_inventory import OUT, read, save, resource
from scripts.falsedeath_replay_20261001 import input_record


def updates(path: Path) -> Iterator[dict]:
    """型復元を省いても値比較可能なJSON入力を読む。"""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'update':
                yield row


def compare(source: str, targets: list[dict]) -> dict:
    """評価器への入力が同一ならD4の影響はないと確定できる。"""
    count, differences, by_case, fields, samples = 0, [], {}, Counter(), {}
    left, right = (updates(input_record(mode, source)) for mode in ('prod', 'v2d'))
    for old, new in zip_longest(left, right):
        assert old is not None and new is not None, source
        a, b = old['args']['tuple'], new['args']['tuple']
        assert a[3:5] == b[3:5], (source, a[3:5], b[3:5])
        count += 1
        if a == b:
            continue
        stamp = a[3]
        changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        names = changed_fields(a[0], b[0])
        fields.update(names)
        detail = dict(t_sec=stamp, game=a[4], changed_args=changed, changed_fields=names)
        differences.append(detail)
        for target in targets:
            if target['local_game'] == a[4] and stamp <= target['candidate_observed_death_sec']:
                by_case.setdefault(str(target['game']), []).append(stamp)
                if stamp <= target['first_sec']:
                    samples[str(target['game'])] = dict(**detail, old_result=a[0], new_result=b[0])
    return dict(frames=count, different_frames=len(differences), by_case=by_case,
                fields=fields, samples=samples, differences=differences)


def changed_fields(old: dict, new: dict) -> list[str]:
    """元記録を参照可能なため、全盤面の重複保存はせず変更フィールドを残す。"""
    names = []
    for side in ('p1', 'p2'):
        a, b = old['namespace'][side]['namespace'], new['namespace'][side]['namespace']
        names.extend(f'{side}.{key}' for key in a.keys() | b.keys() if a.get(key) != b.get(key))
    return names


def main() -> None:
    """差分がないことと、差分があるが影響不明であることを分ける。"""
    save('RESOURCE_compare_summary.json', resource())
    targets = read(OUT/'EVENTS.json')
    rows = {}
    for source in dict.fromkeys(r['record'] for r in targets):
        rows[source] = compare(source, [r for r in targets if r['record'] == source])
        print(source, rows[source]['frames'], rows[source]['different_frames'], flush=True)
    save('INPUT_DIFF_SUMMARY.json', rows)


if __name__ == '__main__':
    main()
