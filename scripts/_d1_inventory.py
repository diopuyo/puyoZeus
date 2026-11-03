"""D1保存記録の読取専用診断。"""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from typing import Any

OUT = Path('logs/d1')
SOURCES = ('q_7gc4TgFig', 'fcXG83vInDY', 'mia8KCjr52g', 'review')


def decode(value: Any) -> Any:
    """本番コードをロードせず保存型を展開する。"""
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    for key in ('board', 'array', 'namespace', 'tuple'):
        if key in value:
            return decode(value[key])
    return {k: decode(v) for k, v in value.items()}


def read(source: str) -> list[dict]:
    """保存済み更新列だけを読む。"""
    path = Path('logs/e31/records') / f'{source}.jsonl.gz'
    with gzip.open(path, 'rt') as stream:
        return [decode(json.loads(line)) for line in stream if '"kind":"update"' in line]


def main() -> None:
    """全発火の保存起点と確定盤面の変化を抽出する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        rows = read(source)
        fires, changes, previous, stable = {}, [], {}, {}
        for row in rows:
            result, _, _, stamp, game, *_ = row['args']
            for side in ('p1', 'p2'):
                value = result[side]
                key = (game, side)
                board = value['confirmed_board']
                if value['state']['state'] == 'STABLE' and board is not None:
                    stable.setdefault(key, []).append((stamp, board))
                if previous.get(key) != board:
                    changes.append(dict(game=game, side=side, t=stamp, **value))
                    previous[key] = board
                event = value['chain_event']
                if event:
                    firekey = (game, side, event['trigger_sec'])
                    if firekey not in fires:
                        fires[firekey] = dict(game=game, side=side, t=stamp, **value)
                        origin = event.get('before_board')
                        candidates = [x for x in stable.get(key, []) if x[0] < event['trigger_sec']]
                        fires[firekey]['origin'] = origin if origin is not None else candidates[-1][1] if candidates else None
        (OUT/f'{source}_inventory.json').write_text(json.dumps(dict(fires=list(fires.values()), changes=changes)), encoding='utf-8')
        print(source, len(rows), len(fires), len(changes), flush=True)


if __name__ == '__main__':
    main()
