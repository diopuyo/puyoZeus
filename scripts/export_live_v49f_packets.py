"""Windows検収用に保存観測だけを持ち出す。予測結果は別の照合用ファイルに置く。"""
from __future__ import annotations
import gzip
import json
from pathlib import Path

from scripts.verify_live_v49f_replay import messages, notifications, REFERENCE
from scripts.prefire_v49e_speed_accounting import rows

OUT = Path('/mnt/d/puyo_analyzer/packaging/build_20261019a/packets')


def export(video: int, game: int) -> None:
    events, cursor, count = messages(video, game), 0, 0
    with gzip.open(OUT/f'{video}_{game}.jsonl.gz', 'xt', encoding='utf-8') as stream:
        for index, (stamp, current, raw) in enumerate(notifications(video)):
            if current != game:
                continue
            delivered = []
            while cursor < len(events) and events[cursor]['t'] <= stamp:
                delivered.append(events[cursor]); cursor += 1
            stream.write(json.dumps(dict(id=index, t=stamp, game=game, raw=raw, origins=delivered))+'\n')
            count += 1
    for prefix in ('base', 'predictions'):
        with (OUT/f'{prefix}_{video}_{game}.jsonl').open('x') as stream:
            for row in rows(REFERENCE/f'{prefix}_{video}.jsonl'):
                if row['game'] == game:
                    stream.write(json.dumps(row)+'\n')
    print('export', video, game, count, flush=True)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    export(80, 7)
    export(51, 4)
