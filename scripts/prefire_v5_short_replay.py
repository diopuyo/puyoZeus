"""検証の梯子の短区間。既存記録を読み、先頭の実入力だけ別ファイルへ保存する。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from src.exchange_event_record import read_records, encode
from scripts.run_prefire_replay_20260930 import RECORDS, options, model_directory

SOURCE = 'q_7gc4TgFig'
OUT = Path('logs/prefire_prediction/v5_short')
TAIL_SEC = 1.0


def main() -> None:
    """入力・静的特徴を保持した短記録を作って本番の再生経路で確認する。"""
    from scripts.replay_exchange_event_20260926 import replay
    OUT.mkdir(parents=True, exist_ok=True)
    sample = json.loads(Path('logs/prefire_prediction/v5_experiment/samples.json').read_text())[0]
    game, end = sample['game'], sample['t_sec']+TAIL_SEC
    record, frames = OUT/'input.jsonl.gz', 0
    with gzip.open(record, 'wt', encoding='utf-8') as stream:
        for row in read_records(RECORDS/f'{SOURCE}.jsonl.gz'):
            if row['kind'] == 'update':
                if row['args'][3] > end:
                    break
                if row['args'][4] != game:
                    continue
                frames += 1
            if row['kind'] == 'display' and row['game_idx'] != game:
                continue
            if row['kind'] == 'complete':
                break
            stream.write(json.dumps(encode(row))+'\n')
        stream.write(json.dumps(dict(kind='complete', frames=frames))+'\n')
    result = replay(record, OUT, model_directory(options('v5')), True, None, **options('v5', 0.3))
    (OUT/'result.json').write_text(json.dumps(result, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
