"""v4.5専用再生。fitとレビューを分離し、通知時刻+実測時間で公開する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import argparse
import json
import time
from pathlib import Path
from scripts.exchange_review_io import jsonl, save, sha256
from scripts.prefire_v44_data import source, notifications, VIDEOS
from src.prefire_post_b3 import PostB3
from src.b3_predict import RowPredictor
from src.b3_train_model import load_model
from src.b3_timing import TimingTable

OUT = Path(_public_dev_text('<work>/exch_review_render/v4_5'))
EVIDENCE = Path('docs/verification/prefire_v45')
ASSETS = OUT.parent/'video_80_g7_v3_1/input'
MODEL_HASH = '6113e42b8d3aadad6251f0749ad09281d7c63a65fc09a4cc659e68694ee2903c'
WAIT_SEC = 20.


def collect(video: int, limit: int | None = None) -> None:
    import puyo_core
    assert sha256(ASSETS/'model.joblib') == MODEL_HASH
    model = RowPredictor(load_model(ASSETS/'model.joblib'))
    table = TimingTable.from_json((ASSETS/'timing_table.json').read_text())
    policy = json.loads(Path('docs/verification/exch_review_v310/policy_locked.json').read_text())
    engine = PostB3(model, table, policy, puyo_core, enabled=True)
    rows = json.loads((OUT.parent/'v4_4/oof_full.json').read_text())
    games = {r['game'] for r in rows if r['video'] == video}
    games = {7} if video == 80 else {31, 32} if video == 52 else games
    count, timings, skipped = 0, [], 0
    OUT.mkdir(parents=True, exist_ok=True)
    from scripts.prefire_v45_origins import collect as collect_origins
    forecasts, cursor = collect_origins(video, OUT, puyo_core, policy), 0
    with (OUT/f'predictions_{video}.jsonl').open('w') as stream:
        for stamp, game, raw in notifications(video):
            if game not in games:
                continue
            delivered = []
            while cursor < len(forecasts) and forecasts[cursor]['input_sec'] <= stamp:
                if forecasts[cursor]['game'] == game:
                    delivered.append(forecasts[cursor])
                cursor += 1
            value = engine.observe(stamp, game, raw, delivered)
            if value is None:
                continue
            value['video'] = video
            stream.write(json.dumps(value)+'\n')
            if 'v_post' in value:
                timings.append(value['low_spec_ms'])
                skipped += value['over_budget']
            count += 1
            if count % 100 == 0:
                stream.flush()
                print('予測盤B3', video, count, '最大ms', max(timings, default=0), flush=True)
            if limit and count >= limit:
                break
    save(OUT/f'predictions_{video}.receipt.json', dict(rows=count, over_budget=skipped,
        coalesced_inputs=engine.coalesced_inputs,
        max_ms=max(timings, default=0), input_sha256=sha256(source(video)/'inputs.jsonl.gz'),
        output_sha256=sha256(OUT/f'predictions_{video}.jsonl'),
        native_sha256=sha256(Path(puyo_core.__file__)), model_sha256=MODEL_HASH))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', type=int, nargs='+', default=list(VIDEOS))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--wait-for')
    args = parser.parse_args()
    if args.wait_for:
        path = OUT/'logs'/f'{args.wait_for}.exit'
        while not path.exists():
            time.sleep(WAIT_SEC)
        assert path.read_text().strip() == '0'
    if set(args.videos) & {80, 52} and not (EVIDENCE/'model.json').exists():
        raise ValueError('レビュー処理前にfitを固定してください')
    for video in args.videos:
        collect(video, args.limit)


if __name__ == '__main__':
    main()
