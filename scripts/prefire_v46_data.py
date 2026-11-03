"""未来なし予測を先に保存し、神託を別ファイル・別関数へ隔離する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import argparse
import json
from pathlib import Path
from scripts.exchange_review_io import save, sha256
from scripts.prefire_v44_data import source, notifications, VIDEOS
from scripts.prefire_v45_data import ASSETS, MODEL_HASH
from scripts.prefire_v46_origins import collect as collect_origins
from src.prefire_post_v46 import PostV46
from src.b3_predict import RowPredictor
from src.b3_train_model import load_model
from src.b3_timing import TimingTable

OUT = Path(_public_dev_text('<work>/exch_review_render/v4_6'))
EVIDENCE = Path('docs/verification/prefire_v46')
PROGRESS_ROWS = 500


def engine_for() -> PostV46:
    import puyo_core
    assert sha256(ASSETS/'model.joblib') == MODEL_HASH
    model = RowPredictor(load_model(ASSETS/'model.joblib'))
    table = TimingTable.from_json((ASSETS/'timing_table.json').read_text())
    policy = json.loads(Path('docs/verification/exch_review_v310/policy_locked.json').read_text())
    pre = json.loads(Path('docs/verification/prefire_v44/model.json').read_text())['models']['exchange_mixture']
    return PostV46(model, table, policy, puyo_core, enabled=True, pre_model=pre)


def collect(video: int, limit: int | None = None) -> None:
    from scripts.prefire_v46_oracle import AuditOnly
    engine = engine_for()
    rows = json.loads((OUT.parent/'v4_4/oof_full.json').read_text())
    games = {r['game'] for r in rows if r['video'] == video}
    games = {7} if video == 80 else {31,32} if video == 52 else games
    OUT.mkdir(parents=True, exist_ok=True)
    forecasts = collect_origins(video, OUT, engine.native, engine.policy)
    cursor, count, over, times = 0, 0, 0, []
    audit = AuditOnly(video)
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
            audit.observe(engine, value, stamp, game, raw)
            if value is None:
                continue
            value['video'] = video
            stream.write(json.dumps(value)+'\n')
            if 'v_post' in value:
                times.append(value['low_spec_ms'])
                over += value['over_budget']
            count += 1
            if count % PROGRESS_ROWS == 0:
                stream.flush()
                print('v46',video,count,'最大ms',max(times,default=0),'超過',over,flush=True)
            if limit and count >= limit:
                break
    audit.finish(engine)
    engine.pool.shutdown()
    save(OUT/f'predictions_{video}.receipt.json',dict(rows=count,over_budget=over,
        coalesced_inputs=engine.coalesced_inputs,max_ms=max(times,default=0),
        input_sha256=sha256(source(video)/'inputs.jsonl.gz'),
        output_sha256=sha256(OUT/f'predictions_{video}.jsonl'),
        native_sha256=sha256(Path(engine.native.__file__)),model_sha256=MODEL_HASH))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos',type=int,nargs='+',default=list(VIDEOS))
    parser.add_argument('--limit',type=int)
    args = parser.parse_args()
    if set(args.videos)&{80,52}:
        assert (EVIDENCE/'fit_lock.json').exists(), 'レビュー前にfit固定が必要'
    for video in args.videos:
        collect(video,args.limit)


if __name__ == '__main__':
    main()
