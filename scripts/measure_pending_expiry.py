"""答え合わせ pending 連鎖失効 (--verification-pending-chain-expiry) の影子測定。

認識は本番構成のままフラグOFFで再生し、答え合わせの全判定 (verified_match /
verified_mismatch_corrected) を logs/pending_expiry/verif_SOURCE.jsonl へ記録する。
各判定には「ONなら失効していたか (新しい連鎖が pending 作成後に始まったか)」を併記する。
使い方:  run SOURCE  /  analyze
真値 = 書込 0.1〜0.5 秒後の「生CNN==生HSV (不明除く。空も可)」の最頻値。
  空(0)も真値に含める: 幻おじゃまの検出は「生CNN==生HSV==0」が根拠のため。
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

from scripts.measure_c65_guard import (
    FOLLOWUP, FPS_NORM, RECOGNITION_ARGS, SOURCES,
)

OUT = Path('logs/pending_expiry')
TRUTH_FRAME_OFFSETS = range(3, 16)  # 30fps 換算で 0.1〜0.5 秒後
UNKNOWN = 10
MIN_TRUTH_VOTES = 3  # 真値とみなす最低一致票数 (13 フレーム中)
PRODUCTION_EXTRA = ['--next-recolor-pair-guard']  # 本番採用済みガードは ON で測る


def install(log_path: Path) -> None:
    """_update_chain_estimate_verification を包み、全判定を記録する。"""
    from src import recognition_pipeline as rp
    original = rp.RecognitionPipeline._update_chain_estimate_verification
    stream = log_path.open('w')

    def wrapper(self, side, state, cnn_board):  # noqa: ANN001
        pending = (self._chain_verify_pending_1p if side == '1P'
                   else self._chain_verify_pending_2p)
        if pending is not None and state in self.VERIFY_EXPIRY_STATES:
            pending['shadow_expired'] = True  # ON なら失効していた
        result = original(self, side, state, cnn_board)
        if result[0] is None:
            return result
        frame = sys._getframe(1)
        ctx, t = frame.f_locals['ctx'], frame.f_locals['time_sec']
        row = dict(t=round(t, 3), side=side, result=result[0],
                   shadow_expired=bool(pending and pending.get('shadow_expired')),
                   cells=[])
        if result[1] is not None and ctx.confirmed_board is not None:
            before, after = ctx.confirmed_board._grid, result[1]._grid
            for r, c in zip(*(before != after).nonzero()):
                row['cells'].append([int(r), int(c), int(before[r][c]), int(after[r][c])])
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        return result

    rp.RecognitionPipeline._update_chain_estimate_verification = wrapper


def run(source: str) -> None:
    """指定動画の認識を再生 (--no-render)。"""
    spec = SOURCES[source]
    OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = ['visualize_advantage_overlay', '--video', spec['video'],
                '--out', str(OUT / f'{source}_scan.mp4')] + spec['span'] \
        + RECOGNITION_ARGS + PRODUCTION_EXTRA
    import random
    import numpy as np
    import torch
    random.seed(20260926)
    np.random.seed(20260926)
    torch.manual_seed(20260926)
    install(OUT / f'verif_{source}.jsonl')
    from scripts.visualize_advantage_overlay import main
    main()


def load_obs(source: str, keys: set[int]) -> tuple[dict[int, list], int]:
    """必要フレームの観測 (キー=round(t*30)) と行オフセット (13行=0 / 14行=1)。"""
    found: dict[int, list] = {}
    offset = -1
    with gzip.open(FOLLOWUP / f'{source}.observations.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if offset < 0:
                offset = 0 if len(row['sides'][0]['cnn']) == 13 else 1
            key = round(row['t'] * FPS_NORM)
            if key in keys:
                found[key] = row['sides']
    return found, offset


def truth_of(obs: dict, offset: int, key: int, side: str, r: int, c: int) -> int | None:
    """書込セルの真値: 生CNN==生HSV (空も可、不明除く) の最頻値。票数不足は None。"""
    votes: collections.Counter = collections.Counter()
    for d in TRUTH_FRAME_OFFSETS:
        sides = obs.get(key + d)
        if not sides or r - offset < 0:
            continue
        cell = sides[0 if side == '1P' else 1]
        cnn, hsv = cell['cnn'][r - offset][c], cell['hsv'][r - offset][c]
        if cnn == hsv and cnn != UNKNOWN:
            votes[cnn] += 1
    if not votes:
        return None
    value, n = votes.most_common(1)[0]
    return value if n >= MIN_TRUTH_VOTES else None


def analyze_source(source: str) -> dict:
    """1 動画の集計 (母数を必ず併記)。"""
    rows = [json.loads(line) for line in (OUT / f'verif_{source}.jsonl').open()]
    keys = {round(x['t'] * FPS_NORM) + d for x in rows if x['cells']
            for d in TRUTH_FRAME_OFFSETS}
    obs, offset = load_obs(source, keys)
    tot: collections.Counter = collections.Counter()
    cases: list = []
    for x in rows:
        tot['verifications'] += 1
        tot[x['result']] += 1
        tot['shadow_expired_verifications'] += x['shadow_expired']
        tot['expired_and_corrected'] += x['shadow_expired'] and bool(x['cells'])
        key = round(x['t'] * FPS_NORM)
        for r, c, before, written in x['cells']:
            if r < 1:
                continue  # 隠し段は真値が観測できない
            tot['written_cells'] += 1
            truth = truth_of(obs, offset, key, x['side'], r, c)
            if truth is None:
                tot['no_truth'] += 1
                continue
            on = before if x['shadow_expired'] else written  # ON の結果値
            tot['truth_cells'] += 1
            tot['off_right'] += written == truth
            tot['on_right'] += on == truth
            tot['before_right'] += before == truth
            tot['on_degraded'] += (on != truth and written == truth)
            tot['on_improved'] += (on == truth and written != truth)
            if x['shadow_expired']:
                tot['expired_cells_truth'] += 1
                tot['expired_cells_written_right'] += written == truth
                tot['expired_cells_before_right'] += before == truth
                cases.append(dict(t=x['t'], side=x['side'], cell=[r, c],
                                  before=before, written=written, truth=truth))
    return dict(source=source, totals=dict(tot), expired_cells=cases)


def analyze() -> None:
    """完走した動画だけ集計して JSON へ。"""
    results = []
    for s in SOURCES:
        log = OUT / f'run_{s}.log'
        if log.exists() and '[done]' in log.read_text(errors='ignore'):
            results.append(analyze_source(s))
    (OUT / 'measure.json').write_text(json.dumps(results, ensure_ascii=False, indent=1))
    for result in results:
        print(result['source'], result['totals'])


if __name__ == '__main__':
    run(sys.argv[2]) if sys.argv[1] == 'run' else analyze()
