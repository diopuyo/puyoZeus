"""既存5記録の表示フレームを固定勝敗と観測死亡に照合し、採点とは別表に残す。"""
from __future__ import annotations

import json
import fcntl
import numpy as np

from scripts.falsedeath_replay_20261001 import OUT
from scripts.falsedeath_v2d_receipt import SOURCES
from scripts.eval_set2_regression_20261001 import fixed_labels

DEST = OUT/'v2d'
CERTAIN = ('unavoidable_death', 'confirmed_death')
EVEN_PROBABILITY = .5


def scan(source: str, mode: str, labels: list[dict], terminals: list[dict]) -> dict:
    """勝敗不明の表示は正解へ算入せず、元のゲーム番号ごとに分離する。"""
    rows = []
    with np.load(OUT/'replay'/mode/source/'display.npz') as data:
        for game in np.unique(data['game_idx']):
            winners = {r['winner'] for r in labels if r['source']==source
                       and r['game_idx']==game and r['winner']}
            observed = {r['side'] for r in terminals if r['game']==game}
            if not winners and len(observed) == 1:
                winners = {'2P' if next(iter(observed)) == '1P' else '1P'}
            mask = (data['game_idx']==game) & np.isin(data['source'], CERTAIN)
            winner = next(iter(winners)) if len(winners) == 1 else None
            wrong = ((data['display_p1'] < EVEN_PROBABILITY) if winner == '1P'
                     else (data['display_p1'] > EVEN_PROBABILITY)) & mask
            stamps = data['t_sec'][wrong] if winner is not None else np.array([])
            rows.append(dict(game=int(game), winners=sorted(winners),
                certain_frames=int(mask.sum()), false_frames=len(stamps),
                unknown_frames=int(mask.sum()) if winner is None else 0,
                first_false_sec=float(stamps[0]) if len(stamps) else None))
    return dict(rows=rows, **{k:sum(r[k] for r in rows)
                for k in ('certain_frames','false_frames','unknown_frames')})


def main() -> None:
    """既存の門とRESULTは変更せず、フレーム数だけを追加監査する。"""
    with (DEST/'DISPLAY_AUDIT.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (DEST/'DISPLAY_AUDIT.json').exists():
            return
        labels = fixed_labels()
        terminals = json.loads((OUT/'TERMINALS.json').read_text())
        value = {mode:{source:scan(source,mode,labels,terminals['prod' if mode=='prod' else 'all'][source])
                       for source in SOURCES} for mode in ('prod','v2d')}
        with (DEST/'DISPLAY_AUDIT.json').open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=1)


if __name__ == '__main__':
    main()
