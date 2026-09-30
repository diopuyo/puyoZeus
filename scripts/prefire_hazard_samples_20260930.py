"""発火タイミング (hazard) の学習サンプルを148動画から作る (Phase 2、2026-09-30)。

入力: data/indicators_v2/boards_lean_phase_l_2026-08-11/*.npz (148動画、置くごとの STABLE 盤面・NEXT/NEXT2・得点・発火時刻)。
評価に使う5記録 (q_7gc4TgFig / fcXG83vInDY / mia8KCjr52g / zenchi c0BQoMJwwQU) はこの148本に含まれない
(data/phase_e_dl_index.tsv で確認済み)。

1行 = ある側がある STABLE 盤面にいる時点 t。特徴は時刻 t 以前の観測だけ (自側と相手の最新盤面・NEXT/NEXT2)。
ラベル = その側が次の HORIZON 回の設置以内に、t 時点の最大発火の FIRE_MATCH_FRACTION 以上の送り量で撃ったか。
「撃つ/待つ」は正しく指しても分かれる選択なので学習してよい (user 決定 9/30)。応手のミス率は学習しない。
使い方: PYTHONPATH=. python -m scripts.prefire_hazard_samples_20260930 --shard I --shards N
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import prefire_exchange_sim as sim
from src.scoring import compute_effective_rate

LEAN = Path('/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/indicators_v2/'
            'boards_lean_phase_l_2026-08-11')
OUT = Path('logs/prefire_prediction/hazard_samples')
HORIZON_PLACEMENTS = 4        # ≒ 4 × 0.63 秒 = 2.5 秒 (オラクルの N=3 秒に対応、事前登録)
FIRE_MATCH_FRACTION = 0.5     # 実際の発火が「t 時点の最大発火を撃った」とみなす送り量の下限比
SIDES = ('1P', '2P')


def queue_of(data: dict, i: int) -> tuple[int, ...]:
    """行 i の NEXT/NEXT2 (4色)。"""
    return tuple(int(data[k][i]) for k in ('next1_a', 'next1_b', 'dnext_a', 'dnext_b'))


def game_start(data: dict) -> dict[int, float]:
    """試合ごとの最初の行時刻 (経過秒の起点、評価器の _start と同じく最初の保存時刻)。"""
    starts: dict[int, float] = {}
    for g, t in zip(data['game_idx'], data['t_sec']):
        starts.setdefault(int(g), float(t))
    return starts


def chain_send(data: dict, prev: int, j: int, start: float) -> float | None:
    """行 j の発火の送り量 (得点差をおじゃま換算)。得点が読めなければ None。"""
    before, after = int(data['score'][prev]), int(data['score'][j])
    if before < 0 or after < 0:
        return None
    rate = compute_effective_rate(float(data['chain_trigger_sec'][j]) - start)
    return max(0, after - before) / rate


def label_row(data: dict, rows: np.ndarray, pos: int, best_send: float, start: float) -> tuple[float, int]:
    """(ラベル, 得点欠測なら1)。次の HORIZON 回以内に十分な大きさの発火があれば 1。"""
    i = rows[pos]
    for step in range(1, HORIZON_PLACEMENTS + 1):
        if pos + step >= len(rows):
            break
        j = rows[pos + step]
        if data['game_idx'][j] != data['game_idx'][i]:
            break
        trigger = float(data['chain_trigger_sec'][j])
        if np.isfinite(trigger) and trigger > float(data['t_sec'][i]):
            send = chain_send(data, rows[pos + step - 1], j, start)
            if send is None:
                return np.nan, 1
            if send >= FIRE_MATCH_FRACTION * best_send:
                return 1.0, 0
    return 0.0, 0


def side_options(data: dict) -> list[sim.SideFireOptions]:
    """全行の発火候補 (行ごとに自側の盤面・NEXT/NEXT2 だけから求める)。"""
    return [sim.fire_options(data['grids'][i].astype(np.int8).tobytes(), queue_of(data, i))
            for i in range(len(data['t_sec']))]


def opponent_index(data: dict, rows_opp: np.ndarray, i: int) -> int | None:
    """時刻 t_i 以前で同じ試合の、相手側の最新行。"""
    times = data['t_sec'][rows_opp]
    k = int(np.searchsorted(times, data['t_sec'][i], side='right')) - 1
    if k < 0 or data['game_idx'][rows_opp[k]] != data['game_idx'][i]:
        return None
    return int(rows_opp[k])


def video_samples(path: Path) -> dict[str, np.ndarray]:
    """1動画分の特徴・ラベル・母数を作る。"""
    with np.load(path) as raw:
        data = {k: raw[k] for k in raw.files}
    options, starts = side_options(data), game_start(data)
    rows_by_side = {s: np.flatnonzero(data['side'] == s) for s in SIDES}
    out: dict[str, list] = {k: [] for k in ('features', 'label', 'label_missing', 'row', 'game', 't_sec',
                                            'side', 'best_send', 'forced')}
    for side, rows in rows_by_side.items():
        opp_rows = rows_by_side[SIDES[1 - SIDES.index(side)]]
        for pos, i in enumerate(rows):
            best, j = options[i].best(), opponent_index(data, opp_rows, i)
            if best is None or j is None:
                continue
            start = starts[int(data['game_idx'][i])]
            elapsed = float(data['t_sec'][i]) - start
            best_send = sim.send_ojama(best.score, elapsed)
            label, missing = label_row(data, rows, pos, best_send, start)
            out['features'].append(sim.hazard_features(options[i], options[j], data['grids'][i],
                                                       data['grids'][j], elapsed))
            for key, value in (('label', label), ('label_missing', missing), ('row', i),
                               ('game', int(data['game_idx'][i])), ('t_sec', float(data['t_sec'][i])),
                               ('side', SIDES.index(side)), ('best_send', best_send),
                               ('forced', options[i].forced)):
                out[key].append(value)
    result = {k: np.asarray(v) for k, v in out.items()}
    result['rows_total'] = np.asarray(len(data['t_sec']))
    result['rows_queue_known'] = np.asarray(sum(o.queue_known for o in options))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0, help='先頭 N 本だけ (試運転)')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(LEAN.glob('*.npz'))
    paths = paths[:args.limit] if args.limit else paths
    for path in paths[args.shard::args.shards]:
        dest = OUT / f'video_{path.stem}.npz'
        if dest.exists():
            continue
        samples = video_samples(path)
        np.savez_compressed(dest.with_suffix('.tmp.npz'), **samples)
        dest.with_suffix('.tmp.npz').rename(dest)
        print(json.dumps(dict(video=path.stem, rows=int(samples['rows_total']),
                              samples=len(samples['label']))), flush=True)


if __name__ == '__main__':
    main()
