"""E33のzenchi悪化行を固定し、二つの除外機構を反実仮想再生で照合する。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e33b')
GAMES = Path('logs/review_zenchi_part3/official_games.json')


def outcome_masks(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """公式試合の最後の1/3という従来の採点行を固定する。"""
    times = data['t_sec']
    selected, correct = np.zeros(len(times), bool), np.zeros(len(times), bool)
    for game in json.loads(GAMES.read_text()):
        mask = (times >= game['start']+(game['end']-game['start'])*2/3) & (times < game['end'])
        sign = 1 if game['winner'] == '1P' else -1
        selected |= mask
        correct |= mask & (data['display_adv']*sign > 0)
    return selected, correct


def prepare() -> dict:
    """純減と悪化・改善の総数を混同せず、結果を見る前に対象行を保存する。"""
    old, new = [np.load(Path(f'logs/{v}/on/zenchi/display.npz')) for v in ('e32', 'e33')]
    assert np.array_equal(old['t_sec'], new['t_sec'])
    selected, before = outcome_masks(old)
    _, after = outcome_masks(new)
    losses, gains = np.flatnonzero(before & ~after), np.flatnonzero(after & ~before)
    groups = np.split(losses, np.flatnonzero(np.diff(losses) != 1)+1)
    intervals = [dict(start=float(old['t_sec'][g[0]]), end=float(old['t_sec'][g[-1]]),
        game=int(old['game_idx'][g[0]]), rows=len(g)) for g in groups if len(g)]
    result = dict(n=int(selected.sum()), before=int(before.sum()), after=int(after.sum()),
        losses=losses.tolist(), gains=gains.tolist(), net_loss=len(losses)-len(gains), intervals=intervals)
    save_json(OUT/'LOSS_ROWS.json', result)
    return result


def attribute() -> dict:
    """終了信号だけ・時間切れだけの2×2再生から、各悪化行を分類する。"""
    fixed = json.loads((OUT/'LOSS_ROWS.json').read_text())
    reference = np.load(Path('logs/e32/on/zenchi/display.npz'))
    hits = {}
    for variant in ('on', 'signal'):
        data = np.load(OUT/variant/'zenchi/display.npz')
        for column in ('t_sec', 'game_idx'):
            assert np.array_equal(data[column], reference[column])
        hits[variant] = outcome_masks(data)[1]
    categories = dict(end_signal=[], timeout=[], either_alone=[], interaction=[])
    for row in fixed['losses']:
        timeout_ok, signal_ok = bool(hits['on'][row]), bool(hits['signal'][row])
        name = ('end_signal' if timeout_ok and not signal_ok else
                'timeout' if signal_ok and not timeout_ok else
                'interaction' if timeout_ok else 'either_alone')
        categories[name].append(row)
    result = dict(net_loss=fixed['net_loss'], gross_losses=len(fixed['losses']),
        gross_gains=len(fixed['gains']), counts={k: len(v) for k,v in categories.items()},
        correct_rows={v: int(h.sum()) for v,h in hits.items()}, rows=categories)
    save_json(OUT/'ATTRIBUTION.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attribute', action='store_true')
    args = parser.parse_args()
    result = attribute() if args.attribute else prepare()
    print(json.dumps(result['counts'] if args.attribute else result, ensure_ascii=False, indent=2))
