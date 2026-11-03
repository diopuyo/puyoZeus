"""ed原票だけから試作v2の係数を推定する。本番からは呼ばない。"""
from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scripts.exchange_review_io import jsonl, pickles, fires, save
from src.b3_search import points_to_ojama
from scripts.exchange_review_labels import corrected_winners

EXCLUDED_VIDEO = 80
MIN_VIDEO, MAX_VERIFIED_VIDEO = 29, 94
MATCH_TOLERANCE = 1.0
CONFIDENCE = .5
ROOTS = [Path('/mnt/d/puyo_analyzer/share/tasks/done/ei_ed_parallel_after_bo5/artifacts'),
         Path('/mnt/d/puyo_analyzer/share/exchange_runs')]


def winners(run: Path) -> dict:
    """旧版再現専用（勝敗ラベル不具合）。新規採点はcorrected_winnersを使う。"""
    found: dict = {}
    for row in jsonl(run / 'death.jsonl'):
        if not row.get('active') and len(row.get('dead', [])) == 1:
            found.setdefault(row['game'], set()).add('2P' if row['dead'][0] == '1P' else '1P')
    return {g: next(iter(v)) for g, v in found.items() if len(v) == 1 and g > 0}


def observed_send(chains: list[dict], side: str, stamp: float, elapsed: float) -> float | None:
    found = [c for c in chains if c['side'] == side
             and abs(c['trigger_sec'] - stamp) <= MATCH_TOLERANCE
             and c.get('score_ready_sec') is not None and c.get('score_delta') is not None]
    if not found:
        return None
    chain = min(found, key=lambda c: abs(c['trigger_sec'] - stamp))
    return float(points_to_ojama(chain['score_delta'], elapsed))


def samples(run: Path, idx: int) -> list[dict]:
    rows = list(jsonl(run / 'out/spool/exchange_projection_shadow.jsonl'))
    ends = {r['event_id']: r for r in rows if r.get('kind') == 'outcome'}
    chains: dict = {}
    for packet in pickles(run / 'out/supervised-exchanges.pickle'):
        chains.setdefault(packet['game_idx'], []).extend(packet['chains'])
    wins, result = corrected_winners(run), []
    for event in fires(rows):
        end, game = ends.get(event['event_id']), event['game_idx']
        if end is None or game not in wins or not event.get('exchange_input'):
            continue
        elapsed, response = event.get('elapsed_sec', 0), end.get('response')
        attack = observed_send(chains.get(game, []), event['attacker'], event['trigger_sec'], elapsed)
        returned = (observed_send(chains.get(game, []), event['receiver'], response['trigger_sec'],
                    elapsed + response['trigger_sec'] - event['trigger_sec']) if response else 0.)
        if attack is None or returned is None:
            continue
        land = [r for r in rows if r.get('event') == 'landing' and r.get('game_idx') == game
                and event['t_sec'] <= r['t_sec'] <= end['t_sec']]
        net_land = sum((1 if r['receiver'] == event['attacker'] else -1)
                       * max(0, r['ojama_after'] - r['ojama_before']) for r in land)
        result.append(dict(video=idx, game=game, event=event['event_id'], actual_return=returned,
            actual_attack=attack, balance=returned - attack, landing_balance=net_land,
            win=int(wins[game] == event['receiver'])))
    return result


def fit(output: Path) -> dict:
    rows, used = [], set()
    for root in ROOTS:
        for run in sorted(root.glob('ed_video_*')):
            idx = int(run.name.split('_')[-1])
            if idx in used or idx == EXCLUDED_VIDEO or not MIN_VIDEO <= idx <= MAX_VERIFIED_VIDEO:
                continue
            used.add(idx)
            part = samples(run, idx)
            rows.extend(part)
            print(f'当てはめ入力: 動画{idx} / {len(part)}窓', flush=True)
    # 左右反転対称なロジスティック。切片0、非負傾きを最尤推定する。
    x, y = np.array([r['balance'] for r in rows]), np.array([r['win'] for r in rows])
    loss = lambda b: float(np.mean(np.logaddexp(0, b * x) - y * b * x))
    slope = float(minimize_scalar(loss, bounds=(0, 1), method='bounded').x)
    model = dict(slope=slope, intercept=0., c=CONFIDENCE, n_value=len(rows),
        games=len({(r['video'], r['game']) for r in rows}), videos=sorted(used),
        excluded_video=EXCLUDED_VIDEO, n_landing_nonzero=sum(r['landing_balance'] != 0 for r in rows),
        fit_basis='確定得点の返却量−攻撃量。着地増分も監査列に保存。左右対称、切片0。')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'fit_samples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(output / 'model.json', model)
    return model


def value(balance: float, model: dict) -> float:
    return float(expit(model['intercept'] + model['slope'] * balance))
