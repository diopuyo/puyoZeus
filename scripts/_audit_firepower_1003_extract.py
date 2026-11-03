"""本番再生の全発火・全評価入力・全表示フレームを時刻で対応付ける。"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from scripts._audit_firepower_1003_inventory import OUT, ROOT, resources
from scripts._audit_firepower_1003_replay import NAMES
from scripts._audit_firepower_1003_training import fire_column, stats
from src.scoring import compute_effective_rate
from src.indicators_v2 import estimate_chain_anim_duration_sec

FPS = 30
SIDES = ('1P', '2P')


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """圧縮原票を一行ずつ読む。"""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def games_for(name: str) -> list[dict[str, Any]]:
    """固定の試合勝者ラベルを使用する。"""
    file = ROOT.parent / ('set2/labels.json' if name.startswith('s') else 'labels.json')
    return json.loads(file.read_text())['games']


def label(sec: float, games: list[dict[str, Any]]) -> dict[str, Any]:
    """未来の勝者は集計専用。推論へは戻さない。"""
    for game in games:
        if game['start'] <= sec < game['end']:
            return dict(match=game['game'], winner=int(game['winner'] == '1P'))
    return dict(match=-1, winner=np.nan)


def chain_key(game: int, chain: dict[str, Any]) -> tuple[Any, ...]:
    """連鎖識別子は区間内試合番号と側を含む。"""
    return game, chain['side'], chain['chain_id']


def process_observation(row: dict[str, Any], initial: dict, observed: dict) -> None:
    """初回予測を凍結し、観測式の段数を独立に収集する。"""
    for chain in row['chains']:
        key = chain_key(row['game'], chain)
        if key not in initial:
            initial[key] = dict(chain, first_t=row['t'], score_elapsed=row.get('score_elapsed'),
                                initial_states=row['states'])
    for side, event in zip(SIDES, row['observed']):
        if event and event.get('mechanism') == 'formula_read':
            latest = next((c for c in reversed(row['chains']) if c['side'] == side), None)
            if latest is None:
                continue
            key = chain_key(row['game'], latest)
            value = observed.setdefault(key, dict(first_count=event['chain_count'], count=0, score=0))
            value['count'] = max(value['count'], event['chain_count'])
            value['score'] = max(value['score'], event['total_score'])


def chain_rows(name: str, folder: Path, initial: dict, observed: dict,
               times: np.ndarray, games: list[dict]) -> list[dict]:
    """初回予測と終了後の独立得点証拠を対比する。"""
    output = []
    for event in read_jsonl(folder / 'events.jsonl'):
        for c in event['chains']:
            key = chain_key(event['game_idx'], c)
            first = initial.get(key)
            if not first or not times[0] <= first['first_t'] <= times[-1]:
                continue
            obs = observed.get(key, {})
            elapsed = first['score_elapsed']
            rate = compute_effective_rate(elapsed) if elapsed is not None else np.nan
            pred = first['predicted_final_score']
            end = c['end_signal_sec']
            count = first['predicted_chain_count']
            expected = c['trigger_sec'] + estimate_chain_anim_duration_sec(count, 'empirical_table_2026_08_14') if count is not None else np.nan
            if label(first['first_t'], games)['match'] < 0:
                continue
            output.append(dict(segment=name, set=2 if name.startswith('s') else 1,
                **label(first['first_t'], games), game_idx=event['game_idx'], exchange=event['exchange_id'],
                side=c['side'], chain_id=c['chain_id'], trigger=c['trigger_sec'], observed=first['first_t'],
                pred_count=count, pred_score=pred, pred_sent=np.floor(pred/rate) if pred is not None else np.nan,
                rate=rate, obs_count=obs.get('count'), first_obs_count=obs.get('first_count'),
                obs_formula_score=obs.get('score'), formula_total=c['formula_total'],
                score_delta=c['score_delta'], display_score=c['display_score'], score_before=c['score_before'],
                drop_bonus=c['drop_bonus_score'], independent_score=c.get('finalized_evidence_score'),
                predicted_end=expected, observed_end=end, end_confirmed=c['end_confirmed'],
                end_reason=c['end_reason'], both_chain=first['initial_states'] == ['CHAIN', 'CHAIN'],
                **initial_landing(event, first['first_t'], c['side'])))
    return output


def initial_landing(event: dict, sec: float, side: str) -> dict:
    """初回着弾評価と観測着地を分離し、単一連鎖への帰属を断定しない。"""
    first = next((v for v in event['values'] if 'incoming' in v and sec <= v['t_sec'] <= sec + 1/FPS), None)
    receiver = '2P' if side == '1P' else '1P'
    landing = next((v for v in event['landings'] if v['side'] == receiver
                    and v['t_sec'] is not None and (v['fall_start_sec'] or -1) >= sec), {})
    return dict(initial_incoming1=first['incoming'][0] if first else None,
                initial_incoming2=first['incoming'][1] if first else None,
                initial_hands1=first['hands'][0] if first else None,
                initial_hands2=first['hands'][1] if first else None,
                first_receiver_landing=landing.get('t_sec'),
                landing_attribution='同じ撃ち合い内の最初の受け側着地。単独帰属未確認')


def observation_frames(folder: Path) -> tuple[pd.DataFrame, dict, dict]:
    """同時連鎖の火力差を未来情報を使わず各時点で作る。"""
    initial, observed, frames = {}, {}, []
    for row in read_jsonl(folder / 'observations.jsonl.gz'):
        process_observation(row, initial, observed)
        latest = [next((c for c in reversed(row['chains']) if c['side'] == s), None) for s in SIDES]
        preds = [c['predicted_final_score'] if c else None for c in latest]
        counts = [c['predicted_chain_count'] if c else None for c in latest]
        ratesec = row.get('score_elapsed')
        rate = compute_effective_rate(ratesec) if ratesec is not None else np.nan
        diff = (preds[0]-preds[1])/rate if all(v is not None for v in preds) else np.nan
        countdiff = counts[0]-counts[1] if all(v is not None for v in counts) else np.nan
        remaining = remaining_prediction(row, latest, observed, rate)
        frames.append(dict(t_sec=row['t'], pred_send_diff=diff, pred_count_diff=countdiff,
            pred1=preds[0], pred2=preds[1], count1=counts[0], count2=counts[1],
            drops1=row['drops'][0], drops2=row['drops'][1], rate=rate, **remaining))
    return pd.DataFrame(frames), initial, observed


def remaining_prediction(row: dict, latest: list, observed: dict, rate: float) -> dict:
    """未消化分の監査専用差。予測を超えた観測は残量0とし負値にしない。"""
    count_values, send_values = [], []
    for chain in latest:
        if chain is None:
            count_values.append(np.nan)
            send_values.append(np.nan)
            continue
        observation = observed.get(chain_key(row['game'], chain), {})
        pred_count, pred_score = chain['predicted_chain_count'], chain['predicted_final_score']
        count_values.append(max(0, pred_count-observation['count'])
            if pred_count is not None and 'count' in observation else np.nan)
        send_values.append(max(0, pred_score-(chain['formula_total'] or 0))/rate
                           if pred_score is not None else np.nan)
    return dict(remaining_count_diff=count_values[0]-count_values[1],
                remaining_send_diff=send_values[0]-send_values[1])


def landing_frames(folder: Path) -> pd.DataFrame:
    """同時刻に複数評価があれば最後の着弾評価を使う。"""
    values = []
    for event in read_jsonl(folder / 'events.jsonl'):
        for v in event['values']:
            if 'incoming' not in v:
                continue
            required, available = v['required_cancel'], v['near_future_send']
            deficit = [required[i]-available[i] if available[i] is not None else np.nan for i in range(2)]
            values.append(dict(t_sec=v['t_sec'], base_p1=v['base_p1'], landing_gfe=v['gfe_p1'],
                incoming_diff=v['incoming'][1]-v['incoming'][0],
                death_incoming_diff=v.get('death_incoming', v['incoming'])[1]-v.get('death_incoming', v['incoming'])[0],
                deficit1=deficit[0], deficit2=deficit[1],
                reason=json.dumps(v.get('state_safety'), ensure_ascii=False),
                bound_reason=json.dumps([x.get('reason') for x in v.get('post_counter_bound', [])]),
                certain1=v['completion_certain'][0], certain2=v['completion_certain'][1],
                incoming1=v['incoming'][0], incoming2=v['incoming'][1]))
    return pd.DataFrame(values).drop_duplicates('t_sec', keep='last').sort_values('t_sec')


def feature_rows(name: str, folder: Path, times: np.ndarray, games: list[dict]) -> None:
    """実推論列と欠測からの0補完を保存する。"""
    manifest = json.loads(Path('models/exchange_event_v3/manifest.json').read_text())['models']
    gcols = json.loads(Path('models/exchange_event_v1_r1_hgb/manifest.json').read_text())['models']['G_fe']['columns']
    grouped: dict[str, list[dict]] = {}
    for row in read_jsonl(folder / 'features.jsonl.gz'):
        if not times[0] <= row['t'] <= times[-1] or label(row['t'], games)['match'] < 0:
            continue
        model = row['model']
        model = ('G_fe_R1s' if row['static_r1s'] else 'G_fe_landing') if model == 'G_fe' else model
        grouped.setdefault(model, []).append(row)
    for model, rows in grouped.items():
        cols = gcols if model.startswith('G_fe') else manifest[model + '_prime_light']['columns']
        x = np.array([r['x'] for r in rows])
        info = pd.DataFrame([dict(t=r['t'], **label(r['t'], games), p=r['p'],
            both_chain=r['states'] == ['CHAIN', 'CHAIN']) for r in rows])
        info.to_csv(folder / f'input_{model}.csv', index=False)
        np.save(folder / f'input_{model}.npy', x)
        result = []
        for i, col in enumerate(cols):
            if not fire_column(col):
                continue
            na_known = [r for r in rows if 'd_missing' in r]
            result.append(dict(segment=name, model=model, column=col, **stats(x[:, i]),
                impute_known_n=len(na_known), imputed_zero=sum(i in r['d_missing'] for r in na_known)))
        pd.DataFrame(result).to_csv(folder / f'stats_{model}.csv', index=False)


def extract(name: str) -> None:
    """表示窓だけを対象とし、warmup重複を母数へ入れない。"""
    folder = OUT / 'replay' / name
    games = games_for(name)
    with np.load(folder / 'display.npz') as data:
        display = pd.DataFrame({k: data[k] for k in ('t_sec', 'source', 'display_p1', 'display_adv', 'state1', 'state2')})
    display = display[[label(t, games)['match'] >= 0 for t in display.t_sec]].reset_index(drop=True)
    times = display.t_sec.to_numpy()
    if (folder / 'shown.jsonl.gz').exists():
        shown = pd.DataFrame(read_jsonl(folder / 'shown.jsonl.gz')).drop_duplicates('t_sec', keep='last')
        display = display.merge(shown, on='t_sec', validate='one_to_one')
    else:
        display['shown_p1'] = np.nan
    obs, initial, observed = observation_frames(folder)
    chains = chain_rows(name, folder, initial, observed, times, games)
    pd.DataFrame(chains).to_csv(folder / 'chains.csv', index=False)
    frame = display.merge(obs, on='t_sec', validate='one_to_one')
    landing = landing_frames(folder)
    frame = pd.merge_asof(frame, landing, on='t_sec', direction='backward')
    # 直前の着弾行は現在の表示が着弾由来の場合だけ有効とする。
    active = frame.source.isin(['S3_landing', 'unavoidable_death'])
    for column in landing.columns.drop('t_sec'):
        frame[column] = frame[column].where(active)
    frame['both_chain'] = (frame.state1 == 'CHAIN') & (frame.state2 == 'CHAIN')
    frame['segment'], frame['set'] = name, 2 if name.startswith('s') else 1
    frame = pd.concat([frame, pd.DataFrame([label(t, games) for t in times])], axis=1)
    frame.to_csv(folder / 'frames.csv.gz', index=False)
    feature_rows(name, folder, times, games)
    (folder / 'EXTRACTED.json').write_text(json.dumps(dict(chains=len(chains), frames=len(frame))))
    print(name, len(chains), len(frame), flush=True)


def main() -> None:
    """完了済み区間を集計する。再生中のgzip末尾は読まない。"""
    for name in NAMES:
        resources()
        folder = OUT / 'replay' / name
        if (folder / 'status.json').exists() and ('--refresh' in sys.argv or not (folder / 'EXTRACTED.json').exists()):
            extract(name)


if __name__ == '__main__':
    main()
