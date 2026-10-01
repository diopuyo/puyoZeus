"""Phase 6事前登録の試合等重み採点・対照・リーク監査。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from scripts.prefire_v6_replay import OUT, EVALSET
from scripts.prefire_v6_measure import save
from scripts import prefire_oracle_ceiling_20260930 as oracle
from scripts import prefire_gate_report_20260930 as legacy
from scripts.run_prefire_replay_20260930 import BASELINE_DIRS, EXEV
from scripts.prefire_truncation_audit_20260930 import compare_prefix, compare_trace

BOOTSTRAPS, SEED = 10000, 0
EPSILON = 1e-7
PREFIRE_SECONDS = 3.
LATE_FRACTION = 2/3
CI_PERCENTILES = (2.5, 97.5)
JOIN_COLUMNS = ('t_sec', 'display_p1', 'display_adv', 'game_idx', 'prefire_mask', 'prediction_delta')


def load(root: Path) -> dict:
    """表示の全列を読む。"""
    with np.load(root/'display.npz') as data:
        return {k:data[k].copy() for k in data.files}


def events(root: Path) -> list[dict]:
    """イベントを読み、発火窓は本番側だけで固定する。"""
    return [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]


def paired(left: np.ndarray, right: np.ndarray, ratio: bool = False) -> dict:
    """同じ試合を10,000回・seed 0で復元抽出する。"""
    index = np.random.default_rng(SEED).integers(0, len(left), (BOOTSTRAPS, len(left)))
    def aggregate(values: np.ndarray) -> np.ndarray:
        return values[..., 0].sum(axis=-1)/values[..., 1].sum(axis=-1) if ratio else values.mean(axis=-1)
    delta = aggregate(left[index])-aggregate(right[index])
    return dict(baseline=float(aggregate(right)), phase6=float(aggregate(left)),
                difference=float(aggregate(left)-aggregate(right)),
                ci95=np.percentile(delta, CI_PERCENTILES).tolist(), games=len(left))


def game_metrics(display: dict, game: dict, prefire: np.ndarray) -> dict:
    """同じ表示行で全区間LL・発火前LL・終盤一致率・±3帯反転を計算する。"""
    times = display['t_sec']
    mask = (times >= game['start']) & (times < game['end'])
    p = np.clip(display['display_p1'], EPSILON, 1-EPSILON)
    if not np.isfinite(p).all() or not mask.any():
        raise ValueError('非有限確率または空の試合窓')
    loss = -np.log(p if game['winner'] == '1P' else 1-p)
    late = mask & (times >= game['start']+LATE_FRACTION*(game['end']-game['start']))
    sign = 1 if game['winner'] == '1P' else -1
    game_ids = display.get('game_idx', np.zeros(len(times), dtype=int))
    flips = legacy.flips(dict(display_adv=display['display_adv'][mask], game_idx=game_ids[mask]))
    return dict(m_time=float(loss[mask].mean()),
        prefire=float(loss[mask & prefire].mean()) if (mask & prefire).any() else None,
        prefire_frames=int((mask & prefire).sum()),
        agree=[int(np.count_nonzero(display['display_adv'][late]*sign > 0)), int(late.sum())],
        flips=flips)


def placebo(baseline: dict, actual: dict, games: list[dict], increment: np.ndarray | None = None) -> dict:
    """予測増分を公式試合ごとに半周ずらす。別試合へ運ばない。"""
    p0, p1 = (logit(np.clip(d['display_p1'], EPSILON, 1-EPSILON)) for d in (baseline, actual))
    output = dict(baseline, display_p1=baseline['display_p1'].copy())
    for game in games:
        mask = (baseline['t_sec'] >= game['start']) & (baseline['t_sec'] < game['end'])
        delta = p1[mask]-p0[mask] if increment is None else increment[mask]
        shifted = np.roll(delta, len(delta)//2)
        changed = shifted != 0
        values = baseline['display_p1'][mask].copy()
        values[changed] = expit(p0[mask][changed]+shifted[changed])
        output['display_p1'][mask] = values
    return output


def join_displays(parts: list[dict]) -> dict:
    """公式窓の採点前に全表示を連結する。区間境界の端数フレームを落とさない。"""
    keys = [key for key in JOIN_COLUMNS if all(key in part for part in parts)]
    joined = {key:np.concatenate([part[key] for part in parts]) for key in keys}
    if 'game_idx' in keys:
        offset, unique = 0, []
        for part in parts:
            unique.append(part['game_idx']+offset)
            offset += int(np.max(part['game_idx']))+1
        joined['game_idx'] = np.concatenate(unique)
    order = np.argsort(joined['t_sec'], kind='stable')
    joined = {key:values[order] for key,values in joined.items()}
    if np.any(np.diff(joined['t_sec']) <= 0):
        raise ValueError('区間間の表示時刻が重複している')
    return joined


def prediction_increment(root: Path, stamps: np.ndarray) -> np.ndarray:
    """通常評価器の版差を混ぜず、実際のPhase 6補正だけを対照へ渡す。"""
    with np.load(root/'prefire_v6_features.npz') as data:
        rows = data['values']
    rows = rows[np.isin(rows[:,0], stamps)]
    result = np.zeros(len(stamps))
    delta = logit(np.clip(rows[:,3], EPSILON, 1-EPSILON))-logit(np.clip(rows[:,2], EPSILON, 1-EPSILON))
    result[np.searchsorted(stamps, rows[:,0])] = rows[:,4]*delta
    return result


def set2_inputs(games: list[dict]) -> tuple:
    """発火窓は区間内の本番イベントで固定し、全表示を時刻順に繋ぐ。"""
    displays = {variant:[] for variant in ('baseline','phase6')}
    deaths = {variant:set() for variant in displays}
    for part in dict.fromkeys(g['part'] for g in games):
        base_root, root = EVALSET/'set2/replay/prod'/part, OUT/'replay'/part
        baseline, actual = load(base_root), load(root)
        np.testing.assert_array_equal(baseline['t_sec'], actual['t_sec'])
        mask = oracle.inject(baseline, oracle.exchange_rows(baseline, events(base_root)), PREFIRE_SECONDS, 0.)[1]
        actual['prediction_delta'] = prediction_increment(root, actual['t_sec'])
        baseline['prediction_delta'] = np.zeros(len(baseline['t_sec']))
        for variant, data, directory in [('baseline',baseline,base_root), ('phase6',actual,root)]:
            data['prefire_mask'] = mask
            displays[variant].append(data)
            deaths[variant].update(false_deaths(events(directory), games))
        deaths['phase6'].update(predicted_false_deaths(root, games))
    return join_displays(displays['baseline']), join_displays(displays['phase6']), deaths


def score_set2() -> dict:
    """58試合を固定して集計。発火なし試合は区間LLの母数を別記する。"""
    games = json.loads((EVALSET/'set2/labels.json').read_text())['games']
    assert len(games) == 58
    rows = {v:[] for v in ('baseline', 'phase6', 'placebo')}
    baseline, actual, deaths = set2_inputs(games)
    control = placebo(baseline, actual, games, actual['prediction_delta'])
    for variant, display in [('baseline',baseline), ('phase6',actual), ('placebo',control)]:
        rows[variant] = [game_metrics(display, game, baseline['prefire_mask']) for game in games]
    result = {}
    for metric in ('m_time', 'prefire', 'agree'):
        keep = [i for i,r in enumerate(rows['baseline']) if r[metric] is not None]
        arrays = {v:np.asarray([rows[v][i][metric] for i in keep]) for v in rows}
        result[metric] = paired(arrays['phase6'], arrays['baseline'], metric == 'agree')
        if metric == 'prefire':
            result['placebo'] = paired(arrays['placebo'], arrays['baseline'])
    result.update(flips={v:sum(r['flips'] for r in rows[v]) for v in ('baseline','phase6')},
                  false_deaths={v:len(d) for v,d in deaths.items()}, rows=rows)
    return result


def false_deaths(records: list[dict], games: list[dict]) -> set:
    """既存採点と同じ試合・側単位で回避不能死の誤りを数える。"""
    result = set()
    for event in records:
        for value in event['values']:
            if value['source'] != 'unavoidable_death':
                continue
            game = next((g for g in games if g['start'] <= value['t_sec'] < g['end']), None)
            if game is not None and game['winner'] in value['dead_sides']:
                result.add((game['game'], game['winner']))
    return result


def predicted_false_deaths(root: Path, games: list[dict]) -> set:
    """予測層から新たに表示したE35根拠も、既存イベントにないからと除外しない。"""
    result = set()
    with np.load(root/'prefire_v6_features.npz') as data:
        columns = list(data['columns'])
        rows = data['values']
    rows = rows[np.isin(rows[:,0], load(root)['t_sec'])]
    for game in games:
        mask = (rows[:,0] >= game['start']) & (rows[:,0] < game['end'])
        losing_proof = 'proof_2p' if game['winner'] == '1P' else 'proof_1p'
        if np.any(rows[mask, columns.index(losing_proof)]):
            result.add((game['game'], game['winner']))
    return result


def regression() -> dict:
    """既存の独立ラベル監査器を使い、事前登録どおり件数で比較する。"""
    from scripts.prefire_v6_regression import score
    return score()


def regression_predictions() -> set:
    """既存5記録の固定ラベルに対し、追加したE35表示を数える。"""
    from scripts import aggregate_e3_exchange_eval_20260926 as metrics
    cases = set()
    for source in BASELINE_DIRS:
        if source in ('review', 'zenchi'):
            games = json.loads((EXEV/'logs/review_zenchi_part3/official_games.json').read_text())
        else:
            windows, _ = metrics.outcomes(source)
            games = [dict(game=i, start=w['start']/metrics.FPS, end=w['end']/metrics.FPS,
                          winner=w['winner']) for i,w in enumerate(windows)]
        with np.load(OUT/'replay'/source/'prefire_v6_features.npz') as data:
            columns, rows = list(data['columns']), data['values']
        rows = rows[np.isin(rows[:,0], load(OUT/'replay'/source)['t_sec'])]
        for game in games:
            column = 'proof_2p' if game['winner'] == '1P' else 'proof_1p'
            mask = ((rows[:,0] >= game['start']) & (rows[:,0] < game['end'])
                    & (rows[:,columns.index(column)] > 0))
            cases.update((source, int(index), game['winner']) for index in rows[mask,1])
    return cases


def leakage() -> dict:
    """打切り再生の表示と予測内訳を全長の同時刻まで照合する。"""
    result = {}
    for cut in (300, 600):
        full, short = OUT/'replay/q_7gc4TgFig', OUT/'cut'/f'q{cut}'
        result[str(cut)] = dict(display=compare_prefix(full, short, cut), trace=compare_trace(full, short, cut))
        with np.load(full/'prefire_v6_features.npz') as a, np.load(short/'prefire_v6_features.npz') as b:
            keep = a['values'][:,0] < cut
            result[str(cut)]['features_equal'] = bool(np.array_equal(a['values'][keep], b['values'], equal_nan=True))
    return result


def main() -> None:
    """全6条件を個別判定し、リーク監査も採用の必須条件にする。"""
    result = score_set2()
    from scripts.prefire_v6_checks import off_controls
    result['off_controls'] = off_controls()
    result['regression'], result['leakage'] = regression(), leakage()
    latency = json.loads((OUT/'latency.json').read_text())
    result['latency'] = latency
    result['gates'] = dict(condition1=latency['p95_ms'] <= 200.,
        condition2=result['prefire']['ci95'][1] < 0.,
        condition3=result['m_time']['ci95'][1] < .002,
        condition4=result['agree']['ci95'][0] > -.01 and result['flips']['phase6'] <= 1.2*result['flips']['baseline'],
        condition5=result['false_deaths']['phase6'] <= result['false_deaths']['baseline'] and result['regression']['passed'],
        condition6=result['placebo']['ci95'][1] >= 0.)
    result['leakage_passed'] = all(v['display']['passed'] and v['trace']['passed'] and v['features_equal']
                                    for v in result['leakage'].values())
    result['passed'] = all(result['gates'].values()) and result['leakage_passed']
    save(OUT/'RESULT.json', result)
    print(result['gates'], flush=True)


if __name__ == '__main__':
    main()
