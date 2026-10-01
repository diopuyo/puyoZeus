"""E19事前登録の採点。指標式はセット1の実装を直接再利用する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts import eval_set_noise_20261001 as noise
from scripts import eval_set_score_20261001 as scorer
from scripts import eval_set_prefix_check_20261001 as prefix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'logs/eval_set/set2'
BOOT_N, BOOT_SEED = 10_000, 0
METRICS = ('m_time', 'm_agree')
VARIANTS = ('prod', 'e19')
CERTAIN_SOURCES = ('unavoidable_death', 'confirmed_death')
EVEN_PROBABILITY = .5
M_TIME_DELTA_UPPER_MAX = 0.
M_AGREE_DELTA_LOWER_MIN = -.01


def save(name: str, value: object) -> None:
    """JSON原票を保存する。"""
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')


def events(path: Path) -> list[dict]:
    """イベントJSONLを読む。"""
    return [json.loads(line) for line in path.read_text().splitlines()]


def death_audit(parts: list[str], variant: str, games: list[dict]) -> dict:
    """試合・側単位の確定を、時刻で公式窓へ割り当ててWIN勝者と照合する。"""
    cases, outside = {}, 0
    for part in parts:
        for event in events(OUT/'replay'/variant/part/'events.jsonl'):
            for value in event['values']:
                if value['source'] != 'unavoidable_death':
                    continue
                stamp = value['t_sec']
                game = next((g for g in games if g['start'] <= stamp < g['end']), None)
                if game is None:
                    outside += 1
                    continue
                for side in value['dead_sides']:
                    key = (game['game'], side)
                    row = dict(game=game['game'], side=side, first_sec=stamp,
                               winner=game['winner'], false=side == game['winner'])
                    if key not in cases or stamp < cases[key]['first_sec']:
                        cases[key] = row
    rows = list(cases.values())
    return dict(cases=len(rows), false=sum(r['false'] for r in rows),
                unlabelled=0, outside_window_values=outside, rows=rows)


def display_deaths(parts: list[str], variant: str, games: list[dict]) -> dict:
    """確定由来の表示が勝者の負けになっていないか、全表示行も監査する。"""
    wrong = []
    for part in parts:
        with np.load(OUT/'replay'/variant/part/'display.npz') as data:
            for game in games:
                mask = ((data['t_sec'] >= game['start']) & (data['t_sec'] < game['end'])
                        & np.isin(data['source'], CERTAIN_SOURCES))
                losing = (data['display_p1'] < EVEN_PROBABILITY if game['winner'] == '1P'
                          else data['display_p1'] > EVEN_PROBABILITY)
                stamps = data['t_sec'][mask & losing]
                if len(stamps):
                    wrong.append(dict(game=game['game'], frames=len(stamps), first_sec=float(stamps[0])))
    return dict(false_frames=sum(r['frames'] for r in wrong), rows=wrong)


def prefix_check() -> dict:
    """セット1と同じJSON行バイト一致と画像窓一致を調べる。"""
    records = OUT/'collect/records'
    rows = prefix.compare_rows(records/'s1.jsonl.gz', records/'check.jsonl.gz')
    windows = prefix.compare_windows(records/'s1.windows.json.gz', records/'check.windows.json.gz')
    passed = (rows['rows_compared'] > 0 and rows['mismatched_rows'] == 0
              and rows['check_rows_unpaired'] == 0 and windows['check_windows'] > 0
              and not windows['missing'] and not windows['differ'])
    result = dict(passed=passed, rows=rows, windows=windows)
    save('PREFIX_CHECK.json', result)
    return result


def score_labels(labels: Path) -> dict:
    """同一フレーム・同一試合を保証して、試合対ブートストラップを行う。"""
    games = json.loads(labels.read_text())['games']
    parts = list(dict.fromkeys(g['part'] for g in games))
    scores, displays = {}, {}
    for variant in VARIANTS:
        paths = [OUT/'replay'/variant/part/'display.npz' for part in parts]
        scores[variant] = scorer.score(paths, labels)
        displays[variant] = scorer.load_display(paths)
        save(f'score_{labels.stem}_{variant}.json', scores[variant])
    np.testing.assert_array_equal(displays['prod']['t_sec'], displays['e19']['t_sec'])
    index = np.random.default_rng(BOOT_SEED).integers(0, len(games), (BOOT_N, len(games)))
    metrics = {}
    for metric in METRICS:
        values = {v: noise.per_game(scores[v]['rows'], metric) for v in VARIANTS}
        metrics[metric] = dict(paired=noise.paired(values['e19'], values['prod'], index),
            **{v: dict(value=float(noise.aggregate(values[v])), ci95=noise.boot_ci(values[v], index))
               for v in VARIANTS})
    return dict(labels=str(labels), games=len(games), bootstrap=dict(n=BOOT_N, seed=BOOT_SEED),
                metrics=metrics, summaries={v:scores[v]['summary'] for v in VARIANTS},
                deaths={v:death_audit(parts,v,games) for v in VARIANTS},
                display_deaths={v:display_deaths(parts,v,games) for v in VARIANTS})


def main() -> None:
    """ラベル確定後だけ実行し、指定された合格条件をそのまま判定する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    args = parser.parse_args()
    result = score_labels(args.labels)
    result['prefix_check'] = prefix_check()
    regression = json.loads((OUT/'REGRESSION.json').read_text())
    result['regression'] = regression
    result['gates'] = dict(m_time=result['metrics']['m_time']['paired']['ci95'][1] < M_TIME_DELTA_UPPER_MAX,
        m_agree=result['metrics']['m_agree']['paired']['ci95'][0] > M_AGREE_DELTA_LOWER_MIN,
        false_death=result['deaths']['e19']['false'] == 0,
        regression=regression['passed'])
    assert result['prefix_check']['passed'], '測定器の健全性確認が不一致'
    result['passed'] = all(result['gates'].values())
    result['reference_next30'] = score_labels(OUT/'labels_next30.json')
    save('RESULT.json', result)
    print(json.dumps({k:result[k] for k in ('games','metrics','gates','passed')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
