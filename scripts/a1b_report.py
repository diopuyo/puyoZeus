"""A1b二構成の事前登録判定と、正しい確定の変化を原票からまとめる。"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import pandas as pd
from scripts.a1_completion_replay import ROOT
from scripts.falsedeath_v2d_report import table
from scripts.falsedeath_replay_20261001 import records

OUT = ROOT/'logs/a1b'
DOC = ROOT/'docs/agent_coordination/A1B_COMPLETION_SCORE_20261003'
NICE_LEVEL = 10
MIN_AVAILABLE_KIB = 6 * 1024**2


def read(path: Path) -> Any:
    """採点済みの原票を読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def csv_table(title: str, frame: pd.DataFrame) -> str:
    """必要な集計列を省略せず掲載する。"""
    return table(title, list(frame.columns), frame.values.tolist())


def death_changes(result: dict) -> dict:
    """各採点対象の消失・遅れ・追加を抽出する。zenchi重複は加算しない。"""
    groups = {name: (value['deaths'], value['correct_delay']) for name, value in result['sets'].items()}
    groups.update({name: (value['change'], value['correct_delay'])
                   for name, value in result['existing5']['sources'].items()})
    changes = {}
    for name, (change, timing) in groups.items():
        changes[name] = dict(correct_before=change['correct_before'],
            correct_retained=change['correct_retained'], lost=timing['lost'],
            delays=change['delays'], added=change['added'])
    return changes


def resources(mode: str) -> dict:
    """再生時の資源条件と所要時間を既存記録から集約する。"""
    rows = []
    for name in records():
        folder = OUT/mode/'replay/a1'/name
        resource, status = read(folder/'RESOURCE.json'), read(folder/'status.json')
        assert resource['nice'] == NICE_LEVEL and resource['available_kib'] >= MIN_AVAILABLE_KIB
        assert set(resource['pools'].values()) == {'1'}
        rows.append(dict(name=name, started=resource['started_utc'], seconds=status['elapsed_seconds']))
    return dict(records=rows, total_seconds=sum(row['seconds'] for row in rows),
                started=min(row['started'] for row in rows))


def metric_rows(mode: str, result: dict) -> list:
    """門v2の115試合の対差と区間を取り出す。"""
    return [[mode, name, row['prod'], row['a1'], row['paired']['diff'], *row['paired']['ci95'],
             result['gates'][name]] for name, row in result['combined']['metrics'].items()]


def adopted_a2_comparison(result: dict) -> dict:
    """A2単独の再生出力を基準にした同じ18記録の採点結果を併記する。"""
    reference = read(OUT/'a2_reference/RESULT.json')
    output = {metric: dict(a2=row['prod'], combo=row['a1'], paired=row['paired'])
              for metric, row in reference['combined']['metrics'].items()}
    assert all(row['combo'] == result['combined']['metrics'][metric]['a1'] for metric, row in output.items())
    return dict(adoption=reference['adoption'], baseline=reference['baseline'], metrics=output,
                gates=reference['gates'], deaths=death_changes(reference))


def republish_counts(mode: str) -> dict:
    """送り量換算直前に計装した分岐回数を記録単位で集計する。"""
    rows = {name: read(OUT/mode/'replay/a1'/name/'REPUBLISH_BRANCH.json')['count'] for name in records()}
    return dict(total=sum(rows.values()), records=rows, replayed_records=len(rows),
                prior_count_available=False, reason='保存済み出力は更新後の値のみのため全18記録を再生')


def main() -> None:
    """単独・併用の全判定とレビュー7場面を同じ報告書に保存する。"""
    results = {mode: read(OUT/mode/'RESULT.json') for mode in ('single', 'combo')}
    text = ('# A1b 完走予測・採点結果\n\n'
        '事前登録: `59d155a` の `DECISIONS.md`「A1b」。基準は旧本番 R1s + D-v2h2b。'
        '18記録・115試合、A1と同じ採点器・入力・bootstrap。'
        '本番出力は `logs/a1_completion/replay/prod` を再利用。\n\n'
        '既定OFF `--completion-score-max-chain`。最大段数、同段数なら最大確率の候補から'
        '得点・段数・盤面を取得する。補完・分布記録はA1のまま。'
        'comboのみ `--a2-safety-review`（`dca8384`）を追加。'
        '承認済みの再公開修正と分岐計装を入れ、両構成とも全18記録を作り直した。\n\n')
    text += gate_table(results)
    text += table('門v2（115試合）', ['構成', '指標', '旧本番', '候補', '対差', '95%下端', '95%上端', '合格'],
                  [row for mode, result in results.items() for row in metric_rows(mode, result)])
    text += detail_tables(results)
    receipt = {mode: dict(gates=result['gates'], passed=result['passed'],
        metrics=result['combined'], regression=result['regression'],
        deaths=death_changes(result), histories=result['histories'], resources=resources(mode),
        republish=republish_counts(mode))
        for mode, result in results.items()}
    if 'combo' in results:
        reference = adopted_a2_comparison(results['combo'])
        receipt['combo']['adopted_a2_comparison'] = reference
        text += table('先行登録されたA2本番比（6d3c4dc）', ['指標', 'A2本番', '併用', '対差', '95%区間'],
            [[name, row['a2'], row['combo'], row['paired']['diff'], row['paired']['ci95']]
             for name, row in reference['metrics'].items()])
        text += table('A2単独比: 正しい確定の変化', ['対象', '基準正解', '保持', '消失', '時刻差', '追加'],
            [[name, row['correct_before'], row['correct_retained'], row['lost'], row['delays'], row['added']]
             for name, row in reference['deaths'].items()])
    text += table('再公開分岐の回数', ['記録', '単独', '併用'],
        [[name, receipt['single']['republish']['records'][name], receipt['combo']['republish']['records'][name]]
         for name in records()])
    receipt['reported_utc'] = datetime.now(timezone.utc).isoformat()
    started = min(datetime.fromisoformat(receipt[mode]['resources']['started']) for mode in results)
    receipt['replay_wall_seconds'] = (datetime.fromisoformat(receipt['reported_utc']) - started).total_seconds()
    DOC.with_suffix('.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=1), encoding='utf-8')
    DOC.with_suffix('.md').write_text(text, encoding='utf-8')
    print(json.dumps({mode: result['gates'] for mode, result in results.items()}))


def gate_table(results: dict) -> str:
    """全合否を事前登録の閾値で示す。"""
    rows = []
    for mode, result in results.items():
        regression = result['regression']
        score = pd.read_csv(OUT/mode/'error_summary.csv')
        score = score[(score.group == 'observed_7plus') & (score.metric == 'score')].set_index('mode')
        rows.append([mode, result['gates']['m_time'] and result['gates']['m_agree'],
            [row['deaths']['after']['false'] for row in result['sets'].values()] + [result['existing5']['false']],
            regression['scene']['legacy_metric_sec'],
            len(regression['game14_false_times']), score.loc['a1', 'abs_p50'],
            score.loc['a1', 'abs_p90'], result['passed']])
    return table('事前登録の合否', ['構成', '門v2', '誤確定（set1/set2/既存5）', '3:00秒（≤2766）',
        '第14試合誤確定', '7連鎖以上得点誤差中央値', 'P90', '総合合格'], rows)


def detail_tables(results: dict) -> str:
    """初回火力誤差・正しい確定の全変化・レビュー場面・開始未観測を記録する。"""
    baseline = pd.read_csv(ROOT/'logs/a1_completion/scenes.csv')
    text = csv_table('旧本番: レビュー7場面', baseline[(baseline['mode'] == 'prod') & (baseline.source == 'ALL')][
        ['scene', 'set', 'match', 'start', 'end', 'raw_min', 'raw_max', 'raw_mean', 'adv_min', 'adv_max', 'adv_mean']])
    for mode, result in results.items():
        errors = pd.read_csv(OUT/mode/'error_summary.csv')
        text += csv_table(f'{mode}: 得点誤差', errors[errors.metric == 'score'][
            ['mode', 'group', 'total', 'n', 'abs_p50', 'abs_p90']])
        changes = death_changes(result)
        text += table(f'{mode}: 正しい確定', ['対象', '本番正解', '保持', '消失全件', '時刻差全件', '追加全件'],
            [[name, row['correct_before'], row['correct_retained'], row['lost'], row['delays'], row['added']]
             for name, row in changes.items()])
        scenes = pd.read_csv(OUT/mode/'scenes.csv')
        text += csv_table(f'{mode}: レビュー7場面', scenes[scenes.source == 'ALL'][
            ['scene', 'set', 'match', 'start', 'end', 'raw_min', 'raw_max', 'raw_mean',
             'adv_min', 'adv_max', 'adv_mean']])
        text += table(f'{mode}: 開始未観測', ['構成', '未観測', '試合'],
            [[name, row['incomplete'], row['games']] for name, row in result['histories'].items()])
    return text + ('時刻差は候補−本番（正なら遅れ）。zenchiとset1の同一試合は重複加算しない。'
        'rawは採用1P確率、advは表示有利不利。原票は `logs/a1b/{single,combo}/`。'
        '今回追加した再公開経路の単体テスト1件通過。画像追加・pushなし。\n')


if __name__ == '__main__':
    main()
