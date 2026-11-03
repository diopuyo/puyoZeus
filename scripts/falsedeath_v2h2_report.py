"""固定したD-v2h2条件の全表・D-v2d対差・回復例を原票から生成する。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.falsedeath_replay_20261001 import OUT, records
from scripts.falsedeath_score_20261001 import distribution, noise
from scripts.falsedeath_v2d_display import scan as display_scan
from scripts.eval_set2_regression_20261001 import fixed_labels
from scripts.falsedeath_v2d_report import table

DEST = OUT/'v2h2'
BOOT_N, BOOT_SEED = 10_000, 0
# 2026-10-03 D-v2h2事前登録の例外継承上限。結果に応じて変更しない。
MTIME_EXCEPTION_MAX = 0.002
UNIQUE_GROUPS = ('set1', 'set2', 'review', 'q_7gc4TgFig', 'fcXG83vInDY', 'mia8KCjr52g')


def read(path: Path) -> Any:
    """UTF-8の原票だけを読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def comparisons(old: dict, new: dict, baseline: str = 'v2d') -> dict:
    """115試合の同じ行・同じ抽出添字でD-v2dからの対差を計算する。"""
    result = {}
    names = (*new['sets'], 'combined')
    for name in names:
        groups = tuple(new['sets']) if name == 'combined' else (name,)
        values = {mode: {metric: np.concatenate([noise.per_game(value['sets'][g]['rows'][mode], metric)
            for g in groups]) for metric in ('m_time', 'm_agree')}
            for mode, value in ((baseline, old), ('v2h2', new))}
        count = len(values[baseline]['m_time'])
        index = np.random.default_rng(BOOT_SEED).integers(0, count, (BOOT_N, count))
        result[name] = {metric: dict(
            baseline_value=float(noise.aggregate(values[baseline][metric])),
            v2h2=float(noise.aggregate(values['v2h2'][metric])),
            paired=noise.paired(values['v2h2'][metric], values[baseline][metric], index))
            for metric in ('m_time', 'm_agree')}
    return result


def death_groups(value: dict) -> dict:
    """重複するzenchi単独集計を全体件数へ二重加算しない。"""
    result = {name: dict(change=row['deaths'], delay=row['correct_delay'])
              for name, row in value['sets'].items()}
    result.update({name: dict(change=row['change'], delay=row['correct_delay'])
                   for name, row in value['existing5']['sources'].items()})
    return result


def recovery_rows(old: dict, new: dict) -> dict:
    """本番で正しかった同一試合・同一側の消失と回復を列挙する。"""
    result = {}
    for name, before in death_groups(old).items():
        after = death_groups(new)[name]
        old_lost = {(r['game'], r['side']): r for r in before['delay']['lost']}
        new_lost = {(r['game'], r['side']): r for r in after['delay']['lost']}
        current = {(r['game'], r['side']): r for r in after['change']['after']['rows']}
        rows = []
        for key in sorted(old_lost.keys()-new_lost.keys()):
            row, recovered = old_lost[key], current[key]
            observed, stamp = row['candidate_observed_death_sec'], recovered['first_sec']
            rows.append(dict(game=key[0], side=key[1], prod_sec=row['first_sec'], v2h2_sec=stamp,
                observed_sec=observed, delay_sec=stamp-row['first_sec'],
                recovered_sec=None if observed is None else observed-max(stamp, row['first_sec'])))
        result[name] = dict(old_lost=len(old_lost), new_lost=len(new_lost), recovered=rows,
            newly_lost=[new_lost[k] for k in sorted(new_lost.keys()-old_lost.keys())],
            delay_distribution=after['delay'],
            recovered_delay=distribution([r['delay_sec'] for r in rows]))
    return result


def histories() -> dict:
    """表示外を含む176の記録内試合を従来H1と同じ母数で比較する。"""
    old = read(OUT/'v2d/HISTORY.json')['full']
    acceptance = read(DEST/'START_ACCEPTANCE.json')
    result = {}
    for source in records():
        directory = OUT/'replay/v2h2'/source
        audit = read(directory/'history_integrity_audit.json')
        games = audit['games']
        assert [g['game'] for g in games] == [g['game'] for g in old[source]['games']], source
        result[source] = dict(games=len(games), v2d_incomplete=old[source]['incomplete_games'],
            v2h2_incomplete=sum(not g['history_complete'] for g in games),
            h1_prime_accepted=acceptance[source]['accepted_count'],
            elapsed_seconds=read(directory/'status.json')['elapsed_seconds'],
            resources=read(directory/'RESOURCE.json'), rows=games)
    return result


def gate_result(old: dict, new: dict, compare: dict, recovery: dict, history: dict,
                displays: dict) -> dict:
    """今回の①〜④だけを合否へ使い、本番比の旧門は参考値として残す。"""
    old_lost = sum(recovery[g]['old_lost'] for g in UNIQUE_GROUPS)
    new_lost = sum(recovery[g]['new_lost'] for g in UNIQUE_GROUPS)
    old_missing = sum(r['v2d_incomplete'] for r in history.values())
    new_missing = sum(r['v2h2_incomplete'] for r in history.values())
    clean_sets = all(r['deaths']['after']['false'] == r['deaths']['after']['false_frames'] == 0
                     for r in new['sets'].values())
    clean_existing = new['existing5']['false'] == new['existing5']['unresolved'] == 0
    clean_frames = all(r['false_frames'] == r['unknown_frames'] == 0 for r in displays.values())
    gates = dict(condition1=clean_sets and clean_existing and clean_frames,
        condition2=new['regression']['passed'],
        condition3=compare['combined']['m_time']['paired']['diff'] <= MTIME_EXCEPTION_MAX,
        condition4=new_missing < old_missing)
    return dict(gates=gates, passed=all(gates.values()),
        loss=dict(v2d=old_lost, v2h2=new_lost), incomplete=dict(v2d=old_missing, v2h2=new_missing),
        prod_gates_reference=new['gates_v2'], adoption='統括判断・本番設定変更なし',
        condition3_exception_limit=MTIME_EXCEPTION_MAX,
        original_condition3_reference=compare['combined']['m_time']['paired']['diff'] <= 0 and new_lost <= old_lost)


def metric_tables(new: dict, compare: dict) -> str:
    """本番比とv2d比のM_time/M_agreeを全セット・合算で並記する。"""
    text = table('本番比（門v2の形）', ['対象', '試合数', '指標', '本番', 'v2h2', '対差', '95%区間'],
        [[name, row['games'], metric, value['prod'], value['v2h2'],
          value['paired']['diff'], value['paired']['ci95']]
         for name, row in dict(new['sets'], combined=new['combined']).items()
         for metric, value in row['metrics'].items()])
    return text + table('D-v2d比', ['対象', '指標', 'v2d', 'v2h2', '対差', '95%区間'],
        [[name, metric, value['baseline_value'], value['v2h2'], value['paired']['diff'], value['paired']['ci95']]
         for name, row in compare.items() for metric, value in row.items()])


def supplementary_tables(new: dict) -> str:
    """表示外イベントと保持例の時刻変更も省略せず、主判定と分けて残す。"""
    text = table('表示外warmup内部確定（合否外）', ['構成', '記録', '試合idx', '側', '初回秒', 'source'],
        [[mode, *[row[k] for k in ('record','game','side','first_sec','source')]]
         for mode, rows in new['warmup'].items() for row in rows])
    text += table('保持した正しい確定の時刻変更・全例（本番比）', ['対象', '試合', '側', '遅れ秒'],
        [[name, row['game'], row['side'], row['seconds']]
         for name, value in death_groups(new).items() for row in value['change']['delays']])
    return text


def false_case_table(new: dict) -> str:
    """既存5記録の不合格事例を、最初の誤確定と逆側の観測死亡で特定する。"""
    rows = [r for r in new['existing5']['rows']['v2h2'] if r['false'] or r['unresolved']]
    text = table('不合格事例の全表', ['記録', '試合idx', '誤確定側', '初回秒', '相手側の観測死亡秒', '誤確定', '未解決'],
        [[r[k] for k in ('source','game','side','first_sec','opponent_death_sec','false','unresolved')]
         for r in rows])
    return text


def recovery_tables(recovery: dict) -> str:
    """全回復例と遅れ分布を記録し、最終勝敗一致と物理証明の正解を区別する。"""
    text = '「正しい」は既存の最終勝敗ラベルとの一致。物理的回避不能時刻の独立検証ではない。\n\n'
    text += table('正しい確定の消失と回復', ['対象', 'v2d消失', 'v2h2消失', '回復', '新規消失', '回復例の本番比遅れ分布'],
        [[name, r['old_lost'], r['new_lost'], len(r['recovered']), len(r['newly_lost']), r['recovered_delay']]
         for name, r in recovery.items()])
    text += 'zenchi単独行はセット1と重複し、総数へ再加算しない。\n\n'
    text += table('回復した確定の全例', ['対象', '試合', '側', '本番秒', 'v2h2秒', '観測死亡秒', '本番比遅れ秒', '回復秒'],
        [[name, *[r[k] for k in ('game','side','prod_sec','v2h2_sec','observed_sec','delay_sec','recovered_sec')]]
         for name, group in recovery.items() for r in group['recovered']])
    text += table('v2h2で残る消失の全例', ['対象', '試合', '側', '本番秒', '観測死亡秒', '遅れ秒'],
        [[name, *[r[k] for k in ('game','side','first_sec','candidate_observed_death_sec','delay_sec')]]
         for name, group in recovery.items() for r in group['delay_distribution']['lost']])
    text += table('遅れ分布（本番比・秒）', ['対象', '残る消失', '保持された時刻変更例'],
        [[name, r['delay_distribution']['lost_delay'], r['delay_distribution']['retained_delay']]
         for name, r in recovery.items()])
    return text


def write_report(value: dict, new: dict) -> None:
    """全原票へたどれるMarkdownとJSONを新規保存する。"""
    text = '# D-v2h2 実装・採点結果\n\n構成はD-v2h + --death-integrity-v2h2。採否は統括。\n\n'
    text += table('事前登録の判定', ['条件', '結果'], list(value['decision']['gates'].items()))
    text += table('短区間5事例', ['記録', '試合idx', '誤確定イベント値', '誤表示フレーム', 'warmup'],
        [[r[k] for k in ('source','game','false_event_values','false_frames','warmup')]
         for r in value['ladder']['cases']])
    text += table('3:00場面と第14試合', ['対象', '結果'],
        [['短区間', value['ladder']['scene']], ['全体', new['regression']]])
    text += metric_tables(new, value['compare'])
    text += table('D-v2h比', ['対象', '指標', 'v2h', 'v2h2', '対差', '95%区間'],
        [[name, metric, v['baseline_value'], v['v2h2'], v['paired']['diff'], v['paired']['ci95']]
         for name, row in value['compare_v2h'].items() for metric, v in row.items()])
    text += table('表示区間の誤確定', ['対象', '誤確定イベント', '誤表示フレーム'],
        [[name, r['deaths']['after']['false'], r['deaths']['after']['false_frames']]
         for name, r in new['sets'].items()] +
        [[name, sum(r['false'] for r in new['existing5']['rows']['v2h2'] if r['source']==name),
          row['false_frames']] for name, row in value['displays'].items()])
    text += recovery_tables(value['recovery'])
    text += recovery_tables(value['recovery_v2h']).replace('v2d', 'v2h')
    text += table('新規消失の全例（両基準）', ['基準', '対象', '試合', '側', '本番秒', '観測死亡秒', '遅れ秒'],
        [[mode, name, *[r[k] for k in ('game', 'side', 'first_sec', 'candidate_observed_death_sec', 'delay_sec')]]
         for mode, groups in [('v2d', value['recovery']), ('v2h', value['recovery_v2h'])]
         for name, group in groups.items() for r in group['newly_lost']])
    text += supplementary_tables(new)
    text += false_case_table(new)
    text += table('18記録の開始未観測・時間', ['記録', '試合数', 'v2d未観測', 'v2h2未観測', 'H1′受理', '再生所要秒'],
        [[name, *[r[k] for k in ('games','v2d_incomplete','v2h2_incomplete','h1_prime_accepted','elapsed_seconds')]]
         for name, r in value['history'].items()])
    text += table('実行時間と資源', ['項目', '値'], list(value['execution'].items()))
    text += '\n実行資源は各再生のRESOURCE.json。画像は同梱しない。\n'
    with (DEST/'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    with (DEST/'COMPARISON.json').open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)


def execution(history: dict) -> dict:
    """梯子起動から集計までの実時間と各再生の計算時間を区別する。"""
    line = (DEST/'worker.log').read_text().splitlines()[0]
    started = datetime.fromisoformat(line.removeprefix('START '))
    finished = datetime.now(timezone.utc)
    resources = [row['resources'] for row in history.values()]
    assert all(row['nice'] == 10 and row['available_kib'] >= 6*1024**2 for row in resources)
    assert all(all(v == '1' for v in row['pools'].values()) for row in resources)
    return dict(ladder_started=started.isoformat(), report_finished=finished.isoformat(),
        wall_seconds=(finished-started).total_seconds(),
        full_replay_sum_seconds=sum(row['elapsed_seconds'] for row in history.values()),
        min_available_gib=min(row['available_kib'] for row in resources)/1024**2,
        nice=10, blas_threads=1, launcher='run_limited')


def main() -> None:
    """18記録完了後、独立したv2h2出力先だけへ集計を書く。"""
    from scripts.falsedeath_v2h2_blocks import summarize, render
    ladder = read(DEST/'LADDER.json')
    blocks = summarize(ladder['passed'])
    if not ladder['passed']:
        write_short_failure(ladder, render(blocks))
        return
    old, new = (read(OUT/m/f'RESULT_{m}.json') for m in ('v2d', 'v2h2'))
    compare, recovery, history = comparisons(old, new), recovery_rows(old, new), histories()
    v2h = read(OUT/'v2h/RESULT_v2h.json')
    terminals, labels = read(OUT/'TERMINALS.json')['all'], fixed_labels()
    displays = {s: display_scan(s, 'v2h2', labels, terminals[s]) for s in new['existing5']['sources']}
    decision = gate_result(old, new, compare, recovery, history, displays)
    value = dict(decision=decision, compare=compare, recovery=recovery, history=history,
                 compare_v2h=comparisons(v2h, new, 'v2h'), recovery_v2h=recovery_rows(v2h, new),
                 displays=displays, ladder=read(DEST/'LADDER.json'), execution=execution(history))
    write_report(value, new)
    with (DEST/'REPORT.md').open('a', encoding='utf-8') as stream:
        stream.write(render(blocks))
    print(json.dumps(decision, ensure_ascii=False))


def write_short_failure(ladder: dict, blocks: str) -> None:
    """停止条件を守り、未実施の全体指標は未測定と明記する。"""
    text = '# D-v2h2 短区間判定\n\n事前登録どおりD-v2h + --death-integrity-v2h2。採否は統括。\n\n'
    text += table('梯子の判定', ['条件', '結果'], list(ladder['gates'].items()))
    text += table('短区間5事例', ['記録', '試合idx', '誤確定イベント値', '誤表示フレーム', 'warmup'],
        [[r[k] for k in ('source', 'game', 'false_event_values', 'false_frames', 'warmup')]
         for r in ladder['cases']])
    text += table('3:00場面', ['項目', '値'], list(ladder['scene'].items()))
    text += '\n梯子不合格のため18記録再生・①〜④の全体採点・M_time/M_agree・回復と遅れ分布は未実施。\n\n'
    text += blocks
    with (DEST/'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write(text)


if __name__ == '__main__':
    main()
