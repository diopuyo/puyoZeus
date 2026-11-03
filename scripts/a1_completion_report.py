"""A1の全採点表と検証票を、丸め前の原票から作成する。"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import pandas as pd

from scripts.a1_completion_replay import OUT, ROOT, INPUT
from scripts.falsedeath_replay_20261001 import records
from scripts.falsedeath_v2d_report import table
from scripts.falsedeath_collect_20261001 import SOURCES
from scripts.falsedeath_v2_score_20261002 import (
    TIME_POINT_MAX, TIME_CI_MAX, AGREE_POINT_MIN, AGREE_CI_MIN,
)

NICE_LEVEL = 10
KIB_PER_GIB = 1024**2
MIN_AVAILABLE_KIB = 6*KIB_PER_GIB


def read(path: Path) -> Any:
    """数値の原票をUTF-8で読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def csv_table(title: str, frame: pd.DataFrame) -> str:
    """全行・全列を省略せず報告書へ展開する。"""
    return table(title, list(frame.columns), frame.values.tolist())


def death_tables(result: dict) -> str:
    """セットと既存記録を別表にし、zenchiの重複加算を避ける。"""
    groups = {name: (row['deaths'], row['correct_delay']) for name, row in result['sets'].items()}
    groups.update({name: (row['change'], row['correct_delay'])
                   for name, row in result['existing5']['sources'].items()})
    rows, lost, delays, added = [], [], [], []
    for name, (change, timing) in groups.items():
        rows.append([name, len(change['before']['rows']), len(change['after']['rows']),
            sum(r['false'] for r in change['after']['rows']), change['correct_before'],
            change['correct_retained'], timing['lost_count'], timing['lost_delay'], timing['retained_delay']])
        lost.extend(dict(group=name, **row) for row in timing['lost'])
        delays.extend(dict(group=name, **row) for row in change['delays'])
        added.extend(dict(group=name, **row) for row in change['added'])
    text = table('確定・正しい確定の消失と遅れ',
        ['対象', '本番確定', 'A1確定', 'A1誤確定', '本番正解', '保持', '消失', '消失→観測死亡の遅れ秒', '保持分の時刻差秒'], rows)
    for title, values in [('消失の全件', lost), ('保持した確定の時刻差・全件', delays), ('追加確定の全件', added)]:
        text += csv_table(title, pd.DataFrame(values)) if values else f'## {title}\n\n0件。\n\n'
    return text


def runtime_tables() -> tuple[str, dict]:
    """41再生の経過時間と資源条件を全件列挙する。"""
    rows, started = [], []
    for mode in ('original', 'prod', 'a1'):
        for name in SOURCES if mode == 'original' else records():
            folder = OUT/'replay'/mode/name
            status, resource = read(folder/'status.json'), read(folder/'RESOURCE.json')
            assert resource['nice'] == NICE_LEVEL and resource['available_kib'] >= MIN_AVAILABLE_KIB
            assert set(resource['pools'].values()) == {'1'}
            started.append(datetime.fromisoformat(resource['started_utc']))
            rows.append([mode, name, status['frames'], status['elapsed_seconds'], resource['nice'],
                         resource['available_kib']/KIB_PER_GIB])
    now = datetime.now(timezone.utc)
    receipt = dict(replay_started=min(started).isoformat(), reported=now.isoformat(),
        replay_wall_seconds=(now-min(started)).total_seconds(),
        replay_sum_elapsed_seconds=sum(row[3] for row in rows))
    return table('全再生の時間・資源', ['構成', '記録', '更新フレーム', '再生実行秒', 'nice', '開始空きGiB'], rows), receipt


def verification() -> dict:
    """入力・モデルmanifestと本番設定をハッシュで固定する。"""
    files = [INPUT/f'{name}.jsonl.gz' for name in records()]
    files += [ROOT/'models'/name/'manifest.json' for name in ('exchange_event_v3', 'exchange_event_v1_r1_hgb')]
    files += [ROOT/'src/production_config.py']
    files += [ROOT/'docs/agent_coordination/DECISIONS.md', INPUT.parents[2]/'labels.json',
              INPUT.parents[2]/'set2/labels.json']
    files += [ROOT/'src'/name for name in ('exchange_completion_score.py', 'exchange_event_overlay.py',
        'exchange_event_cli.py', 'exchange_prefire_candidates.py', 'exchange_prefire_snapshot.py',
        'exchange_hidden_row_belief.py', 'exchange_hidden_row_probability.py')]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def supplementary_tables(result: dict) -> str:
    """表示外の確定、反転、補完の使用数も隠さず記録する。"""
    text = table('既存5記録の表示フレーム監査', ['構成', '記録', '確定フレーム', '誤確定', '勝敗未確認'],
        [[mode, name, row['certain_frames'], row['false_frames'], row['unknown_frames']]
         for mode, values in result['display'].items() for name, row in values.items()])
    text += table('warmup内部確定（表示区間の合否には含めない）', ['構成', '件数', '全件'],
        [[mode, len(rows), rows] for mode, rows in result['warmup'].items()])
    text += table('±3帯反転（記録のみ）', ['対象', '本番', 'A1', '1.2倍以内'],
        [[name, row['prod'], row['a1'], row['passed']] for name, row in result['reversals'].items()])
    rows = []
    for name in records():
        folder = OUT/'replay/a1'/name
        fallback = read(folder/'completion_score_audit.json')['rows']
        prefire = read(folder/'prefire_audit.json')['rows']
        rows.append([name, len({(r['game'], r['side'], r['chain_id']) for r in fallback}), len(fallback),
            sum(r['predicted_count'] > r['observed_count'] for r in fallback),
            sum(bool(r.get('score_distributions')) for r in prefire)])
    text += table('補完と候補分布の記録数（warmup含む）',
        ['記録', '補完連鎖', '補完更新', '残り連鎖あり更新', '候補分布を記録した発火'], rows)
    return text


def validation_tables() -> str:
    """旧監査との照合と実行済み検証を報告へ残す。"""
    check = read(OUT/'AUDIT_DEFINITION_CHECK.json')
    text = table('旧火力監査との定義照合（現行本番）', ['項目', '値'],
        [[key, value] for key, value in check.items()])
    text += ('火力誤差の対象は旧監査と同じ14区間（c1〜c6・zenchi・s0〜s6）。'
             '表示・門v2は指定の18記録、OFF互換性は指定の既存5記録を使用する。\n\n')
    rows = [[name, (OUT/name).read_text(encoding='utf-8').strip().replace('\n', '<br>')]
            for name in ('tests.log', 'cli_tests.log')]
    text += table('テスト結果', ['ログ', '結果'], rows)
    return text


def judgement(result: dict, errors: pd.DataFrame, gates: dict) -> str:
    """中央値の改善と不合格条件を分け、閾値も丸め前の値で判定する。"""
    metrics = result['combined']['metrics']
    timing, agreement = (metrics[name]['paired'] for name in ('m_time', 'm_agree'))
    verdict = '合格' if all(gates.values()) else '不合格'
    failed = ', '.join(name for name, passed in gates.items() if not passed) or 'なし'
    score = errors[(errors.group == 'observed_7plus') & (errors.metric == 'score')].set_index('mode')
    text = f'**総合判定: {verdict}。** 不通過項目: {failed}。採否は統括。\n\n'
    text += (f"観測7連鎖以上{int(score.loc['a1', 'n'])}件の初回得点誤差中央値は {score.loc['prod', 'abs_p50']:.3f} → "
             f"{score.loc['a1', 'abs_p50']:.3f} 点。P90は {score.loc['prod', 'abs_p90']:.3f} → "
             f"{score.loc['a1', 'abs_p90']:.3f} 点で、中央値と上位誤差は同じ方向には改善していない。\n\n")
    rows = [('M_time 対差', timing['diff'], f'≤ {TIME_POINT_MAX}', timing['diff'] <= TIME_POINT_MAX),
            ('M_time 95%上端', timing['ci95'][1], f'< {TIME_CI_MAX}', timing['ci95'][1] < TIME_CI_MAX),
            ('M_agree 対差', agreement['diff'], f'≥ {AGREE_POINT_MIN}', agreement['diff'] >= AGREE_POINT_MIN),
            ('M_agree 95%下端', agreement['ci95'][0], f'> {AGREE_CI_MIN}', agreement['ci95'][0] > AGREE_CI_MIN)]
    return text + table('門v2の閾値照合', ['項目', '実測', '条件', '合格'], rows)


def result_tables(result: dict, errors: pd.DataFrame, gates: dict) -> str:
    """固定条件・誤差・表示の全表を作る。"""
    text = '# A1 完走予測の得点整合・採点結果\n\n'
    text += ('基準 c13ea95、R1s + D-v2h2b。既定OFF `--completion-score-coherent`。採否は統括。'
             '18件とも同一の falsedeath_fix 再収集記録を読み、認識・学習モデル・門は変更していない。\n\n'
             '最大確率（同率は段数大）の候補の得点・盤面・段数を一緒に使う。'
             'S3/着弾の候補評価も代表候補へ揃え、分布は別記録に保持する。'
             '死亡証明の全候補上限は維持する。0段0点の補完は観測式と現在のGRAVITY_SETTLE盤面だけを使用する。'
             '途中盤面が欠測・UNKNOWN・浮遊状態のときは補完しない。\n\n'
             '得点の照合先は帰属済み式累計であり、人手真値ではない。'
             '初回予測は後続の観測で更新せず採点し、動的予測を別表にした。'
             '送り換算には従来と同じ候補生成時の経過秒を使用する。'
             '途中試行first_attempt/second_attemptは採点対象外で、replay/a1だけが正式候補。\n\n')
    text += judgement(result, errors, gates)
    text += table('事前登録判定', ['条件', '合格'], [[k, v] for k, v in gates.items()])
    values = dict(result['sets'], combined=result['combined'])
    text += table('M_time・M_agree', ['対象', '試合', '指標', '本番', 'A1', '対差', '95%区間'],
        [[name, r['games'], metric, v['prod'], v['a1'], v['paired']['diff'], v['paired']['ci95']]
         for name, r in values.items() for metric, v in r['metrics'].items()])
    text += table('3:00・第14試合', ['構成', '項目', '値'],
        [[mode, k, v] for mode, r in [('prod', result['baseline_regression']), ('a1', result['regression'])]
         for k, v in r.items()])
    text += csv_table('完走予測誤差・全集計', errors)
    text += death_tables(result)
    text += ('追加確定の set1 第47試合と zenchi 試合idx7 は同じ事例で、重複加算しない。\n\n'
             'レビュー表の raw_* は平滑前の採用1P確率（display_p1）、adv_* は表示有利不利'
             '（display_adv、−100〜+100）。有利不利を直線換算した値を勝率として扱わない。\n\n')
    text += csv_table('レビュー7場面・由来別全行', pd.read_csv(OUT/'scenes.csv'))
    text += supplementary_tables(result)
    text += table('開始未観測', ['構成', '記録', '試合', '未観測'],
        [[mode, name, row['games'], row['incomplete']] for mode, value in result['histories'].items()
         for name, row in value['records'].items()])
    text += table('OFF互換性', ['記録', 'ファイル', 'バイト一致'],
        [[name, file, row['identical']] for name, value in read(OUT/'COMPATIBILITY.json').items() for file, row in value.items()])
    return text


def main() -> None:
    """事前登録の門を動かさず、採否は統括へ委ねる。"""
    result = read(OUT/'RESULT.json')
    errors = pd.read_csv(OUT/'error_summary.csv')
    scores = errors[(errors.group == 'observed_7plus') & (errors.metric == 'score')].set_index('mode')
    gates = dict(result['gates'], large_chain_error=bool(scores.loc['a1', 'abs_p50'] < scores.loc['prod', 'abs_p50']))
    text = result_tables(result, errors, gates) + validation_tables()
    runtime, receipt = runtime_tables()
    text += runtime + table('所要時間', ['項目', '値'], list(receipt.items()))
    receipt.update(gates=gates, passed=all(gates.values()), hashes=verification())
    text += ('原票: `logs/a1_completion/RESULT.json`、`error_summary.csv`、`scenes.csv`、'
             '`{prod,a1}_all_chains.csv`、`{prod,a1}_dynamic_predictions.csv`。'
             '再現時は `git archive c13ea95 src scripts` を `logs/a1_completion/baseline/` へ展開し、'
             'WSLで `bash scripts/run_a1_completion.sh`、完了後 '
             '`bash scripts/run_a1_completion_finalize.sh` を実行する。'
             '完了票がある記録はスキップするため、新規再現には空の出力先が必要。\n')
    (OUT/'VERDICT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=1), encoding='utf-8')
    (ROOT/'docs/agent_coordination/A1_COMPLETION_SCORE_20261003.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=1), encoding='utf-8')
    (OUT/'REPORT.md').write_text(text, encoding='utf-8')
    (ROOT/'docs/agent_coordination/A1_COMPLETION_SCORE_20261003.md').write_text(text, encoding='utf-8')
    print(json.dumps(dict(gates=gates, passed=receipt['passed'])), flush=True)


if __name__ == '__main__':
    main()
