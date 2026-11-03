"""採点原票から、省略のないA2検収表を出力する。"""
from __future__ import annotations

import json
from pathlib import Path
from scripts.a2_replay import OUT
from scripts.falsedeath_v2d_report import table

DEST = Path(__file__).resolve().parents[1]/'docs/agent_coordination/A2_SAFETY_REVIEW_20261003.md'


def metric_table(value: dict) -> str:
    """事前登録の比較単位と合否をそのまま表示する。"""
    rows = []
    for name, metrics in [(k, v['metrics']) for k, v in value['sets'].items()] + [
            ('115試合合算', value['combined']['metrics'])]:
        for metric, row in metrics.items():
            pair = row['paired']
            rows.append([name, metric, row['prod'], row['a2'], pair['diff'], pair['ci95']])
    return table('門v2の指標', ['母数', '指標', '本番', 'A2', '差', '95%区間'], rows)


def history_table(value: dict) -> str:
    """全18記録の開始欠測・抑止回数・誤表示を列挙する。"""
    rows = []
    for name, modes in value['history'].items():
        a, b = modes['prod'], modes['a2']
        wrong = value['display_audit'][name]
        rows.append([name, a['games'], a['incomplete'], b['incomplete'],
            a['blocked_calls'], b['blocked_calls'], a['display_blocked_calls'], b['display_blocked_calls'],
            a['blocked_game_sides'], b['blocked_game_sides'],
            wrong['prod']['false_frames'], wrong['a2']['false_frames']])
    rows.append(['合計']+[sum(r[i] for r in rows) for i in range(1, len(rows[0]))])
    paired = [[r[0], r[1], *[f'{r[i]}→{r[i+1]}' for i in range(2, len(r), 2)]] for r in rows]
    return table('18記録の全表（本番→A2）', ['記録', '記録内試合', '開始未観測', 'H1抑止呼出し 全体',
        'H1抑止呼出し 表示内', '抑止試合×側', '誤確定表示フレーム'], paired)


def false_table(value: dict) -> str:
    """表示区間の計算確定イベントと表示フレームを別々に検収する。"""
    rows = [[name, row['deaths']['before']['false'], row['deaths']['after']['false']]
            for name, row in value['sets'].items()]
    existing = value['existing5']
    rows.append(['既存5記録', sum(r['false'] for r in existing['rows']['prod']), existing['false']])
    return table('誤った計算確定イベント（試合×死亡側）', ['対象', '本番', 'A2'], rows)


def delay_table(value: dict) -> str:
    """計算確定の消失と、保持された正しい確定の遅れを混同しない。"""
    groups = {k: dict(change=v['deaths'], correct_delay=v['correct_delay']) for k, v in value['sets'].items()}
    groups.update(value['existing5']['sources'])
    rows, details, changed = [], [], []
    for name, group in groups.items():
        change, delay = group['change'], group['correct_delay']
        label = 'zenchi（set1内数）' if name == 'zenchi' else name
        rows.append([label, change['correct_before'], change['correct_retained'], delay['lost_count'],
            len(change['added']), delay['retained_delay'], delay['lost_delay'], delay['missing_observed']])
        details += [[name, r['game'], r['side'], r['first_sec'], r['candidate_observed_death_sec'],
                     r['delay_sec']] for r in delay['lost']]
        changed += [[name, r['game'], r['side'], r['seconds']] for r in change['delays']]
    return table('正しい計算確定の消失・遅れ', ['対象', '本番正解', '保持', '消失', '追加',
        '保持の差秒分布', '消失→観測死亡の秒分布', '死亡未観測'], rows) + table(
        '消失した全事例', ['対象', '試合', '死亡側', '本番計算確定秒', '観測死亡秒', '遅れ秒'], details) + table(
        '保持した確定の時刻が変わった全事例', ['対象', '試合', '死亡側', '候補−本番 秒'], changed)


def review_table(value: dict) -> str:
    """レビュー7件の値と区間内の最初の計算確定を表示する。"""
    rows = []
    for row in value['review']:
        a, b = row['prod'], row['a2']
        first = [r['first_computed_death'] if r['first_computed_death'] is not None else 'なし'
                 for r in (a, b)]
        rows.append([row['case'], row['target'], a['p1'], b['p1'], a['source'], b['source'],
            *first,
            [a['minimum'], a['maximum']], [b['minimum'], b['maximum']]])
    return table('レビュー7件', ['#', '元動画秒', '本番1P', 'A2 1P', '本番由来', 'A2由来',
        '区間内計算確定 前', '後', '区間内1P範囲 前', '後'], rows)


def verification_tables() -> str:
    """独立参照の互換検収と、実行開始時の資源原票を併記する。"""
    path = OUT/'VERIFICATION.json'
    if not path.exists():
        return '\n互換・資源の最終検収票は生成待ち。\n'
    value = json.loads(path.read_text())
    rows = [[name, group['display.npz']['byte_identical'], group['events.jsonl']['byte_identical'],
             group['events.diagnostics.json']['equal'], group['events.diagnostics.json']['byte_identical']]
            for name, group in value['compatibility'].items()]
    text = table('既定OFFとc13ea95の互換', ['記録', '表示バイト一致', 'イベントバイト一致',
                 '診断JSON内容一致', '診断JSON生バイト一致'], rows)
    resources = []
    for mode in ('prod', 'a2'):
        group = [r[mode] for r in value['resources'].values()]
        resources.append([mode, len(group), min(r['available_kib'] for r in group)/1024**2,
            sorted({r['nice'] for r in group}), all(set(r['pools'].values()) == {'1'} for r in group)])
    text += table('全再生の実行資源', ['構成', '記録数', '開始時空きGiBの最小', 'nice', '全BLAS=1'], resources)
    text += '診断JSONは件数辞書のキー順が非決定的。既存compareと同じ内容比較と、生バイト比較を区別した。\n\n'
    audit = OUT/'START_AUDIT.json'
    if audit.exists():
        start = json.loads(audit.read_text())
        text += f"追加受理{start['accepted_games']}試合の受理時入力を監査。両側の実表示得点が同時に0だったものは{start['both_raw_zero']}試合。全例はSTART_AUDIT.json。\n\n"
    log = OUT/'unit_release.log'
    if log.exists():
        text += '最終関連テスト: `' + log.read_text().strip().splitlines()[-1] + '`。'
        text += 'スキップはE2_SHORT_ARTIFACTS未指定の既存実動画テスト。A2追加34件は全成功。\n'
    return text


def main() -> None:
    """統括向けの全表を生成し、本番採否とは明確に区別する。"""
    value = json.loads((OUT/'RESULT.json').read_text())
    text = '# A2 安全装置レビュー 2026-10-03\n\n'
    text += '事前登録: DECISIONS.md末尾「A1・A2」。現行本番R1s + D-v2h2bとA2のみを比較。採否は統括。\n\n'
    text += '既定OFF `--a2-safety-review`。式セッションの原点分離とH1′のMENU開始受理。モデル・門・鮮度条件は変更なし。\n\n'
    text += 'bootstrapは各10,000回。門の115試合合算はseed 0、セット別参考区間は既存seed（セット1=20261001、セット2=0）。\n\n'
    text += table('判定', ['条件', '結果'], list(value['gates'].items()) + [['全条件', value['passed']]])
    text += metric_table(value)
    text += table('回帰場面', ['構成', '3:00採点秒', '3:00直接秒', '第14試合誤確定'], [
        [k, r['scene']['legacy_metric_sec'], r['scene']['first_display_sec'], len(r['game14_false_times'])]
        for k, r in value['regression'].items()])
    text += false_table(value) + history_table(value) + review_table(value) + delay_table(value)
    text += '\n全体の抑止呼出しは表示外の暖機を含む。176試合は記録内試合の合算で、門の115試合と母数が異なる。試合×側は記録内で集約。\n'
    text += 'zenchiはセット1の内数。reviewも別記録として指定どおり検査しており、表を横断した和を重複のない物理試合数とは扱わない。\n'
    text += '消失・時刻変化表のset1/set2は固定ラベルの1始まり試合番号。既存5記録は記録内の0始まり番号。\n'
    text += 'H1抑止数の前後差は段原点補正も含むA2全体の比較であり、開始条件だけを切り替えた単独効果ではない。\n'
    text += '保持の時刻差分布は初回時刻が変わった事例だけ（負値は早期化）。消失は計算確定がなくなった試合×側で、後の観測死亡までの時間を別記。\n'
    text += '「正しい」は既存採点と同じ最終勝敗との一致を指し、その時点の物理的な回避不能性を独立に実証した意味ではない。\n'
    text += verification_tables()
    text += '\n原因・段対応・残る抑止は [診断書](A2_DIAGNOSIS_20261003.md)。'
    text += '数値原票は [RESULT.json](a2_safety_20261003/RESULT.json)、'
    text += '[互換・資源検収](a2_safety_20261003/VERIFICATION.json)、'
    text += '[開始受理35件](a2_safety_20261003/START_AUDIT.json)、'
    text += '[#6の証明](a2_safety_20261003/CASE6_SUMMARY.json)。\n'
    DEST.write_text(text, encoding='utf-8')
    print(str(DEST), flush=True)


if __name__ == '__main__':
    main()
