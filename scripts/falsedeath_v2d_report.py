"""D-v2dの全表を、短区間・全体の原票から生成する。採否は統括へ委ねる。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.falsedeath_replay_20261001 import OUT
from scripts import _diag_falsedeath_v2b_tables as short

DEST = OUT/'v2d'


def table(title: str, columns: list[str], rows: list[list[Any]]) -> str:
    """全列を明記した表を作り、欠測は未測定として区別する。"""
    def cell(value: Any) -> str:
        if value is None:
            return '未測定'
        if isinstance(value, float):
            return f'{value:.6f}'
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    lines = [f'## {title}', '', '| '+' | '.join(columns)+' |',
             '| '+' | '.join(['---']*len(columns))+' |']
    lines.extend('| '+' | '.join(map(cell, row))+' |' for row in rows)
    return '\n'.join(lines)+'\n\n'


def short_tables(ladder: dict) -> tuple[str, dict]:
    """梯子の対象4件・warmup・時間・資源を省略せず保存する。"""
    short.DEST = DEST
    cases = [short.enrich(c) for c in ladder['cases']]
    warmup = {s:short.warmup(s) for s in dict.fromkeys(c['source'] for c in cases)}
    warmup['review'] = short.warmup('review')
    statuses = {s:short.read(DEST/'short'/s/'status.json') for s in warmup}
    resources = {s:short.read(DEST/f'resources_{s}.json') for s in warmup}
    text = table('短区間4事例', ['記録','試合idx','対象秒','誤確定イベント値','誤確定表示','表示母数','warmup'],
        [[r[k] for k in ('source','game','original_sec','false_event_values','false_frames','display_frames_n','warmup')] for r in cases])
    scenes = {m:short.scene(OUT/m/'short/review/display.npz') for m in ('v2','v2b','v2c','v2d')}
    scenes['prod'] = short.scene(OUT/'replay/prod/review/display.npz')
    text += table('3:00場面', ['構成','直接表示秒','門の指標秒（EMA再適用）','表示source'],
                  [[m,r['direct_sec'],r['legacy_sec'],r['source']] for m,r in scenes.items()])
    text += table('短区間warmup内部イベント', ['記録','試合idx','死亡側','source','初回秒','最終秒','値数'],
        [[r[k] for k in ('record','game','side','source','first_sec','last_sec','values_n')] for rows in warmup.values() for r in rows])
    text += table('短区間実行資源・時間', ['記録','再生秒','nice','開始時空きGiB'],
        [[s,statuses[s]['elapsed_seconds'],resources[s]['nice'],resources[s]['memory_available_kib']/1024**2] for s in statuses])
    return text, dict(cases=cases, scenes=scenes, warmup=warmup, statuses=statuses, resources=resources)


def full_metrics(result: dict) -> str:
    """固定条件の新旧判定と、セット別・115試合合算の対差を並記する。"""
    displays = short.read(DEST/'DISPLAY_AUDIT.json')['v2d']
    text = table('門v2・旧条件', ['条件','門v2','旧条件'],
        [[k,result['gates_v2'].get(k,'対象外'),result['old_condition_gates'].get(k,'対象外')]
         for k in dict.fromkeys((*result['gates_v2'],*result['old_condition_gates']))])
    values = dict(result['sets'], combined=result['combined'])
    text += table('指標と本番との差', ['対象','試合数','指標','本番','v2d','対差','95%区間'],
        [[name,r['games'],metric,v['prod'],v['v2d'],v['paired']['diff'],v['paired']['ci95']]
         for name,r in values.items() for metric,v in r['metrics'].items()])
    text += table('表示区間の誤確定', ['対象','本番誤確定','v2d誤確定','v2d誤確定フレーム'],
        [[name,r['deaths']['before']['false'],r['deaths']['after']['false'],r['deaths']['after']['false_frames']]
         for name,r in result['sets'].items()]+[['既存5記録',sum(r['false'] for r in result['existing5']['rows']['prod']),
         result['existing5']['false'],sum(r['false_frames'] for r in displays.values())]])
    text += table('回帰・反転', ['対象','値'], [['回帰',result['regression']],['±3反転',result['reversals']]])
    text += table('全体warmup内部イベント（表示外・合否外）', ['構成','記録','試合idx','側','初回秒','source'],
        [[mode,*[r[k] for k in ('record','game','side','first_sec','source')]]
         for mode,rows in result['warmup'].items() for r in rows])
    return text


def delay_tables(result: dict) -> str:
    """正しい確定の消失と保持例の遅れを分け、全事例表も残す。"""
    from scripts.falsedeath_score_20261001 import distribution
    groups = {name:r['correct_delay'] for name,r in result['sets'].items()}
    groups.update({name:r['correct_delay'] for name,r in result['existing5']['sources'].items()})
    changes = {name:r['deaths'] for name,r in result['sets'].items()}
    changes.update({name:r['change'] for name,r in result['existing5']['sources'].items()})
    text = table('正しい確定の消失と遅れ（秒、本番比）',
        ['対象','消失','死亡未観測','消失から観測死亡までの分布','時刻が変わった保持例の遅れ分布'],
        [[name,r['lost_count'],r['missing_observed'],r['lost_delay'],r['retained_delay']] for name,r in groups.items()])
    rows = []
    for name, change in changes.items():
        unchanged = change['correct_retained']-len(change['delays'])
        delays = [r['seconds'] for r in change['delays']] + [0.]*unchanged
        rows.append([name,change['correct_before'],change['correct_retained'],unchanged,distribution(delays)])
    text += table('保持された正しい確定の全母数（時刻不変の0秒を含む）',
                  ['対象','本番正確数','保持数','時刻不変数','全保持例の遅れ分布'], rows)
    text += table('消失の全事例', ['対象','試合','側','本番初回秒','候補観測死亡秒','遅れ秒'],
        [[name,*[v[k] for k in ('game','side','first_sec','candidate_observed_death_sec','delay_sec')]]
         for name,r in groups.items() for v in r['lost']])
    return text


def omission_tables(value: dict) -> str:
    """呼出し数を試合数と混同せず、未観測を生存に読み替えない。"""
    return value['unit']+'。'+value['not_observed_definition']+'\n\n'+table(
        'E35″未知ツモ省略', ['記録','省略回数','後続死亡あり','後続死亡なし','表示内','表示外','死亡証明成立','履歴不完全下の成立'],
        [[name,*[r[k] for k in ('calls','observed','not_observed','display_calls','outside_display_calls','dead_calls','h1_blocked_calls')]]
         for name,r in dict(value['records'], total=value['total']).items()])


def history_tables(groups: dict) -> str:
    """記録単位の母数と全試合の開始根拠・抑止時刻を表示する。"""
    text = ('H1は両側の得点ゼロかつ空STABLE盤面を観測した後、最初の非空STABLEが2個であることを要求。'
        '連続観測の試合境界直後、両側得点ゼロの残像だけは開始判定を保留する。'
        '一度途中開始と判定した試合は全消しでも復帰しない。'
        '開始未観測は上記の開始証拠を確立できなかった試合数であり、動画自体の途中開始に限定しない。'
        '連続記録でも開始証拠が欠ければ不完全扱いになる保守性は、確定消失と併せて査読対象とする。\n\n'
        '履歴不完全秒は観測開始から開始設置の観測（または記録末尾）まで。'
        '抑止表示秒は直近の抑止評価を使ったS3_landing表示から同一試合の次表示までの積算。'
        '最終行の先は外挿しない。warmupの抑止時刻は別列に含む。\n\n')
    for scope, sources in groups.items():
        columns = ('observed_games','incomplete_games','blocked_games','blocked_evaluations',
                   'incomplete_seconds','blocked_display_frames','blocked_display_seconds')
        totals = {key:sum(row[key] for row in sources.values()) for key in columns}
        text += table(f'H1集計（{scope}、記録間の重複試合は別カウント）',
            ['記録','観測試合','開始未観測試合','証明抑止試合','抑止評価回数','履歴不完全秒','抑止表示行','抑止表示秒'],
            [[s,*[r[k] for k in columns]] for s,r in dict(sources, total=totals).items()])
        text += table(f'H1全試合（{scope}）',
            ['記録','試合idx','観測開始','観測終了','最初の設置観測','履歴完全','両側空観測','境界残像行','抑止回数','抑止初回','抑止最終','抑止表示秒'],
            [[s,*[g[k] for k in ('game','first_observation_sec','last_observation_sec','match_start_observed_sec',
                'history_complete','empty_observed_secs','boundary_residual_frames','blocked_evaluations',
                'first_blocked_sec','last_blocked_sec','blocked_display_seconds')]] for s,r in sources.items() for g in r['games']])
    return text


def supplement_tables(value: dict) -> str:
    """qとwarmupの勝敗照合を、採点対象の表示区間と分離して載せる。"""
    text = table('q従来指標（記録のみ）', ['構成','層','フレーム数','試合数','log loss','AUC'],
        [[mode,name,*[r[k] for k in ('frames','matches','log_loss','auc')]]
         for mode,metric in value['q'].items() for name,r in metric['groups'].items()])
    text += table('warmup勝敗照合（表示外・合否外）',
        ['構成','記録','試合idx','死亡側','初回秒','source','固定勝者','観測死亡側','判定'],
        [[mode,*[r[k] for k in ('record','game','side','first_sec','source','winners','observed_dead_sides','verdict')]]
         for mode,rows in value['warmup'].items() for r in rows])
    return text


def total_timing_table() -> str:
    """未採点試行と共有枠待機も含めた、初回検証からの総壁時計を保存する。"""
    from datetime import datetime, timezone
    value = short.read(DEST/'TIMING_START.json')
    started = datetime.fromisoformat(value['start_utc'])
    ended = datetime.now(timezone.utc)
    value.update(end_utc=ended.isoformat(), seconds=(ended-started).total_seconds())
    (DEST/'TOTAL_TIMING.json').write_text(json.dumps(value,ensure_ascii=False,indent=1),encoding='utf-8')
    return table('初回検証からの総所要時間', ['開始UTC','終了UTC','秒','範囲'],
        [[value['start_utc'],value['end_utc'],value['seconds'],value['definition']]])


def verification_tables() -> tuple[str, dict]:
    """門とは分けた表示監査と、測定時間を除く同一入力の検算を明示する。"""
    value = short.read(DEST/'VERIFY.json')
    display = short.read(DEST/'DISPLAY_AUDIT.json')
    text = table('既存5記録の表示フレーム監査（補助集計）',
        ['構成','記録','確定表示フレーム','誤確定フレーム','勝敗不明フレーム'],
        [[mode,source,*[r[k] for k in ('certain_frames','false_frames','unknown_frames')]]
         for mode,sources in display.items() for source,r in sources.items()])
    text += ('勝敗は従来の固定ラベルを優先し、欠測時だけ同一試合の観測死亡を使用。'
             '確定sourceかつ表示勝率が勝者の反対側にあるフレームを誤確定とする。門の追加条件ではない。\n\n')
    text += table('短区間と全体の共通部分の検算', ['記録','表示','H1監査','E35監査','実処理時間の差分行'],
        [[source,*[r[k] for k in ('display','history_audit','post_counter_audit','elapsed_differences')]]
         for source,r in value['prefix'].items()])
    text += ('最初の検算はE35監査のperf_counter実測値elapsed_secまで同一視して失敗した。'
             'この測定時間だけを分離し、判定値・時刻・H1証拠・表示の一致と門の真偽を再確認した。'
             '原票と初回失敗ログは保持。統括の独立再計算・査読の代替ではない。\n\n')
    return text, dict(verification=value, display_audit=display)


def validation_tables() -> str:
    """C6の実バイト比較と、実行済みテスト・測定時間を表へ載せる。"""
    from datetime import datetime
    receipt = short.read(DEST/'RECEIPT.json')
    text = table('C6 既定OFF原票比較', ['記録','display.npz','events.jsonl','再生秒'],
        [[r['source'],r['files']['display.npz']['comparison'],r['files']['events.jsonl']['comparison'],
          r['status']['elapsed_seconds']] for r in receipt['off_byte_comparisons']])
    text += table('回帰テスト', ['実行','結果'], [
        ['共通回帰・C6の5記録', (DEST/'tests.log').read_text().splitlines()[-1]],
        ['H1・E35″・境界残像', (DEST/'h1_unit.log').read_text().splitlines()[-1]],
        ['途中開始直後の全消し・得点欠測を含む最終H1', (DEST/'start_evidence_unit.log').read_text().splitlines()[-1]],
        ['整理後の共通回帰（C6は上記で実施済み）', (DEST/'final_unit.log').read_text().splitlines()[-1]]])
    start = datetime.fromisoformat((DEST/'worker.log').read_text().splitlines()[0].split(' ',1)[1])
    stop = datetime.fromisoformat((DEST/'worker.log').read_text().splitlines()[-1].split(' ',1)[1])
    text += table('所要時間', ['測定','秒'], [
        ['最終試行の梯子＋全体再生＋採点（共有枠待機を含む）',(stop-start).total_seconds()],
        ['短区間＋全体再生の実測合計',receipt['replay_seconds_total']],
        ['C6の5記録再生合計（共通テスト内）',receipt['off_replay_seconds_total']]])
    samples = [json.loads(line) for line in (DEST/'RESOURCE_SAMPLES.jsonl').read_text().splitlines()]
    text += table('実行中の資源確認', ['UTC時刻','空きGiB','このworktreeのPython数','nice一覧','BLAS'],
        [[r['utc'],r['available_kib']/1024**2,len(r['processes']),
          [p['nice'] for p in r['processes']],[p['blas'] for p in r['processes']]] for r in samples])
    return text+('既存テスト1件は E2_SHORT_ARTIFACTS 未指定のためskip。C6の5件は全実行・全通過。'
        '共通262件・C6の5件・開始証拠の追加2件で269件通過。初回の2失敗はupdateの51行と、'
        '読込後のソース編集によるinspect行位置不一致。構造整理と新プロセスで解消。\n\n'
        '最初の実装点検では境界の得点ゼロ残像を最初の設置と誤認したため、梯子採点前に停止。'
        '未完了試行はimplementation_check_1へ分離し、修正後は短区間を先頭から再実行した。'
        '全体再生では起動口更新中のシェル読込エラーで主制御器が停止したが、'
        '完了原票を保持して再開した。その後、途中開始直後の全消しを除外するため得点ゼロの証拠を追加し、'
        '全体未採点の試行をimplementation_check_2に分離。最終実装で梯子から再実行した。'
        '上記の壁時計時間は最終実装の試行。共有枠待機を含む。\n\n')


def main() -> None:
    """短区間停止なら全体を未実施と明示し、門の変更や本番登録はしない。"""
    ladder = short.read(DEST/'LADDER.json')
    text, data = short_tables(ladder)
    header = '# D-v2d 実行結果\n\n構成: D1 + D3′ + D4 + E35″ + H1。採否は統括。本番設定は変更しない。\n\n'
    header += '表の試合idxは記録内部の番号。c5のidx=0はセット1第26試合に対応する。\n\n'
    header += f'短区間梯子: {"通過" if ladder["passed"] else "不通過・全体再生停止"}。\n\n'
    if ladder['passed']:
        result = short.read(DEST/'RESULT_v2d.json')
        omissions = short.read(DEST/'OMISSIONS.json')
        header += f'門v2: {"通過" if result["passed_v2"] else "不通過"}、旧条件: {"通過" if result["old_condition_passed"] else "不通過"}。\n\n'
        text += full_metrics(result)+delay_tables(result)+omission_tables(omissions)
        data.update(result=result, omissions=omissions)
        supplement = short.read(DEST/'SUPPLEMENT.json')
        text += supplement_tables(supplement)
        data['supplement'] = supplement
        extra_text, extra_data = verification_tables()
        text += extra_text
        data.update(extra_data)
    else:
        text += '全体18記録・門v2・旧条件・第14試合・全体の消失と遅れ・省略集計は、梯子不通過のため未実施。\n'
    h1 = short.read(DEST/'HISTORY.json')
    text += history_tables(h1)
    data['history'] = h1
    text += validation_tables()
    text += total_timing_table()
    with (DEST/'TABLES.json').open('w', encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=1)
    with (DEST/'REPORT.md').open('w', encoding='utf-8') as stream:
        stream.write((header+text).rstrip()+'\n')


if __name__ == '__main__':
    main()
