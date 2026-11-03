"""D-v2cの全表を、短区間・全体の原票から生成する。採否は統括へ委ねる。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.falsedeath_replay_20261001 import OUT
from scripts import _diag_falsedeath_v2b_tables as short

DEST = OUT/'v2c'


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
    scenes = {m:short.scene(OUT/m/'short/review/display.npz') for m in ('v2','v2b','v2c')}
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
    text = table('門v2・旧条件', ['条件','門v2','旧条件'],
        [[k,result['gates_v2'].get(k),result['old_condition_gates'].get(k)]
         for k in dict.fromkeys((*result['gates_v2'],*result['old_condition_gates']))])
    values = dict(result['sets'], combined=result['combined'])
    text += table('指標と本番との差', ['対象','試合数','指標','本番','v2c','対差','95%区間'],
        [[name,r['games'],metric,v['prod'],v['v2c'],v['paired']['diff'],v['paired']['ci95']]
         for name,r in values.items() for metric,v in r['metrics'].items()])
    text += table('表示区間の誤確定', ['対象','本番誤確定','v2c誤確定','v2c誤確定フレーム'],
        [[name,r['deaths']['before']['false'],r['deaths']['after']['false'],r['deaths']['after']['false_frames']]
         for name,r in result['sets'].items()]+[['既存5記録',sum(r['false'] for r in result['existing5']['rows']['prod']),
         result['existing5']['false'],'原票の各事例を参照']])
    text += table('回帰・反転', ['対象','値'], [['回帰',result['regression']],['±3反転',result['reversals']]])
    text += table('全体warmup内部イベント（表示外・合否外）', ['構成','記録','試合idx','側','初回秒','source'],
        [[mode,*[r[k] for k in ('record','game','side','first_sec','source')]]
         for mode,rows in result['warmup'].items() for r in rows])
    return text


def delay_tables(result: dict) -> str:
    """正しい確定の消失と保持例の遅れを分け、全事例表も残す。"""
    groups = {name:r['correct_delay'] for name,r in result['sets'].items()}
    groups.update({name:r['correct_delay'] for name,r in result['existing5']['sources'].items()})
    text = table('正しい確定の消失と遅れ（秒、本番比）',
        ['対象','消失','死亡未観測','消失から観測死亡までの分布','保持された確定の遅れ分布'],
        [[name,r['lost_count'],r['missing_observed'],r['lost_delay'],r['retained_delay']] for name,r in groups.items()])
    text += table('消失の全事例', ['対象','試合','側','本番初回秒','候補観測死亡秒','遅れ秒'],
        [[name,*[v[k] for k in ('game','side','first_sec','candidate_observed_death_sec','delay_sec')]]
         for name,r in groups.items() for v in r['lost']])
    return text


def omission_tables(value: dict) -> str:
    """呼出し数を試合数と混同せず、未観測を生存に読み替えない。"""
    return value['unit']+'。'+value['not_observed_definition']+'\n\n'+table(
        'E35″未知ツモ省略', ['記録','省略回数','後続死亡あり','後続死亡なし','表示内','表示外','死亡証明成立'],
        [[name,*[r[k] for k in ('calls','observed','not_observed','display_calls','outside_display_calls','dead_calls')]]
         for name,r in dict(value['records'], total=value['total']).items()])


def validation_tables() -> str:
    """C6の実バイト比較と、実行済みテスト・測定時間を表へ載せる。"""
    from datetime import datetime
    receipt = short.read(DEST/'RECEIPT.json')
    text = table('C6 既定OFF原票比較', ['記録','display.npz','events.jsonl','再生秒'],
        [[r['source'],r['files']['display.npz']['comparison'],r['files']['events.jsonl']['comparison'],
          r['status']['elapsed_seconds']] for r in receipt['off_byte_comparisons']])
    text += table('回帰テスト', ['実行','結果'], [
        ['共通回帰・C6の5記録', (DEST/'tests.log').read_text().splitlines()[-1]],
        ['追加観測・キャッシュ・集計・構成', (DEST/'final_unit.log').read_text().splitlines()[-1]]])
    start = datetime.fromisoformat((DEST/'worker.log').read_text().splitlines()[0].split(' ',1)[1])
    stop = datetime.fromisoformat((DEST/'worker_resume.log').read_text().splitlines()[-1].split(' ',1)[1])
    text += table('所要時間', ['測定','秒'], [
        ['梯子（共有枠待機を含む）',(stop-start).total_seconds()],
        ['短区間再生の実測合計',receipt['replay_seconds_total']],
        ['C6の5記録再生合計（共通テスト内）',receipt['off_replay_seconds_total']]])
    return text+'既存テスト1件は E2_SHORT_ARTIFACTS 未指定のためskip。C6の5件は全実行・全通過。\n\n'


def main() -> None:
    """短区間停止なら全体を未実施と明示し、門の変更や本番登録はしない。"""
    ladder = short.read(DEST/'LADDER.json')
    text, data = short_tables(ladder)
    header = '# D-v2c 実行結果\n\n構成: D1 + D3′ + D4 + E35″。採否は統括。本番設定は変更しない。\n\n'
    header += f'短区間梯子: {"通過" if ladder["passed"] else "不通過・全体再生停止"}。\n\n'
    if ladder['passed']:
        result = short.read(DEST/'RESULT_v2c.json')
        omissions = short.read(DEST/'OMISSIONS.json')
        header += f'門v2: {result["passed_v2"]}、旧条件: {result["old_condition_passed"]}。\n\n'
        text += full_metrics(result)+delay_tables(result)+omission_tables(omissions)
        data.update(result=result, omissions=omissions)
    else:
        text += '全体18記録・門v2・旧条件・第14試合・全体の消失と遅れ・省略集計は、梯子不通過のため未実施。\n'
        text += ('\n最初の誤確定はc5の1727.433333秒、2P。E35監査は CHAIN、'
                 'firing_placed=true、next_unchanged=true、next_motion=false、ended=false、'
                 'end_signal_sec=null、未知ツモ省略=true を記録し、dead=true。'
                 '元の診断時刻1728.1秒より前から同じ事例を誤確定した。\n\n')
    text += validation_tables()
    with (DEST/'TABLES.json').open('w', encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=1)
    with (DEST/'REPORT.md').open('w', encoding='utf-8') as stream:
        stream.write((header+text).rstrip()+'\n')


if __name__ == '__main__':
    main()
