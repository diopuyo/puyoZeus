"""診断4〜7の全表・代表値・証拠リンクを再生成する。合否は扱わない。"""
from __future__ import annotations

import csv
import json
from typing import Any
import numpy as np
from scripts._diag_review1003b_extract import ROOT, OUT, resources, save, records
from scripts._diag_review1003b_inspect import TIMES
from src.exchange_event_record import decode, encode

PREFIX = '# レビュー指摘4〜7の診断（合否・採否の判断なし）\n\n'
INTRO = '''基準コード802efbc。時刻は元動画の秒、確率はすべて1P勝率。30fpsの記録を使用。
#4だけ確定発生を含めるため2763秒まで延長（指定区間2748〜2762秒は420フレーム、延長30）。
母数は4件・3試合、抽出計1382フレーム（元CSV16632フレーム）。静止G_feの同一入力比較は207呼出。
製品コード・設定・モデルを変更せず保存入力から再生した。3試合のevents.jsonlとdisplay.npzは納品原票とバイト一致、診断JSONは辞書として一致。
盤面は認識器の保存値であり、人手の全セル正解との照合ではない。「残り」は予測最終段−通知段数の参考値であり、欠測・段番号ずれ時は真の残り段数ではない。
空欄はその評価器の値がないことを表す。G_fe欄が撃ち合い中に残っていても選択された評価ではない。
全フレーム・両者13×6盤面・NEXT・通知原値・完走予測はcase*_full.csvとcase*_timeline.jsonlに保存。
'''
FINDINGS = '''
## 理由と修正候補を読む際の区別

- #4: 2761.017秒まで主に攻撃の未裏付け。その後E35⁗′の直近連続2段と鮮度が止め、2762.283秒に解消してE35上限証明が成立する。2Pの7段は途中観測で、最終予測は13段・75180点。1P87720点に対し過去の2連鎖360点も含む2P75540点で、純差174個。
- #5: 4244秒の認識値は色ぷよ56対60個。量差は新モデルでも2P寄与だが、1Pのcurrent_max_chain=12、相手=0という機能特徴が相殺する。旧LRは最大連鎖差の寄与が強い。新側の入力を固定した新旧比較207呼出のうち、旧実記録も一致したのは195呼出。残り12呼出は最大0.004613の残差（入力・更新時点差等の原因は未特定）。表の4代表時刻はすべて旧実記録も一致し、推定器差を直接比較できる。
- #6: 旧確定の初時刻は4271.667秒。旧E35は439個受け・1手・完走後盤面1候補について202ノードで全滅を証明。新は4272.567秒まではD3′の独立得点裏付け、以後はE35⁗′のrecent_mismatchで止まる。7→8段が予測6400/観測5120、8→9段が13720/6400。鮮度は推定10段・観測9段で通過。通知段数が1段先行するように見えるが、認識と連鎖ID境界のどちらが原因かは未確定（推測）。1Pの当該発火通知は段2から始まりprefireはmissing_first_formula。
- #7: 2Pの発火時予測は0段・0点・完走盤面なし。prefire_snapshotもfirst_scoreで不受理。区間内のmidchain採用なし。6049.250〜6060.617秒はmissing_post_counter_completion。6060.650秒以降は候補が得られるが、未知手持ち前置を含む楽観探索に生存枝があり（optimistic_survivor、152ノード）、証明自体が成立しない。6060.750秒からG_feへ戻り0.9896となるが死亡確定ではない。旧は6060.417秒から1P死亡を誤確定しているため、旧の確定を正解として扱わない。

H1のhistory_completeは3件すべてtrue、履歴理由の取消し0。H1′の各時刻accepted=falseは「その時刻の開始条件」の値で、試合開始の観測失敗ではない。
D1は単発死亡候補を扱う条件。この区間のsingle_death_proofは未実行で、主要な停止原因ではない。

## S3′が横ばいになる仕組み

完走得点はprovisional_score→側別合計→score_featuresに入り、送り量s/(1+s)、純差d/(1+abs(d))と大小フラグになる。差は未入力ではない。
追加NF_k1〜k5はlog1p、counter_marginは符号付きlog1pであり、純差の大きさも間接入力される。
一方、残り段数・残演出時間・着弾時刻・死亡証明の未通過理由はS3′の入力にない。発火時Dとarrivalは凍結される。
表示のS3_landingはS3′と仮想着弾G_feのlogit平均（隠し段候補があれば候補ごとの合成後に加重平均）。仮想着弾G_feは静止R1sではなく旧共通G_feを使用。
#6は4263.533秒以降、側別合計111480対142100、NF両側全0、counter_margin=-437.43/0で固定。S3′0.195888と着弾G_fe0.295039を合成して0.242024となる。
#4は2759.983秒時点で送り量変換0.999203/0.999074、純差変換0.994286。S3′の木経路寄与は凍結D=-0.900958、arrival=-0.281360、得点=-0.279777、count=+0.972158、基準=+0.083419（logit）で0.399747。着弾側0.796729と合成して0.617684。
#4では1Pの多数決完走盤面が欠測し、live countは発火前履歴の再シミュレーションにフォールバック。未発火の起点が残り、NF約1224個を保持している。完走得点と完走後能力の対応が弱い経路が存在する（影響の修正後比較は未実施）。
#7のS3′は0.7817→0.7070、着弾G_feは0.8763→0.8775などで、受け量の変化を確率として扱う。確定経路が成立しない限り0.98に固定する規則はない。

## 学習範囲と説明法

S3′学習資産logs/e15/S3_prime.npyは84445行・144動画・21466撃ち合い。いずれかのSTABLE盤面/NEXT更新時だけ行を作り、fill_phaseが中盤の行に限定。連鎖終点の得点差を発火時完走予測の代理にしている。
今回の代表S3特徴は全単変量で学習min〜max内。これだけで多変量の分布内・校正良好とはいえない。「大連鎖だから学習範囲外」と断定する根拠はない。飽和得点3列が各々±0.01以内という緩い近似では#4の2759.983秒120行、#6の4265秒114行があるが、これは同じ盤面・同じ残り時間の実例数ではない。
HistGBCの寄与は学習ノード個数で重み付けした子孫葉平均を基準とする木経路分解（Saabas型）。全特徴寄与+基準がdecision_functionと一致することを検証。SHAPや因果効果ではなく、分岐順依存。
旧LRは係数×入力の正確なlogit寄与。別保存の単変量0中和は感度実験であり、物理的に成立する盤面変更・加法寄与ではない。
'''
CANDIDATES = [
    ('#4/#6', 'src/exchange_event_tracker.py:442; src/exchange_death_integrity_v2.py:9', '式と表示の非同期・連鎖ID帰属を原票で整列し、独立証拠の成立を早める', '予測を実測へ昇格させる変更は誤確定を再発させる。帰属と同一段の照合が必要'),
    ('#4/#6', 'src/exchange_post_counter_steps.py:21; src/exchange_post_counter_recent.py:33', '式の段番号と新規発火の段原点を照合し、直近2段の誤対応を診断・補正', '任意の段ずらしで一致を探すと誤盤面も通す。単なる閾値緩和とは分離'),
    ('#7', 'src/exchange_event_overlay.py:577; src/exchange_prefire_snapshot.py:28; src/exchange_midchain_completion.py:75', '発火直前の盤面/設置・初段得点対応を改善し、欠けた完走予測を回復', '途中の不確定盤面を正解扱いすると火力過大予測。次段の独立照合を維持'),
    ('#7', 'src/post_counter_death_bound.py:144; src/exchange_post_counter_bound.py:148', 'optimistic_survivorの具体枝と未知手持ち候補を保存し、観測で除外可能か調査', '未知枝を無根拠に削ると安全な上限証明ではなくなる'),
    ('#4', 'src/exchange_event_overlay.py:631; src/exchange_hidden_row_probability.py:62', '候補ごとの得点と完走盤面を対にしてlive countも同時に評価', '候補数・探索時間増。学習側との入力定義差、再学習・候補上限の評価が必要'),
    ('#4/#6/#7', 'src/exchange_event_features.py:121; src/exchange_event_count_features.py:106; scripts/e15_training_rows_20260928.py:118', '非飽和の予測送り量/純差、残り段数/残時間、予測有効性を観測特徴として追加', '現モデルと列順互換を保つ別版が必要。対称化・正規化・学習時未来漏洩を全経路で確認'),
    ('#4/#6/#7', 'src/exchange_event_landing.py:439; scripts/e15_training_rows_20260928.py:143', '終盤の実予測入力を学習・校正対象に含め、S3′と着弾確率の合成を検証', '中盤性能・校正変化、同動画漏洩。今回4件への過適合を避ける別動画検証が必要'),
    ('#5', 'src/exchange_event_features.py:22; src/exchange_event_evaluator.py:274', 'ぷよ量と発火可能火力の相反をUIへ根拠表示し、R1sの局所誤差を別動画で計測', '単純な量差単調制約は高積み窒息や発火機能との関係を壊す可能性'),
]


def fmt(value: Any) -> str:
    """表の数値を丸め、空欄と配列を識別する。"""
    if value is None or value == '':
        return '—'
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass
    if isinstance(value, float):
        return f'{value:.6f}'.rstrip('0').rstrip('.')
    return str(value).replace('|', '/')


def table(headers: list[str], rows: list[list[Any]]) -> str:
    """Markdownの全列を省略せず作る。"""
    lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    lines.extend('| '+' | '.join(fmt(v) for v in row)+' |' for row in rows)
    return '\n'.join(lines)+'\n\n'


def full_tables(case: int) -> list[dict[str, Any]]:
    """全フレームへ再生予測を追加し、単独で読めるCSVにする。"""
    observations = {} if case == 5 else {r['t_sec']: decode(r)
        for r in records(OUT / f'replay_case{case}' / 'observations.jsonl')}
    rows = records(OUT / f'case{case}_timeline.jsonl')
    flattened = []
    for row in rows:
        frame = dict(row['frame'])
        stamp = float(frame['t_sec'])
        extra = observations.get(stamp, {})
        for key, value in {'raw_input': row['input'], 'landing_audit': row['landing'],
            'score_prediction': row['score_prediction'], 'replay_observation': encode(extra)}.items():
            frame[key] = json.dumps(value, ensure_ascii=False)
        flattened.append(frame)
        row['observation'] = extra
    with (OUT / f'case{case}_full.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    return [min(rows, key=lambda r: abs(float(r['frame']['t_sec'])-t)) for t in TIMES[case]]


def evaluation_table(rows: list[dict], case: int) -> str:
    """すべての評価器と表示の代表値を並べる。"""
    keys = ['t_sec', 'new_evaluator', 'p1_G_fe', 'p1_S1', 'p1_S3_provisional', 'p1_S3',
            'p1_landing_gfe', 'new_p1', 'p1_display', 'old_p1', 'old_evaluator']
    values = [[float(r['frame'][k]) if r['frame'][k] and k not in ('new_evaluator', 'old_evaluator')
               else r['frame'][k] for k in keys] for r in rows]
    return f'## #{case} 評価器時系列（代表時刻）\n\n'+table(keys, values)


def input_table(rows: list[dict], case: int) -> str:
    """両者の盤面時刻・数量・予告・連鎖・受け・応手を並べる。"""
    values = []
    for row in rows:
        frame, observation = row['frame'], row['observation']
        chains = observation.get('chains', [])
        for i, label in enumerate(('1P', '2P')):
            inp = row['input']['sides'][i]
            chain = next((c for c in reversed(chains) if c['side'] == label), {})
            count = frame[f'{label}_chain_count']
            predicted = chain.get('predicted_chain_count')
            remain = max(0, predicted-int(count)) if predicted and count else None
            history = observation.get('history', [None, None])[i]
            model_queue = history['queue'].tolist() if history else None
            nf = [round(float(frame[f'{label}_NF_ojama_k{k}']))
                  if frame[f'{label}_NF_ojama_k{k}'] else None for k in range(1, 6)]
            values.append([float(frame['t_sec']), label, inp['state'], frame[f'{label}_confirmed_sec'],
                inp['colors'], inp['queue'], model_queue, count, predicted, remain, chain.get('predicted_final_score'),
                frame[f'{label}_formula_total'], frame[f'{label}_forecast'], frame[f'{label}_incoming'],
                row['landing'].get('death_incoming', [None, None])[i], frame[f'{label}_hands'], nf,
                frame[f'{label}_counter_margin']])
    return f'## #{case} 入力時系列（1P/2P別）\n\n'+table(['秒','側','状態','盤面確認秒','色個数',
        '現観測NEXT4色(0欠測)','履歴NEXT4色','通知段','完走予測段','残段参考','完走予測点','帰属後式累積','予告','確率用受け',
        '証明用受け','猶予手数','NF_k1〜5個','counter_margin個'], values)


def attribution_table() -> str:
    """新旧の主要特徴寄与を全代表時刻で比較する。"""
    new = json.loads((OUT / 'gfe_tree_path_contributions.json').read_text())
    old = json.loads((OUT / 'gfe_attribution.json').read_text())
    names = ['board_color_puyo_total','board_puyo_total','diff_board_color_puyo_total',
        'diff_board_puyo_total','current_max_chain','diff_current_max_chain','diff_column_bumpiness',
        'm0_logit','second_chain_potential']
    rows = []
    for left, right in zip(new, old):
        nf = {r['name']: r for r in left['features']}
        of = {r['name']: r for r in right['features']}
        for name in names:
            rows.append([left['t_sec'], name, nf[name]['value'], nf[name]['contribution_logit'],
                         of[name]['lr_logit_contribution']])
        rows.append([left['t_sec'],'基準logit',None,left['baseline_logit'],right['lr_intercept']])
    return '## #5 新旧モデル寄与（logit）\n\n'+table(['秒','特徴','値','新・木経路寄与','旧・LR寄与'],rows)


def main() -> None:
    """診断結果を網羅するレポートとCSVを作成する。"""
    resources()
    summaries = json.loads((OUT / 'summaries.json').read_text())
    text = PREFIX+INTRO+'\n\n'
    text += table(['件','開始秒','終了秒','フレーム','新評価器内訳','旧評価器内訳'],
        [[s[k] for k in ('case','first','last','frames','evaluators','old_evaluators')] for s in summaries])
    for case in TIMES:
        rows = full_tables(case)
        text += evaluation_table(rows, case)+input_table(rows, case)
    text += '## 確定判定の切替全件\n\n'+table(['件','切替秒','選択評価器','E35状態'],
        [[s['case'],v['t_sec'],v['evaluator'],v['reasons']] for s in summaries for v in s['transitions']])
    text += attribution_table()+FINDINGS
    text += '\n## 修正候補（採否なし）\n\n'+table(['対象','file:line','候補','副作用・未検証点'], CANDIDATES)
    text += '\n## 成果物\n\n全フレーム: case4_full.csv、case5_full.csv、case6_full.csv、case7_full.csv。\n'
    text += '原入力: set*_inputs.json。全安全条件: case*_timeline.jsonl。寄与全列: gfe_tree_path_contributions.json、gfe_attribution.json、case*_support.json。\n'
    (OUT / 'REPORT.md').write_text(text, encoding='utf-8')
    save('report_resources.json', resources())


if __name__ == '__main__':
    main()
