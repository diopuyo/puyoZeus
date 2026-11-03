"""診断の全表を日本語Markdownへまとめる。採否は記載しない。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scripts._diag_gfe_stage2_analyze import OUT
from scripts.gfe_phase_stage1_20261002 import memory_guard, save_json
from src.exchange_event_features import D_COLUMNS, G_COLUMNS
from src.exchange_event_user_phase import USER_COLUMNS

COMMIT = '378b705c220c1feb36ed91bcedb67232ab3f8d81'
PERCENT = 100


def cell(value: Any) -> str:
    """欠測を明示し、Markdownの区切り文字を無害化する。"""
    if isinstance(value, (float,np.floating)):
        return '—' if not np.isfinite(value) else f'{value:.6g}'
    return str(value).replace('|', '/').replace('\n',' ')


def table(frame: pd.DataFrame) -> str:
    """全行を省略せず表へ変換する。"""
    lines = ['|'+'|'.join(map(str,frame.columns))+'|', '|'+ '|'.join(['---']*len(frame.columns))+'|']
    lines += ['|'+'|'.join(cell(v) for v in row)+'|' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def feature_table() -> pd.DataFrame:
    """54列と59列の和集合66列を、同じ列名で比較する。"""
    train = pd.read_csv(OUT/'input_columns.csv')
    actual = pd.read_csv(OUT/'all_actual_columns.csv')
    traw = pd.read_csv(OUT/'training_raw_columns.csv').set_index('column')
    araw = actual[actual.dataset == 'actual_raw'].set_index('column')
    scenes = pd.read_csv(OUT/'same_scene_columns.csv').set_index('column')
    result = []
    for name in dict.fromkeys((*USER_COLUMNS,*G_COLUMNS)):
        suffix = '54' if name in USER_COLUMNS else '59'
        a = train[(train.dataset == 'train'+suffix) & (train.column == name)].iloc[0]
        b = actual[(actual.dataset == 'actual'+suffix) & (actual.column == name)].iloc[0]
        def distribution(r: pd.Series) -> str:
            return '/'.join(f'{r[k]:.4g}' for k in ('min','p01','p50','p99','max'))
        result.append(dict(列=name, 入力=('両方' if name in USER_COLUMNS and name in G_COLUMNS else suffix),
            学習NaN率=traw.loc[name,'nan']*PERCENT if name in traw.index else 0,
            提供NaN率=araw.loc[name,'nan']*PERCENT if name in araw.index else 0,
            学習0率=a.zero*PERCENT, 提供0率=b.zero*PERCENT,
            学習範囲分位=distribution(a), 提供範囲分位=distribution(b),
            提供定数=b.constant,
            同場面不一致=f'{int(scenes.loc[name,"mismatches"])}/30' if name in scenes.index else '対象外'))
    frame = pd.DataFrame(result)
    frame.to_csv(OUT/'feature_comparison.csv', index=False)
    return frame


def calibration_summary() -> pd.DataFrame:
    """確率帯別実勝率に加え、同じ試合重みでECEを計算する。"""
    results = []
    for route, filename in [('G_fe','calibration'),('S3_landing','landing_calibration'),
                             ('着弾内部G_fe','landing_raw_calibration')]:
        frame = pd.read_csv(OUT/f'{filename}.csv')
        for name, g in frame.groupby('model'):
            results.append(dict(route=route, model=name,
                ECE=np.dot(np.abs(g.prediction-g.outcome),g.mass)/g.mass.sum(),
                prediction=np.average(g.prediction,weights=g.mass),
                actual_win_rate=np.average(g.outcome,weights=g.mass)))
    pd.DataFrame(results).to_csv(OUT/'calibration_summary.csv',index=False)
    return pd.DataFrame(results)


def purpose_comparison() -> pd.DataFrame:
    """通常静止と仮想着弾の入力分布の差を同じ指標で示す。"""
    train = pd.read_csv(OUT/'input_columns.csv')
    actual = pd.read_csv(OUT/'all_actual_columns.csv')
    data = pd.concat([train,actual]).set_index(['dataset','column'])
    rows = []
    measures = [('呼出し/学習行数','board_puyo_total','rows',False),
        ('盤面総量・中央値','board_puyo_total','p50',False),
        ('おじゃま量>0の率(%)','board_ojama_count','zero',True),
        ('saturation>0の率(%)','saturation_chain_upper','zero',True),
        ('m0_logit p01','m0_logit','p01',False),('m0_logit中央値','m0_logit','p50',False),
        ('m0_logit p99','m0_logit','p99',False)]
    for label,column,stat,complement in measures:
        row = {'指標':label}
        for dataset,title in [('train54','学習'),('actual__evaluate','通常静止'),('actual__landing_gfe','仮想着弾')]:
            value = data.loc[(dataset,column),stat]
            row[title] = (1-value)*PERCENT if complement else value
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'purpose_comparison.csv',index=False)
    return result


def hypotheses() -> pd.DataFrame:
    """確定した現象と原因について未検証の点を区別する。"""
    rows = [
        ['仮想着弾G_feへの置換波及', 'G_fe表示寄与−0.003852、S3_landing寄与+0.019093。78,407行の表示・イベントが一致しbase_p1不変。内部G_fe LL 0.720904→0.894734。',
         '悪化経路は特定。入力のどの列・相互作用が原因かは、列介入を行っていないため未確定。'],
        ['着弾経路での過度に極端な確率', 'V3内部G_fe<0.1の全体寄与+0.014845、>=0.9は+0.005081。V3着弾表示0–0.1帯の予測平均0.063594に対し実勝率0.305579。',
         '内部G_feのAUCも0.704781→0.699537と低下。過信だけでなく識別の差もあり、仮想盤面の目標差・非線形性・合成規則は未分離。'],
        ['54列への変更だけで説明できるか', '同じ54列のV1b対本番の着弾寄与は+0.001294。V3対V1bでは通常G_fe寄与−0.006962、着弾寄与+0.017798。',
         '入力54列が同じでも着弾への適用結果が違う。モデル種の差は確認できるが、特定の分岐・列への因果帰属は未測定。'],
        ['仮想着弾の入力分布', '盤面総量中央値は学習0.4444、通常静止0.4167、仮想着弾0.8472。saturation非0率は0.0928%、0.1166%、3.2151%。',
         '呼出し標本の分布差。候補・ウォームアップも含み、各列を同じ分布へ介入した場合の損失は未測定。'],
        ['入力欠測・機能の差', '学習会計4列は43.8–44.1%欠測を0補完。提供は全消し2列が0固定。相手欠測6,689行は学習だけにありUフラグで区別。saturationは学習も提供もNaN→0。',
         '差は確認できるが、この差だけによるΔLLは未測定。saturation単独の原因認定はできない。'],
        ['M0入力の取り違え', '元記録c1のm0_logit=0率59.48%、実再生は1.47%。段2は記録M0をそのまま使わず再推論。実境界入力を取り直した。',
         '元記録の54.24%という全体0率は段2の分布として使用不可。OOF M0と最終M0のモデル差は別途残る。'],
        ['表示時刻・試合重みの相違', '試合等重み−0.010554。保持入力の経過秒−0.004967、経過秒×U・1P限定−0.005163。85,691表示行の入力確率照合済み。符号逆転は再現しない。',
         '保持時間とUは合わせたが、D46列・仮想盤面・経路選択まで再現したCVではない。s2残部の保持入力は標本外。'],
        ['同場面の特徴生成版差', 'video_29の30場面×40列で1,197セル一致。diff_dig_resistanceが1、diff_ukeyasusaが2セル不一致。',
         '旧CSVは2026-08-14、現行掘り耐性には08-25の決定的seed修正。旧乱数実現・版差の寄与未確定。会計6列・M0は同場面一致の対象外。'],
        ['評価動画の構成差', '段1は144動画・7,686試合、段2は同じzenchi動画の115試合。表示経路の遷移は全211,012行で一致。',
         '115試合のbootstrapは動画を独立再標本化していない。対戦者・認識版・動画固有差と仮想盤面差を分離できない。'],
    ]
    result = pd.DataFrame(rows,columns=['原因候補','確かめた証拠','残る不確実性'])
    result.to_csv(OUT/'hypotheses.csv',index=False)
    return result


def proposals() -> pd.DataFrame:
    """コードを変更せず、修正を行うなら触れる場所と検証内容を示す。"""
    rows = [
        ['用途ごとにG_feを選択', 'src/exchange_event_evaluator.py:143 / src/exchange_event_landing.py:377',
         '通常静止と仮想着弾の用途をoptional引数または外部wrapperで識別し、同じ置換フラグが全StaticInputへ波及する構造を分ける。'],
        ['仮想着弾・合成を学習と検証へ含める', 'scripts/gfe_phase_stage2_train_20261002.py:30 / src/exchange_event_landing.py:422 / src/exchange_hidden_row_probability.py:100',
         '仮想着弾入力と候補重み・表示保持を記録し、動画を分離した予測で用途別モデルまたは合成較正を検証する。今回115試合で係数を調整しない。'],
        ['欠測・全消し・会計の入力契約を統一', 'scripts/visualize_advantage_overlay.py:3289 / src/exchange_event_user_phase.py:35 / scripts/gfe_phase_stage1_20261002.py:85',
         '0固定の全消しと会計再同期の差を共通変換へ集約し、補完前の欠測を保持する。列追加時はモデル版・54/59列契約も更新する。'],
        ['特徴計算版を固定して保存資産を再生成', 'src/indicators_v2.py:841 / scripts/build_labeled_win_from_npz.py:544',
         'seed・ghost rule・native版をmanifestへ記録し、同じ保存盤面で学習と提供の一致を確認してから特徴を再生成する。'],
        ['実推論入力の記録を追加', 'src/exchange_event_record.py:157 / scripts/replay_exchange_event_20260926.py:19',
         'キャッシュ後の実54/59列、M0最終予測、呼出用途、盤面キー、時刻、モデルhashを診断記録へ残す。旧static_callのM0と区別する。'],
        ['CVと表示評価の対象をそろえる', 'scripts/train_exchange_event_models_20260926.py:123 / scripts/gfe_phase_stage1_report_20261002.py:42',
         '静止行LLに加え、保持・仮想着弾・S3合成を含む動画分離の再生予測を保存し、試合等重みの経路別損失を併記する。'],
    ]
    result = pd.DataFrame(rows,columns=['修正案','file:line','内容'])
    result.to_csv(OUT/'proposals.csv',index=False)
    return result


def write_report() -> None:
    """依頼の全表と補助表を省略なしで納品する。"""
    features, cal, causes, fixes = feature_table(), calibration_summary(), hypotheses(), proposals()
    verification = json.loads((OUT/'VERIFICATION.json').read_text())
    lines = ['# G_fe段2 診断（合否外）', '',
        '悪化は通常のG_fe表示ではなく、S3_landing内部の仮想着弾G_feに集中した。'
        '通常G_feの改善と着弾経路の悪化を足すと、保存済みΔM_time +0.015240692に一致する。'
        '時刻の再重み付けだけではCVの改善から悪化への逆転を再現しない。', '',
        f'段2コミット: `{COMMIT}`。関連87テスト通過、54再生独立検算済み。', '',
        '入力取得: 13区間は全再生・表示とイベントのバイト一致、s2は先頭2,000更新'
        '（約66.7秒、ウォームアップを含む）の部分再生・表示全列の接頭一致。'
        '入力分布の表はこの実呼出し標本を用い、s2残部を代表すると仮定しない。'
        '表示損失の分解・較正はs2全体を含む115試合211,012表示行。', '',
        f'取得・検証: {json.dumps(verification, ensure_ascii=False)}', '',
        '計算はnice 19。再生の各更新で空き11GiB以上を要求（指定6GiBに5GiBの余裕）。'
        '他worktreeは読取り専用、本番設定・既存再生・モデルは変更していない。', '',
        'LLとM_timeは小さいほどよい。寄与は「区分の条件付きΔLL×115試合等重みでの区分占有率」。'
        '表内の率は特記のない限り0–1。入力表のNaN率・0率だけ%で表示。', '']
    sections = [('入力66列（54/59の和集合）', features), ('較正の要約', cal),
                ('原因候補・証拠・不確実性', causes), ('修正案（実装なし）', fixes)]
    for title, frame in sections:
        lines += ['## '+title, '', table(frame), '']
    lines += ['入力範囲分位はmin / p01 / median / p99 / max。NaN率はD補完前、'
        '0率はモデル入力。モデル入力54/59列のNaN率はいずれも0。'
        'p25/p75を含む全分位は後半の全列原表に記載。', '']
    lines += ['学習分布は段2最終学習と同じrows.A（seed0外側OOF）を使用。59列の経過閾値は'
        '本番manifestの値。実入力分布はモデル呼出し等重みで、表示保持重みとは区別する。'
        'ウォームアップ・非表示の仮想候補も含む。record_*表は元収集器のstatic_callであり、'
        '段2のM0分布を示す表ではない。', '']
    for path in sorted(OUT.glob('*.csv')):
        if path.stem in ('landing_display_detail','same_scene_values','video29_source_rows',
                         'feature_comparison','hypotheses','proposals','calibration_summary'):
            continue
        frame = pd.read_csv(path)
        lines += ['## '+path.name, '', table(frame), '']
    lines += ['## 制約', '',
        'CV再重み付けは保持盤面や仮想盤面への介入ではない。相対進行は学習保存行の最終時刻と'
        '公式試合窓の終端という異なる起点情報を含むため、1秒絶対経過の結果を主として併記した。'
        '時間帯は[0,10),[10,20),[20,30),[30,45),[45,60),[60,90),[90,120),[120,180),[180,300),[300,∞)。', '',
        '同場面確認はvideo_29の保存NPZから自側と同時刻以前の最新相手盤面を復元した30場面。'
        '会計・全消し6列と、OOFと最終学習でモデルが異なるM0は同場面比較に含めない。'
        '入力列ごとの因果寄与は未測定であり、修正案の採否・合否はこの診断では扱わない。', '']
    (OUT/'DIAGNOSIS.md').write_text('\n'.join(lines),encoding='utf-8')


def main() -> None:
    """診断が揃った後にのみ最終表を生成する。"""
    os.nice(max(0,19-os.nice(0)))
    memory_guard()
    from scripts._diag_gfe_stage2_cv_input_time import main as input_time
    input_time()
    purpose_comparison()
    write_report()
    from scripts._diag_gfe_stage2_delivery import main as delivery
    delivery()


if __name__ == '__main__':
    main()
