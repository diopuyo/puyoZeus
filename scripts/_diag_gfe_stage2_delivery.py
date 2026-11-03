"""依頼に対応する全表を、会話にも転載しやすい形へ整える。"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scripts._diag_gfe_stage2_analyze import OUT
from scripts._diag_gfe_stage2_report import table, COMMIT
from scripts.gfe_phase_stage1_20261002 import memory_guard

TIME_LABELS = ('0–10','10–20','20–30','30–45','45–60','60–90','90–120','120–180','180–300','300以上')


def games_table() -> pd.DataFrame:
    """両セットの115試合を1行ずつ対応表示する。"""
    source = pd.read_csv(OUT/'games.csv').set_index(['set','game'])
    rows = []
    for game in range(1,59):
        result = {'試合':game}
        for name in ('set1','set2'):
            for column,label in [('baseline','本番'),('V3','V3'),('difference','差'),('contribution','寄与')]:
                result[name+' '+label] = source.loc[(name,game),column] if (name,game) in source.index else np.nan
        rows.append(result)
    return pd.DataFrame(rows)


def calibrated(name: str) -> pd.DataFrame:
    """本番とV3の確率帯別実勝率を並べる。"""
    source = pd.read_csv(OUT/(name+'.csv')).set_index(['model','bin'])
    rows = []
    for band in range(10):
        row = {'予測確率帯':f'{band/10:.1f}–{(band+1)/10:.1f}'}
        for model in ('baseline','V3'):
            for column,label in [('frames','行数'),('prediction','予測平均'),('outcome','実勝率')]:
                row[model+' '+label] = source.loc[(model,band),column]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """全表の短縮版も行の省略なしで保存する。"""
    os.nice(max(0,19-os.nice(0)))
    memory_guard()
    feature = pd.read_csv(OUT/'feature_comparison.csv')
    route = pd.read_csv(OUT/'routes.csv').merge(pd.read_csv(OUT/'route_intervals.csv')[['route','lower','upper']],on='route')
    route = route[['route','frames','mass','baseline','V3','difference','contribution','lower','upper']]
    absent = pd.DataFrame([dict(route='S1′（表示なし）',frames=0,mass=0,baseline=np.nan,
        V3=np.nan,difference=np.nan,contribution=0,lower=0,upper=0)])
    route = pd.concat([route,absent],ignore_index=True)
    times = pd.read_csv(OUT/'time.csv')
    times['time_bin'] = times.time_bin.map(dict(enumerate(TIME_LABELS)))
    mismatches = pd.read_csv(OUT/'same_scene_values.csv')
    mismatches = mismatches[mismatches.mismatch == True][['row_id','column','train','serving','own_frame','opponent_frame']]
    sections = [('表示経路別',route), ('試合別（全115試合）',games_table()),
        ('経過秒別',times), ('54列・59列の入力比較（和集合66列、率は%）',feature),
        ('同一保存場面での不一致（全3セル）',mismatches),
        ('入力用途別の分布',pd.read_csv(OUT/'purpose_comparison.csv')),
        ('G_fe表示の確率差',pd.read_csv(OUT/'probability_difference.csv')),
        ('G_fe表示の較正',calibrated('calibration')), ('S3_landing表示の較正',calibrated('landing_calibration')),
        ('確率・識別・較正の補助指標',pd.read_csv(OUT/'probability_metrics.csv')),
        ('CV再重み付け',pd.read_csv(OUT/'cv_reweighted.csv')),
        ('CVの細かい時刻分布再重み付け',pd.read_csv(OUT/'cv_fine_reweighted.csv')),
        ('CVの保持された実入力時刻による再重み付け',pd.read_csv(OUT/'cv_input_time_reweighted.csv')),
        ('原因候補・証拠・不確実性',pd.read_csv(OUT/'hypotheses.csv')),
        ('修正案（file:line）',pd.read_csv(OUT/'proposals.csv'))]
    lines = [f'段2コミット: {COMMIT}', '',
        '診断のみ。表示損失は全115試合のdisplay_p1（平滑前の定義）。入力は13区間全再生とs2先頭約66.7秒の標本。',
        '表は丸め表示。寄与は115試合等重みの全体M_time差への寄与。S3系はS3′使用構成の記録タグ。', '']
    for title,frame in sections:
        lines += [title, '', table(frame), '']
    (OUT/'FINAL_TABLES.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    main()
