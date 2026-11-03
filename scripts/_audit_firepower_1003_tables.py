"""監査結果の全集計表を日本語Markdownにまとめる。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._audit_firepower_1003_inventory import OUT, resources


def read(name: str) -> pd.DataFrame:
    """集計済みCSVだけを読む。"""
    return pd.read_csv(OUT / f'{name}.csv')


def num(value: Any) -> str:
    """表の丸めは表示のみで、原票は丸めない。"""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return '—' if pd.isna(value) else f'{value:.4g}'
    return str(value).replace('|', '/')


def table(frame: pd.DataFrame) -> str:
    """外部依存なしで表を作る。"""
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    rule = '|' + '|'.join(['---']*len(frame.columns)) + '|'
    lines = ['| ' + ' | '.join(num(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header, rule, *lines])


def feature_table() -> pd.DataFrame:
    """全火力列の欠測率・ゼロ率・中央値・90分位を同じセルに並べる。"""
    train, live = read('training_stats'), read('production_stats')
    columns = train[train.model == 'S3'].column.tolist()
    rows = []
    for col in columns:
        row = {'列': col}
        for model in ('G_fe_R1s', 'G_fe_landing', 'S1', 'S3'):
            left = train[(train.model == ('G_fe_R1s' if model.startswith('G_fe') else model)) & (train.column == col)]
            right = live[(live.model == model) & (live.column == col)]
            if left.empty or right.empty:
                row[model] = '—'
                continue
            a, b = left.iloc[0], right.iloc[0]
            row[model] = (' / '.join(f'{num(v)}→{num(w)}' for v, w in (
                (100*a.missing/a.n, 100*b.missing/b.n), (100*a.zero/a.n, 100*b.zero/b.n),
                (a.p50, b.p50), (a.p90, b.p90))))
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_table() -> pd.DataFrame:
    """全分位と全評価器を一行に揃える。"""
    data = read('winrate_quantiles')
    rows = []
    for (metric, bucket), group in data.groupby(['metric', 'bin'], sort=False):
        raw = group[group.model == 'output_raw'].iloc[0]
        row = dict(差=metric, 分位=bucket, フレーム=int(raw.n), 秒=raw.seconds,
                   試合=int(raw.games), 実勝率=raw.observed_win)
        row.update({r.model: f'{int(r.n)}/{num(r.observed_win)}/{num(r.model_p1)}' for r in group.itertuples()})
        rows.append(row)
    return pd.DataFrame(rows)


def drop_table() -> pd.DataFrame:
    """独立の認識盤面差。人手真値との精度を名乗らない。"""
    data, rows = read('board_drops'), []
    positive = data.measurable & data.board_delta.between(1, 30)
    for name, part in [('全降下通知', data), ('前後盤面比較可', data[data.measurable]),
                       ('比較可・予測1秒内', data[data.measurable & (data.prediction_age <= 1)]),
                       ('正の盤面増加1〜30個', data[positive]),
                       ('正の盤面増加・予測1秒内', data[positive & (data.prediction_age <= 1)])]:
        delta = part.predicted_single_drop-part.board_delta
        known = delta.dropna()
        q = np.quantile(known.abs(), (.5, .9, .95)) if len(known) else [np.nan]*3
        rows.append(dict(標本=name, 件数=len(part), 比較数=len(known), 増加ゼロ=int((part.board_delta == 0).sum()),
                         負または30超=int(((part.board_delta < 0) | (part.board_delta > 30)).sum()),
                         誤差中央値=q[0], 誤差P90=q[1], 誤差P95=q[2]))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'drop_summary.csv', index=False)
    return result


def importance_table() -> pd.DataFrame:
    """木の訓練内寄与と外部の列別感度を区別する。"""
    gains, permutations = read('s3_tree_gain'), read('s3_permutation')
    gains = gains[gains.fire].copy()
    gains['gain_percent'] = gains.gain_fraction*100
    result = gains.merge(permutations, on='column', how='outer')
    return result[['column', 'gain_percent', 'delta_logloss', 'mean_abs_delta_p']]


def imputation_table() -> pd.DataFrame:
    """補完前の欠測が、0としてモデルへ入った割合を示す。"""
    train, live = read('training_imputation'), read('production_stats')
    result = []
    for col in train.column.unique():
        row = {'列': col}
        for model in ('G_fe_R1s', 'G_fe_landing', 'S1', 'S3'):
            source = 'G_fe_R1s' if model.startswith('G_fe') else model
            a = train[(train.model == source) & (train.column == col)].iloc[0]
            b = live[(live.model == model) & (live.column == col)].iloc[0]
            row[model] = f'{num(100*a.imputed_zero/a.n)}→{num(100*b.imputed_zero/b.impute_known_n)}'
        result.append(row)
    return pd.DataFrame(result)


def main() -> None:
    """詳細原票へのリンクと全集計表を保存する。"""
    resources()
    sections = [('# 火力→勝率監査・全集計表', None),
        ('対象。時刻は元動画絶対秒、時間は30fps換算。', read('population')),
        ('初回予測と観測の差。段数は候補最大、得点は候補平均を含む。OCRは人手真値ではない。', read('error_summary')),
        ('両者CHAINの各側・各時点予測と最終観測の誤差。', read('dynamic_error_summary')),
        ('学習行の母数。148原票から採用144動画。', read('training_populations')),
        ('本番の実推論呼出数。隠し段候補の複数呼出を含む。', read('model_populations')),
        ('全火力列。各セルは「欠測率% / ゼロ率% / P50 / P90」、各値は学習→本番。ゼロ率は補完率ではない。', feature_table()),
        ('D列の欠測から0への補完率%（学習→本番）。補完前CSVと全D列照合済み。', imputation_table()),
        ('同時CHAINの差別五分位。+は1P側。実勝率はフレーム加重。各モデル欄は比較母数/その標本の実勝率/平均確率。', quantile_table()),
        ('有利側の平滑後勝率が0.8未満の時間。差30個は代理条件で、死亡の証明ではない。', read('weak_probability')),
        ('弱い表示の理由。上位12組、全文はweak_reasons.csv。', read('weak_reasons').head(12)),
        ('発火前候補の監査理由。', read('prefire_reasons')),
        ('降下前後の独立な盤面差との比較。計算された会計消化数を正解にしていない。', drop_table()),
        ('S3′全火力列の木の分岐利得%とpermutation感度。ΔLLは正が有用、負は入替で改善。', importance_table()),
        ('指摘7窓。由来ごとの時間と確率範囲。', read('scenes')),
        ('予測得点と帰属済み式累計の差が最大の15発火。', read('largest_errors')[
            ['set', 'match', 'side', 'trigger', 'pred_count', 'obs_count', 'pred_score', 'formula_total', 'abs_score_error']])]
    text = '\n\n'.join(title if data is None else f'{title}\n\n{table(data)}' for title, data in sections)
    (OUT / 'TABLES.md').write_text(text + '\n', encoding='utf-8')
    print('tables complete', flush=True)


if __name__ == '__main__':
    main()
