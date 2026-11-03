"""採点原票からSET2_E19.mdを作り、入力不変と完走を記録する。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_set2_labels_20261001 import digest
from scripts.eval_set2_score_20261001 import OUT, ROOT, save

CONFIG_SHA256 = '6f782e98330c58ee239d6d4bd47d75ce18e6a291fe59b3f79cb60d1a519d90d9'
EXPECTED_JOBS, MAX_PROCESSES = 27, 8
KIB_PER_GIB, MIN_AVAILABLE_GIB = 1024*1024, 4


def metric_line(name: str, metric: dict) -> str:
    """点推定と単一構成・対差の区間を同じ精度で書く。"""
    values = []
    for variant, label in (('prod','本番'),('e19','E19')):
        row = metric[variant]
        lo, hi = row['ci95']
        values.append(f"{label} {row['value']:.8f} [{lo:.8f}, {hi:.8f}]")
    row = metric['paired']
    lo, hi = row['ci95']
    return f"- {name}: " + ' / '.join(values) + f"。差(E19−本番) **{row['diff']:+.8f} [{lo:+.8f}, {hi:+.8f}]**。"


def receipts() -> dict:
    """全ジョブ終了・測定原票SHA・メモリ下限・本番設定不変を確認する。"""
    done = {p.stem:int(p.read_text().strip()) for p in (OUT/'jobs').glob('*.done')}
    assert len(done) == EXPECTED_JOBS and all(code == 0 for code in done.values()), done
    assert digest(ROOT/'src/production_config.py') == CONFIG_SHA256
    records = {p.name:digest(p) for p in (OUT/'collect/records').glob('*.gz')}
    resources = [line.split('\t') for line in (OUT/'resources.tsv').read_text().splitlines()]
    minimum = min(int(row[1]) for row in resources)
    maximum = max(int(row[2]) for row in resources)
    host_minimum = min(int(row.split('\t')[1]) for row in (OUT/'host_resources.tsv').read_text().splitlines())
    assert minimum >= MIN_AVAILABLE_GIB*KIB_PER_GIB and maximum <= MAX_PROCESSES
    assert host_minimum >= MIN_AVAILABLE_GIB*KIB_PER_GIB
    result = dict(completion_signals=done, record_sha256=records, min_available_kib=minimum,
                  min_host_available_kib=host_minimum, max_processes=maximum,
                  production_config_sha256=CONFIG_SHA256,
                  recovered_completions=[json.loads(p.read_text()) for p in (OUT/'jobs').glob('*.recovered.json')])
    save('RECEIPT.json', result)
    return result


def details(result: dict, receipt: dict) -> str:
    """収集と回帰確認の母数を明記する。"""
    prefix, regression = result['prefix_check'], result['regression']
    deaths, scene = regression['deaths'], regression['scene']
    e19 = result['deaths']['e19']
    false_cases = '、'.join(f"第{r['game']}試合 {r['side']} ({r['first_sec']:.3f}秒)" for r in e19['rows'] if r['false'])
    return f"""## 収集・健全性
- セット1と同じ `scripts/collect_eval_set_20261001.py`。追加したのは任意区間と出力先の引数のみ。
- 本番認識フラグは `placement_reconcile_flags()` から読み、既存コマンドとの照合は全区間一致。
- s0〜s6の7区間、全区間30秒助走。短区間checkはs1と同じ3626秒開始、3745秒終了。
- 短区間のJSON行バイト照合: {prefix['rows']['rows_compared']:,}行、不一致{prefix['rows']['mismatched_rows']}。
  画像窓 {prefix['windows']['check_windows']}件一致。`set2/PREFIX_CHECK.json`。
- 完走: 収集8件・本番/E19再生14件・既存回帰5件、全27件完了確認。
- 収集の起動シェルを実行中に更新したため終了通知が欠落したものは、プロセス消滅・収集器の完了票・
  SHA256・gzip終端complete行で復元。元終了コードは未採取として `jobs/*.recovered.json` に記録。
- 最大{receipt['max_processes']}計算プロセス、記録したavailableメモリ最小 {receipt['min_available_kib']/1024/1024:.3f} GiB。
- Windows側の空き物理メモリ最小 {receipt['min_host_available_kib']/KIB_PER_GIB:.3f} GiB。
- 元動画は開始前から共有dataに存在した資産のため保持（セット1と同じ扱い）。新規動画生成なし。

## 誤確定・回帰
- セット2 E19: 誤った負け確定 **{e19['false']}/{e19['cases']}件**（全{result['games']}試合、試合・側単位）。
- 本番構成も **{result['deaths']['prod']['false']}/{result['deaths']['prod']['cases']}件**。両構成の確定事例は同一={result['deaths']['prod']['rows'] == e19['rows']}。
- 誤確定の内訳（上記58試合での通し番号）: {false_cases}。
- 確定由来の表示行の追加監査: 誤表示 {result['display_deaths']['e19']['false_frames']}行。
- 既存5記録 E19: 誤確定 **{deaths['false']}/{deaths['total']}**、未解決{deaths['unresolved']}。既存37件との対象一致={deaths['same_cases']}。
- 3:00場面: 平滑化ONの実表示 {scene['first_display_sec']:.8f}秒。
  従来採点器と同じEMA再適用の初到達 {scene['legacy_metric_sec']:.8f}秒（門 ≤2766.0）。
- q第14試合の誤確定: **{len(regression['game14_false_times'])}**。
- 詳細原票: `set2/REGRESSION.json`、`set2/RESULT.json`。
"""


def main() -> None:
    """結果だけから合否を記し、本番登録は行わない。"""
    result = json.loads((OUT/'RESULT.json').read_text())
    labels = json.loads(Path(result['labels']).read_text())
    receipt = receipts()
    verdict = '合格' if result['passed'] else '不合格'
    metrics = '\n'.join(metric_line(name,result['metrics'][name]) for name in ('m_time','m_agree'))
    reference = '\n'.join(metric_line(name,result['reference_next30']['metrics'][name]) for name in ('m_time','m_agree'))
    gates = '\n'.join(f"- {name}: {'合格' if passed else '不合格'}" for name,passed in result['gates'].items())
    text = f"""# セット2 E19確認判定 (2026-10-01)

**判定: {verdict}**。本番登録はしていない。`src/production_config.py` は不変。
事前登録原文: exev `docs/agent_coordination/DECISIONS.md` 末尾（複写 `set2/PREREGISTRATION.md`）。

## 対象・勝者・非重複
- video_zenchi_c0BQoMJwwQU、{result['games']}試合。WINパネル全件確認、不一致0・ラベルなし除外0。
- {labels['boundary_note']}
- セット1の採点済み57試合との重複0。E19の実学習148動画をティア表のYouTube IDと照合して混入0。
  撃ち合いモデル学習行144動画にも混入0。`set2/labels.json` に一覧とSHAを記録。
- 証拠画像: `set2/panels_0.jpg`、`set2/panels_1.jpg`、`set2/gap_0.jpg`、`set2/tail_0.jpg`。

## 指標
定義は `METRIC.md`。M_timeは試合等重み、M_agreeは終盤1/3のフレーム加重。
試合単位の同じ復元抽出を両構成へ適用、10,000回・seed 0、percentile 95%区間。
構成は `--production-exchange-event`（平滑化ON）と同構成+`--landing-counter-prob`。
モデル `models/landing_counter_prob_v1` は再学習していない。

{metrics}

## 事前登録の門
M_time差区間上端<0、M_agree差区間下端>−0.01、全試合誤確定0、既存5記録の回帰条件。
{gates}

{details(result,receipt)}
## 範囲の違いによる参考値（採否には使わない）
次の30先だけの57試合（`labels_next30.json`）。主判定は事前に確定した58試合のまま。
{reference}

## 再現と検証
作業ブランチ `claude/eval-set-zenchi-set1-20261001`、開始HEAD `3d0673b`。
指定WSL venvを `PYTHONPATH=.` で使用。依存data/logs/modelsはセット1と同じexevを参照。
ジョブ定義 `set2/job.sh`・`set2/tasks.txt`。採点器 `scripts/eval_set2_score_20261001.py`。
原票SHA・終了通知・元終了コード未採取の監査は `set2/RECEIPT.json`。テスト結果は `set2/tests.log`。
停止不具合修正 `src/exchange_hidden_row_probability.py` は開始HEADに含まれるものを使用。
"""
    (OUT.parent/'SET2_E19.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
