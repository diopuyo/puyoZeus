# rt への評価停止修正の取り込み

基点 `40d337e` に、exev の検証済みマージコミット `989f660` を取り込む。
マージ競合なし。`src/production_config.py` は変更なしで、SHA-256 は
`6f782e98330c58ee239d6d4bd47d75ce18e6a291fe59b3f79cb60d1a519d90d9`。
exev の5記録バイト一致は `EXCHANGE_STOP_FIX_2026-10-01.md` を参照。

## 通知経路の回帰

`tests/test_live_b19_hidden_row.py` に平滑化OFF/ONの2ケースを追加。
`current=None / firing=None` で、隠し段候補が有効な `death_record` が残る
状態を作り、通知入口 `NotificationExchangeOverlay.update()` を呼ぶ。
モデル出力と後続の静的評価への切替だけを固定し、以下の実処理を通す。

1. `src/phase_j/live_notification_eval.py` の通知更新。
2. `ExchangeEventOverlay.update()` から `ExchangeLandingProjection.update()`。
3. `_project()` → `_evaluate()` → `_evaluate_single()` → `weighted_landing()`。
4. 加重なしの確率 `logit_mean(.8, .6)` を記録し、表示を更新。

修正前 `40d337e` では2ケースとも
`ExchangeEndInput.__post_init__` の `ValueError: firingはFiringInputが必要`
で失敗した。ログ: `logs/merge_evalset_20261001/before.log`。
マージ後は同じ2ケースが通過し、通知経路まで修正が届くことを確認した。

## 関連テストの実行条件

`wsl -d Ubuntu` 経由、作業ディレクトリは rt、`PYTHONPATH=.`。
Python: `/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python`。
pytestは1本ずつ実行し、評価ワーカーとresource trackerを含め上限4プロセス内。

対象: `tests/test_live_b1*.py`、`tests/test_e[0-9]*.py`、
`tests/test_exchange_event*.py`、`tests/test_exchange_review_regressions.py`、
`tests/test_exchange_display_smoothing.py`、`tests/test_live_switch_smoothing.py`、
`tests/test_eval_set*.py`。
ログ: `logs/merge_evalset_20261001/tests.log`。

結果: **1,481 passed / 1 skipped**（1,669.09秒）。
新規2ケースは先頭のファイル指定とglobの両方に含まれるため各2回実行
（通過した異なるケースは1,479件）。skipは `E2_SHORT_ARTIFACTS` が必要な既存検査。
ワーカーの起動・再構築を含む `test_live_b1*.py` 全範囲を完走した。

```bash
PYTHONPATH=. /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python \
  -m pytest tests/test_live_b19_hidden_row.py tests/test_live_b1*.py \
  tests/test_e[0-9]*.py tests/test_exchange_event*.py \
  tests/test_exchange_review_regressions.py tests/test_exchange_display_smoothing.py \
  tests/test_live_switch_smoothing.py tests/test_eval_set*.py -q
```
