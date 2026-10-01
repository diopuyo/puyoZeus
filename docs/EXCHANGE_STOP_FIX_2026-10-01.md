# 発火入力欠測による評価停止の修正取り込み

## exev

基点 `1193c8f` に `origin/claude/eval-set-zenchi-set1-20261001`
（`3d0673b6d183cbc57a0f9fa7ae3e2d617df31ef2`）をマージ。
物差しのスクリプト・定義・ラベル・記録も含む。

- `weighted_landing` は発火入力がないとき加重なしへ戻る。
- 競合は `docs/agent_coordination/CLAUDE_TO_CODEX.md` の末尾追記のみ。
  発火前最善手 Phase 3/4 と評価セットの記述を両方保持。コード競合なし。
- 関連テストで既存コンストラクタの51行を検出したため、引数宣言の改行だけを
  整理して50行にした。API・処理は不変。
- `src/production_config.py` は変更なし。SHA-256:
  `6f782e98330c58ee239d6d4bd47d75ce18e6a291fe59b3f79cb60d1a519d90d9`。

## 本番再生の一致

`logs/pending_expiry/full/records` の q_7gc4TgFig、fcXG83vInDY、
mia8KCjr52g、review、zenchi を `--production-exchange-event` で再生。
出力: `logs/merge_evalset_20261001/production/<source>`。
比較先: `logs/review_fix_20261001/production/<source>`（平滑化ON）。

全5記録、更新114,146フレーム・表示112,346行が完走。
**display.npz / events.jsonl は5/5でファイル全バイト一致**。
NPZの全列・dtype・値のビット列、diagnosticsの内容も一致。
比較先と新出力の全10ファイルのSHA-256は、`1d2dee4` に含まれる
`docs/REVIEW_FIX_2026-10-01.json` の検証記録と一致する。
新出力のハッシュは `EXCHANGE_STOP_FIX_2026-10-01.json` に保存。

```bash
PYTHONPATH=. /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python \
  -m scripts.replay_exchange_event_20260926 \
  logs/pending_expiry/full/records/<source>.jsonl.gz \
  --out logs/merge_evalset_20261001/production/<source> \
  --production-exchange-event \
  --compare logs/review_fix_20261001/production/<source>
```

## テスト

Windowsから `wsl -d Ubuntu` 経由、指定venv・`PYTHONPATH=.` で実行。
再生は2並列、pytestは1プロセス。本作業が起動したPythonは同時最大3本。

```bash
python -m pytest tests/test_e[0-9]*.py tests/test_exchange_event*.py \
  tests/test_exchange_review_regressions.py tests/test_exchange_display_smoothing.py \
  tests/test_eval_set*.py -q
```

改行整理後: **1,166 passed / 1 skipped**（230.97秒）。
skipは `E2_SHORT_ARTIFACTS` 指定が必要な既存の実動画成果物テスト。
ログ: `logs/merge_evalset_20261001/tests_final.log`。
