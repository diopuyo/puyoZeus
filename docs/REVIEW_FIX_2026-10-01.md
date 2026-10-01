# 2026-10-01 レビュー4件の修正

対象: `claude/exchange-event-eval-20260926`。`src/production_config.py` は変更しない。

## 修正

- 記録版1の `update` 行に `fallback_adv` / `fallback_p1` を追加。
  更新入力は従来と同じ時点で複製し、描画直前に旧評価値を加えて保存する。
  間引き・暖機中も全更新を記録する。従来の `display` 行と版番号は維持する。
  再生は追加フィールドがあれば使用し、なければ従来の代替値を使用する。
- `hidden_scenario_cap` は `None` または正整数のみ許可する。
  APIは不正値に `ValueError`、描画・再生CLIは引数エラーを返す。
- 複数着弾証明のキャッシュキーに実効探索上限を追加する。
- `generate()` の平滑化引数を末尾へ移す。基点 `3802ab7` の全105位置引数名を
  `tests/fixtures/generate_positional_3802ab7.json` に保存し、順序をテストで固定する。

## 単体・接続テスト

指定WSL venv、`PYTHONPATH=.`、nice 10で実行。今回起動するPythonは最大2プロセス。

関連10ファイル: **157 passed / 1 skipped**。
skipは `E2_SHORT_ARTIFACTS` が必要な既存の実動画成果物テスト。

- 未確定盤面の3更新 `0 → 40 → 40`、表示2フレーム間隔で、
  新記録の描画・再生は最終17.5かつdisplay/eventsがバイト一致。
  暖機で先頭表示を省く条件も一致する。
- 同じ記録から追加フィールドを除くと従来の10.0を再現する。
- 遅延保存中に元盤面・イベントを変更しても、記録入力が変わらない。
- 全列下10段おじゃま、NEXT `(1,2,3,4)`、受け30、1手、経過0秒で、
  上限1の `node_limit / nodes=2` から、上限1000では
  `all_responses_dead / nodes=42` へ変わり、キャッシュ消去後の結果とも一致する。

## 再現用コマンド

5記録それぞれに対し、設定指定は `--production-exchange-event` のみ。
`--compare` は保存済み本番登録時出力のdisplay/events全バイトを検証する。

```bash
PYTHONPATH=. nice -n 10 /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python \
  -m scripts.replay_exchange_event_20260926 \
  logs/pending_expiry/full/records/<source>.jsonl.gz \
  --out logs/review_fix_20261001/production/<source> \
  --production-exchange-event \
  --compare /mnt/d/puyo_analyzer/wt_switch/logs/switch_smoothing/cli_prod/<source>
```

作業ディレクトリはこのworktree、Windowsから `wsl -d Ubuntu` 経由で実行する。
再生後の照合・固定ラベル再採点は `scripts/verify_exchange_review_20261001.py` で行う。

## 5記録の実測結果

`q_7gc4TgFig / fcXG83vInDY / mia8KCjr52g / review / zenchi` の全5記録を再生。
更新114,146フレーム、表示112,346行。本番登録時の `cli_prod` と
**display.npz / events.jsonl が5/5バイト一致**、diagnosticsも5/5一致した。
E36bの平滑前5列も5/5でdtype・全バイト一致した。

- q: **0.5075669993215693**（6,526フレーム / 4試合）。E36bと一致。
- 誤確定: **0/37**、未ラベル0。E36bと一致。
- zenchi: 本番平滑化ON **7,670/8,333**。本番登録時と一致。
  E36b OFFは **7,671/8,333**（下記注意参照）。
- `src/production_config.py` のSHA-256:
  `6f782e98330c58ee239d6d4bd47d75ce18e6a291fe59b3f79cb60d1a519d90d9`（変更なし）。

全ファイルのSHA-256と採点結果は `docs/REVIEW_FIX_2026-10-01.json` に保存した。
今回の修正で動画認識全体を再実行したとは主張しない。

## 採点条件の注意

要求されたzenchi 7,671はE36bの平滑化OFFの値。
本番登録済み平滑化ONは7,670/8,333であり、両者は同じ値ではない。
`logs/switch_smoothing/RESULT.md` にある採用時の事後例外と一致する。
今回の再採点でもOFF 7,671、ON 7,670を確認した。
本番登録時のON出力とのバイト一致を維持し、この既存差を変更で埋めない。
