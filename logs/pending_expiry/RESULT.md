# 答え合わせ pending の連鎖失効 (--verification-pending-chain-expiry) 結果 (2026-09-30)

## 実装 (既定OFF)
- `src/recognition_pipeline.py`: `_update_chain_estimate_verification` の先頭で、pending が有り、かつ
  今フレームの状態が `VERIFY_EXPIRY_STATES = (CHAIN, GRAVITY_SETTLE)` (= pending 作成後に同じ側で
  新しい連鎖が始まった) なら pending を破棄。TSUMO_FALL / OJAMA_FALL 等の連鎖以外の非STABLEでは失効させない。
  1本目の連鎖後の通常補正 (STABLE 5枚) は不変。監査カウンタ `verification_pending_expired_count`。
- 「履歴から連鎖前フレームを除外」案より狭く単純な「pending 全体を失効」を採用 (連鎖を跨いだ pending は
  期待盤面 `expected` 自体も連鎖前で古いため、履歴だけ削っても不一致判定が誤る)。
- 配線: `RecognitionPipeline.__init__` / `load_default` の `enable_verification_pending_chain_expiry`、
  `scripts/visualize_advantage_overlay.py` の `--verification-pending-chain-expiry`。
  `src/production_config.py` は未編集 (本番採用はまだ)。

## 段1: 単体テスト (tests/test_verification_pending_expiry.py, 7件)
OFFで q第14型 (連鎖前3+連鎖後2) の幻おじゃま書込を再現 / ONで書込なし・失効1 / 単発連鎖は ON/OFF とも補正 /
連鎖以外の非STABLEでは失効しない / 既定OFF / load_default 既定 False。
既存 324 件 (recognition_pipeline / next_recolor_pair_guard / exchange_event_production を含む) 全パス。

## 段3相当: 4動画の影子測定 (認識は本番構成+`--next-recolor-pair-guard` ON、フラグOFFで再生し、各答え合わせに「ONなら失効していたか」を併記)
真値 = 書込 0.1〜0.5 秒後 13フレームの「生CNN==生HSV (不明除く、空も可)」最頻値 (3票以上)。row0 は除外。
スクリプト: `scripts/measure_pending_expiry.py`、生ログ `verif_*.jsonl`、集計 `measure.json`。

| 動画 | 答え合わせ総数 | 補正発生 | 失効(ON) | 補正が失効した数 | 書込セル | 真値あり | OFF正 | ON正 | 書込前正 | ON悪化 | ON改善 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q (0-900s) | 58 | 7 | 3 | 1 | 48 | 45 | 23 | 32 | 14 | **0** | 9 |
| zenchi 2580-3427s | 61 | 11 | 0 | 0 | 349 | 317 | 279 | 279 | 5 | **0** | 0 |
| fcXG83vInDY (0-900s) | 58 | 19 | 6 | 2 | 462 | 401 | 270 | 334 | 103 | **0** | 64 |
| mia8KCjr52g (0-900s) | 71 | 39 | 1 | 0 | 799 | 704 | 534 | 534 | 106 | **0** | 0 |
| 合計 | 248 | 76 | 10 | 3 | 1,658 | 1,467 | 1,106 | 1,179 | 228 | **0** | 73 |

- 失効した補正3件の書込セルは真値あり73セル: 書込が正しかった 0 / 書込前が正しかった 73 (q 9/9、fc 64/64)。
  失効させた補正は全部が誤り = 「連鎖後の正当な補正」を潰した例は 0。
- q: 881.967 秒 1P の幻おじゃま (1,5)(2,5)(3,5)=9 を含む12セル補正が失効対象 (診断書の根因B と一致、9セルとも書込前=正)。
- fc: 535.867 秒 1P の大量書込 (真値0の空セルへ色ぷよを書く) ほか。
- 真値なし (票不足): q 3 / fc 61 / mia 95 / zenchi 32 (母数 1,658 のうち 191 = 11.5%)。
- 制限: 影子なので失効後の下流連鎖 (誤盤面が後続の推論を汚す効果) は測っていない。端から端の確認は全長ゲートで行う。
- 注意: 診断書の「6補正・43セル」は q 900秒の別集計。今回の q は補正7件・書込48 (row0除外前後の差と保存記録の違い、判定は同一基準)。

## 判定
ON は悪化 0 / 改善 73セル (q 9・fc 64)。q第14試合の幻おじゃまは止まる。ただし 884 秒の誤確定の
決定要因は根因A (対整合ガード採用済み) で、本フラグ単独では誤確定の追加低減は限定的 (診断書反事実A)。
本番採用の前に全長ゲート (logs/pending_expiry/launch_full.sh) で q .507567 / zenchi 7,671 / 誤発火0 / 場面2760.38 を確認すること。

## 全長ゲート (未起動・承認待ち)
- 起動: `MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/pending_expiry/launch_full.sh`
- 構成: c65_guard の全長再生成と同一 (collect_e34c の旧コマンド + `--next-recolor-pair-guard --verification-pending-chain-expiry` → E36b 再生 → 採点)。出力は `logs/pending_expiry/{full,e36b_on}`。
- 比較基準: `logs/c65_guard/e36b_on/SUMMARY.json` (q .507567 / zenchi 7,671/8,333 / 誤発火0/37 / 場面2760.38)。
- 見積もり (壁時計、単位=動画1本の処理frame数、c65_guard 実走 04:19→07:04 の実測): 収集 zenchi+review レーン 1h36〜2h00、
  q+mia レーン 1h20〜2h28 → 収集 2h00〜2h30、再生+採点 17〜20分、合計 **2.3〜3.0 時間**。
  今回の影子測定では他ジョブ (WSL 上 scan_terminal_prefilter_b20 3本) 並走で zenchi 単体が約 95 分だった (負荷次第で上振れ)。
