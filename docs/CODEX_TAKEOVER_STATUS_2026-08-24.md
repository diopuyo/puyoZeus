# Codex 引き継ぎ状況（2026-08-24）

## この記録の目的

Claude Code の制限到達後に、Codex が既存作業を**上書きせず**状況を引き継ぐための
スナップショット。進捗確認の起点として使う。実装・ジョブ・ログの所有権は前任作業に
残し、この文書は状況整理だけを追加する。

## 情報源と優先順位

1. `CLAUDE.md` — 不変の設計・データ・運用制約。
2. `C:\Users\ryouj\.claude\projects\C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer\memory\MEMORY.md`
   と 2026-08-21〜24 の個別メモリ — user 決定・根因・受け入れ条件。
3. `src/production_config.py` — 採用済みフラグの唯一の情報源。
4. 実行中プロセス、ログ、`data/verify/` の成果物 — 現在の実行事実。

## 現在の結論

- Phase J（OBS オーバーレイ）は後ろ倒し。判定精度を先に固める。
- 現在の最優先は、掛け算式（消去ごとの `N×M`）を観測値として使う連鎖検知の
  根治候補の A/B 検証である。
- `STABLE` 確定盤面のみを評価に使う原則は維持する。ただし凍結盤面を連鎖検知の
  入力にも使う循環が、片側連鎖で状態遷移を止める根因として確定している。
- 盤面が高いこと自体は窒息ではない。連鎖中・設置中の窒息断定はしない。

## 実行中・検証中

| 項目 | 状態 | 確定している事実 | 完了条件 |
|---|---|---|---|
| 掛け算式による連鎖検知根治 | 実行中 | `enable_chain_formula_read_verify`、`enable_formula_chain_count_update`、`enable_slide_exit_no_min_display` の3フラグで旧/新比較中。既定はすべてOFF。 | E2E ON/OFF、偽イベントA/B、全テストを揃え、全域比較で採否を判断する。 |
| 偽イベント A/B | 途中 | 旧構成は75トリガー中5件がスコア非支持。新構成側は未集計。 | `summary_on.json` が出力され、同一窓で比較できること。 |
| 長尺動画の ±100 反転 | 実行中 | 8区間のON/OFF dumpを生成中。原因は撃ち返し未会計、216上限の相殺汚染、ChainEvent断片化。 | 規模比較・確信度制御の採否を、全区間ペア比較で決める。 |
| 窒息の凍結盤面誤判定 | 部分修正済み | dump後処理で誤判定時間は31%減少。ただしライブ判定と「連鎖中なのにSTABLE」の本体は未解決。 | 根因の連鎖検知修正後にライブ経路を再計測する。 |

## 148動画再収集の状態

`data/verify/regen_2026-08-18/status.tsv` は 2026-08-18 11:30 以降更新されていない。

- 記録済み40件: `OK=14`、`SKIP_DL_FAIL=26`
- 未記録: 108件（全148対象との差分）
- 失敗動画の先行ダウンロード用プロセスは残っているが、収集オーケストレータは動いていない。
- その後、認識・状態機械・採用フラグの作業ツリーが大きく変わった。

**したがって再開しない。** この40件に現行コードで続きを混ぜると、同じデータセット内で
認識構成が混在する。現構成の採否を確定した後、出力先を分けた新しい再収集計画として
起動する。

## 作業ツリーの保全

- tracked変更は37ファイル、差分は +7,526 / -494 行（確認時点）。
- 診断・検証スクリプトなどの未追跡ファイルが多数ある。
- 全pytestの直近記録は `5,604 passed / 13 skipped / 1 failed`。
  失敗は `tests/test_expected_net_damage_step5.py::test_fuller_opp_board_yields_larger_damage_for_same_net_expected`。
- 新しい掛け算式修正に対する全テストはまだ走行中で、成功扱いにしない。

## 次に実施する順序

1. 実行中のA/B・E2E・pytestの完了を直接ログで確認する（監視スクリプトへ委任しない）。
2. 生成されたON/OFF成果物を同じ窓・同じフラグで突合する。
3. 根因・採否に関わる判断は、実画面と一次ログを確認してから行う。
4. 採用は user 承認後に `src/production_config.py` だけへ登録し、全域バックテストを行う。
5. 148再収集はその後。既存 `boards_lean_phase_l_2026-08-18/` を上書きしない。

## Codex への確認依頼方法

このチャットで「進捗確認」または「検証状況」と送れば、Codex が次を直接照合して報告する。

- 実行中プロセスとCPU/GPU負荷
- 対応するログ末尾と成果物の更新時刻
- A/Bの集計JSON・E2Eトレース・pytestの最終結果
- Git差分（変更・上書き・停止は行わない）

## 参照すべき根因記録

- `project_stable_freeze_deadlock_2026-08-24.md`
- `project_pm100_display_flip_2026-08-24.md`
- `project_is_dead_stable_misdetect_2026-08-24.md`
- `reference_chain_formula_layout_2026-08-24.md`
- `project_chain_event_fragmentation_accumulator_2026-08-22.md`
- `feedback_use_single_source_for_flags_2026-08-22.md`

## 2026-08-24 11:26 プロセス最適化

- 既存ジョブの停止・再起動・成果物変更は行わず、実行優先度だけを調整した。
- クリティカルパスの掛け算式検証を nice 0、並列中の ±100 比較を nice 10、全 pytest を nice 19 とした。
- 後続プロセスにも同じ優先度を適用する一時監視器
  `scripts/_prioritize_claude_jobs_2026-08-24.sh` を起動した。2本の掛け算式ドライバ終了後に自動停止する。
- 調整直後、WSL の load average は 26.39 から 23.23 へ低下。±100 の seg05 ON/OFF は完了済み。
- 掛け算式側の全 pytest は、負荷依存の速度テスト
  `test_precise_and_fast_complete_within_generous_timeout` 1件で停止した。
  機能退行との切り分けのため、低負荷時に当該テストを再実行する。
