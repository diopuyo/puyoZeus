# 現在地

## 2026-08-25 08:10 JST — Codex独立レビュー後の最新状態

更新者: Codex

- Claude Codeは07:00にGate 3-2bの作業を終了。対象の実行中ジョブは0件。
- Claude基準: 全pytest 5,914 passed / 13 skipped / 0 failed。
- Codex独立確認: 関連624件pass、py_compile成功、ただし既存テスト未検出の
  P1を3件、P2を2件再現した。
- 交換エピソード会計は既定OFF・本番オーバーレイ未配線のため、現在の表示への
  直接影響はない。
- 掛け算式3フラグは`RECOGNITION_ADOPTED`で本番認識構成ONのため、
  FormulaStepAccumulatorのP2は修正・回帰確認が必要。

### 最新Gate判定

`Gate 3-2b REVIEW NG / Gate 3-2c HOLD`

オーバーレイ配線、交換台帳のproduction登録、Gate 4 A/Bへは進まない。
先に `CODEX_TO_CLAUDE.md` 2026-08-25 08:10節のP1/P2を修正し、
実データの再測定を通す。

### レビューNG

1. `simulate_fallback`のFINALIZEを拒否しても、同じ推定値がFIREとして台帳へ入る。
   52,150点のbaseline-only入力で`finalize_rejected_count=1`なのに
   `net_raw=745`を再現。
2. growthなしの`CHAIN_SETTLED`を即クローズした後に`SCORE_FINALIZE`が来ると、
   1物理連鎖から2つのchain_idが発行される。
3. `pending_uncapped`の純残量差分だけでは、同一フレームの新規受信と相殺の
   総量を分離不能。問題の「相手本線→自分本線」応酬で過小帰属しうる。
4. 大幅下方FINALIZEの再入力で`unreconciled`が二重加算される。
5. FormulaStepAccumulatorがinvalid/幕間でpendingを破棄せず、非連続2観測を
   「連続2フレーム」として確定する。

### 次の順序

`P1修正 → P2修正 → W38根治 → 115.7%再検算 → Gate 3-2b再レビュー →`
`既定OFF配線 → PM100/保存則A/B → モデルA/B/C/D比較`

以下の2026-08-24節は履歴として保持する。最新判断には上記を使う。

更新: 2026-08-24 18:18 JST / 更新者: Codex

## write lock

`RELEASED` — 14:18 JSTに実プロセス0、`ALL_DONE`、`BACKTEST_ALL_DONE`、
PM100 pytest完了を確認したため、旧write lockを解除した。

ただし既存差分のreset、checkout、stash、削除、上書きは禁止を継続する。
新規施策は既定OFFとし、`src/production_config.py` はユーザー承認前にONにしない。

## 実行中

- Claude CodeがQ-01〜Q-04のGate 0是正を継続中。
- Q-01の段同一性修正方針をfable reviewerで確認中。
- Q-02判定器、Q-03完全構成比較、Q-04仮想着弾案の新規成果物が生成中。
- CodexはClaude報告箱・成果物・実プロセスを監視し、完了後に独立レビューする。
- Codexの現行51指標監査が完了。61動画・全1,275ペアを評価し、未使用12動画で確認。
  詳細は `docs/CODEX_INDICATOR_AUDIT_2026-08-24.md` と `CODEX_TO_CLAUDE.md` 最新節。
- 局面別モデル追補を共有済み。現行は全局面共通モデル・共通ブレンドであり、
  Gate 0後に全局面/文脈付き単一/soft-gated 3モデル/ExchangeEpisode専門を比較する。

## 完了済み

- PM100全8区間のOFF/ON timeline dump生成。
- PM100修正の単体テスト: 31 passed。
- 指摘場面 seg01 game2: OFFは約1.6秒で +100→-99、ONは +100→+25を維持。
- 掛け算式10ケース、偽イベント4走行、8動画×3条件バックテスト完了。
- OFF baseline / OFF final / 先行OFFのMD5一致。
- ClaudeがCodex品質監査Q-01〜Q-04を受諾。Q-01は独立テストで再現。
- fable architect/reviewerは交換仕様を条件付き不合格とし、Gate 2前の仕様修正を要求。
- 全pytest: 5676 passed / 13 skipped / 2 failed。速度失敗は低負荷単独PASS、
  おじゃまダメージ失敗は既存意味論不具合として再現。
- 指標監査: 現行モデルは序盤AUC0.5256 / 中盤0.5506 / 終盤0.7843。
  forecast系3列はほぼ完全重複かつ95.1%が0。時間差交換にはExchangeEpisode会計が必要。

## PM100全8区間結果

- 急変: 79回 → 40回。
- 反転試合: 47 → 34。
- 決着方向誤り: 10 → 7。
- ±100張り付き: 14.0% → 18.3%へ悪化。
- 生モデルと逆符号の張り付き: 3.3% → 5.2%へ悪化。

結論: 現行A+Bは指摘場面には効くが全域採用不可。既定OFFを維持する。

## Gate判定

`Gate 0 継続` — Q-01〜Q-04、w2の旧`formula`混入、v51 ON 99.391%、
母集団差、交換仕様レビュー指摘を閉じるまでGate 2統合・本番採用へ進まない。
