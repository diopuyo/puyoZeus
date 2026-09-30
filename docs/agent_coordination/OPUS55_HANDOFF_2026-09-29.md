# 引き継ぎ 2026-09-29 夜 (Codex上限中・Claude節約運用)

前回: `OPUS55_HANDOFF_2026-09-28.md`。**Codex は利用上限で 10/4 8:49 まで停止**。その間は Claude 子エージェントで実働、**週上限の1日15〜18%** に収める (memory `feedback_claude_budget_harness_2026-09-29`)。

## 本番 (採用済み)
- 撃ち合い評価 E27+E31+E32 を `src/production_config.py` に登録 (3802ab7)。`--production-exchange-event` で適用。q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火 1/35。
- リアルタイム (rt) も本番評価を実行時に読む (B18a e3915d3)、表示EMA通知単位・発火前スナップショット配線 (B18b 9bf9837)、保持コスト 0.90/1.34ms (B18c dc300c0)。

## 不採用 (既定OFFのまま)
E33/E33b (段タイムアウト)、E34 (起点保持ガード: 欠落が増える)、R1 (58ae8a7、合図照合: おじゃま合図が q を悪化させた、D4 1fe6f3b)。

## 20:33 に Codex 上限で途中停止した4本 (変更は worktree に未コミットで残る + 退避 `C:/Users/ryouj/.codex/worktrees/_backup_codex_limit_20260929/`)
| 作業 | 場所 | 状態 | 次 |
|---|---|---|---|
| R1b 合図照合 (NEXT+式のみ) | exev logs/r1b | **合格・コミット 82d3baf** (Claudeが評価段を起動): q .508040 / zenchi 92.06% / 誤発火 1/35 / 誤修正 0.20% / 遅延同一 / 代理正解率 99.26→99.30%。3:00 は反撃75,440点・残171個を採用、負け確定は未 (E36で) | 本番登録は E36 後にまとめて user 判断 |
| D5 既存の誤った負け確定 | exev logs/d5 | 根因特定済み: q 第14試合1P 884秒、旧単発死亡 (`ExchangeLandingProjection._single_death_metrics→_latch→unavoidable_death`) が応手後の生存枝を検証せず確定 (盤面右上2セル誤認)。修正 `single_death_proof_guard` 実装途中 | 修正の完成と5記録で誤確定0の確認、R1記録で E35 の3:00先行確認 |
| E35b 上限計算の高速化 | exev | ベンチ途中 | 後回し可 |
| B19 リアルタイム修正 (再起動後の交換欠落・ID重複 / KeyError / guard_loop 例外分類 / フレームview / テスト順序依存) | rt logs/live_b19 | テスト点検途中 | 後回し可 |

## 次の一手 (新セッション)
1. D5 を Sonnet の coder 子エージェント1本で仕上げる (未コミット: src/exchange_single_death_proof.py, scripts/*d5*, tests/test_d5_single_death_safety.py, overlay/landing/replay の single_death_proof_guard 配線)。
2. E36 = R1b記録 (logs/r1b/records) 上で E35 + D5修正 を再生、事前登録の合否。runner は既存 scripts を流用。
3. E35b・B19 は後回し。

## 最終目標の E36
R1b + E35 (`--post-counter-death-bound`、0f28a04) + D5修正 の組合せで、zenchi 3:00 の 2P≤5% ≤2766秒・誤った負け確定0・q/zenchi/誤発火 非悪化。E35 は正盤面なら 50/50 候補で負け確定を証明済み。

## 残課題
- 評価側: `src/exchange_midchain_completion.py:56` の履歴が試合境界を越えて蓄積 (本番 E27 の挙動、影響測定してから修正)。
- テスト: 基準時点から失敗77・エラー292 (exev)、65/165 (rt) が既存。今回の新規退行は50行制限1件。
- 小修正: 50行超2関数、`prefire_snapshot_reader.py:101` の色9直書き、`scripts/probe_e34_recognition.py:35` の署名欠落ラッパー。
- rt: 実時間の遅延測定 (B18本体) は重い処理が無い時に単独で。
- user伝授 (memory): 置き終わりの合図3つ / 打ち返し後の負け確定の考え方。
