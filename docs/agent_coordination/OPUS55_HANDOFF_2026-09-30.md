# 引き継ぎ 2026-09-30 (Codex上限中・Claude実働)

前回: `OPUS55_HANDOFF_2026-09-29.md`。Codex は 10/4 8:49 まで停止中。運用は memory `feedback_claude_budget_harness_2026-09-29` を参照 (節約とはトークン効率のことで、ゴールは遅らせない)。
**user は判断待ちを一括で承認済み (9/30)**。フルアクセスも許可済み。事前登録の門に合格したものは Claude が本番登録まで進める (DECISIONS.md 末尾)。

## 本番 (production_config.py)
- 撃ち合い評価: E27+E31+E32 (3802ab7) に、**E35 + D5 + D5b を追加 (310fec8)**。R1b は `PLACEMENT_RECONCILE_ADOPTED` へ入れた。
  - 根拠 (logs/e36b): q .507565 / zenchi 7,676/8,333 (92.12%) / 誤発火 1/39 / zenchi 3:00 の確定 2760.38秒 (期限 2766)。旧本番は q .507977 / zenchi 7,664。
- E35b の高速化 (26358a5): 出力はバイト一致、上限計算 P50 1.32→0.28ms、P95 191→34.6ms。

## 本日のコミット (exev)
71d378d D5 / 0b8a76b D5b (+E36 runner) / 310fec8 本番登録 / 26358a5 E35b。rt: 075b217 B19 / 03a1c2a B19 全体実行時のエラー是正。

## 分かったこと
- D5 単独は証明できない単発死亡を取り消すため、R1b 記録の q で正しい確定を3件消していた (logs/e36 は不合格)。D5b は生存枝・相殺可能が見つかった場合だけ取り消す。
- **q 第14試合 1P 884秒の誤った負け確定は今も残る** (誤発火 1/39 の1件)。正しい確定の q 第3試合 270.0秒と、理由の系列 (board_unsettled → node_limit → board_unsettled) が同じで、判定側では区別できない。node 上限を 1e6 にしても結論は出ない (logs/d5c_probe/PROBE.json)。**根本は盤面の誤認**で、誤りは5セル (1,5)(2,5)(3,5)(5,2)(6,2)、原票は logs/d5/ROOT_CAUSE.json と FALSE_INPUTS.json。

## 次の一手
1. 第14試合の盤面誤認を認識側で診断・修正する (debugger で計装 → coder)。置き終わりの合図3つ (memory reference_placement_end_signals) が使えるか確認する。
2. rt の全テスト再実行の結果 (logs/live_b19/full_suite_after2.log) を基準の失敗65・エラー165と比べる。B19 が増やした failed のうち1件がまだ特定できていない。
3. 小修正: `exchange_event_overlay.py` の `__init__` が66行 (50行制限)、`prefire_snapshot_reader.py:101` の色9直書き、`scripts/probe_e34_recognition.py:35` の署名欠落。
4. rt B18 本体 (実時間の遅延測定)。重い処理が無い時に単独で行う。
5. 評価側: `src/exchange_midchain_completion.py:56` の履歴が試合境界を越えて蓄積する件 (影響を測ってから)。
6. ロードマップ正本 `docs/ROADMAP_2026-08-09.md` を現状に更新する (9/7 以降の E/B 系列が反映されていない)。
