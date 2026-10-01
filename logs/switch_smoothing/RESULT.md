# 評価器切替の値飛び対策 (--exchange-event-switch-smoothing) と E19 再測定 (2026-10-01)

結論: **どちらも事前登録の門に不合格 → 本番登録なし** (production_config 不変)。フラグは既定OFFの試験用として残す。
入力 = 現本番の保存記録 5本 (logs/pending_expiry/full/records、認識不変・再生のみ)。事前登録 = exev DECISIONS.md 2026-10-01 (複写 PREREGISTRATION.md、採点前に固定)。
作業 worktree D:/puyo_analyzer/wt_switch (branch claude/eval-switch-smoothing-20261001)。

## 測定器の健全性 (採点前)
- OFF 再生 5本の display.npz / events.jsonl が logs/pending_expiry/e36b_on と **バイト一致 5/5**。採点も一致 (q .5075669993 / zenchi 7,671/8,333 / 誤発火 0/37 / 第14試合 0)。
- 逸脱1: 場面 (3:00) は実表示 (display_adv) 由来で測ると OFF で 2760.25秒 (旧採点器は display_p1 を再EMAする近似で 2760.38)。門 ≤2766.0 には影響なし。旧近似は平滑に盲目なので使わない。
- 逸脱2: 標準の q (M3) は平滑前の display_p1 を使うため、表示平滑では原理的に動かない (実測も ON/OFF 完全同値)。補助門として表示ベース q (display_adv 由来) を足した。
- 既存の再生器 run_e36.worker は 0f28a04 版の証明器を固定するが、現行 src と引数が合わず動かない (early_exit)。26358a5 の「判定不変」記録に従い現行 src を使い、上のバイト一致で確認済み。post_counter_bound_audit.json だけ内容差 (監査ログ、表示・events には無関係)。

## 切替時の値飛び (OFF 実測、5記録・112,346フレーム・由来切替 641回)
飛び (同一試合1秒内 |Δ勝率|≥0.15) = 342件。種別: 発火 112 / 着地 48 / 死亡 45 / 連鎖終了 24 / 試合境界 33 / **物理イベントなし 80**。
物理イベントなし80のうち「由来切替に帰属」= **34件**: waiting_confirmed→G_fe 19、S3_landing→G_fe 6、G_fe→S3_provisional 3、S3→G_fe 2、他4。残り46は切替なしの G_fe 内変動等 (評価器の実際の動き、対象外)。
- 原因の実体: waiting_confirmed の旧評価器値は 5記録とも **常に厳密に中立 (adv 0 / 勝率 0.5)** (q 1,079・fc 1,192・zenchi 45 フレーム全て)。新試合の最初の G_fe フレームで、前試合末の古い EMA (±100 側) から再開して 1 フレームだけ 0.04/0.98 等へ飛ぶのが主犯 (qの9件で確認、一時値 0.04〜0.98)。つまり (b) が主因、(a) は中立値が定数のため実質 (b) の一部。

## 構成A (旧評価器値もEMA・連続状態・非イベント切替のみ0.5秒ブレンド) の結果 vs OFF
| 門 | OFF | A | 判定 |
|---|---|---|---|
| q (標準、≤.5075674993) | .5075670 | .5075670 | 合格 (同値) |
| 表示ベース q (≤OFF+.001) | .506002 | .506334 | 合格 |
| zenchi (≥7,671/8,333) | 7,671 | **7,670** | **不合格 (−1フレーム)** |
| 誤発火 / bound_false / unresolved | 0/37, 0, 0 | 同 | 合格 |
| 場面 ≤2766.0 | 2760.25 | 2760.25 | 合格 |
| 第14試合 | 0 | 0 | 合格 |
| 必須: 非イベント切替の飛び 34→≤17 | 34 | **16 (−53%)** | 合格 |
| 確定死亡の即時性 (分母 5,544行) | 5,544 | 5,544 | 合格 |
| 正当な飛び (発火+死亡) ≥0.9× | 157 | 158 | 合格 |
| 鮮度 display_adv: 同値率 | .1592 | .1606 | 合格 (≤+.05) |
| 鮮度: 最長同値 / 更新回数/分 | 85.97秒 / 1513.4 | 85.97秒 / 1510.8 | 合格 |
| 平滑前の列 (display_p1・source・adv_raw_last・t_sec・game_idx) | - | 5記録全一致 | 合格 |
全飛び 342→297、試合境界の飛び 33→5。唯一の不合格 = zenchi −1フレーム。
- 該当フレーム: zenchi 2948.483秒 (第47試合、勝者2P)。公式の試合終了窓の最終フレームが追跡側では次試合 (第8) の最初の G_fe フレームで、OFF は前試合の古いEMA (−29) が **偶然** 勝者側の符号で一致、ON は新試合の値 (+15.6) へ即時スナップした。窓境界1フレームのずれによる見かけの差で、悪化ではないが、事前登録の門 (7,671) は文字どおり未達。
## 構成B (事前登録の予備、Aが落ちた場合のみ1回)
A と **5記録とも display.npz がバイト一致** (waiting の旧評価器値が常に中立のため、(a) の有無で差が出ない)。門の結果は A と同一。独立な追加証拠にはならない。

## E19 --landing-counter-prob (現本番の上で再測定)
- リーク確認: 学習は 148動画 (eligible 148、評価ID q_7gc4TgFig/fcXG83vInDY/mia8KCjr52g/c0BQoMJwwQU の重複 0) → 評価に使える。
| 門 | OFF | E19 | 判定 |
|---|---|---|---|
| q (標準) | .507567 | **.496663 (−.0109)** | 非悪化・必須改善 (≤.505567) とも合格 |
| zenchi (≥7,671) | 7,671 | **7,627 (−44)** | **不合格** (必須改善 +50 にも届かず逆方向) |
| 誤発火 / 場面 / 第14試合 | 0/37 / 2760.25 / 0 | 0/37 / 2760.25 / 0 | 合格 |
| 評価値鮮度 (同値率 ≤+.05, 最長同値) | .7940 / 89.77秒 | .7841 / 89.77秒 | 合格 |
q は大きく良化するが zenchi が 44 フレーム悪化 (9/28 と同じ q↑・zenchi↓ の取引)。事前登録は「非悪化かつ改善」なので不合格。組合せ (A+E19) は事前登録どおり再生せず。

## 成果物
- src/exchange_display_smoothing.py (SwitchAwareDisplayEMA、既定OFF、表示のみ) / scripts/visualize_advantage_overlay.py (`--exchange-event-switch-smoothing`、`_exchange_display` の分岐) / scripts/replay_exchange_event_20260926.py (`switch_smoothing` 引数)。
- scripts/measure_switch_jumps.py (飛びの測定器) / run_switch_smoothing.py / report_switch_smoothing.py / _run_*.sh。
- tests/test_exchange_display_smoothing.py (15件)。既存 374件 (production/e18〜e33 系) 通過。
- 出力: logs/switch_smoothing/{off,a,b,e19}/ (FULL_METRICS.json / VERDICT.json)、baseline_jumps.json。
## 次の判断 (user)
1. 門の文字どおりでは A は不合格だが、−1フレームは窓境界の偶然 (OFF側が古いEMAで偶然一致)。門を ≥7,670 へ緩めて登録するか (事後変更なので要承認)、不合格のまま据え置くか。他の全門は合格・飛び34→16。
2. E19 は q −.011 / zenchi −44。zenchi 側の悪化原因 (勝者側の応手確率が終盤で過大?) を見るなら別途。
