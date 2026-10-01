# 事前登録 (exev docs/agent_coordination/DECISIONS.md から複写、採点前に固定)

## 2026-10-01 事前登録 (採点前に固定): 評価器切替時の値飛び対策 (--exchange-event-switch-smoothing) と E19 --landing-counter-prob の再測定 (user承認: 事前登録の門に合格した構成は本番登録まで進めてよい)
- 作業場所: worktree D:/puyo_analyzer/wt_switch (branch claude/eval-switch-smoothing-20261001、exev e25fd85 から分岐)。入力 = 現本番の保存記録 logs/pending_expiry/full/records (認識は不変、再生のみ)。結果 = wt_switch の logs/switch_smoothing/RESULT.md。
- 比較基準 (現本番): logs/pending_expiry/e36b_on の q .5075669993 (6,526フレーム/4試合) / zenchi 7,671/8,333 / 誤発火 0/37 / 3:00場面 2760.38秒 / 第14試合誤確定 0 / 切替飛び 342件 (うち「物理イベントなし・由来切替に帰属」34件、遷移別 waiting_confirmed→G_fe 19・S3_landing→G_fe 6 ほか)。全て採点前に実測済みの基準値 (飛びの測定器 scripts/measure_switch_jumps.py、閾値 |Δ勝率|≥0.15/1秒)。
- 測定器の健全性 (採点前の必須条件): フラグOFFで再生した出力が logs/pending_expiry/e36b_on と display.npz・events.jsonl ともバイト一致、採点 SUMMARY も同値。一致しなければ以降を採点しない。
### 対策1: 切替時の値飛び (構成A=主、構成B=Aが門で落ちた場合のみ)
- 構成A (src/exchange_display_smoothing.py): (a) waiting_confirmed の旧評価器値も EMA を通す、(b) EMA状態は常に直前の表示値から連続 (古い状態から再開しない)・試合境界は新試合の値へ即時スナップ、(c) 物理イベント (発火=記録数増・連鎖終了信号・着地の記録増、死亡を含む切替、直前0.3秒以内のイベント) のない由来切替だけ 0.5秒の線形ブレンド。confirmed_death は従来どおり即時表示 (遅延禁止)、EMA状態も同値へ合わせる。定数 BLEND_SEC=0.5 / EVENT_GRACE_SEC=0.3 は物理量 (1手の設置間隔約1.3秒より短くEMA時定数約0.13秒より長い) から固定し、結果を見て調整しない。
- 構成B (Aが門で落ちた場合のみ1回): (a) を外し、waiting_confirmed は旧評価器の生値を表示するがEMA状態はその値へ合わせる ((b)(c) は同じ)。
- 合格条件 (すべて満たす):
  1. q log loss ≤ .5075669993 + 5e-7 (標準M3=display_p1。平滑前の列なので平滑では動かない設計。実際に display_p1・source・adv_raw_last・t_sec・game_idx が OFF と全列一致することも確認)
  2. 表示ベース q (display_adv→勝率、同じラベル窓・同じフレーム) ON ≤ OFF + 0.001 (標準qが平滑に盲目なことへの補助門)
  3. zenchi hits ≥ 7,671 (母数 8,333)
  4. 誤発火 0/37・bound_false 0・unresolved 0
  5. 3:00場面 ≤ 2766.0秒 (実表示 display_adv 由来で測る。OFFで同関数が 2760.38 を返すことを先に確認)
  6. 第14試合の誤確定 0
  7. 必須: 「物理イベントなし・由来切替に帰属」の飛び 34件 → ON ≤ 17件 (50%以上減)。同じ測定器・同じ5記録・同じ閾値。
  8. 正当な変化の保存: confirmed_death 行の display_adv が確定値そのもの (遅延0) 100% (分母=confirmed_death 行数)、かつ種別 fire/death の飛び件数 ON ≥ OFF×0.9。
  9. 鮮度 (display_adv): 直前同値率 ON ≤ OFF+0.05、最長同値区間 ON ≤ OFF、更新回数/分 ON ≥ OFF×0.95。評価値 (display_p1) の鮮度は OFF と同一。
- 合格したら --production-exchange-event に接続 (EXCHANGE_EVENT_ADOPTED へ登録)。不合格なら本番は現状維持、結果は記録。
### 対策2: E19 --landing-counter-prob (models/landing_counter_prob_v1) を現本番の上で再測定
- 構成: 現本番の再生構成 (run_e36b options) + landing_counter_prob=True。同じ保存記録で再生し、同じ採点器で採点。
- 合格条件 (すべて満たす): 上記 1・3・4・5・6 (非悪化) に加え、必須の改善 = q log loss ≤ .5075669993 − 0.002 (= .5055670) または zenchi hits ≥ 7,671 + 50 (= 7,721)。鮮度は評価値 display_p1 で 直前同値率 ON ≤ OFF+0.05・最長同値区間 ON ≤ OFF。
- リーク確認 (採点前に実施): models/landing_counter_prob_v1 の学習データに今回の評価5記録 (q_7gc4TgFig・review・fcXG83vInDY・mia8KCjr52g・zenchi の試合) が含まれる場合、改善は評価に使えない (「新モデルがオラクル上限を超えたらリークを疑う」教訓)。含まれる場合は採点しても本番登録しない (結果は参考値として記録)。
- 両方が合格した場合: 両ON の組合せも同じ非悪化門 (1・3〜6) で再生し、通ったときだけ両方を登録。落ちたら効果の大きい側だけ。
