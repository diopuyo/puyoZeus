# 現行51指標 死活・組合せ・局面別監査

作成: 2026-08-24 17:12 JST / 作成者: Codex  
対象Git HEAD: `6ee7496`  
判定: **現行51指標は終盤判定には有効だが、催促対応→相手本線→自分本線の時間差交換を中盤で判断するには不足。ExchangeEpisode会計と局面別モデルが必要。**

## 1. 結論

1. 現行モデルの全体AUCは **0.6357** だが、近似時間位相では序盤 **0.5256**、中盤 **0.5506**、終盤 **0.7843**。性能の大半を終盤が担っている。
2. 単独上位も同じ傾向である。`diff_board_ojama_count` は全体0.5763に対して序盤0.5040、中盤0.5109、終盤0.6926。着弾後には強いが、時間差交換の未解決中にはほぼ無情報である。
3. 明確な削除・再設計候補は `all_clear_bonus_pending`、`main_linked_ratio`、`buried_hole_count`。ただし即時削除ではなく、原因確認→寄与A/B→帰属表示から除外、の順で扱う。
4. `saturation_chain_upper` は死に指標ではない。**99.59%欠損**の配線・カバレッジ障害で、観測できた0.41%ではAUC 0.6360、動画別中央値0.6133。削除ではなく生成経路を修正する価値がある。
5. `ojama_forecast`、`ojama_forecast_uncapped`、`ojama_forecast_progress_interaction` は Spearman 0.9996〜0.99998でほぼ同一、95.1%が0、単独AUC 0.5076〜0.5077。現在の予告系は「独立した3軸」ではなく、ほぼ同じ弱い1軸である。
6. 全1,275ペア中、探索に使わない12動画で単独より有意に改善した上位ペアは20件中17件。ただし最良ペアの改善はほぼ加算効果で、掛け算項の追加効果はゼロだった。
7. 最良の加算ペアは `diff_max_column_height + diff_current_max_chain`。確認AUCは単独0.5735から0.6269へ **+0.0533**、動画クラスタ95% CIは **+0.0296〜+0.0748**。一方、中盤AUCは0.4864で、終盤AUC0.7884に偏る。全局面共通の重みとして採用してはいけない。
8. より局面安定なのは `diff_column_bumpiness + diff_board_ojama_count`。確認AUC0.5908、単独比+0.0318で、序盤・中盤・終盤の改善がそれぞれ+0.0352、+0.0335、+0.0326。ただし「相手本線後、自分本線前」というイベント順序は表現しない。
9. 非線形相乗の最良確認例は `diff_board_color_puyo_total × color_diff_x_ojama_diff`。加算モデル比 **+0.0130**、95% CI **+0.0043〜+0.0214**。ただし確認AUC自体は0.5380で、中盤AUCは0.4245。研究候補であり本番候補ではない。

## 2. 対象と検証設計

- 入力: `labeled_win_model62_3col.csv`
- ラベル済み: 325,707行 / 61動画 / 3,172 game_idx
- 指標数: 51
- 相乗効果用集約: video × game × side × 近似位相の中央値、19,021行
- 単独指標: 全体AUC、動画別AUC中央値、近似時間位相、盤面状態位相、欠損率、ゼロ率、Permutation Importanceを併用
- ペア探索: 49動画側だけで全1,275通りを4-fold GroupKFold評価
- 確認: 探索に使わない12動画へ固定適用
- 信頼区間: 確認12動画を単位に1,000回クラスタ・ブートストラップ

入力SHA-256:

- 学習CSV: `6E859D906FB52588FE5C1AFAB0A67FF20E5DE5CDC55CCA330874372C1E0EF38D`
- 特徴量定義: `51BE534089AFCC9E99AE9BA713461EBFE6E25F33AD4E58FE9134145638895B49`

## 3. 死活分類

### 3.1 死に指標候補

- `all_clear_bonus_pending`
  - 全体AUC 0.5007、全位相0.4999〜0.5017、99.20%が0、Permutation rank 49/51、重要度 -0.000120。
  - 現データでは勝敗情報を持たない。全消し状態の発生・保持・消費の配線が正しいか先に検証する。
- `buried_hole_count`
  - 全体AUC 0.5000、99.954%が0、Permutation重要度0。
  - 定義が上級者盤面に合っていないか、抽出条件が厳しすぎる可能性が高い。
- `main_linked_ratio`
  - 全体AUC 0.5004、動画別中央値0.5040、Permutation rank 48/51、重要度 -0.000084。
  - `isolated_pair_count`との相関0.712はあるが、単独・局面別とも救済されない。

### 3.2 局面限定指標

- `simultaneous_pop_richness`: 全体0.5127、序盤0.4978、中盤0.4961、終盤0.5417。
- `min_puyos_to_ignite`: 全体0.5109、序盤0.4974、中盤0.5047、終盤0.5301。

全局面に同じ係数で入れるより、終盤または発火直前のゲート内で使うべき候補である。

### 3.3 カバレッジ障害

- `saturation_chain_upper`
  - 欠損99.5877%、非欠損部の全体AUC0.6360、動画別中央値0.6133。
  - 欠損を0埋めした現行学習ではPermutation rank 45/51となり、潜在信号が失われている。
  - 対応: producer → CSV変換 → 対称化 → fillna → 本番特徴量組立の全経路を点検する。欠損指示フラグを別列にするか、観測可能条件を限定した専門モデルで評価する。

### 3.4 冗長・低増分

- `ojama_forecast` ↔ `ojama_forecast_uncapped`: rho 0.999978
- `ojama_forecast` ↔ `ojama_forecast_progress_interaction`: rho 0.999648
- `diff_board_color_puyo_total` ↔ `color_offset_power`: rho 0.998509
- `chain_efficiency` ↔ `chain_articulation_point_count`: rho 0.963474
- `board_color_puyo_total` ↔ `board_puyo_total`: rho 0.953647

予告系3列は値域名が違うだけで実質同じ軸になっている。`ojama_forecast_progress_interaction` は進行度との掛け算でも95.1%が0のため、局面情報を追加できていない。

## 4. 局面別の実態

近似時間位相で見た主な単独AUC:

- `diff_board_ojama_count`: 序盤0.5040 / 中盤0.5109 / 終盤0.6926
- `diff_ukeyasusa`: 序盤0.4990 / 中盤0.4821 / 終盤0.6817
- `diff_death_margin_neighbor`: 序盤0.4967 / 中盤0.4854 / 終盤0.6727
- `diff_max_column_height`: 序盤0.4978 / 中盤0.4876 / 終盤0.6674
- `diff_board_puyo_total`: 序盤0.4794 / 中盤0.4645 / 終盤0.6726
- `diff_current_max_chain`: 序盤0.4931 / 中盤0.5162 / 終盤0.6307

強い指標の多くが序盤・中盤で0.5近傍または逆方向となり、終盤だけで強い。これは「盤面の最終的な崩れ」を読む能力であり、「未解決の交換」を先読みする能力ではない。

盤面状態位相では一部の値が変わる。例として `diff_current_max_chain` は序盤0.4806 / 中盤0.5518 / 終盤0.6077。位相定義で符号や強さが変わるため、単一の静的モデルに位相を暗黙学習させるだけでは不安定である。

## 5. 組合せ評価

### 5.1 補完関係が強いペア

- `diff_max_column_height + diff_current_max_chain`
  - 確認AUC 0.6269、開発側最良単独0.5735比 +0.0533、95% CI +0.0296〜+0.0748。
  - 加算AUC0.6270に対し掛け算込み0.6269。非線形上積み -0.0001で0と区別できない。
  - 序盤0.5538 / 中盤0.4864 / 終盤0.7884。終盤特化。
- `board_ojama_count + diff_column_bumpiness`
  - 確認AUC0.5781、単独比+0.0415、95% CI +0.0202〜+0.0634。
  - 序盤0.5383 / 中盤0.5412 / 終盤0.6515。比較的局面安定。
- `diff_column_bumpiness + diff_board_ojama_count`
  - 確認AUC0.5908、単独比+0.0318、95% CI +0.0152〜+0.0492。
  - 序盤・中盤・終盤の改善が均等で、次の限定A/B候補として最も扱いやすい。

### 5.2 非線形相乗

非線形相乗上位20候補のうち、未使用動画で掛け算項の95% CIが0を超えたものは7件。ただし絶対AUCは低いものが多い。

- `diff_board_color_puyo_total × color_diff_x_ojama_diff`
  - 加算0.5250 → 掛け算込み0.5380、差+0.0130、95% CI +0.0043〜+0.0214。
- `chain_articulation_point_count × color_diff_x_ojama_diff`
  - 掛け算上積み+0.0102、95% CI +0.0001〜+0.0196。
- `min_puyos_to_ignite × conn_triple_count`
  - 掛け算上積み+0.0101、95% CI +0.0030〜+0.0175。

これらは探索価値があるが、現行51列フルモデルへの追加増分を測った結果ではない。次段階ではフルモデルに対する grouped ablation と、別動画集合での再確認が必要。

## 6. 99%→1%急落問題への判断

### 現行施策だけで十分か

**不十分。** Q-01の累積器・幕間配線とQ-04の仮想着弾は必要条件だが、それだけでは「交換の順序」をモデル入力へ渡せない。

根拠:

- おじゃま予告系は95.1%が0、互いにほぼ完全重複、単独AUC約0.508。
- 最重要の`diff_board_ojama_count`は中盤AUC0.511で、実際におじゃまが盤面へ載った終盤に初めて強い。
- 静的ペア最良でも中盤AUC0.486。高さ差と火力差を組み合わせても、相手本線後・自分本線前の未解決状態を区別できない。
- 現行特徴には「催促へ対応済み」「相手本線が先に発火」「自分本線は保持中」「未着弾純残量」「自分の発火までの残り時間」を同じepisodeとして保持する状態がない。

### 追加すべき施策

ExchangeEpisode / ExchangeLedgerへ次の状態を持たせ、特徴量化する。

1. `opponent_harass_sent` / `own_harass_response_sent`
2. `opponent_main_fired` / `own_main_fired`
3. `pending_incoming_total` / `cancelled_total` / `net_incoming_remaining`
4. `own_main_ready_power` / `own_main_eta_sec` / `own_fire_probability`
5. `virtual_landing_death_margin` / `virtual_landing_max_height`
6. `time_since_opponent_main_fire` / `episode_stage`
7. `post_own_fire_net_ojama` / `post_exchange_survival_margin`

学習ターゲットは最終勝敗だけでなく、episode単位の次も併用する。

- 交換前後の勝率差
- 交換後の純おじゃま残量
- 交換後の窒息余裕
- 相手本線後、自分本線前の生存確率
- 自分本線発火後に優勢へ戻るか

表示側では、`episode_stage == unresolved` の間は静的終盤指標による±100 hard overrideを禁止し、台帳の確定値・仮想着弾・自分本線残存能力を優先する。

## 7. Claudeへの作業順

1. 進行中のQ-01〜Q-04 Gate 0を閉じる。現行A+Bは本番OFF維持。
2. `saturation_chain_upper`の99.59%欠損原因を読取専用で追跡し、producerから本番特徴量組立までの経路表を出す。
3. ExchangeEpisode仕様へ上記7状態とepisode単位ターゲットを追加する。
4. `diff_column_bumpiness + diff_board_ojama_count`を「単純加算・局面安定候補」として限定A/Bする。
5. `diff_max_column_height + diff_current_max_chain`は終盤ゲート内だけでA/Bし、中盤では使わない。
6. 非線形上位3ペアはフル51列に対するgrouped ablationを行い、別動画で再確認するまでproductionへ入れない。
7. 死に指標3件は即削除しない。配線確認、帰属表示除外、A/B、互換性確認の順で処理する。

## 8. 制約と注意

- `tsumo`は真の手数ではなくSTABLEスナップショット順位の近似。既知弱点W18のため、位相AUCの絶対値は参考値である。
- `game_idx`境界には既知弱点W21がある。相乗効果はgame/phase中央値でフレーム重複を抑えたが、境界誤り自体は除去していない。
- `match_progress`は盤面総量由来でリアルタイム利用可能だが、連鎖後に戻る非単調な状態位相である。
- ペア確認は探索未使用12動画で行ったが、最終採用には新規動画または時系列的に完全分離した再確認集合が必要。
- 本監査は既存ソース・本番フラグを変更していない。

## 9. 成果物

- 再現スクリプト: `scripts/_audit_indicator_synergy_2026-08-24.py`
- 機械可読サマリ: `data/verify/indicator_audit_2026-08-24/summary.json`
- 指標死活: `data/verify/indicator_audit_2026-08-24/feature_health.csv`
- 冗長ペア: `data/verify/indicator_audit_2026-08-24/redundant_pairs.csv`
- 全1,275ペア探索: `data/verify/indicator_audit_2026-08-24/pair_screen_development.csv`
- 加算改善上位の確認: `data/verify/indicator_audit_2026-08-24/pair_confirmation_top20.csv`
- 非線形上位の確認: `data/verify/indicator_audit_2026-08-24/pair_confirmation_nonlinear_top20.csv`
- 実行ログ: `logs/indicator_audit_2026-08-24.log`

## 10. 追補: 現行モデルの局面運用と次モデル比較（2026-08-24 18:18 JST）

### 10.1 現行運用の事実

**現在、序盤・中盤・終盤ごとに指標重みを切り替える運用はしていない。**

- `visualize_advantage_overlay.py`の通常成果物経路は、1つのHistGradientBoostingモデルを全局面で共用する。
- `--model-dir`なしの既定成果物は`data/verify/retrain148_2026-08-14`の47列モデル。
- 全知レンダ等で明示されている`--model-dir data/verify/retrain_model62_2026-08-21`も、1つのモデルを全局面で共用する。
- 本監査対象`retrain_model62_3col_2026-08-21`は51列だが、`match_progress`自体を特徴列に含まない。位相を含む列は主に`ojama_forecast_progress_interaction`であり、95.1%が0、`ojama_forecast`とのrho 0.999648なので、実質的な局面切替を担えていない。
- 既定47列モデルも`match_progress`を含まない。
- 成果物が使えずCSV起動時学習へフォールバックした経路だけは、`match_progress`と`color_puyo_x_earliness`を追加できる。ただしこれは通常の評価済み成果物経路ではない。
- 表示合成は全局面共通の固定係数を使う。`W_PRESSURE=0.35`、`W_FORECAST=0.30`、`W_MODEL=0.20`、`W_THREAT=0.15`、`W_COUNTER=0.20`であり、位相別の係数表はない。
- イベント状態に応じたtracker・overrideは複数あるが、これは序盤・中盤・終盤別の指標重みではない。

### 10.2 位相別Plattとの区別

位相別Platt校正器と`--phase-calibration`は実装済みだが、既定はFalseである。

- 有効化しても、指標の重要度・モデル分岐・表示ブレンド係数は変わらない。
- 変えるのは最終確率の傾きと切片だけである。
- 2026-08-11実測では校正なしのECEが最良で、位相別Plattは悪化したため非推奨。
- よって位相別Plattを「局面別モデル導入済み」の根拠にしてはいけない。

### 10.3 比較すべきモデル構成

Gate 0完了後、同一データ・同一video GroupKFold・同一確認動画で次を比較する。

1. **A: 現行全局面共通モデル**
   - 現状再現baseline。既存51列、局面ゲートなし。
2. **B: 局面文脈付き単一モデル**
   - `match_progress`、信頼できる真の`tsumo_count`、`episode_stage`を入力する。
   - 各指標と局面の交互作用は木または明示列で学習し、人手で重みを決めない。
3. **C: 3専門モデルのsoft gate**
   - 序盤・中盤・終盤の専門モデルを学習する。
   - 1/3、2/3で突然切り替えず、境界付近は2モデルを滑らかに混合する。
   - 硬い切替による勝率急変を検収対象にする。
4. **D: ExchangeEpisode専門モデル**
   - `harass_response`、`opponent_main_fired`、`own_main_held`、`own_main_fired`等のepisode段階を使う。
   - 未解決episode中はA〜Cの静的終盤判断よりDを優先し、±100 hard overrideを禁止する。

### 10.4 必須評価

- 全体・序盤・中盤・終盤のOOF AUC、logloss、Brier、動画別中央値/IQR
- おじゃまフラット、催促対応中、相手本線後、自本線前、自本線後のepisode別AUC
- 位相境界前後±3秒の勝率ジャンプ件数と最大変化幅
- PM100全8区間の急変、反転試合、誤った±100張り付き
- A→B、A→C、A→Dの動画クラスタbootstrap 95% CI
- 全pytestと既定OFF bit-identical

### 10.5 採用条件

- 序盤または中盤を改善しても終盤・全体・較正を悪化させない。
- soft gate境界で新しい急変を発生させない。
- 99%→1%指摘episodeで、相手本線後・自本線前を未解決として維持できる。
- `src/production_config.py`への採用登録は、独立確認動画での再現とユーザー承認後のみ。
