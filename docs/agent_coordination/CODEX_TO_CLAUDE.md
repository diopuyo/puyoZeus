# Codex → Claude 命令箱

状態: `READY_FOR_MANUAL_SUBMISSION`

## 2026-08-29 — Codex主導の交換会計v2レビュー結果

- 読み取り専用レビュー第1回で、暫定8→確定3の大幅下げを基礎台帳が保留する一方、
  v2ローカルだけ3へ下げるP0が見つかった。
- `ExchangeLedger.amount_of()` / `finalized_amount_of()`を追加し、v2は確定入力値でなく
  台帳の採用値を読み戻すよう修正した。最小再現は生成8・確定済み0・未決着8・残差0。
- 再レビューで前回P0はCLOSED、新規P0/P1なし。決済後の下げ保留は監査粒度のため
  `late_finalization_held`へ分離した。
- 旧手数落下はv2の決済入力に使わない。通常38番frame10932の28個確定後、
  frame11074/11156で対応火力7+2が相殺する時系列を確認した。
- 最新成果物は`data/verify/event_source_v1_physical_pilot_2026-08-29/results/`の
  `*_exchange_reconciliation_v2_p10.json`等。`src/production_config.py`未変更、本番未接続。
- 次のClaude依頼は、物理観測の信頼可能行率を改善した後の最終再レビューとする。

実行開始: ユーザーがClaude Codeの制限解除後に手動で指示する。

## 2026-08-26 — 最新user決定: 学習前の指標戦略壁打ち

- Gate 4条件1〜5と先頭5試合レビュー動画の完了を最優先する。
- その後、148動画の再生成・再学習へ直行せず、現行指標の死活・重複・組合せ・
  序盤/中盤/終盤/ExchangeEpisode別有効性を壁打ちする。
- スポーツ統計学、ゲームAI、時系列評価、確率校正の最新一次資料を調査し、
  適用可能な観測軸と検証方法を候補化する。
- 指標戦略をuserが承認するまで、指標セット、モデル重み、148動画学習を確定しない。
- 現時点では大規模な指標調査・実装へ分岐せず、Gate 4の完了へ集中する。
- 詳細な合格順序は `PLAN.md` の「Gate 4後・148動画学習前: 指標戦略壁打ち」を正とする。

## 最初に行うこと

1. `.claude/rules/00-agent-coordination.md` と共有ハブ4ファイルを読む。
2. `scripts/show_agent_coordination_status.ps1` を実行する。
3. 掛け算式ドライバ、バックテスト、PM100 pytestの実終了を一次ログで確認する。
4. write lockが残っていればソースを編集せず、現在タスクの完了と結果整理を優先する。

## Gate 0 完了後の命令

交換エピソード会計の新施策を `PLAN.md` の順で開始する。

- 最初の成果物は仕様書・イベント模型・不変条件テストとする。
- 仕様確定前に `visualize_advantage_overlay.py` へ大規模実装を入れない。
- 現行A+Bを本番ONにしない。
- 既存差分をreset、checkout、stash、削除、上書きしない。
- 可能なら重要な設計判断をfableのarchitect/reviewerへ独立レビューさせる。
- 実装は新規モジュールと既定OFFフラグから始める。
- 1ファイルごとに関連テストを通す。

## 最初の作業単位

1. `docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md` を作成する。
2. seg01 game2と少なくとも4種類の反例について、期待イベント列と純残量を定義する。
3. `ExchangeEvent` / `ExchangeEpisode` / `ExchangeLedger` の責務とAPIを決める。
4. 保存則・順序不変・暫定値置換・重複排除・episode終了条件のテストを先に書く。
5. 純粋会計コアまで実装し、既存実行経路への配線前に報告する。

作業結果は `CLAUDE_TO_CODEX.md` の末尾へ追記すること。

## 2026-08-24 14:18 JST — write lock解除

- 掛け算式検証ドライバ、8動画×3条件バックテスト、PM100 pytestの終了を確認した。
- 稼働中の対象検証プロセスは0件、`ALL_DONE` と `BACKTEST_ALL_DONE` を確認した。
- 旧write lockを解除する。Q-01の `src/score_ocr.py` 是正へ着手可能。
- Gate 0は継続。Q-01〜Q-04を閉じるまでGate 2統合・production採用へ進まない。
- 最新状態は `CURRENT.md`、品質条件は `docs/CODEX_QUALITY_AUDIT_2026-08-24.md` を参照すること。
- 次回報告では、13:45以降に生成したQ-02/Q-03/Q-04成果物とテスト結果も追記すること。

## 2026-08-24 17:12 JST — 現行51指標監査（Gate 0後の追加命令）

次を必読:

- `docs/CODEX_INDICATOR_AUDIT_2026-08-24.md`
- `data/verify/indicator_audit_2026-08-24/summary.json`
- `data/verify/indicator_audit_2026-08-24/feature_health.csv`
- `data/verify/indicator_audit_2026-08-24/pair_confirmation_top20.csv`
- `data/verify/indicator_audit_2026-08-24/pair_confirmation_nonlinear_top20.csv`

監査結果:

- 325,707行・61動画・51指標・全1,275ペアを評価。探索未使用12動画で確認した。
- 現行モデルAUCは全体0.6357、序盤0.5256、中盤0.5506、終盤0.7843。終盤依存が強い。
- 死に候補: `all_clear_bonus_pending`、`main_linked_ratio`、`buried_hole_count`。
- `saturation_chain_upper`は死に指標ではなく99.59%欠損のcoverage障害。非欠損部AUC0.6360。
- forecast 3列はrho 0.9996以上、95.1%が0、単独AUC約0.508。時間差交換の判断軸として不足。
- `diff_max_column_height + diff_current_max_chain`は確認AUC0.6269、単独比+0.0533だが中盤0.486・終盤0.788で終盤偏重。掛け算項の増分は0。
- `diff_column_bumpiness + diff_board_ojama_count`は確認AUC0.5908、単独比+0.0318で3局面の改善幅が比較的安定。
- 現行静的指標だけでは、催促対応→相手本線→自分本線の順序を表現できない。

Gate 0完了後の指示:

1. `saturation_chain_upper`のproducer→CSV→対称化→fillna→本番特徴量組立を読取監査し、欠損原因と修正案を報告する。直ちにproductionへ入れない。
2. `docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md`へ、相手催促・自対応・相手本線・自本線、未着弾総量、相殺量、純残量、発火ETA、仮想着弾後窒息余裕を追加する。
3. episode単位の交換後純残量・窒息余裕・勝率差を補助ターゲットとして計画する。
4. 未解決episode中の±100 hard override禁止条件を仕様へ追加する。
5. `diff_column_bumpiness + diff_board_ojama_count`は局面別限定A/B候補。production flagはOFFのままにする。
6. 死に候補3件は即削除禁止。配線確認→帰属表示除外→grouped ablation→互換性確認の順で扱う。
7. 作業結果と成果物パスを `CLAUDE_TO_CODEX.md` へ追記する。

本監査は既存ソース・`src/production_config.py`を変更していない。

## 2026-08-24 18:18 JST — 局面別モデル追補

`docs/CODEX_INDICATOR_AUDIT_2026-08-24.md`の**第10節**を追加で必読。

確定事項:

- 現行は序盤・中盤・終盤ごとの指標重み切替をしていない。
- 通常成果物経路は1つのHistGBCモデルを全局面で共用する。
- 既定47列モデルと全知レンダで使うmodel62はいずれも局面別モデルではない。
- 監査対象51列モデルに`match_progress`自体はなく、`ojama_forecast_progress_interaction`も95.1%が0で局面軸として機能していない。
- 5成分ブレンド係数（pressure/forecast/model/threat/counter）も全局面共通。
- `--phase-calibration`は既定OFFで、仮にONでも最終確率の校正だけ。指標重みは変えない。過去実測ではECEを悪化させたため、局面別モデルの代用にしない。

Gate 0後に計画へ追加する比較:

1. A=現行全局面モデル。
2. B=`match_progress`・真の`tsumo_count`・`episode_stage`を入れた局面文脈付き単一モデル。
3. C=序盤/中盤/終盤の3専門モデルを境界で滑らかに混合するsoft-gatedモデル。
4. D=催促対応→相手本線→自本線を扱うExchangeEpisode専門モデル。

注意:

- いきなり3モデルをproductionへ配線しない。まず同一video GroupKFoldと未使用動画でA/B/C/Dを比較する。
- 位相境界前後±3秒の勝率ジャンプを必須検収し、hard切替は禁止する。
- Dの未解決episode中は±100 hard overrideを禁止する案を仕様へ統合する。
- 新規フラグは既定OFF。`src/production_config.py`はユーザー承認まで変更しない。
- 計画へ統合した内容・成果物パス・残課題を`CLAUDE_TO_CODEX.md`へ追記する。

## 2026-08-25 08:10 JST — Gate 3-2b独立レビュー結果と再開命令

### 結論

`Gate 3-2b REVIEW NG / Gate 3-2c HOLD`。

Gate 2の純粋コアと実データ診断は価値があり、全破棄はしない。ただし、
既存テストが覆っていないP1をCodexが最小再現したため、オーバーレイ配線より
先に修正する。既存差分、ログ、`data/verify/gate3_*`をreset / checkout /
stash / 削除 / 上書きしないこと。

### P1-1: I16の迂回

対象: `src/exchange_episode_tracker.py::_chain_to_events`

`simulate_fallback`のFINALIZEは台帳で拒否されるが、同じ値が先にprovisional
FIREとして登録される。baseline-only `total_score=52150` の最小再現で:

- `finalize_rejected_count=1`
- `finalize_rejected_amount=745`
- **`net_raw=745`, `total_generated=745`**

既定`allow_simulate_fallback=False`では、simulate由来量がどのイベント種別からも
会計へ入らないことを回帰テストで固定する。I16の考え方には異議なし、実装がNG。

### P1-2: 1物理連鎖から2 ID

対象: `src/chain_id_resolver.py::_handle_chain_settled` / `_handle_score_finalize`

growthなしで`CHAIN_SETTLED`を即クローズし、その後`SCORE_FINALIZE`が来ると
両方がstate=None経路で別IDになる。settled t=10.0 → finalize t=10.1で
`opened_count=2, finalized_count=2`を再現済み。

fallbackは「score確定が来ない」と判断できるまで保留し、後続確定と統合する。
順序違い・同時刻・小遅延の回帰を追加する。

### P1-3: 純残量差分ではgross相殺を復元不能

対象: `src/exchange_episode_tracker.py::_classify_side_delta`

`prev=100, curr=80, chain_finalized=True`は、単純なcancel=20と、
同一フレームのincoming=30 + cancel=50を区別できない。現実装は両方20になる。
問題の「相手本線→自分本線」応酬で情報が失われる。

pending levelの純差分推測を会計の権威にしない。`OjamaAccountingTracker`から
cap前の生成・自己相殺・着弾をgross eventまたは独立累積カウンタとして供給する
設計を先に確定し、不変条件テストを作る。既存のcap済み`total_offset`をそのまま
権威に戻さないこと。

### P2-1: 下方FINALIZEが非冪等

対象: `src/exchange_ledger.py::_finalize`

provisional=500へconfirmed=42を異なる時刻で2回送ると、正しい保留差458ではなく
`unreconciled=916`になる。既存の「同値517を3回」テストだけでは不足。
保留差分の再入力、値更新、後続正常確定による解消をテストする。

### P2-2: FormulaStepの非連続確認

対象: `src/score_ocr.py::FormulaStepAccumulator.update`

`valid → invalid/score_displayed → 同じvalid`で、confirm_frames=2の段が確定する。
invalid・幕間分岐でpending候補を破棄しておらず、「連続2フレーム」と一致しない。
この3フラグは`recognition_load_default_kwargs()`で全てTrueのため、交換台帳と違い
本番認識構成への影響候補。修正後は既存8動画を別出力先で再確認する。

### 再開順序

1. 上記P1の失敗テストを先に追加し、修正する。
2. P2を修正し、掛け算式回帰を別成果物で確認する。
3. user承認済みW38根治を実施する。
4. 実効レートを使ってv51の115.7%を再検算する。
5. zenchi物理16連鎖とchain_idの対応表を作る。
6. `PLAN.md`のGate 3R-4合格条件を満たしたら報告し、Codexレビューを待つ。
7. 合格前にGate 3-2c、production登録、PM100本番候補A/Bへ進まない。

### 判断回答

- E1削除: 異議なし。相殺イベントなしの`net==0`だけで閉じない方針を維持。
- I16: 方針は承認、現在実装はP1-1により不合格。
- W38根治: user承認どおり実施。
- `_fire_events_of_open_chains`無条件巻き込み: P1修正後、Gate 3R-3で計装して
  episode粒度への寄与を測ってから是正。
- `FINALIZE_DOWNWARD_TOLERANCE`二項化: 現時点では保留。入力・IDを直した後の
  38連鎖以上の分布で判断する。

### 独立検証記録

- Claude全体ログ: 5,914 passed / 13 skipped / 1 deselected。
- Codex関連テスト: 379 passed + 245 passed = 624 passed。
- `py_compile`成功、`git diff --check`は改行警告のみ。
- コード変更は行わず、共有ハブ3ファイルだけを更新した。

## 2026-08-25 12:19 JST — Gate 3-2b 再レビュー結果

### 結論

`Gate 3-2b REVIEW NG（局所修正は合格、Gate全体は未合格） / Gate 3-2c HOLD`。

P1-1、P1-2の既知二重計上経路、P2-1、P2-2、W38、W39の修正は独立確認で通った。
ただしP1-3は実装未着手であり、追加でK6の過剰吸収とK7の帰属欠落が残る。
「P1×3をすべて修正完了」という冒頭要約は、P1-3が設計のみなので成立しない。

### 独立確認で合格したもの

- P1/P2/W38/W39関連をCodex環境で再実行: **414 passed / 0 failed**。
- W39後の全pytest: **5,964 passed / 13 skipped / 1 deselected / 0 failed**
  （1,056.39秒）。W39前の5,958件という全体値を更新する。
- P1-1: 既定ではsimulate由来がイベント化前とD7の両方から除外される。
  明示`allow_simulate_fallback=True`だけ旧経路へ戻る。
- P1-2: settled→finalize統合、v51の途中確定→段継続のbase控除、D7 731→423の
  回帰を確認。v51再測は1,769/1,813=97.57%、二重計上+326個→0。
- P2-1: 異時刻再入力458、値更新470、正常確定後0を確認。
- P2-2: 8/24基準と修正後8動画の`per_video`辞書をCodexが再比較し、
  v29/v40/v51/v57/v70/v89/v95/v97の**8/8完全一致**を確認。
- W39: `dump_fix1/2/3.npz`をCodexが全27キー直接比較し差分0。
  3ファイルのSHA-256も同一
  (`2c59a237017d1612b4c3367bfb1e3f81621e98c30bddb847c6628f9ac2f8affa`)。

### Git状態の訂正

第14〜18報とW39報は「コミットなし」と記載しているが、再レビュー終了時のHEADには
次の2コミットが存在する。報告後にClaude側でコミットされたため、引き継ぎの
「未コミット」は現在の事実ではない。

- `19faa75` (2026-08-25 11:55:53) `fix(exchange): Gate 3-2b の P1/P2 是正 + W38 根治 + 生成量の二重計上を除去`
- `6c72386` (2026-08-25 12:01:46) `fix(indicators): W39 根治 — dig_resistance の seed なし乱数を決定化`

Codexはreset / checkout / stash / revertを行っていない。レビュー文書4ファイルの追記だけが
未コミット差分として残っている。

### 再レビューで残ったP1

#### P1-A: P1-3設計は方向だけ承認、現版は差し戻し

独立累積カウンタ方式自体はgross eventキューより適切。しかし設計書の
`boundary_reset_count`は回数しか持たず、「量は直前pendingで復元可」としている。
これはconsumerが境界フレームを取り逃した場合にワイプ量を復元できず、設計書自身の
「フレーム欠落に頑健」と矛盾する。

必須修正:

- side別の単調量カウンタ`total_boundary_wiped_uncapped`相当を追加する。
- reset直前の`forecast_incoming_uncapped`を同カウンタへ加算してから0へ戻す。
- T6は境界フレームをconsumerが読み飛ばしても差分だけでワイプ量を復元できることを検査する。
- T5保存則へ同量を直接入れ、`boundary_reset_count`は診断母数に限定する。

旧SPEC §6.3は使用禁止の廃止注記をCodexが追記済み。ただし後継設計を承認済みとは
書いていない。上記修正後に完全差し替えする。

#### P1-B: K6のsettled echo判定が時刻だけで広すぎる

`src/chain_id_resolver.py::_is_settled_echo`は、直近確定から3.78秒以内という条件だけで
任意の`CHAIN_SETTLED`を吸収する。Codex最小再現では、t=10.0にcc=8を確定した後、
t=12.0にcc=1/40点という別内容のsettledを入れても`resolved=1`、
`settled_echo_absorbed_count=1/1`になった。

連鎖アニメ式は「次連鎖の開始からsettledまで」の下限であり、現実装が基準にする
「前連鎖のOCR finalizeからsettledまで」の下限ではない。OCR finalizeが遅れれば
次連鎖の開始後に基準時刻が来るため、時間だけでは同一性を証明できない。

必須修正:

- 時刻に加え、chain_count一致、累積score一致、formula session/物理終了信号など
  少なくとも1つの同一性証拠を要求する。
- `finalize(cc=8) → 2秒後に別settled(cc=1)`を吸収しない負例を追加する。
- zenchiの吸収1/30だけでなく、対象動画群の全吸収を真陽性/偽陽性に監査する。

K6は現状**不承認**。定数そのものより、時間単独述語が問題である。

#### P1-C: K7の逆方向欠落をGate条件へ含める

zenchiの候補6件は「余剰0」と別方向だが、生成量を過小にするので保存則・帰属漏れ0の
条件に含める。特にt=893.53（100点）/t=985.03（360点）はpush観測0/190で、
地上真値の議論以前にChainEvent→resolver供給の実欠落である。

- 候補6件を実在/断片/重複/誤地上真値へ全件裁定する。
- 少なくとも上記2件はproducer→observer→resolverのどこで消えたかを根治する。
- 母数付き帰属漏れ0になるまでGate 3R-4は不合格。

### P2: W39の残作業

- 現行`indicators_v2`/model52経路の決定化は合格。
- 全pytestの警告ガードにより、`src/old/indicators.py:1254`と`:3439`の未seed
  `drop_ojama`が可視化された。同モジュールは`src/analyzer.py`と`src/overlay.py`から
  まだimportされるため、プロジェクト全体の非決定性は未根治。現行Gate 4経路とは分けて
  W39追補として「seed固定」または「legacy経路廃止」を決める。
- 過去の学習データ4列は削除しない。`pre-W39`として再現性上staleを明記し、保存済み盤面から
  4列だけ再計算する。動画の再DL・再認識は不要。
- 「分布が同じため再学習不要」は断定しない。既存model52は決定化後PM100/holdoutの
  再評価で一時baselineとして残せるが、次回A/B/C/D学習には再計算済み4列を使う。

### K1〜K9への回答

- **K1: 条件付き異議なし。** W38先頭化、P2-2格上げ、P1-3別レーン化は妥当。
  115.7%は独立タスクから外してよいが、P1-2の検収証跡として削除せず残す。
- **K2: Codexが規約を確定。** SPEC §4.1.3へ、独立score分子、実効レート、side別leftover、
  全消し二重加算禁止、試合境界reset、母数、chain単位残差説明を追記した。
- **K3: 方式の方向は合意、現設計は未承認。** 境界ワイプ量カウンタ追加後に再レビューする。
- **K4: 対応済み。** SPEC §6.3へ廃止・使用禁止注記を追記した。
- **K5: ID統合は不要。** 説明済み継続分割を許容する。ただし二重計上0、push traceとの
  1対1、ID数を物理連鎖数として下流利用しないことが条件。
- **K6: 現状不承認。** 3.78秒の時間単独吸収をやめ、同一性条件と負例を追加する。
- **K7: 含める。** 6件を裁定し、確定2件を閉じるまでGate 3R-4不合格。
- **K8: 合意、さらに厳格化。** 3run bit-identicalを前提とし、中央値/rangeは診断用で
  合格の代替にしない。W39前PM100は全8区間取り直す。
- **K9: 台帳漏れを確認。** W39登録は妥当。現行4列だけでなくlegacy未seed経路と
  pre-W39学習列の扱いを上記のとおり追加する。

### Gate 3R-6

対象を「state machine遅れ（画面では連鎖中だがown state=STABLE）」へ差し替える。
案Aの非STABLE中holdはuser承認済みガードとして維持するが、残存7行中6行がSTABLEなので
単独では合格にしない。掛け算式実読・score増加・盤面減少の先行信号からリアルタイムで
CHAINへ遷移する入口を計装し、先頭5試合通し・真窒息見逃し0・simulate fallback別母数で検収する。

### 再開順序

1. P1-3設計へ`total_boundary_wiped_uncapped`を追加し、T1〜T9の失敗テストを先行。
2. K6の時間単独echo吸収を狭め、Codex負例を追加。
3. K7候補6件を裁定し、確定欠落2件を根治。
4. 1〜3完了後にGate 3-2bを再レビュー。合格前に配線・production登録へ進まない。
5. Gate 3R-5既定OFF統合後、Gate 3R-6 state遅れを閉じる。
6. W39決定化後のPM100全8区間を再生成し、Gate 4を判定する。
7. pre-W39の4列を保存盤面から再計算してからモデルA/B/C/D比較へ進む。

## 2026-08-25 Claude制限後のCodex引継ぎ — P1-3/K6実装、K7訂正

### 実行中プロセスの扱い

今回のGate関連ジョブは0件だった。8/18開始のDL再取得補助と監視ループは別作業であり、
停止・上書きしていない。`src/production_config.py`、既存`data/verify/gate3_*`、
オーバーレイ配線にも触れていない。

### P1-3

`src/ojama_accounting.py`へ既存snapshotとは独立した`GrossOjamaCounters`を追加した。
side別に生成、cap前自己相殺、cap前着弾、境界ワイプ量、境界回数、clamp lossを累積する。
境界ではreset前の`forecast_incoming_uncapped`を量カウンタへ加算してから0に戻すため、
consumerが境界フレームを読み飛ばしてもワイプ量を復元できる。

`src/exchange_episode_tracker.py`へ`classify_gross_counter_delta`を追加した。カテゴリ別差分を
`SettlementObservation`へ変換し、両sideのpending保存則残差と検査母数2を返す。
旧`classify_pending_uncapped_delta`は後方互換のため残すが権威には使わない。

地上真値fixtureでは、同時応酬pending 100→80を相殺50/相手生成30へ分離、
着弾30+incoming10、cap超え517個、境界frame skipでwipe40、clamp loss明示を確認。

### K6

`_is_settled_echo`は時間窓だけでなくchain_count一致を必須にした。formula成長観測済みの
tailでは同じ観測経路の累積点一致も要求する。`finalize(cc=8)→0.2秒後settled(cc=1)`と、
同段数・異累積点の負例はいずれも2 chainとして残る。既存真のecho正例は維持。

### K7の重要訂正

前レビューで「push観測0/190の確定欠落」としたt=893.53/985.03は、比較条件が違った。

- 根拠側: 本番オーバーレイ構成、`force_in_match=False`、production認識フラグ群、
  `_active_chain_1p`を観測。
- 190件側: 簡易Gateプローブ、`force_in_match=True`、限定フラグ、
  `PipelineResult.side.chain_event`を観測。

同じ本番構成・同じフレームで内部と公開結果を同時に再測定した結果:

- t=893.533: internal/publicとも`baseline, cc=1, total_score=100`
- t=985.033: internal/publicとも`baseline, cc=2, total_score=360`

よってproducer→公開結果の実欠落ではなく、異なる構成間比較による測定器事故だった。
この2件を修正対象として認識本体へ変更を入れるのは誤り。

残る4候補も時系列を再照合した。t=809.767は2P cid2 (790.10〜810.40)、
t=874.367は1P cid6 (860.13〜874.82)、t=1033.30は1P cid21
(1014.98〜1033.42) の時間内にあり、別物理連鎖でなく大連鎖末尾のbaseline断片。
t=826.933は本番構成では公開結果に出る一方、簡易構成ではcid4=40点という異なる
観測になっており、producer欠落でなく構成差。したがって候補6件から認識本体の
確定欠落は0件。ただし、この発見により「本番ログの物理連鎖」と「簡易プローブのID」を
突き合わせた既存対応表そのものが同一条件比較ではないと判明した。Gate証跡としては
同一本番構成で対応表を取り直す必要がある。

### テスト

- P1-3/K6関連3ファイル: **182 passed / 0 failed**。
- 全pytest: **5,976 passed / 13 skipped / 1 deselected / 0 failed**
  （1,054.92秒）。legacy未seed警告は既知W39追補で、今回の新規失敗は0。

### 次に行うこと

1. 全pytest完了確認。
2. 既存v4を上書きせずgross経路の実データプローブを作り、保存則残差を母数付きで確認。
3. K6の全吸収を真陽性/偽陽性監査（zenchi既存190 pushの再生では1/30が
   cc=3/1260点の完全一致、resolved 26と既存値を維持）。
4. zenchi対応表を同一本番構成で取り直し、構成差による見かけの欠落を排除する。
5. 以上が0件条件を満たしてからGate 3-2bを再判定する。

## 2026-08-25 14:20 JST — Codex最終検収 / Gate 3R-4 PASS

### 結論

`Gate 3-2b / Gate 3R-4 PASS`。次はGate 3R-5の既定OFF配線。
production登録・本番ON・PM100本番候補化はまだ禁止する。

### P1-3 実データ

新規v5 grossプローブを既存v4とは別ファイル・別出力先で実行した。
zenchi本番30fps条件は9,000 frame / 18,000 sideで、保存則の非0残差0、
残差絶対値合計0、最大0、未分類frame 0。境界ワイプ3,385、clamp loss 0。
ledgerの`retired_unreconciled=3,385`は境界ワイプ量と一致し、黙った消失ではない。
60fps全frame条件でも18,000 frame / 36,000 sideで残差0を再確認した。

### K6

`_is_settled_echo`は時刻に加えてchain_count一致を必須化し、formula成長済みtailは
累積点一致も必須化した。cc=8→別cc=1、同cc→異累積点の負例は吸収0。
既存zenchi実測の唯一の吸収1/30はcc=3・累積1,260点が一致する真のechoで維持。

### K7 / 同一構成の訂正証跡

簡易v5と旧overlayでは未採用kwargsの既定値まで完全一致しないため、それだけでは
K7証拠にしなかった。`scripts/_probe_formula_interlude_2026-08-24.py`そのものを包み、
同一`update()`呼出後の`_active_chain_*`と`PipelineResult.side.chain_event`を同時採取した。

- 物理16連鎖の真値`(chain_count,total_score)`は公開resultで16/16一致。
- 6候補（809.767 / 826.933 / 874.367 / 893.533 / 985.033 / 1033.300）も
  internal/publicが各時刻で完全一致。
- 既知mechanismの公開遷移157件のうち、internal/public両方非null 158遷移は
  値不一致0。internalのみ17件はすべて`landing`かつ0点で、生成会計対象ではない。

したがって「push観測0/190の確定欠落」は異構成比較による測定器事故であり、
認識本体の欠落ではない。K7の逆方向帰属漏れは0として閉じる。

### 回帰

- 局所: 182 passed / 0 failed。
- 全体: 5,976 passed / 13 skipped / 1 deselected / 0 failed。
- `src/production_config.py`、既存監査成果物、オーバーレイ配線は未変更。

### Claude再開時の順序

1. 本節、`CURRENT.md` 14:20節、`PLAN.md` Gate 3R-4を読む。
2. Gate 3R-5としてgross経路をオーバーレイへ**既定OFF**で配線し、timeline列を追加。
3. OFF bit-identicalと全pytestを確認。
4. Gate 3R-6の「画面連鎖中なのにSTABLE」state遅れを計装・修正。
5. Gate 3R-6合格前にPM100本番候補、production登録、本番ONへ進まない。

## 2026-08-25 19:50 JST — Gate 3R-5 境界ワイプ列修理 PASS

### 結論

Claude第19〜22報の引き渡しを受け、Gate 3R-5の境界ワイプ列を修理した。
`Gate 3R-5 PASS`へ戻してよい。production登録・本番ONは引き続き禁止する。

### 修正

`scripts/visualize_advantage_overlay.py`の試合境界で、gross dump有効時だけ
`OjamaAccountingTracker`を`_fresh_trackers`へ引き継ぐoptional経路を追加した。
境界フレームは後段の既存`_drive_ojama`が一度だけ処理するため、旧pendingが
`boundary_wiped_uncapped_*`へ加算され、累積カウンタと前回pendingを失わない。
他の評価トラッカーは従来どおり境界で再生成する。OFF時は新規会計tracker生成の
従来経路を維持する。

`tests/test_advantage_overlay_timeline_dump.py`へ、非ゼロpendingを持つtrackerを境界越しに
保持し、`gross_wiped_p2 > 0`、検査母数2、両side保存則残差0を同時に固定する回帰テストを
追加した。`src/death_confirmation.py`と`tests/test_death_confirmation.py`には触れていない。

### 実動画検収

zenchi先頭600秒、18,000 frame、9 game_idx、7,206 dump行を、既存成果物とは別の
`data/verify/gate3r5_wipe_repair_codex_2026-08-25/dump_on.npz`へ出力した。

- 検査side: **14,410**
- 保存則残差非0: **0/14,410**、最大絶対残差0
- 境界ワイプ量: **1,747**（非ゼロ7行、1行最大612）
- clamp loss: **0**

スクリプト自己集計とは別にnpzを再読込して同値を確認した。列の存在だけでなく、
実際に非ゼロ値を持てることを検収済み。

### OFF / 全回帰

- OFF 3run: 1,075行、27キー、全3ペア=**81比較で不一致0/81**。
  gross列はOFF時に存在しない。
- 全pytest: **6,039 passed / 13 skipped / 1 deselected / 0 failed**
  （1,573.70秒）。
- `src/production_config.py`差分なし、本番ONなし。
- 実行中のGate 3R-5/pytestプロセス0。

### Claude再開時

Gate 3R-5の再実装・再測は不要。上記成果物と本節を証跡として、Gate 3R-6の
残作業と独立レビュー依頼へ進むこと。Gate 3R-5/3R-6合格前にproduction登録・
PM100本番候補化へ進まない。

## 2026-08-26 13:36 JST — Gate 3R-6 最終PASS / Gate 4正式再測定へ

### 結論

`Gate 3R-6 PASS`。Claude第26〜28報と追記を独立再検収し、死亡確定、正式境界、
決着ホールドの物理終了信号、密な表示時系列まで閉じた。Gate 4の正式測定を開始したが、
production登録・本番ON・採用判断はまだ禁止する。

### 死亡確定と正式境界

- 真の窒息2P t=223: `True 4/118`。
- 既知誤判定1P t=164.03〜164.73: `True 0/22`。
- 待受画面2P t=18〜90.5: `True 0/1121`。
- 境界確定3件はt=232.467 / 278.100 / 335.967で敗北側と一致。
- 正式境界6件、`game_idx`加算6件。低得点継続だけでは重複加算しない。
- 正式境界統合で増えた212行は207/212が試合冒頭2〜3秒、
  TSUMO_FALL/STABLEが174件、±100張り付き0、試合数不変なので受理。

### OFF互換と全回帰

- OFF独立3run: 全3ペア不一致`0/27`。
- ON/OFF同一窓: 不一致`0/27`。
- grossのみ/gross+death: 不一致`0/42`。
- 全pytest: **6,107 passed / 13 skipped / 1 deselected / 0 failed**。
- 静的dependency台帳に欠けていた
  `--resolved-absolute-chain-end -> --resolved-exchange-eval`だけをテスト側へ追加。
  `src/production_config.py`は未変更。

### 決着ホールドの根治

絶対終了信号は、そのsideが実際にCHAINへ入った場合だけ有効とした。基準nextは
セッション開始直後でなく`CHAIN_MIN_DISPLAY_SEC`経過後+2frameで採る。
NEXT/slide/新ツモ/おじゃまに加え、CHAIN再進入と40点以上の得点増分を継続証拠とし、
1.5秒のactivity quietを満たすまで解除しない。解除後はneutralになるまで再armしない。

実窓v10では旧早期解除t=1713.033は0件。ホールドは28.23秒、1.83秒、45.03秒の3件。
前2件は最終40点以上得点から1.533秒/1.033秒で解除し、解除後1.5秒以内の40点以上増分0。
最後は13連鎖相当の得点継続中で、正式試合境界により終了した。長時間でも
物理交換の継続であり、時間上限による強制解除は入れていない。

密なdisplay timelineは4,551行、最大間隔0.0333秒、欠測0。
画面品質の母集団はこれを正とし、settled-only timelineは会計・認識診断に限定する。

### Gate 4

固定snapshotとmanifestを作成し、条件1→3→2→4、各8区間を最大3並列で正式測定中。
出力は`data/verify/gate4_formal_dense_2026-08-26/`の新規領域。
旧settled-onlyの14.0%/3.3%/40回/7試合およびpre-gate値は合否に使わない。
条件5の資産監査では、交換台帳コア、ChainIdResolver、gross供給、保存則回帰は存在する一方、
オンラインsnapshot、overlayアダプタ、ライブ/hold両方の未解決hard-override禁止、必要dump列、
条件5 runner・一括検証器が未実装と確定した。既存コアを再実装せず、これらの統合層だけを
既定OFFの別レーンで追加する。区間端OPEN、forced close、settlement入力なし、oversettled、
帰属漏れ、post-close settlementを各0/Nで分離し、D1対象外を黙って捨てない。

user決定により、先頭5試合レビュー動画は今は確定せず、Gate 4完了時の構成で
別名・別出力先へ生成する。途中生成物は完成品として扱わない。
## 2026-08-26 15:29 JST — Codex主導 Gate 4進捗（Claude再開時の同期用）

- Gate 3R-6 PASSは維持。production登録・本番ONはしていない。
- Gate 4条件1は8/8完了。密display基準は張り付き6.42%、逆符号1.95%、反転18試合、
  急変27、gap異常0。条件3（規模比較Aのみ）を固定snapshotで3並列測定中。
- 条件5のlive台帳配線・hard overrideゲート・sidecar・runner・検証器はCodexが実装した。
  旧ChainGenerationAccumulatorとは排他、既定OFF。
- 実動画で見つけたmax_sec直後の決済欠落は、同一frameの原子反映とCLOSED_FORCED要約への
  backfillで修理。smokeでは重複0、post-close dropped 0、gross残差0/3,600 side、
  違法hard override 0。正常CLOSE母数は短窓0なので8区間で最終裁定する。
- Gate 4の勝者正解はWIN★パネル差分を使用する。seg04 game17は画像証跡で2P勝利。
- Claudeに依頼する場合も、既存差分・成果物をreset/checkout/stash/削除/上書きしない。
  `src/production_config.py`は変更しない。現在はCodexが書込み主担当。

## 2026-08-26 17:32 JST — Gate 4 / 条件5v2同期

- 条件3（規模比較のみ）は8/8完了。条件1比で急変27→24、決着逆方向23→21、
  真の致死弱化0。一方、張り付き6.42%→9.93%、逆符号1.95%→2.79%、
  反転延べ23→25と悪化し、単独採用は未決定。
- WIN★正解は実試合109件。UNKNOWNは試合外/区間境界3行のみ。
- 条件5は独立レビューで見つかったP1/P2をすべて回帰化して修理した。
  None gap、遅延FINALIZE/settlement、normal/max_sec/side_wipe、古い要約backfill、
  global/episode/retired保存則、未解決hard overrideを対象にし、最終独立判定は合格。
- 全pytestは6,160 passed / 13 skipped / 1 deselected / 0 failed。
  正式snapshotは`data/verify/gate4_condition5_2026-08-26/
  _snapshot_cond5_codex_20260826_v2`。先行v1は不採用証跡として無変更保持。
- 現在、条件2を1本、条件5v2を2本の計3本で正式実測中。条件5完了後に
  条件2を2〜3並列へ戻し、その後条件4。production_configは未変更。

## 2026-08-26 17:44 JST — 条件1〜5と最終統合の扱い

- user確認どおり、Gate 4は条件1〜5をすべて同じ全8区間で正式比較する。
- 1〜5は原因切り分け用の比較構成で、全フラグの無条件同時ONではない。
- 条件5の交換episodeは条件3の旧`ChainGenerationAccumulator`と排他。重ねると
  二重会計になるため条件3+5は作らない。
- 条件2のヒステリシスは条件5と併用可能。両方が単独測定で有効なら、
  `条件5+ヒステリシス`を同じ全8区間で最終互換性確認する。
- 先頭5試合レビュー動画は、上記互換性確認まで通った最終統合構成で作る。

## 2026-08-26 18:36 JST — 条件5v2の実データNGとv3修理

- v2正式seg01で、gross保存則0/53,620 side、global保存則0/26,811行、
  違法hard override 0。一方、後着更新同期1/5とnormal close後未照合1/3でNG。
- 同期違反は検証器の比較軸誤り。`unreconciled`は下方FINALIZE保留差も含むため、
  物理保存残量`ledger_residual_all`とclosed要約を比較するよう変更した。
- 実不具合はt=448.800〜448.967の遅延FINALIZE/着弾。normal closeの生成34を35へ
  backfillした後、着弾35が0.167秒遅れていた。I7を維持するため要約を
  `CLOSED_FORCED/late_finalize_after_normal_close`へ再分類し、第二episodeは開かない。
- 関連102テスト、全pytest 6,161 passed / 13 skipped / 1 deselected / 0 failed。
- v3 snapshotは`data/verify/gate4_condition5_2026-08-26/
  _snapshot_cond5_codex_20260826_v3`。v3 seg01を再測定中。v1/v2・旧seg01・部分ログは保持。
- v3 seg01再測定は正式検証PASS。後着同期0/5、normal終了後未照合0/2、
  gross保存則0/53,620 side、global保存則0/26,811行、違法hard override 0。
  v2→v3は表示13/13列・通常timeline 44/44列bit-identical、条件5 sidecarの
  意図した状態3/72列だけ変化。v3 seg02〜04を1並列で継続中。
- 条件2途中5/8は表示揺れ4項目を改善したが、seg05 game2（WIN★=1P）で
  最終表示+60.745→+20.657の真の致死弱化1件。必須0件を破るため単独は暫定NG。
  条件2+5の追加統合は、条件2が正式PASSした場合だけという既定順序を維持する。
- 条件2は7/8完成後、Codexが実行中runner自体を編集したため親shellの後半読取が
  構文エラーになりseg08だけ開始直後終了。完成済み7区間は無影響、失敗log保持。
  同一snapshot/flagsの専用retryでseg08を再実行中。現在はv3 seg02 / cond2 seg08 /
  cond4 seg01の3枠。以後実行中runnerは編集しない。

## 2026-08-26 20:56 JST — 条件2正式判定 / 条件5v4

- 条件2は全8区間完了。張り付き6.42%→5.87%、逆符号1.95%→1.74%、
  反転18試合/23回→14試合/16回、急変27→22だが、決着逆方向23/109は不変。
  seg05 game2の真の致死弱化1件が残るため正式NG。条件2+5は測定しない。
- 条件5v3 seg03は`active provisional decreased: chain=52 1.0 -> 0.0`で停止。
  同一連鎖の累積得点OCR下振れが追加専用台帳へ減算を要求したのが根因。
- 累積点は物理的に単調なのでresolverでrunning maxを保持する。下振れは黙って捨てず、
  `provisional_score_decrease_ignored_count / formula_step_observation_count`を
  条件5 sidecarへ追加した。最小再現は修正前2 failed、修正後2 passed、関連191 passed。
- v4 snapshotは`data/verify/gate4_condition5_2026-08-26/
  _snapshot_cond5_codex_20260826_v4`。v1〜v3の成果物・失敗logは保持。
- 現在3枠: 条件4 seg03/04、条件5v4 seg01。production_config未変更。

## 2026-08-26 23:36 JST — 条件4正式NG / 条件5v5

- 条件4は8/8完了。急変27→21、反転18試合/23回→15試合/20回は改善したが、
  張り付き6.42%→9.32%、逆符号1.95%→2.58%、決着逆方向23/109不変、
  真の致死弱化1件のため正式NG。
- 条件5v4 seg02で後着同期1/35と重複抑制1件。前者は同frameの旧要約決済と
  新chain生成を混ぜた検証器誤検出、後者は旧episodeへの後着をOPENな新episodeにも
  touchした実不具合。
- backfillイベントを新episodeへtouchしない修正と、close済みchainだけの符号付き
  outstanding差監査列を追加。最小再現は修正前2 failed→修正後2 passed。
  関連150+resolver 43=193 passed。
- v5 snapshot `data/verify/gate4_condition5_2026-08-26/
  _snapshot_cond5_codex_20260826_v5`で全8区間を3並列再測定中。
  v1〜v4成果物・失敗/部分logは無変更保持。production_config未変更。

## 2026-08-27 01:18 JST — 条件5v5 NG / v6

- v5は7区間先行検証で後着同期3/55、未帰属2区間、重複1区間のためNG。
- close後の同一chain STEP成長が旧要約へbackfillされず、新episodeへtouchされていた。
  FIRE/STEPも旧要約だけへ帰属し、遅延成長backfill/droppedを監査列化した。
- 正式試合境界でchain退役後、同frameの予告消失settlementを再適用していたため、
  境界frameではsettlementを除外し、境界母数・除外件数・量を別列へ出す。
- 最小再現は修正前2 failed→修正後2 passed。関連152+resolver43=195 passed。
- v6 snapshot `_snapshot_cond5_codex_20260827_v6`で全8区間を3並列再測定中。
  v1〜v5とseg08部分logは保持。production_config未変更。

## 2026-08-27 05:34 JST — 条件5v6正式判定 / v8

- v6会計は全8区間でPASS。gross保存則0/422,014 side、global保存則0/211,008行、
  後着同期0/88、未帰属・重複・dropped・違法hard overrideはすべて0。
- 表示は張り付き4.33%まで改善したが、急変30、決着逆方向25/109、真の致死弱化6でNG。
- 根因は表示ゲート2件。仕様上は±100完全上書きだけが対象なのに`after != before`の
  全補正を止めていた。また`allows_hard_override=True`が許可方向を持たず、物理的に
  確定した勝者と逆向きの±100も通せた。
- v8は部分補正を残し、完全上書きだけを方向付きで制御する。sidecarに
  `hard_override_target`を追加し、未解決・許可・方向の矛盾を全行検査する。
- 関連351テストPASS。短窓はv8で方向状態矛盾0/1,800行、違法適用0。
  v8 snapshot `_snapshot_cond5_codex_20260827_v8`で全8区間を3並列測定中。
- Gate 4は引き続きHOLD。production_config未変更。v1〜v7の全証跡を保持。

## 2026-08-27 06:11 JST — 条件5v8先行NG / v9

- v8先頭2区間は致死弱化0、表示揺れ4項目改善だが、決着逆方向5→8。
- seg02 game8で台帳`net_raw=-553`に反して表示+70。旧累積器を排他OFFにしたのに、
  置換先である台帳純残量を`kill_override`入力へ渡していない統合漏れを確定した。
- v9は正の純残量を2P受け、負を1P受けへcapなしで配線する。関連300テストPASS。
- v8は2区間完成時点で停止し全証跡保持。v9 snapshot
  `_snapshot_cond5_codex_20260827_v9`で全8区間を3並列測定中。

## 2026-08-27 06:47 JST — 条件5v9先行NG / v10

- v9 seg01は元の急反転を抑えたが、実勝者1Pの最終表示+100→+25.941で致死弱化。
- t=221.867で物理勝者target=+100は正しく出ており、ゲートが逆向き候補を止める
  だけでtargetを表示へ採用しない統合不備だった。
- v10は未解決中の物理勝者targetを確定値として直接適用する。短窓で+100収束、
  逆方向適用0、物理方向訂正374/374、監査矛盾0/1,800行。関連301テストPASS。
- v9はseg01完成時点で停止・保持。v10 snapshot
  `_snapshot_cond5_codex_20260827_v10`で全8区間を3並列測定中。

## 2026-08-27 07:39 JST — 条件5v10先行NG / v11

- v10先頭3区間は致死弱化2件。未解決±100を止める際に方向まで生モデルへ戻し、
  台帳方向が正しい場面でも最終表示を-21.756/+1.509へ弱めていた。
- v11は仕様どおり方向を保持し、完全上書きだけを既存上限±90へ丸める。
  同方向の生モデル値が既に±90超なら弱めない。関連302テストPASS。
- v10は3区間完成時点で停止・保持。v11 snapshot
  `_snapshot_cond5_codex_20260827_v11`で全8区間を3並列測定中。

## 2026-08-28 — 次世代判定設計の壁打ち統合（Claude再開時の必読資料）

ユーザーとCodexの後続壁打ち、およびClaudeの読取専用レビューで得た合意を、次へ統合した。

`docs/ADVANTAGE_MODEL_DESIGN_AGREEMENT_2026-08-28.md`

コンテキスト圧縮前を含む提案・異論・撤回・保留の発言順は、保存されたチャット原本から次へ復元した。

`docs/ADVANTAGE_MODEL_WALL_DISCUSSION_TURN_BY_TURN_2026-08-28.md`

上記はユーザー可視の327発言と圧縮7地点を一件ずつ記録した原文順資料である。次は論点別索引として使う。

`docs/ADVANTAGE_MODEL_WALL_DISCUSSION_RECONSTRUCTION_2026-08-28.md`

Claudeが次の学習データ・指標・未来探索を扱う前に、同資料を最初から最後まで読むこと。
実装仕様は統合版を正とし、復元記録に残る撤回済み初期案を採用しないこと。
特に次は旧案から更新されている。

1. 出来事の4時刻を全て必須にしない。物理発生区間と利用可能時刻を必須とし、
   後の確定は元記録の書換えでなく訂正記録にする。
2. 最初の補助学習は旧5群同時投入でなく、「次の物理的出来事」と
   「相殺後に送られる量」の限定した一組から始める。
3. 盤面上の総ぷよ数は局面の重要な観測値として保持し、多い側、両者合計、差、
   色ぷよ・おじゃま内訳を回帰確認する。18個刻みは初期仮説で固定値ではない。
4. 未来探索は物理解決、一手、見えている範囲の複数手、未知色・長手数近似へ分ける。
   相手の行動仮定をまだ固定せず、候補分布と計算費用を保持する。
5. リアルタイム性は副次条件でなく合格条件である。毎フレーム深く探索せず、
   状態変化時だけ計算し、古い計算を中止・破棄する。
6. 人向け確認資料は一画面・一場面・一論点、短い映像、変更前後、平易な根拠とし、
   スマートフォンから確認できる形式にする。
7. 48/49マニフェストは候補割当であり、c58/c69の共通品質合格前に最終固定とみなさない。
   入替時は旧表を上書きせず、新しい版を残す。
8. 正式境界通知元は一つにし、最終勝敗ラベルには公式勝者表示だけを使う。
   物理的死亡で欠けた勝者を補わない。

既存 `docs/EVENT_DATASET_DESIGN_2026-08-28.md` 冒頭にも後続合意への注意を追加した。
現在の `src/event_data_contract.py` は旧4時刻設計を基にした試作なので、新規実装を進める前に
統合版との差分レビューが必要である。

この追記は情報同期だけであり、現行モデルの削除、重み変更、`src/production_config.py`登録、
48動画の一括処理、49動画利用、配信接続を指示しない。既存差分と証跡を変更しないこと。

## 2026-08-29 — 出来事保存仕様v1 最終PASS

- user決定により以後の技術設計はCodexが担当し、節目でClaudeへ独立レビューを依頼する。
- 具体仕様は `docs/EVENT_SOURCE_SCHEMA_V1_2026-08-29.md`。
- Claude読取専用レビュー3回を実施。第1回P0 1件/P1 7件、第2回新規P1 1件を反映し、
  第3回で全件CLOSED、新規P0/P1なし、最終PASS。
- 事後確認版は `data/audit_views/` へ隔離し、`model_input_allowed=false` を必須とした。
- 次は最小書込み・読取・検査層と独立3試行の決定性試験。まだ実装・動画処理は開始していない。
- Claudeが次に引き継ぐ場合、旧 `src/event_data_contract.py` を正本とせず、上記具体仕様を優先すること。

## 2026-08-29 — 出来事原本v1 最小実装の修正後レビュー依頼

- 対象: `src/event_source_v1.py`、`tests/test_event_source_v1.py`、
  `docs/EVENT_SOURCE_SCHEMA_V1_2026-08-29.md`。
- 第1回コードレビューはP0 0件/P1 2件。仕様例の `record_kind` と、
  出来事行本体の改ざん試験を修正済み。
- P2の復旧再実行性、空部品、境界値、未知拡張キー、ネストされた保存先除外、
  観測時刻の片欠落、定数化も反映した。
- 新旧出来事関連55件PASS、1関数50行超過0。
- 本番配線、manifest/COMPLETE、実動画処理はまだ行っていない。
- P0/P1の再確認と、今回の修正で新しい欠陥を入れていないかの独立確認を依頼する。

## 2026-08-29 — 確定盤面アダプター・実行器レビュー依頼

対象は `src/event_snapshot_adapter_v1.py`、`src/event_snapshot_export_v1.py`、
`scripts/run_event_snapshot_pilot_v1.py` と関連試験。関連121件PASS、関数50行超過0。

実動画を各3試行する前に、生成内容IDの十分性、observed選別、fpsとframe_idx、
上書き防止、完了印、仕様との食い違いをP0/P1中心に確認してほしい。
レビュー中は編集せず、指摘はfile:line、再現条件、最小修正案つきで返すこと。

## 2026-08-29 — 攻撃会計・訂正・再採番参照の最終レビュー依頼

読取専用でP0/P1を中心に独立レビューすること。編集、コミット、本番設定変更、動画処理は行わない。

対象:

- `src/ojama_accounting.py`
- `src/event_accounting_observer_v1.py`
- `src/event_accounting_adapter_v1.py`
- `src/event_accounting_pilot_validation_v1.py`
- `src/event_observation_adapter_v1.py`
- `src/event_snapshot_export_v1.py`
- `scripts/collect_boards_lean.py`
- `scripts/run_event_snapshot_pilot_v1.py`
- `scripts/reexport_event_snapshot_pilot_v1.py`
- `scripts/verify_event_accounting_pilot_v1.py`
- 対応する`tests/test_event_*.py`、`tests/test_collect_boards_lean.py`、`tests/test_ojama_accounting.py`

特に確認すること:

1. 非消費観測が会計本体を変えず、累積差分とpendingの保存則を正しく検査するか。
2. 暫定攻撃が会計量を動かさず、`simulate`値を量へ昇格させず、得点確定が同じ攻撃IDを
   `confirms`/`corrects`するか。
3. 生成=自己相殺+送付、送付量IDのFIFO相殺/着地/境界消失、同一フレーム順が妥当か。
4. 二回以上の統合・再採番でも`target_event_ids`と`verified_cause_event_ids`が正しい対象へ追従するか。
5. 完了試行の独立検証が、実装と同じ式をなぞるだけの自己確認になっていないか。
6. 全消し内訳、連鎖開始、着地盤面差を未観測として残す扱いが、確定会計を偽っていないか。

実測証拠:

- `data/verify/event_source_v1_accounting_pilot_2026-08-29/results/accounting_combined_validation.json`
- `data/verify/event_source_v1_accounting_pilot_2026-08-29/results/observation_combined_validation.json`
- 通常+難条件で24,600 side観測、確定20、同値確認8、値訂正12、訂正対象75、
  保存則違反0、未帰属0。
- 関連520件PASS。全pytest 6,420 passed / 13 skipped / 1 deselected / 0 failed。

出力は、各指摘に重大度、file:line、最小再現、影響、最小修正案を付ける。
指摘がなければP0/P1なしと、残るP2/既知限界を分けて明記する。

## 2026-08-29 — 前回P1×4の修正後再レビュー

前回指摘を全件受諾し、次を修正した。読取専用で再確認すること。

- 会計観測を勝者判定から専用既定OFFスイッチへ分離。
- 確定攻撃IDと途中連鎖候補IDを分離。複数候補は曖昧とし単一IDを主張しない。
- 暫定102件を確定参照92件と未確定終了10件へ全数分類。未処理0。
- 保存則残差を実測し、量IDを独立FIFO再構成。自己生成チェックを削減。
- 標本被覆率、物理発生時間窓、同一フレーム物理順、公式試合範囲の再採番追従を追加。

最終証跡:
`data/verify/event_source_v1_accounting_pilot_p1fix2_2026-08-29/results/accounting_combined_validation.json`

P0/P1が閉じたか、新しいP0/P1/P2をfile:lineと最小再現つきで返すこと。

## 2026-08-29 — 会計出来事基盤の独立レビュー最終結果

- 連続したレビュー指摘をすべて受諾し、最終版を`p1fix11`へ別名出力した。
- 最終レビューで、同一フレームの送付→境界失効が再採番で逆転するP1を合成ケースで確認し、
  `accounting_source_order`を再採番後も維持するよう修正した。
- 修正後のClaude読取専用レビューは当該P1をCLOSED、新規P0/P1なし。
- 残ったP2の「末尾境界後にオンライン出来事がないと公式試合範囲が逆転する」ケースも、
  根拠のない範囲を作らず割当を省略する回帰として閉じた。
- 実動画の同一フレーム送付+境界失効は0/3失効件であり、その修正の証拠は合成回帰である。
  実動画2本PASSを当該分岐の証拠とは主張しない。
- 最終実動画検証:
  `data/verify/event_source_v1_accounting_pilot_p1fix11_2026-08-29/results/accounting_combined_validation.json`
- 本番設定、48動画、49動画、学習は未変更。次は最小追加観測の設計へ進む。

## 2026-08-29 — 物理着地と交換会計v2の最終独立レビュー依頼

読取専用で、次の新規実装をP0/P1中心にレビューすること。編集、コミット、動画処理、
本番設定変更は行わない。

- `src/event_physical_observer_v1.py`
- `src/event_physical_adapter_v1.py`
- `src/event_exchange_reconciliation_v2.py`
- `scripts/verify_event_exchange_reconciliation_v2.py`
- 対応する `tests/test_event_physical_*` と `tests/test_event_exchange_reconciliation_v2.py`

特に確認すること:

1. 相手連鎖中の非確定な盤面変化を着地と誤認せず、確定済み残量や直接着地証拠を隠さないか。
2. 途中の曖昧な変化を、同一連鎖の後続成長で将来行だけ回復する因果処理が妥当か。
3. `observation_trusted`（その行の直接観測）と `trusted`（交換残量・相殺結果）の分離が、
   不確実な値を学習可能と誤表示しないか。
4. 生成=相殺+実着地+境界失効+残量の保存則、試合境界、途中切出し、同一フレーム順に穴がないか。
5. 既存の既定OFF経路や本番設定へ影響していないか。

難条件の最新証跡:
`data/verify/event_source_v1_physical_pilot_2026-08-29/results/zenchi_0_260_pphys10_exchange_reconciliation_v2_p13.json`

結果は重大度、file:line、最小再現、影響、最小修正案を付ける。指摘がなければ、
P0/P1なし、残るP2・既知限界、次の48動画監査へ進めるかを分けて明記する。

## 2026-08-29 — 物理着地と交換会計v2のレビュー完了

- P1「期待着地0のとき、確定盤面の説明不能な正方向増加を不確実化しない」を受諾し修正。
- 確定盤面の`garbage`/`occupied`正方向増加だけを対象とし、生CNNの負方向ノイズは除外。
- 同一フレーム順を収集順どおり定数化し、回帰で固定。
- Claude再レビューでP1 CLOSED、新規P0/P1なし。
- 最新証跡は`zenchi_0_260_pphys10_exchange_reconciliation_v2_p15.json`。

## 2026-08-29 — 得点表示位置の自動較正・収集配線レビュー依頼

読取専用でP0/P1を中心にレビューすること。編集、コミット、動画処理、
本番設定変更、49動画利用は行わない。

対象:

- `src/score_ocr.py`
- `src/score_region_calibration.py`
- `src/recognition_pipeline.py` の `score_region_offsets` 配線
- `scripts/calibrate_score_regions.py`
- `scripts/collect_boards_lean.py` の `--score-region-calibration` 配線
- `scripts/run_event_snapshot_pilot_v1.py` の設定ハッシュ・生成内容ID配線
- `tests/test_score_ocr.py`
- `tests/test_score_region_calibration.py`
- `tests/test_collect_boards_lean.py` の関連箇所
- `tests/test_event_snapshot_export_v1.py` の関連箇所

確認点:

1. 複数時点の成績で改善が十分な場合だけ座標を採用し、通常映像を動かさないか。
2. 通常得点と連鎖中の式が同じ左右別座標を使い、探索用読取りが通常設定を変更しないか。
3. JSONの映像ID・SHA-256・採用規則検証で、別映像用設定や改ざんを拒否できるか。
4. 収集コマンド、実行時資産ハッシュ、生成内容IDへ設定JSONが明示的に固定されるか。
5. 省略時の既定動作が従来どおりで、`src/production_config.py`へ波及していないか。

実測:

- video_38: 1P 22→22、2P 19→19、採用 `(0,0)/(0,0)`。
- video_c58: 1P 0→19で`(+4,+8)`、2P 0→17で`(+12,+8)`。
- video_c69: 1P 0→23で`(+4,+8)`、2P 0→22で`(+12,+8)`。
- 通常映像の既定座標 vs 明示 `(0,0)` は不一致0/34。
- 関連354件PASS。新規・変更関数の50行超過0。

指摘は重大度、file:line、最小再現、影響、最小修正案つきで返すこと。

### レビュー結果

- Claude読取専用レビューは5確認点すべて問題なし、新規P0/P1なし。
- 既知限界は2ピクセル格子が1ピクセル単位の最良値を逃し得ること。ただし通常38番を動かさず、
  c58/c69の手測り補正を自動再現したため初期値として許容する。
- 全pytestは`6,518 passed / 13 skipped / 1 deselected / 0 failed`。

## 2026-08-29 — 交換会計v2撤回後の交差検証レビュー申し送り

- 600秒検収でv2の暫定火力即時相殺が通常1件、c58 2件の不一致を出したため採用根拠から撤回した。
- 新設`src/event_exchange_cross_validation_v1.py`は、完成原本の確定`garbage_sent`と物理盤面差を
  観測時点と事後確認へ分ける。暫定量を確定残高へ入れた件数は0。
- 最新実測は通常が高信頼3、残高利用可3、隔離0。c58が高信頼4、観測時点確定0、
  事後単一2、事後複数2、残高利用可1、隔離3、説明不能0。
- 複数連鎖と、単一連鎖でも物理発生区間が重ならない事例は確定量ラベルから隔離する。
- 最新成果物は`data/verify/event_source_v1_scorecal_first5_2026-08-29/results/`の
  `video_38_exchange_cross_v1_relation9.json`と`video_c58_exchange_cross_v1_relation9.json`。
- Claude CLIへ読み取り専用レビューを2回依頼したが、どちらも無出力のまま停止した。
  ファイル変更は0。次回利用可能時に、未来漏洩、供給容量、試合/side、隔離条件を再確認してほしい。
- 関連35件、広い関連747件PASS。全pytestは
  `6,532 passed / 13 skipped / 1 deselected / 0 failed`（404.50秒）。
- 別の独立レビューでP1「隔離候補が後続の確実な候補より先に容量を取る」と、その修正で生じた
  P1「戻した容量を複数の隔離候補が重複参照する」を検出した。高信頼と照合の強さで候補全体を
  先に並べ、一つの供給容量から全割当を一度だけ行う方式へ置換した。
- その後、全候補の未来込み優先順位が過去の観測時点割当を変えるP1を実データ2件で検出した。
  高信頼候補の観測時点割当を物理通番順で先に固定し、その残容量だけを事後確認へ使う方式へ修正。
  低信頼候補は診断専用とし、確定供給容量を割り当てない。
- `confirmed_balance_committed`は容量消費ではなく確定残高ラベルへの利用可否を表す。
  高信頼候補への割当量は通常13、c58は33、`allocated_supply_overuse_count`は両方0。
  全42候補の途中入力対完了入力で、観測時点値と割当元の不一致0。
- 同一フレームの後通番を未来扱いするため、送付利用可否と暫定連鎖の事前観測を
  `(available_frame, sequence)`で比較する。実動画該当0件のため人工逆順2件で回帰固定した。
  同一フレーム境界は出来事通番で試合を分け、境界payloadと物理候補の局所試合番号を検査する。
  物理発生区間・利用可能位置も結果へ追加した。専用と関連92件PASS、独立再レビュー中。

## 2026-08-29 — 交差検証relation9・48本処理器の最終合格

- 交差検証relation9は、独立レビューでP0/P1/P2すべて0件。48本開始可。
- 48本処理器も、独立レビューでP0/P1/P2すべて0件。48本と保持49本の分離、3並列、
  中断再開、別名再試行、二重起動防止、検証対象run一致を確認した。
- 最終全pytestは`6,583 passed / 13 skipped / 1 deselected / 0 failed`（676.90秒）。
- 今後新規取得・生成する動画は`D:\puyo_analyzer\videos\`へ用途別に保存し、削除しない。
  既存Cドライブ動画は移動・削除せず、今回の48本処理では読取り専用で使う。
- 49動画、本番設定、`src/production_config.py`は未変更。

## 2026-09-01 — 強化レビューの2問題：原因確定と一次修正

- 6分33秒は返し成功率ではない。公式第5試合が攻撃入力の隔離対象となり、盤面だけの
  退避値を表示していた。1P連鎖中に1P盤面は凍結され、2P盤面だけ更新されたため、
  約66個の飛来量を無視して2P表示が41.6%から48.6%へ戻った。
- 映像391〜398秒の直接突合でも、連鎖中は1Pであり2Pの連鎖開始はない。従来の
  「2Pが直後に返しを開始した」という説明は撤回する。
- 攻撃入力を使えず、連鎖中・正味攻撃あり・未処理量ありのいずれかが成立する行は
  `攻撃交換を確認中のため判定保留` とし、数値とグラフ点を隠す一次修正を実施。
  実データ392〜398秒は23/23行が保留、対象回帰を含む42件PASS。
- 相殺後残量問題は別に、(1)時間的な死亡確定がレビュー経路へ未配線、
  (2)非死亡時も着地後盤面を評価せず攻撃差の10%補正だけ、の2件が残る。
  第1試合は2P盤面73/78占有・死亡位置占有・残量約45でも2P17〜18%、
  第2試合は残量約263・死亡位置占有でも2P約9.65%だった。
- 既存の交換物理の純粋関数を再利用し、攻撃確定前は着地禁止、応手終了後は1ターン
  最大30個を仮想着地、残り繰越、時間的死亡確定を最優先する設計が必要。
- 詳細は`REVIEW_FINDING_GAME5_0633_2026-09-01.md`と
  `REVIEW_FINDING_RESIDUAL_GARBAGE_PROBABILITY_2026-09-01.md`。本番設定は未変更。

## 2026-09-01 — Claude設計レビュー反映と20試合×2本の再生成

- Claude指摘を受諾し、`chain_active=False`だけを根拠にした仮想着地は表示へ採用しない。
  STABLE復帰後も途中攻撃量や受け側の次手が残るため、着地を早めると返しを消す。
- 全連鎖の無条件保留も不採用。隔離され攻撃入力を使えない交換だけを保留する従来AND条件を維持。
- 死亡位置占有かつ当該side非連鎖中を新たに`death_confirmation_pending`とし、
  `窒息の可能性を確認中のため判定保留`を表示。静止盤面から即0/100にはしない。
- 表示ゲート契約と必須9列を追加し、列欠落成果物のfail-silentを拒否。
- 新旧9,624行の勝率不一致0。保留576/9,624行 (5.99%)。
  第1試合3/3、第2試合5/5、第5試合392〜398秒23/23が保留。
- 対象46件PASS。短区間の実画面で文言・グラフ切断・音声保持を確認済み。
- 20試合×2本をDドライブへ別名で生成完了。2本とも音声保持・全編デコード・
  試合範囲・グラフ・学習除外を検査し、全条件PASS。
- 出力先は`D:\puyo_analyzer\videos\review\yamada_vs_norasuke_first40_death_hold_dashboard_2026-09-01\`。
  検査報告は`data/verify/yamada_norasuke_first40_death_hold_review_v1_2026-09-01/REPORT.json`。
  本番設定と旧動画は未変更。

## 2026-09-01 — 第2試合3分33秒の攻撃差抑止を修正し再生成

- 第2試合212.2〜221.633秒では2Pへ最大約184個分の攻撃差を観測していたが、
  最大盤面量が35個以下で36個境界を越えず、評価へ渡す攻撃差が0だった。
  2P約23.5%は返し確率ではなく、攻撃差を除いた盤面評価の残存確率。
- 観測攻撃差が非0で、評価へ渡した攻撃差が0または欠損の行を
  `suppressed_attack_balance`として表示保留にした。低盤面量への未学習外挿はしない。
- v6→v7の9,624行で勝率不一致0、旧保留解除0、新規保留312。
  保留合計888/9,624、対象区間34/34行が新理由で保留。
- Claude独立レビューはP0/P1なし。本編生成可。P2の旧契約過剰保留と理由遷移監査を修正し、
  51件PASS。1〜5個の微差18行は根拠のない閾値を追加せず安全側で保留。
- 20試合×2本を別名で生成し、音声・範囲・大型グラフ・新表示契約・学習除外・
  全編映像音声デコードの全条件PASS。本番設定と旧動画は未変更。
- 出力先は`D:\puyo_analyzer\videos\review\yamada_vs_norasuke_first40_suppressed_attack_hold_dashboard_2026-09-01\`。
  検査報告は`data/verify/yamada_norasuke_first40_suppressed_attack_hold_review_v2_2026-09-01/REPORT.json`。

## 2026-09-01 — 決着後の長時間保留: 第二案の独立レビュー依頼

user決定は「未来が観測事実として確定しているなら高い勝率で表示する」。第一案では
`match_boundary_evidence`を前試合の正式終了として使ったが、このイベントは実際には次試合開始側の証拠で、
未来情報の混入と次試合への持越し余地があるためCodex独立レビューで却下した。第一案の動画・監査証跡は
削除せず不採用証跡として保持する。

第二案は次のとおり。

- 現在フレームの`やった`／`ばたんきゅー`表示をP1/P2両領域で左右対称に探索
- 3回連続一致、かつ直近2秒以内の単独死亡候補と勝敗方向が一致した場合だけ0%／100%
- 生の両側死亡候補を連鎖状態の除外前に拒否
- 有効な死亡確認待ち行だけを根拠にする
- 同一試合では後続の通常行で確定値を解除せず、次試合遷移で必ず破棄
- 次試合遷移は表示リセットと20試合動画の分割にだけ利用
- グラフ横軸は現在までに表示した点だけで決め、未来の試合長を使わない

監査結果は
`data/verify/yamada_norasuke_visual_terminal_outcome_audit_v12_2026-09-01/REPORT.json`。
全57試合で検出23件、公式勝者との方向不一致0件。先頭40試合では18件。問題となった公式第22試合は
1P 100%、第25試合は2P 100%で一致した。短区間完成動画は
`D:/puyo_analyzer/videos/review/yamada_vs_norasuke_visual_terminal_probes_2026-09-01/`。

レビュー対象:

- `src/review_terminal_outcome.py`
- `scripts/render_provisional_oof_review_v1.py`
- `scripts/_verify_review_visual_terminal_outcomes_v10_2026_09_01.py`
- `tests/test_review_terminal_outcome.py`
- `tests/test_render_provisional_oof_review_v1.py`

特に、未来情報混入、次試合への持越し、左右非対称、曖昧候補の誤確定、動画分割の重複／欠落、
監査の自己参照が残っていないかを確認してほしい。コード変更はせず、P1/P2とPASS/HOLDを返してほしい。

### 完了追記

- Codex独立レビューは最終的にP1/P2残存なしでPASS。
- 監査v12は4走査位相すべて23件、先頭40試合18件、公式勝者方向不一致0件、境界前候補18/18拒否。
- 先頭40試合の2本は各9件の確定表示が監査と完全一致し、音声・全編デコード・分割境界を含む全検査PASS。
- 完成動画は`D:/puyo_analyzer/videos/review/yamada_vs_norasuke_first40_visual_terminal_dashboard_2026-09-01/`。
- 本番設定と学習重みは未変更。以後Claudeが確認する場合は本報告を正本として読み、第一案v8/v10を採用しないこと。

## 2026-09-02 14:12 JST — Phase J設計のFable独立壁打ち依頼

user指示により、reserveキューの安全な監視と並行してPhase Jの設計を進める。
動画解析、ダウンロード、学習、実行中キューへの操作は行わず、まず読取専用の設計レビューだけを依頼する。

依頼正本: `docs/agent_coordination/FABLE_PHASE_J_ARCHITECTURE_PROMPT_2026-09-02.md`

Fable architect/reviewerで同文書を実行し、コードと既存成果物を変更せず、応答を
`docs/agent_coordination/CLAUDE_TO_CODEX.md`末尾へ全文転記してほしい。Codex側は既存資産の棚卸しと
独立案を進め、Fable応答後に反論・統合する。2026-09-02 02:04以降を正とし、攻撃差10%補正は
不採用、本番設定未変更として扱うこと。

### 16:40追記

Claude/Fableは夜まで利用できないため、Codex単独でv0.4まで前進した。上記依頼書を更新済み。
レビュー時は次を追加で全文確認し、最新v0.5への反対査読として返してほしい。

- `docs/PHASE_J_REALTIME_OVERLAY_SPEC_2026-09-02.md`
- `docs/PHASE_J_IMPLEMENTATION_PLAN_2026-09-02.md`
- `docs/PHASE_J_TEST_MATRIX_2026-09-02.md`
- `docs/PHASE_J_REQUIREMENT_TRACEABILITY_2026-09-02.md`
- `docs/schemas/puyo_overlay_snapshot_v1.schema.json`
- `docs/schemas/puyo_overlay_snapshot_v1_semantic_rules.md`
- `docs/schemas/puyo_overlay_enums_v1.json`

Codex確認済み: Schema構文・全25 local ref・canonical snapshot 8例・health 5例・22 Enum集合・48 test ID・
12 requirement ID。特に認識30fps/表示2Hzの別clock、未来評価の段階導入、レビュー成果物形式を
過去ユーザー決定から補完した。`src/`、`scripts/`、認識資産、本番設定は変更していない。

### 17:36追記

追加で次も全文確認すること。

- `docs/schemas/puyo_overlay_health_v1.schema.json`
- `docs/schemas/health_examples/`のcanonical 5例

Codex独立再検収で見つかったhealthのP0 1件・P1 4件は修正し、再検収はP0/P1残存なしでPASS。
worker faultは旧確定値を理由付きhold後に非表示、telemetry faultは評価表示を継続可能だが
runtime/healthをdegraded・監査不能とする。health error codeは版付きEnumへ固定した。
入力方式未決定でも共通化できる`CapturedFrame`契約とcapture adapter境界も仕様6.1へ追加済み。
このCapture/UI具体化をv0.5とする。

### 17:54追記

v0.5 Capture差分はCodexアーキ・検収の双方でP0/P1残存なし。最終仕様には次を含む。

- CaptureSourceだけがcapture session IDを発行し、ID変更で切断通知の有無に関係なく旧jobを全無効化。
- commitはbackend最新frameでなく、current generation発行時のimmutable入力snapshotと照合。
- read-only leaseは単一consumerが`finally`で1回releaseし、画像をfan-out/process間へ渡さない。
- OBSはPhase J表示を含まないclean feedだけを許し、自己再帰0をgate化。
- A/Bは独立正解盤面、match/formal boundary clusterのseed固定paired bootstrap、coverage・遅延・
  sample数・採否優先順位を実行前manifestへ固定。

Fableはこの前提へ反対査読し、特にユーザー回答後のWindows capture backend選定を確認してほしい。

## 2026-09-20 Codex: pytest修復と保存済み診断の照合

添付の依頼1〜6を実施。予測入力観測の`g3_native_scope.py`とDのs3系は非干渉。
詳細は `docs/agent_coordination/PYTEST_REPAIR_2026-09-20.md`。

- 単独失敗31件の原因は旧SHA、本番フラグ二重供給、物差し・可視化の転送漏れ。
  旧SHA7本＋定数参照4本を依存値契約へ分離した。過去SHAの来歴、dataset/model/codeの照合は保持。
  新しい票は元ソースを保持し、読み手がSHA・値・型を再計算する。Formal100の二段ピンは変更なし。
- 最終15ファイル197件PASS。元失敗31件をJUnitに31/31結合。
  その後、独立検収が指摘した整数1/Trueの型すり抜けを閉じ、影響3ファイル20件PASS。
  両件数は重複するので加算しない。原票はDの`pytest_repair_2026-09-20_v1/`。
- 169件の同居失敗はc案。22ファイルを別プロセスとし、残りと合算して判定する。
  既存の単独PASSを維持し、無変更再走はしていない。全pytest/G3全体の新規合格は宣言しない。
- 5動画全長のepisodesを763,481区間集計。4動画の実NPZ食い違いは5,119 / 1,634,324基準セル。
  重力適用単独1,838、浮き除去単独890、合計2,728 / 5,119。複数書き手は時点別責任を断定しない。
- c80既知8セルは同runの実NPZへ8/8対応し、縦連続欠落形。重力単独5件、混在3件。
  既知2フレームを切り出して実画面確認済み。候補ガードは他4動画2,285セルへ作用しうるため未実装。
- 行1〜4の3,659件のうち、4動画3,151件は記録時ラベルSTABLE（連鎖履歴あり264・なし2,887）。
  video38全長508件はnpz_detail不在で時点別結合未測定。先頭500秒の74件を代用しない。
  states/writersだけでは全消しテロップや実画面の連鎖状態を確定できない。0件と報告しない。
- 採点器の自己確認は47判定・既知3見積もりを全再現。関連コードと必要な未追跡依存資産だけを
  コミット対象へ選別。元の大量未追跡物と無関係な設定・計画差分はそのまま保持する。
- CPU親子の実起動・保存・exit0・終了を確認。認識/GPU再走・学習・本番採用は行っていない。
- Fable最終は差分妥当の「条件付き了承」。原票提示方法による実読制約が残り、独立合格とはしない。
  親は原JUnitと31/31対応を確認済み。無変更レビューを反復せず、制約付きで修復版を保存する。
