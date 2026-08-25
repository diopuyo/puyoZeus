# Codex → Claude 命令箱

状態: `READY_FOR_MANUAL_SUBMISSION`

実行開始: ユーザーがClaude Codeの制限解除後に手動で指示する。

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
