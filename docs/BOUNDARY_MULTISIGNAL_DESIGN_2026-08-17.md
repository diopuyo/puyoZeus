# 試合境界マルチシグナル + STABLE持続確認 統合設計 v2 (2026-08-17、アーキ、棚卸し反映版)

工程上の位置づけ (user承認 2026-08-17): W25クローズ → ②判定報告 → **本設計の実装** → 148再収集再開 → Phase J。
動機: STABLEスナップショット汚染40% (下限)、綺麗な静止盤面だけなら99.93%
(`data/verify/board_labels_general_2026-08-17/RESULTS.md`)。

## 0. 前提 (棚卸し結果 2026-08-17、両論併記は不要)
- `src/match_end_detector.py` (`MatchEndDetector`): やった!/ばたんきゅー検出**実装済み・本番配線済み**。
  テンプレ `models/ui_templates/match_end_{yatta,batan}.png`、NCC健全 (0.96-0.98)、
  閾値 `DEFAULT_NCC_THRESHOLD=0.55`、`DEFAULT_LOCKDOWN_SEC=5.0`。SEARCH_P1/P2 の左右ROIで敗者側特定可
- `scripts/collect_boards_lean.py:602-828` `_SharedGameCounter` (+`_reconcile_boundary_anomalies`):
  game_idx進行の境界マルチシグナル本体、実装済み・既定OFF (`--enable-boundary-multisignal`)。
  W22救済込みで c109/c13/c96 実測 7/8=87.5%
- `src/win_panel.py`+`src/match_winner.py`: WIN★パネル数値差分の勝者特定、完成済み・オフライン専用
  (`--enable-winner-panel-crosscheck` 既定OFF)
- `src/board_quality.py::phantom_board_mask`: 非試合幻盤面の事後npzフィルタ (多層防御として維持、競合しない)
- レディーゴー専用検出: 実装ゼロ。現行代理指標 (visual-rise-persist) で87.5%
- **新規盲点 (本設計の主要動機)**: ばたんきゅー結果パネル表示中も `is_match_active` がTrueのまま
  (`030_c21_2P_f57548` 実写確認)。
- **【Step 0診断 完了 2026-08-18】結論 = 診断B (分岐の打ち消し)。ロックダウン切れ仮説でもテンプレ欠損でも
  チャンク人工物でもない** (`data/verify/diag_match_end_miss_2026-08-17/`):
  - `MatchEndDetector` は t=958.667 (パネル出現前0.47秒) から3.15秒連続検出 (NCC最大0.9773)、
    `match_end_locked` も即座にTrue
  - しかし**勝者側の13連鎖キル連鎖のアニメが CHAIN/GRAVITY_SETTLE を報告し続け、
    `chain_in_progress` ガード (recognition_pipeline.py:3720-3735、反復3 2026-07-23) が
    正規の試合終了検出を2.55秒間打ち消した**。030アンカーはこの窓内 (+0.47秒)
  - 設計前提の転覆: 「連鎖中の match_end 検出=瞬間誤爆」という前提でガードが書かれたが、
    実ゲームでは**本物のパネルは勝者の連鎖アニメ中に表示され始める** (持続3秒超 vs 誤爆は単発)
  - force_in_match True/False で結果同一 (収集方式の影響なし、診断C除外)
  - 副次観測: t=962.08 に score OCR 0/0誤読による0.03秒のTrue復帰フリッカ (主因ではない、記録のみ)

## 1. 実装順序・依存関係
1. **W25コミット後に着手** (recognition_pipeline.py の行域は離れているが、W25コミットのdiffを確認してから)
2. (b) ばたんきゅーラッチ → (a) 既存境界マルチシグナルのA/B→本採用判断。並行して (d) (依存なし)
3. (c) レディーゴーは本サイクル見送り

## 2. (a) 既存境界マルチシグナルの本採用判断 (新規実装なし、A/B測定→採用登録)
- 対象: `--enable-boundary-multisignal` / `--enable-winner-panel-crosscheck` (両方既定OFFのまま測定)
- 手順: 実写確認済み動画 (c109/c13/c96/c10/c11/c17/c21/c22/c23) でON/OFF。指標 =
  (i) game_idx境界の残anomalies数 (ii) 勝者判定一致率 (score単調増加+実写照合)
  (iii) 物差しv2 55盤面bit-identical (ピクセル分類に触れないので崩れたら実装ミス)
- 採用基準: (i)(ii)が現行(score-reset単独)を上回り(iii)無退行 → production_config へ採用日+根拠つき登録
- 順序: (b)実装後にまとめて測定 (③が塞がってからの方が測定が汚れない)

## 3. (b) 試合外画面オンライン検出 (新設・本丸、**Step 0診断により2段構成に確定**)

### (b-1) match_end持続時間ゲート (診断Bへの対処、Step 0推奨案2)
`match_end_locked` が `MATCH_END_PERSIST_OVERRIDE_SEC` (≈1.0〜1.5秒、実測校正) 以上連続Trueなら、
`chain_in_progress` による抑制を上書きして `effective_hard_off` を有効化する持続タイマーを新設。
- 根拠: 本物のパネルは3秒超持続検出される / 元ガードが守る「瞬間誤爆」は単発 — 持続時間で弁別可能
- 既存回帰テスト (`test_gravity_settle_in_progress_suppresses_match_end_locked_false_positive`) は
  維持したまま新規テスト追加 (瞬間誤爆=抑制継続、持続検出=タイムアウト後に反映)
- 「match_end_lockedを連鎖ガードから完全除外する1行修正」は**非推奨** (瞬間誤爆の再燃リスク)
- 校正材料: 他の長連鎖フィニッシュ実例 (5連鎖級・20連鎖級 各1件以上) の追加実測が望ましい

### (b-2) 次試合開始までのラッチ (対戦カード紹介・待機画面のカバー)
**対戦カード紹介の専用検出器は新設しない**。ばたんきゅー/やった!検出をトリガーに
「次の本物の試合開始が確認されるまで試合外とみなすラッチ」 —
結果パネル・対戦カード紹介・次ラウンド待機を一括カバー (ロックダウン5秒切れ後の再活性を防ぐ)。
- 作用点: `recognition_pipeline.py:3582-3596` (match_end_locked計算) 〜 `:3631` (hard_match_off)
- 状態: `self._post_match_lockdown_active: bool = False` (`__init__`/`reset()`)
- ラッチON: `match_end_locked` False→True の立ち上がり
- ラッチOFF: `raw_active` が `BOUNDARY_VISUAL_RISE_PERSIST_SEC` (既存定数0.5秒を再利用) 以上連続True
- 安全弁: `POST_MATCH_LOCKDOWN_MAX_SEC` 新設。**実測前に確定させない** — ばたんきゅー〜次ラウンド開始の
  実秒数を c21/c109/c13/c96 で計測してから決定 (仮の作業値60秒でスケルトン→実測値で差し替え)
- 合流: `hard_match_off = score_zero_both or match_end_locked or self._post_match_lockdown_active`。
  以降の `effective_hard_off`/`is_active` は無変更 (score_actively_moving/chain_in_progress の保護がそのまま効く)
- フラグ: `enable_post_match_lockdown_latch` (既定False)。`__init__`/`load_default`/collect CLI まで配線。
  OFF時bit-identical
- **RTスコープ**: RT本体に実装してよい。「盤面が無いと確定している区間の延長」なので指摘13リスクなし。
  Phase Jの勝者即時特定にも直接寄与
- 検証: ①単体テスト4パターン (ON/OFF/上限解除/bit-identical) ②c21該当窓で is_match_active=False 維持を計装確認
  ③一般分布35枚再収集で③試合外5件が混入しなくなること ④数本で総STABLE snapshot数の変化 (急減は誤爆疑い)

## 4. (c) レディーゴー専用検出 → 見送り (Phase Jバックログ)
現行代理指標87.5%は今回の目的を阻害していない。専用テンプレの価値は開始時刻の精度のみ = Phase Jの
レイテンシ要件の話。再検討トリガー: (a)のA/Bで「開始検知遅延」型の境界ズレが頻発した場合、
または Phase J で開始時刻±1秒以内などの明示要件が出た場合。

## 5. (d) STABLE確定の持続確認 (新設、収集限定)
**RT本体には入れない** (`_should_emit` のみ)。カテゴリ①② (連鎖アニメ中・送付フラッシュ重畳) 専用。
③試合外は静止画面なので差分ゼロ = 本機構では検出不能 — (b)の守備範囲 ((b)の代替にはならない)。
- 信号: `scripts/_diag_general_chain_contamination_2026-08-17.py:160-176`
  (`_board_roi_gray`/`_column_diffs`) を `src/board_motion.py` (新設、stateless純関数) に昇格
- 作用点: `collect_boards_lean.py` メインループ (`:1383-1392`) で毎フレームside別diffを計算し
  rolling window保持 → `_should_emit` (`:876`直後) に
  `if enable_stable_persistence_gate and not state.raw_pixel_stable: return False`
- 判定: `board_motion.is_raw_pixel_stable(recent_diffs, window_sec, diff_threshold)` —
  直近 `STABLE_PERSISTENCE_WINDOW_SEC` 秒のdiffが全て `STABLE_PERSISTENCE_DIFF_THRESHOLD` 未満
- **定数は実測から**: 汚染判定の実測diff (002連鎖中16セル/031フラッシュ/005送付フラッシュ vs 綺麗な21枚)
  を突き合わせて分離できる値を算出してから固定。「002だけ通る閾値調整」は過学習で禁止
- 自己修復: 既存dedup機構により、スキップ後に本当に静止した瞬間が来れば自動記録 (defer不要)
- フラグ: `enable_stable_persistence_gate` (既定False)。collect CLI のみ配線
  (RecognitionPipeline側に足さない = RTスコープ外の意図的な非対称配線)
- 検証: ①board_motion純関数の数値一致回帰テスト+OFF時bit-identical ②一般分布再収集で002/005の解消確認
  (003は解消候補だが保証しない、実測報告) ③物差しv2無退行 ④STABLE snapshot総数の変化率 (過度な減少=閾値過剰)

## 6. 148再収集の最小ブロックセット
- **必須**: (b) 実装+検証、および (a) の本採用判断 (既存フラグON固定の意思決定なしに148へ進まない)
- **推奨・非ブロッカー**: (d)。間に合えば148に反映、間に合わなければ後追い
- **見送り**: (c)

## 7. フラグ一覧サマリ
| フラグ | 既定 | 配線先 | スコープ |
|---|---|---|---|
| `enable_boundary_multisignal` | False (既存) | collect_boards_lean.py | 収集 (game_idx) |
| `enable_winner_panel_crosscheck` | False (既存) | collect_boards_lean.py | 収集 (勝者ラベル) |
| `enable_post_match_lockdown_latch` | False (新規) | recognition_pipeline.py + collect CLI | RT本体+収集 (試合外検知) |
| `enable_stable_persistence_gate` | False (新規) | collect_boards_lean.py のみ | 収集限定 (STABLE持続確認) |

## 8. 参照
`src/recognition_pipeline.py:3562-3739` (is_active合成) / `src/match_end_detector.py` /
`src/win_panel.py` / `src/match_winner.py` / `src/board_quality.py` /
`scripts/collect_boards_lean.py:602-888,1383-1642` /
`data/verify/diag_general_chain_contamination_2026-08-17/` (汚染判定の実測) /
`docs/KNOWN_WEAKNESSES.md` (W3/W20/W21/W22/W25)
