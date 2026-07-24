# Cycle 検証で確定したプロジェクトルール

> 2026-05-15 cycle_5〜cycle_11 (5 動画 multicycle 検証) で得られた発見をルール化したもの。
> CLAUDE.md からリンクされる。 認識精度改善 cycle を回す前に必ず読む。

## 1. パラメータ・指標の確定値

### 1.-1 cycle 再走時は `--cnn-model models/cnn_phase_i_hsv_seed.pt` を **必ず明示**せよ (2026-05-17)
- 落とし穴: `scripts/multi_video_cycle.py --cnn-model` の default は `cnn_phase_b_large_v3.pt`、
  `RecognitionPipeline.DEFAULT_CNN_MODEL_PATH` の default は `cnn_phase_b_large_v2.pt`。
  いずれも cycle 14 以前の 71v 路線で、 puyo 認識が cycle_14 (HSV-seed) より劣る。
- cycle_14-19 は `cnn_phase_i_hsv_seed.pt` を **明示指定** して走行していた。
  launch script に `--cnn-model` を書かないと別 model が選ばれ、 結果が悪化に見えて
  「変更が悪い」 と誤判定される (実際は model 差)。
- 2026-05-17 cycle_20/cycle_21 の初回走行で実際にこの罠にハマり、 patch 3rd AND 軸が
  「悪化」 と誤判定された。 後で hsv_seed で再走させて正しい比較が必要。
- **以後の launch script template**:
  ```
  PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
      --cycle XX --parallel 3 \
      --cnn-model models/cnn_phase_i_hsv_seed.pt \
      --cnn-override-prob 0.70 \
      --hsv-state data/per_video_hsv_ranges/_merged_default.json
  ```
- TODO: `multi_video_cycle.py` の default を `cnn_phase_i_hsv_seed.pt` に変更する
  (cycle_14 以降の運用に合わせる)。 ただし破壊的変更なので慎重に。

### 1.0 `visualize_recognition.py --hsv-state` の default = `_merged_default.json` (2026-05-16)
- `scripts/visualize_recognition.py:186-194`
- 引数なしで起動すると `data/per_video_hsv_ranges/_merged_default.json` を pre-inject。
- cycle_12 で効果検証 (mismatch -4.3%、 constraint +1.8%、 未知動画 frame 0 認識成立)。
- ファイル不在は silent skip (stderr に WARN のみ、 crash しない)。
- 明示無効化は存在しない path を渡す (`--hsv-state /nonexistent`)。

### 1.1 `DEFAULT_CNN_OVERRIDE_PROB = 0.70` 維持必須 (optimum 確定)
- `src/hybrid_classifier.py:36`
- **0.55 (弱める方向) は禁止**: cycle_9 で 0.70→0.55 にすると 5 動画合計
  `constraint_replaced` が 487→689 (+41%) で悪化。 特に v89m3 (low quality h264) で
  189→374 (約 2 倍) と劇的悪化。
- **0.80 (強める方向) も採用しない**: cycle_11 で 0.70→0.80 にすると mismatch 合計
  不変 (93=93)、 constraint 487→505 (+3.7%) で微悪化。 動画品質依存度
  (v89m3/v97 比) も改善せず。
- **→ 0.70 で optimum 確定**。 強める / 弱める どちらもネット改善なし、 この軸は
  打ち止め。 汎用化改善は別軸 (OnlineHsv 等) で。
- **CNN を弱めると constraint 補正経由で後段が爆発する** という trade-off は
  全動画で確認済 = 確定知見。

### 1.2 「Innovation D (HSV-fill)」は実質 no-op
- cycle_8 で導入された `validate_next_history` 内の HSV 距離最小色 fill。
- cycle_5 vs cycle_8 で全 5 動画 metrics 完全一致 (mismatch / constraint /
  stable_pct すべて 0 差分) → このコードパスは発火していない or 影響を与えていない。
- **理由**: `frame_bgr` / `region` が呼び出し側で渡されていない経路で fallback (UNKNOWN)
  していた。 この機能を活かしたいなら呼び出し側を直す必要があるが、
  そもそも UNKNOWN→fill が認識精度に効くかは未検証。

### 1.3 cycle_5 = cycle_10 完全一致 (= baseline 確定)
- `cnn_override_prob 0.70` 単独で cycle_5 baseline を完全再現できる。
- 他の cycle_8/9 期の変更 (HSV-fill 等) は結果に影響しない。
- **これは「認識精度問題の調整軸は単一」ではなく「現状コード上で結果に効く軸は cnn_override_prob のみ」を意味する** (他の改善は別軸で導入する必要がある)。

## 2. cycle 運用ルール

### 2.1 backup は必ず `hybrid_classifier.py` を含める
- cycle_5/8 の backup に `hybrid_classifier.py` が含まれていなかったため、
  cycle_9 で何を変更したか追跡できず半日溶けた。
- 「結果に効くファイル」 (= hybrid_classifier.py, recognition_pipeline.py,
  placement_inferrer.py, board_state_machine.py, state_detectors.py,
  background_fingerprint.py, image_reader.py) を **全部** backup する。
- backup 取得は `scripts/_backup_cycle.sh` に集約する想定 (未実装、 必要なら作る)。

### 2.2 1 cycle = 1 軸変更 (= 効果切り分け)
- 既存 `CYCLE_PLAN.md` に明記、 維持。
- 複数軸を同時に動かすと、 効果がどの軸に帰属するか判別不能。

### 2.3 backup の CHANGES.txt は信用するが鵜呑みにしない
- 「cycle_9 = cnn_override_prob 0.70→0.55」のような記述は **コードの diff で必ず裏付ける**。
- 過去 backup が不完全な場合、 mtime + 手元 src diff で当時の状態を再構築する。

### 2.4 cycle 比較対象は cycle_5 (baseline) を必ず含める
- cycle_5 = unseen 動画 (v97) で mismatch 27→8 (-70%) を達成した「大ヒット」点。
- 新規 cycle は cycle_5 比で評価する。 cycle_9 のような単発悪化を見逃さないため。

## 3. 並列実行・GPU 運用ルール

### 3.1 同時走行は最大 3 process (GPU 8GB 制約)
- `multi_video_cycle.py --parallel 3` が安全上限。
- 6 process 同時走行で frame 処理速度が 1.6 秒/frame (通常 0.18 秒/frame) と
  約 10 倍遅くなった = メモリ断片化。
- 一度遅くなると **cycle 全部 kill + 再起動が最速**。 そのまま待っても回復しない。

### 3.2 cycle 連鎖は chain script を使わず明示順次
- `_launch_cycle_X_then_Y.sh` のような自動連鎖は 2026-05-15 に誤判定で破綻した。
  - 原因: `pgrep` の pattern が甘く、 cycle_10 走行中なのに「procs gone」と判定し
    cycle_11 を早期起動 → 同時走行で速度激減。
- 推奨フロー: **cycle_X 完走を mp4 5 本 + viz process 全消滅の AND 条件で
  Monitor → 明示的に cycle_Y 起動**。

### 3.3 GPU clean state を保つ
- WSL2 + GPU は context 断片化に弱い。 cycle 1 サイクルごとに全 process を確実に
  完了させてから次へ。
- スリープ復帰時に GPU process が hang する確率が体感 30-50% → スリープ前は
  完走または kill で clean state に。

### 3.4 viz 起動は shell script 経由 + `setsid -f` で detach
- inline で `setsid -f bash -c '...'` を `wsl -d Ubuntu -- bash -c "..."` から
  叩くと、 引用符ネストで起動失敗する事例あり (2026-05-15)。
- 対策: `scripts/_launch_cycle_N.sh` 形式の shell script に書いて
  `wsl -d Ubuntu -- bash -c "setsid -f /path/to/_launch_cycle_N.sh < /dev/null > /dev/null 2>&1"`。

## 4. 集計・判定ルール

### 4.1 評価指標 (`scripts/cycle_metrics.py`)
- `mismatch_count`: `[constraint-mismatch]` ログ行数 = CNN 過剰検出
- `constraint_replaced_count`: `[constraint] ` 行数 = 物理推論補正回数
- `online_hsv_inject_count`: `online_hsv injected` 行数
- `p1_stable_pct` / `p2_stable_pct`: progress sampling 内 stable 比率

### 4.2 採否判定基準
- 全動画で `mismatch_count` 同等 (±2) かつ `constraint_replaced_count` ±10% 以内 → **neutral** (採否は別軸の判断)
- 1 動画でも `constraint_replaced` +20% 以上 → **悪化**、 採用しない
- 全動画で `mismatch_count` -20% 以上または `constraint_replaced` -20% 以上 → **採用検討**
- 低画質 (v89m3) は他動画より影響が大きく出やすい → **判定の重み増し**

### 4.2-bis viz 目視レビューは必須 (2026-05-18 確定)
- **mismatch / replace は puyo→empty 誤認に対して fail-silent** な指標。
  - mismatch は puyo 過剰検出回数 → puyo→empty 誤認だと逆に減る
  - replace は物理補正回数 → puyo cell が empty になれば発火しない
  - → 数値だけで「改善」 と判定すると **empty bias で puyo を消す model** が昇格してしまう
- 2026-05-18 cycle_24 (CReST 9 動画拡大) で実際にこの罠にハマった:
  - 数値: mismatch -45%、 replace -67% で「史上最高 baseline 更新」 と誤判定
  - viz 目視: 全動画で背景誤認 + puyo→empty 誤認が広範発生 (ユーザー報告)
  - 結論: cycle_23/24 完全不採用、 **cycle_19 (`cnn_phase_i_hsv_seed.pt`) が真 baseline**
- **採否判定の最低条件**:
  1. 5 動画 viz 全本を目視で「背景誤認 / puyo→empty / 置いた直後誤認 / 連鎖中誤認」 4 軸チェック
  2. 数値指標は trend 補助、 単独昇格判断には使わない
  3. 数値「劇的改善」 が出たら逆に懐疑的 (= empty bias で fail-silent 化していないか確認)
  4. memory `feedback_viz_eval_required.md` 参照

### 4.2-ter I1 メトリクスも採否ゲートに含める (2026-05-26 確定)

**mismatch / replace と同様に fail-silent** な認識崩壊を自動検出する 3 メトリクスを
`scripts/measure_stable_cell_acc.py` に実装 (I1 メトリクス)。
cycle を採用する前に以下の全てが NG でないことを確認する。

| メトリクス | 閾値 | 発火条件 | 検出対象 |
|---|---|---|---|
| `per_col_unknown_rate` | WARNING 15% / CRITICAL 30% | STABLE confirmed_board で col 別 COLOR_UNKNOWN 比率 | v89_match01 27-30s col=0,1 認識不能 |
| `non_stable_consecutive_frames` | CRITICAL 180 frames | warmup (15s) 後の最長連続 non-STABLE フレーム数 | state machine 初期化失敗 / 長時間 CHAIN 状態 |
| `per_col_midgame_empty_rate` | CRITICAL 99% | 中盤 (30s 以降) で col 単位の EMPTY 率 (最低 30 STABLE frame 必要) | v40_match01 1P col=1 全 EMPTY 誤判定 |

- `_judge_pass_fail(stats_list=stats_list)` を呼ぶと 3 メトリクスを含む PASS/FAIL 判定が出る
- **per_col_unknown_rate は mismatch と同様に採用ゲートに含める** (cycle 23/24 の反省)
  - mismatch/replace が改善していても per_col_unknown_rate が CRITICAL なら FAIL
- テスト: `tests/test_i1_metrics.py` (17 件、 必須 v89/v40 シナリオ含む)

### 4.2-quater Phase 1 評価基盤強化メトリクス (2026-05-28 確定)

**C1: `avg_puyo_count_per_stable_frame`** — STABLE 確定盤面の 1P+2P 合算平均ぷよ数。
- 実装: `src/recognition_evaluator.py` の `compute_avg_puyo_count()` (module-level)
- `generate_report(baseline_avg_puyo_count=X)` で baseline 比 check 有効化
- **baseline 比 < 0.85 → AUTO_REJECT** (= puyo→empty fail-silent の構造的 catch)
- 定数: `AVG_PUYO_COUNT_CRITICAL_RATIO = 0.85`

**baseline 値 (= patch_fp 採用、 main HEAD `ea505f2` / commit `b42a0c9`):**

| 動画 | avg_puyo_count | n_stable_frames |
|---|---|---|
| v89m7 | 40.32 | 2958 |
| v30_match11 | 33.12 | 2575 |
| v30_5min | 35.07 | 2120 |
| v97_match11 | 27.80 | 2755 |
| v29m2 | 54.08 | 3536 |
| v40m7 | 48.62 | 3078 |
| v51m2 | 41.49 | 2713 |
| v57m2 | 22.17 | 2493 |
| v70m2 | 34.76 | 2824 |
| v89m3 | 37.44 | 2059 |
| v95m15 | 15.56 | 2519 |
| v97m11 | 27.80 | 2755 |
| **12 動画加重平均** | **35.67** | **32385** |

- 今後の cycle で avg_puyo_count が **35.67 × 0.85 = 30.32 未満** なら REJECT
- compute コマンド: `compute_avg_puyo_count(entries)` に board_log JSONL エントリを渡す

**C2: `StableTransitionMonitor`** — STABLE→STABLE 間の物理事由なきぷよ大幅減少検知。
- 実装: `src/stable_transition_monitor.py` (新規)
- `RecognitionPipeline` に統合済: `_transition_monitor_1p/2p`
- `SideResult.transition_drop_alerts` に alert tuple リストを格納 (backwards compat: default None)
- 定数: `STABLE_TRANSITION_DROP_THRESHOLD = 2` (= 1 ツモ = 2 cell 以内は正常)

**C3: `judge_cycle()`** — 複合 verdict ロジック。
- 実装: `src/recognition_evaluator.py` の module-level 関数 `judge_cycle(baseline_stats, candidate_stats)`
- 戻り値: `"AUTO_ACCEPT_PROVISIONAL"` / `"AUTO_REJECT"` / `"NEEDS_REVIEW"`
- AUTO_REJECT 条件: avg_puyo_ratio < 0.85 OR p_to_e +20% 超 OR critical +10% 超
- テスト: `tests/test_recognition_evaluator.py` に新規 TestJudgeCycle / TestComputeAvgPuyoCount 等追加

### 4.2-quinquies postprocess_corruption ゲート (2026-05-31 確定 = fail-silent 再発防止)

**背景**: constraint_fill OFF を採用しかけたが、3者合意 cell-acc (99.86%) が後処理(constraint_fill)による正解破壊を埋もれさせ、main Claude が数値を信じて PR 化に向かった。user viz 目視 (1P→緑/2P→赤 バイアス) のみが発見。再発防止メモリ `feedback_consensus_eval_fail_silent.md`。

**D1: `postprocess_corruption`** — 後処理が CNN/HSV 一致の正解を破壊した検知。
- 実装: `scripts/measure_stable_cell_acc.py` の `_check_postprocess_corruption()`
- 条件: STABLE で `raw_cnn_val == raw_hsv_val and confirmed_val != raw_cnn_val` (UNKNOWN除く)
- 出力: count / rate / by_side / color_pairs / side_bias / corruption_ratio / log (結果 JSON `postprocess_corruption` セクション)
- **rate >= 0.001 (0.1%) → REJECT**、**片側 side_bias (書換先1色 >=50% & >=3セル) → REJECT**
- 定数: `POSTPROCESS_CORRUPTION_REJECT_RATE=0.001`, `POSTPROCESS_SIDE_BIAS_THRESHOLD=0.50`
- 破壊率正規化: `corruption_ratio = corruption/(corruption+physics_fix)` (>0.5 でネット負)
- テスト: `tests/test_postprocess_corruption.py` (合成ボード、false green 防止)

**重要 blind spot (D1 で検知できない)**:
- `raw_cnn==raw_hsv==誤り` の全列崩壊型 (未知動画 HSV ズレ等) は D1 で検知不可 → `per_col_midgame_empty_rate`(I1) + `avg_puyo_count`(C1) + **viz 目視**で補完必須。
- `enable_constraint_fill=False` 時は D1 が常時 0 (false green) → 結果 JSON に WARNING note 出力。constraint ON 条件で検証すること。

**採否チェックリスト (cycle 採用前に全確認)**:
1. measure_stable_cell_acc を実行 (constraint 評価時は ON 条件で corruption を見る)
2. `postprocess_corruption.rate >= 0.1%` or `side_bias.detected` → 数値が良くても REJECT
3. `corruption.log` のセル座標を viz で目視
4. **corruption=0 でも viz 目視は必須** (全列崩壊型は D1 で検知不可)
5. `avg_puyo_count >= baseline×0.85` (C1) 確認
6. 全 OK → user に viz 提示 → **user 承認後に PR 化**
7. 数値改善のみで PR 化は禁止 (`feedback_viz_eval_required.md` / `feedback_consensus_eval_fail_silent.md`)

### 4.3 動画品質依存度を見る
- 高画質 (v97) と低画質 (v89m3) の `constraint_replaced` 比 = 動画品質依存度
- cycle_5: 189 / 19 = 10 倍格差
- cycle_9: 374 / 32 = 11.7 倍格差 (悪化)
- **動画品質依存度 = 汎用化の指標**。 これを下げる施策が真の改善。

## 5. 認識精度改善の次の方向性 (2026-05-15 時点)

`cnn_override_prob` 軸 (cycle_9/11) と OnlineHsv DB pre-inject (cycle_12) は
いずれも既存動画の動画品質依存度 (v89m3/v97 constraint 比 10 倍) を改善しなかった。
→ **残る打ち手は CNN model の品質向上のみ**。 OnlineHsv 関連は別ファイル等で完了済。

1. **CNN mode collapse 突破** (本命) — memory `project_phase_i_finetune_findings.md`、
   pseudo fine-tune が ~50% 壁を超えない問題。 手動 seed dataset で fine-tune の anchor
   作成 or HSV-based seed で fresh CNN 再学習。 工数 ~1-2 日 + GPU 学習時間。
2. **CReST oversampling + logit adjustment** — memory
   `project_recognition_improvement_candidates.md`、 工数 ~1-2 日、 CNN 再学習必要。
3. **背景 FP の HybridClassifier 伝達** — memory `project_bg_fingerprint_status.md`、
   現状 image_reader 1st pass のみで青背景バイアス残存。

順序は (1) → (2) → (3) を推奨。 (1) が本質的、 (2)(3) は補助。
DB pre-inject (cycle_12 で検証済) は本番 default に組み込む価値あるが、
動画品質依存問題は解決しない。

## 6. cycle 履歴一覧 (~2026-05-15 時点)

| cycle | 変更 | 効果 | 採否 |
|---|---|---|---|
| 0 | LANDING_GRACE 36→12 | baseline | 起点 |
| 1 | F1 (UNKNOWN→CNN fill) + F1-B (ever_seen) | neutral | keep |
| 2 | F4 (STABLE_OVERRIDE_MIN_RATIO 0.9→0.75) | neutral | keep |
| 3 | F3 (bg threshold 28→35) | neutral | (cycle_6 で revert) |
| 4 | F7 (LANDING_VOTE_MIN_RATIO 0.4→0.3) | neutral | keep |
| **5** | **F8 (LANDING_VOTE_FRAMES 36→24)** | **v97 mismatch 27→8 (-70%)** | **★ baseline 確定** |
| 6 | F3 revert | cycle_5 と一致 | F3 不要確定 |
| 7 | F11 + F8' (vote 24→18) | 2P stable -3.4pt | revert (cycle_5 採用) |
| 8 | Innovation D (HSV-fill) | cycle_5 と完全一致 (no-op) | 効果なし |
| 9 | Innovation I (cnn_override_prob 0.70→0.55) | constraint +41% | **revert (cycle_10 で実施)** |
| 10 | cycle_9 revert (0.55→0.70) | cycle_5 と完全一致 | **★ baseline 復元** |
| 11 | cnn_override_prob 0.70→0.80 | mismatch 不変 (93=93)、 constraint +3.7%、 微悪化 | revert (0.70 で打ち止め) |
| 12 | DB pre-inject (`_merged_default.json`) | mismatch -4.3% (93→89)、 constraint +1.8% (487→496)、 動画品質依存度は不変 | **本番 default 採用推奨** (未知動画 frame 0 認識のため。 既知動画指標は neutral) |
| 13 | `visualize_recognition --hsv-state` default を `_merged_default.json` に変更 (本番 default 化) | sanity check OK (引数なしで pre-inject ログ確認) | **★ 採用完了** (2026-05-16) |
| 14 | HSV-seed fine-tuned CNN (`cnn_phase_i_hsv_seed.pt`)、 5 色 11,500 seed × epochs=5 + class_balance + augment | mismatch -91% (93→8)、 constraint -92% (487→39)、 動画品質依存度 完全解消、 v89m3 stable +21pt | mode collapse 突破成功、 ただし **empty 学習なし → 背景まで puyo 認識** の副作用判明 (viz 目視で確認、 物理推論の検出網漏れだった) |
| 15 | cycle_14 seed + empty 2,500 件追加 (= 14,000 件)、 class_balance + augment | mismatch -94.6% (93→5)、 constraint -92% (487→39)、 fill 13 復活 | empty 認識復活、 ただし **puyo 認識力が cycle_14 から劣化** (= class_balance で empty dominant) |
| 16 | cycle_15 seed + class_balance OFF | キャンセル (= cycle_14 model + bg_fp 強化 = cycle_18 を先行) | — |
| 17 | cycle_14 model + bg_fp 閾値 35→50 | cycle_14 と完全同一数値 = bg_fp が capture されない | **失敗** (鶏卵問題: cycle_14 model が試合開始から全 puyo 出力 → puyo_count_total>0 で bg_fp 採取条件不成立) |
| 18 | cycle_17 + `BG_FP_FORCE_MAX_PUYO=5→144` で鶏卵問題突破 | mismatch=73, replace=600, **fill=203 (= 物理推論大量発火)** | **bg_fp 機能、 v89m3 良好** だが **v97 で puyo→empty 誤認** (= bg_fp aggressive すぎて puyo cell まで empty 化) |
| 19 | cycle_18 + AND 条件 (bg_fp 距離 < 50 **AND** HSV-単独でも puyo 色判定されない) で empty 確定 | mismatch 38 (cycle_18: 73, -48%)、 replace 323 (600, -46%)、 v97 完全解消 (mismatch 9→0, replace 102→0)、 ただし v91 replace 58→163 (+181%) で副作用 | **AND 条件は puyo 復活成功**、 ただし v91 で aggressive すぎる箇所が残る。 cycle_20 (bg_fp 構造改革: NextDetector トリガー + cell patch 化) へ進む |
| 20 | bg_fp 構造改革: (A) NextDetector トリガーで採取タイミングを CNN 非依存化、 (B) cell 画像 patch (32x32 BGR) を保存 → 3rd AND 軸 (patch L1 mean < 30 で empty 確定) | mismatch 89 (cycle_19: 38, **+134%**)、 replace 483 (323, **+50%**) ← **だが launch script が `cnn_phase_b_large_v3.pt` を渡しており、 cycle_19 (= hsv_seed model) と異なる model で走行していたため比較不能**。 再評価は別 cycle で hsv_seed model で再走必要。 | **暫定 revert** (cycle_19 baseline 維持)、 ただし「悪化」 判定は model 不一致由来の可能性大。 launch script に `--cnn-model models/cnn_phase_i_hsv_seed.pt` 必須を section 1.-1 として記録 (2026-05-17) |
| 21 | 物理推論強化 (a+b+d): (a) 連鎖完了で constraint_valid 再有効化、 (b) 連鎖消去分の tsumo_count 減算 (ChainSimulator 再実行)、 (d) chain_ev side 別 ban (OJAMA_FALL / LANDING_GRACE 中) | **hsv_seed model で再走 (= cycle_19 と同条件)**: v50/v70/v89m3/v91 完全同一 (= a+b+d 副作用ゼロ)、 **v97 のみ mismatch 0→16 (+16)** | **不採用 / revert** (2026-05-17)。 (a+b) は効果ゼロ (= 連鎖機会で constraint_valid 再有効化されても constraint_replaced 増えず)、 (d) が v97 で正常連鎖を誤 ban (= false negative)。 (c) お邪魔追跡は別 cycle 切り出し。 次はロードマップ通り **cycle_23 (CReST oversampling) 前倒し** へ (cycle_22 おじゃま強化は飛ばし) |
| 23 | **CReST oversampling (5 video)**: cycle_14 seed (11,500) + empty 2,500 = 14,000 件、 class_balance **OFF** + oversample-alpha 0.5 + focal-gamma 2.0 + logit-adjust-tau 1.0 + augment + epochs 10、 model `cnn_phase_b_crest_v1.pt` (val acc 98.79%) | mismatch 48 (cycle_19: 38, +26%)、 replace 426 (323, +32%)、 **v89m3 で mismatch 100% 消失** (7→0, replace 80→3, -96%)、 v50/v70 改善、 stable_pct 全動画大幅向上 (v50 p2 +26.1pt)、 **v91 で大悪化** (mismatch 9→26 +189%, replace 163→321 +97%)、 v97 軽悪化 (0→5) | **部分採用候補**: 4/5 で大幅改善 + stable_pct 全動画向上は史上最大、 ただし v91 悪化が CYCLE_FINDINGS 4.2 基準で「悪化」 該当。 原因は学習データ偏り (= 5 動画のうち v91 サンプル不足)。 → cycle_24 (= 9 video 拡大) で解決を試みる |
| 24 | **CReST oversampling (9 video 拡大)**: cycle_23 の 5 → 9 video 拡大 (v29m2, v40m7, v51m2, v57m2 追加、 raw video が手元にある全動画)、 extract_hsv_seed_dataset.py で各 ~11,000 件抽出 + empty 500/video + ojama 学習も追加 = total **104,020 件**、 cycle_23 と同じ args で fine-tune、 model `cnn_phase_b_crest_v2.pt` (val acc 96.05%) | 数値: mismatch 21 (cycle_19: 38, -45%)、 replace 108 (323, -67%)。 viz: 全動画背景誤認 + puyo→empty 誤認広範発生 (= ユーザー目視 2026-05-18) | **不採用** (2026-05-18 改訂)。 当初「★ 採用昇格」 判定したが viz レビューで puyo→empty 誤認判明 (= empty oversample-alpha 0.5 で minority 過剰採択、 empty bias) → mismatch/replace は **fail-silent** で見抜けず。 cycle_23 も同様に **不採用**。 **cycle_19 (= `cnn_phase_i_hsv_seed.pt`) が真 baseline** 確定。 採否ルールに viz 目視必須を追加 (section 4.2-bis) |
| **26** | **着地直後の誤認削減 (A1+A2+A4 統合)**: A1 = grace 中 confirmed_board 完全凍結 (constraint/vote/long-term override skip)、 A2 = LANDING_VOTE 初期 5 frame 除外 + 不一致時 ratio 0.5、 A4 = NEXT 色 prior 強化 (ratio>=0.7) + 早期確定 (len>=5, ratio>=0.8)。 model は cycle_19 と同条件 `cnn_phase_i_hsv_seed.pt`、 編集 `src/recognition_pipeline.py` のみ、 backup `data/cycles/cycle_26_backup/` | viz 走行中 (2026-05-18) | TBD (viz 目視評価必須。 着目: 「置いた直後の誤認」 が cycle_19 比で削減されたか) |

## 4-end. 今後の確定ロードマップ (2026-05-16 確定)

ユーザー指示で「有利不利判定に必要 + 認識精度に影響するもの」 を全部このタームで実装する方針:

| cycle | 内容 | 工数 | 結果 |
|---|---|---|---|
| 20 | ~~bg_fp 構造改革~~ | ~4h | **不採用 (model 不一致由来の悪化、 再評価候補)** |
| 21 | ~~物理推論強化 (a+b+d)~~ | ~半日 | **不採用 (a+b 効果ゼロ、 d v97 で誤 ban)** |
| 23 | CReST oversampling (5 video) | 学習 ~30分 | **部分採用候補** (v91 悪化、 cycle_24 で解決) |
| **24** | **CReST oversampling (9 video 拡大)** = cycle_23 + raw video が手元にある全動画 (v29m2/v40m7/v51m2/v57m2 追加)、 extract_hsv_seed_dataset で 104,020 件 dataset、 ojama 学習も追加 | 学習 ~1h | **★ 採用 (mismatch -45%, replace -67%, 史上最高)** |
| 次候補 25 | Phase L 本番化前哨: 全 66 video raw を再 DL → seed 拡大 9→66 動画 | 1-2 日 (DL + 学習) | (未着手) |
| 22 候補 (保留) | おじゃま puyo cell 認識強化 (score OCR 主軸で副次的) | ~3h | (保留、 cycle_24 で ojama 学習組込済) |

## 5. 認識精度改善の方向性 (2026-05-18 viz レビュー結果で改訂)

**cycle_24 数値改善は見せかけ**だった (= viz 品質劣化、 empty bias で fail-silent)。 真 baseline は cycle_19 (`cnn_phase_i_hsv_seed.pt`)。

- cycle_14 で「HSV-seed で puyo 認識 98.87%」 達成、 ただし 5 動画で empty 学習なし。
- cycle_15 で empty 追加 + class_balance ON → empty dominant で puyo 劣化、 失敗。
- cycle_19 で bg_fp AND 条件 → mismatch 38 まで削減、 **真 baseline 確定**。
- cycle_23 (5 video CReST)、 cycle_24 (9 video CReST) → 数値改善だが viz puyo→empty 誤認悪化、 **両方不採用**。

→ **着手中: cycle_26** (着地直後の誤認削減、 A1+A2+A4)。
  - cycle_19 viz でユーザー指摘の 3 課題のうち最頻発のもの (= 「置いた直後の誤認」) が対象
  - 物理推論 final_board を grace 中 100% hold + LANDING_VOTE 強化
  - 編集 `src/recognition_pipeline.py` のみ、 model 変更なし

→ **次以降の打ち手**:
  - **cycle_27**: A3 = placement_inferrer + rotation_tracker 連携 (= 4 回転判別精度向上)
  - **cycle_28**: Phase L 前哨 = 全 66 動画 raw 再 DL + seed 拡大 + CNN 再学習 + viz 目視併用評価。 背景誤認 (v89m3 1P 等) の根本対策
  - cycle_25 (連鎖中 step-by-step 物理推論) は **優先度低** で保留 (= 連鎖時間/火力/連鎖後形は ChainSimulator で既に取れているため指標的に問題なし)

## 7. Phase L 本番化最優先 attack 対象 (= 2026-05-25 追記)

### 7.1 v30_5min 55 秒問題

- **症状**: v30_5min_90s.mp4 の frame 3262-3302 (= 55 秒地点) で 1P 側 confirmed_board が 0.7 秒間 null (= 43 frame 連続)、 直後 frame 3304 で 8 cells が auto_correction critical 発火
- **真因 (= 4 エージェント分析確定)**: CNN が 1P 盤面 row 5-11, col 0-1 エリアに緑 (= color 3) / 紫 (= color 5) を確信誤認、 30 STABLE frame に渡って同色出力を継続
- **背景**: 1P 側 light キャラの暗赤背景が盤面内に広く露出 (= 試合 2 開始 12 秒後で盤面ほぼ空)、 CNN が「赤背景 ≒ 赤ぷよ」 と確信誤認 (ただし実際は緑/紫を出力 = 学習 seed と背景 HSV の overlap が複雑)

### 7.2 推論軸での試行履歴 (= phase-l-bg-mask branch、 全て主目標未達)

| 軸 | 内容 | 結果 |
|---|---|---|
| 軸 1 (= cnn_board EMPTY gate) | 試合開始後 15 秒間 S<80 V<150 cell を EMPTY 強制 | regression +2、 revert |
| 軸 1' 案 b (= score_zero_both tsumo clear) | 試合切り替わり検知時に tsumo_count 強制 reset | regression 0、 55 秒改善ゼロ、 revert |
| 軸 3-b (= tier 1 threshold エリア別) | row 5-11, col 0-1 で threshold +15.0 緩和 | 副次 -9 critical (= 0-20 秒の 2P 認識ノイズ削減)、 55 秒主目標未達 |

**軸 3-b の扱い**: branch 上に残置 (= main には未反映)。 副次効果数値確認済のため保留知見として記録。

### 7.3 Phase L 本番化での attack 方向 (= 次回最優先)

- **動画追加 DL** で v30 系の試合 2-3 等 (= 異なるキャラ・盤面状態) を seed 拡充
- **キャラ別 seed 採取** で light キャラの暗赤背景を明示的に学習データに含める
- **新 CNN scratch 学習** で row 5-11, col 0-1 の特定エリアへの確信誤認を解消
- **per-match HSV** (= 案 4) は単発実装で先行検証可能、 ただし工数 1-2 日

### 7.4 推論軸の限界 (= cycle 32 系 7 連敗 + 軸 1/1'/3-b 全敗で確定)

- bg_fingerprint tier 1/2 の **grey zone (= distance 25-100)** は推論層で止められない
- CNN 確信誤認 + 30 STABLE frame 持続は推論層で構造的に解決不能
- 解決には **新規 seed (= キャラ別暗赤背景含む) + scratch 学習** が必要 = Phase L 本番化フェーズ

### 7.5 全 12 動画 eval 結果 (= 軸 3-b 適用時 per_video_inject、 2026-05-24 完走)

- baseline_existing_critical: 1512
- per_video_inject_critical: 1583 (+71、 +4.7%)
- v30_5min: critical 107 (= 主目標の 55 秒問題改善ゼロを数値確認)
- 参考: merged38 比では -4 (= -0.3%) で実質同等

## 8. 連鎖後残像バグ (2026-07-23 調査)

### 8.1 P3 (`is_match_active` 誤 False) 仮説は iter4 診断で反証済み

- 反復4 診断 (`board_none_reason` 内訳集計、`scripts/recognition_physics_review.py`)
  で c62 game9 の CHAIN 中 `confirmed_board=None` の理由内訳を計測した結果、
  1P/2P とも `chain_hold_none` (= CHAIN/GRAVITY_SETTLE 中は STABLE 以外という
  仕様通りの凍結) がほぼ 100%、`menu_reset` (= P3 が狙う
  `is_match_active` 誤 False → MENU 強制経路) はほぼ 0% だった
  (実測: 1P `board_none_reason_chain` = `{'chain_hold_none': 1320}` /
  1321 frame、`menu_reset` 0 件)。
- つまり CHAIN 中に `confirmed_board` が None になるのは「STABLE 以外は
  未確定」という設計通りの挙動であり、`is_match_active` の誤判定 (P3 が
  想定した経路) はこの区間で発生していない。
- **以後、この軸 (is_match_active 誤 False 対策) への投資はしない**。
  「連鎖後残像/estimated_board 未カバー」問題の真因は別軸 (起点盤面
  `before_board` の認識精度、または連続小連鎖時の trigger 検出タイミング)
  にある。次の調査は `scripts/_diag_estimate_collapse_c62_1p.py`
  (2026-07-23, c62 game9 1P estimated_board coverage 9.8% の真因診断) を参照。
- **解釈**: per_video_inject 自体の評価であり、 軸 3-b 単体の -9 副次効果は別コンテキスト

### 8.2 機能D (`enable_chain_formula_simulate_verify`) を default ON に採用 (2026-07-24)

- 機能D (連鎖開始 掛け算式検知, `enable_chain_formula_detection`, default ON
  済) の早期発火 77 件を `_diag_false_event_source_2026-07-24.py` で真因
  診断した結果、35 件 (45.5%) が「連鎖ゼロの起点盤面」からの疑似発火
  (偽イベント) と確定した。
- 修正D として、早期発火の起点盤面 (`before_board`) を `ChainSimulator`
  で事前検証し、`chain_count==0` (連鎖が実在しない) なら疑似発火を抑制、
  `chain_count>0` なら固定 `chain_count=1` でなく実測値を使う対策を実装
  (`src/recognition_pipeline.py` の `_resolve_formula_chain_count` /
  `_apply_chain_formula_early_fire`)。
- 物理採点 + 独立診断 (before/after 比較) で偽イベント率 27.5% → 0% と
  確認され、user viz 承認により `enable_chain_formula_simulate_verify` の
  既定値を `False` → `True` に変更 (2026-07-24、`__init__` /
  `load_default` 両方)。誤抑制 (連鎖が実在する起点盤面を誤って握り潰す)
  は最小構成テストで検証済みでゼロ。
- 旧挙動 (検証なし・bit-identical) は
  `enable_chain_formula_simulate_verify=False` を明示指定すれば維持できる
  (backwards compat のため退避経路として残置)。
  savepoint タグ: `savepoint/chain-formula-verify-default-on`。
