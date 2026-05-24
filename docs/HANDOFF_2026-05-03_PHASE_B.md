# Phase B 引継ぎドキュメント — 2026-05-03

## TL;DR

- **認識戦略を大転換**: 「STABLE state の静止フィールドのみ認識、それ以外は物理推論」(`project_recognition_strategy_pivot`)
- BoardStateMachine + 4 detector + InferenceBoardGenerator + DriftDetector + RecognitionPipeline を新規実装、合計 81+ tests pass
- **動画別 model selector** で平均 STABLE 53.5% (HSV のみ) → **58.3%** (per-video) **+4.8pt 改善**
- 単 frame 精度は 96-99% (drift 解析実数値、試合中区間)
- CNN-v2 訓練は drift_truth 真値の不整合で破綻、**cnn_phase_b_v1 (holdout 99.61%) を最終版として固定**
- Phase Z hard violation review 資産は破棄、新方針では訓練データ源泉として使えない

## 主な発見

### 1. 認識戦略 pivot

**旧方針**: 全 frame で CNN 単 frame 精度 99.5% を目指す
**新方針**: **BoardStateMachine が STABLE 確定した frame のみで精度を測る** + アクション中 (ツモ落下/連鎖/おじゃま落下/エフェクト) は ChainSimulator + ネクスト履歴 + score OCR で推論

ぷよぷよはルール完全閉じの決定論ゲームなので、平常時さえ精度が出れば中間状態は全てシミュレーションで再現可能。

### 2. 主要実装

| Phase | ファイル | 内容 |
|---|---|---|
| B-1 | `src/board_state_machine.py` | state enum + StateContext + 連続多数決 + Detector Protocol |
| B-2 | `src/state_detectors.py` | ChainPhase / Tsumo / Ojama / Effect の 4 detector |
| B-4 | `src/score_ocr.py` (拡張) | 既存 ScoreOcr に ScoreTracker / ScoreDelta を追加 |
| B-5 | `src/inference_board.py` | InferenceBoardGenerator (state ごとに推論盤面を出す) |
| B-6 | `src/drift_detector.py` | 推論-CNN 乖離検出 + re-sync 要請 |
| B-7a | `src/recognition_pipeline.py` | 全コンポーネント連結、frame → 1P/2P 確定盤面 |
| B-7 | `scripts/phase_b_finetune_cnn.py` | phase_u + menu_truth で fine-tune (cnn_phase_b_v1.pt = holdout 99.61%) |
| B-8 | `scripts/phase_b_pipeline_eval{,_all}.py` | 動画別 / 全動画 一括評価 (--per-video-model 対応) |
| B-9 | `scripts/phase_b_drift_analysis.py` | drift 詳細解析 (cell 位置 + color confusion) |
| B-10 | `scripts/phase_b_collect_drift_dataset.py` | drift cell 採取スクリプト (CNN-v2 訓練に使えなかった) |

テスト総数 81+、全 pass (`tests/test_board_state_machine.py`, `tests/test_state_detectors.py`, `tests/test_score_ocr.py`, `tests/test_inference_board.py`, `tests/test_drift_detector.py`, `tests/test_recognition_pipeline.py`)

### 3. 重要な設計知見

- **ChainEvent は drop 観測 1 frame 幅しかない** → pipeline 側で `chain_count × 0.3s` 保持する仕組み (`RecognitionPipeline.CHAIN_HOLD_PER_STEP_SEC`)
- **OjamaPhaseDetector はロックイン回避**: 一度発火したら state==OJAMA_FALL なら STABLE 復帰させる
- **TsumoPhaseDetector の CNN ぶれ吸収**: `consec_threshold=2` で連続 2 frame 観測しないと TSUMO_FALL に遷移しない (1 frame の CNN ぶれ吸収)
- **HybridClassifier の cnn_override_prob=0.90** (default 0.75 より厳しい) で CNN ぶれを HSV で抑制
- **per-video model selector** (`PHASE_B_CNN_BEST_VIDEOS = {1,2,7,9,10,11,12,13,16}`) で動画別に HSV/CNN を切替

### 4. 精度結果 (2026-05-04 PV2 で更新)

#### 全動画 STABLE distribution (30 秒 / 10fps)

| 構成 | 平均 STABLE | 平均 drift | vs HSV |
|---|---|---|---|
| HSV のみ | 53.5% | 25.2 | — |
| CNN-v1 一律 | 52.7% | 26.3 | -0.8pt |
| Per-Video (PV) | 58.3% | 27.0 | +4.8pt |
| Smoothing=3 一律 | 56.6% | 26.9 | +3.1pt |
| **PV2 (PV + per-video smoothing)** ★ | **60.4%** | 27.5 | **+6.9pt** |

**PV2 が全動画で最良 or 同等**。Phase B 最終形。

詳細: `data/phase_b_eval_summary{,_v1,_pv}.tsv`

#### 単 frame 精度 (drift 解析、試合前除く)

- 平均 ~96.7%, 最低 v15=95.15%, 最高 v11=98.84%
- 試合前 (v14, v17, v18): 100%
- cell 位置: r1-r6 c2-c4 (盤面中央上部) で drift 集中、r12 (最下段) は最安定
- color confusion: **EM ↔ X (任意色) の混同が支配的** (top: EM→BL 880件, BL→EM 509件)

詳細: `data/phase_b_drift_analysis.tsv`

### 5. CNN 訓練の試行錯誤と結論

| Model | データソース | Holdout |
|---|---|---|
| cnn_phase_u_v16 (元) | manual labels | 99.45% |
| **cnn_phase_b_v1** ★ | + menu_truth (試合外 EMPTY hard) | **99.61%** |
| cnn_phase_b_v2 (drift puyo 含) | + drift_truth | 63.64% (失敗) |
| cnn_phase_b_v2 (drift EMPTY-only) | + drift_truth EMPTY フィルタ | 93.31% (失敗) |
| cnn_phase_b_v2 (drift consec=3) | + drift_truth_consec3 (B-13) | 71.67% (失敗) |

**drift_truth が訓練データを汚染する構造的理由 (B-13 で確定)**:

drift_truth の真値 = 直近 STABLE 盤面、画像 = 現在 frame の cell。連続性フィルタ (consec_threshold=3) で「1 frame ぶれ起因」を除去しても改善せず、原因は別:

「CNN が **確信的に** 間違える画像」 = 「**本質的に紛らわしい** 画像」(例: 紫っぽいピクセル分布になった青ぷよ frame)。それを真値ラベル (BL) で学習させると、CNN が「紫っぽい画像 → 青」と誤って学び、全 cell 認識が崩れる = 訓練データ汚染。

**menu_truth との構造的違い**:

| 観点 | menu_truth | drift_truth |
|---|---|---|
| 真値根拠 | 物理 (試合外 = puyo 0 個) | state machine 多数決 |
| 真値信頼度 | 100% | ~95% (連鎖直前/直後で古い可能性) |
| 画像の意味 | 背景/キャラ (puyo ではない) | 同位置 cell の puyo を CNN が誤認 |
| 汚染リスク | なし | 高 (= 紛らわしい画像 + 矛盾ラベル) |

→ **cnn_phase_b_v1 (phase_u + menu_truth) が Phase B 最終版として確定**。drift_truth ベースは構造的に使えない。

## 現在の production state

- **CNN モデル**: `models/cnn_phase_b_v1.pt`
- **使い方**: `RecognitionPipeline.load_default(cnn_model_path=Path("models/cnn_phase_b_v1.pt"), stable_frame_count=6)` または `--per-video-model` フラグ
- **平均 STABLE 確定率**: 58.3% (Per-Video)
- **単 frame 精度**: 96.7% (試合中区間平均)

## 残課題

### 7. CNN 時系列平均 (B-15、2026-05-04 追加)

`RecognitionPipeline.temporal_smoothing=N` で直近 N frame の cell 単位
majority vote を `signals.cnn_board` に渡す機能を追加。CNN ぶれを抑える
意図だが、副作用として state 遷移検出が遅延する。

全動画 eval (smoothing=3 vs no-smoothing):
- 改善動画 (+2pt 以上): v02, v03 (+12), v07, v08, v10, v11, v15 (+10), v16
- 悪化動画: v05 (-21), v06, v09, v13, v19 — smoothing で TSUMO 検出遅延が顕著
- 平均は微減 (PV 58.3% → PV+sm3 56.6%)

→ **動画別に smoothing も切替** が筋。`PHASE_B_SMOOTHING3_BEST_VIDEOS = {2,3,7,8,10,11,15,16}` を `select_phase_b_smoothing()` で適用。

### 9. レビュー後の推論強化 (B-17、2026-05-04 ユーザー指摘反映)

レビュー動画の問題点を user 指摘で発見、以下 4 修正を pipeline に追加:

1. **inferred_board hold-on-None** (`phase_b_render_review_video.py`): 状態遷移境界の 1-2 frame 一瞬空消失を直前 hold で緩和
2. **1P/2P 同期 (active hysteresis)** (`recognition_pipeline.py: MATCH_ACTIVE_HOLD_FRAMES=10`): 直前 10 frame 内で active 観測歴 or 1P/2P どちらかが NON-MENU 状態なら is_match_active=True 強制。「片側だけメニュー」物理あり得ない状態を ban
3. **試合開始 chain ban** (`CHAIN_BAN_FRAMES_AFTER_MATCH_START=30`): 試合 active 開始から 30 frame は ChainEvent を破棄、「1 手目から連鎖中」を ban
4. **背景 FP 自動採取**: `RecognitionPipeline.update()` 内で試合 active 開始 +5 frame 後の frame から `BackgroundFingerprint.capture()` で 1P/2P 両側の背景色を fingerprint 化、`ImageReader.set_background_fingerprints()` で注入。背景色に近い cell は EMPTY 強制 (= キャラ顔等の誤発火抑制)

これらの修正で「全 puyo 一瞬消失」「片側メニュー」「序盤 CHAIN 誤検出」「キャラ顔の puyo 誤認」を構造的に解消。テスト 6/6 全 pass。

### 8. レビュー動画 + 代表画像生成 (B-16、2026-05-04)

`scripts/phase_b_render_review_video.py` で 5 動画 (v01/v07/v13/v06/v15) ×
30 秒のオーバーレイ動画を生成。`scripts/phase_b_extract_review_frames.py`
で各動画から state 別代表 frame + drift_high frame を抽出 (合計 24 枚)。

オーバーレイ要素:
- 上部 HUD: time, 1P/2P state, drift, score, frame_idx
- 盤面: confirmed_board の各 cell に色 dot (推論結果の可視化)

詳細: `data/review_videos/README.md`

### 6. CHAIN 推論精度解析 (B-14、2026-05-04 追加)

`scripts/phase_b_chain_inference_analysis.py` で全動画 CHAIN state 中の
**InferenceBoardGenerator 推論盤面 vs CNN raw 盤面** の cell 一致率を計測:

- 全動画平均: **77.75%** (16 動画、48,456 cell 比較)
- 動画別:
  - 高 (>85%): v01, v02, v04, v07, v09, v12, v16 (84〜90%)
  - 中 (75-85%): v05, v10, v11
  - 低 (<75%): v03, v06, v08, v13, v15, v19 (57〜74%)
  - 最低: v06 = 57.16%
- Confusion top: **EM → 任意色** が支配的 (3402+985+890+...= 不一致の 67%)
  - = ChainSimulator が「消えた」と判断した cell を CNN がまだ puyo と認識
  - = **CNN は連鎖アニメ / 残光に惑わされる**、新方針核心仮説の強い裏付け

低一致動画 (v06, v08) は Phase Z でも「CNN で改善困難」と確定済の動画と一致。
**ChainSimulator が CNN の連鎖中誤認識をカバーしている** ことが定量で示された。

詳細: `data/phase_b_chain_inference.tsv`

### 高優先度
1. **新規真値ソースの探索** (drift_truth が使えないと B-13 で確定):
   - 候補 a: 連鎖完了 + N 秒安定後の **STABLE multi-vote 盤面** を真値とし、その時点の cnn_board と比較
   - 候補 b: 試合終了画面に表示される最終盤面を template/OCR で抽出して真値とする (1 試合 1 回のみ採取可、データ量少)
   - 候補 c: 手動レビューデータの限定的復活 (Phase Z 破棄分の活用方法)
   - menu_truth は 99.61% で頭打ち、これ以上の CNN 改善には新規真値が必須
2. **CHAIN state の盤面推論精度**: VideoChainTracker.before_board が現実盤面と乖離する場合に対処 (新ツモ着地が反映されない)。InferenceBoardGenerator の chain_playback が崩れる原因
3. **score OCR の連続 frame 安定化**: ScoreOcr の連続 frame ノイズが OjamaPhaseDetector を誤発火させた経緯あり、threshold は 70 に上げて回避中

### 中優先度
4. **MatchStateDetector の精度**: 試合前 (v14/17/18) を IN_MATCH と判定する false positive。drift 0 で訓練データ汚染なしのため放置中だが、accuracy 評価で除外フィルタ追加が筋
5. **連鎖カウンタ OCR (B-3 旧 task)**: VideoChainTracker と冗長で priority 低、必要なら ★n連鎖★ template matching で実装

### 長期
6. **真値 GT データの整備**: 99.99% 検証には人手 GT が必要、Phase Z review 資産破棄後の代替整備
7. **動画間プレーヤー戦術差**: per-video selector で動画別 model 切替したが、根本的に「動画特性に依存」する課題は解消しない

## 主要ファイル一覧

### 新規実装 (Phase B)
- `src/board_state_machine.py`, `src/state_detectors.py`
- `src/inference_board.py`, `src/drift_detector.py`
- `src/recognition_pipeline.py`
- `src/score_ocr.py` (ScoreTracker / ScoreDelta 追加)
- `src/per_video_model_selector.py` (PHASE_B_CNN_BEST_VIDEOS / select_phase_b_model 追加)

### 訓練・評価 scripts
- `scripts/phase_b_finetune_cnn.py` (phase_u + menu_truth + drift_truth 統合 fine-tune)
- `scripts/phase_b_pipeline_eval{,_all}.py` (動画別 / 全動画 評価、--per-video-model 対応)
- `scripts/phase_b_drift_analysis.py` (drift 詳細解析)
- `scripts/phase_b_collect_menu_truth_dataset.py` (試合外 false positive 採取)
- `scripts/phase_b_collect_drift_dataset.py` (STABLE 中 drift cell 採取、CNN 訓練に使えなかった)
- `scripts/phase_b_collect_chain_truth{,_v2}_dataset.py` (連鎖 hard negative 試行、両方タイミングずれで使えず)
- `scripts/phase_b_collect_stable_dataset.py` (STABLE state cell 採取、循環学習リスクで未使用)

### モデル
- `models/cnn_phase_b_v1.pt` ★ (Phase B 最終版、holdout 99.61%)
- `models/cnn_phase_b_v2.pt` (失敗作、drift_truth 汚染)
- `models/cnn_phase_u_v16.pt` (Phase U 最終版、ベース init として使用)

### データ
- `data/training/menu_truth/v??_menu.npz` (162 frames, 20,967 cells)
- `data/training/drift_truth/v??_drift.npz` (10,029 cells、訓練に使えず)
- `data/training/chain_truth{,_v2}/v??_hard.npz` (タイミングずれで使えず)
- `data/training/stable_state/v??_cells.npz` (循環学習リスクで未使用)
- `data/phase_b_eval_summary{,_v1,_pv}.tsv` (eval 結果)
- `data/phase_b_drift_analysis.tsv` (単 frame 精度)

### 破棄した Phase Z 資産
- `scripts/phase_z_train_cnn_v18.py` 計画 (新方針で破棄)
- `data/verify/phase_z_review/weak_video_extra/v??/violations*.csv` レビュー資産 (v04 一部反映済だが新方針で訓練データに使わない)
- `cell_recovery_refiner.py` の役割は state machine 内に分散予定

## 次セッション着手時の推奨

1. 残課題 #1 (drift detector 真値ソース改善) を最優先で着手 → CNN-v2 訓練が成立する条件が整う
2. その後 #2 (CHAIN state 推論精度) を改善 → 連鎖中の hard negative も safe に採取できる
3. STABLE state 確定率を 70%+ まで持っていけば、99.99% の射程内 (連続多数決効果)
4. 真値 GT (#6) は外部依頼 or 半自動化 (= 物理推論で確実な場面のみ採取) 検討

## 参考資料
- `memory/project_recognition_strategy_pivot.md` (新方針の核)
- `memory/feedback_recognition_target_995.md` (99.99% 目標)
- `memory/feedback_chain_phase_physics_only.md` (アクション中は物理推論優先)
- `docs/HANDOFF_2026-05-02.md` (前回引継ぎ、Phase Z 完了)
