# Phase W (W16 まで) ハンドオフ — 2026-04-30

次セッションで作業継続するための完全引継ぎ。

## 1. 現在の Production パイプライン

### CNN (盤面色分類)
- **`models/cnn_phase_u_v16.pt`** ← 最新 production
  - v15 init + review×100 oversample (15 epoch、lr=2e-4、max-per-class=20K)
  - cross-video (2300 cells): 約 93.85%
  - v18_m03 (最弱動画): 79.44% (v7 78.33% から +1.11pt)
  - in-sample acc 93.74% (review labels 2700 件)
- 旧モデル: v7 (旧 production、91.76%)、v15 (review fine-tune、93.33%)

### StatePipeline 構成 (`src/state_pipeline.py`)

#### default ON
| 項目 | 効果 (field2 sparse) | 効果 (runtime 期待) |
|---|---|---|
| **CNN v16** | base 80.95% | base |
| **PerVideoCalibrator** (W11-C) | +3.81pt | +0.5pt (cross-video full) |
| EnhancedBoardTracker | -7pt (sparse 評価で過剰補正) | 連続フレームで正しく動作するはず |
| temporal_voting (W11-D, window=3) | 0pt (sparse) | 連続で動作 |
| score_eraser (W12-B) | 0pt (sparse) | chain 発火時のみ |
| pair_landing_check (W13-A、修正済) | 0pt (sparse) | next_pair 検出時のみ |

#### default OFF (実験的、過剰補正で害)
- BgEmptyDetector (W9-F) — bg-em 過剰
- ScorePhysicsRefiner (W10-B) — streak バグ
- ColorRecoveryRefiner (W11-A) — false positive 多
- ChainAnimationDetector (W14-C) — sparse 時間で誤発火
- PuyoStabilityRefiner (W15-A) — event 検出失敗時の誤動作

### NextDetector
- CNN v7 + HSV + **NextPairCentroid (W10-E)** の多数決
- StableNextDetector wrap (3 フレーム連続一致のみ採用)
- centroid: `models/next_pair_centroid_v1.npz` (28K next pair labels から学習)

## 2. 重大バグ修正履歴 (W14-W16)

| 修正 | 症状 | 原因 | 対策 |
|---|---|---|---|
| W10-G | W10 video で puyos 大量消失 | BG-EM/ScorePhysics 過剰 | default OFF |
| W14-A | PairLandingCheck が真の puyo 殺し | next_pair=None で全色 suspicious | 検出失敗時 skip + 同色 neighbor 許容 |
| W15-D | Field review production 34% | warmup の EM で TemporalVoting が color を 2/3 majority で潰す | Field review tool から warmup 削除 |
| W15-D | ChainAnim/Stability 過剰補正 | sparse 時間で誤発火 / event 検出失敗 | default OFF |

## 3. ユーザー review labels (累計 2935 cells)

### Full sheet review (`data/verify/phase_w_review/{name}/labels.csv`)

| Sheet | cells | match | review 完了 |
|---|---|---|---|
| v05_m55_full | 200 | v05 m55 | ✓ |
| v12_m54_full | 323 | v12 m54 | ✓ |
| v09_m02_full | 200 | v09 m02 | ✓ |
| v13_m02_full | 198 | v13 m02 | ✓ |
| v17_m11_full | 199 | v17 m11 | ✓ |
| v18_m03_full | 180 | v18 m03 (最難) | ✓ |
| v18_m08_full | 200 | v18 m08 | ✓ |
| v18_m15_full | 200 | v18 m15 | ✓ |
| v19_m06_full | 200 | v19 m06 | ✓ |
| v04_m07_full | 200 | v04 m07 | ✓ (W16-C 新規) |
| v06_m06_full | 179 | v06 m06 | ✓ (W16-C 新規、9行) |
| v17_m37_full | 180 | v17 m37 | ✓ (W16-C 新規、9行) |
| v19_m07_full | 199 | v19 m07 | ✓ (W16-C 新規) |
| v18_m03_field2 | 210 | v18 m03 (2 frame field 全体) | ✓ |
| violations_50_bg/v04..v19 | 16×50=800 | 16 動画 violation cells | ✓ |

**累計: 約 3068 cells (= 2268 full sheet + 800 violations)**

### Field review (盤面 crop + grid + label overlay)
新形式 (`scripts/phase_w_extract_field_review.py`、default 2 frame):
- v18_m03_field, v18_m03_field2, v18_m03_field2_v2, v18_m03_field2_v16fixed

## 4. 弱点動画ランキング (v7 baseline cross-video)

| 動画 | v7 acc | v16 acc | 備考 |
|---|---|---|---|
| v18_m03 | 78.33% | 79.44% | 最難 (色混同 GRN↔PUR↔YEL 主因) |
| v17_m11 | 90.95% | 96.98% (v15) | 改善大 |
| v19_m06 | 93.00% | 93.50% (v15) | やや改善 |
| v05_m55 | 57.00% | 58.00% | 元から低い (sparse rendering 由来?) |

v18_m03 の固定的 39 cell エラー (EM↔色混同) は静的パッチ評価で改善せず。**温度的文脈** 必要。

## 5. データセット (`data/training_phase_u/`)

| ファイル | サイズ | 内容 |
|---|---|---|
| manual_plus_strict.npz | 451,980 | manual + parallel pl1-pl4 strict mix (v7 訓練ベース) |
| pseudo_v7_all19.npz | 38,000 | 19 動画 × 2000 で v7+HSV pseudo label (W9-A) |
| v13_dataset.npz | 492,280 | manual+pseudo+review combined (v13 訓練用) |
| v14_dataset.npz | 570,980 | review×30 oversample 統合 (v14 訓練用) |
| v16_dataset.npz | (存在) | review×100 oversample (v16 訓練用) |
| next_pair_labels.npz | 28,576 | 19動画 next pair stable labels (W8-D) |

## 6. 主要スクリプト

### 訓練
- `scripts/phase_w_train_cnn_v14.py` (review×30 oversample、v7 init)
- `scripts/phase_w_train_cnn_v15.py` (v14 init + review-only fine-tune)
- `scripts/phase_w_train_cnn_v16.py` (v15 init + review×100 oversample)
- `scripts/phase_w_train_centroid.py` (centroid v3、2700 cells)
- `scripts/phase_w_train_next_centroid.py` (next pair centroid)

### 評価・デバッグ
- `scripts/phase_w_eval_cross_video.py` (2300 cells cross-video harness)
- `scripts/phase_w_isolate_refiners.py` (refiner ON/OFF 切り分け、W16-A 用)
- `scripts/phase_w_test_sample_ratio.py` (CELL_SAMPLE_RATIO 感度)
- `scripts/phase_w_test_roi_offset.py` (ROI offset grid search)
- `scripts/phase_w_diff_v9_v10.py` 等 model 比較

### Review シート生成
- `scripts/phase_w_extract_full_sheet.py` (全セル個別 patch sheet、20列)
- `scripts/phase_w_extract_field_review.py` (盤面 crop + grid overlay、default 2 frame)
- `scripts/phase_w_apply_w9b_review.py` (char-coded label 適用、現在 11 sheet)
- `scripts/phase_w_apply_field_review.py` (field2 用)

### Render
- `scripts/phase_w_render_full_review.py` (1 試合フル動画 rendering with overlay)
- 主要オプション: `--cnn-model`, `--bg-fp-time`, `--use-calib`, `--use-temporal`

## 7. 主要モジュール (Phase W で追加)

### 物理推論・補正系
- `src/centroid_classifier.py` — クラス別平均色 1-NN
- `src/bg_empty_detector.py` — 試合前盤面 EM 判定 (default OFF)
- `src/color_recovery_refiner.py` — EM→色 復活 (default OFF)
- `src/score_physics_refiner.py` — 旧 score 連動 (default OFF、バグあり)
- `src/score_eraser.py` — chain 発火→4+ cluster cell 強制 EM (default ON、修正済)
- `src/pair_landing_check.py` — 新着 cell ↔ next_pair 拘束 (default ON、W14 修正済)
- `src/temporal_voting_refiner.py` — N フレーム多数決 (default ON)
- `src/per_video_calibrator.py` — BGR shift キャリブ (default ON、+3.81pt 効果)
- `src/chain_animation_detector.py` — motion 検出 (default OFF、誤発火)
- `src/puyo_stability_refiner.py` — cell 不変拘束 (default OFF)

### CNN 系
- `src/patch_classifier.py` — 8x8 CNN (v7 アーキ)
- `src/patch_classifier_v2.py` — 16x16 ResNet (v10 アーキ、v7 を超えず)
- `src/next_pair_classifier.py` — next pair 専用 32x32 (auto label 不正で活用未)

### NextDetector
- `src/next_detector.py` — CNN+HSV+centroid 多数決
- `src/stable_next_detector.py` — N フレーム連続一致のみ採用

## 8. 試した実験 (v8〜v16) の結論

| モデル | アーキ | 訓練データ | cross-video | 採用 |
|---|---|---|---|---|
| v7 | 8x8 旧 | manual_aug20 + parallel_strict (451K) | 91.76% | 旧 production |
| v8/v9 | 8x8 | + v05 review + pseudo_v8 | < v7 | × |
| v10 | 16x16 ResNet+aug | manual_v05_pseudo (13K) | 87.53% | × (over-fit v05) |
| v11 | 16x16 + 大規模データ | strict + v05 pseudo | 89.57% | × |
| v12 | 8x8 + pseudo_v7_all19 | strict + 38K | 94.40% (1323 cells) | × (cross-video 全体で v7 超えず) |
| v13 | 8x8 + review 統合 | strict + pseudo + review | 89.96% | × |
| v14 | 8x8 + review×30 | + oversample 41K | 92.5% | × |
| v15 | 8x8 review fine-tune | v14 init + review only | 93.33% | 中継 |
| **v16** | **8x8 + review×100** | **v15 init + 強 oversample** | **93.85%** | **production** |

## 9. 次セッション開始ポイント

### 即座に実行可能
1. **v04/v06/v17/v19 の追加 review labels を CNN v17 訓練に統合**
   - `scripts/phase_w_train_cnn_v16.py` を v17 用に書き直し (v16 init + 累計 review×N)
   - 期待: cross-video +0.5〜1pt、v18_m03 が +1〜2pt
2. **EnhancedBoardTracker の runtime 挙動検証**
   - 連続フレームで N3/浮遊削除 がどれだけ発火しているか可視化
   - 過剰なら threshold 調整
3. **field review 結果から CNN v17 専用データセット構築**
   - field2 など最新の高品質 reviewed labels も統合
4. **video 19 のような hard 動画の連続 frame 評価**
   - render 時に「何フレーム連続で誤検出」を可視化

### 中期
- Per-video CNN fine-tune (v18 専用 v16 → v18 への補正模型)
- Animation frame 検出 v2 (chain 発火時の輝度急変を score+UI で検出、time gap robust)
- Multi-scale CNN (3x3 cells = 24x24 入力で空間文脈)

### Phase X 候補 (image recognition 卒業後)
- 強化学習向け state extraction 統合
- リアルタイム OBS overlay
- 別大会動画追加

## 10. ユーザールール (継続)

- 自律運転 OK (`feedback_autonomous_operation.md`)
- レビュー依頼は **Windows フルパス** で提示 (`feedback_review_image_links.md`)
- 文字化け対策済 (terminal フォント設定済)
- 「明らかに良い選択肢は聞かずに進める」(2026-04-29 確認)
- ユーザーは review labels を char-coded で送る (E/R/B/G/Y/P/O、? = effect 等不明)
- 物理推論 (連鎖/ツモ着地) と平均色マッチを継続改善方針

## 11. 動画レビュー成果物

最新 production レンダ:
```
data/verify/phase_w_results/full_review_v18_m03_w15fixed.mp4
```
(v16 + W11-C/D, score_eraser, pair_landing, ChainAnim/Stability OFF)

field 全体 (盤面 crop + grid + label):
```
data/verify/phase_w_review/v18_m03_field2_v2/field_sheet.png
```
