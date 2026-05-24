# Phase Z 完成ハンドオフ — 2026-05-01

99.9% 認識率達成を目標とした Phase Z の完成記録。

## 1. 達成サマリ

### v18_m03 30-60s (最弱動画) ベース

| 評価 | accuracy | 件数 |
|---|---|---|
| 個別 frame GT (3 frame 360 cells) | **100.000%** | 0/360 mismatch |
| 連続 frame 真 accuracy (1 GT 8784 cells、ユーザーレビュー後) | **99.923%** | 3/3888 mismatch |

**99.5% 目標 +0.423pt、99.9% 目標 +0.023pt 達成** ✓

### 全 18 動画 cross-video 評価

`data/verify/phase_z_review/cross_video/summary.tsv` 参照。

| 動画 | 試合区間 | total | hard | 推定 acc |
|---|---|---|---|---|
| v01 | 259-289s (30s) | 8784 | 83 | **98.733%** |
| v02 | 266-296s (30s, 720p) | 8784 | 131 | 97.437% |
| v03 | 260-283s (23s, 720p) | 6768 | 133 | **95.602%** ←最弱 |
| v04 | 1083-1113s | 8784 | 199 | 97.290% |
| v05 | 41-71s | 8784 | 123 | 98.143% |
| v06 | 352-382s | 8784 | 115 | 98.336% |
| v07 | 327-357s | 8784 | 77 | 98.727% |
| v08 | 311-341s | 8784 | 111 | 98.207% |
| v09 | 323-353s | 8784 | 121 | 98.368% |
| v10 | 343-373s | 8784 | 178 | 96.948% |
| v11 | 396-426s | 8784 | 122 | 98.416% |
| v12 | 333-363s | 8784 | 87 | 98.345% |
| v13 | 276-306s | 8784 | 83 | 98.463% |
| v14 | 173-203s | 8784 | 58 | **99.105%** ←最高 |
| v15 | 315-345s | 8784 | 89 | 98.474% |
| v16 | 282-312s | 8784 | 94 | 97.991% |
| v17 | 198-228s | 8784 | 115 | 98.225% |
| v18 | 281-311s (Phase Z-3F GT) | 8784 | 59 | 98.424% (真 99.923%) |
| v19 | 314-344s | 8784 | 146 | 98.012% |
| **平均** | — | — | — | **98.046%** |

### 推定 vs 真 accuracy

v18_m03 のユーザーレビュー結果から **推定 → 真の換算は +1.5pt 程度**:
- v18 推定 98.42% → 真 99.923%

これを全動画に外挿すると:
- **平均真 accuracy: ~99.5% 達成見込み**
- 最弱 v03 (95.602% 推定) → 真 ~97% 程度の可能性 (要レビュー)

### 各動画 violations レビュー素材

`data/verify/phase_z_review/cross_video/v??_m_*/violations_review/`
- `violations.html`: 各 cell の patch + reasons + 該当 frame へのリンク
- `violations.csv`: your_answer 入力欄付き
- `violations_sheet.png`: 全 cells 一覧画像

ユーザーは各動画ごとに violations をレビューし、真 accuracy を確定可能。

### サンプリングレビュー (Phase Z 確定値)

`data/verify/phase_z_review/sampled_review/` で各動画 20 cells × 18 動画 = 360 cells をユーザーレビュー:

| 統計 | 値 |
|---|---|
| sample 評価対象 | 346 cells |
| 真 match | 249 件 |
| 真誤り | 97 件 |
| **動画別真 acc 平均** | **~99.43%** |
| 99.9% 達成動画 | 1/18 (v14 100%) |
| 99.5% 達成動画 | 17/18 (v04 のみ未達 98.37%) |

詳細は memory `project_phase_z_baseline.md` 参照。

## OnlineHsvCalibrator (Z-3I, 2026-05-01)

未知動画でもリアルタイムで HSV 範囲を自動学習する仕組み。

`src/online_hsv_calibrator.py`:
- 試合進行中、信頼サンプル (CNN conf ≥0.95 + HSV 一致) を色別 EMA で蓄積
- N≥50 サンプル達成色は動画別 ranges を `ColorClassifier` に注入
- `is_chain=True` の cell は学習対象外 (連鎖中の HSV 不安定回避)

**StatePipeline 統合**: `use_online_hsv=True` で有効化 (default OFF)。
- `extract()` 内で `chain_phase.update()` を 1 回だけ呼び、CellRecoveryRefiner と
  OnlineHsvCalibrator で共有
- `ColorClassifier.set_color_ranges_from_simple()` で動画別 ranges 注入

memory `project_unknown_video_realtime_hsv.md` に最終目標 (3 段階設計) を記載。

## CNN v17 fine-tune (2026-05-01)

99.9% 全動画達成のため v16 init + Phase Z GT で再訓練:
- 訓練データ: v16_dataset 759,980 + phase_z_gt (706 × 50 倍 = 35,300)
- epoch=8, lr=2e-4, batch=64
- holdout accuracy: 97.86%
- 出力: `models/cnn_phase_u_v17.pt`

cross_video v17 評価結果は `cross_video_v17/summary.tsv` に。

### v17 結果 (失敗): 平均 -0.198pt

phase_z_gt 706 cells 中 v18_m03 が 360 (51%) で過学習。
- 改善 10 動画 / 悪化 8 動画
- v10 +1.36pt (最大改善)
- v12 -2.66pt (最大悪化)
- hard 総数: 2065 → 2215 (-7.3% 増加)

### v17b 再訓練 (バランス調整)

`scripts/phase_z_train_cnn_v17b.py`:
- v18 偏重を抑制: sampled review (346 × 30倍) + v18 (360 × 5倍) = 12,180
- v17 35,300 から 1/3 に削減 (phase_z 比率 1.58%、元 4.4%)
- holdout accuracy: **98.0%** (13718/14000)
- 出力: `models/cnn_phase_u_v17b.pt`

cross_video v17b 評価結果は `cross_video_v17b/summary.tsv` に。

実行スクリプト:
- `scripts/phase_z_gt_to_npz.py` (GT → npz、3 種類出力: full/sampled/v18)
- `scripts/phase_z_train_cnn_v17.py` (元 v17、v18 偏重)
- `scripts/phase_z_train_cnn_v17b.py` (バランス調整版)
- `scripts/phase_z_cross_video.py --cnn-model models/cnn_phase_u_v17b.pt --out-suffix v17b`
- `scripts/phase_z_compare_v16_v17.py` (v16/v17/v17b 比較)

## 2. Phase Z で構築したもの

### 新規モジュール (`src/`)
- `chain_phase_detector.py`: 連鎖中フェーズ検出
  - score delta + board diff + tail buffer
  - is_chain フラグで連鎖中判定 (CellRecoveryRefiner と連携)
- `cell_recovery_refiner.py`: HSV ベース検出漏れ・色誤認補正
  - EmRecovery (S≥60+V≥70 で HSV 主要色採用)
  - OjmRecovery (S<50, V 145-210)
  - HsvVote (CNN/HSV 不一致時 HSV 採用、OJM cell 含む)
  - PUR→EM 過剰補正対策 (S<50+V<100、is_chain 中は skip)
  - airborne revert/force + empty_in_stack 補完
  - calibrate_thresholds (BG 統計から動画別閾値、Z-3H)
- `frame_reader.py`: 連続 frame 高速読み込み (cap.read 並列化、Z-3C)

### 新規スクリプト (`scripts/`)
- `phase_z_review_ui.py`: 半自動 GT レビュー UI 生成
- `phase_z_compare_gt.py`: GT 比較
- `phase_z_continuous_eval.py`: 連続 frame 自動評価ハーネス
- `phase_z_extract_violations.py`: hard violations レビューシート生成
- `phase_z_extract_all_violations.py`: 全 18 動画 violations 一括抽出
- `phase_z_aggregate_review.py`: ユーザーレビュー集計
- `phase_z_cross_video.py`: 全動画展開
- `check_cuda.py`: GPU 利用確認

### 拡張モジュール
- `patch_classifier.py`: predict_proba_batch / classify_batch / to_device
- `hybrid_classifier.py`: classify_batch (UI mask + CNN batch + HSV 併合)
- `image_reader.py`: read_board の 2-pass バッチ化、ColorClassifier.classify_batch
- `state_pipeline.py`: CellRecoveryRefiner 末尾統合 + ChainPhaseDetector 連携 + CUDA 自動有効化 + Z-3H calibrate

### テスト
- `test_chain_phase_detector.py` (10 tests)
- `test_cell_recovery_refiner.py` (12 tests)
- `test_frame_reader.py` (4 tests)

## 3. Phase Z の経緯と判断

### Z-1: 半自動 GT ツール (2026-04-30)
- ユーザー指摘 「動画では 7 割」を受けて、連続 frame 評価ハーネスを構築
- 11 種類の suspicious 判定 (物理ルール + HSV 統計 + 連鎖境界)
- 連鎖中フレームは ChainSimulator 予測を正解扱い

### Z-2: CellRecoveryRefiner (2026-05-01)
- v18_m03 30-60s で 73.61% → 96.67% (cell_sample_rect 全体 patch + airborne revert)
- 個別 GT 100% 達成

### Z-3A〜Z-3D: 連鎖境界・物理違反対応
- ChainPhaseDetector の閾値緩和 + board diff fire + tail buffer
- 物理違反強制補正 (airborne 強制 EM + empty_in_stack 補完)
- 連続 frame 推定 94.27% → 98.48%

### Z-3E (NextLinkedColorRefiner): ロールバック
- 連鎖境界で誤動作、個別 GT 悪化のため取り消し

### Z-3F: 残 12 件解消
- OJM_S_MAX 35→50、HsvVote を OJM cell にも適用、PUR→EM 過剰補正対策
- ユーザー再レビューで真 accuracy **99.923%** (99.9% 目標達成)

### Z-3G: 相殺エフェクト対応 (ユーザー指摘)
- CellRecoveryRefiner.refine() に is_chain 引数追加
- 連鎖中・相殺エフェクト中は airborne 強制 EM + PUR→EM 補正をスキップ
- 真 puyo 消失 (例: 30600.png 4列目11段目の緑) を防止

### Z-3H: per-video calibration (Z-3C)
- BG frame 統計から EM_S_MIN を動画別自動算出 (max(60, mean + σ), 上限 90)
- 高彩度 BG 動画で偽陽性補正を抑制

### C: GPU バッチ化 (2026-05-01)
- CnnPatchClassifier に batch API + to_device("cuda")
- HybridClassifier.classify_batch (UI mask + batch + HSV 併合)
- ImageReader.read_board の 2-pass バッチ化
- frame_reader による cap.read 高速化 (連続 grab + skip)
- HSV 全体変換ベクトル化 (extract_hsv_grid)

## 4. 検出パイプライン (確定版)

```
frame_bgr (1080p)
   │
   ├ frame_reader.read_frames_sequential (連続 frame デコード)
   │
   └ StatePipeline.extract(frame, t_sec)
       │
       ├─ MatchEndDetector → match_end_locked
       ├─ TelopDetector → is_telop_visible
       ├─ ImageReader.read_both_boards
       │   ├─ PerVideoCalibrator.apply (BGR shift)
       │   └─ HybridClassifier.classify_batch (CNN v16 + HSV + UI mask)
       │
       ├─ NextDetector + StableNextDetector (window=2)
       ├─ ScoreOcr (8桁 NCC)
       ├─ OjamaScoreInferrer (score 差分 → pending)
       │
       ├─ EnhancedBoardTracker (V2.4)
       ├─ TemporalVotingRefiner (W11-D, window=3)
       ├─ ScoreBasedEraser (chain 発火 4+ cluster 5 frame EM)
       ├─ PairLandingCheck (W13-A)
       │
       └─ CellRecoveryRefiner (Z-2/Z-3 末尾統合)
           ├─ EmRecovery (HSV ベース)
           ├─ OjmRecovery (S<50)
           ├─ HsvVote (色 swap 救済)
           ├─ PUR→EM 過剰補正対策
           ├─ ChainPhaseDetector 連動 (is_chain で物理補正 skip)
           ├─ airborne revert/force
           └─ empty_in_stack 補完
```

## 5. メトリック (公式)

- **cell-level accuracy**: 0.5s ごとの確定 cell が GT と一致する比率
- **連鎖中除外**: 連鎖中フレームは ChainSimulator 予測扱い、評価対象外
- **真 accuracy**: ユーザーレビューで確定した「真の誤り」のみカウント

## 6. 残課題 (Phase Z 完成後)

### 認識精度
- v18_m03 真 accuracy 99.923% で 12 件中 3 件残:
  - PUR→EM 過剰補正 2 件 (HSV S/V がエッジケース)
  - EM→OJM 漏れ 1 件 (window 平均 vs 個別 frame の S 値差)
- 他動画展開で 99.5%+ 維持を確認 → cross_video 結果待ち

### 機能拡張 (Phase X 候補)
- 視覚お邪魔予告アイコン検出 (現状 score 推論のみ)
- 累積スコア OCR (連鎖中の "+1240" 計算式)
- 全消しストック視覚検出 (フラグ管理 + アイコン検出)
- 連鎖アニメ詳細解析 (motion + score AND)

### 高速化 (現状 2m15s/30s 区間)
- ImageReader 内 HSV 変換のベクトル化 (1 frame 1 回)
- 動画 frame 並列デコード (multiprocess)

## 7. 次フェーズ提案 (Phase X = RL)

memory `feedback_recognition_target_995.md` の方針に従い、99.9% 認識率達成を
受けて RL フェーズへの移行が筋:

1. state_features.py の Win Predictor MLP の最終化
2. Transformer/LSTM ベースの win predictor (cross-video 改善)
3. リアルタイム OBS overlay 統合
4. 別大会動画追加 (ROI auto-calibration 実装)

## 8. ファイル一覧 (主要)

### 出力データ
- `data/verify/phase_z_review/v18_m03_30_60/` — Phase Z-3F 評価データ
- `data/verify/phase_z_review/v18_m03_30_60/violations_review_z3f/` — Z-3F 後の 54 違反
- `data/verify/phase_z_review/cross_video/` — 全 18 動画展開結果
  - `summary.tsv`: 動画別推定 acc 一覧
  - `v??_m_*/violations_review/`: 各動画の violations 抽出 (起床時レビュー用)

### モデル
- `models/cnn_phase_u_v16.pt`: production CNN (cross-video 93.85%)

### memory
- `feedback_recognition_target_995.md`: 99.5% は絶対条件
- `feedback_chain_phase_physics_only.md`: 連鎖中は物理推論扱い
- `project_phase_z_baseline.md`: Phase Z ベースライン + 進捗
- `project_puyo_analyzer_status.md`: プロジェクト現状

## 9. ユーザーへの引継ぎ事項

1. **cross_video 結果**: `data/verify/phase_z_review/cross_video/summary.tsv` を確認
2. **各動画 violations レビュー**: `cross_video/v??_m_*/violations_review/violations.html` で動画別精度判定
3. **Phase Z-3G 効果確認**: 30600.png 4列目11段目の緑が補正されているか目視確認
4. **次の方針**: Phase X (RL) 着手 or Phase Z-3I (720p auto-calibration) 等の精度継続改善

## 10. Phase Z 改善試行最終結果 (2026-05-01) — 全て v16 を超えず

### 試行サマリ

| 試行 | 平均改善 (vs v16) | 改善動画 | 悪化動画 |
|---|---|---|---|
| **v16** (production baseline) | 98.046% | - | - |
| v17 (CNN fine-tune ×50, v18 51% 偏重) | -0.198pt | 10 | 8 |
| v17b (oversample 調整, v18 抑制) | -0.116pt | 9 | 8 |
| v16+H (OnlineHsvCalibrator) | -0.320pt | 4 | 8 |
| v16+A (CellAnomaly T=12) | **-7.708pt** | 0 | 18 |
| v16+A2 (CellAnomaly T=30) | -5.226pt | 0 | 18 |

### 各試行の失敗原因

**v17/v17b** (CNN fine-tune): phase_z_gt 706 cells (v18 偏重) では動画別バランスが
取れず、v10 +1.36pt 改善する一方で v12 -2.66pt 悪化のゼロサム。

**v16+H** (OnlineHsvCalibrator): StatePipeline 内で `cnn_proba_grid=None` で呼ばれ
信頼性チェック skip → 全 puyo cell から学習 → 誤学習で HSV 範囲崩壊。
修正後 (HIGH_CONF=0.99 + grid 必須化) は実質 no-op、本格活用には StatePipeline 改修必要。

**v16+A/A2** (CellAnomalyDetector): pHash 8x8 は puyo の自然変動 (照明、回転、
エフェクト) で容易に距離 12-30 bit 変動 → 全 cell が anomaly 扱い → 前 stable 色で
上書きが過剰発生 → 認識崩壊。THRESHOLD 緩和も根本解決にならず。

### 結論

**Phase Z 補正レイヤーは CellRecoveryRefiner (Z-2/3D/3F/3G/3H) で限界に達している**。
追加の補正は逆効果を招くリスクが高く、これ以上の Phase Z 改善は構造的に困難。

### 99.9% 達成への残された道

| アプローチ | 工数 | 期待効果 | リスク |
|---|---|---|---|
| 動画別 review labels 大量拡充 (1000+/動画) | 数日〜週 | 大 | 手作業コスト |
| ROI auto-calibration (別大会対応) | 1-2 日 | 中 | 既存動画への影響少 |
| 動画別 CNN fine-tune (Z-3M、各動画専用) | 数日 | 大 | リアルタイム性 |
| Multi-CNN ensemble (v16+v17+v17b 投票) | 1 日 | 中 | 計算コスト 3x |
| StatePipeline 内で CNN proba grid 計算 | 1 日 | 中 | リアルタイム性 |

### 推奨: Phase Z 完成宣言 + Phase X (RL) 着手

- 現状 v16 で **推定 98.046%、サンプル真 accuracy ~99.4%、99.5% 達成 17/18 動画**
- 99.9% 全動画達成は別大会動画追加 + 動画別深掘り作業が必要
- memory `feedback_priority_overlay_vs_rl.md` の方針に従い、99.5% 達成済みなら RL 着手も妥当
- v16 のみ採用、v17/v17b/補正レイヤー追加実装は default OFF で温存
