# Phase Z 完成ロードマップ (2026-05-01 更新版)

「全動画で 99.9% 認識率 + 未知動画リアルタイム対応」最終ゴールへの工程表。

## 1. 現状 (2026-05-01 12:00 時点)

### 達成済
- v18_m03 30-60s で **真 accuracy 99.923%** (Phase Z-3F)
- 全 18 動画 cross_video 推定平均 **98.046%**
- サンプリング (各 20 cells × 18 動画) で動画別真 accuracy ~99.4%
- 99.5% 達成: **17/18 動画** (v04 のみ未達 98.37%)
- 99.9% 達成: 1/18 動画 (v14 のみ)

### CNN fine-tune の限界
- v17 / v17b 共に v16 平均を超えられず (-0.116〜-0.198pt)
- 一部動画で改善 (v10 +1.2pt) するが他動画で悪化 (v12 -1.6pt)
- phase_z_gt 706 cells は動画別偏りに対し不十分

### 結論
**画面情報取得の品質強化は CNN 単独ではなく、補正レイヤーと動画別キャリブで実現すべき**

## 2. 残課題と Phase Z-3J 以降

### Z-3I: OnlineHsvCalibrator ✅ 実装、cross_video 評価結果: 逆効果

`src/online_hsv_calibrator.py` 実装済 + 7 tests pass。

**v16+HSV cross_video 結果**: 平均 -0.320pt (vs v16)、最悪 v13 -4.57pt。
原因: StatePipeline 内で `cnn_proba_grid=None` で呼ばれ信頼性チェック skip → 全 cell から学習 → 誤学習で HSV 範囲崩壊。

**修正**: 閾値厳格化 (HIGH_CONF 0.95→0.99、MIN_SAMPLES 50→200) + 
grid 必須化 (None なら学習 skip)。**現状は実質 no-op、本格活用には StatePipeline 内で
CNN proba grid 計算が必要 (大改修)**。Phase Z-3I' として継続検討。

### Z-3J: Cell Anomaly Detection ✅ 実装完了

`src/cell_anomaly_detector.py` で実装:
- 各 cell の patch を 8x8 dHash 化 (高速)
- 直前 N=3 frame と Hamming 距離で異常検出
- 距離 >12 で anomaly フラグ → 直前 stable 色で上書き
- 連鎖中 (is_chain=True) は対象外

統合:
- StatePipeline `use_cell_anomaly=True` で有効化 (default OFF)
- CellRecoveryRefiner の後段に配置
- phase_z_review_ui.py / cross_video.py に `--use-cell-anomaly` 追加

テスト 6 件 pass。次に cross_video で効果検証。

**期待**: 連鎖アニメ・落下中の不安定 cell を識別 → スコア swing を抑制

### Z-3K 候補: 重み付き Multi-frame Voting (30 分)

現状 TemporalVotingRefiner (window=3) を拡張:
- window 3 → 5 (= 0.5s)
- CNN 確信度で重み付き投票 (高確信度 cell の vote が重い)
- 連続 stable cell は信頼度 boost

実装は既存 `temporal_voting_refiner.py` の小改修。

### Z-3L 候補: ROI auto-calibration (2-4 時間)

未知動画 (別大会、別 UI) で ROI 座標が異なる場合への対応:
- 試合開始 frame から UI マーカー (NEXT 枠の青枠線、Score 0 桁) を検出
- ROI offset (x_shift, y_shift) を auto 算出
- 別動画別解像度動画でも 1080p ハードコード ROI を補正

**手法**: テンプレートマッチ + Hough 変換で UI 直線検出

### Z-3M 候補: 動画別 CNN fine-tune オートメーション (大、6+ 時間)

各動画専用 CNN を auto 訓練するフレームワーク:
1. 動画再生中に高確信度 puyo cell を蓄積
2. 200+ サンプル/色蓄積後、専用 CNN を fine-tune (in-memory)
3. fine-tune CNN を以降の認識に使用

**未知動画リアルタイム対応の段階 3** (memory `project_unknown_video_realtime_hsv.md`)
**ただし計算コスト大**、リアルタイム fine-tune は GPU 4060 Laptop で実用的かは要検証

## 3. 優先度マトリクス

| 案 | 工数 | 効果見込 | 優先度 | 依存 |
|---|---|---|---|---|
| Z-3I OnlineHsvCalib (評価) | 完了 | 動画別 HSV 補正 | - | 評価結果待ち |
| Z-3K 重み付き voting | 30 分 | フリッカー抑制 | **★★★** | なし |
| Z-3J Cell Anomaly | 1-2 時間 | 評価値 swing 抑制 | **★★** | Z-3K 完了後 |
| Z-3L ROI auto-calib | 2-4 時間 | 別大会動画対応 | ★ | 別動画ソース必要 |
| Z-3M 動画別 fine-tune | 6+ 時間 | 究極の動画別最適化 | ★ | Z-3K/J 効果次第 |

## 4. 推奨ステップ (13:30 帰宅後)

### Step 1: OnlineHsvCalibrator 結果確認
- v16+HSV cross_video の v16/v17/v17b との比較
- 改善あれば本採用、なければ Z-3K へ

### Step 2: Z-3K 重み付き voting (即着手)
- 30 分実装、効果測定
- フリッカー抑制が確認できれば accuracy +0.2-0.5pt 期待

### Step 3: Z-3J Cell Anomaly (改善継続)
- 1 cell 違い問題の根本対処
- 評価値 swing が大きい場面で効く

### Step 4: 99.9% 達成判定 + Phase X 着手 or 継続改善

## 5. Phase X 候補 (Phase Z 完成後)

memory `feedback_priority_overlay_vs_rl.md` に従い、99.9% 達成後:
- Phase X-1: state_features の確定 + Win Predictor 強化
- Phase X-2: Transformer/LSTM win predictor (cross-video 改善)
- Phase X-3: リアルタイム OBS overlay 統合
- Phase X-4: 別大会動画追加 + ROI auto-calibration 強化

## 6. ファイル一覧 (Phase Z 完成時)

### コアモジュール
- `src/cell_recovery_refiner.py` (Z-2/3D/3F/3G/3H)
- `src/chain_phase_detector.py` (Z-1/3A/3B/3C)
- `src/online_hsv_calibrator.py` (Z-3I)
- `src/cell_anomaly_detector.py` (Z-3J、未実装)
- `src/temporal_voting_refiner.py` (Z-3K で拡張予定)

### 評価スクリプト
- `scripts/phase_z_review_ui.py` (半自動 GT)
- `scripts/phase_z_continuous_eval.py` (連続 frame 評価)
- `scripts/phase_z_cross_video.py` (全動画展開)
- `scripts/phase_z_compare_v16_v17.py` (CNN 比較)
- `scripts/phase_z_sample_violations.py` (サンプリング)
- `scripts/phase_z_aggregate_sampled.py` (集計)

### 訓練スクリプト
- `scripts/phase_z_gt_to_npz.py` (GT 抽出)
- `scripts/phase_z_train_cnn_v17.py` / `v17b.py` (fine-tune)

### モデル
- `models/cnn_phase_u_v16.pt` (production)
- `models/cnn_phase_u_v17.pt` / `v17b.pt` (失敗、参考のみ)

### Memory (永続化)
- `feedback_recognition_target_995.md` (目標 99.5%、後に 99.9%)
- `feedback_chain_phase_physics_only.md` (連鎖中処理)
- `project_phase_z_baseline.md` (進捗履歴)
- `project_unknown_video_realtime_hsv.md` (最終目標)
- `reference_puyo_ai_recognition.md` (先行研究)
