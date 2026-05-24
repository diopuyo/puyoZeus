# 引継ぎ: cycle 71v Large CNN v2 default 昇格 (2026-05-14)

## 完了サマリ

cycle 71v v2 (`cnn_phase_b_large_v2.pt`, val 98.87%) を `RecognitionPipeline.load_default` の system default に昇格。 併せて、 base class `CnnPatchClassifier` の `_patch_to_tensor` 喪失 regression を修正。

## 変更内容

### 1. v2 モデルを default に昇格 (`src/recognition_pipeline.py`)

- 新規定数: `RecognitionPipeline.DEFAULT_CNN_MODEL_PATH = Path("models/cnn_phase_b_large_v2.pt")`
- `load_default(cnn_model_path=None)` の意味を変更:
  - 旧: HSV-only fallback
  - 新: `DEFAULT_CNN_MODEL_PATH` を試し、 存在すれば HybridClassifier (HSV+CNN) を構築。 不在なら HSV-only fallback (旧挙動)
- 明示的に Path を渡した caller は影響なし
- HSV-only を強制したい caller は皆無 (既存 `diagnose_hsv_only.py` は別途 override_prob で disable)

### 2. `_patch_to_tensor` を base class に移動 (`src/patch_classifier.py`)

- cycle 71v 案 D 時に `_patch_to_tensor` が `CnnPatchClassifierLarge` 側のみに置かれた regression
- base `CnnPatchClassifier` の `classify` / `predict_proba` / `fit` が AttributeError でした
- `tests/test_cell_color_fine_tuner.py::test_fine_tune_synthetic_dataset` が失敗していた
- 修正: base class に `_patch_to_tensor` を移植、 Large 側の重複定義を削除 (継承で利用)

## テスト

- `tests/test_recognition_pipeline.py`: 9 passed
- `tests/test_cell_color_fine_tuner.py`: regression 解消
- 広域 tests/ 走行 (test_self_supervised_integration.py 除外): **1815 passed / 6 skipped / 6 failed**
  - 失敗 6 件は全て pre-existing (本作業と無関係):
    - `test_patch_classifier::TestHsvPatchClassifier::test_delegates_to_color_classifier` ── HSV 合成サンプル精度 85.7% / 90% 閾値境界 (CNN は無関係)
    - `test_timeseries_wrapper.py` 5 件 ── 指標数 assertion (47 vs 期待 45、 後の指標追加で更新漏れ)

## 残課題と次工程

### 引継ぎから持ち越し (cycle 71v viz レビュー)

- **v89 試合 2**: 1P 1 列目 1 段目の黄色 EMPTY 誤認 (1 cell)
- **v50 全消し overlay**: cell として認識される
- v20 viz 6 本生成済、 ユーザーレビュー前 (リンクは `HANDOFF_2026-05-14_cycle71v.md` 参照)

### 次工程候補 (優先度順)

1. **EffectPhaseDetector 統合で全消し overlay 機械的解消** (= 既存検出器を pipeline に組み込み済か確認、 未統合なら適用)
2. **Phase L 本番化** (yt-dlp で動画追加 DL、 全動画 regen、 CNN 事前学習)
3. **配信オーバーレイ (Phase J)** ── 認識精度が運用ライン (99%+) に達したと判断する場合
4. **v89 単 cell 誤認の追加ラベリング** (= データ薄さ起因か構造的問題かの判別はユーザーレビュー後に確定)

### 設計判断記録

- 認識目標 99.99% (memory `feedback_recognition_target_995.md`) は緩和判断待ち。 v2 で val 98.87% (= 99% 弱) のため、 次工程移行と並走で詰める方針が現実的
- 自律運転 (memory `feedback_autonomous_operation.md`) は継続適用

## 関連ファイル

### 編集
- `src/recognition_pipeline.py:DEFAULT_CNN_MODEL_PATH` 新規 + `load_default` 自動解決
- `src/patch_classifier.py:CnnPatchClassifier._patch_to_tensor` 移植 / Large 重複削除

### 引継ぎ親
- `docs/HANDOFF_2026-05-14_cycle71v.md` (v2 訓練・viz 生成までの記録)
