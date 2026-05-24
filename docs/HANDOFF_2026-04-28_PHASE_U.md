# セッション引継ぎ 2026-04-28 Phase U (盤面認識クリーン再構築)

## 1 行サマリ

Phase T v2 (融合判定 + 擬似 self-training) は archive 化、Phase U で
**HSV 改善 + 手動ラベル CNN + UI Mask + 試合状態判定** に再構築。
全 4749 件データセット精度 **97.94% (CNN v3)**、Holdout 98.5%。

## 達成事項

### Phase U-0 クリーンアップ
- archive/legacy_phase_t_v2/ に Phase T v2 系 5 モジュール + 6 スクリプト + 6 学習済モデル + 訓練データ + ログを退避
- src/image_reader.py を簡素化 (融合判定削除)
- 流用継続: stateful_board_tracker, physics_sanity, next_detector, chain, animation_filter, background_fingerprint

### HSV 範囲調整 (本流 src/image_reader.py)
```
COLOR_RED:    H 0-18 / 166-180, S>=110, V>=100
              + BGR R-G>=80 で黄と区別 (H 11-18 範囲)
COLOR_BLUE:   H 100-130, S>=100, V>=80
COLOR_GREEN:  H 50-85,  S>=100, V>=80
COLOR_YELLOW: H 14-38,  S>=30,  V>=180  (低彩度クリーム色対応)
COLOR_PURPLE: H 130-165, S>=80, V>=80
CELL_SAMPLE_RATIO: 0.5  (中央 50%)
判定順序: 色閾値 → OJAMA
```

### UI Mask 強化
- `models/ui_templates/x_mark_video01.png`, `x_mark_video01_b.png`, `x_mark_video01_c.png` 追加
- ImageReader に `use_ui_mask=True` 追加 (デフォルト ON)

### CNN 訓練 (Phase U-3)
| Model | データ | epoch | lr | Holdout | 全データ |
|---|---|---|---|---|---|
| cnn_phase_u_v1.pt | 1500 件 (元) | 40 | 0.002 | 95.0% | - |
| cnn_phase_u_v2.pt | 17365 件 (拡張) | 25 | 0.001 | 98.6% | 98.36% |
| **cnn_phase_u_v3.pt** | **23745 件 (拡張、バッチ 1-4)** | 25 | 0.0008 | **98.5%** | **97.94%** |

### 新規モジュール
- `src/hybrid_classifier.py` - HSV+CNN+UIMask 統合分類器
- `src/roi_calibration.py` - ROI 自動キャリブ (検出未成功、要改善)
- `src/console_init.py` - UTF-8 化 + Windows パス変換

### 新規スクリプト (Phase U)
- `scripts/phase_u_extract_samples.py` - ラベル候補シート生成 (大文字認識色)
- `scripts/phase_u_batch_extract.sh` 〜 `phase_u_batch4_extract.sh` - 各バッチ生成
- `scripts/phase_u_build_dataset.py` - csv → npz データセット構築
- `scripts/phase_u_augment.py` - データ拡張 (5 倍)
- `scripts/phase_u_train_cnn.py` - CNN fine-tune
- `scripts/phase_u_eval_classifiers.py` - HSV/CNN/Hybrid 比較評価
- `scripts/phase_u_compare_labels.py` / `phase_u_diff_batch1.py` - ラベル一致率分析

### 試合状態判定統合
- ImageReader に `use_match_state=True` 追加
- 試合中以外 (NOT_IN_MATCH) は両盤面強制 EMPTY を返す

## ユーザのレビュー実績

| バッチ | 試合 | レビュー件数 | 修正 |
|---|---|---|---|
| 1 | m1, m2, m3, m6, m8, m9, m10, m11, m13 (m14 除外) | 450 件 | 数件修正 |
| 2 | m4, m5, m7, m15-25 | 500 件 | 数件修正 |
| 3 | m18, m21, m22, m24, m26-32 | 500 件 (未レビュー) | - |
| 4 | m31, m33-46 (10 試合) | 500 件 | 55 件修正 |

合計 **1000 件レビュー済** + 1000 件未レビュー。

## ユーザフィードバック

- 「empty 判定に何あり、色だけでは背景/エフェクトが認識できない」
  → CNN v3 で 55 件の background/effect EMPTY サンプルを追加学習で対応
- 「ラベル形式 OK で 1000 件レビュー可」
  → 大文字色付き背景レイアウトで進行中

## 残課題

| 課題 | 優先度 | 対応案 |
|---|---|---|
| バッチ 3 のレビュー未着 | 高 | ユーザ時間ある時にレビュー |
| ROI 自動キャリブ (Hough) 検出失敗 | 中 | テンプレートマッチ等の別手法 |
| 配信用本番統合 | 中 | HybridClassifier + use_match_state=True で運用 |
| 720p 動画 (video_02, video_03) の ROI | 低 | ROI 自動キャリブ完成後 |

## 再開コマンド

```bash
# 全テスト走行
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m pytest tests/ -q --ignore=tests/test_video_processor.py"

# 分類器精度評価
wsl -- bash -c "cd /mnt/c/.../puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_eval_classifiers"

# シート生成 (任意の試合)
wsl -- bash -c "cd /mnt/c/.../puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_extract_samples \\
    data/frames/video_01.mp4 \\
    --times 200,210,220,230 \\
    --out-dir data/verify/phase_u_test \\
    --max-samples 50 --side both --bg-fp-time 188"

# データセット再構築 + 訓練
wsl -- bash -c "cd /mnt/c/.../puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_build_dataset && \\
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_augment && \\
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_train_cnn \\
        --input data/training_phase_u/manual_labels_aug.npz \\
        --init-model models/cnn_phase_u_v3.pt \\
        --out-model models/cnn_phase_u_v4.pt"
```

## 主要パラメータ (本流)

```python
# src/image_reader.py
CELL_SAMPLE_RATIO = 0.5
EMPTY_V_THRESHOLD = 40
RED_GREEN_DIFF_FOR_RED = 80  # H 11-18 範囲で BGR R-G 差で黄と区別

# src/hybrid_classifier.py
DEFAULT_CNN_OVERRIDE_PROB = 0.75  # 高確信度 CNN 採用閾値

# src/match_state.py
IN_MATCH_V_MAX = 150.0  # 盤面 V 平均 < 150 = 試合中
```

## 次に着手すべき作業

1. **HybridClassifier を本番 ImageReader に組み込み** (use_match_state=True 連携)
2. **バッチ 3 + バッチ 5 でデータ倍増** → CNN v4 訓練
3. **ROI 自動キャリブの再設計** (テンプレートマッチ or 格子検出)
4. **配信オーバーレイ統合** (stream_overlay 検証)
