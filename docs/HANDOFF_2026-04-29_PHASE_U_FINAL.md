# セッション引継ぎ 2026-04-29 (Phase U 最終、コンソール clear 直前)

## 1 行サマリ

Phase U 完成形: **CNN v6 本流採用、Holdout 99.45% / 全データ 98.34%**。
ユーザ提案「試合別 BG FP」が効果実証 (uncertain 30→22 件、-27%)。
HybridClassifier + use_match_state + use_ui_mask + 試合別 BG FP の構成で動作。

## 採用モデル

**`models/cnn_phase_u_v6.pt`** (本流):
- 訓練データ: 4771 件 (バッチ 1+2+3+4 + uncertain v01) → x20 augment 95420 件
- 25 epoch、lr 0.0004、cnn_phase_u_v5.pt 初期化
- Final loss 0.0204
- Holdout 99.45% (新規データ汎化に有利)

## 認識パイプライン構成

```
Frame 1080p
  ↓
ImageReader (use_match_state=True, use_ui_mask=True)
  classifier=HybridClassifier(cnn=cnn_phase_u_v6, hsv, ui_mask)
  set_background_fingerprints(fp1, fp2)  ← 試合別、毎試合更新
  ↓
read_both_boards
  ├ 試合状態判定 (試合中以外 = 全 EMPTY)
  ├ 各セル:
  │   ├ BG 距離 < 28 → EMPTY 確定 (キャラ動き対応)
  │   ├ HybridClassifier.classify(patch)
  │   │   ├ UI Mask 検出 → EMPTY (X 印など)
  │   │   ├ CNN.predict_proba → max prob >= 0.75 → CNN 採用
  │   │   └ それ以外 → HSV 判定
  │   └ result
  ├ clear_floating_above_gap (浮遊削除)
  └ _infer_hidden_rows (隠し段推論)
  ↓
両盤面 (12×6 + 隠し段 1 行)
```

## 主要モジュール (本流)

| モジュール | 役割 |
|---|---|
| `src/image_reader.py` | 中核。HSV 範囲 + use_match_state + use_ui_mask 統合 |
| `src/hybrid_classifier.py` | HSV+CNN+UIMask 統合分類器 |
| `src/board_recognition_pipeline.py` | 時系列レイヤー (本番動画再生用、離散時刻評価では効果なし) |
| `src/adaptive_background.py` | 継続的背景学習 (キャラ動き追随) |
| `src/console_init.py` | UTF-8 化 + Windows パス変換 |
| 流用 | animation_filter, stateful_board_tracker, temporal_smoother, next_detector, chain, ui_mask, match_state, physics_sanity, background_fingerprint |

## 主要 HSV パラメータ

```python
COLOR_RED:    H 0-18 + 166-180, S>=110, V>=100  (H 11-18 で BGR R-G>=80 黄区別)
COLOR_BLUE:   H 100-130, S>=100, V>=80
COLOR_GREEN:  H 50-85,  S>=100, V>=80
COLOR_YELLOW: H 14-38,  S>=30,  V>=180  (低彩度クリーム色対応)
COLOR_PURPLE: H 130-165, S>=80, V>=80
CELL_SAMPLE_RATIO: 0.5  (中央 50% median)
RED_GREEN_DIFF_FOR_RED: 80
判定順序: 色閾値 → OJAMA
```

## データセット

- `data/training_phase_u/manual_labels.npz` - 4771 件 (元)
- `data/training_phase_u/manual_labels_aug20.npz` - 95420 件 (x20)
- `data/verify/phase_u_batch1` - m1, m2, m3, m6, m8, m9, m10, m11, m13 (バッチ 1、450 件レビュー済)
- `data/verify/phase_u_batch2` - m4, m5, m7, m15-25 (500 件レビュー済)
- `data/verify/phase_u_batch3` - m18, m21, m22, m24, m26-32 (500 件 **未レビュー**)
- `data/verify/phase_u_batch4` - m31, m33-46 (500 件レビュー済 + 55 件修正)
- `data/verify/phase_u_uncertain_v01` - 30 件 (低確信度、19 件修正)
- `data/verify/phase_u_uncertain_v01_bg` - 22 件 (BG FP 込みで除外、9 件修正)

## ユーザ提案の効果実証

| 提案 | 実装 | 効果 |
|---|---|---|
| 試合別 BG FP | `phase_u_extract_uncertain.py --bg-fp-time` | uncertain 30 → 22 件 (-27%) |
| 大文字認識色シート | `phase_u_extract_samples.py` | レビュー労力大幅削減 |
| 1000 件ラベル意向 | バッチ 1+2+4 で 1500 件達成 | CNN 訓練データ基盤 |

## 主要スクリプト

| スクリプト | 用途 |
|---|---|
| `phase_u_extract_samples.py` | ラベル候補シート生成 (大文字) |
| `phase_u_extract_uncertain.py` | 確信度低セル抽出 (BG FP 対応) |
| `phase_u_batch_extract.sh`〜`batch4_extract.sh` | バッチ自動生成 |
| `phase_u_apply_batch4_labels.py` | ラベルを csv 反映 |
| `phase_u_build_dataset.py` | csv → npz データセット |
| `phase_u_augment.py` | x N 倍 augmentation |
| `phase_u_train_cnn.py` | CNN fine-tune |
| `phase_u_eval_classifiers.py` | HSV/CNN/Hybrid 比較 |
| `phase_u_eval_pipeline.py` | Pipeline 連続フレーム評価 |
| `phase_u_render.py` | 認識結果オーバーレイ動画 (CNN v6 デフォルト) |

## 残課題 (再開時の優先順)

| 課題 | 優先度 | 状態 |
|---|---|---|
| バッチ 3 ユーザレビュー (m18-m32 500 件) | 高 | ユーザ作業待ち |
| 試合別 BG FP の本流統合 | 中 | uncertain で実証済、ImageReader.set_background_fingerprints で動く |
| 動画レンダ本番統合 | 中 | phase_u_render.py 完成、CNN v6 デフォルト |
| 720p 動画対応 (動画別 ROI 設定) | 低 | video_02/03 で ROI ズレ |
| ROI 自動キャリブ | 低 | Hough 検出失敗、要再設計 |

## 再開コマンド

```bash
# 1. 引継ぎ確認
cat /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/docs/HANDOFF_2026-04-29_PHASE_U_FINAL.md

# 2. 全テスト走行 (約 3 分)
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m pytest tests/ -q --ignore=tests/test_video_processor.py"

# 3. 最新 CNN v6 評価
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_eval_classifiers"

# 4. 動画レンダ (1 試合のみテスト)
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_render \\
    data/verify/review_videos/clip_v01_m34.mp4 \\
    data/verify/review_videos/phase_u_v01_m34.mp4 \\
    --interval 0.2 --max-seconds 15"

# 5. バッチ 3 シート確認
ls "C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\verify\phase_u_batch3\"
```

## 設計判断

1. Phase T v2 (融合判定 + 擬似 self-training) は archive 化、Phase U で再構築
2. CELL_SAMPLE_RATIO 1.0 → 0.5 で median のシフト問題解消
3. 赤 H 拡張 (10→18) + BGR R-G 差 80 で黄と区別
4. 黄 H 14-38 + S>=30 + V>=180 で低彩度クリーム色対応
5. UI Mask に X 印 4 種類追加 (x_mark_video01_a/b/c)
6. CNN v6 が最終本流、x20 augmentation で holdout 99.45%
7. 時系列 Pipeline は離散時刻評価で逆効果 → 本番動画再生用に保留
8. 試合別 BG FP は uncertain 抽出で効果実証、本流統合の余地あり

## 次セッションへの推奨着手

1. **バッチ 3 ユーザレビュー** (m18-m32、500 件) → CNN v7 訓練 (3000 件規模)
2. **試合別 BG FP の本流統合** (毎試合開始時に capture_pair_robust → ImageReader へ)
3. **動画レンダ動作確認** (phase_u_render.py で実動作)
4. 720p 動画対応 (動画別 ROI 設定 JSON 作成)
