# セッション引継ぎ 2026-04-28 (Phase U 最新、コンソール再起動向け)

## 1 行サマリ

Phase U で 1500-2000 件手動ラベル + CNN fine-tune で **97.94%** 達成。
時系列レイヤー (BoardRecognitionPipeline) は離散時刻評価では効果出ず、
本番動画再生用に保留。CNN v4 (x10 augment) 訓練中。

## 達成事項 (Phase U)

### 認識精度推移
| 段階 | 全 4749 件精度 | Holdout |
|---|---|---|
| HSV 単独 | 95.94% | - |
| CNN v1 (1500件) | - | 95.0% |
| CNN v2 (x5 augment) | 98.36% | 98.6% |
| CNN v3 (4749 件 + x5 augment) | **97.94%** | 98.5% |
| CNN v4 (x10 augment、4749 件 + バッチ 4 反映) | 98.02% | 99.1% |
| **CNN v5 (x20 augment、最新本流)** | **98.44%** | **99.38%** |
| Hybrid (Hybrid+UIMask) | 97.94% | - |
| Pipeline 全 ON | 96.00% (悪化) | - |

### 主要モジュール
- `src/image_reader.py` - 簡素化 + use_match_state / use_ui_mask 追加
- `src/hybrid_classifier.py` - HSV+CNN+UIMask 統合
- `src/board_recognition_pipeline.py` - 時系列レイヤー (本番動画用)
- `src/adaptive_background.py` - archive から復元
- 既存流用: `animation_filter`, `stateful_board_tracker`, `temporal_smoother`,
            `next_detector`, `chain`, `ui_mask`, `match_state`,
            `physics_sanity`, `background_fingerprint`

### HSV 範囲 (確定)
```python
COLOR_RED:    H 0-18, S>=110, V>=100  (H 11-18 は BGR R-G>=80 で黄と区別)
              + H 166-180, S>=110, V>=100
COLOR_BLUE:   H 100-130, S>=100, V>=80
COLOR_GREEN:  H 50-85,  S>=100, V>=80
COLOR_YELLOW: H 14-38,  S>=30,  V>=180  (低彩度クリーム色対応)
COLOR_PURPLE: H 130-165, S>=80, V>=80
CELL_SAMPLE_RATIO: 0.5  (中央 50% median)
判定順序: 色閾値 → OJAMA
RED_GREEN_DIFF_FOR_RED: 80
```

### データセット
- `data/training_phase_u/manual_labels.npz` - 4749 件 (元)
- `data/training_phase_u/manual_labels_aug.npz` - 23745 件 (x5)
- `data/training_phase_u/manual_labels_aug10.npz` - 47490 件 (x10、訓練中)
- バッチ 1, 2, 3, 4 = 各 500 件 ラベル候補シート (バッチ 1+2+4 レビュー済)
- バッチ 3 (m18-m32) は **未レビュー**

### 主要スクリプト
| スクリプト | 用途 |
|---|---|
| `phase_u_extract_samples.py` | ラベル候補シート生成 |
| `phase_u_batch_extract.sh` 〜 `phase_u_batch4_extract.sh` | バッチ自動生成 |
| `phase_u_apply_batch4_labels.py` | ユーザレビュー結果を csv 反映 |
| `phase_u_build_dataset.py` | csv → npz データセット構築 |
| `phase_u_augment.py` | データ拡張 (--multiplier x で N 倍) |
| `phase_u_train_cnn.py` | CNN fine-tune |
| `phase_u_eval_classifiers.py` | HSV/CNN/Hybrid 比較 |
| `phase_u_eval_pipeline.py` | Pipeline 評価 (連続フレーム) |

## 残課題

| 課題 | 優先度 | 状態 |
|---|---|---|
| CNN v4 (x10 augment) 訓練 | - | ✅ **完了 Holdout 99.1%** |
| バッチ 3 ユーザレビュー (500 件) | 中 | ユーザ作業待ち |
| Pipeline 本番動画評価 | 中 | 単純認識で止まれば不要 |
| ROI 自動キャリブ (720p 動画) | 低 | Hough 検出失敗、再設計要 |
| 動画レンダ (本番統合) | 低 | ユーザの「本番はまだ」指示で保留 |

## 既知の誤認パターン (Hybrid CNN v3)

```
1 → 0 (EM): 35件 — 赤判定セル (X 印・連鎖中赤エフェクト) を真値 EMPTY
3 → 0 (EM): 15件 — 緑判定セルを真値 EMPTY
2 → 0 (EM): 15件 — 青判定セルを真値 EMPTY
9 → 0 (EM): 13件 — おじゃま判定セルを真値 EMPTY
4 → 0 (EM):  8件 — 黄判定セルを真値 EMPTY
```

合計 86件 (97.94% の残 2%)。多くは **背景・エフェクト → 色判定された false positive**。

## 再開時の最初のコマンド

```bash
# 全テスト走行 (約 3 分)
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m pytest tests/ -q --ignore=tests/test_video_processor.py"

# 最新 CNN 評価
wsl -- bash -c "cd /mnt/c/.../puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_eval_classifiers"

# レビュー済シート確認
ls "C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\verify\phase_u_batch4\"
```

## 推奨次手順 (優先順)

1. **CNN v4 (x10 augment) 訓練完了確認** + 評価
2. **バッチ 3 ユーザレビュー** (m18-m32 の 500 件、6 行 × 5 文字 × 10 シート)
3. CNN v5 訓練 (3000 件規模)
4. 動画レンダ本番統合 (HybridClassifier + use_match_state=True + Pipeline)
5. ROI 自動キャリブ再設計 (720p 動画対応)

## 主要パラメータ

```python
# src/hybrid_classifier.py
DEFAULT_CNN_OVERRIDE_PROB = 0.75  # 高確信度 CNN 採用閾値

# src/board_recognition_pipeline.py
DEFAULT_TEMPORAL_WINDOW = 5  # TemporalSmoother のウィンドウサイズ

# Pipeline は本番動画再生用、ラベル評価では効果なし
```

## 設計判断履歴

1. Phase T v2 (融合判定 + 擬似 self-training) は archive 化
2. CELL_SAMPLE_RATIO 1.0 → 0.5 で median のシフト問題解消
3. 赤 H 拡張 (10→18) + BGR R-G で 黄と区別
4. 黄 H 14-38 + S>=30 + V>=180 で低彩度クリーム色対応
5. UI Mask に X 印 4 種類 (video01_a/b/c) 追加
6. Pipeline は離散時刻評価で逆効果、本番動画再生用に保留

## 文字化け対策

- `src/console_init.py` の `init_console()` を全スクリプト先頭で呼ぶ
- `to_windows_path()` で WSL/Mingw → C:\... フルパス変換
- VS Code settings.json / PowerShell プロファイル / WSL ~/.bashrc で UTF-8 設定済
