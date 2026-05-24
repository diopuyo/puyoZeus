# セッション引継ぎ 2026-04-28 (夜間 self-training 完了)

## 1 行サマリ

Phase T v2 全 (N→J→M→K→L) を統合した v5 動画でユーザレビュー後、夜間に擬似ラベル + self-training を 5 ラウンド実行 → cnn_pseudo_v5.pt が完成 (loss 0.049)。CNN 統合動画 v6/v7 とラベル候補シート v7 をユーザレビュー待ち。

## 達成済み (今夜)

### 1. T-v2-N → J → M → K → L 全統合 (前半)

| サイクル | 機能 | ファイル |
|---|---|---|
| T-v2-N | キャラ背景プロファイル (複数フレーム median) | `src/background_fingerprint.py` (capture_robust) |
| T-v2-J | StatefulBoardTracker (色→色直接遷移棄却) | `src/stateful_board_tracker.py` |
| T-v2-M | PhysicsSanityChecker (浮遊違反 → EMPTY 補正) | `src/physics_sanity.py` |
| T-v2-K | NextDetector → 許容色集合 (新規 EMPTY→色 棄却) | `stateful_board_tracker.py` 拡張 |
| T-v2-L | ChainSimulator で連鎖整合 | render に統合 |

### 2. 擬似ラベル self-training (5 ラウンド)

確実な正解として採用:
- 試合開始 0.4-1.4s → 全 EMPTY (約 1700 サンプル × 動画)
- presence < 0.20 → EMPTY (背景に近い)
- presence > 0.65 + HSV 有色 → HSV 色
- presence > 0.65 + CNN 確率 ≥ 0.85 → CNN 色 (Round 3 以降)

ラウンド毎の loss 推移:

| Round | データ | epoch | lr | final loss | 備考 |
|---|---|---|---|---|---|
| 1 | 7996 sample | 30 | 0.003 | **0.192** | 基本擬似ラベル |
| 2 | 29700 (augmented x5) | 50 | 0.002 | **0.099** | 元 + 拡張 |
| 3 | 31547 (CNN v2 self-train) | 30 | 0.001 | **0.070** | self-training 1 |
| 4 | (ログ要確認) | 25 | 0.0008 | (ログ要確認) | self-training 2 |
| 5 | 45823 | 20 | 0.0005 | **0.049** | self-training 3 |

`models/cnn_pseudo_v1.pt` 〜 `models/cnn_pseudo_v5.pt` (各 26K)。

### 3. 動画統合 + 視覚レビュー素材

| 動画 | 内容 | パス |
|---|---|---|
| `field_review_v0X_v5.mp4` | T-v2-N/J/M/K/L 全統合 + HSV のみ | data/verify/review_videos/ |
| `field_review_v0X_v6.mp4` | 同上 + CNN cnn_pseudo_v3.pt | 同上 |
| **`field_review_v0X_v7.mp4`** | **同上 + CNN cnn_pseudo_v5.pt (最新)** | 同上 |

ラベル候補シート (false positive 候補のみ、各 30 件以下):
- `data/verify/review_samples_v02_v7/sheet.png` (18 件)
- `data/verify/review_samples_v01_v7/sheet.png` (2 件)
- `data/verify/review_samples_v02_2p/sheet.png` (40 件、include-empty)

## ユーザレビュー待ち

朝起きた時に確認いただきたい:

1. **`field_review_v02_m1_v7.mp4`** と **`field_review_v01_m34_v7.mp4`**: 最終 CNN 統合動画
2. **`review_samples_v01_v7/sheet.png`** (2 件) と **`review_samples_v02_v7/sheet.png`** (18 件): ラベル付け対象

判定により次のサイクル決定:
- **改善大、99% 達成** → Phase T v2 完了、別のフィーチャー追加へ
- **改善あるが残る** → 残った誤検出セルから手動ラベル → Round 6 以降
- **悪化** → cnn_pseudo_v5 をロールバックして v3 へ

## 重要パラメータ (現在の融合判定)

```python
# cell_evidence_fusion.py
DEFAULT_W_HSV = 0.25
DEFAULT_W_BG = 0.25
DEFAULT_W_CNN = 0.20
DEFAULT_W_SHAPE = 0.20
DEFAULT_W_EYE = 0.10
DEFAULT_PRESENCE_THRESHOLD = 0.40
HSV_VOTE_WEIGHT = 0.55
CNN_PROB_THRESHOLD = 0.25
```

## 主要新規モジュール (T-v2)

- `src/cell_shape_features.py` (T-v2-2) — 円形度/Hough/エッジ/彩度
- `src/cell_evidence_fusion.py` (T-v2-3/8) — 重み付き融合 + 色多数決
- `src/shake_detector.py` (T-v2-7) — phaseCorrelate 振動検出
- `src/adaptive_background.py` (T-v2-C) — 継続的背景学習

## render の主要オプション (現在の最強構成)

```bash
PYTHONPATH=. ./venv/bin/python -m scripts.render_field_review_video \
  data/verify/review_videos/clip_v02_m1.mp4 \
  data/verify/review_videos/field_review_v02_m1_vN.mp4 \
  --interval 0.2 \
  --bg-fp-time 1.0 \
  --anim-filter \
  --temporal-smooth 7 \
  --evidence-fusion \
  --shake-filter \
  --adaptive-bg \
  --robust-bg \
  --stateful \
  --physics-sanity \
  --next-aware \
  --chain-predict \
  --cnn-model models/cnn_pseudo_v5.pt
```

## テスト状況

- 全テストパス (新規追加 cell_shape, cell_evidence_fusion, shake_detector, adaptive_background)
- `python -m pytest tests/ -q` で確認可能 (約 3 分)

## 次に着手すべき作業候補

1. **手動ラベルでの fine-tune**: ユーザレビュー結果のラベルで cnn_pseudo_v6.pt を訓練
2. **より長いデータ収集**: 動画全体 (300+ 秒) からのラベル抽出
3. **ROI 動的キャリブレーション** (T-v2-O): 試合開始時の盤面外枠検出 → 各動画ごとに自動補正
4. **Phase T v2 完了 + Phase U 着手**: 評価指標再学習・連鎖シミュ精度向上

## ログ

詳細な訓練履歴は `data/auto_train_log.txt` に追記済 (Round 1-5 全工程)。
