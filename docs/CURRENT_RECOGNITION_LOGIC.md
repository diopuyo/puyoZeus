# 現状フィールド認識ロジック (2026-04-28 04:30)

## 結論: v7 (CNN cnn_pseudo_v5.pt 統合) で悪化

擬似ラベル self-training (Round 1〜5) は **CNN が擬似ラベル自身に過適合** したため、本物の精度では悪化。最も信頼できる構成は **v5 (HSV のみ + 全フィルタ)**。

---

## 全体パイプライン (現状)

```
動画フレーム (1920×1080)
    ↓
[1] AnimationFilter      連鎖閃光 → 直前盤面保持
[2] ShakeDetector        振動検出 → ROI 補正 (dx, dy) or 直前保持
[3] AdaptiveBackgroundFingerprint.update()   背景 FP を毎フレーム移動平均で更新
    ↓
ImageReader.read_both_boards(frame, p1_offset, p2_offset)
    ↓ [P1, P2 それぞれ]
read_board(frame, region)
    ↓ [12 行 × 6 列の各セル]
    ├─ patch = frame[セル矩形] を切り出し
    └─ _classify_with_fusion(patch, ...)
            ↓
        CellEvidence を構築:
            ├─ HSV 色判定        (ColorClassifier or CnnPatchClassifier)
            ├─ 背景距離          (CellFingerprint.distance_to)
            ├─ 形状特徴          (CellShapeFeatures: 円形度/Hough/エッジ/彩度)
            ├─ 目検出スコア      (PuyoEyeDetector)
            └─ CNN 確率 (オプション、predict_proba)
            ↓
        compute_presence_score → 重み付き合計 0-1
            ↓
        presence < 0.40 → COLOR_EMPTY
            ↓ それ以外
        色多数決 (HSV 票 0.55 + CNN 確率 ≥ 0.25 加算)
            ↓
        票最大色 / OJAMA は HSV 即決 / 票なし→ EMPTY
    ↓ [全セル後]
    ├─ clear_floating_above_gap   浮遊ぷよ削除
    └─ _infer_hidden_rows         隠し段の重力推論
    ↓
[4] TemporalSmoother      過去 7 フレーム多数決
[5] StatefulBoardTracker  物理ルール違反棄却 + chain_event 検知
                         ├─ 色 → 色 直接遷移 → 棄却
                         ├─ 孤立 色 → empty → 棄却 (chain_event=False 時)
                         ├─ 色 → OJAMA → 棄却
                         └─ NextDetector の許容色集合と照合 (--next-aware)
[6] PhysicsSanityChecker  浮遊違反 → 該当セル EMPTY 補正 (--physics-sanity)
[7] ChainSimulator       連鎖発火検知 → ChainSimulator で次盤面予測 →
                         観測との差 ≤3 セル → 予測を採用 (--chain-predict)
    ↓
最終 Board (12×6 + 隠し段 1 行)
```

## 各モジュールの役割と主要パラメータ

### ROI 設定 (1920×1080 ハードコード)

| プレイヤー | x | y | width | height |
|---|---|---|---|---|
| 1P | 282 | 160 | 384 | 720 |
| 2P | 1258 | 160 | 384 | 720 |

セル単位: 64 × 60 px、12 行 × 6 列。

### HSV 色閾値 (`src/image_reader.py::DEFAULT_COLOR_RANGES`)

| 色 | H | S | V |
|---|---|---|---|
| RED | 0-10, 166-180 | ≥120 | ≥100 |
| BLUE | 100-130 | ≥100 | ≥80 |
| GREEN | 50-85 | ≥100 | ≥80 |
| YELLOW | 20-38 | ≥100 | ≥100 |
| PURPLE | 130-165 | ≥80 | ≥80 |
| EMPTY | (S<60 + V<40) または HSV 範囲不一致 |
| OJAMA | S<60 + V≥100 |

### 形状特徴 (`src/cell_shape_features.py`)

各 0-1 正規化、`presence_score = 0.30·circularity + 0.25·circle_score + 0.20·edge_density + 0.25·sat_center`

| 特徴 | 計算 |
|---|---|
| circularity | Otsu 二値化 → 最大輪郭 4π·area/perimeter² (面積セルの 4-85% 限定) |
| circle_score | HoughCircles で中央 ±30% 以内の円検出強度 |
| edge_density | Canny エッジ比率 / 0.30 で正規化 |
| sat_center | 中央 50% S 平均 - 周辺 S 平均 |

### 融合判定 (`src/cell_evidence_fusion.py`)

**現在のデフォルト** (T-v2-D 改後):

```python
DEFAULT_W_HSV = 0.25   # HSV 色付き判定の重み
DEFAULT_W_BG = 0.25    # 背景距離 (大きいほどぷよあり)
DEFAULT_W_CNN = 0.20   # CNN 確率最大値 (現状ColorClassifierでは 0)
DEFAULT_W_SHAPE = 0.20 # 形状特徴 presence
DEFAULT_W_EYE = 0.10   # 目検出スコア

DEFAULT_PRESENCE_THRESHOLD = 0.40  # この未満で EMPTY
HSV_VOTE_WEIGHT = 0.55             # 色決定時の HSV 基本票
CNN_PROB_THRESHOLD = 0.25          # CNN 票の最低確率
```

色決定:
1. presence < 0.40 → EMPTY
2. HSV=OJAMA → OJAMA 即決
3. HSV 有色なら 0.55 票、CNN 各色 prob ≥ 0.25 を加算
4. 票最大色を採用、票なし → EMPTY (UNKNOWN にしない)

### 振動検出 (`src/shake_detector.py`)

cv2.phaseCorrelate で 2 フレーム間サブピクセルシフト推定:
- magnitude ≥ 0.8 px かつ response ≥ 0.10 → 振動中
- (dx, dy) を ROI 補正に使用、±5 px 超は直前保持

### 継続背景学習 (`src/adaptive_background.py`)

毎フレームで各セル HSV を取得し、既存背景との距離 < 35 のセルだけ移動平均更新 (lr=0.05)。連鎖中・キャラ動きに追随。

### StatefulBoardTracker (`src/stateful_board_tracker.py`)

直前確定盤面と観測の差分を遷移ルールで判定:
- EMPTY → 色: 受理 (新規落下)、ただし `expected_new_colors` 集合になければ棄却 (T-v2-K、--next-aware)
- 色 → EMPTY: chain_event (4+ 同時消滅) 時のみ受理
- 色 → 色 (異色): chain_event 時のみ受理
- 色 → OJAMA: 常に棄却

### PhysicsSanityChecker

各フレームの盤面に対し:
- AIRBORNE: 直下が空のぷよ → 浮遊違反
- UNRESOLVED_CHAIN: 4+ 同色連結が消えていない

render では AIRBORNE のみ自動補正 (該当セル → EMPTY)。

### ChainSimulator 連鎖整合 (T-v2-L)

tracker.last_stats.chain_event=True を検知 → 直前盤面に ChainSimulator.simulate を適用 → 予測盤面と現観測の差を計算 → 差 ≤ 3 セル → 予測を観測に上書き。

### TemporalSmoother (Phase T サイクル A)

各セルの過去 N=7 フレームの色履歴から多数決を取って確定色を返す。

## 動画モード (現状の最強構成と各動画)

| 動画 | フィルタ構成 | 状況 |
|---|---|---|
| `field_review_v0X.mp4` (旧) | サイクル 1+2+5 | 古い |
| `field_review_v0X_temporal.mp4` | + サイクル A | 古い |
| `field_review_v0X_eyes.mp4` | + サイクル 8 (require_eyes_for_color=True) | 過剰に empty |
| `field_review_v0X_fusion.mp4` | T-v2-3 融合判定 | OK |
| `field_review_v0X_fusion2.mp4` | + T-v2-7/8 振動・色多数決 | OK |
| `field_review_v0X_v3.mp4` | + adaptive_bg、UNKNOWN 残し | UNKNOWN 多発で問題 |
| `field_review_v0X_v4.mp4` | + UNKNOWN→EMPTY、閾値 0.40 | 改善 |
| **`field_review_v0X_v5.mp4`** | **+ T-v2-N/J/M/K/L 全統合 (HSV のみ)** | **ユーザ評価 ◎** |
| `field_review_v0X_v6.mp4` | + CNN cnn_pseudo_v3.pt | 要検証 |
| `field_review_v0X_v7.mp4` | + CNN cnn_pseudo_v5.pt | **悪化判定** |

## 推奨ロールバック

**推奨: v5 を最良基準とする** (CNN なし、全フィルタ ON)。

```bash
PYTHONPATH=. ./venv/bin/python -m scripts.render_field_review_video \
  data/verify/review_videos/clip_v02_m1.mp4 \
  data/verify/review_videos/field_review_v02_m1_v5_clean.mp4 \
  --interval 0.2 --bg-fp-time 1.0 --anim-filter --temporal-smooth 7 \
  --evidence-fusion --shake-filter --adaptive-bg --robust-bg \
  --stateful --physics-sanity --next-aware --chain-predict
  # ← --cnn-model は付けない
```

## 推奨される追加対応

### 1. CNN 再学習 (擬似ラベルではなく手動ラベル)

擬似ラベルベースの self-training は精度の自己強化バイアスを起こす。
**ユーザの手動ラベル** (10-30 サンプル) で fine-tune する方が信頼できる。

`data/verify/review_samples_v01_v7/labels.csv` を埋めて、再訓練に使う。

### 2. 既存 CNN cnn_global_best.pt (holdout 0.9266) を初期重みに使う

scripts/train_pseudo_cnn.py で `--init-model models/cnn_global_best.pt` を指定。
self-training の負スパイラルを脱する。

### 3. CNN を融合の補助に留める

CNN を「主役 (classifier)」ではなく「重み 0.10 程度の補助証拠」として融合に組み込む。
HSV を主役に保ち、CNN は微妙なケースの補助に。

## v5 動画の構成図 (再掲)

```
[ 動画ループ ]
   ↓
1080p リサイズ
   ↓
adaptive_bg.update()                       ← T-v2-C
   ↓
解析間隔チェック (0.2s)
   ↓
AnimationFilter 判定                       ← サイクル 2
   ↓ (連鎖中ならスキップ)
ShakeDetector 判定                         ← T-v2-7
   ↓ (大振動ならスキップ、それ以下は ROI 補正に使う)
ImageReader.read_both_boards(p1_offset, p2_offset)
   │
   ├ HSV 判定 (ColorClassifier)           ← 旧来
   ├ 背景距離 (Adaptive FP)                ← サイクル 1 + T-v2-C
   ├ 形状特徴 (CellShapeFeatures)          ← T-v2-2
   ├ 目検出 (PuyoEyeDetector)              ← サイクル 8
   └ 融合判定 (fuse_color)                 ← T-v2-3/8
   ↓
clear_floating_above_gap                  ← サイクル 5
   ↓
_infer_hidden_rows
   ↓
TemporalSmoother (window=7)               ← サイクル A
   ↓
StatefulBoardTracker (next_aware=True)    ← T-v2-J + T-v2-K
   ↓ (色→色直接、孤立 c→e、ネクスト外色を棄却)
PhysicsSanityChecker (浮遊→EMPTY)         ← T-v2-M
   ↓
ChainSimulator 整合 (差≤3 で予測採用)     ← T-v2-L
   ↓
render
```

これが現在の v5 構成 (CNN なし、全推論 ON) です。
