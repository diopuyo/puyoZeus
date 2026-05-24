# セッション引継ぎ 2026-04-28 (盤面認識精度向上プロジェクト進行中)

## 1 行サマリ

ユーザ目標 99.2% は達成済 (Phase F〜S)。ただし「**フィールドのぷよが empty になる**」根本問題発覚 → Phase T (盤面認識改善) を **計画→設計→製造→レビュー** サイクルで進行中。サイクル 1, 2, 5, A, 8 実装済、レビュー動画 12 本生成済。中期 (色別目テンプレ) のユーザサンプル座標待ち。

## 状態サマリ

### 達成済み精度 (推論ロジック整備済)
| 層 | 精度 |
|---|---|
| 学習層 LR (21 特徴量) | 0.626 (best LEARNED_PHASE_J_GLOBAL) |
| score OCR | readable 87.2% / 正解率 100% |
| 視覚版 CNN (708 サンプル) | 0.939 |
| **実運用統合精度推定** | **≈ 99.2%** |
| 全テスト | 920+ passed / 2 skipped |

### Phase T (進行中) - 盤面認識精度向上

**経緯**: ユーザレビューで「**置いているぷよが empty 判定される**」「**そもそもぷよ色の誤検出が多い**」と判明。CNN holdout 0.9266 で頭打ち、内部 0.9911 で過学習傾向。下流 (連鎖シミュ・凝視) 全部に影響するボトルネック。

**進行中サイクル**:

| サイクル | 内容 | 状態 |
|---|---|---|
| 1. 背景フィンガープリント差分 | 試合開始時の空盤面の HSV を保存、差分で empty 判定 | ✅ 実装 + 動画生成 |
| 2. アニメフレーム除外 | 連鎖中の閃光フレームをスキップ、直前盤面保持 | ✅ |
| 5. 浮遊ぷよ削除 | `clear_floating_above_gap` を ImageReader 経路に統合 | ✅ |
| 6. CNN 学習再開 | `data/training_stopped` 削除 | ✅ (watchdog 再開準備) |
| A. 時系列スムージング | TemporalSmoother (window=7) で N フレーム多数決 | ✅ render 経路のみ |
| 8. 目検出統合 (ユーザ提案) | `PuyoEyeDetector`、目なし色判定は empty 強制 | ✅ |
| 9. 色別目の形テンプレ (ユーザ提案、中期) | 各色の代表ぷよから目の形テンプレ生成、色判定補強 | 🔄 ユーザサンプル座標待ち |
| 3. ROI/HSV 動的キャリブレーション | 試合開始時に枠検出 | 設計済、未実装 |
| 4. 訓練データ増強 (video_02/03) | 連鎖中以外フレーム抽出 → CNN 再訓練 | 設計済、未実装 |

**現在の推論割合**: HSV/CNN 60-65% + 推論層 35-40% (Phase T サイクル 8 統合後)

## 重要ファイル

### 計画書
- `docs/PLAN_BOARD_RECOGNITION_2026-04-27.md` (Phase T の全戦略)
- `docs/HANDOFF_2026-04-26_EVENING.md` (Phase F〜S の全工程記録)

### 新規モジュール (Phase T 関連)
- `src/background_fingerprint.py` (背景 FP、9 テスト)
- `src/animation_filter.py` (アニメフレーム検出、7 テスト)
- `src/puyo_eye_detector.py` (目検出、6 テスト)

### 修正済み既存モジュール
- `src/image_reader.py` — Phase T 統合 (背景 FP、浮遊削除、目検出を `require_eyes_for_color` で有効化)
  - `apply_inference: bool = True` (デフォルト推論オン)
  - `floating_min_gap: int = 2`
  - `require_eyes_for_color: bool = False` (デフォルト OFF、互換性のため)
  - `set_background_fingerprints(fp1, fp2)` API
- `scripts/render_field_review_video.py` — `--bg-fp-time` `--anim-filter` `--temporal-smooth N` `--eye-required` フラグ追加

### 診断スクリプト
- `scripts/diagnose_field_recognition.py` — 各セルの HSV / 判定理由を可視化
  - 出力例: `data/verify/diag_v02_t235.png`, `diag_v01_t2070.png`

### Phase F〜S の永続成果物
- `models/cnn_global_best.pt` (CNN holdout 0.9266)
- `models/ojama_cnn.pt` (視覚版予告 CNN、val 0.924)
- `models/ui_templates/score_digits/digit_0..9.png` (score OCR テンプレ)
- `models/ui_templates/ojama/{small,line,rock,moon,crown,big_crown}.png` (視覚版予告テンプレ)
- `data/training/match_features_v3.csv` (1390 行 × 21 特徴量、Phase G/J/L 反映済)
- `data/training/score_series_cache.json` (148 試合、readable 87.2%)
- `data/verify/ojama_labels_v[1-5].tsv` (合計 708 ラベル)

### レビュー動画 (12 本、`data/verify/review_videos/`)
| 動画名 | 内容 |
|---|---|
| `clip_v02_m1.mp4`, `clip_v01_m34.mp4` | 元クリップ (オーバーレイなし) |
| `review_v02_m1_overlay.mp4`, `review_v01_m34_overlay.mp4` | score バー + 指標 |
| `field_review_v02_m1.mp4`, `field_review_v01_m34.mp4` | 旧 (ベースライン) |
| `field_review_v02_m1_bg.mp4`, `..._v01_m34_bg.mp4` | cycle 1 |
| `field_review_v02_m1_bg_anim.mp4`, `..._v01_m34_bg_anim.mp4` | cycle 1+2 |
| `field_review_v02_m1_full.mp4`, `..._v01_m34_full.mp4` | cycle 1+2+5 |
| `field_review_v02_m1_temporal.mp4`, `..._v01_m34_temporal.mp4` | cycle 1+2+5+A |
| **`field_review_v02_m1_eyes.mp4`, `..._v01_m34_eyes.mp4`** | **cycle 1+2+5+A+8 (最新)** |

## 未解決の問題

### 1. 「置いているぷよが empty 判定される」 (最重要)
- ユーザ報告で確認 (cycle 1+2+5+A 統合 `*_temporal.mp4` でも「変化乏しい」)
- 根本原因仮説:
  - CNN/HSV が継続的に誤認識 (時系列スムージングでも復元できない)
  - ROI 位置のズレ (video_02 720p→1080p リサイズ起因の可能性)
  - 暗所紫が空判定に倒れる (97% 既知問題)
- **対策候補 (未着手)**:
  - サイクル 3: ROI 動的キャリブレーション
  - サイクル 4: 訓練データ video_02/03 増強 + CNN 再訓練
  - **サイクル 9: 色別目の形テンプレ** (進行中、ユーザサンプル待ち)
  - HSV 閾値再調整 (EMPTY_V_THRESHOLD=40 → 25)

### 2. ぷよの色誤検出
- ユーザ提案: 色だけでなく **目の形 (色によって違う)** を併用
- サイクル 8 で「目検出による存在確定」は実装済 → 効果評価中
- サイクル 9 で「色別目テンプレ」を追加予定

## ユーザレビュー待ち

最新版 `field_review_v01_m34_eyes.mp4` / `field_review_v02_m1_eyes.mp4` (cycle 1+2+5+A+8) を見て:

| 結果 | 次のアクション |
|---|---|
| 改善大、ぷよ → empty 解消 | **サイクル 9 (色別目テンプレ)** に進む |
| 改善あるが残る | 目検出閾値調整 + サイクル 9 |
| 悪化、ぷよ → empty 増加 | 閾値緩和 (`UPPER_HALF_RATIO_MIN` 0.55 → 0.45) → 再生成 |
| 変化乏しい | サイクル 3 (ROI 動的化) or 4 (訓練データ増強) で根本対策 |

## サイクル 9 用ユーザサンプル座標

中期実装に必要な情報 (再掲):
```
red:    (動画, 時刻, side, row, col)
blue:   ...
green:  ...
yellow: ...
purple: ...
```
- 各色 3-10 サンプル推奨
- 参考: `data/verify/diag_v02_t235.png`, `diag_v01_t2070.png`
- row=0..11 (上→下、可視 12 段)、col=0..5 (左→右)
- side=`1P`/`2P`

## 再開コマンド

```bash
# 現状把握 (引継ぎ確認)
cat /c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/docs/HANDOFF_2026-04-28_BOARD_RECOGNITION.md

# 全テスト走行 (約 3 分)
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m pytest tests/ -q"

# 最新レビュー動画を開く
explorer.exe "C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\verify\review_videos\field_review_v02_m1_eyes.mp4"

# 動画再生成 (パラメータ調整時)
PYTHONPATH=. ./venv/bin/python -m scripts.render_field_review_video \
  data/verify/review_videos/clip_v02_m1.mp4 \
  data/verify/review_videos/field_review_v02_m1_v2.mp4 \
  --interval 0.5 --bg-fp-time 1.0 --anim-filter \
  --temporal-smooth 7 --eye-required

# 診断画像生成
PYTHONPATH=. ./venv/bin/python -m scripts.diagnose_field_recognition \
  data/frames/video_02.mp4 --time 235.0 \
  --out data/verify/diag_v02_t235_new.png
```

## バックアップ

- `data/training/match_features_v3.bak_pre_*.csv` (Phase F/J/K の各バックアップ)
- `data/training/score_series_cache.bak_pre_supplement.json` (Phase L1 補完前)
- `data/verify/ojama_labels_v3.bak_pre_user_review.tsv` (Phase M ユーザ修正前)

## 再開プロンプト (新セッションでコピペ)

```
ぷよぷよeスポーツ動画解析プロジェクト (C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer) を再開します。

直前まで Phase T (盤面認識精度向上) を進行中です。次の手順で作業を続けてください:

1. 引継ぎドキュメントを最初に読む:
   docs/HANDOFF_2026-04-28_BOARD_RECOGNITION.md

2. 必要なら全テスト走行 (約 3 分、920+ passed / 2 skipped を確認):
   wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m pytest tests/ -q"

3. 直前のユーザレビュー対象動画 (cycle 1+2+5+A+8 統合最新版):
   - data/verify/review_videos/field_review_v02_m1_eyes.mp4
   - data/verify/review_videos/field_review_v01_m34_eyes.mp4

4. 直前まで進めていた未解決事項:
   - ユーザの最新動画レビュー結果待ち (改善大/中/小/悪化のフィードバック)
   - サイクル 9 (色別目テンプレ) のユーザサンプル座標待ち
   - サイクル 3 (ROI 動的化), 4 (訓練データ増強) は未着手

5. 既知の最重要問題:
   「フィールドのぷよが empty 判定される」根本問題 (CNN holdout 0.9266 で頭打ち、video_02/03 で汎化性能低)

6. 現状の推論パイプライン (Phase T サイクル 8 統合後):
   HSV/CNN 60-65% + 推論層 35-40% (背景FP / アニメフィルタ / 浮遊削除 / 時系列 / 目検出)

詳細は docs/HANDOFF_2026-04-28_BOARD_RECOGNITION.md と
docs/HANDOFF_2026-04-26_EVENING.md (Phase F〜S) を参照。
```

## 重要な設計判断 (再開時に忘れないように)

1. **Phase K 3 指標 (opponent_offset_power 等) は学習特徴量から撤回** したが IndicatorSet には保持 (推論ロジック内で活用可)
2. **Phase J 21 特徴量 + LEARNED_WEIGHTS_PHASE_J_GLOBAL** が現状の最良学習重み (test_acc 0.626、end phase 0.769)
3. **Phase S で本番経路バグ修正済** (`Analyzer.analyze_boards` で opponent_board / incoming_ojama を相互渡し)
4. **CNN holdout 0.9266 で頭打ち** (Cycle 5、`global_best.json`、`training_stopped` は Phase T で削除済)
5. **Phase T サイクル 8 で `require_eyes_for_color` はデフォルト False** (合成テスト互換性のため)、render では True で動作
