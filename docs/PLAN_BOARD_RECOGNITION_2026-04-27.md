# 盤面読取精度向上プロジェクト 計画書

2026-04-27 / Phase T (Board Recognition Improvement)

## 1. 現状診断

### 精度
- CNN holdout 0.9266 / 内部 0.9911 (Cycle 5 で頭打ち、2026-04-25)
- 7-8% の誤認識 → 6×12=72 セル/フレーム × 138 試合 × 10 phase = 約 10 万セル中 1 万近く誤認識
- ユーザ目視レビュー結果: 「フィールドのぷよは精度が低い」「推論もうまくいっていない」

### ボトルネック
1. **背景・エフェクト混入**: 連鎖アニメ・閃光・キャラ背景がぷよ判定を阻害
2. **CNN 内部 0.99 vs holdout 0.93** = 過学習 (動画間プレイヤー戦術差・UI 差で汎化不足)
3. **video_01 中心の訓練データ**: video_02 (720p) / video_03 で性能低下
4. **暗所紫が「空」誤認**: 1P 青背景 / 2P 赤背景で暗所紫の彩度低下
5. **ROI 固定**: 動画ごとに微妙な位置ズレに対応できていない

## 2. 目標

| 指標 | 現状 | 短期目標 | 中期目標 |
|---|---|---|---|
| holdout 精度 | 0.9266 | 0.95+ | 0.97+ |
| ユーザ目視 OK 率 | 低 | 中 | 高 |
| 誤認識セル/フレーム | 5-6 | 2-3 | 1 以下 |

## 3. 改善戦略 (5 つ)

### A. 背景フィンガープリント差分 (ユーザ提案、最優先) ⭐
- 試合開始 0.5-2.0 秒の「空盤面」フレームから各セルの **背景 HSV 平均値** を取得
- 各動画/各試合開始時にキャッシュ
- 推論時に「背景との HSV 距離」を計算
  - 距離 < 閾値 → 空 (背景のまま、CNN 不要)
  - 距離 ≥ 閾値 → 「ぷよあり」と仮判定 → CNN で色を判定
- 期待効果: 背景・キャラ画像の影響を完全打消、誤認識 -50%

### B. 連鎖中・エフェクト中フレーム除外 (Phase S レビュー指摘)
- V 標準偏差が異常に高い (閃光) → 該当フレームの該当セルを「直前盤面のまま」に
- HSV 飽和度急変も同様
- ChainSimulator の連鎖区間中はサンプリング除外
- 期待効果: 学習・推論両方でノイズ削減、+1-2%

### C. ROI/HSV 動的キャリブレーション
- 試合開始時に盤面の枠 (灰色・黒の境界) を検出
- 動画ごとの calibration.json を自動生成
- 各動画で HSV 範囲も再キャリブ
- 期待効果: 動画/UI 差での汎化性能 +2-4%

### D. 訓練データ video_02/03 増強
- 現状 pl1×35 + pl3×36 + pl4×30 = 101 npz、video_01 中心
- video_02 / video_03 のフレームから連鎖中以外を抽出 → 訓練データに追加
- 期待効果: 動画 hold-out 性能 +3-5%

### E. CNN モデル深化 (長期)
- 現状 Residual+Gated → ResNet 風 (より深い)
- 訓練データ拡充後に効果大

## 4. 実装サイクル (計画→設計→製造→レビュー)

### サイクル 1: A (背景フィンガープリント) — 優先実装
- 計画 (済): 上記
- 設計: `src/background_fingerprint.py` 新規モジュール
- 製造: 試合開始時のキャプチャ + 推論時の差分判定
- レビュー: field_review 動画再生成 → ユーザ確認 (2 時間後 or 完了時)

### サイクル 2: B (連鎖中除外)
- 既存 ChainDetector の状態を活用
- 連鎖中フラグ取得 API 追加

### サイクル 3: C (動的キャリブレーション)
- 試合開始時に盤面枠を Hough 変換 / エッジ検出
- HSV 範囲は背景 FP から自動算出

### サイクル 4: D (訓練データ増強)
- video_02 / video_03 からサンプリング
- ラベル付け (CNN 自動 + 人手レビュー)
- 再訓練

## 5. レビューポイント

| 段階 | レビュー | 頻度 |
|---|---|---|
| サイクル 1 完了 | A の効果を field_review で目視 | 即時 |
| サイクル 2 完了 | エフェクト除去の効果 | サイクル 1 から 30 分後 |
| サイクル 3 完了 | キャリブレーション結果 | 1-2 時間後 |
| サイクル 4 完了 | 訓練データ増加後の精度 | 数時間後 (CNN 再訓練込み) |

人レビューは **30 分〜2 時間に 1 回** を目安、不要なら省略。

## 6. 学習資料 (Web 先行研究)

[HSV Color Range Thresholding - LearnCodeByGaming](https://learncodebygaming.com/blog/hsv-color-range-thresholding) — HSV 値域でのオブジェクト分離

[Robust background subtraction in HSV color space](https://www.researchgate.net/publication/228957857) — HSV 空間での背景差分

[OpenCV background subtraction tutorial](https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html) — MoG2 等の伝統的手法

知見:
- HSV の H (色相) は局所変動と明度変化に強い
- 低彩度時は H が不安定なので S と V も併用が必須
- MoG2 は HSV 直接利用に向かない (低彩度ピクセルで H 散らばり)
- **シンプルなピクセル単位差分** が小領域 (= 各セル) の判定では十分

実装方針:
- セル中央 16×16 patch の (H, S, V) 平均値を背景 FP として保存
- 推論時: 現フレームの (H, S, V) と FP の Manhattan 距離 / 重み付き距離
- 閾値以上で「ぷよあり」と仮判定

## 7. 成功条件

- field_review_v02_m1.mp4 / field_review_v01_m34.mp4 をユーザが目視して **「フィールドのぷよ精度が大幅改善」** と判定
- holdout 精度 0.93 → 0.95+
- 連鎖中フレームでの誤検出が消滅

## 8. 進捗

### サイクル 1: 背景 FP 差分 (実装完了)
- ✅ `src/background_fingerprint.py` 新規 (CellFingerprint、BackgroundFingerprint、9 テスト pass)
- ✅ `src/image_reader.py` 統合 (set_background_fingerprints API、空セル先判定経路)
- ✅ `scripts/render_field_review_video.py` に `--bg-fp-time` オプション追加
- ✅ 38 テスト pass (image_reader + background_fingerprint)
- 🔄 動画再生成中 (`b18fmik4f` v02 m1、`bewerm69p` v01 m34)
- 🔄 ユーザレビュー待ち

### サイクル 2: 連鎖中フレーム除外 (設計中)
- VideoChainTracker に `is_in_chain_animation(t_sec)` API 追加
- ImageReader に「連鎖中フラグ」を渡し、True なら直前盤面を保持
- temporal_smoother / stateful_board_tracker と統合

### サイクル 3: ROI/HSV 動的キャリブレーション (未着手)
- 試合開始時に枠 (灰色境界) を Canny + Hough で検出
- 動画ごとの calibration.json 自動生成

### サイクル 4: 訓練データ増強 (未着手)
- video_02 / video_03 から連鎖中以外のフレーム抽出 → npz 化
- CNN 再訓練

## 9. レビュー予定

| 時刻 (目安) | 内容 |
|---|---|
| 30-40 分後 | サイクル 1 動画完成 → ユーザ目視 |
| 1-1.5 時間後 | サイクル 2 完了 (動画再生成不要、内部修正のみ) |
| 2-3 時間後 | サイクル 3 完了 + 動画再生成 |
| 半日後 | サイクル 4 完了 + 全体精度測定
