# セッション引継ぎ 2026-04-29 (Phase W in-progress)

## 1 行サマリ

Phase W: 状況認識統合 (StatePipeline) + 勝率予測 MLP (in-distribution 97%、cross-video 63%) + **確率的盤面表現 (W3.0、隠し段+お邪魔の量子的推論)** + pl5/pl6 16 動画 DL 完了。count_match の OCR ベース修正版を実行中。

## 完成済 (Phase V + Phase W)

### Phase V (引継ぎ前完了)
- CNN v7 multi-video Holdout 99.8% (manual + parallel_strict ミックス、500万件)
- ロジック層強化 V2.1-2.4 (NextLink / Pair / Connectivity / EnhancedTracker)
- 構造的限界対処 V3.1-3.2 (テロップ UNKNOWN / 試合終了告知 + ロックダウン)
- phase_u_render.py に v7 + V2.4 + V3.2 実統合

### Phase W (今セッション完了)
- **W1.1 StatePipeline**: 4 項目を 1 frame から一括抽出 (9 tests)
- **W1.2 動画別精度検証**: confidence 0.5 + delta 50000 フィルタで安定
- **W2.1 勝率予測 MLP**: 1068 dim → 64 → 32 → 1 (sigmoid、15 tests)
- **W2.2 訓練データ**: video_01-03 + matches/winners → 3279 サンプル
- **W2.3 訓練 + ベースライン**:
  - in-distribution holdout: **97.15%** (+42pt vs majority baseline)
  - cross-video (v03 holdout): **63.04%** (P1/P2 swap aug 込み)
- **W3.0 確率的盤面表現**: ProbabilisticBoard / hidden_row_inferrer / ojama_position_inferrer (23 tests)
- **動画 DL**: pl5 (6) + pl6 (10) = 16 動画追加 (video_04-19、約 13GB)
- **動画 state overlay** 2 本: v05_315 / v06_190 (両 70 秒、下部に 4 項目可視化)

## 進行中

- **count_match_via_ocr.py 実行中** (video_10): score_zero detector 失敗を score_ocr で代替

## 残タスク

| ID | 内容 | 優先 |
|---|---|---|
| W4 (in progress) | count_match_via_ocr 全動画適用 → winners 生成 → 訓練データ追加 → MLP 再訓練 | 高 |
| W3.2 | 試合勝率カーブ可視化 (matplotlib) | 中 |
| W3.0 拡張 | 学習ベース確率推論 (類似盤面 NN) | 中 |
| W3.1 | Transformer/LSTM で cross-video 改善 | 低 |
| stream_overlay | OBS 統合検証 | 低 |

## 主要ファイル (Phase W 追加)

| パス | 役割 | テスト |
|---|---|---|
| `src/state_pipeline.py` | W1.1 4 項目抽出 + GameState (pboard 拡張) | 9 |
| `src/state_features.py` | W2.1 GameState → 1068 dim feature | 10 |
| `src/win_predictor.py` | W2.1 勝率予測 MLP | 5 |
| `src/probabilistic_board.py` | W3.0 ProbabilisticCell/Board | 11 |
| `src/hidden_row_inferrer.py` | W3.0 隠し段確率推論 | 7 |
| `src/ojama_position_inferrer.py` | W3.0 お邪魔位置確率推論 | 5 |
| `scripts/phase_w_build_training_data.py` | (state, label) 生成 | - |
| `scripts/phase_w_train_predictor.py` | MLP 訓練 + 比較 | - |
| `scripts/phase_w_render_state_overlay.py` | 動画下部 state パネル overlay | - |
| `scripts/count_match_via_ocr.py` | OCR ベース試合境界検出 (W4 用) | - |
| `models/win_predictor_v2_mixed.pt` | MLP 本流モデル | - |

## データセット

| パス | 件数 | 用途 |
|---|---|---|
| `data/training_phase_w/win_pred_train.npz` | 3,279 | 勝率予測訓練データ (v01-03) |
| `data/frames/video_04-19.mp4` | 16 動画 | pl5+pl6、訓練データ追加待ち |
| `data/verify/match_boundaries_v5/` | (生成中) | OCR ベース matches.tsv |

## 設計判断 (Phase W 追加)

1. **score 信頼度フィルタ** (0.5) で連鎖中の不安定 OCR を除外
2. **score delta 上限** (50000) で OCR 失敗復帰時の異常累積を回避
3. **P1/P2 swap データ拡張** で訓練データ 2 倍 + label 反転、cross-video +2pt
4. **count_match_v4 (score_zero ベース)** が pl5/pl6 で機能しない → **OCR ベース代替版**を実装
5. **確率的盤面**は隠し段とお邪魔のみ確率分布、可視領域は確定色
6. **学習ベース類似盤面推論**は将来課題 (現状ルールベース)

## 設計上の TODO

- count_match_via_ocr の動作確認後、16 動画一括適用 (Agent 並列化検討)
- 訓練データ拡大後、MLP 再訓練で cross-video 改善測定
- 学習ベース確率推論 (Phase W3.0 拡張) の必要性判断は MLP 改善結果見てから

## 再開コマンド

```bash
# 全テスト走行
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m pytest tests/ -q"

# 勝率予測 MLP 再訓練 (新データ生成後)
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_train_predictor --augment-swap --holdout-video 3 --epochs 30"

# state overlay 動画
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.phase_w_render_state_overlay --video data/frames/video_05.mp4 --start 315 --end 385 --out data/verify/state_overlay.mp4"
```

## ユーザールール (継続有効)

- 自律運転 OK、明らかに良い選択肢は聞かずに進める
- フルパスは Windows 形式
- 文字化け対策: terminal.integrated.fontFamily 設定済
- メモリ警告: WSL は 10GB 余裕あるが Windows 全体 16GB 使用、重い並列処理は控える
- 強化学習優先 → 教師あり勝率予測 (B 案) で進行中
- オーバーレイ・UI 系は優先度下げる
- ぷよぷよ AI (Self-play) は目的外
