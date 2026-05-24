# 引継ぎ: cycle 71v Large CNN v2 (2026-05-14)

## 現状サマリ

- **cycle 71v 案 D 完了**: `CnnPatchClassifierLarge` (100KB, 4-conv + BN + Dropout) を scratch 訓練
- **v2 訓練**: 7 動画 31,176 cells で val accuracy **98.87%** 達成
- **モデル**: `models/cnn_phase_b_large_v2.pt` (= 415KB on disk, 100,615 params)
- **viz**: 6 動画分の v20 生成済 (v50/v91/v89 = 既知、 v29/v40/v57 = 新学習)
- **状態**: ユーザーレビュー待ち

## 訓練データ累計

| 動画 | cells |
|---|---|
| test_v50 | 1,008 |
| v50_match1 | 7,992 |
| v89_match1 | 3,960 |
| v91_match1 | 2,232 |
| v29_match2 | 6,984 |
| v40_match7 | 5,040 |
| v57_match2 | 3,960 |
| **合計** | **31,176** |

UNKNOWN (label 10) 1,081 を除外して訓練 → 30,095 samples (train 27,086 / val 3,009)

## v20 viz レビュー対象

既知動画 (= 過剰適合チェック):
- `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v50_match1_75s_viz_finetuned_v20.mp4`
- `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v91_match1_75s_viz_finetuned_v20.mp4`
- `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v89_match1_75s_viz_finetuned_v20.mp4`

学習データ動画自身 (= ホールド検証):
- `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v29_match2_viz_v20.mp4`
- `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v40_match7_viz_v20.mp4`
- `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v57_match2_viz_v20.mp4`

## v19 (前回 4 動画 Large) 残課題 - v20 で要確認

- **v89 試合 2**: 1P 1 列目 1 段目の黄色が EMPTY と誤認 (= 1 cell のみ、学習薄さ起因の可能性)
- **v50 全消し文字**: 全消し overlay が cell として認識される (= X ラベルで未学習)

## v20 視覚レビュー観点

1. **v89 試合 2 の 1P 黄色 誤認は解消されたか**
   - v29/v40/v57 ラベル追加で改善 = 学習薄さ起因と判明
   - 改善なし = データ量問題ではなく構造的問題
2. **v29/v40/v57 で新規動画品質**
   - 配色多様性での汎用化検証
3. **全消し overlay 誤認**
   - 未対処なので残存予想 → 別途 EffectPhaseDetector 統合で機械的解消が筋

## 次セッションでの選択肢

### 選択肢 A: 残課題を機械的に解消 → 次工程
- `EffectPhaseDetector` で全消し検出時に cell 表示凍結
- `cnn_phase_b_large_v2.pt` を default に昇格 (`recognition_pipeline.load_default` の default 変更)
- v89 試合 2 残課題は backlog → Phase L (動画追加) で自然解消狙い
- 次工程候補: Phase L 本番化 / RL preprocessing / 配信オーバーレイ (Phase J)

### 選択肢 B: 更にラベリング → 99%+ 詰める
- 追加候補 (DL 済): `v51_match2_97s.mp4`, `v70_match2_113s.mp4`, `v89_match3_95s.mp4`
- 別動画 (新規 DL) なら yt-dlp + ティア filter 必須
- 工数: 1 本 30-60 分、 3 本で 1.5-3 時間

### 選択肢 C: 並走 (= A の機械手当て + B の追加ラベリング背景)

## 関連ファイル

### 新規実装
- `src/patch_classifier.py:872` `CnnPatchClassifierLarge`
- `src/recognition_pipeline.py:_build_hybrid_reader` (state_dict shape 自動判別)
- `scripts/phase_i_fine_tune.py` `--cell-arch [small|large]` arg

### ラベリング起動 batch (Windows)
- `start_labeling_v50_match1.bat` / `v50_match2.bat` / `v91.bat` / `v89.bat`
- `start_labeling_v29.bat` / `v40.bat` / `v57.bat` (2026-05-14 追加)

### 訓練・viz コマンド
- 訓練: `wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune --component cell_color --video-ids test_v50,v50_match1,v89_match1,v91_match1,v29_match2,v40_match7,v57_match2 --cell-arch large --cell-save-to models/cnn_phase_b_large_v2.pt --epochs 15 --lr 1e-3 --augment --class-balance"`
- viz: `PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition --video <input.mp4> --output <output.mp4> --cnn-model models/cnn_phase_b_large_v2.pt`

### v2 訓練ログ
- `logs/train_cnn_large_v2.log`
- `logs/viz_v50_v20.log` 〜 `viz_v57_v20.log`

## 設計判断記録

- **設計方針保持**: STABLE 確定盤面のみで指標評価 (CLAUDE.md「設計思想 4.」)
- **認識目標調整**: 99.99% は asymptotic、 99%+ で次工程移行も選択肢 (memory `recognition_target_995.md` 緩和の可能性)
- **HSV ベース fallback**: low confidence 時 HSV 採用 (`HybridClassifier.LOW_CONFIDENCE_UNKNOWN_THRESHOLD=0.0` = UNKNOWN マーク無効)

## CLAUDE.md / メモリ参照

- `MEMORY.md` の `project_cycle_71v_large_cnn.md` を 2026-05-14 に更新済
- `feedback_recognition_target_995.md` (= 99.99% 目標) は緩和判断待ち
- `feedback_autonomous_operation.md` (= 確認不要自律) は継続適用
