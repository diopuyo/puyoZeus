#!/bin/bash
# Option B バックアップ (git なしの environments 用)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

BACKUP="data/snapshots/phase_b_optionB_2026-05-04"
mkdir -p "$BACKUP/src" "$BACKUP/scripts" "$BACKUP/docs" "$BACKUP/review_videos" "$BACKUP/models"

# src core modules
cp src/board_state_machine.py "$BACKUP/src/"
cp src/state_detectors.py "$BACKUP/src/"
cp src/inference_board.py "$BACKUP/src/"
cp src/drift_detector.py "$BACKUP/src/"
cp src/recognition_pipeline.py "$BACKUP/src/"
cp src/per_video_model_selector.py "$BACKUP/src/"

# Phase B scripts
for s in phase_b_render_review_video phase_b_extract_review_frames \
         phase_b_debug_lag_frames phase_b_pipeline_eval_all \
         phase_b_finetune_cnn phase_b_drift_analysis \
         phase_b_chain_inference_analysis phase_b_collect_menu_truth_dataset \
         phase_b_collect_drift_dataset phase_b_collect_chain_truth_v2_dataset; do
    if [ -f "scripts/${s}.py" ]; then
        cp "scripts/${s}.py" "$BACKUP/scripts/"
    fi
done

# docs
cp docs/HANDOFF_2026-05-03_PHASE_B.md "$BACKUP/docs/"
cp data/review_videos/README.md "$BACKUP/"
cp data/review_videos/v02_review_205_306.mp4 "$BACKUP/review_videos/"

# CNN model
cp models/cnn_phase_b_v1.pt "$BACKUP/models/"

# version note
cat > "$BACKUP/VERSION.md" <<EOF
# Phase B Option B — 2026-05-04 完成形

## 概要
新方針 (BoardStateMachine + per-video model + 推論主軸) の Option B 適用版。
Option C (物理推論主軸の TSUMO 着地推論) 着手前のスナップショット。

## 適用された全修正
1. inferred_board hold-on-None
2. 1P/2P 同期 (active hysteresis, MATCH_ACTIVE_HOLD_FRAMES=10)
3. 試合開始 chain ban (CHAIN_BAN_FRAMES_AFTER_MATCH_START=30)
4. 背景 FP 自動採取 (試合開始 +5 frame)
5. force_in_match (MatchStateDetector の試合中誤判定回避、撤去済)
6. score-based match strengthen (score>0 なら IN_MATCH 強制)
7. GPU 推論 (RTX 4060 Laptop, CnnPatchClassifier.to_device("cuda"))
8. 文字オーバーレイ (R/B/G/Y/P/O 半透明色背景 + 白文字)
9. 試合 1+2 連続抽出 (--n-matches=2)
10. **Option B (本リリース): TsumoPhaseDetector landed_consec=2**
    - 着地検出: TSUMO_FALL 中に CNN 盤面が 2 frame 連続同一なら STABLE 復帰
    - 旧バグ「diff==0 で STABLE 復帰のみ → +2 puyo 状態で TSUMO_FALL ロックイン」を解消
11. **Option B: temporal_smoothing=1 (OFF)** — レイテンシ重視

## 精度結果
- 平均 STABLE 確定率: 60.4% (PV2)
- 単 frame 精度: 96.7% (試合中区間平均)
- CHAIN 推論精度: 77.75%
- CNN holdout: 99.61% (cnn_phase_b_v1)

## 既知の残課題
- 上部で少し遅延残る
- 背景色を puyo と認識するチラつき (= ImageReader/CNN の境界 cell 誤検出)
- → Option C (物理推論主軸の着地後 puyo 位置推論) で改善期待

## レビュー動画
- review_videos/v02_review_205_306.mp4 (v02 試合 1+2 連続、101 秒)

## 次セッション着手
Option C: TsumoPhaseDetector が「着地時点で next 履歴 + 着地位置から
物理推論盤面を生成」する設計に変更。CNN 出力に依存せず、着地 1 frame で反映。
EOF

echo "[done] backup -> $BACKUP"
du -sh "$BACKUP"
ls -R "$BACKUP" | head -50
