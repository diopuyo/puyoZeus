#!/bin/bash
# C-5 完成形バックアップ
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

BACKUP="data/snapshots/phase_c_c5_2026-05-04"
mkdir -p "$BACKUP/src" "$BACKUP/scripts" "$BACKUP/docs" "$BACKUP/review_videos" "$BACKUP/models"

# src core modules
cp src/board_state_machine.py "$BACKUP/src/"
cp src/state_detectors.py "$BACKUP/src/"
cp src/inference_board.py "$BACKUP/src/"
cp src/drift_detector.py "$BACKUP/src/"
cp src/recognition_pipeline.py "$BACKUP/src/"
cp src/per_video_model_selector.py "$BACKUP/src/"

# Phase B/C scripts
for s in phase_b_render_review_video phase_b_extract_review_frames \
         phase_b_debug_lag_frames phase_b_pipeline_eval_all \
         phase_b_finetune_cnn phase_b_drift_analysis \
         phase_b_chain_inference_analysis phase_b_collect_menu_truth_dataset \
         phase_b_collect_drift_dataset phase_b_collect_chain_truth_v2_dataset \
         phase_c_debug_next_detector phase_c_debug_pipeline_next \
         phase_c_review_next_detection; do
    if [ -f "scripts/${s}.py" ]; then
        cp "scripts/${s}.py" "$BACKUP/scripts/"
    fi
done

cp docs/HANDOFF_2026-05-03_PHASE_B.md "$BACKUP/docs/"
cp data/review_videos/README.md "$BACKUP/"
cp data/review_videos/v02_review_205_306.mp4 "$BACKUP/review_videos/"
cp models/cnn_phase_b_v1.pt "$BACKUP/models/"

cat > "$BACKUP/VERSION.md" <<EOF
# Phase C C-5 — 2026-05-04 完成形

## 概要
Option B (着地検出 + smoothing OFF) に C-2 (TSUMO→STABLE 物理推論) +
C-5 (差分のみ反映 + BG FP robust 化) を統合。次は C-6 (B+C+A 推論強化)。

## 主要修正
1. inferred_board hold-on-None
2. 1P/2P 同期 (active hysteresis)
3. 試合開始 chain ban
4. **背景 FP 自動採取 (capture_robust_fingerprint + 連続空盤面 5 frame)**
5. score-based match strengthen
6. GPU 推論 (RTX 4060 Laptop)
7. 文字オーバーレイ (R/B/G/Y/P/O 半透明色背景)
8. 試合 1+2 連続抽出 (--n-matches=2)
9. **TsumoPhaseDetector landed_consec=2 (着地検出)**
10. **temporal_smoothing=1 デフォルト (CLI 明示優先)**
11. **TSUMO→STABLE 物理推論 (着地差分 2 cell のみ転写)**
12. **\_merge_diff_only で全 state 遷移経路で baseline 維持**

## 既知の残課題
- 背景・エフェクトの持続的誤認が多数決を満たして confirmed に乗る
- ChainSim 連鎖後盤面ではなく CNN 盤面が CHAIN→STABLE 復帰時に採用されている
- 浮きぷよ (空の上に puyo) の物理整合性 filter 未実装

## レビュー動画
- review_videos/v02_review_205_306.mp4 (v02 試合 1+2)

## 次セッション着手
C-6: B (確定 cell 保護) + C (ChainSim final_board 採用) + A (浮きぷよ ban)
EOF

echo "[done] backup -> $BACKUP"
du -sh "$BACKUP"
