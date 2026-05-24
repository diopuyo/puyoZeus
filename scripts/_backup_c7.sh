#!/bin/bash
# C-7 (E-1) 完成形バックアップ
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

BACKUP="data/snapshots/phase_c_c7_2026-05-04"
mkdir -p "$BACKUP/src" "$BACKUP/scripts" "$BACKUP/docs" "$BACKUP/review_videos" "$BACKUP/models"
cp src/board_state_machine.py "$BACKUP/src/"
cp src/state_detectors.py "$BACKUP/src/"
cp src/inference_board.py "$BACKUP/src/"
cp src/drift_detector.py "$BACKUP/src/"
cp src/recognition_pipeline.py "$BACKUP/src/"
cp src/per_video_model_selector.py "$BACKUP/src/"
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
# Phase C C-7 (E-1) — 2026-05-04
連続多数決経由の confirmed 更新を停止、state 遷移時のみ更新。
背景・エフェクト誤認の経路は減ったが、
- 浮きぷよ filter が pipeline 上書き経路で効いていない
- 着地色が CNN 値依存 (ネクスト履歴未活用)
が残課題。次は C-8 (E-2)。
EOF
echo "[done] backup -> $BACKUP"
du -sh "$BACKUP"
