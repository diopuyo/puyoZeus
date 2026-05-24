#!/bin/bash
# C-6 完成形バックアップ (E-1 着手前)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

BACKUP="data/snapshots/phase_c_c6_2026-05-04"
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
# Phase C C-6 (B+C+A) — 2026-05-04 完成形

## 概要
C-5 + B (確定 cell 保護) + C (ChainSim final_board 採用) + A (浮きぷよ ban)。
次は E-1 (CNN を盤面更新ソースから完全排除)。

## C-6 主要修正
1. **A**: \_apply_gravity_filter で浮きぷよ ban
2. **B**: STABLE 多数決確定時に allow_puyo_to_empty=False (= 連鎖以外で消えない)
3. **C**: CHAIN→STABLE 復帰時に ChainSimulator.simulate(before_board).final_board を採用

## 残課題
- 背景誤認 / 上部での確定遅延 がまだ残る
- 多数決経路で CNN が採用される箇所が複数 (MENU→STABLE 初回, 連続多数決確定)
- → E-1: 盤面更新ソースを state 遷移時のみに完全制限

## 戻り方
このバックアップから戻すには src/scripts を上書きコピー:
  cp -r data/snapshots/phase_c_c6_2026-05-04/src/* src/
  cp -r data/snapshots/phase_c_c6_2026-05-04/scripts/* scripts/
EOF

echo "[done] backup -> $BACKUP"
du -sh "$BACKUP"
