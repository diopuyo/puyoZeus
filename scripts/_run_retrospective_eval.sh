#!/bin/bash
# 強化アナリスト遡及適用 (2026-05-19): cycle 32d/e/g/baseline を 3 動画で
# board log JSONL 生成 → evaluator で評価 → cycle 失敗を自動 reject できるか検証
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p logs/board_logs
mkdir -p data/verify/retrospective_eval

# 各 (model, video) 組合せで board log 生成
# 既存 cycle 32d/e/g viz は mp4 のみで board log なし → 再走行で生成
# 動画は 1 つ (v89m3) に絞り、 model 4 種 × 1 動画 = 4 走行で工数節約

VIDEO_FILE="v89_match3_95s.mp4"
VIDEO_ID="v89m3"

run_log_gen() {
  local model_path="$1"
  local cycle_name="$2"
  local extra_args="$3"
  local log_path="logs/board_logs/${cycle_name}_${VIDEO_ID}.jsonl"
  local viz_log="logs/${cycle_name}_logreplay_${VIDEO_ID}.log"
  # 既存 mp4 を上書きしないため /tmp に出力
  local tmp_mp4="/tmp/${cycle_name}_${VIDEO_ID}_logreplay.mp4"
  echo "[logreplay-start] ${cycle_name}"
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "data/evaluation_videos/${VIDEO_FILE}" \
    --output "${tmp_mp4}" \
    --cnn-model "${model_path}" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "${log_path}" \
    ${extra_args} \
    > "${viz_log}" 2>&1
  echo "[logreplay-done] ${cycle_name} → ${log_path}"
  rm -f "${tmp_mp4}"  # mp4 は捨てる、 log だけ残す
}

echo "=== retrospective eval @ $(date) ==="

# baseline (= cnn_phase_b_large_v2.pt) ※既 default
run_log_gen "models/cnn_phase_b_large_v2.pt" "baseline" ""

# cycle 32d (= 5 クラス scratch)
run_log_gen "models/cnn_cycle32d.pt" "cycle32d" ""

# cycle 32e (= 6 クラス + EMPTY + PuyoPresenceGate + ojama logit mask)
run_log_gen "models/cnn_cycle32e.pt" "cycle32e" "--mask-ojama-logit --use-puyo-gate"

# cycle 32g (= 32e + 円形マスク + EMPTY 採取拡張)
run_log_gen "models/cnn_cycle32g.pt" "cycle32g" "--mask-ojama-logit --use-puyo-gate --use-circle-mask"

echo "=== board log gen DONE @ $(date) ==="

# evaluator 適用
for cycle in baseline cycle32d cycle32e cycle32g; do
  echo "=== evaluate ${cycle} ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "logs/board_logs/${cycle}_${VIDEO_ID}.jsonl" \
    --report-out "data/verify/retrospective_eval/${cycle}_${VIDEO_ID}.json" \
    > "logs/retroeval_${cycle}_${VIDEO_ID}.log" 2>&1
  echo "[eval-done] ${cycle}"
done

echo "=== ALL DONE @ $(date) ===" | tee logs/retrospective_eval_done.flag
