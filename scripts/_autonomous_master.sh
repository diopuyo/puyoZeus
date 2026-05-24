#!/bin/bash
# 朝までの自律実行 master loop (= 完全 detach、 Claude session 切断耐性)
#
# 役割:
#   1. baseline_v3_eval / cycle50_seed_regen の完了監視
#   2. 完了したら autonomous_sweep / cycle51_ojama_dryrun を background 起動
#   3. 進捗 dashboard を 5 分ごと更新 (= logs/autonomous_dashboard.md)
#   4. 朝 7:30 まで loop continuance (= 自走 cron 風)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health autonomous_master

DASHBOARD=logs/autonomous_dashboard.md
END_TS=$(date -d "07:30 today + 1 day" +%s 2>/dev/null || date -d "07:30 today" +%s)

update_dashboard() {
  {
    echo "# 自律実行 dashboard (= 2026-05-21 深夜)"
    echo ""
    echo "更新時刻: $(date)"
    echo ""
    echo "## 走行中プロセス"
    pgrep -af python | grep -E "extract_hsv|visualize_recognition|evaluate_recognition|autonomous_sweep|phase_i_fine" | head -20 || echo "  (none)"
    echo ""
    echo "## 完了 flag"
    for f in logs/baseline_v3_eval/all_done.flag \
             logs/cycle50_seed_regen/all_done.flag \
             logs/cycle51_ojama_dryrun/all_done.flag \
             logs/autonomous_sweep_main/all_done.flag; do
      if [ -f "$f" ]; then
        echo "- ✓ $f"
      else
        echo "- ✗ $f (= 未完了)"
      fi
    done
    echo ""
    echo "## 直近 sweep judgments (= 最大 15)"
    if [ -f data/verify/autonomous/_judgments.jsonl ]; then
      tail -15 data/verify/autonomous/_judgments.jsonl | while read line; do
        echo "  - \`$line\`"
      done
    else
      echo "  (まだ judgments なし)"
    fi
    echo ""
    echo "## baseline critical"
    if [ -f data/verify/baseline_v3_eval/_summary.json ]; then
      ./venv/bin/python -c "import json; d=json.load(open('data/verify/baseline_v3_eval/_summary.json')); print(f\"  total = {d['totals']['critical']}\"); [print(f\"  - {k}: {v}\") for k,v in d['per_video'].items()]"
    else
      echo "  (まだ baseline なし)"
    fi
    echo ""
    echo "## seed quality (= cycle 50 vs baseline)"
    for f in data/verify/seed_quality_phase_l_baseline.json data/verify/seed_quality_cycle50.json; do
      if [ -f "$f" ]; then
        ./venv/bin/python -c "import json; d=json.load(open('$f')); print(f\"  $f -> overall {d['summary']['overall_purity']}, per_color {d['summary']['per_color_purity']}\")"
      fi
    done
  } > "$DASHBOARD" 2>&1
}

start_autonomous_sweep_if_ready() {
  if [ -f logs/baseline_v3_eval/all_done.flag ] \
     && [ ! -f logs/autonomous_sweep_main/started.flag ] \
     && ! pgrep -f "autonomous_sweep" > /dev/null; then
    mkdir -p logs/autonomous_sweep_main
    touch logs/autonomous_sweep_main/started.flag
    setsid -f bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.autonomous_sweep --sweep all > logs/autonomous_sweep_main.log 2>&1; touch logs/autonomous_sweep_main/all_done.flag" < /dev/null
    echo "[master] autonomous_sweep started @ $(date)"
  fi
}

start_cycle51_if_ready() {
  if [ -f logs/cycle50_seed_regen/all_done.flag ] \
     && [ ! -f logs/cycle51_ojama_dryrun/started.flag ] \
     && ! pgrep -f "cycle51_ojama" > /dev/null; then
    mkdir -p logs/cycle51_ojama_dryrun
    touch logs/cycle51_ojama_dryrun/started.flag
    setsid -f bash scripts/_cycle51_ojama_dryrun.sh < /dev/null > /dev/null 2>&1
    echo "[master] cycle51_ojama_dryrun started @ $(date)"
  fi
}

compose_diff_pngs() {
  # 4 動画の改修前後合成 PNG (= 朝のレビュー材料)
  if [ -f logs/cycle50_seed_regen/all_done.flag ] \
     && [ ! -f logs/compose_diff/all_done.flag ]; then
    mkdir -p logs/compose_diff
    for key in v86m17 v52m5 v89m7 v34m13; do
      before="data/seed_review/phase_l_${key}.png"
      after="data/seed_review/cycle50_${key}.png"
      if [ -f "$before" ] && [ -f "$after" ]; then
        ./venv/bin/python -m scripts.compose_seed_diff \
          --before "$before" --after "$after" \
          --output "data/seed_review/cycle50_diff_${key}.png" \
          --label "${key} (= cycle 50 改修効果)" \
          > /dev/null 2>&1
      fi
    done
    touch logs/compose_diff/all_done.flag
    echo "[master] compose diff PNG done @ $(date)"
  fi
}

# メイン loop (= 5 分間隔、 朝 7:30 まで)
ITER=0
while true; do
  NOW=$(date +%s)
  if [ "$NOW" -ge "$END_TS" ]; then
    echo "[master] reached end time, exit"
    break
  fi
  ITER=$((ITER + 1))
  echo "[master iter=$ITER] $(date)"

  start_autonomous_sweep_if_ready
  start_cycle51_if_ready
  compose_diff_pngs
  update_dashboard

  sleep 300  # 5 分
done

finalize_health 0
