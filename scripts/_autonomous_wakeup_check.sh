#!/bin/bash
# 自律 cycle 起動条件確認 + 起動 script
# 1. baseline 評価完了確認
# 2. seed regen 完了確認
# 3. 両方 done なら autonomous_sweep.py + cycle51_ojama_dryrun 起動
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh

BASELINE_DONE_FLAG="logs/baseline_v3_eval/all_done.flag"
SEED_DONE_FLAG="logs/cycle50_seed_regen/all_done.flag"

if [ ! -f "$BASELINE_DONE_FLAG" ]; then
  echo "[wait] baseline not done"
  exit 1
fi

if [ ! -f "$SEED_DONE_FLAG" ]; then
  echo "[wait] seed regen not done"
  exit 1
fi

BASE_CRIT=$(./venv/bin/python -c "import json; d=json.load(open('data/verify/baseline_v3_eval/_summary.json')); print(d['totals']['critical'])")
echo "[autonomous] baseline critical = $BASE_CRIT"

# autonomous sweep を background で起動 (= 朝までずっと走らせる)
init_health autonomous_sweep_main
setsid -f bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.autonomous_sweep --sweep all > logs/autonomous_sweep_main.log 2>&1" < /dev/null
echo "[autonomous] sweep started (background)"

# cycle 51 ojama dryrun も並列で
setsid -f bash scripts/_cycle51_ojama_dryrun.sh > logs/cycle51_ojama_dryrun_master.log 2>&1 < /dev/null
echo "[cycle51] ojama dryrun started (background)"

# 合成 PNG 生成 (= 朝の 4 動画レビュー材料)
for key in v86m17 v52m5 v89m7 v34m13; do
  PYTHONPATH=. ./venv/bin/python -m scripts.compose_seed_diff \
    --before "data/seed_review/phase_l_${key}.png" \
    --after "data/seed_review/cycle50_${key}.png" \
    --output "data/seed_review/cycle50_diff_${key}.png" \
    --label "${key} (= cycle 50 改修効果)" \
    > /dev/null 2>&1
done
echo "[compose] 4 diff PNG generated"

finalize_health 0
