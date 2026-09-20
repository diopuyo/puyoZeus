#!/usr/bin/env bash
set -euo pipefail

project_root="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
output_root="${1:?新規shadow出力directoryを指定してください}"
log_path="${output_root}.log"
if [[ -e "$output_root" || -e "$log_path" ]]; then
  echo "出力またはログが既に存在します" >&2
  exit 1
fi
cd "$project_root"
set -o noclobber
setsid -f env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$project_root/venv/bin/python" -u -m scripts.diagnose_video38_atomic_landing_shadow_v1 \
  --start-sec 560 --end-sec 583 --allow-native-runtime-mismatch --output-root "$output_root" \
  > "$log_path" 2>&1 < /dev/null
sleep 1
shadow_pid="$(pgrep -n -f "^${project_root}/venv/bin/python -u -m scripts.diagnose_video38_atomic_landing_shadow_v1 .*--output-root ${output_root}$")"
printf 'PID=%s\nLOG=%s\nOUTPUT=%s\n' "$shadow_pid" "$log_path" "$output_root"
