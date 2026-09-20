#!/usr/bin/env bash
set -euo pipefail

project_root="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
mode="${1:?修正modeを指定してください}"
output_root="${2:?新規出力rootを指定してください}"
module_sha="${3:?候補sourceの固定SHAを指定してください}"
test_sha="${4:?候補testの固定SHAを指定してください}"
log_path="${output_root}.log"
cd "$project_root"
if [[ -e "$output_root" || -e "$log_path" ]]; then
  echo "既存出力を上書きしません" >&2
  exit 1
fi
case "$mode" in start_epoch|chain_end|gravity_grid) ;; *) exit 2 ;; esac
set -o noclobber
setsid -f env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  "$project_root/venv/bin/python" -u -m scripts.diagnose_video38_boundary_repair_shadow_v1 \
  --mode "$mode" --module-sha256 "$module_sha" --module-test-sha256 "$test_sha" \
  --allow-native-runtime-mismatch --output-root "$output_root" \
  > "$log_path" 2>&1 < /dev/null
sleep 1
repair_pid="$(pgrep -n -f "^${project_root}/venv/bin/python -u -m scripts.diagnose_video38_boundary_repair_shadow_v1 --mode ${mode} .*--output-root ${output_root}$")"
printf 'PID=%s\nLOG=%s\nOUTPUT=%s\n' "$repair_pid" "$log_path" "$output_root"
