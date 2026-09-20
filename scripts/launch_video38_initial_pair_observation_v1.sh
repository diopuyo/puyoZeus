#!/usr/bin/env bash
set -euo pipefail

project_root="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
output_root="${1:?新規出力rootを指定してください}"
script_sha="${2:?検収済みadapter SHAを指定してください}"
test_sha="${3:?検収済みtest SHAを指定してください}"
launcher_sha="${4:?検収済みlauncher SHAを指定してください}"
log_path="${output_root}.log"
cd "$project_root"
if [[ -e "$output_root" || -e "$log_path" ]]; then
  echo "既存出力を上書きしません" >&2
  exit 1
fi
set -o noclobber
setsid -f env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  "$project_root/venv/bin/python" -u -m scripts.diagnose_video38_initial_pair_observation_v1 \
  --script-sha256 "$script_sha" --test-sha256 "$test_sha" --launcher-sha256 "$launcher_sha" \
  --allow-native-runtime-mismatch --output-root "$output_root" \
  > "$log_path" 2>&1 < /dev/null
sleep 1
observation_pid="$(pgrep -n -f "^${project_root}/venv/bin/python -u -m scripts.diagnose_video38_initial_pair_observation_v1 .*--output-root ${output_root}$")"
printf 'PID=%s\nLOG=%s\nOUTPUT=%s\n' "$observation_pid" "$log_path" "$output_root"
