#!/bin/bash
# olRyxDGacbg 陰性対照 regen 完了待ち → ゲート自動実行 (使い捨て、2026-08-07)
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
NPZ="data/verify/phase_l_quality_gate_2026-08-07/negative_control/olRyxDGacbg_10min.npz"
LOG="logs/phase_l_quality_gate_negctrl_2026-08-07.log"

while ! pgrep -f "_collect_lean_1t.*olRyxDGacbg" > /dev/null; do sleep 5; done
while pgrep -f "_collect_lean_1t.*olRyxDGacbg" > /dev/null; do sleep 20; done
echo "[negctrl-wait] regen finished $(date)" >> "${LOG}"

if [ -f "${NPZ}" ]; then
  ./venv/bin/python scripts/phase_l_video_quality_gate.py \
    --video olRyxDGacbg_10min --target-npz "${NPZ}" >> "${LOG}" 2>&1
  echo "[negctrl-wait] gate run done $(date)" >> "${LOG}"
else
  echo "[negctrl-wait][ERROR] npz missing after process exit" >> "${LOG}"
fi
