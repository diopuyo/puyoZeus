#!/bin/bash
# 待機条件: 他エージェントの重い処理が終わる or サンプルが全148本そろう (どちらか早い方)
D=/mnt/d/puyo_analyzer/wt_prefire/logs/prefire_prediction/hazard_samples
while true; do
  n=$(ls $D/video_*.npz 2>/dev/null | grep -v tmp | wc -l)
  o=$(pgrep -f 'scan_telop_prefilter|profile_multilanding|run_multilanding_speed|measure_|bench' | wc -l)
  if [ "$n" -ge 148 ] || [ "$o" -eq 0 ]; then echo "samples=$n other=$o $(date)"; exit 0; fi
  sleep 30
done
