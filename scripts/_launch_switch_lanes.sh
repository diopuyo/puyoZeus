#!/usr/bin/env bash
# 2 レーン並走 (CPU 上限 2 プロセス・nice 10)。レーン1=OFF の残り4本、レーン2=構成A の5本。
cd /mnt/d/puyo_analyzer/wt_switch
setsid -f bash scripts/_run_switch_replay.sh off q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi > logs/switch_smoothing/lane1.log 2>&1 < /dev/null
setsid -f bash scripts/_run_switch_replay.sh a review q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi > logs/switch_smoothing/lane2.log 2>&1 < /dev/null
