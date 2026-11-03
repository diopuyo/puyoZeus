#!/usr/bin/env bash
cd /mnt/d/puyo_analyzer/wt_switch
setsid -f bash scripts/_run_switch_replay.sh b review q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi > logs/switch_smoothing/lane4.log 2>&1 < /dev/null
