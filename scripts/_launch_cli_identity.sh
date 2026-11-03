#!/usr/bin/env bash
cd /mnt/d/puyo_analyzer/wt_switch
mkdir -p logs/switch_smoothing/cli_prod logs/switch_smoothing/cli_explicit
setsid -f bash scripts/_run_cli_identity.sh prod zenchi review > logs/switch_smoothing/cli_p1.log 2>&1 < /dev/null
setsid -f bash scripts/_run_cli_identity.sh prod q_7gc4TgFig fcXG83vInDY mia8KCjr52g > logs/switch_smoothing/cli_p2.log 2>&1 < /dev/null
setsid -f bash scripts/_run_cli_identity.sh explicit zenchi review > logs/switch_smoothing/cli_e1.log 2>&1 < /dev/null
setsid -f bash scripts/_run_cli_identity.sh explicit q_7gc4TgFig fcXG83vInDY mia8KCjr52g > logs/switch_smoothing/cli_e2.log 2>&1 < /dev/null
