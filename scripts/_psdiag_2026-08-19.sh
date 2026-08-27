#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
echo "== recollect runner =="
pgrep -af "_recollect_lockfix_2026-08-19.py" | head -3
echo "== collect_lean_1t =="
pgrep -af "_collect_lean_1t" | cut -c1-160
echo "== per-video logs =="
ls logs/recollect_lockfix_2026-08-19/ 2>/dev/null
echo "== main log =="
ls -la logs/_recollect_lockfix_2026-08-19.log 2>/dev/null
tail -3 logs/_recollect_lockfix_2026-08-19.log 2>/dev/null
echo "== c109win log =="
tail -3 logs/_recollect_lockfix_c109win_2026-08-19.log 2>/dev/null
