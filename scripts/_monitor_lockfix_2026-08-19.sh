#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
want="$1"
for i in $(seq 1 240); do
  n=$(ls logs/_verify_lockdown_release_fix_2026-08-19/*.json 2>/dev/null | wc -l)
  [ "$n" -ge "$want" ] && break
  sleep 30
done
echo "== replay jsons =="
ls logs/_verify_lockdown_release_fix_2026-08-19/ 2>/dev/null
echo "== replay log tail =="
tail -6 logs/_verify_lockdown_release_fix_2026-08-19.log
echo "== recollect log tail =="
tail -4 logs/_recollect_lockfix_2026-08-19.log
echo "== pytest tail =="
tail -1 logs/_pytest_full_lockfix_2026-08-19.log
