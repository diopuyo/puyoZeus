#!/bin/bash
# 【未起動・要承認後に実行】cycle65 対整合ガード ON の全長再生成 → E36b 構成で再生 → 採点。
# 使い方 (Git Bash): MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/c65_guard/launch_full.sh
# 収集は R1b と同じく最大3並列 (docs/CYCLE_FINDINGS.md の並列上限)。既存 logs は書き換えない。
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/c65_guard/full logs/c65_guard/e36b_on
cat > logs/c65_guard/full/_pipeline.sh <<'INNER'
#!/bin/bash
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
run() { nice -n 10 "$PY" -B -m scripts.run_c65_guard_full --step "$1" --source "$2"; }
# 収集: 3レーン (zenchi は最長なので単独レーン)
( run collect zenchi; run collect review ) &
( run collect q_7gc4TgFig; run collect mia8KCjr52g ) &
( run collect fcXG83vInDY ) &
wait
# 再生 (E36b と同じ順・並列1) → 採点
for s in q_7gc4TgFig review fcXG83vInDY mia8KCjr52g zenchi; do run replay "$s"; done
nice -n 10 "$PY" -B -m scripts.run_c65_guard_full --step report
touch logs/c65_guard/full/PIPELINE_DONE
INNER
setsid -f bash logs/c65_guard/full/_pipeline.sh > logs/c65_guard/full/pipeline.log 2>&1 < /dev/null
