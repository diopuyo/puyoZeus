#!/bin/bash
# 【未起動・要承認後に実行】本番構成 + --verification-pending-chain-expiry の全長再生成 → E36b 構成で再生 → 採点。
# 使い方 (Git Bash): MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/pending_expiry/launch_full.sh
# 収集は最大3並列 (docs/CYCLE_FINDINGS.md の並列上限)。既存 logs は書き換えない。
# 見積もり (c65_guard 実走 2026-09-30 04:19→07:04 の実測を単位別に): 収集は 2h00〜2h30 (zenchi+review レーン
#   1h36〜2h00 / q+mia レーン 1h20〜2h28、動画1本の処理frame数で費用が決まる)、再生+採点 17〜20 分。
#   合計 2.3〜3.0 時間 (壁時計、他ジョブ並走なし前提)。
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
mkdir -p logs/pending_expiry/full logs/pending_expiry/e36b_on
cat > logs/pending_expiry/full/_pipeline.sh <<'INNER'
#!/bin/bash
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
run() { nice -n 10 "$PY" -B -m scripts.run_pending_expiry_full --step "$1" --source "$2"; }
( run collect zenchi; run collect review ) &
( run collect q_7gc4TgFig; run collect mia8KCjr52g ) &
( run collect fcXG83vInDY ) &
wait
for s in q_7gc4TgFig review fcXG83vInDY mia8KCjr52g zenchi; do run replay "$s"; done
nice -n 10 "$PY" -B -m scripts.run_pending_expiry_full --step report
touch logs/pending_expiry/full/PIPELINE_DONE
INNER
setsid -f bash logs/pending_expiry/full/_pipeline.sh > logs/pending_expiry/full/pipeline.log 2>&1 < /dev/null
