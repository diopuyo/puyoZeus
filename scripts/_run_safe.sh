#!/bin/bash
# 熱暴走再発防止ランナー(2026-07-16 導入)。
# 温度センサが読めない環境(WSL/ACPI拒否)のため、負荷を物理的に抑えて予防する。
#   - 低並列(既定2)  : 同時実行数を絞る
#   - スレッド制限     : 1プロセスが全16コアを占有しないよう numpy/torch を絞る
#   - nice -n 15       : 低優先度
#   - クールダウン     : 各ジョブ完了後にCPUを休ませる
# 使い方: bash scripts/_run_safe.sh <jobs_file> [MAXPAR=2] [COOLDOWN=60] [THREADS=3]
#   jobs_file = 1行1コマンド(PYTHONPATH=. は本スクリプトが設定)
set -u
JOBS="${1:?jobs file required}"
MAXPAR="${2:-2}"       # 同時実行上限(既定2 = 最大 MAXPAR*THREADS コア)
COOLDOWN="${3:-60}"    # ジョブ完了後の休止秒(既定60)
THREADS="${4:-3}"      # 1プロセスのスレッド上限(既定3)

cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
# numpy/OpenBLAS/MKL/torch のスレッドを絞る(単一プロセスの発熱を抑える最重要レバー)
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" \
       VECLIB_MAXIMUM_THREADS="$THREADS"

echo "[safe] MAXPAR=$MAXPAR THREADS=$THREADS COOLDOWN=${COOLDOWN}s  (最大負荷 ~$((MAXPAR*THREADS))/16コア)"
n=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  # 空きスロットが出るまで待つ(=前ジョブ完了待ち)
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  # 完了直後にクールダウン(CPUを休ませてから次を投入)
  sleep "$COOLDOWN"
  n=$((n+1))
  echo "[safe] ($n) start: $cmd"
  nice -n 15 bash -c "$cmd" &
done < "$JOBS"
wait
echo "[safe] ALL DONE ($n jobs)"
