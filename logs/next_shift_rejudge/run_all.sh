#!/bin/bash
# 再判定の一括実行 (exev DECISIONS 2026-10-01 事前登録 (再判定))。
# GPU: M0 seed1..4 × orig/T を順に。CPU: S′・G_fe を先に全部作る。再生は構成ごとに5並列で順に。
cd /mnt/d/puyo_analyzer/wt_nextfix
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
R=logs/next_shift_rejudge
T=logs/next_shift_train
for k in 0 1 2 3 4; do for v in orig T; do
  for part in sprime gfe; do PYTHONPATH=. $PY -m scripts.next_shift_rejudge_models_20261001 --variant $v --seed $k --part $part >> $R/models.log 2>&1 || echo "FAIL $v $k $part" >> $R/models.log; done
done; done
echo cpu_done > $R/cpu_models.done
( for k in 0 1 2 3 4; do for v in orig T; do
    OMP_NUM_THREADS=2 PYTHONPATH=. $PY -m scripts.next_shift_rejudge_models_20261001 --variant $v --seed $k --part m0 >> $R/m0.log 2>&1 || echo "FAIL $v $k m0" >> $R/m0.log
  done; done; echo done > $R/m0.done ) &
( for k in 1 2 3 4; do for v in orig T; do
    src=$([ $v = orig ] && echo /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/e15 || echo $T/sprime_T)
    PYTHONPATH=. nice -n 10 $PY $T/tools/s1_cv.py $src $R/s1cv_${v}_rs$k $k >> $R/s1cv.log 2>&1
  done; done; echo done > $R/s1cv.done ) &
for k in 1 2 3 4; do for v in orig T; do
  until [ -f models/rejudge/${v}_rs${k}_common/M0/manifest.json ]; do sleep 20; done
  pids=""
  for s in q_7gc4TgFig fcXG83vInDY mia8KCjr52g review zenchi; do bash $T/replay.sh J_${v}_rs$k $s & pids="$pids $!"; done
  wait $pids
  bash $T/score.sh J_${v}_rs$k
done; done
wait
echo done > $R/all.done
