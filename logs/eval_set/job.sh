#!/bin/bash
# 再生ジョブ1件。使い方: bash job.sh <kind> <variant> <source>
#   kind=rejudge : wt_nextfix の再生器 (variant=J_orig_rs0 など)
#   kind=cli     : 再生CLI --production-exchange-event (variant=prod | e19 = 本番+--landing-counter-prob)
# 完了で logs/eval_set/jobs/<kind>_<variant>_<source>.done に終了コードを書く。
set -u
kind=$1; variant=$2; source=$3
E=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $E/jobs
tag=${kind}_${variant}_${source}
if [ "$kind" = rejudge ]; then
  cd /mnt/d/puyo_analyzer/nextfix_run
  PYTHONPATH=/mnt/d/puyo_analyzer/wt_nextfix nice -n 10 $PY -B /mnt/d/puyo_analyzer/wt_evalset/scripts/replay_eval_set_rejudge_20261001.py \
    --variant $variant --source $source > $E/jobs/$tag.log 2>&1
else
  cd /mnt/d/puyo_analyzer/evalset_run
  if [ "$source" = zenchi ]; then REC=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/pending_expiry/full/records/zenchi.jsonl.gz
  else REC=$E/collect/records/$source.jsonl.gz; fi
  EXTRA=""; [ "$variant" = e19 ] && EXTRA="--landing-counter-prob"
  mkdir -p $E/replay_cli/$variant/$source
  PYTHONPATH=/mnt/d/puyo_analyzer/wt_evalset nice -n 10 $PY -B -m scripts.replay_exchange_event_20260926 $REC \
    --out $E/replay_cli/$variant/$source --production-exchange-event $EXTRA > $E/jobs/$tag.log 2>&1
fi
echo $? > $E/jobs/$tag.done
