#!/bin/bash
# Gate 4正式検収用の固定snapshot。既存snapshot/成果物は上書きしない。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

SNAP=data/verify/gate4_formal_dense_2026-08-26/_snapshot_codex_20260826
if [[ -e "$SNAP/SNAPSHOT_COMPLETE" ]]; then
  echo "既存snapshotは上書きしない: $SNAP" >&2
  exit 1
fi
mkdir -p "$SNAP" logs/gate4_formal_dense_2026-08-26
rsync -a --exclude='__pycache__' --exclude='*.pyc' src/ "$SNAP/src/"
rsync -a --exclude='__pycache__' --exclude='*.pyc' scripts/ "$SNAP/scripts/"
rsync -a --exclude='__pycache__' --exclude='*.pyc' tests/ "$SNAP/tests/"
{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "git_head=$(git rev-parse HEAD)"
  sha256sum scripts/visualize_advantage_overlay.py src/death_confirmation.py \
    src/exchange_episode_tracker.py src/ojama_accounting.py
} > "$SNAP/manifest.txt"
touch "$SNAP/SNAPSHOT_COMPLETE"
echo "SNAPSHOT_DONE $SNAP"
