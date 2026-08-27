#!/bin/bash
# Gate 4 条件5専用snapshot。条件1〜4の固定snapshotは変更しない。
set -euo pipefail
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

SNAP=data/verify/gate4_condition5_2026-08-26/_snapshot_cond5_codex_20260827_v12
if [[ -e "$SNAP" ]]; then
  echo "既存snapshotは上書きしない: $SNAP" >&2
  exit 1
fi
mkdir -p "$SNAP" logs/gate4_condition5_2026-08-26
rsync -a --exclude='__pycache__' --exclude='*.pyc' src/ "$SNAP/src/"
rsync -a --exclude='__pycache__' --exclude='*.pyc' scripts/ "$SNAP/scripts/"
rsync -a --exclude='__pycache__' --exclude='*.pyc' tests/ "$SNAP/tests/"
{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "git_head=$(git rev-parse HEAD)"
  sha256sum scripts/visualize_advantage_overlay.py \
    src/live_exchange_episode_tracker.py src/chain_id_resolver.py \
    src/exchange_episode_tracker.py src/exchange_ledger.py \
    src/ojama_accounting.py src/death_confirmation.py
  find data/verify/retrain_model62_2026-08-21 -maxdepth 1 -type f -print0 \
    | sort -z | xargs -0 sha256sum
  stat --printf='video_size=%s video_mtime=%y video=%n\n' \
    data/frames/video_zenchi_c0BQoMJwwQU.mp4
} > "$SNAP/manifest.txt"
touch "$SNAP/SNAPSHOT_COMPLETE"
echo "SNAPSHOT_DONE $SNAP"
