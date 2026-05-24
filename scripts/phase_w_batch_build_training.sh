#!/bin/bash
# W: phase_w_build_training_data_one.py を 4 並列で全 19 動画に適用 + 結合。

set -e
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

PY="./venv/bin/python"
PARALLELISM=4
LOG_DIR="/tmp/phase_w_build"
mkdir -p "$LOG_DIR"

run_one() {
    local id=$1
    local out="data/training_phase_w/per_video/win_pred_v$(printf '%02d' $id).npz"
    if [ -s "$out" ]; then
        echo "  v${id}: SKIP"
        return 0
    fi
    echo "  v${id}: START"
    PYTHONPATH=. $PY -m scripts.phase_w_build_training_data_one \
        --video-id $id --interval 4.0 \
        > "$LOG_DIR/v${id}.log" 2>&1
    echo "  v${id}: DONE"
}

echo "=== build per-video (parallelism=$PARALLELISM) ==="
for id in $(seq 1 19); do
    while [ $(jobs -pr | wc -l) -ge $PARALLELISM ]; do
        sleep 5
    done
    run_one $id &
done
wait
echo "=== per-video done ==="

echo "=== merge ==="
PYTHONPATH=. $PY -c "
import numpy as np
from pathlib import Path
files = sorted(Path('data/training_phase_w/per_video').glob('win_pred_v*.npz'))
all_X, all_y, all_mid, all_vid = [], [], [], []
for f in files:
    d = np.load(f)
    all_X.append(d['features'])
    all_y.append(d['labels'])
    all_mid.append(d['match_ids'])
    all_vid.append(d['video_ids'])
    print(f'  {f.name}: {d[\"features\"].shape[0]} samples')
X = np.concatenate(all_X)
y = np.concatenate(all_y)
mid = np.concatenate(all_mid)
vid = np.concatenate(all_vid)
print(f'total: {X.shape}, 1P={int(y.sum())}/{len(y)}')
out = Path('data/training_phase_w/win_pred_train_v2.npz')
np.savez_compressed(out, features=X, labels=y, match_ids=mid, video_ids=vid)
print(f'saved: {out}')
"
echo "=== complete ==="
