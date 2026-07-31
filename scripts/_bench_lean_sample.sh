#!/bin/bash
# collect_boards_lean --sample-interval ベンチマーク (短尺 30 秒)
set -e

PROJ=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$PROJ"

VIDEO=data/frames/video_29.mp4
TMPDIR=/tmp/lean_bench
mkdir -p "$TMPDIR"

PY=./venv/bin/python

echo "=== 全フレーム (sample-interval なし) ==="
START_ALL=$(date +%s%3N)
$PY -m scripts.collect_boards_lean \
    --video "$VIDEO" --out-npz "$TMPDIR/lean_all.npz" --max-sec 30
END_ALL=$(date +%s%3N)
ELAPSED_ALL=$((END_ALL - START_ALL))
echo "wall time: ${ELAPSED_ALL} ms"
SNAP_ALL=$($PY - <<'PYEOF'
import numpy as np
d = np.load('/tmp/lean_bench/lean_all.npz')
print(len(d['grids']))
PYEOF
)
echo "snapshots: ${SNAP_ALL}"

echo ""
echo "=== sample-interval 0.1s ==="
START_01=$(date +%s%3N)
$PY -m scripts.collect_boards_lean \
    --video "$VIDEO" --out-npz "$TMPDIR/lean_01.npz" --max-sec 30 --sample-interval 0.1
END_01=$(date +%s%3N)
ELAPSED_01=$((END_01 - START_01))
echo "wall time: ${ELAPSED_01} ms"
SNAP_01=$($PY - <<'PYEOF'
import numpy as np
d = np.load('/tmp/lean_bench/lean_01.npz')
print(len(d['grids']))
PYEOF
)
echo "snapshots: ${SNAP_01}"

echo ""
echo "=== 結果サマリ ==="
echo "全フレーム: ${ELAPSED_ALL} ms / ${SNAP_ALL} snapshots"
echo "0.1s間引き: ${ELAPSED_01} ms / ${SNAP_01} snapshots"
$PY - <<PYEOF
all_ms = ${ELAPSED_ALL}
si_ms  = ${ELAPSED_01}
all_n  = ${SNAP_ALL}
si_n   = ${SNAP_01}
ratio = all_ms / si_ms if si_ms > 0 else 0
snap_ratio = si_n / all_n * 100 if all_n > 0 else 0
print(f"速度比 (全/間引き): {ratio:.2f}x")
print(f"snapshot保持率: {snap_ratio:.1f}% ({si_n}/{all_n})")
PYEOF
