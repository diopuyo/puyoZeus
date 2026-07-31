#!/bin/bash
# フェーズ2 コストゲート: --with-next 有無の実wall時間比較 (60秒クリップ)
set -e
PROJ=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$PROJ"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3

VIDEO=data/frames/video_c1.mp4
TMPDIR=/tmp/lean_next_bench
mkdir -p "$TMPDIR"
PY=./venv/bin/python

echo "=== baseline (--with-next なし, 60秒, sample-interval 0.2) ==="
START=$(date +%s%3N)
$PY -m scripts.collect_boards_lean --video "$VIDEO" --out-npz "$TMPDIR/base.npz" \
    --max-sec 60 --sample-interval 0.2
END=$(date +%s%3N)
echo "wall time: $((END - START)) ms"

echo ""
echo "=== --with-next あり (同条件) ==="
START=$(date +%s%3N)
$PY -m scripts.collect_boards_lean --video "$VIDEO" --out-npz "$TMPDIR/withnext.npz" \
    --max-sec 60 --sample-interval 0.2 --with-next
END=$(date +%s%3N)
echo "wall time: $((END - START)) ms"

echo ""
echo "=== next1_a 実値確認 ==="
$PY - <<'PYEOF'
import numpy as np
d = np.load('/tmp/lean_next_bench/withnext.npz')
n = d['next1_a']
valid = (n >= 0).sum()
print(f"snapshots={len(n)} next1_a>=0 count={valid} ({valid/len(n)*100:.1f}%)")
print("next1_a sample:", n[:20].tolist())
print("next1_b sample:", d['next1_b'][:20].tolist())
PYEOF
