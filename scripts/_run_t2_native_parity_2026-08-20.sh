#!/bin/bash
# T2: 同一動画を native ON/OFF で収集し、npz の全列が bit 一致するか検証する。
#
# 合成パッチでの一致 (T1、4,732枚×4構成で不一致0) は確認済みだが、本番経路
# 全体を通したときに同じ盤面が出るかは別問題。npz の全キー (grids/won/score/
# ojama_* 含む) が array_equal であることを受け入れ条件にする。
#
# 30fps 1本 + 60fps 1本を全長で回す (fable 設計の T2)。動画間で状態を持たない
# ので 48本フル比較は不要 — 動画内の区間の多様性が被覆されれば検証力は増えない。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
exec >> logs/t2_native_parity_2026-08-20.log 2>&1
echo "=== T2 start $(date +%F_%T) ==="

CF=$(PYTHONPATH=. ./venv/bin/python -c "from src.production_config import collect_flags; print(collect_flags())")
OUT_OFF=data/indicators_v2/boards_lean_t2_off_2026-08-20
OUT_ON=data/indicators_v2/boards_lean_t2_on_2026-08-20
mkdir -p "$OUT_OFF" "$OUT_ON"

# 39 = 30fps/9.4万frame (軽い)、38 = 60fps/19.8万frame
for T in ${T2_TARGETS:-39 38}; do
  for MODE in off on; do
    if [ "$MODE" = "on" ]; then
      EXTRA="--enable-native-hsv-classifier"
      OUT="$OUT_ON"
    else
      EXTRA=""
      OUT="$OUT_OFF"
    fi
    echo "--- $T $MODE start $(date +%T) ---"
    PYTHONPATH=. ./venv/bin/python -m scripts._collect_lean_1t \
      --video "data/frames/video_${T}.mp4" \
      --out-npz "${OUT}/${T}.npz" \
      $CF $EXTRA \
      --with-next --enable-phantom-board-guard \
      --max-sec 0 --sample-interval 0 \
      > "logs/t2_${T}_${MODE}_2026-08-20.log" 2>&1
    echo "--- $T $MODE done rc=$? $(date +%T) ---"
  done
done
echo "=== T2 collect end $(date +%F_%T) ==="
