#!/bin/bash
# 統合版の品質門と段別速度を直列で実行する (BelowNormal は台側で設定)。起動は1回だけ。
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
PY=C:/PuyoLive_20261012a/PuyoLive/python/python.exe
M=<work>/wt_live_par/logs/live_parallel/baseline_m
REC="<home>/Videos/2026-10-12 16-18-17.mp4"
OFF="--env PUYO_LIVE_COLOR_BUFFER=0 --env PUYO_LIVE_DELTA_CELLS=0 --env PUYO_LIVE_ONNX_PARALLEL=0"
CD="--env PUYO_LIVE_COLOR_BUFFER=1 --env PUYO_LIVE_DELTA_CELLS=1 --env PUYO_LIVE_ONNX_PARALLEL=0"
CDO="--env PUYO_LIVE_COLOR_BUFFER=1 --env PUYO_LIVE_DELTA_CELLS=1 --env PUYO_LIVE_ONNX_PARALLEL=1"
G="$PY scripts/live_integ_gate.py"
run() { echo "== $*"; $G run "$@" || { echo "FAILED $*"; exit 1; }; }
PHASE="${1:-all}"  # quality / speed / all。速度は他の重い処理と重ねないこと
if [ "$PHASE" != speed ]; then
# 品質門: 900枚 (video_d04 650秒〜) と 10/12録画 4,994枚
run --tag q900_m --source-root $M --logits $OFF
run --tag q900_off --logits $OFF
run --tag q900_cd --logits $CD
run --tag q900_cdo --logits $CDO
run --tag qrec_m --source-root $M --video "$REC" --start 0 --frames 4994 --logits $OFF
run --tag qrec_off --video "$REC" --start 0 --frames 4994 --logits $OFF
run --tag qrec_cd --video "$REC" --start 0 --frames 4994 --logits $CD
run --tag qrec_cdo --video "$REC" --start 0 --frames 4994 --logits $CDO
fi
if [ "$PHASE" != quality ]; then
# 速度: ABA 順 (logits記録なし)
run --tag s1_A $OFF
run --tag s1_B $CD
run --tag s2_A $OFF
run --tag s2_C $CDO
run --tag s3_A $OFF
run --tag s3_B $CD
run --tag s4_A $OFF
fi
echo ALL_DONE
