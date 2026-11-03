#!/bin/bash
# 他レーンと共有するPCで、配布・B3・表示経路だけを低優先度で検査する。
set -euo pipefail
ROOT=/mnt/d/puyo_analyzer/wt_obs_pkg
OUT=/mnt/d/puyo_analyzer/packaging/build_20261011/checks
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
cd /mnt/d/puyo_analyzer/packaging/build/PuyoLive/app
export PYTHONPATH="$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec nice -n 19 "$PY" -m pytest -q -o addopts= \
  "$ROOT/tests/test_launcher.py" "$ROOT/tests/test_launcher_env.py" \
  "$ROOT/tests/test_launcher_manifest.py" "$ROOT/tests/test_bundle_asset_list.py" \
  "$ROOT/tests/test_packaging_closure.py" "$ROOT/tests/test_packaging_equiv.py" \
  "$ROOT/tests/test_b3_live_config.py" "$ROOT/tests/test_b3_realtime.py" \
  "$ROOT/tests/test_b3_async.py" "$ROOT/tests/test_b3_async_display.py" \
  "$ROOT/tests/test_display_completion_b3.py" "$ROOT/tests/test_display_fast_ema.py" \
  "$ROOT/tests/test_verify_display_latency.py" "$ROOT/tests/test_run_live_pipeline_20260928.py" \
  "$ROOT/tests/phase_j/test_display_completion_publish.py" \
  "$ROOT/tests/phase_j/test_live_publish.py" "$ROOT/tests/phase_j/test_live_process.py" \
  "$ROOT/tests/phase_j/test_live_source.py" "$ROOT/tests/phase_j/test_live_layers.py" \
  --junitxml="$OUT/tests.xml"
