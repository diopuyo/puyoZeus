#!/usr/bin/env bash
# Tier B 反映後の学習 + ダッシュボード生成スクリプト
# 前提: data/training/match_features_phase_e_v01-94_tierb.csv が完成済
#
# 利用:
#     bash scripts/_run_phase_e_tierb_learning.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CSV="data/training/match_features_phase_e_v01-94_tierb.csv"
LR_OUT="data/verify/learned_weights_v3_tierb.json"
PA_OUT="data/verify/learned_weights_phase_e_phase_aware_tierb.json"
LGBM_OUT="data/verify/learned_weights_lgbm_tierb.json"
VIF_OUT="data/verify/multicollinearity_phase_e_tierb.json"
DASH_OUT="data/verify/phase_e_dashboard_tierb.md"

if [[ ! -f "$CSV" ]]; then
  echo "[error] CSV not found: $CSV"
  exit 1
fi
echo "[info] CSV rows: $(wc -l <"$CSV")"

# 1. LR L2 (V3 互換)
echo "[step] learn_weights_v3 (LR L2)"
PYTHONPATH=. ./venv/bin/python -m scripts.learn_weights_v3 \
    --csv "$CSV" --out "$LR_OUT" || echo "[warn] learn_weights_v3 failed"

# 2. Phase Aware LR
echo "[step] phase_e_learn_phase_aware (Phase Aware LR)"
PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_learn_phase_aware \
    --csv "$CSV" --out "$PA_OUT" || echo "[warn] phase_e_learn_phase_aware failed"

# 3. HistGradientBoosting (M-A)
echo "[step] learn_weights_lgbm (GBM)"
PYTHONPATH=. ./venv/bin/python -m scripts.learn_weights_lgbm \
    --csv "$CSV" --out "$LGBM_OUT" || echo "[warn] learn_weights_lgbm failed"

# 4. 多重共線性分析
echo "[step] multicollinearity_analysis"
PYTHONPATH=. ./venv/bin/python -m scripts.multicollinearity_analysis \
    --csv "$CSV" --out "$VIF_OUT" || echo "[warn] multicollinearity_analysis failed"

# 5. ダッシュボード集約
echo "[step] phase_e_dashboard"
PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_dashboard \
    --csv "$CSV" \
    --learn-json "$PA_OUT" \
    --vif-json "$VIF_OUT" \
    --out "$DASH_OUT" || echo "[warn] dashboard failed"

echo "[done] Step 3 complete:"
echo "  LR:        $LR_OUT"
echo "  PhaseAware: $PA_OUT"
echo "  GBM:       $LGBM_OUT"
echo "  Dashboard: $DASH_OUT"
