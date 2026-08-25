#!/bin/bash
# 非決定性調査の最終バッチ: r2 を含む全ペア突合 + 走査器 + 判断影響
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
D=data/verify/adv_nondeterminism_2026-08-25
G=data/verify/gate3r6_planA_2026-08-25
PY=./venv/bin/python
CMP=scripts/_diag_adv_nondeterminism_compare_2026-08-25.py

echo "== run1 vs r2 =="
nice -n 19 $PY $CMP $G/first5games_planA_off_run1.npz $D/dump_r2.npz 2>&1 | grep -E "adv_raw:|adv_ema:"
echo "== run2 vs r2 =="
nice -n 19 $PY $CMP $G/first5games_planA_off_run2_determinism_check.npz $D/dump_r2.npz 2>&1 | grep -E "adv_raw:|adv_ema:"
echo "== r1 vs r2 =="
nice -n 19 $PY $CMP $D/dump_r1.npz $D/dump_r2.npz 2>&1 | grep -E "adv_raw:|adv_ema:"
echo "== scanner r2 =="
nice -n 19 $PY -m scripts.scan_judgment_anomalies --from-dump $D/scan_in_r2 --out-dir $D/scan_out_r2 2>&1 | grep -E "ゲート対象|suspects="
echo "== decision impact r1 vs r2 =="
nice -n 19 $PY scripts/_diag_adv_nondet_decision_impact_2026-08-25.py $D/dump_r1.npz $D/dump_r2.npz
