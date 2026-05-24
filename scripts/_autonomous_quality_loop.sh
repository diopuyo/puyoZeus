#!/bin/bash
# 朝までの自律品質向上 loop (= 2026-05-21 深夜 → 朝 8 時)
#
# 学習軸凍結中なので推論軸のみ sweep。 baseline_videos_v3 8 動画で
# 強化アナリスト critical 集計、 baseline 比改善 ≥ 5 で採用、 悪化で revert。
#
# 各 cycle:
#   1. parameter tuning (= 推論側 file 編集)
#   2. baseline_videos_v3 viz + 強化アナリスト評価
#   3. critical 比較 → JSON に結果保存
#   4. 採用 = parameter 維持、 revert = 元に戻す
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health autonomous_quality_loop

# baseline 取得 (= 既に baseline_v3_eval で取得済の想定)
BASELINE_JSON="data/verify/baseline_v3_eval/_summary.json"
if [ ! -f "$BASELINE_JSON" ]; then
  echo "[FATAL] baseline not found, run _eval_baseline_v3.sh first"
  finalize_health 1
  exit 1
fi

BASELINE_CRIT=$(./venv/bin/python -c "import json; print(json.load(open('$BASELINE_JSON'))['totals']['critical'])")
echo "[autonomous] baseline critical = $BASELINE_CRIT"

# eval_one_setup: 現在のコード状態で baseline_videos_v3 8 動画評価し critical 集計
# 引数: $1 = cycle 名 (例: cycle52_puyo_gate)
eval_one_setup() {
  local cycle_name="$1"
  local out_dir="data/verify/autonomous/$cycle_name"
  mkdir -p "$out_dir"
  mkdir -p "logs/autonomous/$cycle_name"

  for key in v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11; do
    local input="data/baseline_videos_v3/${key}_buf15s.mp4"
    local board_log="logs/autonomous/${cycle_name}/viz_${key}.jsonl"
    local report="$out_dir/${key}.json"

    if [ -f "$report" ]; then continue; fi

    PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
      --video "$input" --output "$out_dir/${key}.mp4" \
      --cnn-model models/cnn_phase_b_large_v2.pt \
      --hsv-state data/per_video_hsv_ranges/_merged_default.json \
      --dump-board-log "$board_log" \
      > "logs/autonomous/${cycle_name}/viz_${key}.log" 2>&1 &

    # 並列 2 制御
    if [ $(jobs -r | wc -l) -ge 2 ]; then
      wait -n
    fi
  done
  wait

  for key in v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11; do
    local board_log="logs/autonomous/${cycle_name}/viz_${key}.jsonl"
    local report="$out_dir/${key}.json"
    if [ -f "$report" ]; then continue; fi
    PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
      --board-log "$board_log" --report-out "$report" \
      > "logs/autonomous/${cycle_name}/eval_${key}.log" 2>&1
  done

  PYTHONPATH=. ./venv/bin/python -c "
import json, sys
from pathlib import Path
videos = ['v29m2', 'v40m7', 'v51m2', 'v57m2', 'v70m2', 'v89m3', 'v95m15', 'v97m11']
tot_c, tot_w = 0, 0
per = {}
for v in videos:
    p = Path(f'$out_dir/{v}.json')
    if not p.exists(): continue
    d = json.load(open(p, encoding='utf-8'))
    s = d.get('summary', {})
    c, w = s.get('critical', 0), s.get('warning', 0)
    per[v] = {'critical': c, 'warning': w}
    tot_c += c; tot_w += w
out = {'cycle': '$cycle_name', 'totals': {'critical': tot_c, 'warning': tot_w}, 'per_video': per}
Path('$out_dir/_summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
print(tot_c)
"
}

# 自律判定 + log 記録
judge_and_log() {
  local cycle_name="$1"
  local before_crit="$2"
  local after_crit="$3"
  local diff=$((after_crit - before_crit))
  local verdict
  if [ "$diff" -le -5 ]; then verdict="ACCEPT"
  elif [ "$diff" -ge 5 ]; then verdict="REJECT"
  else verdict="NEUTRAL"
  fi
  echo "[${cycle_name}] before=$before_crit after=$after_crit diff=$diff verdict=$verdict"
  echo "{\"cycle\":\"$cycle_name\",\"before\":$before_crit,\"after\":$after_crit,\"diff\":$diff,\"verdict\":\"$verdict\",\"ts\":$(date +%s)}" \
    >> data/verify/autonomous/_judgments.jsonl
}

# ============================================================
# Cycle 52: PuyoPresenceGate default 化 (= cycle 41 v97m11 -27 改善実績)
# ============================================================
echo "=== cycle 52: PuyoPresenceGate default 化 @ $(date) ==="
# load_default.py で use_puyo_gate=True を default 化
# ただし backup 必要
BACKUP_FILE=src/load_default.py.cycle50backup
[ ! -f "$BACKUP_FILE" ] && cp src/load_default.py "$BACKUP_FILE"

# 単純 sed で use_puyo_gate=False → True 探索
if grep -q "use_puyo_gate=False" src/load_default.py; then
  sed -i 's/use_puyo_gate=False/use_puyo_gate=True/' src/load_default.py
  AFTER_CRIT=$(eval_one_setup cycle52_puyo_gate)
  judge_and_log cycle52_puyo_gate "$BASELINE_CRIT" "$AFTER_CRIT"
  # revert
  cp "$BACKUP_FILE" src/load_default.py
fi

# ============================================================
# Cycle 53: STABLE_RECOVERY_SKIP_FRAMES sweep [12 (current), 6, 18]
# ============================================================
# これは seed 採取側 = 再抽出が必要 = 時間長い。 朝後回し。

# ============================================================
# Cycle 54: cnn_override_prob sweep [0.70 (current default), 0.60, 0.80]
# ============================================================
echo "=== cycle 54: cnn_override_prob sweep @ $(date) ==="
# こちらは load_default.py の cnn_override_prob を変更
# 各 prob で eval、 比較

finalize_health 0
