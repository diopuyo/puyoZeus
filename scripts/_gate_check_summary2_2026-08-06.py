import json
from pathlib import Path

data = json.loads(Path("data/verify/burst_guard_2026-08-05/_gate_check_2026-08-06_result.json").read_text())

def collect(metric_key):
    on_all, am_all, af_all = [], [], []
    worst = None
    for r in data:
        for side in ("1P", "2P"):
            s = r["sides"][side]
            on_rates = s[f"{metric_key}_on"]
            am_rates = s[f"{metric_key}_anchor_matched"]
            af_rates = s[f"{metric_key}_anchor_full"]
            for c in range(6):
                v_on = on_rates[str(c)] if str(c) in on_rates else on_rates.get(c)
                v_am = am_rates[str(c)] if str(c) in am_rates else am_rates.get(c)
                v_af = af_rates[str(c)] if str(c) in af_rates else af_rates.get(c)
                if v_on == v_on:
                    on_all.append(v_on)
                    if worst is None or v_on > worst[0]:
                        worst = (v_on, r["video"], side, c)
                if v_am == v_am:
                    am_all.append(v_am)
                if v_af == v_af:
                    af_all.append(v_af)
    return on_all, am_all, af_all, worst

import statistics as st
for key, label in [("per_col_unknown_rate", "per_col_unknown_rate"), ("per_col_midgame_empty_rate", "per_col_midgame_empty_rate")]:
    on_all, am_all, af_all, worst = collect(key)
    print(f"=== {label} ===")
    print(f"  ON:            mean={st.mean(on_all)*100:.3f}%  max={max(on_all)*100:.3f}%  n={len(on_all)}")
    print(f"  anchor_matched:mean={st.mean(am_all)*100:.3f}%  max={max(am_all)*100:.3f}%  n={len(am_all)}")
    print(f"  anchor_full:   mean={st.mean(af_all)*100:.3f}%  max={max(af_all)*100:.3f}%  n={len(af_all)}")
    print(f"  worst ON cell-col: {worst}")
    print()
