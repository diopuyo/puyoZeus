"""color_to_color 長期凍結ケースの時系列確認."""
import json
from pathlib import Path

COLOR_UNKNOWN = 10
COLOR_EMPTY = 0
BOARD_ROWS = 13
BOARD_COLS = 6

# 長期凍結ケース:
# side=p2 r=5 c=4 fc=16 conf=1->target=3 start=20.8s resolved=True (v70_match02_formulaD)
# side=p1 r=11 c=1 fc=16 conf=2->target=5 start=53.5s resolved=True (v70_match02_formulaD)
# side=p2 r=7 c=1 fc=16 conf=3->target=5 start=44.1s resolved=False state_change (v70_match02)

print("=== v70_match02_formulaD p2 r=5 c=4 (t=20~22s) ===")
fname = "data/verify/viz/v70_match02_formulaD_2026-06-02.jsonl"
p = Path(fname)
with open(p) as f:
    rows = [json.loads(l) for l in f]

for row in rows:
    t = row["t_sec"]
    if not (20.0 <= t <= 22.5):
        continue
    state = row.get("p2_state", "")
    cnn = row.get("p2_raw_cnn_board")
    hsv = row.get("p2_raw_hsv_board")
    conf = row.get("p2_confirmed")
    if cnn is None or hsv is None or conf is None:
        continue
    r, c = 5, 4
    cv = cnn[r][c]
    hv = hsv[r][c]
    cfv = conf[r][c]
    corr_mark = ""
    if cv == hv and cv != COLOR_UNKNOWN and cfv != cv:
        corr_mark = f" <<CORR (conf={cfv})"
    print(f"  t={t:.2f} state={state:12s} cnn={cv} hsv={hv} conf={cfv}{corr_mark}")

print()
print("=== v70_match02_formulaD p2 r=7 c=1 (t=43~46s) ===")
for row in rows:
    t = row["t_sec"]
    if not (43.0 <= t <= 46.0):
        continue
    state = row.get("p2_state", "")
    cnn = row.get("p2_raw_cnn_board")
    hsv = row.get("p2_raw_hsv_board")
    conf = row.get("p2_confirmed")
    if cnn is None or hsv is None or conf is None:
        continue
    r, c = 7, 1
    cv = cnn[r][c]
    hv = hsv[r][c]
    cfv = conf[r][c]
    corr_mark = ""
    if cv == hv and cv != COLOR_UNKNOWN and cfv != cv:
        corr_mark = f" <<CORR (conf={cfv})"
    print(f"  t={t:.2f} state={state:12s} cnn={cv} hsv={hv} conf={cfv}{corr_mark}")

print()
print("=== v70_match02_formulaD p1 r=2 c=0/c=1 (t=43.5~46s) ===")
for row in rows:
    t = row["t_sec"]
    if not (43.5 <= t <= 46.5):
        continue
    state = row.get("p1_state", "")
    cnn = row.get("p1_raw_cnn_board")
    hsv = row.get("p1_raw_hsv_board")
    conf = row.get("p1_confirmed")
    if cnn is None or hsv is None or conf is None:
        continue
    for c in [0, 1]:
        r = 2
        cv = cnn[r][c]
        hv = hsv[r][c]
        cfv = conf[r][c]
        corr_mark = ""
        if cv == hv and cv != COLOR_UNKNOWN and cfv != cv:
            corr_mark = f" <<CORR(conf={cfv})"
        print(f"  t={t:.2f} state={state:12s} r=2c={c}: cnn={cv} hsv={hv} conf={cfv}{corr_mark}")
