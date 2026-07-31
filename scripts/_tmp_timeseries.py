"""gate 通過可能セルの時系列確認."""
import json
from pathlib import Path

COLOR_UNKNOWN = 10
COLOR_EMPTY = 0
BOARD_ROWS = 13
BOARD_COLS = 6

print("=== v89m02 p1 r=3 c=3 の時系列 (t=27.5~29.5s) ===")
fname = "data/verify/viz/v89_match02_D_2026-06-03.jsonl"
p = Path(fname)
with open(p) as f:
    rows = [json.loads(l) for l in f]

for row in rows:
    t = row["t_sec"]
    if not (27.5 <= t <= 29.5):
        continue
    state = row.get("p1_state", "")
    cnn = row.get("p1_raw_cnn_board")
    hsv = row.get("p1_raw_hsv_board")
    conf = row.get("p1_confirmed")
    if cnn is None or hsv is None or conf is None:
        continue
    r, c = 3, 3
    cv = cnn[r][c]
    hv = hsv[r][c]
    cfv = conf[r][c]
    corr_mark = " <<CORR" if (cv == hv and cv != cfv and cfv == COLOR_EMPTY) else ""
    print(
        f"  t={t:.2f} state={state:12s} cnn={cv} hsv={hv} conf={cfv}{corr_mark}"
    )

print()
print("=== v89m01 p1 r=1 c=1 の時系列 (t=30.6~31.3s) ===")
fname2 = "data/verify/viz/v89_match01_D_2026-06-03.jsonl"
p2 = Path(fname2)
with open(p2) as f:
    rows2 = [json.loads(l) for l in f]

for row in rows2:
    t = row["t_sec"]
    if not (30.6 <= t <= 31.3):
        continue
    state = row.get("p1_state", "")
    cnn = row.get("p1_raw_cnn_board")
    hsv = row.get("p1_raw_hsv_board")
    conf = row.get("p1_confirmed")
    if cnn is None or hsv is None or conf is None:
        continue
    r, c = 1, 1
    cv = cnn[r][c]
    hv = hsv[r][c]
    cfv = conf[r][c]
    corr_mark = " <<CORR" if (cv == hv and cv != cfv and cfv == COLOR_EMPTY) else ""
    print(
        f"  t={t:.2f} state={state:12s} cnn={cv} hsv={hv} conf={cfv}{corr_mark}"
    )

# 追加: r=1 c=2 (elapsed=27 from_end=24 の浮きぷよケース)
print()
print("=== v89m01 p1 r=1 c=2 の時系列 (t=5.0~5.9s) ===")
for row in rows2:
    t = row["t_sec"]
    if not (4.5 <= t <= 6.5):
        continue
    state = row.get("p1_state", "")
    cnn = row.get("p1_raw_cnn_board")
    hsv = row.get("p1_raw_hsv_board")
    conf = row.get("p1_confirmed")
    if cnn is None or hsv is None or conf is None:
        continue
    r, c = 1, 2
    cv = cnn[r][c]
    hv = hsv[r][c]
    cfv = conf[r][c]
    corr_mark = " <<CORR" if (cv == hv and cv != cfv and cfv == COLOR_EMPTY) else ""
    # c=5 も並記 (上段)
    cv5 = cnn[r][5]
    hv5 = hsv[r][5]
    cfv5 = conf[r][5]
    print(
        f"  t={t:.2f} state={state:12s} "
        f"r=1c=2: cnn={cv} hsv={hv} conf={cfv}{corr_mark} | "
        f"r=1c=5: cnn={cv5} hsv={hv5} conf={cfv5}"
    )
