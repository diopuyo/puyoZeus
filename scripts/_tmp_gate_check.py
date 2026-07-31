"""recovery gate 発火可能位置の empty_to_color corruption 詳細確認."""
import json
from pathlib import Path

COLOR_UNKNOWN = 10
COLOR_EMPTY = 0
BOARD_ROWS = 13
BOARD_COLS = 6
STABLE_WARMUP_FRAMES = 12

logs = [
    "data/verify/viz/v89_match01_D_2026-06-03.jsonl",
    "data/verify/viz/v89_match02_D_2026-06-03.jsonl",
    "data/verify/viz/v70_match02_formulaD_2026-06-02.jsonl",
]

all_samples = []

for log_path in logs:
    p = Path(log_path)
    if not p.exists():
        continue

    with open(p) as f:
        lines = f.readlines()
    rows = [json.loads(l) for l in lines]

    for side in ("p1", "p2"):
        blocks: list[list[int]] = []
        block: list[int] = []
        for i, row in enumerate(rows):
            state = row.get(f"{side}_state", "")
            if state == "stable":
                block.append(i)
            else:
                if block:
                    blocks.append(block)
                block = []
        if block:
            blocks.append(block)

        for blk in blocks:
            bl = len(blk)
            for pos_in_block, idx in enumerate(blk):
                row = rows[idx]
                from_end = bl - 1 - pos_in_block
                elapsed = pos_in_block

                # 条件: warmup 後 かつ 末尾 8fr 以前 (recovery gate が発火余裕あり)
                if elapsed <= STABLE_WARMUP_FRAMES or from_end < 8:
                    continue

                cnn_board = row.get(f"{side}_raw_cnn_board")
                hsv_board = row.get(f"{side}_raw_hsv_board")
                conf_board = row.get(f"{side}_confirmed")
                if cnn_board is None or hsv_board is None or conf_board is None:
                    continue

                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        c_v = cnn_board[r][c]
                        h_v = hsv_board[r][c]
                        cf_v = conf_board[r][c]
                        if c_v == COLOR_UNKNOWN or h_v == COLOR_UNKNOWN:
                            continue
                        if c_v != h_v:
                            continue
                        if cf_v == c_v:
                            continue
                        if cf_v != COLOR_EMPTY:
                            continue

                        # 下にぷよがあるか (浮きぷよ防止安全弁C)
                        below_empty = any(
                            conf_board[rr][c] == COLOR_EMPTY
                            for rr in range(r + 1, BOARD_ROWS)
                        )
                        all_samples.append({
                            "fname": p.name, "side": side,
                            "r": r, "c": c,
                            "target": c_v, "elapsed": elapsed, "from_end": from_end,
                            "t_sec": round(row["t_sec"], 2), "block_len": bl,
                            "below_empty": below_empty,
                        })

total = len(all_samples)
print(f"recovery gate 発火可能位置の empty_to_color corruption: {total} セル×フレーム")
print()
print("サンプル (最大20件):")
for s in all_samples[:20]:
    print(
        f"  {s['fname']} {s['side']} r={s['r']} c={s['c']} target={s['target']} "
        f"elapsed={s['elapsed']} from_end={s['from_end']} t={s['t_sec']}s "
        f"bl={s['block_len']} below_empty={s['below_empty']}"
    )

floating = sum(1 for s in all_samples if s["below_empty"])
non_floating = total - floating
print()
print("=== 安全弁C (浮きぷよ判定) ===")
print(f"  下に空あり (浮きぷよ → gate 弾かれる): {floating} ({100*floating/max(1,total):.1f}%)")
print(f"  下に空なし (gate 通過可能):            {non_floating} ({100*non_floating/max(1,total):.1f}%)")

# non_floating の中に残ること自体が recovery gate バグの証拠
print()
print(f"  → gate 通過可能なのに残る corruption: {non_floating} 件")
if non_floating > 0:
    print("    これらは recovery gate カウンタが 8fr 未達のまま STABLE 終了 → gate 未発火")
    print("    (elapsed は 12+ だが同じセルの corruption が連続 8fr 続いていない可能性)")
