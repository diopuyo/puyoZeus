import io
BS = chr(92)
quote = chr(34)

path = "scripts/_diag_frozen_cells_rate_2026-07-30.py"
with io.open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 1) suspects の絞り込みを「同一試合内で回復確認済み」に限定する
#    (右打ち切りは試合終了アーティファクトの疑いが強く証拠として弱いため)
for i, line in enumerate(lines):
    if "suspects = df_col[df_col" in line:
        indent = "    "
        lines[i] = (
            indent + "suspects = df_col[df_col[" + quote + "score_suspicious" + quote
            + "] & ~df_col[" + quote + "right_censored" + quote + "]]"
            + "  # 回復確認済みのみ(証拠が強い候補)\n"
        )
        break
else:
    raise SystemExit("suspects line not found")

# 2) ColumnCollapseEvent 再構築に score_available を追加
for i, line in enumerate(lines):
    if "score_suspicious=bool(row.score_suspicious)," in line:
        insert_indent = line[: len(line) - len(line.lstrip())]
        lines.insert(i + 1, insert_indent + "score_available=bool(row.score_available),\n")
        break
else:
    raise SystemExit("ev construction line not found")

with io.open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("patched frame sampling ok")
