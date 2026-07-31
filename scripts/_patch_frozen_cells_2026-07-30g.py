import io
path = "scripts/_diag_frozen_cells_rate_2026-07-30.py"
with io.open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = None
for i, line in enumerate(lines):
    if "n_susp = int(df_col" in line and "score_suspicious" in line:
        start_idx = i
        break
assert start_idx is not None, "anchor line not found"

# 直後の2行 (print文2行、f-string連結) を置き換える対象として特定する。
assert "print(f" in lines[start_idx + 1]
assert lines[start_idx + 2].strip().startswith("f\"")
end_idx = start_idx + 2  # このindexまでが元のprint文2行

indent = "    "
new_lines = [
    lines[start_idx],
    indent + "n_unavail = int((~df_col[\"score_available\"]).sum())\n",
    indent + "n_censored_susp = int((df_col[\"score_suspicious\"]"
              " & df_col[\"right_censored\"]).sum())\n",
    indent + "n_confirmed = n_susp - n_censored_susp\n",
    indent + "print(f\"\n[列崩壊] 全体={n_total}件,"
              " スコアOCR欠測(判定不能)={n_unavail}件\")\n",
    indent + "print(f\"[列崩壊] スコア疑惑(フリーズ候補、右打ち切り含む)={n_susp}件"
              " ({100*n_susp/n_total:.1f}%)\")\n",
    indent + "print(f\"[列崩壊] うち同一試合内で回復確認済み"
              "(=より確度が高い候補)={n_confirmed}件\")\n",
]
lines[start_idx:end_idx + 1] = new_lines

with io.open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("patched summary via line-index ok")
