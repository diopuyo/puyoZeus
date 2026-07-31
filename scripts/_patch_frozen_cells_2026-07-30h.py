import io
BS = chr(92)
NL_ESCAPE = BS + "n"  # 実際のバックスラッシュ+n (改行エスケープ表現そのもの)

path = "scripts/_diag_frozen_cells_rate_2026-07-30.py"
with io.open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = None
for i, line in enumerate(lines):
    if "n_susp = int(df_col" in line and "score_suspicious" in line:
        start_idx = i
        break
assert start_idx is not None, "anchor not found"

# 壊れた行(直前パッチの失敗分)を探して除去範囲を決める:
# n_susp行の次から、次のfor axis行の手前までを丸ごと置換する。
end_idx = start_idx + 1
while "for axis in" not in lines[end_idx]:
    end_idx += 1
# end_idx は for axis 行そのもの (置換範囲はここの手前まで)

indent = "    "
quote = chr(34)  # "

def q(s: str) -> str:
    return quote + s + quote

new_block = [
    lines[start_idx],
    indent + "n_unavail = int((~df_col[" + q("score_available") + "]).sum())\n",
    indent + "n_censored_susp = int((df_col[" + q("score_suspicious") + "] & "
              "df_col[" + q("right_censored") + "]).sum())\n",
    indent + "n_confirmed = n_susp - n_censored_susp\n",
    indent + "print(f" + q(NL_ESCAPE + "[列崩壊] 全体={n_total}件, "
              "スコアOCR欠測(判定不能)={n_unavail}件") + ")\n",
    indent + "print(f" + q("[列崩壊] スコア疑惑(フリーズ候補、右打ち切り含む)={n_susp}件"
              " ({100*n_susp/n_total:.1f}%)") + ")\n",
    indent + "print(f" + q("[列崩壊] うち同一試合内で回復確認済み(=より確度が高い候補)"
              "={n_confirmed}件") + ")\n",
]
lines[start_idx:end_idx] = new_block

with io.open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print("patched summary via chr(92) technique ok")
