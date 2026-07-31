import io
path = "scripts/_diag_frozen_cells_rate_2026-07-30.py"
with io.open(path, "r", encoding="utf-8") as f:
    text = f.read()

old_ev = """            score_suspicious=bool(row.score_suspicious),
            t_recovery=row.t_recovery if pd.notna(row.t_recovery) else None,"""
new_ev = """            score_suspicious=bool(row.score_suspicious),
            score_available=bool(row.score_available),
            t_recovery=row.t_recovery if pd.notna(row.t_recovery) else None,"""
assert old_ev in text
text = text.replace(old_ev, new_ev, 1)

# 集計関数にもscore_available別の内訳を追加 (indeterminateバケットを可視化)
old_summary = """    n_total = len(df_col)
    n_susp = int(df_col["score_suspicious"].sum())
    print(f"\n[列崩壊] 全体={n_total}件, スコア疑惑(フリーズ候補)={n_susp}件"
          f" ({100*n_susp/n_total:.1f}%)")"""
new_summary = """    n_total = len(df_col)
    n_susp = int(df_col["score_suspicious"].sum())
    n_unavail = int((~df_col["score_available"]).sum())
    n_censored_susp = int((df_col["score_suspicious"] & df_col["right_censored"]).sum())
    n_confirmed = n_susp - n_censored_susp
    print(f"\n[列崩壊] 全体={n_total}件, スコアOCR欠測(判定不能)={n_unavail}件")
    print(f"[列崩壊] スコア疑惑(フリーズ候補、右打ち切り含む)={n_susp}件 ({100*n_susp/n_total:.1f}%)")
    print(f"[列崩壊] うち同一試合内で回復確認済み(=より確度が高い候補)={n_confirmed}件")"""
assert old_summary in text
text = text.replace(old_summary, new_summary, 1)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("patched sample_frames + summary ok")
