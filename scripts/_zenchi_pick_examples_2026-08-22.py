import pandas as pd
df = pd.read_csv("data/verify/judgment_scan_zenchi_2026-08-22/suspects.tsv", sep="\t")
blockers = df[df["stage"].isin(["display","both"])].copy()

targets_t = [200.47, 887.0, 4896.0, 3227.97, 16.5]
for tt in targets_t:
    sub = blockers.iloc[(blockers["t_sec"]-tt).abs().argsort()[:1]]
    for _, row in sub.iterrows():
        print(f"t={row['t_sec']:.3f} det={row['detector']} stage={row['stage']} game_idx={row['game_idx']}")
        print(f"  evidence: {row['evidence']}")
