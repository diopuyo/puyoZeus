import numpy as np
import glob

SEG_RANGES = [
    ("seg01", 0.0, 893.7, "set1"),
    ("seg04", 2637.3, 3626.0, "set1"),
    ("seg05", 3626.0, 4379.5, "set2"),
    ("seg08", 6131.6, 7033.6, "set2"),
]
files = sorted(glob.glob("data/verify/zenchi_render_2026-08-21/seg*.npz"))
def load(name):
    f = [x for x in files if f"/{name}_" in x.replace("\\","/")][0]
    return np.load(f, allow_pickle=True)

for name, lo, hi, setname in SEG_RANGES:
    d = load(name)
    t, s1, s2 = d["t_sec"], d["score1"], d["score2"]
    mask = (t>=lo)&(t<hi)
    tt, ss1, ss2 = t[mask], s1[mask].astype(np.int64), s2[mask].astype(np.int64)
    d1 = np.diff(ss1); d2 = np.diff(ss2)
    # 大きなスコアジャンプ(でかい連鎖)を探す
    big1 = np.where(d1 > 5000)[0]
    big2 = np.where(d2 > 5000)[0]
    print(f"\n{name} ({setname}) t=[{lo},{hi}):")
    print("  1Pの大きいスコアジャンプ数:", len(big1), " 上位5件:")
    for i in big1[:5]:
        print(f"    t={tt[i+1]:.2f} score1 {ss1[i]}->{ss1[i+1]} (+{d1[i]})")
    print("  2Pの大きいスコアジャンプ数:", len(big2), " 上位5件:")
    for i in big2[:5]:
        print(f"    t={tt[i+1]:.2f} score2 {ss2[i]}->{ss2[i+1]} (+{d2[i]})")
