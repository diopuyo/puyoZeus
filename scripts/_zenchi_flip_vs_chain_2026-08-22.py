import numpy as np
import glob

SEG_RANGES = [
    ("seg01", 0.0, 893.7, 0),
    ("seg02", 893.7, 1738.3, 12),
    ("seg03", 1738.3, 2637.3, 26),
    ("seg04", 2637.3, 3626.0, 41),
    ("seg05", 3626.0, 4379.5, 59),
    ("seg06", 4379.5, 5255.6, 70),
    ("seg07", 5255.6, 6131.6, 85),
    ("seg08", 6131.6, 7033.6, 100),
]
files = sorted(glob.glob("data/verify/zenchi_render_2026-08-21/seg*.npz"))
def load(name):
    f = [x for x in files if f"/{name}_" in x.replace("\\","/")][0]
    return np.load(f, allow_pickle=True)
data = {name: load(name) for name,_,_,_ in SEG_RANGES}

def slice_range(t_lo, t_hi):
    out=[]
    for name, lo, hi, off in SEG_RANGES:
        a, b = max(lo, t_lo), min(hi, t_hi)
        if a >= b: continue
        d = data[name]
        t = d["t_sec"]
        mask = (t >= a) & (t < b)
        out.append((t[mask], d["adv_ema"][mask], d["score1"][mask], d["score2"][mask], d["game_idx"][mask]+off))
    ts = np.concatenate([o[0] for o in out])
    adv = np.concatenate([o[1] for o in out])
    s1 = np.concatenate([o[2] for o in out])
    s2 = np.concatenate([o[3] for o in out])
    g = np.concatenate([o[4] for o in out])
    order = np.argsort(ts)
    return ts[order], adv[order], s1[order], s2[order], g[order]

WINDOWS = {
    "A": (0.0, 280.0),
    "B": (3260.0, 3700.0),
    "C": (6860.0, 7033.6),
}
# 試合境界 (前段で求めた正規化済みリストを再利用: t, gfrom, gto)
BOUNDARIES = [86.67,141.07,232.47,278.10,335.97,416.47,487.87,514.23,609.63,695.27,751.23,828.37,
              958.90,985.90,1045.13,1097.67,1174.63,1235.77,1300.07,1356.87,1396.90,1432.20,1505.60,
              1582.03,1625.27,1798.53,1879.03,1918.40,1983.23,2034.13,2125.17,2172.07,2219.40,2275.87,
              2351.27,2393.20,2453.73,2544.43,2579.07,2706.17,2782.20,2812.30,2870.97,2898.23,2948.70,
              2988.57,3025.40,3051.70,3122.87,3170.83,3239.13,3273.13,3330.73,3356.20,3417.13,3625.80,
              3673.23,3749.57,3806.73,3894.00,3998.20,4040.60,4084.70,4142.83,4218.57,4293.20,4453.07,
              4550.37,4623.63,4665.27,4730.03,4767.40,4818.43,4871.23,4924.03,4981.33,5026.87,5061.83,
              5132.33,5194.47,5288.10,5366.17,5425.20,5487.00,5530.33,5606.13,5646.23,5744.97,5788.73,
              5832.47,5898.47,5935.73,6005.83,6071.53,6204.73,6281.70,6404.27,6456.20,6522.50,6567.83,
              6631.80,6664.17,6729.27,6755.53,6788.90,6832.90,6874.57,6936.60,6972.97]
BOUNDARIES = np.array(BOUNDARIES)

for label, (lo, hi) in WINDOWS.items():
    t, adv, s1, s2, g = slice_range(lo, hi)
    nz_mask = np.abs(adv) >= 3.0
    nz_t, nz_adv = t[nz_mask], adv[nz_mask]
    # score変化フラグ (直近0.5秒以内にscoreが変わっていれば「連鎖中」とみなす簡易判定)
    print(f"\n=== 窓{label} ===")
    flip_count_near_chain = 0
    flip_count_far_from_chain = 0
    flip_count_near_boundary = 0
    for i in range(1, len(nz_adv)):
        if np.sign(nz_adv[i]) == np.sign(nz_adv[i-1]):
            continue
        tt = nz_t[i]
        # 境界近傍か (±2秒)
        near_b = np.any(np.abs(BOUNDARIES - tt) <= 2.0)
        # 直近0.5秒以内にスコア変化があったか
        win_mask = (t >= tt - 0.5) & (t <= tt)
        s1w, s2w = s1[win_mask], s2[win_mask]
        score_changed = (len(s1w) > 1) and ((s1w.max()!=s1w.min()) or (s2w.max()!=s2w.min()))
        if near_b:
            flip_count_near_boundary += 1
        elif score_changed:
            flip_count_near_chain += 1
        else:
            flip_count_far_from_chain += 1
    print(f"  境界近傍(±2s)の符号反転: {flip_count_near_boundary}")
    print(f"  連鎖中(直近0.5s以内score変化)の符号反転: {flip_count_near_chain}")
    print(f"  それ以外(通常時)の符号反転: {flip_count_far_from_chain}")
