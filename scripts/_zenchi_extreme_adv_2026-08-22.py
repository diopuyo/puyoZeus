import numpy as np
import glob
files = sorted(glob.glob("data/verify/zenchi_render_2026-08-21/seg*.npz"))
for f in files:
    d = np.load(f, allow_pickle=True)
    t, adv = d["t_sec"], d["adv_ema"]
    i_max = np.argmax(adv)
    i_min = np.argmin(adv)
    print(f"{f}: max_adv={adv[i_max]:.3f} at t={t[i_max]:.2f} | min_adv={adv[i_min]:.3f} at t={t[i_min]:.2f}")
