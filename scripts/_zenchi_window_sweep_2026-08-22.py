"""検収②: 3窓 (序盤/セット境界/終盤) の全時間帯数値突合。

各窓について:
  - adv_ema の時系列統計 (最小/最大/符号反転回数)
  - 連鎖中と推定される期間 (score1/score2 変化) での adv_ema 変動幅
  - drivers_top1 の符号と adv_ema の符号の食い違い件数 (表示側の内部整合性チェック、
    D0は raw 固定だがここでは表示値同士の整合性を別途確認する)
"""
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
    """指定した実時間範囲 [t_lo, t_hi) のレコードを、該当セグメントの本編区間内から集める。"""
    out_t, out_adv, out_d1name, out_d1val, out_score1, out_score2 = [], [], [], [], [], []
    for name, lo, hi, off in SEG_RANGES:
        # このセグメントの本編区間と要求区間の共通部分
        a, b = max(lo, t_lo), min(hi, t_hi)
        if a >= b:
            continue
        d = data[name]
        t = d["t_sec"]
        mask = (t >= a) & (t < b)
        out_t.append(t[mask])
        out_adv.append(d["adv_ema"][mask])
        out_d1name.append(d["drivers_top1_name"][mask])
        out_d1val.append(d["drivers_top1_val"][mask])
        out_score1.append(d["score1"][mask])
        out_score2.append(d["score2"][mask])
    if not out_t:
        return None
    t = np.concatenate(out_t)
    adv = np.concatenate(out_adv)
    d1name = np.concatenate(out_d1name)
    d1val = np.concatenate(out_d1val)
    s1 = np.concatenate(out_score1)
    s2 = np.concatenate(out_score2)
    order = np.argsort(t)
    return t[order], adv[order], d1name[order], d1val[order], s1[order], s2[order]


WINDOWS = {
    "A_序盤(game1-3付近)": (0.0, 280.0),
    "B_セット境界(game57-60付近)": (3260.0, 3700.0),
    "C_終盤(game114-116)": (6860.0, 7033.6),
}

for label, (lo, hi) in WINDOWS.items():
    res = slice_range(lo, hi)
    if res is None:
        print(f"{label}: データなし")
        continue
    t, adv, d1name, d1val, s1, s2 = res
    print(f"\n=== {label}  t=[{lo},{hi})  件数={len(t)} ===")
    print(f"  adv_ema: min={adv.min():.2f} max={adv.max():.2f} mean={adv.mean():.2f}")
    # 符号反転回数 (連続する非ゼロ値間で符号が変わった回数、EVEN帯 |adv|<3 は除外)
    nz = adv[np.abs(adv) >= 3.0]
    nz_t = t[np.abs(adv) >= 3.0]
    flips = 0
    flip_times = []
    for i in range(1, len(nz)):
        if np.sign(nz[i]) != np.sign(nz[i-1]):
            flips += 1
            flip_times.append((nz_t[i-1], nz_t[i], nz[i-1], nz[i]))
    print(f"  符号反転回数(|adv|>=3の実効反転): {flips}")
    for ft in flip_times[:10]:
        dt = ft[1]-ft[0]
        print(f"    t={ft[0]:.2f}->{ft[1]:.2f} (dt={dt:.2f}s) adv {ft[2]:.1f}->{ft[3]:.1f}")
    if len(flip_times) > 10:
        print(f"    ...他{len(flip_times)-10}件")

    # drivers_top1の符号とadv_emaの符号の不一致 (|adv|>=3のみ対象、注意書き通りraw/dispは別物だが
    # 表示側の内部一貫性の目安として集計)
    mismatch = 0
    mismatch_examples = []
    for i in range(len(t)):
        if abs(adv[i]) < 3.0 or d1val[i] == 0.0:
            continue
        if np.sign(adv[i]) != np.sign(d1val[i]):
            mismatch += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append((t[i], adv[i], str(d1name[i]), d1val[i]))
    print(f"  drivers_top1符号 vs adv_ema符号 不一致(参考値、D0はraw固定のため別物): {mismatch} / {len(t)}")
    for ex in mismatch_examples:
        print(f"    t={ex[0]:.2f} adv={ex[1]:.1f} driver={ex[2]}={ex[3]:.2f}")

    # スコア変化(連鎖疑い)区間中のadv変動幅
    score_delta = np.abs(np.diff(s1.astype(np.int64))) + np.abs(np.diff(s2.astype(np.int64)))
    chain_mask = score_delta > 0
    if chain_mask.any():
        chain_idx = np.where(chain_mask)[0]
        print(f"  スコア変化検出フレーム数(連鎖進行中の目安): {len(chain_idx)}")
