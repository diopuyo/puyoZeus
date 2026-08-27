import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("data/verify/_audio_sync_2026-08-22")
EVENTS = [
    ("set1_ev1", 200.30, 3.0),
    ("set1_ev2", 2765.93, 3.0),
    ("set1_ev3", 3109.10, 3.0),
    ("set2_ev1", 108.60, 3.0),
    ("set2_ev2", 355.77, 3.0),
    ("set2_ev3", 2573.13, 3.0),
]
for label, t_local, before in EVENTS:
    wav = OUT_DIR / f"{label}.wav"
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    t0 = max(0.0, t_local - before)
    tt = t0 + np.arange(len(data)) / sr
    plt.figure(figsize=(10, 2.5))
    plt.plot(tt, data, linewidth=0.3)
    plt.axvline(t_local, color="red", linestyle="--", label="score急上昇時刻(視覚イベント基準)")
    plt.title(f"{label}  波形(縦線=スコア急上昇時刻)")
    plt.xlabel("動画内ローカル秒")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{label}_waveform.png", dpi=110)
    plt.close()
print("done")
