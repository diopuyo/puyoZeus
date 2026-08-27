"""検収⑤: 音声/映像ずれ確認 (読み取り専用)。

スコアOCRの急上昇(=連鎖発火)時刻を基準に、その前後の音声振幅エンベロープの
立ち上がり(オンセット)時刻を求め、視覚イベントとの時間差を数値化する。
"""
import subprocess
import wave
from pathlib import Path

import numpy as np

FF = "venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SRC = Path("data/verify/zenchi_delivery_2026-08-21")
OUT_DIR = Path("data/verify/_audio_sync_2026-08-22")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SET_BOUNDARY = 3626.0

# (global_t_sec, label)  score OCR上の急上昇(連鎖発火)を基準時刻とする
EVENTS = [
    (200.30, "set1_ev1"),
    (2765.93, "set1_ev2"),
    (3109.10, "set1_ev3"),
    (3734.60, "set2_ev1"),
    (3981.77, "set2_ev2"),
    (6199.13, "set2_ev3"),
]

WINDOW_BEFORE = 3.0
WINDOW_AFTER = 3.0


def extract_clip(video: str, t_center: float, out_wav: Path, out_png: Path) -> None:
    src = SRC / video
    t0 = max(0.0, t_center - WINDOW_BEFORE)
    dur = WINDOW_BEFORE + WINDOW_AFTER
    subprocess.run([FF, "-y", "-ss", str(t0), "-i", str(src), "-t", str(dur),
                     "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", str(out_wav)],
                   check=True, capture_output=True)
    subprocess.run([FF, "-y", "-ss", str(t_center), "-i", str(src), "-frames:v", "1", str(out_png)],
                   check=True, capture_output=True)


def envelope_onset(wav_path: Path, t0_offset: float) -> tuple[np.ndarray, np.ndarray]:
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    win = int(sr * 0.02)  # 20ms窓
    n_win = len(data) // win
    rms = np.array([np.sqrt(np.mean(data[i*win:(i+1)*win]**2)) for i in range(n_win)])
    t = t0_offset + np.arange(n_win) * (win / sr)
    return t, rms


for t_global, label in EVENTS:
    if t_global < SET_BOUNDARY:
        video = "zenchi_set1_audio.mp4"
        t_local = t_global
    else:
        video = "zenchi_set2_audio.mp4"
        t_local = t_global - SET_BOUNDARY
    out_wav = OUT_DIR / f"{label}.wav"
    out_png = OUT_DIR / f"{label}_frame.png"
    extract_clip(video, t_local, out_wav, out_png)
    t0_local_start = max(0.0, t_local - WINDOW_BEFORE)
    t, rms = envelope_onset(out_wav, t0_local_start)
    # イベント基準時刻(=t_local)前後での最大RMSの時刻を探す
    idx_peak = int(np.argmax(rms))
    t_peak = t[idx_peak]
    # ベースライン(イベント3秒以上前の平均RMS)に対する閾値超えの最初の時刻(オンセット)
    baseline = rms[: max(1, int(len(rms) * 0.2))].mean()
    thresh = baseline * 3.0 + 50.0
    onset_idx = np.argmax(rms > thresh) if (rms > thresh).any() else None
    t_onset = t[onset_idx] if onset_idx is not None else None
    print(f"{label}: t_event(local)={t_local:.2f} peak_rms_t={t_peak:.2f} "
          f"(offset={t_peak - t_local:+.2f}s) onset_t={t_onset} "
          f"(offset={(t_onset - t_local) if t_onset is not None else None})")
