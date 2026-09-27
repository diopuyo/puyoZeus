"""R4と同じ暖機・音声合成でE14レビュー動画を生成する。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time

from scripts.run_e3_exchange_eval_20260926 import save_json, command, SOURCES, digest

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "logs/review_zenchi_g41_43_e14"
PREVIOUS = ROOT / "logs/review_zenchi_g41_43_e13"
START, END = 2579.066, 2777.333
SHA_ROOT = ROOT / "logs/e14/sha"
ARTIFACTS = ("overlay.mp4", "display.npz", "settled.npz")


def verify() -> None:
    """動画とCSVのフレーム数、AAC音声を確認する。"""
    import cv2
    import imageio_ffmpeg
    capture = cv2.VideoCapture(str(DEST / "overlay.mp4"))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    with (DEST / "review_data.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert frames == len(rows) and fps > 0
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(DEST / "overlay.mp4")],
                           capture_output=True, text=True)
    assert "Audio: aac" in probe.stderr
    assert any(row["1P_NF_ojama_k1"] for row in rows)
    save_json(DEST / "complete.json", dict(frames=frames, fps=fps, duration_seconds=frames/fps,
        audio="aac", csv_rows=len(rows), first_sec=rows[0]["t_sec"], last_sec=rows[-1]["t_sec"]))


def render() -> None:
    """既存R4コマンドの出力先とモデル指定だけを置換する。"""
    import imageio_ffmpeg
    DEST.mkdir(parents=True, exist_ok=True)
    previous = json.loads((PREVIOUS / "status.json").read_text())
    cmd = [part.replace(str(PREVIOUS), str(DEST)) for part in previous["command"]]
    cmd += ["--exchange-event-model-dir", "models/exchange_event_v2"]
    video = cmd[cmd.index("--video")+1]
    state = dict(command=cmd, state="running", started=time.time())
    save_json(DEST / "status.json", state)
    with (DEST / "render.log").open("w") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i",
            str(DEST / "overlay_video_only.mp4"), "-ss", str(START), "-t", str(END-START),
            "-i", video, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-shortest", str(DEST / "overlay.mp4")], stdout=log, stderr=subprocess.STDOUT, check=True)
    verify()
    state.update(state="completed", elapsed_seconds=time.time()-state["started"])
    save_json(DEST / "status.json", state)


def sha_check() -> None:
    """E13保存済み25秒区間とOFF・v1既定ONの全成果物を照合する。"""
    results = []
    for mode in ("off", "on"):
        dest = SHA_ROOT / mode
        dest.mkdir(parents=True, exist_ok=True)
        cmd = command(SOURCES[0], mode, dest, 25) + ["--start-sec", "140"]
        with (dest / "render.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        before = ROOT / f"logs/e13/sha/{mode}_after"
        hashes = {version: {name: digest(path/name) for name in ARTIFACTS}
                  for version, path in (("before", before), ("after", dest))}
        result = dict(mode=mode, start_sec=140, duration_sec=25, **hashes,
                      identical=hashes["before"] == hashes["after"])
        save_json(SHA_ROOT / f"{mode}_sha256.json", result)
        results.append(result)
    save_json(SHA_ROOT / "summary.json", results)
    assert all(row["identical"] for row in results)


if __name__ == "__main__":
    sha_check()
    render()
