"""E18: E14と同じ条件でE15＋②のレビュー動画を先に生成する。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time
import imageio_ffmpeg
from scripts import render_e15_review_20260928 as prior

DEST = prior.ROOT / "logs/review_zenchi_g41_43_e15b"
MOBILE = Path("/mnt/d/puyo_analyzer/videos/review/zenchi_g41-43_e15b_mobile.mp4")


def mobile(log: object) -> list[dict]:
    """最初から映像1Mbps上限を指定し、30MB以内を検証する。"""
    MOBILE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(DEST/"overlay.mp4"),
        "-vf", f"scale={prior.WIDTH}:-2", "-c:v", "libx264", "-crf", str(prior.CRF),
        "-preset", "medium", "-maxrate", f"{prior.VIDEO_KBPS}k",
        "-bufsize", f"{prior.BUFFER_KBPS}k", "-c:a", "aac", "-b:a", f"{prior.AUDIO_KBPS}k",
        "-movflags", "+faststart", str(MOBILE)], stdout=log, stderr=subprocess.STDOUT, check=True)
    result = prior.verify(MOBILE)
    assert result["bytes"] <= prior.LIMIT_BYTES and result["width"] == prior.WIDTH
    return [result]


def main() -> None:
    """固定コマンド・CSV・音声・圧縮設定を完了記録へ残す。"""
    prior.DEST = DEST
    previous = json.loads((prior.PREVIOUS/"status.json").read_text())
    cmd = [p.replace(str(prior.PREVIOUS), str(DEST)) for p in previous["command"]]
    cmd[cmd.index("--exchange-event-model-dir")+1] = "models/exchange_event_v3"
    cmd.extend(["--exchange-event-live-count", "--exchange-event-death-guard"])
    start, end = (float(cmd[cmd.index(f)+1]) for f in ("--start-sec", "--end-sec"))
    state = dict(command=cmd, state="running", started=time.time())
    prior.save("status.json", state)
    with (DEST/"render.log").open("a") as log:
        if not (DEST/"overlay.mp4").exists():
            subprocess.run(cmd, cwd=prior.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i",
                str(DEST/"overlay_video_only.mp4"), "-ss", str(start), "-t", str(end-start),
                "-i", cmd[cmd.index("--video")+1], "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-shortest", str(DEST/"overlay.mp4")],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        full = prior.verify(DEST/"overlay.mp4")
        with (DEST/"review_data.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == full["frames"] and any(r["1P_NF_ojama_k1"] for r in rows)
        prior.save("complete.json", dict(full=full, csv_rows=len(rows), mobile=mobile(log),
            mobile_encoding=dict(width=prior.WIDTH, crf=prior.CRF,
                                 maxrate_kbps=prior.VIDEO_KBPS, audio_kbps=prior.AUDIO_KBPS)))
    state.update(state="completed", elapsed_seconds=time.time()-state["started"])
    prior.save("status.json", state)


if __name__ == "__main__":
    main()
