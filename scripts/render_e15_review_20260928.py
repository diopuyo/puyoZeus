"""E14の条件を継承してE15動画・音声付きスマホ版を生成する。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "logs/review_zenchi_g41_43_e14"
DEST = ROOT / "logs/review_zenchi_g41_43_e15"
MOBILE = Path("/mnt/d/puyo_analyzer/videos/review/zenchi_g41-43_e15_mobile.mp4")
WIDTH, CRF, LIMIT_BYTES, SEGMENT_SEC = 1280, 28, 30_000_000, 200
# 200秒×映像1000kbps＋音声128kbpsを約28.2MBに収める。
VIDEO_KBPS, AUDIO_KBPS, BUFFER_KBPS = 1000, 128, 2000


def save(name: str, value: dict) -> None:
    """再開判断に使う処理段階を記録する。"""
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def verify(path: Path) -> dict:
    """映像フレーム数と音声ストリームを検査する。"""
    import cv2
    cap = cv2.VideoCapture(str(path))
    result = dict(path=str(path), bytes=path.stat().st_size,
                  frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS),
                  width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)],
                           capture_output=True, text=True)
    assert "Audio: aac" in probe.stderr and result["frames"] > 0
    result["audio"] = "aac"
    result["seconds"] = result["frames"] / result["fps"]
    return result


def mobile(log: object) -> list[dict]:
    """横1280・CRF28で圧縮し、上限超過時は200秒ごとに分割する。"""
    MOBILE.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-i", str(DEST / "overlay.mp4"), "-vf", f"scale={WIDTH}:-2",
        "-c:v", "libx264", "-crf", str(CRF), "-preset", "medium", "-c:a", "aac",
        "-movflags", "+faststart", str(MOBILE)], stdout=log, stderr=subprocess.STDOUT, check=True)
    paths = [MOBILE]
    if MOBILE.stat().st_size > LIMIT_BYTES:
        pattern = MOBILE.with_name(MOBILE.stem + "_%03d.mp4")
        # 短い区間は分割だけでは小さくならない。元のレビューからCRF28＋VBVで再圧縮する。
        subprocess.run([ffmpeg, "-y", "-i", str(DEST / "overlay.mp4"), "-vf", f"scale={WIDTH}:-2",
            "-c:v", "libx264", "-crf", str(CRF), "-preset", "medium",
            "-maxrate", f"{VIDEO_KBPS}k", "-bufsize", f"{BUFFER_KBPS}k",
            "-force_key_frames", f"expr:gte(t,n_forced*{SEGMENT_SEC})",
            "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k", "-f", "segment",
            "-segment_time", str(SEGMENT_SEC), "-reset_timestamps", "1",
            "-segment_format_options", "movflags=+faststart", str(pattern)],
            stdout=log, stderr=subprocess.STDOUT, check=True)
        paths = sorted(MOBILE.parent.glob(MOBILE.stem + "_*.mp4"))
    results = [verify(path) for path in paths]
    assert all(r["bytes"] <= LIMIT_BYTES and r["width"] == WIDTH for r in results)
    if MOBILE not in paths:
        MOBILE.unlink()  # 自分が生成した容量超過版だけを完成確認後に除く。
    return results


def main() -> None:
    """現HEADの固定縮尺を使用し、認識条件はE14の保存コマンドを継承する。"""
    previous = json.loads((PREVIOUS / "status.json").read_text())
    cmd = [p.replace(str(PREVIOUS), str(DEST)) for p in previous["command"]]
    cmd[cmd.index("--exchange-event-model-dir") + 1] = "models/exchange_event_v3"
    cmd.append("--exchange-event-live-count")
    start, end = (float(cmd[cmd.index(flag)+1]) for flag in ("--start-sec", "--end-sec"))
    state = dict(command=cmd, state="running", started=time.time())
    save("status.json", state)
    with (DEST / "render.log").open("a") as log:
        if not (DEST / "overlay.mp4").exists():
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i",
                str(DEST / "overlay_video_only.mp4"), "-ss", str(start), "-t", str(end-start),
                "-i", cmd[cmd.index("--video")+1], "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-shortest", str(DEST / "overlay.mp4")],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        full = verify(DEST / "overlay.mp4")
        with (DEST / "review_data.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == full["frames"] and any(r["1P_NF_ojama_k1"] for r in rows)
        save("complete.json", dict(full=full, csv_rows=len(rows), mobile=mobile(log),
            mobile_encoding=dict(width=WIDTH, crf=CRF, segment_sec=SEGMENT_SEC,
                                 maxrate_kbps=VIDEO_KBPS, audio_kbps=AUDIO_KBPS)))
    state.update(state="completed", elapsed_seconds=time.time()-state["started"])
    save("status.json", state)


if __name__ == "__main__":
    main()
