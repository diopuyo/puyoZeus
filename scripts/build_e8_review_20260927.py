"""qの全900秒を左右同期比較し、音声と撃ち合い一覧を納品する。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.run_e3_exchange_eval_20260926 import digest, save_json
from scripts.visualize_advantage_overlay import _winprob_to_adv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/e8"
SOURCE = "q_7gc4TgFig"
ON = OUT / "renders" / SOURCE / "on"
OFF = ROOT / "logs/e3/renders" / SOURCE / "off"
ORIGINAL = OUT / "source_audio" / f"{SOURCE}.m4a"
REVIEW = Path("/mnt/d/puyo_analyzer/videos/review")
NAME = f"{SOURCE}_exchange_event_offon_20260927_v1"
SECONDS, FPS = 900, 30
SIDE_WIDTH, SIDE_HEIGHT = 1280, 1110
HEADER, FOOTER = 48, 40
WIDTH, HEIGHT = SIDE_WIDTH * 2, SIDE_HEIGHT + HEADER + FOOTER
FONT = Path("/mnt/c/Windows/Fonts/meiryob.ttc")
LABEL_SIZE, CLOCK_SIZE = 28, 24
NODE = "/mnt/c/Users/ryouj/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
MARKER = ("C:/Users/ryouj/.codex/plugins/cache/openai-primary-runtime/spreadsheets/"
          "26.909.12148/skills/spreadsheets/container_tools/mark_artifact_operation_started.mjs")


def video_info(path: Path) -> dict:
    """映像の寸法・フレーム数・長さを読む。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けない: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = (int(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return dict(fps=fps, frames=frames, seconds=frames / fps, width=width, height=height)


def labels() -> Path:
    """小さな日本語ラベルと毎秒の時計帯を、読みやすい位置へ固定する。"""
    directory = OUT / "comparison_labels"
    directory.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT), LABEL_SIZE)
    header = Image.new("RGB", (WIDTH, HEADER), "#151a22")
    draw = ImageDraw.Draw(header)
    for x, label in ((SIDE_WIDTH // 2, "現行"), (SIDE_WIDTH + SIDE_WIDTH // 2, "新方式")):
        draw.text((x, HEADER // 2), label, font=font, fill="white", anchor="mm")
    header.save(directory / "header.png")
    clock = ImageFont.truetype(str(FONT), CLOCK_SIZE)
    for second in range(SECONDS):
        footer = Image.new("RGB", (WIDTH, FOOTER), "#151a22")
        text = f"経過 {second:03d} 秒  ({second // 60:02d}:{second % 60:02d})"
        ImageDraw.Draw(footer).text((WIDTH // 2, FOOTER // 2), text,
                                   font=clock, fill="white", anchor="mm")
        footer.save(directory / f"clock_{second:03d}.png")
    return directory


def compare_video() -> Path:
    """左右とも同じPTSで900秒を全て描画し、音声は元ソースから無変換で保持する。"""
    for root in (OFF, ON):
        info = video_info(root / "overlay.mp4")
        assert info["fps"] == FPS and info["frames"] == SECONDS * FPS, info
    directory = labels()
    REVIEW.mkdir(parents=True, exist_ok=True)
    output = REVIEW / f"{NAME}.mp4"
    temporary = REVIEW / f"{NAME}.partial.mp4"
    filters = (
        f"[0:v]setpts=PTS-STARTPTS,scale={SIDE_WIDTH}:{SIDE_HEIGHT},setsar=1[left];"
        f"[1:v]setpts=PTS-STARTPTS,scale={SIDE_WIDTH}:{SIDE_HEIGHT},setsar=1[right];"
        f"[left][right]hstack=inputs=2,pad={WIDTH}:{HEIGHT}:0:{HEADER}:black[base];"
        "[base][3:v]overlay=0:0[headed];"
        f"[headed][4:v]overlay=0:{HEADER + SIDE_HEIGHT}:eof_action=repeat,format=yuv420p[video]"
    )
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-threads", "2",
        "-i", str(OFF / "overlay.mp4"), "-threads", "2", "-i", str(ON / "overlay.mp4"),
        "-i", str(ORIGINAL), "-loop", "1", "-framerate", str(FPS),
        "-i", str(directory / "header.png"), "-framerate", "1", "-start_number", "0",
        "-i", str(directory / "clock_%03d.png"), "-filter_complex_threads", "1",
        "-filter_complex", filters, "-map", "[video]", "-map", "2:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "2",
        "-c:a", "copy", "-r", str(FPS), "-t", str(SECONDS), "-movflags", "+faststart",
        "-progress", str(OUT / "comparison.progress"), str(temporary)]
    save_json(OUT / "comparison_command.json", command)
    with (OUT / "comparison_ffmpeg.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    info = video_info(temporary)
    assert info["frames"] == SECONDS * FPS and info["seconds"] == SECONDS, info
    temporary.replace(output)
    return output


def exchange_csv() -> Path:
    """未到達・未閉鎖も空欄で残し、最初のS3と閉鎖後G_feを明示する。"""
    events = [json.loads(line) for line in (ON / "events.jsonl").read_text().splitlines()]
    rows = []
    with np.load(ON / "display.npz") as data:
        for event in events:
            s3 = next((v for v in event["values"] if v["source"] == "S3"), None)
            close = event["closed_sec"]
            following = [e["trigger_sec"] for e in events if e["game_idx"] == event["game_idx"]
                         and e["exchange_id"] > event["exchange_id"]]
            cutoff = min(following + [float(SECONDS)])
            eligible = [] if close is None else np.flatnonzero(
                (data["game_idx"] == event["game_idx"]) & (data["t_sec"] >= close)
                & (data["t_sec"] < cutoff) & (data["source"] == "G_fe"))
            idx = int(eligible[0]) if len(eligible) else None
            probability = None if idx is None else float(data["display_p1"][idx])
            rows.append(dict(exchange_id=event["exchange_id"], game_idx=event["game_idx"],
                trigger_sec=event["trigger_sec"], S3_sec=None if s3 is None else s3["t_sec"],
                closed_sec=close, S3_p1=None if s3 is None else s3["p1"],
                S3_adv=None if s3 is None else score(s3["p1"]), postclose_G_fe_p1=probability,
                postclose_G_fe_adv=None if probability is None else score(probability),
                postclose_G_fe_sec=None if idx is None else float(data["t_sec"][idx]),
                close_reason=event["close_reason"]))
    output = REVIEW / f"{NAME}_exchanges.csv"
    save_json(OUT / "exchanges_rows.json", rows)
    subprocess.run([NODE, MARKER, "--operation-kind", "create", "--expected-output-count", "1",
                    "--output-format", "csv"], check=True)
    subprocess.run([NODE, windows_path(OUT / "csv_builder.mjs"),
                    windows_path(OUT / "exchanges_rows.json"), windows_path(output)], check=True)
    save_json(OUT / "csv_summary.json", dict(path=str(output), exchanges=len(rows),
        columns="S3/G_feのp1は未平滑化確率、advは既存変換後の-100～100。未到達は空欄"))
    return output


def windows_path(path: Path) -> str:
    """同じ共有ファイルをWindows側のArtifact Toolへ渡す。"""
    value = str(path)
    if value.startswith("/mnt/c/") or value.startswith("/mnt/d/"):
        return value[5].upper() + ":/" + value[7:]
    raise ValueError(f"共有ドライブ外: {path}")


def score(probability: float) -> float:
    """既存の確率→有利不利変換と表示範囲を使う。"""
    return float(np.clip(_winprob_to_adv(probability), -100, 100))


def audio_hash(path: Path) -> str:
    """映像に依存しない音声パケットのハッシュを比較する。"""
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-c", "copy", "-t", str(SECONDS), "-f", "hash", "-hash", "sha256", "-"],
        capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main() -> None:
    """比較動画とCSVを作り、長さ・サイズ・元音声一致を記録する。"""
    output = compare_video()
    csv_path = exchange_csv()
    original_audio, review_audio = audio_hash(ORIGINAL), audio_hash(output)
    assert original_audio == review_audio, "元音声パケットが不一致"
    save_json(OUT / "review_summary.json", dict(video=str(output), csv=str(csv_path),
        **video_info(output), size_bytes=output.stat().st_size, sha256=digest(output),
        audio_identical=True, audio_sha256=review_audio))


if __name__ == "__main__":
    main()
