"""B3 レビュー動画の生成: 元動画 + 入力時系列 → データ帯付き動画 + フレーム別 CSV + 自己検収 + スマホ分割。

既存のレビュー生成器 (visualize_advantage_overlay / render_e18 の mobile) には書き込まず、
同じ流儀 (幅1280縮小・crf28・aac96k・+faststart・分割) だけを踏襲する。
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.b3_review_input import CSV_FIELDS, PROB_FIELDS, _num, derive, row_at
from src.b3_review_panel import GraphSeries, draw_b3_panel, panel_size

MOBILE_WIDTH, MOBILE_CRF, MOBILE_PRESET = 1280, 28, "veryfast"
MOBILE_AUDIO_KBPS = 96
SEGMENT_SEC = 270                     # 約4.5分 (user 2026-09-27)
LIMIT_BYTES = 30 * 1024 * 1024        # SendUserFile 上限 30MiB
# 270 秒 × (映像 + 音声) が上限に収まるよう逆算した映像上限。余裕 6% を見る。
SIZE_MARGIN = 0.94
FULL_CRF = 20
SELFCHECK_MAX_GAP_SEC = 5.0           # 評価行の間隔がこれを超えたら欠落区間として報告


def ffmpeg_exe() -> str:
    """imageio_ffmpeg のバンドル版を優先、無ければ PATH の ffmpeg。"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("ffmpeg が見つからない")
        return exe


def mobile_video_kbps(segment_sec: int = SEGMENT_SEC, limit_bytes: int = LIMIT_BYTES,
                      audio_kbps: int = MOBILE_AUDIO_KBPS) -> int:
    """分割 1 本が上限に収まる映像ビットレート上限 (kbps)。"""
    total_kbps = limit_bytes * 8 / 1000 * SIZE_MARGIN / segment_sec
    return max(int(total_kbps - audio_kbps), 1)


def mobile_commands(src: Path, dest_dir: Path, stem: str, ffmpeg: str = "ffmpeg",
                    segment_sec: int = SEGMENT_SEC) -> list[list[str]]:
    """縮小 → 分割の 2 コマンド (副作用なし、テスト可能)。"""
    kbps = mobile_video_kbps(segment_sec)
    small = dest_dir / f"{stem}_mobile_full.mp4"
    shrink = [ffmpeg, "-y", "-i", str(src), "-vf", f"scale={MOBILE_WIDTH}:-2",
              "-c:v", "libx264", "-crf", str(MOBILE_CRF), "-preset", MOBILE_PRESET,
              "-maxrate", f"{kbps}k", "-bufsize", f"{2 * kbps}k",
              "-force_key_frames", f"expr:gte(t,n_forced*{segment_sec})",
              "-c:a", "aac", "-b:a", f"{MOBILE_AUDIO_KBPS}k", "-movflags", "+faststart", str(small)]
    split = [ffmpeg, "-y", "-i", str(small), "-f", "segment", "-segment_time", str(segment_sec),
             "-c", "copy", "-reset_timestamps", "1", "-movflags", "+faststart",
             str(dest_dir / f"{stem}_mobile_%02d.mp4")]
    return [shrink, split]


def make_mobile(src: Path, dest_dir: Path, stem: str) -> list[dict[str, Any]]:
    """スマホ送付用の分割を作り、各本のサイズが上限以下であることを検査して返す。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for cmd in mobile_commands(src, dest_dir, stem, ffmpeg_exe()):
        subprocess.run(cmd, check=True, capture_output=True)
    (dest_dir / f"{stem}_mobile_full.mp4").unlink()
    parts = sorted(dest_dir.glob(f"{stem}_mobile_[0-9][0-9].mp4"))
    result = [dict(path=str(p), bytes=p.stat().st_size, ok=p.stat().st_size <= LIMIT_BYTES)
              for p in parts]
    if not result or not all(r["ok"] for r in result):
        raise RuntimeError(f"スマホ分割が上限 {LIMIT_BYTES} B を超過または空: {result}")
    return result


def selfcheck(rows: list[dict[str, Any]], start_sec: float, end_sec: float) -> dict[str, Any]:
    """表示区間の被覆を数える。母数つき (件数だけで合格と言わない)。

    - 区間内の評価行数、先頭行が区間開始より後なら表示されない秒数
    - 各確率列の非欠測行数/母数、行間隔が SELFCHECK_MAX_GAP_SEC を超える区間
    - 状態 (static/exchange/none) の行数、死亡証明状態の行数
    """
    inside = [r for r in rows if _num(r.get("t_sec")) is not None
              and start_sec <= _num(r["t_sec"]) <= end_sec]
    times = [float(r["t_sec"]) for r in inside]
    gaps = [(a, b) for a, b in zip(times, times[1:]) if b - a > SELFCHECK_MAX_GAP_SEC]
    out: dict[str, Any] = dict(
        rows_total=len(rows), rows_in_window=len(inside),
        uncovered_head_sec=max((times[0] - start_sec), 0.0) if times else end_sec - start_sec,
        long_gaps=gaps,
        prob_observed={k: sum(_num(r.get(k)) is not None for r in inside) for k in PROB_FIELDS},
        modes={m: sum(r.get("mode") == m for r in inside) for m in ("static", "exchange", "none")},
        death={f"{s}_{d}": sum(r.get(f"{s}_death_state") == d for r in inside)
               for s in ("p1", "p2") for d in ("none", "suspect", "proven_dead")})
    return out


class FrameCsv:
    """画面に出した値と同じものをフレーム別に逐次保存する (メモリ増加を防ぐ)。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w", encoding="utf-8-sig", newline="", buffering=1)
        self.writer = csv.DictWriter(self.stream, fieldnames=CSV_FIELDS)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in CSV_FIELDS})

    def close(self) -> None:
        self.stream.close()


def _open_encoder(out: Path, size: tuple[int, int], fps: float, src: Optional[Path],
                  start: float, duration: float) -> subprocess.Popen:
    """生フレームを標準入力で受け、元動画の音声 (あれば) を付けて h264 に符号化する。"""
    cmd = [ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{size[0]}x{size[1]}", "-r", f"{fps}", "-i", "-"]
    if src is not None:
        cmd += ["-ss", f"{start}", "-t", f"{duration}", "-i", str(src), "-map", "0:v:0",
                "-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", str(FULL_CRF), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def render_review(video: Path, rows: list[dict[str, Any]], out_mp4: Path, out_csv: Path,
                  start_sec: float = 0.0, end_sec: Optional[float] = None,
                  with_audio: bool = True) -> dict[str, Any]:
    """元動画 [start,end] 秒へデータ帯を連結して保存し、同じ値のフレーム別 CSV と検収結果を返す。

    入力行の t_sec は元動画先頭からの秒。フレーム時刻 t は start + n/fps で、t 以前の最後の行を使う。
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end = min(end_sec if end_sec is not None else total, total)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    series, times = GraphSeries.from_rows(rows), [float(r["t_sec"]) for r in rows]
    enc = _open_encoder(out_mp4, panel_size(width, height), fps, video if with_audio else None,
                        start_sec, end - start_sec)
    csv_out, n = FrameCsv(out_csv), 0
    try:
        while True:
            t = start_sec + n / fps
            ok, frame = cap.read()
            if not ok or t >= end:
                break
            row = row_at(rows, times, t)
            enc.stdin.write(np.ascontiguousarray(draw_b3_panel(frame, row, series, t)).tobytes())
            if row is not None:
                csv_out.write(derive(row, n))
            n += 1
    finally:
        csv_out.close()
        enc.stdin.close()
        enc.wait()
        cap.release()
    result = dict(frames=n, fps=fps, size=panel_size(width, height), selfcheck=selfcheck(rows, start_sec, end),
                  encoder_returncode=enc.returncode)
    out_csv.with_suffix(".selfcheck.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
    return result
