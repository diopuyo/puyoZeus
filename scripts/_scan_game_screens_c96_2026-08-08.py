"""video_c96 (5.5時間の新おいうリーグ放送アーカイブ) の対戦ゲーム画面検出.

5.5 時間のフルデコードを避け、ffmpeg の fps フィルタで
SAMPLE_INTERVAL_SEC 秒間隔にサブサンプルしたフレームのみを解析する
(単一 ffmpeg プロセスの 1 パス通しデコード = ランダムシークの繰り返しより
大幅に高速、実測 14x realtime 程度)。

is_game 判定は既存資産 WinPanelDetector (`src/win_panel.py`) の
「数値★WIN★数値」パネル検出をそのまま流用する。このパネルは対戦中の
ゲームクライアント画面にのみ表示され、トーク画面・順位表画面・
過去試合ハイライトモンタージュ (パネル位置がズレる別レイアウト) では
検出されない (2026-08-08 プローブ実測で確認済み)。

出力: TSV (t_sec, is_game, panel_score)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from src.console_init import to_windows_path  # noqa: E402
from src.win_panel import WinPanelDetector  # noqa: E402

VIDEO: Path = _ROOT / "data" / "frames" / "video_c96.mp4"
OUT_TSV: Path = (
    _ROOT / "data" / "verify" / "c96_split_2026-08-08" / "scan_is_game.tsv"
)

# 粗いサンプリング間隔 (秒)。2〜5 秒間隔の指示に基づき 3 秒を採用。
SAMPLE_INTERVAL_SEC: float = 3.0
FRAME_W: int = 1920
FRAME_H: int = 1080
FRAME_BYTES: int = FRAME_W * FRAME_H * 3

# nice -19 相当の低優先度実行 (CLAUDE.md プロセス管理ルール)。
FFMPEG_BIN: str = "ffmpeg.exe"


def _build_ffmpeg_cmd() -> list[str]:
    fps_expr = f"1/{SAMPLE_INTERVAL_SEC:.6f}"
    # ffmpeg.exe は Windows バイナリなので WSL 形式パス (/mnt/c/...) を
    # 解釈できない。Windows フルパス (C:\...) に変換して渡す。
    video_win_path = to_windows_path(str(VIDEO))
    return [
        FFMPEG_BIN,
        "-nostdin",
        "-loglevel", "error",
        "-i", video_win_path,
        "-vf", f"fps={fps_expr}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]


def _lower_priority(pid: int) -> None:
    """best-effort で ffmpeg プロセスの優先度を下げる (wave2 regen 優先)。"""
    try:
        subprocess.run(
            [
                "wmic.exe", "process", "where", f"ProcessId={pid}",
                "CALL", "setpriority", "64",
            ],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass  # 優先度変更は失敗しても致命的でない


def main() -> int:
    if not VIDEO.exists():
        print(f"[error] video not found: {VIDEO}", file=sys.stderr)
        return 1

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    detector = WinPanelDetector.load_default()

    cmd = _build_ffmpeg_cmd()
    print(f"[start] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FRAME_BYTES * 4)
    _lower_priority(proc.pid)

    idx = 0
    t0 = time.time()
    with OUT_TSV.open("w", encoding="utf-8") as f:
        f.write("t_sec\tis_game\tpanel_score\n")
        while True:
            buf = proc.stdout.read(FRAME_BYTES)
            if len(buf) < FRAME_BYTES:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                FRAME_H, FRAME_W, 3,
            )
            result = detector.detect(frame)
            t_sec = idx * SAMPLE_INTERVAL_SEC
            f.write(f"{t_sec:.1f}\t{int(result.present)}\t{result.score:.4f}\n")
            idx += 1
            if idx % 200 == 0:
                elapsed = time.time() - t0
                print(
                    f"[progress] idx={idx} t={t_sec:.0f}s "
                    f"elapsed={elapsed:.0f}s"
                )
    proc.wait()
    print(f"[done] {idx} samples -> {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
