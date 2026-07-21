"""勝率の較正ビフォーアフターを1枚で比較するプロット。

同じゲームの有利不利 EMA 系列から、旧(直線 0.5+adv/200)と
新(較正 sigmoid)の1P勝率を重ねて描き、決着局面での差を可視化する。

使い方 (v29 game B):
    python -m scripts.plot_winprob_compare \
        --video data/frames/video_29.mp4 --video-id video_29 \
        --start-sec 200 --end-sec 283 --warmup-sec 16 --exclude-video video_29 \
        --out data/indicators_v2/overlay/winprob_compare_v29_gameB.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.plot_advantage_timeline import _collect_timeline  # noqa: E402
from scripts.visualize_advantage_overlay import adv_to_winprob  # noqa: E402

FONT_PATH = "/mnt/c/Windows/Fonts/meiryo.ttc"


def _setup_font() -> None:
    if Path(FONT_PATH).exists():
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def _plot(ts: list[float], advs: list[float], out: Path) -> None:
    _setup_font()
    x = np.array(ts); a = np.array(advs)
    lin = np.clip(0.5 + a / 200.0, 0, 1) * 100          # 旧: 直線
    cal = np.array([adv_to_winprob(v) for v in a]) * 100  # 新: 較正sigmoid
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(50, color="gray", lw=1, ls="--")
    ax.plot(x, lin, color="#999999", lw=1.8, label="旧: 直線 (50%+有利不利/2)")
    ax.plot(x, cal, color="#c0392b", lw=2.2, label="新: 較正 (実データsigmoid)")
    ax.set_ylim(0, 100); ax.set_xlim(0, x.max() if len(x) else 1)
    ax.set_xlabel("ゲーム開始からの経過秒"); ax.set_ylabel("1P 勝率 (%)")
    ax.set_title("勝率較正 ビフォーアフター — 決着局面が高勝率へ張り付く", fontsize=14)
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("出力:", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=0.0)
    ap.add_argument("--warmup-sec", type=float, default=16.0)
    ap.add_argument("--exclude-video", default=None)
    ap.add_argument("--sample-interval", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ts, advs, _ = _collect_timeline(
        Path(a.video), a.video_id, a.start_sec, a.end_sec, a.warmup_sec,
        a.exclude_video, a.sample_interval)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    _plot(ts, advs, Path(a.out))


if __name__ == "__main__":
    main()
