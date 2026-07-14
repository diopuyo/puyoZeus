"""有利不利の成分分解プロット — どの成分(圧力/予告/モデル/火力threat)が
有利不利を動かしているかを可視化(ユーザーの「なぜ」に答える解析ツール)。

各成分の重み付き寄与(W×成分)を積み上げ的に描き、合計(=有利不利)を重ねる。

使い方 (v29 gameB):
    python -m scripts.plot_advantage_components \
        --video data/frames/video_29.mp4 --video-id video_29 \
        --start-sec 202 --end-sec 283 --warmup-sec 16 --exclude-video video_29 \
        --out data/indicators_v2/overlay/components_v29_gameB.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
from scripts.visualize_advantage_overlay import (  # noqa: E402
    _train_model, _score_advantage, PressureTracker, RealtimeForecastTracker, _threat,
    W_PRESSURE, W_FORECAST, W_MODEL, W_THREAT,
)

FONT = "/mnt/c/Windows/Fonts/meiryo.ttc"


def _setup_font() -> None:
    if Path(FONT).exists():
        fm.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def _collect(a) -> dict[str, list[float]]:
    """成分の重み付き寄与を時系列で収集。"""
    model = _train_model(a.exclude_video)
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((a.start_sec - a.warmup_sec) * fps))
    end = int(a.end_sec * fps) if a.end_sec > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True)
    pipe.set_video_id(a.video_id)
    tr = OjamaAccountingTracker(); tr.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    pt = PressureTracker(); fct = RealtimeForecastTracker()
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    step = int(0.5 * fps)
    out: dict[str, list[float]] = {k: [] for k in
                                   ("t", "圧力", "予告", "現モデル", "火力threat", "合計")}
    for fi in range(int((a.start_sec - a.warmup_sec) * fps), end):
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[:2] != (1080, 1920):
            f = cv2.resize(f, (1920, 1080))
        t = fi / fps
        r = pipe.update(fi, t, f)
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        snap = _drive_ojama(tr, r.p1, r.p2, ps1, ps2, t,
                            tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        if b1 is None or b2 is None:
            continue
        m, _, _ = _score_advantage(model, b1, b2, snap)
        pres = pt.update(iv.board_ojama_count(b1).raw, iv.board_ojama_count(b2).raw)
        lead = fct.update(r.p1.score, r.p2.score)
        thr = _threat(b1, b2, r.p1, r.p2, tr._elapsed(t))
        if fi % step or t < a.start_sec:
            continue
        cp = W_PRESSURE * pres; cl = W_FORECAST * lead
        cm = W_MODEL * m; ct = W_THREAT * thr
        out["t"].append(t - a.start_sec)
        out["圧力"].append(cp); out["予告"].append(cl)
        out["現モデル"].append(cm); out["火力threat"].append(ct)
        out["合計"].append(cp + cl + cm + ct)
    cap.release()
    return out


def _plot(d: dict[str, list[float]], title: str, out: Path) -> None:
    _setup_font()
    t = d["t"]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0, color="black", lw=0.8)
    colors = {"圧力": "#2050c0", "予告": "#20a060",
              "現モデル": "#a0a0a0", "火力threat": "#e08020"}
    for k, c in colors.items():
        ax.plot(t, d[k], color=c, lw=1.4, label=f"{k}(寄与)")
    ax.plot(t, d["合計"], color="black", lw=2.6, label="合計=有利不利")
    ax.set_ylim(-100, 100); ax.set_xlabel("ゲーム開始からの経過秒")
    ax.set_ylabel("有利不利への寄与 (+1P / -2P)")
    ax.set_title(title, fontsize=13); ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=5, fontsize=9)
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="有利不利の成分分解")
    a = ap.parse_args()
    d = _collect(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    _plot(d, a.title, Path(a.out))


if __name__ == "__main__":
    main()
