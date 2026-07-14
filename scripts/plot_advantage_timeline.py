"""有利不利の時間推移グラフ (将棋の評価値グラフ風) を出力する。

動画は描画せず、認識 + tier1 軽量モデルで各時刻の有利不利だけを計算し、
0=互角ラインの折れ線 (上=1P有利/青, 下=2P有利/赤) にプロットする。

使い方 (v29 game B):
    python -m scripts.plot_advantage_timeline \
        --video data/frames/video_29.mp4 --video-id video_29 \
        --start-sec 200 --end-sec 283 --warmup-sec 16 --exclude-video video_29 \
        --out data/indicators_v2/overlay/timeline_v29_gameB.png
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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402
from scripts.visualize_advantage_overlay import (  # noqa: E402
    _train_model, EMA_ALPHA, PressureTracker, RealtimeForecastTracker, ScoreLeadTracker,
    HeavyAdvCache, W_PRESSURE, W_FORECAST, W_MODEL, W_THREAT, SL_BIAS_CAP,
)

FONT_PATH = "/mnt/c/Windows/Fonts/meiryo.ttc"


def _setup_font() -> None:
    if Path(FONT_PATH).exists():
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def _collect_timeline(
    video: Path, video_id: str, start_sec: float, end_sec: float,
    warmup_sec: float, exclude_video: str | None, sample_interval: float,
) -> tuple[list[float], list[float], list[float]]:
    """(ゲーム開始からの秒, 有利不利[-100..100], 1P勝率) の時系列を返す。"""
    model = _train_model(exclude_video)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    proc_frame = int(max(0.0, start_sec - warmup_sec) * fps)
    end_frame = int(end_sec * fps) if end_sec > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, proc_frame)
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True)
    pipe.set_video_id(video_id)
    tracker = OjamaAccountingTracker(); tracker.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    adv_ema = 0.0; p1_ema = 0.5
    ptracker = PressureTracker()
    fctracker = RealtimeForecastTracker()
    svtracker = ScoreLeadTracker()
    hcache = HeavyAdvCache(model)
    step = max(1, int(round(sample_interval * fps)))
    ts: list[float] = []; advs: list[float] = []; p1s: list[float] = []
    for fi in range(proc_frame, end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)  # 会計のため毎フレーム更新
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        snap = _drive_ojama(tracker, r.p1, r.p2, ps1, ps2, t,
                            tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        if b1 is None or b2 is None:
            continue
        model_adv, threat, _ = hcache.update(b1, b2, snap, r.p1, r.p2, tracker._elapsed(t))
        pres = ptracker.update(iv.board_ojama_count(b1).raw, iv.board_ojama_count(b2).raw)
        fc = fctracker.update(r.p1.score, r.p2.score,
                              pipe.tsumo_count("1P"), pipe.tsumo_count("2P"))  # (M3改B)配送予告
        sl_bias = max(-SL_BIAS_CAP, min(SL_BIAS_CAP,  # (b)得点タイブレーク(±15頭打ち)
                                        svtracker.update(r.p1.score, r.p2.score)))
        adv = (W_PRESSURE * pres + W_FORECAST * fc
               + W_MODEL * model_adv + W_THREAT * threat) + sl_bias  # 4成分+タイブレーク
        adv = max(-100.0, min(100.0, adv))
        p1 = 0.5 + adv / 200.0
        adv_ema = EMA_ALPHA * adv + (1 - EMA_ALPHA) * adv_ema
        p1_ema = EMA_ALPHA * p1 + (1 - EMA_ALPHA) * p1_ema
        if fi % step == 0 and t >= start_sec:  # 記録だけ間引き
            ts.append(t - start_sec); advs.append(adv_ema); p1s.append(p1_ema)
    cap.release()
    return ts, advs, p1s


def _plot(ts: list[float], advs: list[float], title: str, out: Path) -> None:
    """将棋評価値グラフ風にプロットして保存。"""
    _setup_font()
    x = np.array(ts); y = np.array(advs)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(0, color="black", lw=1)
    ax.fill_between(x, y, 0, where=(y >= 0), color="#4a70d0", alpha=0.7, label="1P 有利")
    ax.fill_between(x, y, 0, where=(y < 0), color="#d05050", alpha=0.7, label="2P 有利")
    ax.plot(x, y, color="black", lw=1.2)
    ax.set_ylim(-100, 100); ax.set_xlim(0, x.max() if len(x) else 1)
    ax.set_xlabel("ゲーム開始からの経過秒"); ax.set_ylabel("有利不利 (＋100=1P圧倒 / −100=2P圧倒)")
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    ax2 = ax.twinx(); ax2.set_ylim(0, 100)
    ax2.set_ylabel("1P 勝率 (%)"); ax2.set_yticks([0, 25, 50, 75, 100])
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
    ap.add_argument("--title", default="有利不利 時間推移 (将棋評価値グラフ風)")
    a = ap.parse_args()
    ts, advs, _ = _collect_timeline(
        Path(a.video), a.video_id, a.start_sec, a.end_sec, a.warmup_sec,
        a.exclude_video, a.sample_interval)
    print(f"[timeline] 点数={len(ts)}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    _plot(ts, advs, a.title, Path(a.out))


if __name__ == "__main__":
    main()
