"""(B) 持続的「圧力(ダメージ)」信号の検証 (game B, dense)。

board_ojama(直接認識で信頼できる)の増加を減衰付き累積し、「どちらが攻撃を
通してきたか」を数秒記憶する持続信号 pressure を作る。
  各フレーム: pressure += (2P盤面お邪魔の増加) - (1P盤面お邪魔の増加); pressure *= 減衰
現モデル有利不利と並べて、pressure の方が滑らかで実体(2P勝ち)に整合するか見る。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
from scripts.visualize_advantage_overlay import _train_model, _score_advantage  # noqa: E402

VIDEO = "data/frames/video_29.mp4"
START, END, WARM = 202.0, 283.0, 60.0
DECAY = 0.985           # 毎フレーム減衰 (半減期 ~1.5s @30fps)
PRESSURE_SCALE = 6.0    # 圧力→有利不利[-100,100] 換算係数
FONT = "/mnt/c/Windows/Fonts/meiryo.ttc"


def _setup_font() -> None:
    if Path(FONT).exists():
        fm.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    model = _train_model("video_29")
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True)
    pipe.set_video_id("video_29")
    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((START - WARM) * fps))
    tr = OjamaAccountingTracker(); tr.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    pressure = 0.0
    prev_oj1 = prev_oj2 = 0.0
    step = int(0.5 * fps)
    ts, pres, adv_model = [], [], []
    for fi in range(int((START - WARM) * fps), int(END * fps)):
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
        oj1 = iv.board_ojama_count(b1).raw
        oj2 = iv.board_ojama_count(b2).raw
        # お邪魔の「増加」= 相手が攻撃を通した瞬間。1Pに増→2P攻勢(圧力減)
        pressure *= DECAY
        pressure += max(0.0, oj2 - prev_oj2) - max(0.0, oj1 - prev_oj1)
        prev_oj1, prev_oj2 = oj1, oj2
        if fi % step or t < START:
            continue
        adv, _, _ = _score_advantage(model, b1, b2, snap)
        ts.append(t - START)
        pres.append(float(np.clip(pressure * PRESSURE_SCALE, -100, 100)))
        adv_model.append(adv)
    cap.release()
    _setup_font()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(0, color="black", lw=1)
    ax.plot(ts, adv_model, color="#c08040", lw=1.0, alpha=0.7, label="現モデル(密度依存)")
    ax.plot(ts, pres, color="#2050c0", lw=2.0, label="圧力信号(持続ダメージ)")
    ax.fill_between(ts, pres, 0, where=[p >= 0 for p in pres], color="#4a70d0", alpha=0.3)
    ax.fill_between(ts, pres, 0, where=[p < 0 for p in pres], color="#d05050", alpha=0.3)
    ax.set_ylim(-100, 100); ax.set_xlabel("ゲーム開始からの経過秒")
    ax.set_ylabel("有利不利 (+1P / -2P)")
    ax.set_title("(B) 持続圧力信号 vs 現モデル (v29 game B)")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    out = Path("data/indicators_v2/overlay/pressure_v29.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    late = [(t, p, a) for t, p, a in zip(ts, pres, adv_model) if t >= 30]
    print("出力:", out)
    print(f"後半平均: 圧力={np.mean([x[1] for x in late]):+.1f} "
          f"モデル={np.mean([x[2] for x in late]):+.1f} (2P勝ちなら負が正)")


if __name__ == "__main__":
    main()
