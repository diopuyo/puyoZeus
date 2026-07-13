"""発火後の有利不利反転の真因検証 (game B)。

仮説: 1Pが発火→空盤面になると密度系が0になりモデルは2P有利へ反転するが、
お邪魔会計(net収支/予告/相手盤面お邪魔)は「通った攻撃」を正しく追えている。
game B を流し 各STABLEで [モデル有利不利] と [お邪魔exchange系] を並べ、
発火後に両者が食い違う(net正=1P だがモデル負)かを見る。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
from scripts.visualize_advantage_overlay import _train_model, _score_advantage  # noqa: E402

VIDEO = "data/frames/video_29.mp4"
START, END, WARM = 202.0, 283.0, 60.0  # 毎フレーム更新で会計が復活するか検証(密度が真因か)


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
    step = int(0.5 * fps)
    print(" t_rel | モデル有利不利 | net収支(1P+) | 予告1P/2P | 盤面お邪魔1P/2P | 状態1P/2P")
    for fi in range(int((START - WARM) * fps), int(END * fps)):
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[:2] != (1080, 1920):
            f = cv2.resize(f, (1920, 1080))
        t = fi / fps
        r = pipe.update(fi, t, f)  # 会計のため毎フレーム更新(密に駆動)
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        snap = _drive_ojama(tr, r.p1, r.p2, ps1, ps2, t,
                            tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        if fi % step or t < START or b1 is None or b2 is None:
            continue  # 記録/表示だけ 0.5s 間引き
        adv, _, _ = _score_advantage(model, b1, b2, snap)
        oj1 = iv.board_ojama_count(b1).raw
        oj2 = iv.board_ojama_count(b2).raw
        st1, st2 = str(r.p1.state)[11:14], str(r.p2.state)[11:14]
        flag = ""
        # 食い違い: net収支は1P有利(+)だがモデルは2P(負)、or 逆
        if snap.net_balance_capped > 5 and adv < -5:
            flag = "  <<< net=1P だがモデル=2P 反転!"
        elif snap.net_balance_capped < -5 and adv > 5:
            flag = "  <<< net=2P だがモデル=1P 反転!"
        print(f"{t - START:6.1f} | {adv:+6.0f}         | {snap.net_balance_capped:+5d}"
              f"        | {snap.forecast_p1:3d}/{snap.forecast_p2:3d}   |"
              f" {oj1:4.0f}/{oj2:4.0f}      | {st1}/{st2}{flag}")
    cap.release()


if __name__ == "__main__":
    main()
