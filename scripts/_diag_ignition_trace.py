"""[scratch] 決定的発火の瞬間に有利不利がラグしているかを細かくトレース。

(B) near-future 信号の伸びしろ確認用。各フレームで score/各成分/合計/勝率と、
受け側の空き容量(=お邪魔をあと何個受けられるか)を出す。発火(score急増)から
有利不利が動くまでの遅れ、および「降る量>空き容量(キル)」かを目視する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
from scripts.visualize_advantage_overlay import (  # noqa: E402
    _train_model, PressureTracker, RealtimeForecastTracker, ScoreLeadTracker,
    HeavyAdvCache, EMA_ALPHA, W_PRESSURE, W_FORECAST, W_MODEL, W_THREAT, SL_BIAS_CAP,
    adv_to_winprob, kill_override, board_room,
)

VIDEO = "data/frames/video_29.mp4"
VID = "video_29"
START, END, WARMUP = 202.0, 283.0, 16.0


def _room(board) -> int:
    """受け側が窒息までに受けられるお邪魔のおおよその空き容量(セル数)。"""
    if board is None:
        return 72
    arr = board._grid
    occupied = int(np.count_nonzero(arr[1:]))  # row0(隠し段)は除く
    return max(0, 72 - occupied)


def main() -> None:
    model = _train_model(VID)
    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    proc0 = int(max(0.0, START - WARMUP) * fps); end = int(END * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, proc0)
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True)
    pipe.set_video_id(VID)
    tr = OjamaAccountingTracker(); tr.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    pt = PressureTracker(); fct = RealtimeForecastTracker()
    svt = ScoreLeadTracker(); hc = HeavyAdvCache(model)
    ps1 = ps2 = BoardState.MENU; b1 = b2 = None; adv_ema = 0.0
    prev_s1 = prev_s2 = 0
    print(" t    s1     s2   |圧力 予告 モデル threat| pending/room | raw→kill 勝率")
    for fi in range(proc0, end):
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
        # HeavyAdvCache.update() は7要素タプル (adv, threat, drivers, ukey1, ukey2, sat1, sat2)。
        # 末尾拡張耐性のため先頭2個のみ index 取得する形で受ける。
        _hres = hc.update(b1, b2, snap, r.p1, r.p2, tr._elapsed(t))
        m, thr = _hres[0], _hres[1]
        pres = pt.update(iv.board_ojama_count(b1).raw, iv.board_ojama_count(b2).raw)
        fc = fct.update(r.p1.score, r.p2.score,
                        pipe.tsumo_count("1P"), pipe.tsumo_count("2P"))
        sl = max(-SL_BIAS_CAP, min(SL_BIAS_CAP, svt.update(r.p1.score, r.p2.score)))
        adv_raw = max(-100.0, min(100.0, W_PRESSURE * pres + W_FORECAST * fc
                                  + W_MODEL * m + W_THREAT * thr + sl))
        room1, room2 = board_room(b1), board_room(b2)
        adv = kill_override(adv_raw, fct.inc1, fct.inc2, room1, room2)  # (B)キル判定
        adv_ema = EMA_ALPHA * adv + (1 - EMA_ALPHA) * adv_ema
        s1 = r.p1.score if r.p1.score is not None else prev_s1
        s2 = r.p2.score if r.p2.score is not None else prev_s2
        if fi % 10 == 0 and t >= START:  # 1/3秒ごとに出力
            print(f"{t:5.1f} {s1:6d} {s2:6d} |{pres:4.0f} {fc:4.0f} {m:5.0f} {thr:5.0f}"
                  f"| inc1={fct.inc1:4.0f} inc2={fct.inc2:4.0f} room {room1:3d}/{room2:3d}"
                  f"| raw{adv_raw:4.0f} kill{adv:4.0f} {adv_to_winprob(adv_ema)*100:4.0f}%")
        prev_s1, prev_s2 = s1, s2
    cap.release()


if __name__ == "__main__":
    main()
