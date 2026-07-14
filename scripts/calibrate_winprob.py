"""有利不利→勝率の較正パラメータを実データから学習する。

初期版の勝率は `50% + 有利不利/2` の直線変換で、決着局面でも 70% 程度で
頭打ちになっていた。本スクリプトは複数動画・複数ゲームを1パス処理し、
各フレームの有利不利 EMA に「そのゲームの最終勝者」ラベルを付け、
ロジスティック回帰 P(1P勝ち)=sigmoid(k×有利不利) の傾き k を学習する。
対称性のため切片は 0 に固定(有利不利0→勝率50%)。

結果は data/indicators_v2/winprob_calib.json に保存し、
overlay / timeline がこれを読んで勝率変換に使う(無い場合は直線にフォールバック)。

使い方:
    python -m scripts.calibrate_winprob \
        --spec data/frames/video_29.mp4:video_29:140:700 \
        --spec data/frames/video_30.mp4:video_30:140:700
"""
from __future__ import annotations

import argparse
import json
import math
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
)

SCORE_RESET_DROP = 1000    # スコアがこれ以上減少=ゲーム境界
MIN_GAME_SEC = 12.0        # これ未満の区間はノイズとして無視
MIN_WINNER_SCORE = 500     # 勝者判定に必要な最低スコア
SAMPLE_STRIDE = 5          # 較正サンプルの間引き(フレーム)
CALIB_PATH = Path("data/indicators_v2/winprob_calib.json")


class _GameBuf:
    """1ゲーム分の (有利不利サンプル列, 最終スコア) バッファ。"""

    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.last_s1 = 0
        self.last_s2 = 0
        self.advs: list[float] = []


def _finalize(buf: _GameBuf, xs: list[float], ys: list[float]) -> int:
    """成立ゲームなら (有利不利, 勝者ラベル) を xs/ys に積む。積んだ数を返す。"""
    if max(buf.last_s1, buf.last_s2) < MIN_WINNER_SCORE:
        return 0
    if abs(buf.last_s1 - buf.last_s2) < 1:
        return 0
    label = 1.0 if buf.last_s1 > buf.last_s2 else 0.0
    for a in buf.advs:
        xs.append(a); ys.append(label)
    return len(buf.advs)


def _collect(spec: str, xs: list[float], ys: list[float]) -> None:
    """1動画区間を処理し、各成立ゲームの (有利不利, 勝者) を xs/ys へ。"""
    video, vid, s0, s1 = spec.split(":")
    start_sec, end_sec = float(s0), float(s1)
    model = _train_model(vid)  # 対象動画は学習から除外(リーク防止)
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    proc0 = int(max(0.0, start_sec - 16.0) * fps)
    end = int(end_sec * fps) if end_sec > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, proc0)
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True)
    pipe.set_video_id(vid)
    tr = OjamaAccountingTracker(); tr.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    pt = PressureTracker(); fct = RealtimeForecastTracker()
    svt = ScoreLeadTracker(); hcache = HeavyAdvCache(model)
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    adv_ema = 0.0
    buf = _GameBuf(start_sec)
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
        for s, last in ((r.p1.score, buf.last_s1), (r.p2.score, buf.last_s2)):
            if s is not None and last - s >= SCORE_RESET_DROP and t - buf.t0 >= MIN_GAME_SEC:
                _finalize(buf, xs, ys)
                pt = PressureTracker(); fct = RealtimeForecastTracker()
                svt = ScoreLeadTracker(); hcache = HeavyAdvCache(model)
                adv_ema = 0.0; buf = _GameBuf(t)
                break
        if r.p1.score is not None:
            buf.last_s1 = r.p1.score
        if r.p2.score is not None:
            buf.last_s2 = r.p2.score
        if b1 is None or b2 is None:
            continue
        m, thr, _ = hcache.update(b1, b2, snap, r.p1, r.p2, tr._elapsed(t))
        pres = pt.update(iv.board_ojama_count(b1).raw, iv.board_ojama_count(b2).raw)
        fc = fct.update(r.p1.score, r.p2.score,
                        pipe.tsumo_count("1P"), pipe.tsumo_count("2P"))
        sl = max(-SL_BIAS_CAP, min(SL_BIAS_CAP, svt.update(r.p1.score, r.p2.score)))
        adv = W_PRESSURE * pres + W_FORECAST * fc + W_MODEL * m + W_THREAT * thr + sl
        adv = max(-100.0, min(100.0, adv))
        adv_ema = EMA_ALPHA * adv + (1 - EMA_ALPHA) * adv_ema
        if fi % SAMPLE_STRIDE == 0 and t >= start_sec:
            buf.advs.append(adv_ema)
    _finalize(buf, xs, ys)
    cap.release()


def _fit_slope(xs: list[float], ys: list[float]) -> float:
    """P=sigmoid(k×adv) の傾き k を勾配降下で学習(切片0固定・対称)。"""
    x = np.asarray(xs, dtype=float); y = np.asarray(ys, dtype=float)
    k = 1.0 / 100.0  # 初期値 = 直線相当のなだらかさ
    lr = 5e-4
    for _ in range(4000):
        p = 1.0 / (1.0 + np.exp(-k * x))
        grad = float(np.mean((p - y) * x))  # ロジスティック損失の傾き
        k -= lr * grad
    return k


def _report(xs: list[float], ys: list[float], k: float) -> None:
    """較正曲線と、ビン別の実測勝率(健全性チェック)を表示。"""
    x = np.asarray(xs); y = np.asarray(ys)
    print(f"\n=== 較正 (サンプル {len(xs)}) 傾き k={k:.4f} ===")
    print(" 有利不利 | 直線勝率 | 較正勝率 | 実測勝率(±10幅)")
    for a in (-80, -60, -40, -20, 0, 20, 40, 60, 80):
        lin = max(0.0, min(1.0, 0.5 + a / 200.0))
        cal = 1.0 / (1.0 + math.exp(-k * a))
        m = np.abs(x - a) <= 10
        emp = float(y[m].mean()) if m.sum() > 20 else float("nan")
        print(f"   {a:+4d}   |  {lin*100:4.0f}%   |  {cal*100:4.0f}%   |  "
              f"{emp*100:4.0f}% (n={int(m.sum())})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True,
                    help="動画:ID:開始秒:終了秒 (複数可)")
    a = ap.parse_args()
    xs: list[float] = []; ys: list[float] = []
    for spec in a.spec:
        print(f"[collect] {spec}", flush=True)
        _collect(spec, xs, ys)
        print(f"  累計サンプル {len(xs)}", flush=True)
    if len(xs) < 100:
        print("サンプル不足。較正中止。"); return
    k = _fit_slope(xs, ys)
    _report(xs, ys, k)
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps(
        {"kind": "logistic_symmetric", "k": k, "n": len(xs)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存: {CALIB_PATH}")


if __name__ == "__main__":
    main()
