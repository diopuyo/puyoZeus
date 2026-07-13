"""多ゲーム勝者一致率の検証ハーネス。

動画を1パス処理し、スコアリセット(大幅減)でゲームを分割。各ゲームで
最終有利不利(4成分ブレンドのEMA)の符号が、勝者(ゲーム終了直前で
スコアが高い側)と一致するかを照合し、一致率を報告する。
ゲームごとに pressure/score-lead/EMA/会計をリセット。

使い方 (v29 の t=140-700 ≒ 複数ゲーム):
    python -m scripts.validate_advantage_winner \
        --video data/frames/video_29.mp4 --video-id video_29 \
        --start-sec 140 --end-sec 700 --exclude-video video_29
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
from scripts.visualize_advantage_overlay import (  # noqa: E402
    _train_model, _score_advantage, PressureTracker, ScoreLeadTracker, _threat,
    EMA_ALPHA, W_PRESSURE, W_CHAIN, W_MODEL, W_THREAT,
)

SCORE_RESET_DROP = 1000   # スコアがこれ以上減少=ゲーム境界
MIN_GAME_SEC = 12.0       # これ未満の区間はノイズとして無視
MIN_WINNER_SCORE = 500    # 勝者判定に必要な最低スコア(不成立ゲーム除外)


class _GameAgg:
    """1ゲーム分の集計(最終スコア・最終有利不利)。"""

    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.last_s1 = 0
        self.last_s2 = 0
        self.last_adv = 0.0


def _decide(g: _GameAgg) -> tuple[str, str, bool] | None:
    """(勝者, 有利不利符号側, 一致) を返す。不成立なら None。"""
    if max(g.last_s1, g.last_s2) < MIN_WINNER_SCORE:
        return None
    if abs(g.last_s1 - g.last_s2) < 1:
        return None
    winner = "1P" if g.last_s1 > g.last_s2 else "2P"
    adv_side = "1P" if g.last_adv > 0 else "2P"
    return winner, adv_side, winner == adv_side


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=0.0)
    ap.add_argument("--exclude-video", default=None)
    a = ap.parse_args()
    model = _train_model(a.exclude_video)
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start_sec * fps))
    end = int(a.end_sec * fps) if a.end_sec > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True)
    pipe.set_video_id(a.video_id)
    tr = OjamaAccountingTracker(); tr.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    pt = PressureTracker(); lt = ScoreLeadTracker()
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    adv_ema = 0.0
    games: list[_GameAgg] = [_GameAgg(a.start_sec)]
    for fi in range(int(a.start_sec * fps), end):
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
        g = games[-1]
        # ゲーム境界検知(いずれかのスコアが大幅減)
        for s, last in ((r.p1.score, g.last_s1), (r.p2.score, g.last_s2)):
            if s is not None and last - s >= SCORE_RESET_DROP and t - g.t0 >= MIN_GAME_SEC:
                pt = PressureTracker(); lt = ScoreLeadTracker(); adv_ema = 0.0
                games.append(_GameAgg(t))
                g = games[-1]
                break
        if r.p1.score is not None:
            g.last_s1 = r.p1.score
        if r.p2.score is not None:
            g.last_s2 = r.p2.score
        if b1 is None or b2 is None:
            continue
        m, _, _ = _score_advantage(model, b1, b2, snap)
        pres = pt.update(iv.board_ojama_count(b1).raw, iv.board_ojama_count(b2).raw)
        lead = lt.update(r.p1.score, r.p2.score)
        thr = _threat(b1, b2, r.p1, r.p2, tr._elapsed(t))
        adv = W_PRESSURE * pres + W_CHAIN * lead + W_MODEL * m + W_THREAT * thr
        adv_ema = EMA_ALPHA * adv + (1 - EMA_ALPHA) * adv_ema
        g.last_adv = adv_ema
    cap.release()
    print("\n=== 勝者一致率 検証 ===")
    ok_n = tot = 0
    for i, g in enumerate(games):
        d = _decide(g)
        if d is None:
            continue
        winner, adv_side, hit = d
        tot += 1; ok_n += int(hit)
        print(f"  game{i} t={g.t0:.0f}s  スコア {g.last_s1}/{g.last_s2}  "
              f"勝者={winner}  有利不利={adv_side}({g.last_adv:+.0f})  "
              f"{'○' if hit else '×'}")
    if tot:
        print(f"\n一致率: {ok_n}/{tot} = {100 * ok_n / tot:.0f}%")


if __name__ == "__main__":
    main()
