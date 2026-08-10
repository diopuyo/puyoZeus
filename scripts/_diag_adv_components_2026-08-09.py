"""有利不利の4成分 (圧力/配送予告/モデル/脅威) + バイアスを時系列で出す.

user 指摘の追跡:
  「29秒時点で 2P 有利が意味不明」
  「同時に連鎖しているのに指数がどんどん 2P 有利へ寄っていく」

得点タイブレークを切っても t=29 は 2P有利69% のままだった一方、
**モデル単体で同じ盤面を評価すると 1P有利62%** だった。 つまり残る差は
モデル以外の 3 成分にある。 overlay の合成式:

    adv = W_PRESSURE×圧力 + W_FORECAST×配送予告 + W_MODEL×モデル + W_THREAT×脅威
          + sl_bias(得点タイブレーク)

本スクリプトは各成分の生値と重み付き寄与を毎フレーム記録し、
**どの成分がいつからどれだけ寄せているか**を特定する。 読み取り専用。

出力: data/verify/youtube_demo_2026-08-07/adv_components_2026-08-09.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board_state_machine import BoardState  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

import src.indicators_v2 as iv  # noqa: E402
import scripts.visualize_advantage_overlay as vao  # noqa: E402

VIDEO = _ROOT / "data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4"
OUT_TSV = (
    _ROOT / "data/verify/youtube_demo_2026-08-07/adv_components_2026-08-09.tsv"
)
# 0.5 秒ごとに記録 (全フレームだと冗長)
SAMPLE_EVERY_SEC: float = 0.5


def main() -> int:
    model = vao._train_model()
    cols = list(getattr(model, "_puyo_feature_cols", []))
    pipe = vao.RecognitionPipeline.load_default(force_in_match=True)
    tracker, tp1, tp2, ptracker, fctracker, svtracker, hcache, efire = (
        vao._fresh_trackers(model)
    )
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    b1 = b2 = None
    ps1 = ps2 = BoardState.MENU
    rows = [
        "t_sec\tp1_state\tp2_state\tscore1\tscore2\t"
        "pressure\tforecast\tmodel_adv\tthreat\tsl_bias\t"
        "w_pressure\tw_forecast\tw_model\tw_threat\tadv_total"
    ]
    last_rec = -1.0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        r = pipe.update(fi, t, frame)
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        snap = vao._drive_ojama(tracker, r.p1, r.p2, ps1, ps2, t,
                                tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        settled = (
            r.p1.state == BoardState.STABLE or r.p2.state == BoardState.STABLE
        )
        if b1 is not None and b2 is not None and settled and t - last_rec >= SAMPLE_EVERY_SEC:
            last_rec = t
            model_adv, threat, _drv, _u1, _u2, _s1, _s2 = hcache.update(
                b1, b2, snap, r.p1, r.p2, tracker._elapsed(t))
            pres = ptracker.update(iv.board_ojama_count(b1).raw,
                                   iv.board_ojama_count(b2).raw)
            fc = fctracker.update(r.p1.score, r.p2.score,
                                  pipe.tsumo_count("1P"), pipe.tsumo_count("2P"))
            sl = max(-vao.SL_BIAS_CAP, min(vao.SL_BIAS_CAP,
                                           svtracker.update(r.p1.score, r.p2.score)))
            wp = vao.W_PRESSURE * pres
            wf = vao.W_FORECAST * fc
            wm = vao.W_MODEL * model_adv
            wt = vao.W_THREAT * threat
            total = wp + wf + wm + wt + sl
            rows.append(
                f"{t:.2f}\t{r.p1.state.value}\t{r.p2.state.value}\t"
                f"{r.p1.score}\t{r.p2.score}\t"
                f"{pres:.2f}\t{fc:.2f}\t{model_adv:.2f}\t{threat:.2f}\t{sl:.2f}\t"
                f"{wp:.2f}\t{wf:.2f}\t{wm:.2f}\t{wt:.2f}\t{total:.2f}"
            )
        fi += 1
    cap.release()
    OUT_TSV.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"出力: {OUT_TSV} ({len(rows) - 1} 行)")
    print(f"重み: W_PRESSURE={vao.W_PRESSURE} W_FORECAST={vao.W_FORECAST} "
          f"W_MODEL={vao.W_MODEL} W_THREAT={vao.W_THREAT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
