"""有利不利の判断根拠を全特徴量で分解する (診断専用・読み取り専用、2026-08-09).

user 指摘:
  1. 「29秒時点で 2P 有利なのが意味不明」
  2. 「1P と 2P が同時に連鎖しているのに指数がどんどん 2P 有利に寄っていくのが謎」
  3. 「最後の連鎖は両方が発火した時点で決まっているのでは」

overlay の「主因」表示は上位 3 件しか出さないため、 **何が効いて 2P 有利に
なっているのか全体像が見えない**。 本スクリプトは指定時刻の確定盤面について
以下を出す:

  - 全特徴量の差分値 (1P - 2P)
  - **各特徴量の寄与** = その特徴だけを 0 (=互角) に置き換えたときの勝率変化。
    正なら「その特徴が 1P 有利方向へ効いている」、 負なら 2P 有利方向。
  - 早期発火の速報バイアスと、 EMA 前後の値

**修正は行わない**。 1 シーンに合わせた調整は過学習になるため、 まず原因の
特定だけを行う (user 指示「くれぐれも過学習には気を付けてまず原因を教えて」)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board_state_machine import BoardState  # noqa: E402
from src.console_init import init_console  # noqa: E402

init_console()

import src.indicators_v2 as iv  # noqa: E402
from scripts.visualize_advantage_overlay import (  # noqa: E402
    COLOR_OJAMA_INTERACTION_COL,
    JP_LABEL,
    _fill_expected_fire_candidate,
    _fill_fire_stability_candidate,
    _fill_near_future_candidate,
    _ojama_flat_score,
    _side_feats,
    _train_model,
)
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO = _ROOT / "data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4"
OUT_MD = _ROOT / "data/verify/youtube_demo_2026-08-07/adv_attribution_2026-08-09.md"
# 調べる時刻 (user 指摘の場面)
TARGET_SECS: tuple[float, ...] = (29.0, 45.0, 54.5, 58.0, 62.0, 66.0, 70.0)


def _attribution(model, f1: dict, f2: dict, cols: list[str]) -> list[tuple]:
    """各特徴を互角(差0)に置いたときの勝率変化を寄与として返す。"""
    diff = {c: f1.get(c, 0.0) - f2.get(c, 0.0) for c in cols}
    use_inter = getattr(model, "_puyo_uses_interaction", False)
    x_cols = list(cols)
    if use_inter:
        flat = float(_ojama_flat_score(
            diff.get("board_ojama_count", 0.0), diff.get("ojama_forecast", 0.0),
        ))
        diff[COLOR_OJAMA_INTERACTION_COL] = (
            diff.get("board_color_puyo_total", 0.0) * flat
        )
        x_cols.append(COLOR_OJAMA_INTERACTION_COL)
    base_x = np.array([[diff[c] for c in x_cols]], dtype=float)
    base_p = float(model.predict_proba(base_x)[0, 1])
    rows = []
    for i, c in enumerate(x_cols):
        x = base_x.copy()
        x[0, i] = 0.0
        p = float(model.predict_proba(x)[0, 1])
        rows.append((c, diff[c], base_p - p))  # 寄与 = 有るとき - 無いとき
    rows.sort(key=lambda r: -abs(r[2]))
    return base_p, rows


def main() -> int:
    model = _train_model()
    cols = list(getattr(model, "_puyo_feature_cols", []))
    pipeline = RecognitionPipeline.load_default(force_in_match=True)
    tracker = OjamaAccountingTracker()
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    b1 = b2 = None
    targets = list(TARGET_SECS)
    out: list[str] = [
        "# 有利不利の判断根拠 (全特徴量分解) 2026-08-09", "",
        "user 指摘「29秒で2P有利が意味不明」「同時連鎖なのに2P有利へ寄る」の原因究明。",
        "寄与 = その特徴を互角(差0)に置いたときの勝率変化。正=1P有利方向へ効いている。", "",
    ]
    fi = 0
    while targets:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        r = pipeline.update(fi, t, frame)
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        if targets and t >= targets[0]:
            tgt = targets.pop(0)
            if b1 is None or b2 is None:
                out.append(f"## t={tgt}s — 盤面未確定 (skip)\n")
                fi += 1
                continue
            f1 = _side_feats(b1, 0.0, 0.0)
            f2 = _side_feats(b2, 0.0, 0.0)
            for name, fn in (
                ("saturated_chain_count", iv.saturated_chain_count),
                ("ukeyasusa", iv.ukeyasusa),
                ("sub_chain_count", iv.sub_chain_count),
            ):
                if name in cols and name not in f1:
                    f1[name] = fn(b1).score
                    f2[name] = fn(b2).score
            _fill_near_future_candidate(f1, f2, b1, b2, cols)
            _fill_fire_stability_candidate(f1, f2, b1, b2, cols)
            _fill_expected_fire_candidate(f1, f2, b1, b2, cols)
            p, rows = _attribution(model, f1, f2, cols)
            adv = (p - 0.5) * 200.0
            out.append(f"## t={tgt}s  勝率1P={p:.1%}  adv={adv:+.0f}")
            out.append(f"- 1P state={r.p1.state.value} / 2P state={r.p2.state.value}")
            out.append("")
            out.append("| 特徴 | 差分(1P-2P) | 寄与 | 向き |")
            out.append("|---|---|---|---|")
            for c, d, contrib in rows[:12]:
                label = JP_LABEL.get(c, c)
                arrow = "1P有利" if contrib > 0 else ("2P有利" if contrib < 0 else "-")
                out.append(f"| {label} | {d:+.3f} | {contrib:+.4f} | {arrow} |")
            out.append("")
        fi += 1
    cap.release()
    OUT_MD.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\n出力: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
