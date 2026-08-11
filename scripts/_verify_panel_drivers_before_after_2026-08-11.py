"""Phase1-3 除外リストの効果を「実際にパネルが使う関数」で前後比較する (読み取り専用).

`scripts/_diag_adv_attribution_2026-08-09.py` は ablation ベースの
真の寄与度 (`_attribution()`) を計算する別ロジックで、そもそも今回のバグ
(|差分| の大きい順で選ぶ `_score_advantage()` の主因ロジック) を再現しない。

本スクリプトは **実際にデモ映像/パネルが使う `_score_advantage()`** を
直接呼び、 除外リスト適用前 (attribution_exclude=()) / 適用後 (既定) の
主因3件を並べて比較する。 併せて adv/p1 がビット単位で不変であることも
表示する (テストは tests/test_attribution_exclusion.py で保証済み、
本スクリプトは実動画での目視確認用)。
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

from scripts.visualize_advantage_overlay import (  # noqa: E402
    JP_LABEL,
    _score_advantage,
    _train_model,
)
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO = _ROOT / "data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4"
TARGET_SECS: tuple[float, ...] = (29.0, 40.0, 45.0, 54.5, 58.0, 62.0, 66.0, 70.0)


def _fmt(drivers: list[tuple[str, float]]) -> str:
    return "  ".join(f"{JP_LABEL.get(c, c)}差 {v:+.2f}" for c, v in drivers)


def main() -> int:
    model = _train_model()
    pipeline = RecognitionPipeline.load_default(force_in_match=True)
    tracker = OjamaAccountingTracker()
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    b1 = b2 = None
    targets = list(TARGET_SECS)
    fi = 0
    print(f"{'時刻':>6s}  {'adv同一':>7s}  {'p1同一':>6s}")
    print("-" * 100)
    while targets:
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps
        r = pipeline.update(fi, t, frame)
        snap = tracker.update_from_score(r.p1.score, r.p2.score, t)
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        if targets and t >= targets[0]:
            tgt = targets.pop(0)
            if b1 is None or b2 is None:
                print(f"t={tgt}s 盤面未確定 (skip)")
                fi += 1
                continue
            adv_on, p1_on, drivers_on = _score_advantage(model, b1, b2, snap)
            adv_off, p1_off, drivers_off = _score_advantage(
                model, b1, b2, snap, attribution_exclude=())
            same = (adv_on == adv_off, p1_on == p1_off)
            print(f"t={tgt:5.1f}s  {str(same[0]):>7s}  {str(same[1]):>6s}"
                  f"  adv={adv_on:+.1f}")
            print(f"  除外前(デバッグ): {_fmt(drivers_off)}")
            print(f"  除外後(既定)   : {_fmt(drivers_on)}")
            print()
        fi += 1
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
