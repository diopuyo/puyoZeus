"""お邪魔予告UI拡大版の smoke (描画破綻チェック + PNG 出力).

video_124_4min.mp4 先頭 30 秒から数フレームを抜き、拡大した
forecast パネルと優勢バーを描画して PNG 保存する。認識パイプラインは
通さず、代表的な OjamaAccountSnapshot を合成して UI レイアウトだけ検証する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visualize_recognition import (  # noqa: E402
    P1_ROI_X,
    P1_ROI_Y,
    P2_ROI_X,
    P2_ROI_Y,
    draw_ojama_advantage_bar,
    draw_ojama_forecast_panel,
)
from src.ojama_accounting import OjamaAccountSnapshot  # noqa: E402

_VIDEO = "data/frames/video_124_4min.mp4"
_OUT_DIR = Path("data/verify/ojama_ui_enlarge")

# (forecast_p1, forecast_p2, net_balance_capped, offboard_p1, offboard_p2)
_CASES: list[tuple[int, int, int, int, int]] = [
    (0, 0, 0, 0, 0),       # 序盤 均衡
    (4, 0, -4, 0, 0),      # 2P がやや有利
    (38, 6, 32, 0, 0),     # 1P 大量送出 (岩x1連x1小x2 / 連x1)
    (90, 12, -78, 10, 0),  # 2P 圧倒 + 1P 画面外あふれ
]


def _make_snap(c: tuple[int, int, int, int, int]) -> OjamaAccountSnapshot:
    """smoke 用に予告/相殺収支/あふれを直接埋めたスナップショットを作る."""
    f1, f2, net, ob1, ob2 = c
    return OjamaAccountSnapshot(
        t_sec=0.0,
        pending_p1=f1,
        pending_p2=f2,
        total_generated_by_p1=0,
        total_generated_by_p2=0,
        total_offset_by_p1=0,
        total_offset_by_p2=0,
        total_dropped_to_p1=0,
        total_dropped_to_p2=0,
        net_ojama_balance=net,
        overflow_risk_p1=False,
        overflow_risk_p2=False,
        confidence=1.0,
        leftover_p1=0,
        leftover_p2=0,
        all_clear_pending_p1=False,
        all_clear_pending_p2=False,
        net_balance_capped=net,
        offboard_p1=ob1,
        offboard_p2=ob2,
        forecast_p1=f1,
        forecast_p2=f2,
    )


def main() -> int:
    """先頭 30 秒の数フレームに拡大 UI を描画し PNG 出力する."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(_VIDEO)
    if not cap.isOpened():
        print(f"FAIL: cannot open {_VIDEO}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    saved = 0
    for i, c in enumerate(_CASES):
        # 30 秒以内で散らした位置からフレーム取得
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * i * 7))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"FAIL: read frame {i}")
            return 1
        if frame.shape[1] != 1920 or frame.shape[0] != 1080:
            frame = cv2.resize(frame, (1920, 1080))
        snap = _make_snap(c)
        draw_ojama_forecast_panel(frame, snap, "1P", P1_ROI_X, P1_ROI_Y)
        draw_ojama_forecast_panel(frame, snap, "2P", P2_ROI_X, P2_ROI_Y)
        draw_ojama_advantage_bar(frame, snap)
        out = _OUT_DIR / f"case{i}_f1-{c[0]}_f2-{c[1]}_net{c[2]}.png"
        cv2.imwrite(str(out), frame)
        saved += 1
        print(f"OK frame {i}: shape={frame.shape} -> {out}")
    cap.release()
    # None スナップショットでも破綻しないこと
    cap2 = cv2.VideoCapture(_VIDEO)
    ok, frame = cap2.read()
    cap2.release()
    if ok and frame is not None:
        if frame.shape[1] != 1920:
            frame = cv2.resize(frame, (1920, 1080))
        draw_ojama_forecast_panel(frame, None, "1P", P1_ROI_X, P1_ROI_Y)
        draw_ojama_forecast_panel(frame, None, "2P", P2_ROI_X, P2_ROI_Y)
        draw_ojama_advantage_bar(frame, None)
        cv2.imwrite(str(_OUT_DIR / "case_none.png"), frame)
        saved += 1
        print("OK None-snapshot frame")
    print(f"DONE: saved {saved} PNG (frames>0: {saved > 0})")
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
