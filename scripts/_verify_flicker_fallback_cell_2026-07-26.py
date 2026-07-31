"""#51後半 検証用 (read-only, 一時スクリプト): 実例セル (1P r11c0, c34 t=470-484s)
の確定値時系列を、フラグ OFF/ON で比較する。

コミット対象外の検証補助スクリプト (report 用の一時ツール)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO = PROJ_ROOT / "data" / "frames" / "video_c34.mp4"
START_SEC = 468.0
DUR_SEC = 20.0
CELL_R, CELL_C = 11, 0


def run(enable_carryover: bool, enable_flicker: bool) -> list[tuple[float, int, int, int]]:
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(START_SEC * fps)
    end_frame = int((START_SEC + DUR_SEC) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    pipe = RecognitionPipeline.load_default(
        enable_recovery_counter_carryover=enable_carryover,
        enable_cnn_flicker_hsv_fallback=enable_flicker,
    )
    pipe.set_video_id("c34")
    out = []
    fi = start_frame
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        cnn_v = int(r.p1.cnn_board.get(CELL_R, CELL_C))
        conf_v = (
            int(r.p1.confirmed_board.get(CELL_R, CELL_C))
            if r.p1.confirmed_board is not None else -1
        )
        counter = pipe._sm_1p.context.stable_recovery_counters.get((CELL_R, CELL_C), 0)
        out.append((t, cnn_v, conf_v, counter))
        fi += 1
    cap.release()
    return out


def main() -> None:
    print(f"=== OFF (baseline: carryover=False, flicker=False) ===")
    off = run(False, False)
    print(f"=== ON (carryover=True, flicker=True) ===")
    on = run(True, True)

    print(f"{'t':>8} {'cnn(off)':>9} {'conf(off)':>10} {'cnt(off)':>9} | "
          f"{'cnn(on)':>8} {'conf(on)':>9} {'cnt(on)':>8}")
    for (t0, cnn0, conf0, cnt0), (t1, cnn1, conf1, cnt1) in zip(off, on):
        print(f"{t0:8.2f} {cnn0:9d} {conf0:10d} {cnt0:9d} | "
              f"{cnn1:8d} {conf1:9d} {cnt1:8d}")


if __name__ == "__main__":
    main()
