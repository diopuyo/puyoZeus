"""各動画の試合開始 frame で ROI offset を計測。

未使用動画 (試合 3 以降) でも有効な検証。720p resize 動画や別大会動画で
offset が有意なら ROI auto-calibration の意義が高まる。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_check_roi_offsets
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2  # noqa: E402

from src.roi_auto_calibrator import detect_roi_offsets  # noqa: E402


def get_match_starts(video_id: int) -> list[float]:
    """matches.tsv から各試合の start_sec リストを取得。"""
    candidates = [
        _ROOT
        / f"data/verify/match_boundaries_v5/video_{video_id:02d}/matches.tsv",
        _ROOT
        / f"data/verify/match_boundaries_v4/video_{video_id:02d}/matches.tsv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            return [
                float(r["start_sec"])
                for r in csv.DictReader(f, delimiter="\t")
            ]
    return []


def main() -> int:
    out_path = (
        _ROOT
        / "data/verify/phase_z_review/roi_offsets_check.tsv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[list] = [[
        "video", "match", "start_sec",
        "p1_dx", "p1_dy", "p2_dx", "p2_dy",
        "confidence",
    ]]
    print(f"{'video':<5} {'match':<5} {'start':<7} "
          f"{'p1':<10} {'p2':<10} {'conf':<6}")
    print("-" * 50)
    for vid in range(1, 20):
        if vid == 18:
            continue  # baseline、計測済
        video_path = _ROOT / f"data/frames/video_{vid:02d}.mp4"
        if not video_path.exists():
            continue
        starts = get_match_starts(vid)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        for idx, start in enumerate(starts[:5], 1):  # 試合 1-5 を確認
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(
                    frame, (1920, 1080), interpolation=cv2.INTER_AREA,
                )
            calib = detect_roi_offsets(frame)
            rows.append([
                f"v{vid:02d}", idx, f"{start:.0f}",
                calib.p1_offset[0], calib.p1_offset[1],
                calib.p2_offset[0], calib.p2_offset[1],
                f"{calib.confidence:.2f}",
            ])
            p1 = f"({calib.p1_offset[0]:+d},{calib.p1_offset[1]:+d})"
            p2 = f"({calib.p2_offset[0]:+d},{calib.p2_offset[1]:+d})"
            print(
                f"v{vid:02d}    m{idx:<3} {start:<7.0f} "
                f"{p1:<10} {p2:<10} {calib.confidence:<6.2f}"
            )
        cap.release()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
    print(f"\nsaved: {to_windows_path(out_path)}")

    # 集計: 各動画 5 試合で offset の最頻値 / 0 でない頻度
    print("\n=== 動画別 offset 統計 ===")
    by_video: dict[str, list[tuple]] = {}
    for r in rows[1:]:
        vid = r[0]
        by_video.setdefault(vid, []).append(
            (r[3], r[4], r[5], r[6]),
        )
    print(f"{'video':<5} {'p1 max':<14} {'p2 max':<14} "
          f"{'非 0 count':<10}")
    print("-" * 45)
    for vid, offsets in sorted(by_video.items()):
        if not offsets:
            continue
        p1_max_dx = max(abs(o[0]) for o in offsets)
        p1_max_dy = max(abs(o[1]) for o in offsets)
        p2_max_dx = max(abs(o[2]) for o in offsets)
        p2_max_dy = max(abs(o[3]) for o in offsets)
        n_nonzero = sum(
            1 for o in offsets if any(abs(x) > 0 for x in o)
        )
        print(
            f"{vid:<5} ({p1_max_dx:+d},{p1_max_dy:+d}){'':<6} "
            f"({p2_max_dx:+d},{p2_max_dy:+d}){'':<6} "
            f"{n_nonzero}/{len(offsets)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
