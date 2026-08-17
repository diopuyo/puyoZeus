"""(b-2) 解除信号置換の保留条件チェック (2026-08-18、アーキ発注)。

対戦カード紹介中に score_zero_both (ScoreZeroDetector) の NCC が
ZERO_NCC_THRESHOLD (0.85) を一瞬でも跨いでしまわないかを実測する。
跨ぐ瞬間があれば「score_zero_both 持続」を解除条件にした場合に
ラッチが誤って早期解除される恐れがあるため、実装前に必ず確認する。

本体コード (src/) は変更しない。ScoreZeroDetector を外部から直接呼ぶだけ
(stateless、CNN 等の重い処理は一切使わない・高速)。

対象: c18 (022/023 の窓、t≈1891.7秒付近) / c20 (024/025 の窓、
t≈830.6秒付近)。近傍 report.md の「MENU state が近傍(±1.5s)に出現」を
広めにカバーするため ±15秒の窓でスキャンする。

## 使い方 (WSL)
    PYTHONPATH=. ./venv/bin/python -m \
        scripts._check_score_zero_ncc_cardintro_2026-08-18
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.score_zero import ScoreZeroDetector, ZERO_NCC_THRESHOLD  # noqa: E402

VIDEO_DIR: Path = Path.home() / "frames"
OUT_DIR: Path = Path(
    "/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
    "/data/verify/boundary_impl_verify_2026-08-18"
)
WINDOW_MARGIN_SEC: float = 15.0


@dataclass(frozen=True)
class ScanTarget:
    video: str
    anchor_t_sec: float


TARGETS: "list[ScanTarget]" = [
    ScanTarget("c18", 1891.733),
    ScanTarget("c20", 830.6),
]


def scan_video(target: ScanTarget) -> "list[dict]":
    video_path = VIDEO_DIR / f"video_{target.video}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = ScoreZeroDetector.load_default()
    start_sec = max(0.0, target.anchor_t_sec - WINDOW_MARGIN_SEC)
    end_sec = target.anchor_t_sec + WINDOW_MARGIN_SEC
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows: "list[dict]" = []
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t_sec = frame_idx / fps
        res = detector.detect(frame)
        rows.append({
            "video": target.video, "frame_idx": frame_idx,
            "t_sec": round(t_sec, 4),
            "dt_from_anchor": round(t_sec - target.anchor_t_sec, 4),
            "score_1p_ncc": round(res.score_1p, 4),
            "score_2p_ncc": round(res.score_2p, 4),
            "both_zero": res.both_zero,
        })
        frame_idx += 1
    cap.release()
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: "list[dict]" = []
    crossings: "list[dict]" = []
    for target in TARGETS:
        print(f"[{target.video}] NCC スキャン開始 (±{WINDOW_MARGIN_SEC}秒)...")
        rows = scan_video(target)
        all_rows.extend(rows)
        for r in rows:
            if r["score_1p_ncc"] >= ZERO_NCC_THRESHOLD or r["score_2p_ncc"] >= ZERO_NCC_THRESHOLD:
                crossings.append(r)
        max1 = max((r["score_1p_ncc"] for r in rows), default=-1.0)
        max2 = max((r["score_2p_ncc"] for r in rows), default=-1.0)
        print(f"  max(score_1p_ncc)={max1:.4f} max(score_2p_ncc)={max2:.4f} "
              f"threshold={ZERO_NCC_THRESHOLD}")

    out_csv = OUT_DIR / "score_zero_ncc_cardintro_scan.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[out] {out_csv} ({len(all_rows)} 行)")
    print(f"[result] 0.85 跨ぎ件数: {len(crossings)}")
    for c in crossings[:20]:
        print(f"  {c}")


if __name__ == "__main__":
    main()
