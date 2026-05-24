"""Phase Z violations を frame 全体スクリーンショット形式で可視化。

各 violation の time に対応するフレームに、該当 cell を赤枠で描画して
1 PNG = 1 frame で出力する。同一フレームに複数 violation がある場合は
全 cell をまとめてマークする。フィールド全体の文脈 (落下軌跡・周囲整合性)
を見ながら色を判定したい場合の補助シート。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_render_violation_frames \
        --segment v06_m03_385_415

segment 名から自動的に labels.csv と動画 (data/frames/video_NN.mp4) を解決。
出力先: data/verify/phase_z_review/weak_video_extra/<segment>/violations_review/frames_marked/
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402

init_console()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from scripts.phase_z_extract_violations import is_hard_violation  # noqa: E402
from src.board import HIDDEN_ROWS  # noqa: E402
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    BoardRegion,
)

WEAK_ROOT = _ROOT / "data" / "verify" / "phase_z_review" / "weak_video_extra"
FRAME_GRID_MS = 500  # frames/ ディレクトリの 0.5s grid 命名規則

# 描画用色 (BGR)
RED = (40, 40, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def parse_segment_video(segment: str) -> str | None:
    """segment 名 (v06_m03_xxx_xxx) から動画ファイル名を導出。"""
    match = re.match(r"v(\d+)_m\d+", segment)
    if not match:
        return None
    return f"video_{int(match.group(1)):02d}.mp4"


def full_cell_rect(
    side: str, row: int, col: int,
) -> tuple[int, int, int, int]:
    """side/row(visible 0-11)/col の cell 全体矩形 (x1,y1,x2,y2)."""
    region: BoardRegion = (
        DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    )
    abs_row = row + HIDDEN_ROWS
    cx, cy = region.cell_center(abs_row, col)
    half_w = max(1, int(region.cell_width / 2))
    half_h = max(1, int(region.cell_height / 2))
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def load_frame_from_grid(
    frames_dir: Path, time_sec: float,
) -> np.ndarray | None:
    """0.5s grid の frame を試行ロード。"""
    ms_round = int(round(time_sec * 2) * FRAME_GRID_MS)
    fp = frames_dir / f"{ms_round:06d}.png"
    if fp.exists():
        return cv2.imread(str(fp))
    return None


def draw_violation_marker(
    out: np.ndarray, vio: dict,
) -> None:
    """1 violation cell を赤枠 + ラベルでマーク。"""
    x1, y1, x2, y2 = full_cell_rect(vio["side"], vio["row"], vio["col"])
    cv2.rectangle(out, (x1, y1), (x2, y2), RED, 3)
    label = f"#{vio['id']} {vio['recognized']}"
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
    )
    ly = max(th + 4, y1 - 4)
    lx = x1
    cv2.rectangle(
        out, (lx, ly - th - 4), (lx + tw + 4, ly + 2), BLACK, -1,
    )
    cv2.putText(
        out, label, (lx + 2, ly - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA,
    )


def render_marked_frame(
    img: np.ndarray, violations: list[dict], time_sec: float,
) -> np.ndarray:
    """同 time の violations を赤枠 + ラベルでマーク。"""
    out = img.copy()
    if out.shape[:2] != (1080, 1920):
        out = cv2.resize(
            out, (1920, 1080), interpolation=cv2.INTER_AREA,
        )
    for vio in violations:
        draw_violation_marker(out, vio)
    # 上部に time + 件数のヘッダ
    header = f"t={time_sec:.2f}s  violations={len(violations)}"
    cv2.rectangle(out, (0, 0), (520, 36), BLACK, -1)
    cv2.putText(
        out, header, (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA,
    )
    return out


def collect_violations(labels_csv: Path) -> list[dict]:
    """labels.csv から hard violation cells を抽出。"""
    out: list[dict] = []
    with labels_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_chain") == "1":
                continue
            reasons = row.get("suspicious_reasons", "")
            if not is_hard_violation(reasons):
                continue
            out.append({
                "id": int(row["id"]),
                "time": float(row["time"]),
                "side": row["side"],
                "row": int(row["row"]),
                "col": int(row["col"]),
                "recognized": row["recognized"],
            })
    return out


def render_segment(segment: str) -> int:
    """1 segment 分の frames_marked/ を生成。"""
    seg_dir = WEAK_ROOT / segment
    labels_csv = seg_dir / "labels.csv"
    frames_dir = seg_dir / "frames"
    out_dir = seg_dir / "violations_review" / "frames_marked"

    if not labels_csv.exists():
        print(f"ERROR: {labels_csv} not found")
        return 1
    video_name = parse_segment_video(segment)
    if video_name is None:
        print(f"ERROR: cannot derive video name from {segment}")
        return 1
    video_path = _ROOT / "data" / "frames" / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    violations = collect_violations(labels_csv)
    if not violations:
        print(f"[render] {segment}: no hard violations")
        return 0

    by_time: dict[float, list[dict]] = {}
    for v in violations:
        by_time.setdefault(v["time"], []).append(v)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: video open failed: {video_path}")
        return 1

    n_written = 0
    for t in sorted(by_time):
        img = load_frame_from_grid(frames_dir, t)
        if img is None:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            img = frame
        marked = render_marked_frame(img, by_time[t], t)
        n_cells = len(by_time[t])
        out_path = out_dir / f"frame_{t:07.2f}_{n_cells:02d}cells.png"
        cv2.imwrite(str(out_path), marked)
        n_written += 1
    cap.release()

    print(
        f"[render] {segment}: {len(violations)} cells / "
        f"{len(by_time)} unique times -> {n_written} PNG"
    )
    print(f"[render] saved: {to_windows_path(out_dir)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment", type=str, required=True,
        help="例: v06_m03_385_415",
    )
    args = parser.parse_args()
    return render_segment(args.segment)


if __name__ == "__main__":
    sys.exit(main())
