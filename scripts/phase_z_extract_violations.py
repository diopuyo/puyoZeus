"""Phase Z 残 hard violation を人間レビュー用に抽出。

連続 frame 自動評価で hard violation とフラグされた cells について、
元動画 frame から patch を切り出し、cell sheet + CSV + HTML を生成。

入力:
    --labels: phase_z_review_ui.py が生成した labels.csv
    --video:  元動画 (frame patch 切り出し用)

出力 (out_dir/violations_review/):
    - violations_sheet.png: 全 hard violation cells の patch 一覧
    - violations.csv:        cell 情報 + your_answer 入力欄 (E/R/B/G/Y/P/O)
    - violations.html:       一覧 + 該当 frame へのリンク

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_extract_violations \
        --labels data/verify/phase_z_review/v18_m03_30_60/labels.csv \
        --video data/frames/video_18.mp4 \
        --out-dir data/verify/phase_z_review/v18_m03_30_60/violations_review
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.board import (  # noqa: E402
    BOARD_COLS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW, HIDDEN_ROWS,
)
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION,
)

HARD_VIOLATION_REASONS: tuple[str, ...] = (
    "color_swap", "disappearance", "airborne", "hidden_below",
    "empty_in_stack", "pair_mismatch", "solo_appearance",
    "hsv_disagree", "unknown_drop",
)

COLOR_BG: dict[str, tuple[int, int, int]] = {
    "EM": (40, 40, 40),
    "RED": (40, 40, 200),
    "BLUE": (200, 80, 40),
    "GRN": (40, 180, 40),
    "YEL": (40, 200, 220),
    "PUR": (180, 40, 180),
    "OJM": (170, 170, 170),
    "??": (80, 80, 120),
}

SAMPLE_PX = 110
SHEET_COLS = 12  # 1 行 12 cell
SHEET_PAD = 8
LABEL_HEIGHT = 70


def parse_reasons(s: str) -> list[str]:
    out = []
    for r in s.split(";"):
        r = r.strip()
        if not r:
            continue
        out.append(r.split("(", 1)[0])
    return out


def is_hard_violation(reasons_str: str) -> bool:
    return any(r in HARD_VIOLATION_REASONS for r in parse_reasons(reasons_str))


def extract_violation_patches(
    labels_csv: Path, video_path: Path,
) -> list[dict]:
    """labels.csv から hard violation cell を抽出し patch を切り出す。"""
    # まず CSV から該当 cell のリストを作る
    violations: list[dict] = []
    with labels_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("is_chain") == "1":
                continue
            reasons = r.get("suspicious_reasons", "")
            if not is_hard_violation(reasons):
                continue
            violations.append({
                "id": int(r["id"]),
                "time": float(r["time"]),
                "side": r["side"],
                "row": int(r["row"]),
                "col": int(r["col"]),
                "recognized": r["recognized"],
                "conf": r.get("conf", ""),
                "reasons": reasons,
            })
    if not violations:
        return []

    # 動画から各 frame を 1 度だけ読み込んで patch を切り出す
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"video open failed: {video_path}")
    by_time: dict[float, list[dict]] = {}
    for v in violations:
        by_time.setdefault(v["time"], []).append(v)
    for t, group in sorted(by_time.items()):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            for v in group:
                v["patch"] = np.zeros((50, 50, 3), dtype=np.uint8)
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        h, w = frame.shape[:2]
        for v in group:
            region = (
                DEFAULT_P1_REGION if v["side"] == "1P" else DEFAULT_P2_REGION
            )
            row = v["row"] + HIDDEN_ROWS
            x1, y1, x2, y2 = region.cell_sample_rect(row, v["col"])
            x1 = max(0, min(x1, w - 1))
            x2 = max(x1 + 1, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(y1 + 1, min(y2, h))
            v["patch"] = frame[y1:y2, x1:x2].copy()
    cap.release()
    return violations


def build_sheet(violations: list[dict]) -> np.ndarray:
    """全 hard violation cells を grid で並べた sheet 画像。"""
    n = len(violations)
    cols = min(SHEET_COLS, n)
    rows = (n + cols - 1) // cols
    cell_w = SAMPLE_PX + SHEET_PAD * 2
    cell_h = SAMPLE_PX + LABEL_HEIGHT + SHEET_PAD * 2
    sheet = np.full(
        (rows * cell_h, cols * cell_w, 3), 18, dtype=np.uint8,
    )
    for k, v in enumerate(violations):
        gr = k // cols
        gc = k % cols
        x0 = gc * cell_w + SHEET_PAD
        y0 = gr * cell_h + SHEET_PAD
        patch = v["patch"]
        if patch.size == 0:
            continue
        resized = cv2.resize(
            patch, (SAMPLE_PX, SAMPLE_PX),
            interpolation=cv2.INTER_CUBIC,
        )
        sheet[y0:y0 + SAMPLE_PX, x0:x0 + SAMPLE_PX] = resized
        # 認識色のラベル背景
        bg_y0 = y0 + SAMPLE_PX
        bg_y1 = bg_y0 + LABEL_HEIGHT
        rec = v["recognized"]
        bg = COLOR_BG.get(rec, (60, 60, 60))
        sheet[bg_y0:bg_y1, x0:x0 + SAMPLE_PX] = bg
        text_color = (
            (255, 255, 255) if sum(bg) / 3 < 128 else (0, 0, 0)
        )
        # ラベル: id, side r,c, recognized, time
        lines = [
            f"#{v['id']} {v['side']}r{v['row']}c{v['col']}",
            f"{rec} t={v['time']:.1f}",
            v["reasons"][:18],
        ]
        for li, line in enumerate(lines):
            cv2.putText(
                sheet, line, (x0 + 2, bg_y0 + 16 + li * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_color,
                1, cv2.LINE_AA,
            )
    return sheet


def write_csv(violations: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "time", "side", "row", "col",
            "recognized", "your_answer", "reasons",
        ])
        for v in violations:
            writer.writerow([
                v["id"], f"{v['time']:.2f}", v["side"],
                v["row"], v["col"], v["recognized"],
                "", v["reasons"],
            ])


def write_html(
    violations: list[dict], out_dir: Path,
    frames_dir: Path | None,
) -> None:
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Phase Z hard violations review</title>",
        "<style>body{font-family:sans-serif;background:#222;color:#ddd;}",
        "table{border-collapse:collapse;width:100%;}",
        "td,th{border:1px solid #444;padding:6px 10px;font-size:13px;}",
        "img.thumb{max-height:120px;}",
        "img.frame{max-height:300px;}",
        ".rec{padding:2px 8px;border-radius:4px;font-weight:bold;}</style>",
        "</head><body>",
        f"<h1>Phase Z hard violations ({len(violations)} cells)</h1>",
        "<p>各 cell の真の色を char-code (E/R/B/G/Y/P/O) で violations.csv の "
        "your_answer 列に入力してください。</p>",
        "<p><a href='violations_sheet.png'>violations_sheet.png (全 cell 一覧)</a></p>",
        "<table>",
        "<tr><th>id</th><th>t</th><th>side r,c</th><th>rec</th>"
        "<th>reasons</th><th>frame</th></tr>",
    ]
    for v in violations:
        ms = int(v["time"] * 1000)
        frame_link = ""
        if frames_dir is not None:
            # 0.5s grid に近い frame
            ms_round = int(round(v["time"] * 2) * 500)
            frame_path = frames_dir / f"{ms_round:06d}.png"
            if frame_path.exists():
                frame_link = (
                    f"<a href='../frames/{frame_path.name}' target='_blank'>"
                    f"<img class='frame' src='../frames/{frame_path.name}'></a>"
                )
        rec_bg = COLOR_BG.get(v["recognized"], (60, 60, 60))
        rec_color_css = (
            f"rgb({rec_bg[2]},{rec_bg[1]},{rec_bg[0]})"
        )
        html.append(
            f"<tr><td>{v['id']}</td><td>{v['time']:.2f}</td>"
            f"<td>{v['side']} r{v['row']} c{v['col']}</td>"
            f"<td><span class='rec' style='background:{rec_color_css};color:white'>"
            f"{v['recognized']}</span></td>"
            f"<td>{v['reasons']}</td>"
            f"<td>{frame_link}</td></tr>"
        )
    html.append("</table></body></html>")
    (out_dir / "violations.html").write_text(
        "\n".join(html), encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[VIOL] extracting hard violations...")
    violations = extract_violation_patches(args.labels, args.video)
    print(f"[VIOL] {len(violations)} hard violations found")

    if not violations:
        print("[VIOL] 違反なし、終了")
        return 0

    sheet = build_sheet(violations)
    sheet_path = args.out_dir / "violations_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"[VIOL] saved: {to_windows_path(sheet_path)}")

    csv_path = args.out_dir / "violations.csv"
    write_csv(violations, csv_path)
    print(f"[VIOL] saved: {to_windows_path(csv_path)}")

    # frames ディレクトリは labels.csv と同階層を想定
    frames_dir = args.labels.parent / "frames"
    if not frames_dir.exists():
        frames_dir = None
    write_html(violations, args.out_dir, frames_dir)
    print(f"[VIOL] saved: {to_windows_path(args.out_dir / 'violations.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
