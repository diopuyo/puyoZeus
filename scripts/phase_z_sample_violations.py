"""Phase Z: 各動画の violations から 20 件サンプリングして統合シート生成。

cross_video/v??_m_*/violations_review/violations.csv から各動画 20 件を
ランダム抽出 (seed=42 固定) し、動画別に並べた 1 つのシート画像と
統合 CSV を生成する。ユーザーが各動画の真 accuracy を統計的に推定可能。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_sample_violations \
        --samples-per-video 20 \
        --out-dir data/verify/phase_z_review/sampled_review
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
SHEET_PAD = 8
LABEL_HEIGHT = 70
HEADER_HEIGHT = 30  # 動画 ID 行の高さ


def load_violations(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def sample_violations(
    rows: list[dict], n: int, seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    if len(rows) <= n:
        return list(rows)
    idxs = rng.choice(len(rows), n, replace=False)
    return [rows[i] for i in sorted(idxs.tolist())]


def extract_patch(
    video_path: Path, t: float, side: str, row: int, col: int,
) -> np.ndarray | None:
    """指定 frame・cell の patch を切り出し。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(
            frame, (1920, 1080), interpolation=cv2.INTER_AREA,
        )
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(row + HIDDEN_ROWS, col)
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return frame[y1:y2, x1:x2].copy()


def get_video_path(video_dir_name: str) -> Path | None:
    """v01_m_259_289 → data/frames/video_01.mp4"""
    parts = video_dir_name.split("_")
    if not parts[0].startswith("v"):
        return None
    try:
        vid = int(parts[0][1:])
    except ValueError:
        return None
    p = _ROOT / f"data/frames/video_{vid:02d}.mp4"
    return p if p.exists() else None


def build_sheet(
    samples_per_video: dict[str, list[dict]],
    cell_w: int = SAMPLE_PX,
    cell_h: int = SAMPLE_PX + LABEL_HEIGHT,
) -> np.ndarray:
    """各動画 1 行 (20 cells) で並べたシート。"""
    n_videos = len(samples_per_video)
    if n_videos == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    max_n = max(len(s) for s in samples_per_video.values())
    cell_w_pad = cell_w + SHEET_PAD * 2
    cell_h_pad = cell_h + SHEET_PAD * 2 + HEADER_HEIGHT
    sheet_w = cell_w_pad * max_n
    sheet_h = cell_h_pad * n_videos
    sheet = np.full((sheet_h, sheet_w, 3), 18, dtype=np.uint8)
    for vi, (vname, samples) in enumerate(
        sorted(samples_per_video.items()),
    ):
        y_base = vi * cell_h_pad + SHEET_PAD
        # ヘッダ (動画名)
        cv2.putText(
            sheet, vname, (10, y_base + 20),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (220, 220, 80), 1, cv2.LINE_AA,
        )
        for ci, s in enumerate(samples):
            x0 = ci * cell_w_pad + SHEET_PAD
            y0 = y_base + HEADER_HEIGHT
            patch = s.get("_patch")
            if patch is not None and patch.size > 0:
                resized = cv2.resize(
                    patch, (cell_w, cell_w),
                    interpolation=cv2.INTER_CUBIC,
                )
                sheet[y0:y0 + cell_w, x0:x0 + cell_w] = resized
            # ラベル背景
            bg_y0 = y0 + cell_w
            bg_y1 = bg_y0 + LABEL_HEIGHT
            rec = s["recognized"]
            bg = COLOR_BG.get(rec, (60, 60, 60))
            sheet[bg_y0:bg_y1, x0:x0 + cell_w] = bg
            text_color = (
                (255, 255, 255) if sum(bg) / 3 < 128 else (0, 0, 0)
            )
            lines = [
                f"#{ci+1} {s['side']}r{s['row']}c{s['col']}",
                f"{rec} t={float(s['time']):.1f}",
            ]
            for li, line in enumerate(lines):
                cv2.putText(
                    sheet, line, (x0 + 2, bg_y0 + 16 + li * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    text_color, 1, cv2.LINE_AA,
                )
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross-dir",
        type=Path,
        default=_ROOT / "data/verify/phase_z_review/cross_video",
    )
    parser.add_argument("--samples-per-video", type=int, default=20)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "data/verify/phase_z_review/sampled_review",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 30s 区間の動画ディレクトリのみ対象 (smoketest 10s 版は除外)
    video_dirs = []
    for d in sorted(args.cross_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("v"):
            continue
        # 命名規則 vNN_m_START_END (END-START >= 20 で 30s 版とみなす)
        parts = d.name.split("_")
        try:
            start = int(parts[2])
            end = int(parts[3])
        except (IndexError, ValueError):
            continue
        if end - start < 20:
            continue  # 10s smoketest を除外
        violations_csv = d / "violations_review" / "violations.csv"
        if not violations_csv.exists():
            continue
        video_dirs.append((d.name, violations_csv))

    print(f"対象動画: {len(video_dirs)} 個 (30s 区間)")
    samples_per_video: dict[str, list[dict]] = {}
    all_sampled: list[dict] = []
    sample_id = 0
    for vname, csv_path in video_dirs:
        rows = load_violations(csv_path)
        sampled = sample_violations(rows, args.samples_per_video)
        video_path = get_video_path(vname)
        if video_path is None:
            print(f"[skip] {vname}: 動画 file なし")
            continue
        # 各 sample に patch を埋め込む
        enriched = []
        for r in sampled:
            patch = extract_patch(
                video_path, float(r["time"]), r["side"],
                int(r["row"]), int(r["col"]),
            )
            sample_id += 1
            r["_patch"] = patch
            r["_video"] = vname
            r["_global_id"] = sample_id
            enriched.append(r)
            all_sampled.append(r)
        samples_per_video[vname] = enriched
        print(f"  {vname}: {len(enriched)} cells")

    # シート生成
    sheet = build_sheet(samples_per_video)
    sheet_path = args.out_dir / "sampled_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"saved: {to_windows_path(sheet_path)}")

    # 統合 CSV (your_answer 入力用)
    csv_path = args.out_dir / "sampled.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "global_id", "video", "time", "side", "row", "col",
            "recognized", "your_answer", "reasons",
        ])
        for r in all_sampled:
            writer.writerow([
                r["_global_id"], r["_video"], r["time"], r["side"],
                r["row"], r["col"], r["recognized"], "",
                r["reasons"],
            ])
    print(f"saved: {to_windows_path(csv_path)}")

    # HTML 一覧
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Phase Z sampled review</title>",
        "<style>body{font-family:sans-serif;background:#222;color:#ddd;}",
        "table{border-collapse:collapse;width:100%;}",
        "td,th{border:1px solid #444;padding:4px 8px;font-size:12px;}",
        ".v{background:#234;color:#ff8;font-weight:bold;}</style>",
        "</head><body>",
        f"<h1>Phase Z sampled review "
        f"({len(all_sampled)} cells from {len(samples_per_video)} videos)</h1>",
        "<p>各動画 {} cells × {} 動画 = {} cells を統計サンプリング。"
        "your_answer に E/R/B/G/Y/P/O 入力。</p>"
        .format(args.samples_per_video, len(samples_per_video),
                len(all_sampled)),
        f"<p><a href='sampled_sheet.png'>sampled_sheet.png</a> | "
        f"<a href='sampled.csv'>sampled.csv</a></p>",
        "<table>",
        "<tr><th>id</th><th>video</th><th>t</th><th>cell</th>"
        "<th>rec</th><th>reasons</th></tr>",
    ]
    cur_v = None
    for r in all_sampled:
        v = r["_video"]
        if v != cur_v:
            html.append(
                f"<tr class='v'><td colspan='6'>{v}</td></tr>"
            )
            cur_v = v
        rec = r["recognized"]
        bg = COLOR_BG.get(rec, (60, 60, 60))
        bg_css = f"rgb({bg[2]},{bg[1]},{bg[0]})"
        html.append(
            f"<tr><td>{r['_global_id']}</td><td>{v}</td>"
            f"<td>{float(r['time']):.2f}</td>"
            f"<td>{r['side']} r{r['row']} c{r['col']}</td>"
            f"<td style='background:{bg_css};color:white;text-align:center'>"
            f"{rec}</td>"
            f"<td>{r['reasons']}</td></tr>"
        )
    html.append("</table></body></html>")
    (args.out_dir / "sampled.html").write_text(
        "\n".join(html), encoding="utf-8",
    )
    print(f"saved: {to_windows_path(args.out_dir / 'sampled.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
