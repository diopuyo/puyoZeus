"""動画別 HSV 統計を集計 (Phase Z-3I 候補、A: HSV 範囲動画別調整用)。

GT ソース:
    1. data/verify/phase_w_review/ 配下の labels.csv (your_answer 列)
    2. data/verify/phase_z_review/sampled_review/sampled.csv (USER_REVIEW)
    3. v18_m03 phase_z_compare_gt の固定 GT

各 cell について patch を切り出し、HSV 中央値・平均値を計算。
動画 ID × color ごとに統計をまとめ、動画別 HSV 範囲算出に使う。

出力:
    data/training_phase_u/per_video_hsv_stats.json
        {video_id: {color: {h_mean, h_std, s_mean, s_std, v_mean, v_std, n}}}
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
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

LABEL_TO_CODE: dict[str, int] = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}
CODE_TO_NAME: dict[int, str] = {
    0: "EM", 1: "RED", 2: "BLUE", 3: "GRN",
    4: "YEL", 5: "PUR", 9: "OJM",
}

CHAR_TO_CODE: dict[str, int] = {
    "E": 0, "R": 1, "B": 2, "G": 3, "Y": 4, "P": 5, "O": 9,
}


def get_video_path(video_label: str) -> Path | None:
    """video_05 / v05 / video_05.mp4 のいずれかから path を解決。"""
    if video_label.endswith(".mp4"):
        p = _ROOT / f"data/frames/{video_label}"
    elif video_label.startswith("v"):
        try:
            vid = int(video_label[1:].split("_")[0])
        except ValueError:
            return None
        p = _ROOT / f"data/frames/video_{vid:02d}.mp4"
    elif video_label.startswith("video_"):
        p = _ROOT / f"data/frames/{video_label}.mp4"
    else:
        return None
    return p if p.exists() else None


def collect_from_csv(
    csv_path: Path, video_path: Path,
) -> list[tuple[int, int, np.ndarray]]:
    """labels.csv の (your_answer, side, row, col, time) → patch + label."""
    samples: list[tuple[int, int, np.ndarray]] = []
    if not csv_path.exists() or not video_path.exists():
        return samples
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return samples
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ans = (row.get("your_answer") or "").strip()
            if not ans or ans not in LABEL_TO_CODE:
                continue
            try:
                t = float(row["time"])
                side = row["side"]
                vrow = int(row["row"])
                col = int(row["col"])
            except (KeyError, ValueError):
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080),
                                   interpolation=cv2.INTER_AREA)
            region = (
                DEFAULT_P1_REGION if side == "1P"
                else DEFAULT_P2_REGION
            )
            x1, y1, x2, y2 = region.cell_sample_rect(
                vrow + HIDDEN_ROWS, col,
            )
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(1920, x2); y2 = min(1080, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            patch = frame[y1:y2, x1:x2].copy()
            samples.append((LABEL_TO_CODE[ans], 0, patch))
    cap.release()
    return samples


def compute_stats(samples: list[np.ndarray]) -> dict:
    """patch list から HSV mean/std を集計。"""
    if not samples:
        return {}
    h_vals, s_vals, v_vals = [], [], []
    for p in samples:
        if p.size == 0:
            continue
        # 中央 50% を取る
        ph, pw = p.shape[:2]
        cy0 = ph // 4
        cy1 = ph - ph // 4
        cx0 = pw // 4
        cx1 = pw - pw // 4
        inner = p[cy0:cy1, cx0:cx1]
        if inner.size == 0:
            continue
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        h_vals.append(float(np.median(hsv[:, :, 0])))
        s_vals.append(float(np.median(hsv[:, :, 1])))
        v_vals.append(float(np.median(hsv[:, :, 2])))
    if not h_vals:
        return {}
    return {
        "n": len(h_vals),
        "h_mean": float(np.mean(h_vals)),
        "h_std": float(np.std(h_vals)),
        "s_mean": float(np.mean(s_vals)),
        "s_std": float(np.std(s_vals)),
        "v_mean": float(np.mean(v_vals)),
        "v_std": float(np.std(v_vals)),
        "h_p10": float(np.percentile(h_vals, 10)),
        "h_p90": float(np.percentile(h_vals, 90)),
        "s_p10": float(np.percentile(s_vals, 10)),
        "s_p90": float(np.percentile(s_vals, 90)),
        "v_p10": float(np.percentile(v_vals, 10)),
        "v_p90": float(np.percentile(v_vals, 90)),
    }


def main() -> int:
    # video_id → color_code → list[patch]
    by_video_color: dict[str, dict[int, list[np.ndarray]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    # 1. data/verify/phase_w_review/ 配下
    review_specs = [
        ("v04_m07_full", "video_04"),
        ("v05_m55_full", "video_05"),
        ("v05_m55_uncertain", "video_05"),
        ("v06_m06_full", "video_06"),
        ("v09_m02_full", "video_09"),
        ("v12_m54_full", "video_12"),
        ("v13_m02_full", "video_13"),
        ("v17_m11_full", "video_17"),
        ("v17_m37_full", "video_17"),
        ("v18_m03_full", "video_18"),
        ("v18_m08_full", "video_18"),
        ("v18_m15_full", "video_18"),
        ("v19_m06_full", "video_19"),
        ("v19_m07_full", "video_19"),
    ]
    for n in range(4, 20):
        review_specs.append(
            (f"violations_50/v{n:02d}", f"video_{n:02d}"),
        )
        review_specs.append(
            (f"violations_50_bg/v{n:02d}", f"video_{n:02d}"),
        )

    base = _ROOT / "data/verify/phase_w_review"
    for name, vid in review_specs:
        csv_p = base / name / "labels.csv"
        vid_p = get_video_path(vid)
        if vid_p is None:
            continue
        samples = collect_from_csv(csv_p, vid_p)
        if not samples:
            continue
        for code, _, patch in samples:
            by_video_color[vid][code].append(patch)
        print(f"  {name} ({vid}): {len(samples)} cells")

    # 2. sampled review
    sampled_csv = (
        _ROOT
        / "data/verify/phase_z_review/sampled_review/sampled.csv"
    )
    if sampled_csv.exists():
        # USER_REVIEW + EXCLUDED は phase_z_aggregate_sampled.py から再利用
        from scripts.phase_z_aggregate_sampled import (
            USER_REVIEW, EXCLUDED, CHAR_TO_LABEL,
        )
        rows_by_video: dict[str, list[dict]] = defaultdict(list)
        with sampled_csv.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vid = r["video"].split("_")[0]
                rows_by_video[vid].append(r)
        for vid, user_str in USER_REVIEW.items():
            rows = rows_by_video.get(vid, [])
            excluded = EXCLUDED.get(vid, set())
            video_path = get_video_path(vid)
            if video_path is None:
                continue
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue
            for i, r in enumerate(rows):
                if (i + 1) in excluded:
                    continue
                if i >= len(user_str):
                    continue
                ch = user_str[i].upper()
                label = CHAR_TO_LABEL.get(ch)
                if label is None:
                    continue
                code = LABEL_TO_CODE.get(label)
                if code is None:
                    continue
                t = float(r["time"])
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if frame.shape[:2] != (1080, 1920):
                    frame = cv2.resize(frame, (1920, 1080),
                                       interpolation=cv2.INTER_AREA)
                region = (
                    DEFAULT_P1_REGION if r["side"] == "1P"
                    else DEFAULT_P2_REGION
                )
                vrow = int(r["row"])
                col = int(r["col"])
                x1, y1, x2, y2 = region.cell_sample_rect(
                    vrow + HIDDEN_ROWS, col,
                )
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(1920, x2); y2 = min(1080, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                patch = frame[y1:y2, x1:x2].copy()
                # 動画 ID は v01..v19 形式 → video_01..video_19 に正規化
                vid_norm = f"video_{int(vid[1:]):02d}"
                by_video_color[vid_norm][code].append(patch)
            cap.release()
        print(f"  sampled review: 各動画 ~20 cells 統合")

    # 統計生成
    stats: dict[str, dict[str, dict]] = {}
    for video, color_dict in sorted(by_video_color.items()):
        stats[video] = {}
        for code, patches in sorted(color_dict.items()):
            s = compute_stats(patches)
            if s:
                stats[video][CODE_TO_NAME[code]] = s

    # 出力
    out_path = _ROOT / "data/training_phase_u/per_video_hsv_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {to_windows_path(out_path)}")
    print(f"対象動画: {len(stats)}")
    for v, cd in stats.items():
        n_total = sum(c.get("n", 0) for c in cd.values())
        print(f"  {v}: {n_total} cells, "
              f"colors {sorted(cd.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
