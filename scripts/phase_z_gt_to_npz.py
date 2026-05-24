"""Phase Z で取得したユーザー GT を patch + label の npz に変換 (CNN v17 訓練用)。

GT ソース:
    1. phase_z_aggregate_sampled.py USER_REVIEW (18 動画 × 20 cells = 360)
    2. phase_z_compare_gt.py GT_FRAMES (v18_m03 3 frame × 144 = 432)

各 cell について:
    - 該当 video.mp4 の指定 t_sec frame を読み込み
    - region.cell_sample_rect で patch を切り出し
    - (patch, label_idx) のペアを numpy array に格納

出力:
    data/training_phase_u/phase_z_gt.npz (X, y)

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_gt_to_npz
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()

import csv  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.board import (  # noqa: E402
    BOARD_COLS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW, HIDDEN_ROWS,
)
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION, DEFAULT_P2_REGION,
)
from src.patch_classifier import (  # noqa: E402
    CLASS_INDEX_TO_COLOR, COLOR_TO_CLASS_INDEX,
)

CHAR_TO_COLOR = {
    "E": COLOR_EMPTY, "R": COLOR_RED, "B": COLOR_BLUE,
    "G": COLOR_GREEN, "Y": COLOR_YELLOW, "P": COLOR_PURPLE,
    "O": COLOR_OJAMA,
}

# === 1. Sampled review (各動画 20 件) ===
SAMPLED_REVIEW: dict[str, str] = {
    "v01": "RGYEGGGYBBBGGBGYRRYE",
    "v02": "YGEBYBEEEEGEEEOEEYEE",
    "v03": "EERGEEPEEEEEEREEEEEE",
    "v04": "BRYYYYYYYRYYYYYGYYYY",
    "v05": "GBEGBEEGEEPEEERBRBPE",
    "v06": "YYBYYRBBYBREEEEEEEEE",
    "v07": "RRYBRGGYRYREBYBYRYYR",
    "v08": "YRBRYYYPRYYPYYRRPYOY",
    "v09": "BBPEEEEGEGBEPEPBBEEE",
    "v10": "EEEEEEEPRPYEEEEEPGRR",
    "v11": "REBEPYPBPBRYRYEPEEYP",
    "v12": "PYGYYYPRYYYBRYYRYRYE",
    "v13": "YRRYEEEEPEYPRBBBREYR",
    "v14": "YPGBGYBGGBYYPBEPGYGB",
    "v15": "EBRPGGPOPRRGGGPRPGPP",
    "v16": "GGPGBBGOGGYEEEEEEYGY",
    "v17": "GPRYPPYEEEEEEEEEEEEE",
    "v19": "BBYBBGGPGGGGBGGYGGGY",
}
SAMPLED_EXCLUDED: dict[str, set[int]] = {
    "v02": {6, 16, 17, 19, 20},
    "v05": {2},
    "v08": {15, 17, 18, 19},
    "v11": {18},
    "v16": {12, 13, 14},
}

# === 2. v18_m03 281.70 / 290.20 / 305.20 GT ===
V18_GT: dict[str, dict[str, list[str]]] = {
    "281.70": {
        "1P": [
            "EEEEEE", "EEEEEE", "EEEEEE", "EEEEEE",
            "EEEERY", "GEEEPP", "YEERPY", "GGGPGY",
            "YYYGPY", "PGRYPR", "PPGRPR", "GGRRYR",
        ],
        "2P": [
            "EEEEEE", "EEEEEE", "EEEEEO", "EEEEEG",
            "OREGEG", "RYEYGP", "ROOOPG", "RPPPGG",
            "YYYGYY", "RGPYRY", "RRGPPR", "GGPYRR",
        ],
    },
    "290.20": {
        "1P": [
            "EEEEPG", "OEERPG", "REEPGP", "POEPRY",
            "YGERRY", "GYEOPP", "YYORPY", "GGGPGY",
            "YYYGPY", "PGRYPR", "PPGRPR", "GGRRYR",
        ],
        "2P": [
            "YEEEEE", "YEEERP", "PGEERP", "PREEYP",
            "OREEGG", "RYYYPP", "ROOOPG", "RPPPGG",
            "YYYGYY", "RGPYRY", "RRGPPR", "GGPYRR",
        ],
    },
    "305.20": {
        "1P": [
            "EEEEEP", "OPEGEY", "RPEYPG", "POEYPR",
            "YGERPR", "GYERYP", "YYERYP", "GGGPPR",
            "YYYGGY", "PGRYPR", "PPGRPR", "GGRRYR",
        ],
    },
}


def get_video_path(vid: str) -> Path | None:
    p = _ROOT / f"data/frames/video_{int(vid[1:]):02d}.mp4"
    return p if p.exists() else None


def read_frame_at(video_path: Path, t_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(
            frame, (1920, 1080), interpolation=cv2.INTER_AREA,
        )
    return frame


def extract_cell_patch(
    frame: np.ndarray, side: str, vrow: int, col: int,
) -> np.ndarray:
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region.cell_sample_rect(vrow + HIDDEN_ROWS, col)
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return frame[y1:y2, x1:x2].copy()


def collect_sampled_gt() -> list[tuple[np.ndarray, int]]:
    """Sampled review 18 動画 × 20 cells から (patch, color) 抽出。"""
    samples: list[tuple[np.ndarray, int]] = []
    csv_path = (
        _ROOT
        / "data/verify/phase_z_review/sampled_review/sampled.csv"
    )
    if not csv_path.exists():
        print("ERROR: sampled.csv not found")
        return []
    rows_by_video: dict[str, list[dict]] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vid = r["video"].split("_")[0]
            rows_by_video.setdefault(vid, []).append(r)

    for vid, user_str in SAMPLED_REVIEW.items():
        rows = rows_by_video.get(vid, [])
        if not rows:
            continue
        excluded = SAMPLED_EXCLUDED.get(vid, set())
        video_path = get_video_path(vid)
        if video_path is None:
            print(f"[skip] {vid}: video file 未発見")
            continue
        # frame 読み込みは t_sec 単位でまとめる (高速化)
        for i, r in enumerate(rows):
            if (i + 1) in excluded:
                continue
            if i >= len(user_str):
                continue
            ch = user_str[i].upper()
            color = CHAR_TO_COLOR.get(ch)
            if color is None:
                continue
            frame = read_frame_at(video_path, float(r["time"]))
            if frame is None:
                continue
            patch = extract_cell_patch(
                frame, r["side"], int(r["row"]), int(r["col"]),
            )
            samples.append((patch, color))
    return samples


def collect_v18_gt() -> list[tuple[np.ndarray, int]]:
    """v18_m03 281.70/290.20/305.20 GT (12 行 × 6 列)。"""
    samples: list[tuple[np.ndarray, int]] = []
    video_path = get_video_path("v18")
    if video_path is None:
        return []
    for t_str, sides in V18_GT.items():
        t = float(t_str)
        frame = read_frame_at(video_path, t)
        if frame is None:
            continue
        for side, grid in sides.items():
            for vrow, line in enumerate(grid):
                for col, ch in enumerate(line):
                    color = CHAR_TO_COLOR.get(ch)
                    if color is None:
                        continue
                    patch = extract_cell_patch(frame, side, vrow, col)
                    samples.append((patch, color))
    return samples


def main() -> int:
    from src.patch_classifier import patch_to_feature
    print("[1/3] Sampled review 抽出 (18 動画 × 20 cells)...")
    sampled = collect_sampled_gt()
    print(f"  {len(sampled)} cells 取得")

    print("[2/3] v18_m03 reviewed cells 抽出 (281.70/290.20/305.20)...")
    v18 = collect_v18_gt()
    print(f"  {len(v18)} cells 取得")

    print(f"\n総計: {len(sampled) + len(v18)} cells "
          f"(sampled {len(sampled)} + v18 {len(v18)})")

    # Class 別件数
    from collections import Counter
    counts_all = Counter(c for _, c in sampled + v18)
    name = {
        COLOR_EMPTY: "EM", COLOR_RED: "RED", COLOR_BLUE: "BLUE",
        COLOR_GREEN: "GRN", COLOR_YELLOW: "YEL", COLOR_PURPLE: "PUR",
        COLOR_OJAMA: "OJM",
    }
    for c, n in sorted(counts_all.items()):
        print(f"  {name.get(c, c)}: {n}")

    # patch_to_feature と同じく 8x8 にリサイズ
    print("\n[3/3] 8x8 にリサイズして 2 つの npz に分割保存...")

    def to_arr(samples: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        for patch, color in samples:
            if patch.size == 0:
                continue
            X.append(patch)
            y.append(COLOR_TO_CLASS_INDEX[color])
        if not X:
            return (np.empty((0, 8, 8, 3), dtype=np.uint8),
                    np.empty((0,), dtype=np.int64))
        X_arr = np.array([
            cv2.resize(p, (8, 8), interpolation=cv2.INTER_AREA)
            for p in X
        ], dtype=np.uint8)
        y_arr = np.array(y, dtype=np.int64)
        return X_arr, y_arr

    sampled_X, sampled_y = to_arr(sampled)
    v18_X, v18_y = to_arr(v18)
    all_X = np.concatenate([sampled_X, v18_X], axis=0)
    all_y = np.concatenate([sampled_y, v18_y], axis=0)

    # 互換: 全部入り phase_z_gt.npz (X, y)
    full_path = _ROOT / "data/training_phase_u/phase_z_gt.npz"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(full_path, X=all_X, y=all_y)
    print(f"  saved (full): {to_windows_path(full_path)} {all_X.shape}")

    # 動画別 split (v17b 訓練用、v18 抑制で過学習回避)
    sampled_path = _ROOT / "data/training_phase_u/phase_z_gt_sampled.npz"
    np.savez_compressed(sampled_path, X=sampled_X, y=sampled_y)
    print(f"  saved (sampled 18 動画分): "
          f"{to_windows_path(sampled_path)} {sampled_X.shape}")

    v18_path = _ROOT / "data/training_phase_u/phase_z_gt_v18.npz"
    np.savez_compressed(v18_path, X=v18_X, y=v18_y)
    print(f"  saved (v18_m03 専用): "
          f"{to_windows_path(v18_path)} {v18_X.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
