"""W9-E: CentroidClassifier 学習 + cross-video 評価。

全 review 済セル (v05_m55_full + v12_m54_full + viol_50_bg/v04..v19 +
W9-B v09/v13/v17/v18/v19 m_full) から centroid を計算、保存。
さらに cross-video harness と同じテストセットで accuracy 比較。

訓練と評価で同じデータを使うのは過適合だが、ベースラインとして:
    - 全データで centroid 計算 → in-sample fit (=真の特性把握)
    - holdout 1 動画で leave-one-out 評価 → 汎化性能評価
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

import cv2
import numpy as np

from src.board import HIDDEN_ROWS
from src.centroid_classifier import CentroidClassifier
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}


def review_sets() -> list[tuple[str, Path, Path]]:
    """(name, csv_path, video_path) のリスト。"""
    base = Path("data/verify/phase_w_review")
    sets: list[tuple[str, Path, Path]] = [
        ("v05_m55_full",
         base / "v05_m55_full" / "labels.csv",
         Path("data/frames/video_05.mp4")),
        ("v12_m54_full",
         base / "v12_m54_full" / "labels.csv",
         Path("data/frames/video_12.mp4")),
    ]
    # W9-B 弱点動画
    for name, vid in (
        ("v09_m02_full", "09"), ("v13_m02_full", "13"),
        ("v17_m11_full", "17"), ("v18_m03_full", "18"),
        ("v18_m08_full", "18"),  # W10-F 追加分
        ("v18_m15_full", "18"),  # W11-B 追加分
        ("v19_m06_full", "19"),
    ):
        sets.append((
            name, base / name / "labels.csv",
            Path(f"data/frames/video_{vid}.mp4"),
        ))
    # violations_50_bg
    for n in range(4, 20):
        v = f"v{n:02d}"
        sets.append((
            f"viol_{v}",
            base / "violations_50_bg" / v / "labels.csv",
            Path(f"data/frames/video_{n:02d}.mp4"),
        ))
    return sets


def collect_patches(
    csv_path: Path, video_path: Path,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """labels.csv + 動画から (patch, code, side) を収集。"""
    if not csv_path.exists() or not video_path.exists():
        return [], [], []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], []
    patches: list[np.ndarray] = []
    truths: list[int] = []
    sides: list[str] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            truth = (row.get("your_answer") or "").strip()
            if not truth or truth not in LABEL_TO_CODE:
                continue
            t = float(row["time"])
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            if fr.shape[:2] != (1080, 1920):
                fr = cv2.resize(fr, (1920, 1080))
            region = (
                DEFAULT_P1_REGION if row["side"] == "1P"
                else DEFAULT_P2_REGION
            )
            r = int(row["row"]) + HIDDEN_ROWS
            c = int(row["col"])
            x1, y1, x2, y2 = region.cell_sample_rect(r, c)
            patch = fr[max(0, y1):min(1080, y2), max(0, x1):min(1920, x2)]
            if patch.size == 0:
                continue
            patches.append(patch.copy())
            truths.append(LABEL_TO_CODE[truth])
            sides.append(row["side"])
    cap.release()
    return patches, truths, sides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-model", default="models/centroid_v1.npz",
    )
    parser.add_argument(
        "--out-tsv", default="data/verify/phase_w_eval_centroid.tsv",
    )
    parser.add_argument(
        "--leave-out-video", default=None,
        help="指定動画を holdout (例: video_19) で残りのデータで学習",
    )
    args = parser.parse_args()

    sets = review_sets()
    all_patches: list[np.ndarray] = []
    all_labels: list[int] = []
    set_owner: list[str] = []
    set_video: list[str] = []
    print("collecting all reviewed patches...")
    for name, csv_p, video_p in sets:
        ps, ls, _ = collect_patches(csv_p, video_p)
        if not ps:
            continue
        print(f"  {name}: {len(ps)}")
        all_patches.extend(ps)
        all_labels.extend(ls)
        set_owner.extend([name] * len(ps))
        set_video.extend([video_p.stem] * len(ps))

    print(f"total: {len(all_patches)}")

    # 学習データ: leave-out 指定なら除外、それ以外全データ
    if args.leave_out_video:
        train_idx = [
            i for i, v in enumerate(set_video) if v != args.leave_out_video
        ]
        print(
            f"holdout {args.leave_out_video}: "
            f"train={len(train_idx)}, "
            f"holdout={len(all_patches) - len(train_idx)}"
        )
    else:
        train_idx = list(range(len(all_patches)))

    cls = CentroidClassifier()
    cls.fit(
        [all_patches[i] for i in train_idx],
        [all_labels[i] for i in train_idx],
        normalize=True,
    )
    print("centroid counts:", cls.counts)

    # 各 review set で評価
    rows: list[str] = ["set\tcorrect\ttotal\taccuracy\ttop_errors"]
    grand_correct = 0
    grand_total = 0
    err_total: dict[tuple[int, int], int] = {}
    for name, _, _ in sets:
        idxs = [i for i, o in enumerate(set_owner) if o == name]
        if not idxs:
            continue
        c = 0
        t = len(idxs)
        cm: dict[tuple[int, int], int] = {}
        for i in idxs:
            pred = cls.classify(all_patches[i])
            truth = all_labels[i]
            if pred == truth:
                c += 1
            cm[(truth, pred)] = cm.get((truth, pred), 0) + 1
            err_total[(truth, pred)] = err_total.get((truth, pred), 0) + 1
        acc = c / max(1, t)
        err_pairs = sorted(
            ((k, v) for k, v in cm.items() if k[0] != k[1]),
            key=lambda kv: -kv[1],
        )[:5]
        err_str = " ".join(
            f"{CODE_TO_LABEL[k[0]]}->{CODE_TO_LABEL[k[1]]}:{v}"
            for k, v in err_pairs
        ) or "(no errors)"
        rows.append(f"{name}\t{c}\t{t}\t{acc:.4f}\t{err_str}")
        print(f"  {name}: {c}/{t} = {acc:.4f}  {err_str}")
        grand_correct += c
        grand_total += t

    grand_acc = grand_correct / max(1, grand_total)
    rows.append(f"TOTAL\t{grand_correct}\t{grand_total}\t{grand_acc:.4f}\t-")
    print(f"\nTOTAL: {grand_correct}/{grand_total} = {grand_acc:.4f}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print(f"saved tsv: {to_windows_path(out_path)}")

    out_model = Path(args.out_model)
    cls.save(out_model)
    print(f"saved model: {to_windows_path(out_model)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
