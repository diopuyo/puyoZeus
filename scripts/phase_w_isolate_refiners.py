"""W16-A: 各 refiner の単独効果を field2 で切り分け。

各 refiner を 1 つだけ OFF にして production accuracy を測る。
最も accuracy が「上がる」OFF 構成 = その refiner が悪さしている。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console  # noqa: E402
init_console()

import cv2
import numpy as np

from src.board import HIDDEN_ROWS
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.state_pipeline import StatePipeline


LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}
CODE_TO_LABEL[10] = "UN"  # COLOR_UNKNOWN

CSV_PATH = "data/verify/phase_w_review/v18_m03_field2/labels.csv"
VIDEO_PATH = "data/frames/video_18.mp4"
BG_FP_TIME = 251.0


def load_labels() -> dict[tuple[float, str, int, int], int]:
    """labels.csv から (time, side, vrow, col) → truth code を作る。"""
    truth: dict[tuple[float, str, int, int], int] = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ans = (row.get("your_answer") or "").strip()
            if not ans or ans not in LABEL_TO_CODE:
                continue
            t = float(row["time"])
            side = row["side"]
            vrow = int(row["row"])
            col = int(row["col"])
            truth[(t, side, vrow, col)] = LABEL_TO_CODE[ans]
    return truth


def evaluate_config(
    label_truth: dict[tuple[float, str, int, int], int],
    config: dict,
) -> tuple[int, int, dict]:
    """指定 refiner config で field2 の 2 frame を独立評価。"""
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return 0, 0, {}

    pipeline = StatePipeline(
        cnn_model_path="models/cnn_phase_u_v16.pt",
        **config,
    )
    pipeline.set_background_fingerprints_from_video(cap, BG_FP_TIME)

    times = sorted(set(t for (t, *_) in label_truth.keys()))

    correct = 0
    total = 0
    cm: dict[tuple[int, int], int] = {}

    for t in times:
        # 各 frame 独立 reset
        pipeline.reset(match_start_sec=t)
        pipeline.set_background_fingerprints_from_video(cap, BG_FP_TIME)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        if fr.shape[:2] != (1080, 1920):
            fr = cv2.resize(fr, (1920, 1080))
        state = pipeline.extract(fr, t)

        for side, board in (("1P", state.board_p1), ("2P", state.board_p2)):
            for vrow in range(12):
                row = vrow + HIDDEN_ROWS
                for col in range(6):
                    key = (t, side, vrow, col)
                    if key not in label_truth:
                        continue
                    truth = label_truth[key]
                    pred = int(board.get(row, col))
                    total += 1
                    if pred == truth:
                        correct += 1
                    else:
                        cm[(truth, pred)] = cm.get((truth, pred), 0) + 1
    cap.release()
    return correct, total, cm


def main() -> int:
    truth = load_labels()
    print(f"truth labels: {len(truth)}")

    # baseline: 全 refiner ON (default)
    base_config = dict(
        use_per_video_calibrator=True,
        use_temporal_voting=True,
        use_score_eraser=True,
        use_pair_landing_check=True,
        use_chain_animation_detector=False,
        use_puyo_stability=False,
    )
    print("\n=== baseline (all default refiners ON) ===")
    c, t, cm = evaluate_config(truth, base_config)
    base_acc = c / max(1, t)
    print(f"  {c}/{t} = {base_acc:.4f}")

    refiner_keys = [
        "use_per_video_calibrator",
        "use_temporal_voting",
        "use_score_eraser",
        "use_pair_landing_check",
        "use_enhanced_tracker",
    ]

    for key in refiner_keys:
        cfg = dict(base_config)
        cfg[key] = False
        print(f"\n=== {key} = OFF ===")
        c, t, cm = evaluate_config(truth, cfg)
        acc = c / max(1, t)
        delta = acc - base_acc
        sign = "+" if delta >= 0 else ""
        err = sorted(cm.items(), key=lambda kv: -kv[1])[:5]
        err_str = " ".join(
            f"{CODE_TO_LABEL[k[0]]}->{CODE_TO_LABEL[k[1]]}:{v}"
            for k, v in err
        )
        print(
            f"  {c}/{t} = {acc:.4f} ({sign}{delta*100:.2f}pt vs base)"
        )
        print(f"  errors: {err_str}")

    # all OFF (raw + bg fp + classify only)
    cfg = dict(base_config)
    for k in refiner_keys:
        cfg[k] = False
    print("\n=== ALL refiners OFF ===")
    c, t, cm = evaluate_config(truth, cfg)
    acc = c / max(1, t)
    delta = acc - base_acc
    sign = "+" if delta >= 0 else ""
    err = sorted(cm.items(), key=lambda kv: -kv[1])[:5]
    err_str = " ".join(
        f"{CODE_TO_LABEL[k[0]]}->{CODE_TO_LABEL[k[1]]}:{v}"
        for k, v in err
    )
    print(f"  {c}/{t} = {acc:.4f} ({sign}{delta*100:.2f}pt vs base)")
    print(f"  errors: {err_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
