"""BoardRecognitionPipeline で連続フレーム認識精度を評価。

各バッチの labels.csv の (time, side, row, col, your_answer) で評価。
動画から t-0.4, t-0.2, t の 3 フレームを順次 read することで、Pipeline の
時系列レイヤーが効くようにする。

比較対象:
    - 単純 ImageReader (本流の Hybrid 分類器)
    - BoardRecognitionPipeline (時系列レイヤー有効)
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.console_init import init_console  # noqa: E402
init_console()

import cv2
import torch
import numpy as np

from src.board import HIDDEN_ROWS
from src.board_recognition_pipeline import BoardRecognitionPipeline
from src.hybrid_classifier import HybridClassifier
from src.image_reader import (
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    ImageReader,
)
from src.patch_classifier import CnnPatchClassifier

LABEL_TO_CODE = {
    "EM": 0, "RED": 1, "BLUE": 2, "GRN": 3,
    "YEL": 4, "PUR": 5, "OJM": 9,
}


def load_classifier(cnn_path: str) -> HybridClassifier:
    cnn = CnnPatchClassifier()
    state = torch.load(cnn_path, map_location="cpu", weights_only=True)
    cnn._model.load_state_dict(state)
    cnn._model.eval()
    return HybridClassifier(cnn_classifier=cnn)


def evaluate_pipeline(
    base_dirs: list[str], video_path: str, cnn_path: str,
    use_pipeline: bool,
    use_anim_filter: bool = True,
    use_smoother: bool = True,
    use_tracker: bool = True,
    use_adaptive_bg: bool = True,
) -> tuple[int, int]:
    """各シートの labels.csv を読み、認識結果と truth を比較。"""
    classifier = load_classifier(cnn_path)
    reader = ImageReader(
        classifier=classifier, use_match_state=True, use_ui_mask=True,
    )
    pipeline = (
        BoardRecognitionPipeline(
            reader,
            use_anim_filter=use_anim_filter,
            use_smoother=use_smoother,
            use_tracker=use_tracker,
            use_adaptive_bg=use_adaptive_bg,
        ) if use_pipeline else None
    )
    cap = cv2.VideoCapture(video_path)

    n_total = 0
    n_correct = 0

    # シート毎 / 時刻毎にグループ化
    for base in base_dirs:
        base_p = Path(base)
        if not base_p.exists():
            continue
        for sheet_dir in sorted(base_p.iterdir()):
            csv_path = sheet_dir / "labels.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            # 時刻ごとに行をまとめる
            by_time: dict[float, list] = defaultdict(list)
            for row in rows:
                by_time[float(row["time"])].append(row)
            sheet_times = sorted(by_time.keys())

            for t in sheet_times:
                if pipeline is not None:
                    # 各時刻で reset → 直前 1.5 秒を連続再生して時系列レイヤー暖機
                    pipeline.reset()
                    start_t = max(0.0, t - 1.5)
                    cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000)
                    end_msec = t * 1000
                    b1 = None
                    b2 = None
                    while True:
                        ok, fr = cap.read()
                        if not ok or fr is None:
                            break
                        cur_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                        b1, b2 = pipeline.read(fr)
                        if cur_msec >= end_msec:
                            break
                    if b1 is None:
                        continue
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                    ok, fr = cap.read()
                    if not ok:
                        continue
                    if fr.shape[:2] != (1080, 1920):
                        fr = cv2.resize(
                            fr, (1920, 1080), interpolation=cv2.INTER_AREA,
                        )
                    b1, b2 = reader.read_both_boards(fr)

                # 各 row の真値と比較
                for row in by_time[t]:
                    truth_str = row.get("your_answer", "").strip()
                    if not truth_str:
                        truth_str = row["recognized"]
                    if truth_str not in LABEL_TO_CODE:
                        continue
                    truth = LABEL_TO_CODE[truth_str]
                    side = row["side"]
                    vrow = int(row["row"])
                    vcol = int(row["col"])
                    board = b1 if side == "1P" else b2
                    pred = int(board.get(vrow + HIDDEN_ROWS, vcol))
                    n_total += 1
                    if pred == truth:
                        n_correct += 1
    cap.release()
    return n_correct, n_total


def main() -> int:
    base_dirs = [
        "data/verify/phase_u_batch1",
        "data/verify/phase_u_batch2",
        "data/verify/phase_u_batch3",
        "data/verify/phase_u_batch4",
    ]
    video_path = "data/frames/video_01.mp4"
    cnn_path = "models/cnn_phase_u_v3.pt"

    print("=== Baseline (single-frame, no pipeline) ===")
    c0, t0 = evaluate_pipeline(base_dirs, video_path, cnn_path, use_pipeline=False)
    print(f"  {c0}/{t0} = {c0/max(1,t0)*100:.2f}%")

    layer_configs = [
        ("only AnimFilter", True, False, False, False),
        ("only Smoother", False, True, False, False),
        ("only Tracker", False, False, True, False),
        ("only AdaptiveBG", False, False, False, True),
    ]
    for name, anim, smoo, track, adap in layer_configs:
        c, t = evaluate_pipeline(
            base_dirs, video_path, cnn_path, use_pipeline=True,
            use_anim_filter=anim, use_smoother=smoo,
            use_tracker=track, use_adaptive_bg=adap,
        )
        diff_pp = (c - c0) / max(1, t0) * 100
        print(f"=== {name}: {c}/{t} = {c/max(1,t)*100:.2f}% (diff {diff_pp:+.2f}ppt) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
