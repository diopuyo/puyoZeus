"""HSV / CNN / Hybrid の分類器精度を比較評価する。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.console_init import init_console  # noqa: E402
init_console()

import numpy as np
import torch

from src.hybrid_classifier import HybridClassifier
from src.image_reader import ColorClassifier
from src.patch_classifier import CnnPatchClassifier


def main() -> int:
    data = np.load("data/training_phase_u/manual_labels.npz")
    patches = data["patches"]
    labels = data["labels"]
    n = len(patches)
    print(f"samples: {n}")

    hsv = ColorClassifier()
    cnn = CnnPatchClassifier()
    state = torch.load(
        "models/cnn_phase_u_v6.pt", map_location="cpu", weights_only=True,
    )
    cnn._model.load_state_dict(state)
    cnn._model.eval()
    hybrid = HybridClassifier(hsv_classifier=hsv, cnn_classifier=cnn)

    n_hsv = 0
    n_cnn = 0
    n_hyb = 0
    confusion = {"hsv": {}, "cnn": {}, "hyb": {}}
    for patch, lbl in zip(patches, labels):
        truth = int(lbl)
        h = hsv.classify(patch)
        c = cnn.classify(patch)
        hy = hybrid.classify(patch)
        if h == truth:
            n_hsv += 1
        else:
            confusion["hsv"][(h, truth)] = (
                confusion["hsv"].get((h, truth), 0) + 1
            )
        if c == truth:
            n_cnn += 1
        else:
            confusion["cnn"][(c, truth)] = (
                confusion["cnn"].get((c, truth), 0) + 1
            )
        if hy == truth:
            n_hyb += 1
        else:
            confusion["hyb"][(hy, truth)] = (
                confusion["hyb"].get((hy, truth), 0) + 1
            )

    print(f"HSV   : {n_hsv:4d}/{n} = {n_hsv/n*100:.2f}%")
    print(f"CNN   : {n_cnn:4d}/{n} = {n_cnn/n*100:.2f}%")
    print(f"Hybrid: {n_hyb:4d}/{n} = {n_hyb/n*100:.2f}%")
    print()
    for name in ("hsv", "cnn", "hyb"):
        print(f"=== {name} top mistakes ===")
        for (pred, truth), cnt in sorted(
            confusion[name].items(), key=lambda x: -x[1],
        )[:8]:
            print(f"  {pred} -> {truth}: {cnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
