"""Phase I: 収集した hidden_row 擬似ラベルから calibration を学習する.

usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_hidden_row_finetune \
        --videos video_02 video_03 video_05

LabelStore から各 video の hidden_row.jsonl を読み込み、HiddenRowFineTuner で
Platt scaling パラメータ a, b を学習する。結果は
`data/verify/hidden_row_calibration.json` に保存される。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.hidden_row_inferrer import DEFAULT_CALIBRATION_PATH  # noqa: E402
from src.self_supervised.hidden_row_fine_tuner import (  # noqa: E402
    HiddenRowFineTuner,
)
from src.self_supervised.label_store import LabelStore  # noqa: E402
from src.self_supervised.pseudo_label import PseudoLabelSample  # noqa: E402

# ============================
# 定数
# ============================

COMPONENT_HIDDEN_ROW: str = "hidden_row"


def _load_samples(
    video_ids: list[str],
    label_store_root: Path | None = None,
) -> list[PseudoLabelSample]:
    """指定 video_id 群から hidden_row 擬似ラベルを集約する."""
    samples: list[PseudoLabelSample] = []
    if not video_ids:
        # 全 video を対象
        video_ids = LabelStore.list_videos(root=label_store_root)
    for vid in video_ids:
        store = LabelStore(video_id=vid, root=label_store_root)
        for s in store.load(COMPONENT_HIDDEN_ROW):
            samples.append(s)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase I: hidden_row Platt scaling fine-tune",
    )
    parser.add_argument(
        "--videos", type=str, nargs="*", default=[],
        help="対象 video_id (省略で全 video)",
    )
    parser.add_argument(
        "--label-store-root", type=Path, default=None,
        help="LabelStore のルート (省略で data/pseudo_labels)",
    )
    parser.add_argument(
        "--calibration-path", type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="出力 calibration JSON",
    )
    args = parser.parse_args(argv)
    samples = _load_samples(args.videos, args.label_store_root)
    print(f"[phase_i_finetune] loaded n_samples={len(samples)}")
    ft = HiddenRowFineTuner(calibration_path=args.calibration_path)
    metrics = ft.fine_tune(samples)
    print(f"[phase_i_finetune] metrics={json.dumps(metrics, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
