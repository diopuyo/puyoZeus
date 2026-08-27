"""2つのnpzについて、全共通キーの完全一致を母数付きで報告する。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    if left.dtype.kind in "fc":
        left = np.nan_to_num(left, nan=-9e99)
        right = np.nan_to_num(right, nan=-9e99)
    return bool(np.array_equal(left, right))


def main() -> int:
    left_path, right_path = Path(sys.argv[1]), Path(sys.argv[2])
    with np.load(left_path, allow_pickle=True) as left_data:
        left = {key: left_data[key] for key in left_data.files}
    with np.load(right_path, allow_pickle=True) as right_data:
        right = {key: right_data[key] for key in right_data.files}
    common = sorted(set(left) & set(right))
    mismatched = [key for key in common if not _equal(left[key], right[key])]
    print(
        f"left={left_path}\nright={right_path}\n"
        f"common_keys={len(common)} mismatched={len(mismatched)}/{len(common)}\n"
        f"mismatched_keys={mismatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
