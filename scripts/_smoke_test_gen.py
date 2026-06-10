"""generate_training_dataset の編集が成立しているか確認する smoke test。"""
from __future__ import annotations


def main() -> int:
    from scripts.old.generate_training_dataset import (
        extract_one_sample,
        extract_video_rows,
        _detect_next_pairs,
    )
    from src.next_detector import NextDetector
    print("import OK")
    print("extract_one_sample:", extract_one_sample.__name__)
    print("extract_video_rows:", extract_video_rows.__name__)
    print("_detect_next_pairs:", _detect_next_pairs.__name__)
    print("NextDetector:", NextDetector.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
