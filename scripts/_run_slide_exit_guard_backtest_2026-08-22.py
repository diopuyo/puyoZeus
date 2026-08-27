"""修正③根治 (enable_slide_exit_min_display_guard) の全域バックテスト起動用
ラッパー (2026-08-22)。

既存の測定器 scripts/measure_stable_cell_acc.py を一切改変せず、
RecognitionPipeline.load_default() に enable_slide_exit_min_display_guard=True
を注入するだけの monkeypatch ラッパー (feedback_check_existing_before_building
の趣旨: 新規測定器を作らず既存を使う)。

--workers 1 前提 (multiprocessing worker はサブプロセス内で改めて
load_default() するため、本 monkeypatch は親プロセスにしか効かない)。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.measure_stable_cell_acc as msca  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

_orig_load_default = RecognitionPipeline.load_default.__func__


def _patched_load_default(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
    kwargs.setdefault("enable_slide_exit_min_display_guard", True)
    return _orig_load_default(cls, *args, **kwargs)


def main() -> int:
    RecognitionPipeline.load_default = classmethod(_patched_load_default)
    try:
        return msca.main()
    finally:
        RecognitionPipeline.load_default = classmethod(_orig_load_default)


if __name__ == "__main__":
    sys.exit(main())
