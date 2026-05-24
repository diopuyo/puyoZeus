"""評価器 hardcode 定数の frozen 回帰保護.

2026-05-20 D 系 sweep で D1_drop_threshold_3 / D4_chain_loss_2 が
「閾値を緩めて critical が -9 件減った」 → ACCEPT 判定、 ただし実体改善ゼロ
という fail-silent 罠を実証。

この罠の再演を防ぐため、 RecognitionEvaluator の判定定数が cycle 間で
変更されていないことを assert で凍結する。
"""
import re
from pathlib import Path

from src import recognition_evaluator as evaluator


def test_puyo_count_drop_threshold_normal_is_2() -> None:
    assert evaluator.PUYO_COUNT_DROP_THRESHOLD_NORMAL == 2


def test_sudden_drop_threshold_is_5() -> None:
    assert evaluator.SUDDEN_DROP_THRESHOLD == 5


def test_chain_min_puyo_loss_is_4() -> None:
    assert evaluator.CHAIN_MIN_PUYO_LOSS == 4


def test_chain_state_min_frames_is_5() -> None:
    assert evaluator.CHAIN_STATE_MIN_FRAMES == 5


def test_reject_critical_threshold_is_20() -> None:
    """RecognitionEvaluator が verdict = REJECT を判定する critical 閾値 = 20.

    src/recognition_evaluator.py:726 で hardcode。
    緩めて 30 等にすると D 系 fail-silent と同型になる。
    """
    src = Path(__file__).resolve().parent.parent / "src" / "recognition_evaluator.py"
    text = src.read_text(encoding="utf-8")
    m = re.search(r"critical_count\s*>=\s*(\d+)", text)
    assert m is not None, "critical_count >= N pattern not found"
    assert m.group(1) == "20", \
        f"REJECT critical threshold changed from 20 to {m.group(1)}"


def test_review_thresholds_are_5_and_30() -> None:
    """REVIEW 判定の閾値 (critical >= 5 OR warning >= 30) 凍結."""
    src = Path(__file__).resolve().parent.parent / "src" / "recognition_evaluator.py"
    text = src.read_text(encoding="utf-8")
    m = re.search(
        r"critical_count\s*>=\s*(\d+)\s*or\s*warning_count\s*>=\s*(\d+)",
        text,
    )
    assert m is not None, "REVIEW threshold pattern not found"
    assert m.group(1) == "5", f"REVIEW critical threshold changed to {m.group(1)}"
    assert m.group(2) == "30", f"REVIEW warning threshold changed to {m.group(2)}"
