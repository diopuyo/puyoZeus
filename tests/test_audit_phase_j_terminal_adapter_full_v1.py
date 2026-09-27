"""Phase J terminal adapter固定57試合集約の判定試験。"""

from scripts.audit_phase_j_terminal_adapter_full_v1 import (
    EXPECTED_GAME_COUNT,
    build_report,
)


def _row(index: int, *, detected: bool, mismatch: bool = False) -> dict[str, object]:
    expected = "1P" if index % 2 else "2P"
    detected_winner = ("2P" if expected == "1P" else "1P") if mismatch else expected
    return {
        "detected": detected,
        "expected_winner": expected,
        "detected_winner": detected_winner if detected else None,
        "evidence_kind": "visual_result_logo_bilateral_2x2" if detected else None,
        "event_digest": "sha256:" + "a" * 64 if detected else None,
    }


def _negative_rows(*, false_positive_index: int | None = None) -> list[dict[str, object]]:
    return [
        _row(index, detected=index == false_positive_index)
        for index in range(EXPECTED_GAME_COUNT)
    ]


def test_full_report_passes_safe_fixed_baseline() -> None:
    rows = [_row(index, detected=index < 13) for index in range(EXPECTED_GAME_COUNT)]

    report = build_report(rows, _negative_rows())

    assert report["decision"] == "PASS"
    assert report["detected_count"] == 13
    assert report["direction_mismatch_count"] == 0


def test_full_report_fails_one_wrong_direction() -> None:
    rows = [
        _row(index, detected=index < 13, mismatch=index == 0)
        for index in range(EXPECTED_GAME_COUNT)
    ]

    report = build_report(rows, _negative_rows())

    assert report["decision"] == "FAIL"
    assert report["direction_mismatch_count"] == 1


def test_full_report_fails_one_sided_coverage() -> None:
    rows = [
        _row(index, detected=index % 2 == 1)
        for index in range(EXPECTED_GAME_COUNT)
    ]

    report = build_report(rows, _negative_rows())

    assert report["decision"] == "FAIL"
    assert report["direction_coverage"]["1P"] == 1.0
    assert report["direction_coverage"]["2P"] == 0.0
    assert report["direction_coverage_gap"] == 1.0


def test_full_report_fails_negative_control_detection() -> None:
    rows = [_row(index, detected=index < 13) for index in range(EXPECTED_GAME_COUNT)]

    report = build_report(rows, _negative_rows(false_positive_index=20))

    assert report["decision"] == "FAIL"
    assert report["negative_control_false_positive_count"] == 1
