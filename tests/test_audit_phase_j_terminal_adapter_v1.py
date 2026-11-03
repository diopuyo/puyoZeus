"""Phase J terminal adapter実フレーム監査の集約判定試験。"""

from scripts.audit_phase_j_terminal_adapter_v1 import build_report


def _row(*, detected: bool = True, winner: str = "1P") -> dict[str, object]:
    return {
        "detected": detected, "detected_winner": winner,
        "expected_winner": "1P", "evidence_kind": (
            "visual_result_logo_bilateral_2x2" if detected else None
        ),
        "event_digest": "sha256:" + "a" * 64 if detected else None,
    }


def test_build_report_passes_only_complete_direction_matched_replay() -> None:
    report = build_report([_row(), _row()])

    assert report["decision"] == "PASS"
    assert report["detected_count"] == 2
    assert report["direction_mismatch_count"] == 0


def test_build_report_fails_missing_or_wrong_direction() -> None:
    report = build_report([_row(detected=False), _row(winner="2P")])

    assert report["decision"] == "FAIL"
    assert report["checks"]["all_reference_detections_reproduced"] is False
    assert report["checks"]["all_winner_directions_match"] is False
