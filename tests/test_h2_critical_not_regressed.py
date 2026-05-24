"""cycle 55 H2 ゲート (= baseline_v3 critical 悪化なし + GO 合格点判定).

baseline_videos_v3 8 動画で評価ツール critical 件数を比較:
- baseline anchor = 1512 (= 現 default model cnn_phase_b_large_v2.pt の値)
- GO 合格点 = 1000 (= -33% 改善、 ユーザー合意済)

cycle 55 eval summary が不在なら skip。
"""
import json
from pathlib import Path

import pytest

BASELINE_SUMMARY = Path("data/verify/baseline_v3_eval/_summary.json")
CYCLE55_SUMMARY = Path("data/verify/cycle55_eval/_summary.json")

BASELINE_ANCHOR_CRITICAL = 1512
H2_GO_THRESHOLD = 1000


def test_baseline_anchor_critical_is_1512() -> None:
    """baseline anchor が改変されていないこと (= 1512 固定)."""
    if not BASELINE_SUMMARY.exists():
        pytest.skip("baseline summary 不在")
    data = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    assert data["totals"]["critical"] == BASELINE_ANCHOR_CRITICAL, (
        f"baseline critical changed from {BASELINE_ANCHOR_CRITICAL} "
        f"to {data['totals']['critical']} (= history 改変?)"
    )


def test_cycle55_critical_not_worse_than_baseline() -> None:
    """cycle 55 critical が baseline (= 1512) を悪化させていないこと."""
    if not CYCLE55_SUMMARY.exists():
        pytest.skip("cycle 55 eval 未実施")
    c55 = json.loads(CYCLE55_SUMMARY.read_text(encoding="utf-8"))
    c55_critical = c55["totals"]["critical"]
    assert c55_critical <= BASELINE_ANCHOR_CRITICAL, (
        f"cycle55 critical {c55_critical} > baseline "
        f"{BASELINE_ANCHOR_CRITICAL} = 学習悪化"
    )


def test_cycle55_meets_go_threshold() -> None:
    """cycle 55 critical が GO 合格点 (= 1000) 以下."""
    if not CYCLE55_SUMMARY.exists():
        pytest.skip("cycle 55 eval 未実施")
    c55 = json.loads(CYCLE55_SUMMARY.read_text(encoding="utf-8"))
    c55_critical = c55["totals"]["critical"]
    assert c55_critical <= H2_GO_THRESHOLD, (
        f"cycle55 critical {c55_critical} > GO threshold {H2_GO_THRESHOLD}"
    )


def test_cycle55_per_video_v29m2_improved() -> None:
    """v29m2 (= baseline 最大 critical 535) が改善していること.

    アナリスト指摘: v29m2 が baseline 全体の 35% を占める。
    この動画を改善しないまま total が 1000 以下になるシナリオは「v29 問題温存」 = NO-GO。
    """
    if not CYCLE55_SUMMARY.exists():
        pytest.skip("cycle 55 eval 未実施")
    c55 = json.loads(CYCLE55_SUMMARY.read_text(encoding="utf-8"))
    v29m2_critical = c55["per_video"].get("v29m2", {}).get("critical", -1)
    if v29m2_critical < 0:
        pytest.skip("v29m2 が cycle 55 eval 対象外")
    assert v29m2_critical < 535, (
        f"v29m2 critical {v29m2_critical} not improved from baseline 535"
    )
