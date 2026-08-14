"""scripts/_build_chain_anim_duration_table_2026-08-14.py の単体テスト +
恒久JSON (data/verify/chain_anim_duration_median_table_2026-08-14.json) と
src/indicators_v2.py のハードコード較正定数の整合性テスト。

docs/DEMO_REVIEW_2026-08-13.md #12 案B: 実測較正は
src/indicators_v2.py 側 (非ファイルI/O、stateless実装原則) が単一情報源
であり、data/verify/ 配下のJSONはそれと同じ値を保持する監査用の副産物
(スクリプトのdocstring参照)。両者がずれていないことをここで固定する。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.indicators_v2 import (
    CHAIN_ANIM_DURATION_EXTRAPOLATION_A_SEC_2026_08_14,
    CHAIN_ANIM_DURATION_EXTRAPOLATION_B_SEC_PER_CHAIN_2026_08_14,
    CHAIN_ANIM_DURATION_EXTRAPOLATION_MIN_CHAIN_COUNT_2026_08_14,
    CHAIN_ANIM_DURATION_MEDIAN_SEC_TABLE_2026_08_14,
)

JSON_PATH = Path("data/verify/chain_anim_duration_median_table_2026-08-14.json")


@pytest.fixture(scope="module")
def mod():
    """ハイフン入りファイル名のためモジュールとして直接ロードする
    (tests/test_build_chain_length_conditional_2026-08-13.py と同じ方式)。"""
    path = Path(__file__).resolve().parent.parent / "scripts" / (
        "_build_chain_anim_duration_table_2026-08-14.py"
    )
    spec = importlib.util.spec_from_file_location("_chain_anim_dur_table_for_test", path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["_chain_anim_dur_table_for_test"] = m
    spec.loader.exec_module(m)
    return m


# ============================
# build_table (純粋関数)
# ============================


def test_build_table_keeps_raw_median_below_extrapolation_threshold(mod) -> None:
    rows = [{"chain_count": "6", "n": "64", "median_sec": "9.5", "p25_sec": "9.0", "p75_sec": "10.0"}]
    table = mod.build_table(rows)["table_by_chain_count"]
    assert table["6"]["extrapolated"] is False
    assert table["6"]["median_sec"] == pytest.approx(9.5)
    assert table["6"]["raw_median_sec_low_n"] is None


def test_build_table_extrapolates_at_and_above_threshold(mod) -> None:
    rows = [
        {"chain_count": "13", "n": "6", "median_sec": "19.39", "p25_sec": "1.0", "p75_sec": "2.0"},
        {"chain_count": "14", "n": "3", "median_sec": "7.02", "p25_sec": "1.0", "p75_sec": "2.0"},
    ]
    table = mod.build_table(rows)["table_by_chain_count"]
    assert table["13"]["extrapolated"] is True
    assert table["13"]["median_sec"] == pytest.approx(mod.EXTRAPOLATION_A + mod.EXTRAPOLATION_B * 13)
    assert table["13"]["raw_median_sec_low_n"] == pytest.approx(19.39)
    assert table["14"]["median_sec"] == pytest.approx(mod.EXTRAPOLATION_A + mod.EXTRAPOLATION_B * 14)


def test_build_table_extrapolation_params_match_source_of_truth(mod) -> None:
    """スクリプト内の複製定数 (indicators_v2 の重い依存を避けるための意図的
    複製) が src/indicators_v2.py のハードコード定数と一致すること。"""
    assert mod.EXTRAPOLATION_A == pytest.approx(CHAIN_ANIM_DURATION_EXTRAPOLATION_A_SEC_2026_08_14)
    assert mod.EXTRAPOLATION_B == pytest.approx(
        CHAIN_ANIM_DURATION_EXTRAPOLATION_B_SEC_PER_CHAIN_2026_08_14)
    assert mod.EXTRAPOLATION_MIN_N == CHAIN_ANIM_DURATION_EXTRAPOLATION_MIN_CHAIN_COUNT_2026_08_14


# ============================
# 恒久JSON ⇔ ハードコード較正定数の整合性
# (JSONが再生成待ち等で一時的に無い環境では fail-safe にスキップする)
# ============================


def test_json_artifact_matches_hardcoded_calibration_table_when_present() -> None:
    if not JSON_PATH.exists():
        return
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    table = data["table_by_chain_count"]
    for n, expected_median in CHAIN_ANIM_DURATION_MEDIAN_SEC_TABLE_2026_08_14.items():
        entry = table[str(n)]
        assert entry["extrapolated"] is False  # N=1..12はテーブル生値のみ
        assert entry["median_sec"] == pytest.approx(expected_median)
    assert data["extrapolation"]["a"] == pytest.approx(
        CHAIN_ANIM_DURATION_EXTRAPOLATION_A_SEC_2026_08_14)
    assert data["extrapolation"]["b"] == pytest.approx(
        CHAIN_ANIM_DURATION_EXTRAPOLATION_B_SEC_PER_CHAIN_2026_08_14)
    assert data["extrapolation"]["applies_to_chain_count_gte"] == (
        CHAIN_ANIM_DURATION_EXTRAPOLATION_MIN_CHAIN_COUNT_2026_08_14)
