"""cycle 50 final seed dataset の 100% PURE 回帰保護.

ユーザー目視 10 動画 22 PNG レビュー結果 (= 2026-05-21 朝) で
seed 系統的汚染が判明、 改修 2/3/4 で 100% PURE 達成。

seed が再生成された場合、 汚染が混入していないかを自動検知する。
data/verify/seed_quality_cycle50_final.json が不在なら skip。
"""
import json
from pathlib import Path

import pytest

SEED_QUALITY_JSON = Path("data/verify/seed_quality_cycle50_final.json")


def _load_summary() -> dict:
    if not SEED_QUALITY_JSON.exists():
        pytest.skip(f"seed quality JSON 不在: {SEED_QUALITY_JSON}")
    data = json.loads(SEED_QUALITY_JSON.read_text(encoding="utf-8"))
    return data["summary"]


def test_overall_purity_is_1() -> None:
    summary = _load_summary()
    assert summary["overall_purity"] == 1.0, \
        f"seed cross_color_purity が 1.0 から低下 (= {summary['overall_purity']})"


def test_per_color_purity_all_1() -> None:
    summary = _load_summary()
    for color, purity in summary["per_color_purity"].items():
        assert purity == 1.0, f"{color} purity {purity} < 1.0 = seed 汚染再混入"


def test_total_samples_above_140k() -> None:
    """sample 数が cycle 50 final 時点の規模 (= 149,523) から大幅減していないこと."""
    summary = _load_summary()
    n = summary["total_samples"]
    assert n >= 140_000, \
        f"sample count {n} < 140000 = seed 再生成で動画落ち の可能性"
