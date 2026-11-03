"""B3 列仕様台帳・非飽和正規化・側交換の単体テスト (台帳の整合と可換性)。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.b3_column_spec import (COLUMN_SPECS, IND_NAMES, K_LEVELS, NONSAT_A_DEFAULT, RAW_COUNT_INDICATORS,
                                SPEC_BY_NAME, Kind, Mask, nonsat, render_markdown, signed_nonsat, swap_sides,
                                validate_spec)

DOC = Path(__file__).resolve().parents[1] / "docs" / "B3_COLUMN_SPEC.md"


def _random_row(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    row = {}
    for c in COLUMN_SPECS:
        row[c.name] = "x|g1|f2|o0" if c.dtype == "str" else float(rng.uniform(0.0, 1.0)) if c.dtype.startswith("f") else int(rng.integers(0, 4))
    row["orientation"] = 0
    row["row_key"] = "v|g1|f2|o0"
    return row


def test_spec_validates_and_unique():
    validate_spec()
    names = [c.name for c in COLUMN_SPECS]
    assert len(names) == len(set(names))
    assert len(names) > 300


def test_every_mask_reference_exists_and_masks_are_int():
    for c in COLUMN_SPECS:
        if c.mask:
            assert c.mask in SPEC_BY_NAME, c.name
            assert SPEC_BY_NAME[c.mask].dtype == "i1"


def test_doc_in_sync_with_spec():
    text = DOC.read_text(encoding="utf-8").splitlines()
    assert chr(10).join(text) + chr(10) == render_markdown()


def test_swap_is_involution():
    row = _random_row()
    again = swap_sides(swap_sides(row))
    for k, v in row.items():
        assert again[k] == pytest.approx(v) if not isinstance(v, str) else again[k] == v


def test_swap_rules_per_kind():
    row = _random_row(1)
    sw = swap_sides(row)
    for c in COLUMN_SPECS:
        if c.kind is Kind.SIDE_ABS:
            assert sw[c.name] == row[c.counterpart]
        elif c.kind is Kind.DIFF_RAW:
            assert sw[c.name] == -row[c.name]
        elif c.kind in (Kind.DIFF_CENTERED, Kind.LABEL):
            assert sw[c.name] == pytest.approx(1.0 - row[c.name])
        else:
            assert sw[c.name] == row[c.name]


def test_diff_columns_commute_with_swap():
    """絶対量を交換してから差分を作る = 差分を作ってから台帳規則で交換、が一致する (一括反転しない)。"""
    from src.b3_features import diff_group
    row = _random_row(2)
    a = dict(row)
    diff_group(a)
    swapped_first = swap_sides(row)
    diff_group(swapped_first)
    diff_first_then_swap = swap_sides(a)
    for n in [c.name for c in COLUMN_SPECS if c.group == "diff"]:
        assert swapped_first[n] == pytest.approx(diff_first_then_swap[n]), n


def test_nonsat_properties():
    x = np.array([0, 1, 10, 100, 1000, 10000, 1e6])
    y = nonsat(x)
    assert y[0] == 0 and np.all(np.diff(y) > 0) and np.all(y < 1)
    assert len(np.unique(y.astype(np.float32))) == len(x)          # float32 でも識別できる
    assert np.isnan(nonsat(np.nan))                                  # NaN を 0 にしない
    assert signed_nonsat(0.0) == 0.5
    assert signed_nonsat(5.0) + signed_nonsat(-5.0) == pytest.approx(1.0)
    assert np.isnan(signed_nonsat(np.nan))
    assert nonsat(10.0, a=NONSAT_A_DEFAULT) == pytest.approx(np.log1p(10) / (NONSAT_A_DEFAULT + np.log1p(10)))


def test_mask_codes_cover_design_reasons():
    assert {m.name for m in Mask} >= {"LEGACY_SCHEMA_ABSENT", "UNOBSERVABLE", "NOT_YET_AVAILABLE", "BOUNDARY_UNKNOWN",
                                      "ONE_SIDE_UNOBSERVED", "UNPREDICTABLE", "SEARCH_TRUNCATED", "NOT_APPLICABLE"}


def test_indicator_columns_match_registry_order():
    assert len(IND_NAMES) == 30
    assert set(RAW_COUNT_INDICATORS) <= set(IND_NAMES)
    assert K_LEVELS == (1, 2, 3, 4, 5)


def test_ledger_columns_are_r_only():
    assert all(c.role == "R_only" for c in COLUMN_SPECS if c.group == "r_ledger")
    assert not any("ledger" in c.name for c in COLUMN_SPECS if c.role == "feature")
