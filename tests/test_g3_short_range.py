"""区間短縮接続の検査。本番区間を壊さないことと、適用範囲の明示を確認する。"""
from contextlib import ExitStack
from typing import Any
import os

import pytest

from scripts import g3_short_range as P


def test_default_end_and_restore() -> None:
    """既定2000へ縮め、scope終了で本番の36300へ戻る。"""
    assert P.G.END == P.ORIGINAL_END
    with ExitStack() as stack:
        receipt = P.install(stack, 2000)
        assert P.G.END == 2000
        assert receipt['frames'] == 1000 and receipt['original_end'] == 36300
    assert P.G.END == P.ORIGINAL_END


def test_rejects_longer_or_equal(monkeypatch: Any) -> None:
    """本番以上の値は拒否する。短縮台が本番を名乗れないようにする。"""
    monkeypatch.setenv(P.VARIABLE, str(P.ORIGINAL_END))
    with pytest.raises(ValueError, match='short_range_must_be_shorter'):
        P.target_end()
    monkeypatch.setenv(P.VARIABLE, '40000')
    with pytest.raises(ValueError, match='short_range_must_be_shorter'):
        P.target_end()


def test_rejects_odd_or_nonpositive(monkeypatch: Any) -> None:
    """stride2と噛み合わない値は拒否する。"""
    monkeypatch.setenv(P.VARIABLE, '1999')
    with pytest.raises(ValueError, match='short_range_even_positive'):
        P.target_end()
    monkeypatch.setenv(P.VARIABLE, '0')
    with pytest.raises(ValueError, match='short_range_even_positive'):
        P.target_end()


def test_env_value_used(monkeypatch: Any) -> None:
    monkeypatch.setenv(P.VARIABLE, '3000')
    assert P.target_end() == 3000


def test_receipt_records_uncovered_defects() -> None:
    """短区間で再現できない内容依存の欠陥を受領票へ明示する。"""
    with ExitStack() as stack:
        receipt = P.install(stack, 2000)
    assert receipt['short_range_not_quality_pass'] is True
    assert any('effect' in x for x in receipt['content_dependent_defects_not_covered'])
    assert any('M1' in x for x in receipt['content_dependent_defects_not_covered'])


def test_guards_original_state(monkeypatch: Any) -> None:
    """他層が先に区間を変えていたら名前つきで落とす。"""
    monkeypatch.setattr(P.G, 'END', 12345)
    with ExitStack() as stack:
        with pytest.raises(ValueError, match='short_range_original_end'):
            P.install(stack, 2000)
