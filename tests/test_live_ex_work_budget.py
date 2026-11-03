"""配信用の封入版にも同じ旗と上限が届き、鮮度判定から独立している。"""
from __future__ import annotations

import pytest

from src.phase_j._live_ex_frozen.src import dl1k_theory as T
from src.phase_j._live_ex_frozen.src.dl1k_work_budget import FLAG, LIMIT_ENV
from src.phase_j._live_ex_frozen.src.prefire_tsumo_v411 import Knowledge
from tests.live_ex_fixtures import context


def test_shared_flag_reaches_frozen_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    native = pytest.importorskip('puyo_core')
    monkeypatch.setenv(FLAG, '1')
    monkeypatch.setenv(LIMIT_ENV, '1')
    ctx = context()
    ctx.ends = (2., 0.)
    result = T.solve(native, ctx, Knowledge().snapshot(ctx), dict(allowed=False))
    assert result['budget_kind'] == 'work'
    assert result['work_units'] == 1 and result['cutoff']


def test_flag_off_keeps_time_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    native = pytest.importorskip('puyo_core')
    monkeypatch.delenv(FLAG, raising=False)
    ctx = context()
    result = T.solve(native, ctx, Knowledge().snapshot(ctx), dict(allowed=False))
    assert result['budget_kind'] == 'time' and result['work_limit'] is None
