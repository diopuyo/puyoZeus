"""外枠猶予の期限・入力世代・得点・位置条件を検証する。"""
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.phase_j.live_border_grace import BorderGraceVerifier, verify_screen, wrap_verifier

IMAGE = np.full((180, 240, 3), 80, dtype=np.uint8)
SIZE = (240, 180)


class Verifier:
    def __init__(self) -> None:
        self.passed, self.feedback = True, False
        self.values: list[Any] = [100, 200]
        self.regions = tuple(SimpleNamespace(x=x, y=30, width=60, height=100) for x in (30, 140))
        self.score = SimpleNamespace(read_side=lambda image, side: (self.values[('1P','2P').index(side)],))
        self.grid = SimpleNamespace(detect=lambda crop: None)

    def __call__(self, image: Any) -> bool:
        return self.passed


def primed() -> tuple[Verifier, BorderGraceVerifier]:
    base = Verifier()
    grace = BorderGraceVerifier(base)
    assert grace.verify(IMAGE, 0., SIZE, 0)
    base.passed = False
    return base, grace


def test_bounded_grace_does_not_renew() -> None:
    base, grace = primed()
    for stamp in (1., 2., 3.):
        assert grace.verify(IMAGE, stamp, SIZE, 0)
    assert not grace.verify(IMAGE, 3.001, SIZE, 0)
    base.passed = True
    assert grace.verify(IMAGE, 4., SIZE, 0)
    base.passed = False
    assert grace.verify(IMAGE, 5., SIZE, 0)


def test_score_continuity_includes_granted_frame() -> None:
    base, grace = primed()
    base.values = [500, 600]
    assert grace.verify(IMAGE, 1., SIZE, 0)
    base.values = [200, 300]
    assert not grace.verify(IMAGE, 2., SIZE, 0)


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('score', [None, 0])
def test_missing_or_reset_score_invalidates(side: int, score: Any) -> None:
    base, grace = primed()
    base.values[side] = score
    assert not grace.verify(IMAGE, 1., SIZE, 0)
    base.values = [100, 200]
    assert not grace.verify(IMAGE, 2., SIZE, 0)


@pytest.mark.parametrize('case', ['feedback', 'dark', 'size', 'generation', 'time', 'position'])
def test_other_failures_never_receive_grace(case: str) -> None:
    base, grace = primed()
    image, stamp, size, generation = IMAGE, 1., SIZE, 0
    if case == 'feedback':
        base.feedback = True
    if case == 'dark':
        image = np.zeros_like(IMAGE)
    if case == 'size':
        size = (480, 360)
    if case == 'generation':
        generation = 1
    if case == 'time':
        stamp = -1.
    if case == 'position':
        base.grid.detect = lambda crop: SimpleNamespace(top_left=(150,150), bottom_right=(200,200))
    assert not grace.verify(image, stamp, size, generation)


def test_initial_failure_and_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    base = Verifier()
    base.passed = False
    monkeypatch.delenv('PUYO_LIVE_BORDER_GRACE', raising=False)
    assert wrap_verifier(base) is base
    monkeypatch.setenv('PUYO_LIVE_BORDER_GRACE', '1')
    grace = wrap_verifier(base)
    assert not verify_screen(grace, IMAGE, 0., SIZE, 0)
