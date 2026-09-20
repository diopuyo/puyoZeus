"""原NEXT変換を使う局所検収。実画像/全update陽性ではない。"""
from __future__ import annotations
from copy import deepcopy
from functools import wraps
from pathlib import Path
import sys
from types import FunctionType, MethodType, SimpleNamespace
from typing import Any
import unittest

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'g2_firing_hand_connection_2026-09-10_v1'
POLICY = ROOT.parent/'g2_firing_policy_2026-09-10_v1'
sys.path[:0] = [str(PARENT), str(POLICY)]
import firing_ticket_v4 as Q
from scripts import next_enqueue_live_shadow_v1 as N

CASES: list[dict[str, Any]] = []


def fixture() -> tuple[Any, Any, Any]:
    cls = type('ArtificialPipe', (), {})
    controller = N.NextEnqueueController(cls, None, {})
    env = {'__file__': str(Q.V3.V2.SOURCE), '__builtins__': __builtins__, '__next_live': controller}
    original = FunctionType(Q.V3.V2.expected(), env)
    actual = N._transformed(original, controller)
    cls.update = actual
    return cls(), controller, actual


def fake_code(actual: Any, wrapped: Any) -> Any:
    values = dict(actual.__globals__)
    text = 'def update(self, frame_idx, time_sec, frame):\n    return None\n'
    exec(compile(text, str(Q.V3.V2.SOURCE), 'exec', dont_inherit=True), values)
    value = values['update']
    value.__wrapped__ = wrapped
    return value


def wrapped(actual: Any) -> Any:
    @wraps(actual)
    def outer(*args: Any, **kwargs: Any) -> Any:
        return actual(*args, **kwargs)
    return outer


def run_entry(foreign: bool) -> dict[str, Any]:
    pipe, controller, actual = fixture()
    journal_controller = N.NextEnqueueController(type(pipe), None, {}) if foreign else controller
    class Factory:
        provider = SimpleNamespace(journal=SimpleNamespace(controller=journal_controller))
        @property
        def controller(self) -> Any:
            raise RuntimeError('artificial_old_qualified_barrier')
    factory, evidence = Factory(), {}
    def detect(self: Any, image: Any) -> Any:
        frame = sys._getframe()
        assert frame.f_back.f_code is actual.__code__
        return Q.V3.V2.OLD.qualified(frame, factory, actual.__globals__)
    pipe._match_detector = SimpleNamespace(detect=MethodType(detect, pipe))
    original = Q.V3.V2.OLD.qualified
    try:
        with Q.installed(pipe, evidence):
            pipe.update(0, 0.0, None)
    except RuntimeError as exc:
        assert str(exc) == 'artificial_old_qualified_barrier', repr(exc)
    else:
        raise AssertionError('artificial barrier required')
    assert Q.V3.V2.OLD.qualified is original and controller.active is None
    return dict(foreign_journal_controller=foreign, old_qualified_reached=True,
        native_pop=False, full_update=False, restored=evidence['qualified_restored'])


class Transform(unittest.TestCase):
    def test_existing_transform_normal(self) -> None:
        pipe, controller, actual = fixture()
        receipt = deepcopy(controller.transform_receipt)
        before = dict(actual.__globals__)
        self.assertTrue(Q.expected(actual, controller))
        self.assertIs(Q.bound_update(pipe), actual)
        self.assertEqual(receipt, controller.transform_receipt)
        self.assertEqual(set(before), set(actual.__globals__))
        self.assertTrue(all(value is actual.__globals__[name] for name, value in before.items()))
        CASES.append(dict(case='existing_transform', accepted=True, receipt_globals_unchanged=True))

    def test_outer_wrapper_real_delegation(self) -> None:
        pipe, controller, actual = fixture()
        type(pipe).update = wrapped(actual)
        self.assertIs(Q.bound_update(pipe), actual)

    def test_changed_final_code(self) -> None:
        pipe, controller, actual = fixture()
        value = FunctionType(actual.__code__.replace(co_name='changed'), actual.__globals__)
        value.__wrapped__ = actual.__wrapped__
        with self.assertRaisesRegex(AssertionError, 'firing_final_NEXT_transform_changed'):
            Q.expected(value, controller)

    def test_same_name_filename_fake_wrapper(self) -> None:
        pipe, controller, actual = fixture()
        type(pipe).update = fake_code(actual, actual.__wrapped__)
        with self.assertRaisesRegex(AssertionError, 'firing_final_NEXT_transform_changed'):
            Q.bound_update(pipe)
        CASES.append(dict(case='same_name_filename_fake_wrapper', rejected=True))

    def test_fake_original_wrapper(self) -> None:
        pipe, controller, actual = fixture()
        value = FunctionType(actual.__code__, actual.__globals__)
        value.__wrapped__ = fake_code(actual, actual.__wrapped__)
        with self.assertRaisesRegex(AssertionError, 'firing_wrapped_original_changed'):
            Q.expected(value, controller)

    def test_wrong_explicit_controller(self) -> None:
        pipe, controller, actual = fixture()
        other = N.NextEnqueueController(type(pipe), None, {})
        self.assertFalse(Q.expected(actual, other))

    def test_missing_wrapped_original(self) -> None:
        pipe, controller, actual = fixture()
        value = FunctionType(actual.__code__, actual.__globals__)
        type(pipe).update = value
        with self.assertRaisesRegex(AssertionError, 'firing_bound_final_update_missing_or_multiple'):
            Q.bound_update(pipe)

    def test_exception_restores_qualified(self) -> None:
        pipe, controller, actual = fixture()
        original, evidence, marker = Q.V3.V2.OLD.qualified, {}, RuntimeError('body marker')
        with self.assertRaises(RuntimeError) as caught:
            with Q.installed(pipe, evidence): raise marker
        self.assertIs(caught.exception, marker)
        self.assertIs(Q.V3.V2.OLD.qualified, original)
        self.assertTrue(evidence['qualified_restored'])

    def test_matching_journal_controller_entry(self) -> None:
        row = run_entry(False)
        CASES.append(dict(case='matching_controller_entry', **row))

    def test_foreign_journal_controller_current_gap(self) -> None:
        row = run_entry(True)
        CASES.append(dict(case='foreign_controller_unchecked', unclosed=True, **row))
