"""撃ち合いの束ね・優先順位・時刻記録を検証する。"""
from pathlib import Path
import json

import numpy as np
import pytest

from src.exchange_event_evaluator import StaticInput
from src.exchange_event_features import D_COLUMNS, SIDE_COLUMNS
from src.exchange_event_tracker import ExchangeEventTracker


class Models:
    """列順を維持し、状態ごとに識別できる勝率を返す。"""

    elapsed_thresholds = (10.0, 30.0)

    def predict_source_probability(self, name: str, features: np.ndarray) -> float:
        return {"G_fe": .4, "S1": .6, "S3": .8}[name]


def static() -> StaticInput:
    return StaticInput(np.zeros(len(D_COLUMNS)), .5, 1.0)


def fire(tracker: ExchangeEventTracker, t: float = 2.0,
         triggers: tuple = (2.0, None)) -> None:
    tracker.fire(t_sec=t, triggers=triggers, static=static(),
                 prefire_sides=np.zeros((2, len(SIDE_COLUMNS))), score_elapsed_sec=t)


@pytest.fixture
def tracker() -> ExchangeEventTracker:
    result = ExchangeEventTracker(Models())
    result.boundary(1, 0)
    result.static(static(), 1, (1, 1))
    return result


@pytest.mark.parametrize("first", [0, 1])
def test_all_participating_chains_must_finalize(tracker: ExchangeEventTracker, first: int) -> None:
    fire(tracker, triggers=(2.0, 2.0))
    assert tracker.source == "S1"
    sides = ("1P", "2P")
    for side in sides:
        tracker.end(side, 3, "next")
    assert all(c.awaiting_finalize for c in tracker.resolver.active())
    tracker.finalize(sides[first], 4, 700)
    tracker.finish_frame(4)
    assert tracker.source == "S1"
    tracker.finalize(sides[1-first], 5, 1400)
    tracker.finish_frame(5)
    assert tracker.source == "S3"
    assert tracker.probability == .8


@pytest.mark.parametrize("triggers", [(2.0, None), (None, 2.0), (2.0, 2.0)])
def test_trigger_repeats_do_not_duplicate(tracker: ExchangeEventTracker, triggers: tuple) -> None:
    for stamp in (2.0, 2.1, 2.2):
        fire(tracker, stamp, triggers)
    assert len(tracker.records) == 1
    assert len(tracker.current.chains) == sum(v is not None for v in triggers)
    assert len(tracker.current.values) == 1


def test_formula_retrigger_is_same_growing_chain(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    fire(tracker, 3, (3, None))
    assert len(tracker.current.chains) == 1


def test_reply_is_bundled_but_first_trigger_flags_are_frozen(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    fire(tracker, 3, (None, 3))
    assert len(tracker.current.chains) == 2
    assert tracker.firing.firing == (True, False)


def test_end_without_score_stays_s1_even_after_landing(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.end("1P", 3, "tsumo")
    tracker.landing("2P", 4)
    tracker.finish_frame(5)
    assert not tracker.static(static(), 5, (5, 5))
    assert tracker.source == "S1"


@pytest.mark.parametrize("landing_time", [3.0, 5.0])
def test_s1_s3_g_and_late_score_timestamp(tracker: ExchangeEventTracker, landing_time: float) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.landing("2P", landing_time)
    tracker.finalize("1P", 4, 700)
    tracker.finish_frame(4)
    assert tracker.source == "S3"
    assert not tracker.static(static(), 4, (4, 4))
    assert not tracker.static(static(), 6, (3, 6))
    assert tracker.static(static(), 6, (6, 6))
    assert tracker.source == "G_fe"
    record = tracker.records[0]
    assert [v["source"] for v in record.values] == ["S1", "S3", "G_fe"]
    assert record.chains[0].score_finalize_sec == 4
    assert record.landings[0]["t_sec"] == landing_time


def test_s3_has_priority_over_later_reply_s1(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 4, 700)
    tracker.finish_frame(4)
    fire(tracker, 5, (None, 5))
    assert tracker.source == "S3"
    assert not tracker.static(static(), 6, (6, 6))
    tracker.end("2P", 7, "next")
    tracker.finalize("2P", 8, 1400)
    tracker.finish_frame(8)
    assert len(tracker.records) == 1
    assert [v["source"] for v in tracker.current.values] == ["S1", "S3", "S3"]


def test_second_chain_same_side_gets_new_id(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 4, 700)
    fire(tracker, 5, (5, None))
    assert len({c.chain_id for c in tracker.current.chains}) == 2


def test_boundary_does_not_finalize_and_clears_probability(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.boundary(2, 3)
    assert tracker.records[0].close_reason == "match_boundary"
    assert tracker.probability is None
    fire(tracker)
    assert len(tracker.records) == 2


def test_equal_cancel_needs_no_landing(tracker: ExchangeEventTracker) -> None:
    fire(tracker, triggers=(2, 2))
    for side in ("1P", "2P"):
        tracker.end(side, 3, "next")
        tracker.finalize(side, 4, 700)
    tracker.finish_frame(4)
    assert tracker.static(static(), 5, (5, 5))


@pytest.mark.parametrize("score", [-1, np.nan, np.inf])
def test_invalid_finalize_rejected(tracker: ExchangeEventTracker, score: float) -> None:
    fire(tracker)
    with pytest.raises(ValueError):
        tracker.finalize("1P", 3, score)


def test_dump_keeps_unfinished_exchange(tracker: ExchangeEventTracker, tmp_path: Path) -> None:
    fire(tracker)
    tracker.save(tmp_path / "events.jsonl")
    row = json.loads((tmp_path / "events.jsonl").read_text())
    assert row["closed_sec"] is None
    assert row["values"][0] == dict(source="S1", t_sec=2.0, p1=.6)


def test_score_before_absolute_end_waits_for_signal(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.finalize("1P", 3, 700)
    tracker.finish_frame(3)
    assert tracker.source == "S1"
    tracker.end("1P", 4, "next")
    tracker.finish_frame(4)
    assert tracker.source == "S3"
    assert tracker.current.chains[0].score_finalize_sec == 3


def test_fall_entry_is_not_landing_completion(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 4, 700)
    tracker.finish_frame(4)
    tracker.fall_start("2P", 5)
    assert not tracker.static(static(), 6, (6, 6))
    tracker.landing("2P", 7)
    assert not tracker.static(static(), 7, (7, 7))
    assert tracker.static(static(), 8, (8, 8))
    assert tracker.records[0].landings == [dict(side="2P", fall_start_sec=5, t_sec=7)]


@pytest.mark.parametrize("times", [(6, 3), (3, 6), (3, 3)])
def test_one_fresh_confirmed_board_cannot_release(tracker: ExchangeEventTracker, times: tuple) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 4, 700)
    tracker.finish_frame(4)
    tracker.landing("2P", 5)
    assert not tracker.static(static(), 6, times)


def test_repeated_finalize_does_not_duplicate_s3(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    for stamp in (4, 5, 6):
        tracker.finalize("1P", stamp, 700)
        tracker.finish_frame(stamp)
    assert [v["source"] for v in tracker.current.values] == ["S1", "S3"]


def test_s3_score_input_is_ocr_sum_and_margin_rate(tracker: ExchangeEventTracker) -> None:
    from src.exchange_event_features import score_features
    captured = []
    original = tracker.models.predict_source_probability
    tracker.models.predict_source_probability = lambda name, x: (captured.append(x), original(name, x))[1]
    fire(tracker, 150, (150, None))
    tracker.end("1P", 152, "next")
    tracker.finalize("1P", 153, 700)
    fire(tracker, 154, (154, None))
    tracker.end("1P", 156, "next")
    tracker.finalize("1P", 157, 700)
    tracker.finish_frame(157)
    np.testing.assert_allclose(captured[-1][-7:], score_features(np.zeros(2), np.array([1400, 0]), 150))


def test_real_formula_continuation_revokes_early_end(tracker: ExchangeEventTracker) -> None:
    from src.chain_id_resolver import ChainObservation, ObservationKind
    kwargs = dict(static=static(), prefire_sides=np.zeros((2, len(SIDE_COLUMNS))),
                  score_elapsed_sec=2.0)
    tracker.fire(t_sec=2, triggers=(2, None), observations=(ChainObservation(
        "1P", 2, ObservationKind.FORMULA_STEP, chain_count=1, total_score=40),), **kwargs)
    tracker.end("1P", 3, "next")
    tracker.fire(t_sec=4, triggers=(3.9, None), observations=(ChainObservation(
        "1P", 4, ObservationKind.FORMULA_STEP, chain_count=2, total_score=360),), **kwargs)
    assert len(tracker.current.chains) == 1
    chain = tracker.current.chains[0]
    assert chain.end_signal_sec is None
    assert chain.end_signals[0]["revoked_sec"] == 4
    tracker.end("1P", 5, "tsumo")
    tracker.finalize("1P", 6, 360)
    tracker.finish_frame(6)
    assert tracker.source == "S3"


def test_physical_interval_closes_even_when_static_model_is_unavailable(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.finalize("1P", 4, 700)
    tracker.finish_frame(4)
    tracker.landing("2P", 5)
    assert tracker.close_confirmed(6, (6, 6))
    assert tracker.source == "S3" and tracker.probability == .8
    fire(tracker, 7, (None, 7))
    assert len(tracker.records) == 2
    assert tracker.source == "S1"


def test_new_functions_within_fifty_lines() -> None:
    import ast
    root = Path(__file__).resolve().parents[1]
    for name in ("src/exchange_event_tracker.py", "src/exchange_event_overlay.py",
                 "tests/test_exchange_event_tracker.py", "tests/test_exchange_event_overlay.py"):
        for node in ast.walk(ast.parse((root / name).read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef):
                assert node.end_lineno - node.lineno + 1 <= 50, (name, node.name)


@pytest.mark.parametrize("side", ["1P", "2P"])
def test_ten_post_end_frames_advance_s3_without_accounting(tracker: ExchangeEventTracker, side: str) -> None:
    fire(tracker, triggers=(2, None) if side == "1P" else (None, 2))
    for frame in range(20):
        tracker.observe_score(side, 2 + frame / 30, 800, 100)
    tracker.end(side, 3, "next")
    for frame in range(10):
        tracker.observe_score(side, 3 + frame / 30, 800, 100)
        tracker.finish_frame(3 + frame / 30)
        assert tracker.source == ("S3" if frame == 9 else "S1")
    chain = tracker.latest_chain(side)
    assert chain.score_ready_reason == "display_stable"
    assert chain.score_finalize_sec is None
    assert chain.score_delta == 700
    assert tracker.resolver.active()[0].awaiting_finalize


@pytest.mark.parametrize("interruption", [None, 801, -1, np.nan])
def test_display_stability_resets(tracker: ExchangeEventTracker, interruption: float | None) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    for frame in range(9):
        tracker.observe_score("1P", 3 + frame / 30, 800, 100)
    tracker.observe_score("1P", 3.3, interruption, 100)
    tracker.observe_score("1P", 3.4, 800, 100)
    tracker.finish_frame(3.4)
    assert tracker.source == "S1"
    assert tracker.latest_chain("1P").stable_frames == 1


@pytest.mark.parametrize("formula,expected", [(700, "S3"), (699, "S1"), (None, "S1")])
def test_formula_match_is_immediate(tracker: ExchangeEventTracker, formula: float | None, expected: str) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.observe_score("1P", 3, 800, 100, formula)
    tracker.finish_frame(3)
    assert tracker.source == expected
    if expected == "S3":
        assert tracker.latest_chain("1P").score_ready_reason == "formula_match"


def test_early_score_waits_for_all_chains(tracker: ExchangeEventTracker) -> None:
    fire(tracker, triggers=(2, 2))
    tracker.end("1P", 3, "next")
    tracker.observe_score("1P", 3, 800, 100, 700)
    tracker.finish_frame(3)
    assert tracker.source == "S1"
    tracker.end("2P", 4, "next")
    tracker.observe_score("2P", 4, 1500, 100, 1400)
    tracker.finish_frame(4)
    assert tracker.source == "S3"


def test_late_accounting_timestamp_survives_exchange_close(tracker: ExchangeEventTracker) -> None:
    fire(tracker)
    tracker.end("1P", 3, "next")
    tracker.observe_score("1P", 3, 800, 100, 700)
    tracker.finish_frame(3)
    tracker.landing("2P", 4)
    assert tracker.static(static(), 5, (5, 5))
    tracker.finalize("1P", 6, 710)
    chain = tracker.records[0].chains[0]
    assert chain.score_finalize_sec == 6
    assert chain.score_ready_sec == 3 and chain.score_delta == 700
    tracker.fall_start("2P", 7)
    tracker.landing("2P", 8)
    assert tracker.records[0].landings[-1] == dict(side="2P", fall_start_sec=7, t_sec=8)


def test_early_close_echo_and_real_continuation_keep_exchange_identity(tracker: ExchangeEventTracker) -> None:
    from src.chain_id_resolver import ChainObservation, ObservationKind
    kwargs = dict(static=static(), prefire_sides=np.zeros((2, len(SIDE_COLUMNS))),
                  score_elapsed_sec=2.0)
    tracker.fire(t_sec=2, triggers=(2, None), observations=(ChainObservation(
        "1P", 2, ObservationKind.FORMULA_STEP, 1, 40),), **kwargs)
    tracker.end("1P", 3, "next")
    tracker.observe_score("1P", 3, 140, 100, 40)
    tracker.finish_frame(3)
    tracker.landing("2P", 3.1)
    tracker.static(static(), 3.2, (3.2, 3.2))
    tracker.fire(t_sec=3.3, triggers=(3.3, None), observations=(ChainObservation(
        "1P", 3.3, ObservationKind.CHAIN_SETTLED, 1, 40),), **kwargs)
    assert tracker.current is None and len(tracker.records) == 1
    tracker.fire(t_sec=4, triggers=(3.3, None), observations=(ChainObservation(
        "1P", 4, ObservationKind.FORMULA_STEP, 2, 360),), **kwargs)
    assert tracker.current is tracker.records[0]
    assert len(tracker.records) == 1
    assert tracker.latest_chain("1P").score_ready_sec is None
    assert tracker.latest_chain("1P").score_before == 100
