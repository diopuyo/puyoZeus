"""追加対の取り消しが既存の升と正当な確定を壊さないことを確認する。"""
import pytest

from src.board import Board
from src.tsumo_rollback import MAX_TRACK_SEC, TsumoRollback

ORIGIN = ((1, 2, 1), (2, 2, 3))
MOVED = ((1, 3, 1), (2, 3, 3))


def board(cells: tuple[tuple[int, int, int], ...] = ()) -> Board:
    result = Board()
    result.set(12, 0, 9)
    result.set(11, 0, 2)
    for r, c, value in cells:
        result.set(r, c, value)
    return result


def armed() -> tuple[TsumoRollback, Board]:
    tracker, confirmed = TsumoRollback(), board(ORIGIN)
    tracker.after_update(board(), confirmed, 1.0, active=True, state='stable',
                         next_shift=False, chain=False)
    return tracker, confirmed


def observe(tracker: TsumoRollback, confirmed: Board, observed: Board,
            **kwargs: object) -> None:
    options = dict(active=True, state='stable', next_shift=False, chain=False)
    options.update(kwargs)
    tracker.before_update(confirmed, observed, 1.1, **options)


def test_move_restores_only_the_pair() -> None:
    tracker, confirmed = armed()
    confirmed.set(10, 5, 9)
    observe(tracker, confirmed, board(MOVED))
    assert confirmed.get(1, 2) == confirmed.get(2, 2) == 0
    assert confirmed.get(12, 0) == confirmed.get(10, 5) == 9
    assert confirmed.get(11, 0) == 2
    assert tracker.rollback_count == 1


def test_real_landing_is_immediate_and_never_retracted() -> None:
    tracker, confirmed = armed()
    assert confirmed.get(1, 2) == 1
    for _ in range(10):
        observe(tracker, confirmed, board(ORIGIN))
    assert confirmed.get(1, 2) == 1
    assert tracker.rollback_count == 0


@pytest.mark.parametrize('option', [dict(chain=True), dict(next_shift=True),
    dict(active=False), dict(state='chain'), dict(state='gravity_settle'),
    dict(state='ojama_fall'), dict(state='effect'), dict(state='menu')])
def test_boundaries_discard_evidence(option: dict) -> None:
    tracker, confirmed = armed()
    observe(tracker, confirmed, board(MOVED), **option)
    assert confirmed.get(1, 2) == 1
    assert tracker.pending is None


@pytest.mark.parametrize('cells', [(), ((1, 3, 1),),
    ((1, 3, 2), (2, 3, 3)), (*MOVED, (0, 4, 4)),
    ((1, 2, 10), *MOVED), (*ORIGIN, *MOVED)])
def test_ambiguous_observation_does_not_retract(cells: tuple) -> None:
    tracker, confirmed = armed()
    observe(tracker, confirmed, board(cells))
    assert confirmed.get(1, 2) == 1
    assert tracker.rollback_count == 0


@pytest.mark.parametrize('time_sec', [0.9, 1.0 + MAX_TRACK_SEC + 0.01])
def test_expired_or_rewound_evidence_is_discarded(time_sec: float) -> None:
    tracker, confirmed = armed()
    tracker.before_update(confirmed, board(MOVED), time_sec, active=True,
                          state='stable', next_shift=False, chain=False)
    assert confirmed.get(1, 2) == 1
    assert tracker.pending is None


def test_newer_correction_is_not_overwritten() -> None:
    tracker, confirmed = armed()
    confirmed.set(1, 2, 9)
    observe(tracker, confirmed, board(MOVED))
    assert confirmed.get(1, 2) == 9
    assert tracker.last_rollback == ((2, 2, 3),)


@pytest.mark.parametrize('cells', [((1, 2, 1),), (*ORIGIN, (0, 0, 3)),
    ((11, 3, 1), (12, 3, 3)), ((1, 2, 9), (2, 2, 9))])
def test_non_pair_updates_are_not_tracked(cells: tuple) -> None:
    tracker = TsumoRollback()
    tracker.after_update(board(), board(cells), 1.0, active=True,
                         state='stable', next_shift=False, chain=False)
    assert tracker.pending is None


def test_flag_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_speed import recognition_config
    monkeypatch.delenv('PUYO_W68_ROLLBACK', raising=False)
    assert 'enable_w68_rollback' not in recognition_config({}, True)
    monkeypatch.setenv('PUYO_W68_ROLLBACK', '1')
    assert recognition_config({}, True)['enable_w68_rollback'] is True
