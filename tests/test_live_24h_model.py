"""EX欠測時のモデル固定、C2だけの経路、失効時の再推論を検査する。"""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.phase_j.live_24h import LEARNING_COLUMNS, Live24h


class Worker:
    def __init__(self, extra: dict | None = None) -> None:
        self.extra = extra

    def available(self, now: float, input_t: float,
                  admitted_at: float | None = None) -> tuple[dict | None, str]:
        return self.extra, 'available' if self.extra is not None else 'missing'

    def take_events(self) -> list:
        return []

    def close(self) -> None:
        pass


def result() -> dict:
    return dict(row={'base_column': 2.}, raw_probability=.05, t_sec=3., frame=90, epoch=1)


def extra(**changes: object) -> dict:
    values = {name: .7 for name in LEARNING_COLUMNS}
    values.update(changes)
    return dict(epoch=1, serial=2, extra=values)


def model(calls: list) -> SimpleNamespace:
    def predict(row: dict) -> float:
        calls.append(row)
        return .4 if math.isnan(row[LEARNING_COLUMNS[0]]) else .8
    return SimpleNamespace(predict_row=predict)


@pytest.mark.parametrize('value', [None, extra(), extra(**{LEARNING_COLUMNS[0]: math.nan})])
def test_one_model_including_missing_and_partial_missing(tmp_path: Path, value: dict | None) -> None:
    calls = []
    worker = Worker(value)
    owner = Live24h(tmp_path, model(calls), {}, lambda _: worker, use_ama=True, temperature_enabled=False,
                    counter_fire_shift_enabled=False, midgame_board_layer_enabled=False)
    prediction = owner.predict(result())
    assert prediction in (.4, .8) and prediction != result()['raw_probability']
    assert len(calls) == 1 and len(calls[0]) == len(LEARNING_COLUMNS) + 1
    assert calls[0]['base_column'] == 2.
    assert owner.last['model_source'] == 'C2+EX'
    owner.close()


def test_ama_off_does_not_start_ex_or_load_model(tmp_path: Path) -> None:
    owner = Live24h(tmp_path, worker_factory=lambda _: pytest.fail('EX起動'), use_ama=False, temperature_enabled=False,
                    counter_fire_shift_enabled=False, midgame_board_layer_enabled=False)
    assert owner.predict(result()) == .05
    assert owner.model is None and owner.worker is None
    owner.close()


def test_expiry_uses_same_model_without_renewing_b3_timestamp(tmp_path: Path) -> None:
    calls, worker = [], Worker(extra())
    owner = Live24h(tmp_path, model(calls), {}, lambda _: worker, use_ama=True, temperature_enabled=False,
                    counter_fire_shift_enabled=False, midgame_board_layer_enabled=False)
    owner.epoch = 1
    display = SimpleNamespace(probability=owner.predict(result()), last_success=3., failed=False)
    worker.extra = None
    owner.revalidate(display)
    assert display.probability == .4 and display.last_success == 3.
    assert len(calls) == 2 and all(math.isnan(calls[-1][name]) for name in LEARNING_COLUMNS)
    owner.close()


def test_boundary_never_reuses_previous_b3_row(tmp_path: Path) -> None:
    calls, worker = [], Worker(extra())
    owner = Live24h(tmp_path, model(calls), {}, lambda _: worker, use_ama=True, temperature_enabled=False,
                    counter_fire_shift_enabled=False, midgame_board_layer_enabled=False)
    owner.predict(result())
    owner.epoch = 2
    display = SimpleNamespace(probability=None, failed=False)
    owner.revalidate(display)
    assert display.probability is None and len(calls) == 1
    owner.close()


def test_expiry_during_inference_is_missing_not_c2(tmp_path: Path) -> None:
    worker, calls = Worker(extra()), []
    def predict(row: dict) -> float:
        calls.append(row)
        worker.extra = None
        return .4 if math.isnan(row[LEARNING_COLUMNS[0]]) else .8
    owner = Live24h(tmp_path, SimpleNamespace(predict_row=predict), {}, lambda _: worker, use_ama=True, temperature_enabled=False,
                    counter_fire_shift_enabled=False, midgame_board_layer_enabled=False)
    assert owner.predict(result()) == .4 and len(calls) == 2
    owner.close()


def test_model_failure_is_not_replaced_by_c2(tmp_path: Path) -> None:
    owner = Live24h(tmp_path, SimpleNamespace(predict_row=lambda _: math.nan), {},
                   lambda _: Worker(), use_ama=True, temperature_enabled=False,
                    counter_fire_shift_enabled=False, midgame_board_layer_enabled=False)
    with pytest.raises(ValueError):
        owner.predict(result())
    owner.close()


def test_game_change_clears_old_row_even_if_input_epoch_is_unchanged(tmp_path: Path) -> None:
    calls, worker = [], Worker(extra())
    worker.poll = lambda: []
    worker.submit = lambda *args: None
    owner = Live24h(tmp_path, model(calls), {}, lambda _: worker, use_ama=True, temperature_enabled=False,
                    counter_fire_shift_enabled=False, midgame_board_layer_enabled=False)
    owner.epoch, owner.game = 1, 1
    owner.predict(result())
    owner.update(dict(game_idx=2, _ex={}), 1, 0.)
    display = SimpleNamespace(probability=None, failed=False)
    owner.revalidate(display)
    assert owner.base_result is None and display.probability is None and len(calls) == 1
    owner.close()


@pytest.mark.parametrize('kind', ['member_count', 'column_order', 'audit_column', 'unknown_column'])
def test_bad_model_contract_is_rejected(kind: str) -> None:
    from src.b3_column_spec import NET_LEDGER_FEATURE_NAMES
    from src.phase_j.live_24h_model import AUDIT_COLUMNS, validate_model
    columns = [NET_LEDGER_FEATURE_NAMES[0], *LEARNING_COLUMNS]
    count = 2 if kind == 'member_count' else 3
    if kind == 'column_order':
        columns[1:] = reversed(columns[1:])
    if kind == 'audit_column':
        columns.append(AUDIT_COLUMNS[0])
    if kind == 'unknown_column':
        columns.append('not_available_in_live_rows')
    evaluator = SimpleNamespace(columns=columns)
    candidate = SimpleNamespace(members=[None]*count, static=evaluator, exchange=evaluator)
    with pytest.raises(ValueError):
        validate_model(candidate)
