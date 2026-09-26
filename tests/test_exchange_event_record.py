"""E5の入力完全性、再計算、記録有無の等価性と欠測鮮度を検証する。"""
from __future__ import annotations

import gzip
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.visualize_advantage_overlay as vao
from scripts.replay_exchange_event_20260926 import compare, replay, static_builder
from src.board import Board
from src.board_state_machine import BoardState
from src.display_freshness import evaluation_freshness
from src.exchange_event_record import (
    ExchangeEventRecorder, decode, encode, read_records, static_key,
)
from tests.test_exchange_event_overlay import FPS, FRAMES, build_static, result, stub


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """認識と学習済みモデルだけ軽量化し、表示・記録経路は実物を通す。"""
    import src.exchange_event_m0 as m0
    stub(monkeypatch)
    monkeypatch.setattr(m0, "FileM0Predictor", lambda path: lambda b, q: .5)
    for name in ("baseline", "recorded"):
        root = tmp_path / name
        kwargs = {"exchange_event_record_path": root / "inputs.jsonl.gz"} if name == "recorded" else {}
        vao.generate(Path("short.mp4"), root / "overlay.mp4", 4, 1 / FPS,
            render=False, enable_exchange_event_update=True,
            exchange_event_m0_predictor=lambda b, q: .5,
            dump_exchange_event_path=root / "events.jsonl",
            dump_display_timeline_path=root / "display.npz", **kwargs)
    return tmp_path


def test_record_flag_has_no_effect_and_replay_is_exact(
    recorded: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """再生はVideoCaptureを一切呼ばず、記録の全フレームだけから復元する。"""
    def forbidden(*args: object) -> None:
        raise AssertionError("再生で動画を開いてはいけない")
    monkeypatch.setattr(vao.cv2, "VideoCapture", forbidden)
    baseline, captured, output = (recorded / name for name in ("baseline", "recorded", "replay"))
    compare(baseline, captured)
    status = replay(captured / "inputs.jsonl.gz", output)
    compare(captured, output)
    assert status["frames"] == status["display_frames"] == FRAMES
    rows = list(read_records(captured / "inputs.jsonl.gz"))
    assert sum(row["kind"] == "update" for row in rows) == FRAMES
    assert any(row["kind"] == "static_call" for row in rows)
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert {value["source"] for event in events for value in event["values"]} == {"S1", "S3", "G_fe"}


def test_replay_uses_changed_evaluator(recorded: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_exchange_event_tracker import Models
    monkeypatch.setattr(Models, "predict_source_probability", lambda self, name, features: .25)
    output = recorded / "changed"
    replay(recorded / "recorded/inputs.jsonl.gz", output)
    with np.load(output / "display.npz") as saved:
        assert np.all(saved["display_p1"] == .25)
    with pytest.raises(AssertionError):
        compare(recorded / "recorded", output)


def test_replay_uses_changed_end_signals(recorded: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_exchange_event_overlay import Signals
    monkeypatch.setattr(Signals, "update", lambda *args: None)
    output = recorded / "no_end"
    replay(recorded / "recorded/inputs.jsonl.gz", output)
    assert (recorded / "recorded/events.jsonl").read_bytes() != (output / "events.jsonl").read_bytes()


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.float32, np.float64])
def test_array_codec_preserves_dtype_and_bits(dtype: type) -> None:
    values = np.arange(12, dtype=dtype).reshape(3, 4)
    restored = decode(json.loads(json.dumps(encode(values))))
    assert restored.dtype == values.dtype and restored.tobytes() == values.tobytes()


@pytest.mark.parametrize("value", [None, (None, .125), BoardState.STABLE,
    BoardState.CHAIN, True, 2**54, "日本語"])
def test_scalar_codec_preserves_types(value: object) -> None:
    restored = decode(json.loads(json.dumps(encode(value))))
    assert type(restored) is type(value) and restored == value


def test_board_codec_copies_grid() -> None:
    original = Board()
    original._grid[-1, 0] = 9
    restored = decode(json.loads(json.dumps(encode(original))))
    np.testing.assert_array_equal(original._grid, restored._grid)
    original._grid[:] = 0
    assert restored._grid[-1, 0] == 9


def test_nan_and_namespace_roundtrip() -> None:
    value = SimpleNamespace(scores=(None, float("nan")), state=BoardState.OJAMA_FALL)
    restored = decode(json.loads(json.dumps(encode(value))))
    assert restored.scores[0] is None and np.isnan(restored.scores[1])
    assert restored.state is BoardState.OJAMA_FALL


def test_update_snapshot_is_independent_of_later_mutation(tmp_path: Path) -> None:
    path = tmp_path / "input.gz"
    writer = ExchangeEventRecorder(path, "test", True, Path("models"))
    observed = result(1)
    snapshot = SimpleNamespace(net_balance_capped=0., forecast_p1=0.,
                               total_dropped_to_p1=0., total_dropped_to_p2=30.)
    finalization = SimpleNamespace(finalized_count_p1=1, finalized_count_p2=0,
                                   chain_total_score_p1=700, chain_total_score_p2=0)
    writer.update(observed, snapshot, finalization, 1., 2, (700., None), (None, 0), (True, False))
    observed.p1.confirmed_board._grid[:] = 9
    observed.p1.chain_event.chain_count = 99
    writer.close()
    saved = list(read_records(path))[1]["args"]
    assert saved[0].p1.chain_event.chain_count == 1
    assert not saved[0].p1.confirmed_board._grid.any()
    assert saved[2].chain_total_score_p1 == 700
    assert saved[5:] == ((700., None), (None, 0), (True, False))


@pytest.mark.parametrize("field", ["net_balance_capped", "forecast_p1"])
def test_static_key_includes_accounting(field: str) -> None:
    boards = (Board(), Board())
    snapshot = SimpleNamespace(net_balance_capped=0., forecast_p1=0.)
    original = static_key(boards, snapshot)
    setattr(snapshot, field, 1.)
    assert static_key(boards, snapshot) != original


def test_static_cache_supports_new_board_combinations(recorded: Path) -> None:
    builder = static_builder(recorded / "recorded/inputs.jsonl.gz")
    boards = (Board(), Board())
    boards[0]._grid[-1, 0] = 1
    snapshot = SimpleNamespace(net_balance_capped=999., forecast_p1=1.)
    value = builder(boards, snapshot, 11., .75)
    assert value.elapsed_sec == 11. and value.m0_probability_1p == .75


@pytest.mark.parametrize("version", [0, 999])
def test_unknown_record_version_rejected(tmp_path: Path, version: int) -> None:
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps(dict(version=version)) + "\n")
    with pytest.raises(ValueError, match="記録版"):
        list(read_records(path))


def test_incomplete_record_rejected(tmp_path: Path) -> None:
    path = tmp_path / "partial.gz"
    writer = ExchangeEventRecorder(path, "test", False, Path("models"))
    writer.stream.close()
    with pytest.raises(ValueError, match="未完了"):
        list(read_records(path))


def test_record_flag_requires_on(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exchange-event-update"):
        vao.generate(Path("unused.mp4"), tmp_path / "unused.mp4", 1, 0,
                     exchange_event_record_path=tmp_path / "unused.gz")
    assert not (tmp_path / "unused.gz").exists()
    assert inspect.signature(vao.generate).parameters["exchange_event_record_path"].default is None


@pytest.mark.parametrize("mode,column,updates", [("off", "adv_raw_last", 0), ("on", "display_p1", 1)])
def test_freshness_ignores_smoothed_display(mode: str, column: str, updates: int) -> None:
    display = dict(adv_raw_last=np.array([1., 1., 1.]), display_p1=np.array([.5, .5, .6]),
                   display_adv=np.array([1., 2., 3.]))
    measured = evaluation_freshness(display, mode, FPS)
    assert measured["column"] == column and measured["updates"] == updates


def test_missing_freshness_stays_in_population() -> None:
    measured = evaluation_freshness(dict(adv_raw_last=np.array([np.nan, np.nan, 1., 1.])), "off", FPS)
    assert measured["frames"] == 4 and measured["adjacent_pairs"] == 3
    assert measured["missing_frames"] == measured["equal_pairs"] == 2
    assert measured["updates"] == 1


@pytest.mark.parametrize("mode,values", [("unknown", [1.]), ("off", [np.inf]), ("on", [-np.inf])])
def test_invalid_evaluation_freshness_rejected(mode: str, values: list) -> None:
    display = dict(adv_raw_last=np.array(values), display_p1=np.array(values))
    with pytest.raises(ValueError):
        evaluation_freshness(display, mode, FPS)
