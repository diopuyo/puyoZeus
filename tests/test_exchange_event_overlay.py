"""既定OFFの互換性と、認識→会計→イベント表示の接続を検証する。"""
from pathlib import Path
from types import SimpleNamespace
import inspect
import json
import os

import numpy as np
import pytest

import scripts.visualize_advantage_overlay as vao
import src.exchange_event_overlay as adapter
from src.board import Board
from src.board_state_machine import BoardState as State
from src.exchange_event_features import D_COLUMNS, SIDE_COLUMNS
from src.exchange_event_evaluator import StaticInput
from tests.test_exchange_event_tracker import Models
from tests.test_advantage_overlay_fps_normalize import _FakeCapture


FPS, FRAMES = 30, 120


def result(t_sec: float) -> SimpleNamespace:
    """最初の発火・得点確定・着地が異なるフレームになる短い認識列。"""
    board = Board()
    event = SimpleNamespace(trigger_sec=1.0, end_sec=2.0, before_board=board,
                            chain_count=1, total_score=700, ojama_sent=10, mechanism="formula")
    state1 = State.CHAIN if 1 <= t_sec < 2 else State.STABLE
    if 2 <= t_sec < 2.2:
        state1 = State.TSUMO_FALL
    state2 = State.OJAMA_FALL if 2.3 <= t_sec < 2.5 else State.STABLE
    sides = [SimpleNamespace(state=st, confirmed_board=board, raw_board=board,
                             score=100 + (700 if idx == 0 and t_sec >= 2 else 0),
                             chain_event=event if idx == 0 and 1 <= t_sec < 2.5 else None,
                             next_pair=(1, 2), dnext_pair=(3, 4), next_slide_motion=False)
             for idx, st in enumerate((state1, state2))]
    return SimpleNamespace(p1=sides[0], p2=sides[1])


class Signals:
    """絶対信号の観測時刻を固定するテスト用入力。"""

    def __init__(self, *args: object) -> None:
        pass

    def update(self, r: object, snapshot: object, t_sec: float) -> str | None:
        return "tsumo" if t_sec >= 2 else None


class Pipeline:
    """動画フレーム時刻から上記の観測列を返す。"""

    def update(self, fi: int, t_sec: float, frame: np.ndarray) -> SimpleNamespace:
        return result(t_sec)

    def tsumo_count(self, side: str) -> int:
        return 1


def build_static(boards: tuple, snapshot: object, elapsed: float, m0: float) -> StaticInput:
    return StaticInput(np.zeros(len(D_COLUMNS)), m0, elapsed)


def stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """動画I/Oと重い指標だけを置換し、generateの状態管理と会計は実物を通す。"""
    monkeypatch.setattr(vao, "_acquire_model", lambda *a: object())
    monkeypatch.setattr(vao.RecognitionPipeline, "load_default", lambda **k: Pipeline())
    monkeypatch.setattr(vao.cv2, "VideoCapture", lambda *a: _FakeCapture(FPS, FRAMES))
    monkeypatch.setattr(vao, "_score_advantage", lambda *a, **k: (0, .5, []))
    monkeypatch.setattr(vao, "_threat", lambda *a: 0)
    monkeypatch.setattr(vao, "_exchange_static_input", build_static)
    monkeypatch.setattr(vao, "_ExchangeEventEndSignals", Signals)
    monkeypatch.setattr(vao.FileExchangeModels, "load", lambda *a, **k: Models())
    monkeypatch.setattr(adapter, "prefire_side_features", lambda *a: np.zeros(len(SIDE_COLUMNS)))


def test_generate_event_order_and_r2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stub(monkeypatch)
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("旧発火速報・決着ホールドを動かしてはいけない")
    monkeypatch.setattr(vao.EarlyFireTracker, "update", forbidden)
    monkeypatch.setattr(vao.ResolvedExchangeTracker, "update", forbidden)
    path = tmp_path / "display.npz"
    written = vao.generate(Path("short.mp4"), tmp_path / "on.mp4", 4, 1 / FPS,
                           render=False, enable_exchange_event_update=True,
                           enable_early_fire_reaction=True, enable_resolved_exchange_eval=True,
                           exchange_event_m0_predictor=lambda b, q: .5,
                           dump_display_timeline_path=path)
    assert written == FRAMES
    row = json.loads((tmp_path / "on.exchange_events.jsonl").read_text())
    assert [v["source"] for v in row["values"]] == ["S1", "S3", "G_fe"]
    assert row["chains"][0]["end_signal_sec"] < row["chains"][0]["score_finalize_sec"]
    with np.load(path) as saved:
        assert len(saved["t_sec"]) == FRAMES
        assert not saved["resolved_active"].any()
        for source, p1 in (("G_fe", .4), ("S1", .6), ("S3", .8)):
            assert np.all(saved["display_p1"][saved["source"] == source] == p1)


def test_off_never_loads_event_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stub(monkeypatch)
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("OFFでイベントモデルをロードしてはいけない")
    monkeypatch.setattr(vao.FileExchangeModels, "load", forbidden)
    dumps = [tmp_path / "default.npz", tmp_path / "explicit.npz"]
    for kwargs, dump in zip(({}, {"enable_exchange_event_update": False}), dumps):
        vao.generate(Path("short.mp4"), tmp_path / "out.mp4", 4, 1 / FPS,
                     render=False, dump_display_timeline_path=dump, **kwargs)
    assert dumps[0].read_bytes() == dumps[1].read_bytes()
    assert not (tmp_path / "out.exchange_events.jsonl").exists()


def test_m0_absent_does_not_evaluate_g(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter, "prefire_side_features", lambda *a: np.zeros(len(SIDE_COLUMNS)))
    overlay = adapter.ExchangeEventOverlay(Models(), build_static, Signals)
    counts = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                             chain_total_score_p1=0, chain_total_score_p2=0)
    overlay.update(result(0), object(), counts, 0, 1)
    assert overlay.tracker.source == "waiting_m0"
    assert overlay.tracker.probability is None
    overlay.update(result(1), object(), counts, 1, 1)
    assert overlay.tracker.source == "S1"


def test_default_flags_and_cli() -> None:
    signature = inspect.signature(vao.generate).parameters
    assert signature["enable_exchange_event_update"].default is False
    assert signature["exchange_event_model_dir"].default == Path("models/exchange_event_v1")
    assert "--exchange-event-update" in inspect.getsource(vao.main)
    assert "--exchange-event-model-dir" in inspect.getsource(vao.main)


def test_real_short_off_matches_prechange_bytes() -> None:
    """E2実走で生成した変更前/OFF成果物を、同じテストから再検証できる。"""
    root = os.environ.get("E2_SHORT_ARTIFACTS")
    if root is None:
        pytest.skip("E2_SHORT_ARTIFACTSで実走成果物ディレクトリを指定")
    directory = Path(root)
    for suffix in (".mp4", ".npz", "_display.npz"):
        assert (directory / ("baseline" + suffix)).read_bytes() == (
            directory / ("off" + suffix)).read_bytes()


def test_missing_display_does_not_count_held_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter, "prefire_side_features", lambda *a: np.zeros(len(SIDE_COLUMNS)))
    overlay = adapter.ExchangeEventOverlay(Models(), build_static, Signals)
    counts = SimpleNamespace(finalized_count_p1=0, finalized_count_p2=0,
                             chain_total_score_p1=0, chain_total_score_p2=0)
    for t in (0, 1, 2):
        overlay.update(result(t), object(), counts, t, 1, displayed_scores=(None, None))
    for frame in range(15):
        overlay.update(result(2 + frame / 30), object(), counts, 2 + frame / 30, 1,
                       displayed_scores=(None, None))
    assert overlay.tracker.source == "S1"
    assert overlay.tracker.latest_chain("1P").stable_frames == 0


def test_display_reader_preserves_missing_values() -> None:
    class Ocr:
        def read_side(self, frame: np.ndarray, side: str) -> tuple:
            return (700 if side == "1P" else None, .9)
    pipeline = SimpleNamespace(_score_ocr=Ocr())
    assert adapter.displayed_scores_from_pipeline(pipeline, np.zeros((1, 1, 3))) == (700, None)
