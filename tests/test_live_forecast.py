"""依頼 A の連鎖の予測 (ChainForecast) の配信側の試験。PC2 の tools/test_forecast.py を pytest に移し、配信の旗と retR の選び方を足す。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

native = pytest.importorskip('puyo_core')

from src.b3_state import SideState  # noqa: E402
from src.fire_board import ChainForecast, PairForecast, native_trace, step_scores  # noqa: E402
from tests.test_b3_chain_prediction import formula, option  # noqa: E402
from tests.test_live_fire_board import boards  # noqa: E402

FPS = 1 / 30
# video_29 の 2P の 7 連鎖の撃った盤 (PC2 の確かめと同じ)
BOARD = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 5, 0, 0, 0, 0], [0, 3, 0, 0, 0, 0],
                  [5, 2, 3, 0, 0, 0], [2, 3, 3, 0, 0, 0], [2, 2, 3, 0, 0, 0], [4, 3, 3, 0, 0, 5], [2, 2, 2, 4, 4, 3],
                  [4, 3, 5, 3, 4, 3], [4, 4, 3, 5, 5, 4], [3, 3, 5, 3, 3, 4]], dtype=np.int8)
TRUE = [180, 840, 640, 1700, 4020, 4900, 6500]


def middles() -> list[np.ndarray]:
    trace = native.simulate_chain_trace_py(list(map(int, BOARD.ravel())), True)
    return [np.asarray(step[0], dtype=np.int8).reshape(13, 6) for step in trace]


def dropped_read() -> np.ndarray:
    """1 段目は合い、後の段が変わる 1 マス落としの読み (列の一番上のぷよを 1 つ消す)。"""
    trace = native_trace(native)
    for r in range(1, 13):
        for c in range(6):
            if BOARD[r, c] == 0 or BOARD[r - 1, c] != 0:
                continue
            board = BOARD.copy()
            board[r, c] = 0
            steps = step_scores(trace(board)[0], 0)
            if steps and steps[0] == TRUE[0] and steps != TRUE:
                return board
    raise AssertionError('1 マス落とした読みが作れない')


def run(mid_ok: bool, dup: bool = False) -> tuple[object, list[float]]:
    forecast, t, mids = ChainForecast(native_trace(native), 'T'), 10.0, middles()
    for _ in range(3):
        forecast.observe(t, None, BOARD.tolist())
        t += FPS
    forecast.observe(t, None, dropped_read().tolist())   # t−1 が読み落とし
    t += FPS
    ids = []
    for k, product in enumerate(TRUE, 1):
        if k >= 2:   # 段の間は式が切れ、途中の盤が読める (mid_ok) か読めない
            for _ in range(14):
                forecast.observe(t, None, mids[k - 1].tolist() if mid_ok else None)
                t += FPS
        for j in range(25):
            forecast.observe(t, dict(valid=True, product=product), mids[k - 1].tolist() if mid_ok else None)
            t += FPS
            if dup and k == 3 and j == 10:   # 0.4 秒切れて同じ値がまた出る
                for _ in range(12):
                    forecast.observe(t, None, None)
                    t += FPS
        ids.append(forecast.current().chain_id)
    return forecast.current(), ids


def test_step_scores_match_native() -> None:
    assert step_scores(native_trace(native)(BOARD)[0], 0) == TRUE


def test_refind_fixes_dropped_read_and_keeps_identity() -> None:
    forecast, ids = run(True)
    assert len(set(ids)) == 1
    assert list(forecast.steps) == TRUE and forecast.source == 'refind' and forecast.refinds >= 1


def test_rescue_keeps_observed_plus_remaining() -> None:
    forecast, _ = run(False)
    assert list(forecast.observed) == TRUE and forecast.source == 'rescue' and forecast.rescues >= 1


def test_repeated_value_is_one_step() -> None:
    forecast, _ = run(True, dup=True)
    assert list(forecast.observed) == TRUE


def side_frames(side: SideState, fired: np.ndarray, before: np.ndarray, start: float, product: int | None) -> None:
    state = 'STABLE' if product is None else 'CHAIN'
    form = dict(valid=product is not None, product=product)
    side.update_frame(start, dict(state=state, confirmed_board=before, raw_cnn_board=fired, formula=form), True)


@pytest.mark.parametrize('enabled', [False, True])
def test_b3_uses_forecast_only_with_live_flag(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """旗 OFF は予測を作らず置き場所の補完の候補のまま (今と同じ)。旗 ON は連鎖の予測 (100+320) を使う。"""
    before, fired = boards()
    monkeypatch.setattr('src.b3_state.origin_candidates', lambda *_: ([option((100, 740))], ''))
    side = SideState('1P', enable_fire_board=enabled)
    assert (side.forecast is not None) is enabled
    for t in (.8, .9):
        side_frames(side, fired, before, t, None)
    side_frames(side, fired, before, 1., 100)
    side.on_fire_start(dict(fire_id='a', trigger_sec=1., observed_at_sec=1., chain_before=before.tolist()))
    formula(side, 1., 100)
    assert side.episode.pred_score == (420 if enabled else 740)
    if enabled:
        assert side.episode.forecast.source == 'fire_board' and side.episode.forecast.steps == (100, 320)


def test_b3_state_with_forecast_survives_pickle(monkeypatch: pytest.MonkeyPatch) -> None:
    """配信の B3 は状態を pickle して別の process に送る (10-28 24ka の通し直しで中の関数が pickle できず B3 が止まった)。"""
    import pickle
    before, fired = boards()
    monkeypatch.setattr('src.b3_state.origin_candidates', lambda *_: ([option((100, 740))], ''))
    side = SideState('1P', enable_fire_board=True)
    for t in (.8, .9):
        side_frames(side, fired, before, t, None)
    side_frames(side, fired, before, 1., 100)
    side.on_fire_start(dict(fire_id='a', trigger_sec=1., observed_at_sec=1., chain_before=before.tolist()))
    formula(side, 1., 100)
    copy = pickle.loads(pickle.dumps(side))
    assert copy.episode.pred_score == 420 and copy.forecast.current().steps == (100, 320)
    side_frames(copy, fired, before, 1.4, 320)   # 受け側でも計算を続けられる (puyo_core を読み直す)
    assert copy.forecast.current().observed == (100, 320)
    pair = pickle.loads(pickle.dumps(PairForecast()))
    pair.observe(1., 1, [dict(raw_cnn_board=fired.tolist(), formula=dict(valid=False)), {}])


def test_b3_old_path_when_forecast_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_B3_FORECAST', '0')
    assert SideState('1P', enable_fire_board=True).forecast is None


def test_retr_selects_attack_by_identity_not_ex_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """EX の chain['started'] が段で進んでも、識別子で選ぶので撃った盤の予測が使われる (RESULT_4 の (c) の再現と直り)。"""
    from src.phase_j._live_ex_frozen.src import dl1k_retr as retr
    from src.phase_j._live_ex_frozen.src.dl1k_engine import ExEngine
    from tests.test_live_fire_board_retr import fixture
    monkeypatch.setenv('PUYO_EX_RETR', '1')
    engine = fixture()
    _, fired = boards()
    raw = [dict(raw_cnn_board=fired.tolist(), formula=dict(valid=False)), {}]
    ExEngine.step(engine, .9, 1, raw)
    raw[0]['formula'] = dict(valid=True, product=100)
    ExEngine.step(engine, 1.01, 1, raw)
    assert type(engine.retr_fire).__name__ == PairForecast.__name__   # 凍結の写しの名前空間のクラス (src/fire_board.py と同じ中身)
    assert engine.retr_fire.current(0, 1.01).steps == (100, 320)
    engine.chains[0]['started'] = 3.5   # EX の連鎖の始まりが次の段へ進んだ (前の決まりでは外れる)
    engine.state.t = 1.2
    value = dict(pending=[0, 1], rate=70, board_prediction_active=[True, False],
                 reconciliation=[dict(times=[1.5]), None], boards=[np.zeros((13, 6)).tolist(), None])
    retr.compute_retr(engine, value)
    assert engine.retr_diag[1]['I'] == 6 and engine.retr_diag[1]['fire_back'] == 'fire_board'
    assert engine.retr_diag[1]['fc_total_o'] == 420
