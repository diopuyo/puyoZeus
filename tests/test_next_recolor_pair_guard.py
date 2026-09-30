"""cycle65 NEXT履歴色補正の対整合ガードのテスト (段1: 単体)。

q第14試合 1P 883.8秒の状況 (キューが 1 手先へ進んだ古い対) を再現し、
既定OFFで従来と同一挙動、ONで観測色を守ることを確認する。
"""
from __future__ import annotations

import numpy as np
import pytest

from src import recognition_pipeline as rp
from src.board import Board, COLOR_BLUE, COLOR_GREEN, COLOR_RED, COLOR_UNKNOWN
from src.board_state_machine import BoardState
from src.next_recolor_pair_guard import (
    OUTCOME_ACCEPTED, OUTCOME_KEPT_OBSERVED, OUTCOME_SUBSTITUTED,
    RECOLOR_HISTORY_LOOKBACK, pair_consistent, select_recolor_pair,
)
from tests.test_recognition_pipeline import (
    _StubImageReader, _StubMatchDetector, _dummy_frame,
)

G, R, B = COLOR_GREEN, COLOR_RED, COLOR_BLUE


# ---------------- 純粋関数 ----------------

def test_match14_stale_queue_keeps_observed() -> None:
    """queue[-2] が 1 手先 (青青) で観測が緑赤・履歴にも緑赤なし → 上書きしない。"""
    pair, outcome = select_recolor_pair([G, R], [(B, B), (B, B)])
    assert pair is None and outcome == OUTCOME_KEPT_OBSERVED


def test_match14_stale_queue_substitutes_from_history() -> None:
    """履歴に緑赤の対があればそれへ差し替える (queue[-2] は 1 手先の青青)。"""
    pair, outcome = select_recolor_pair([G, R], [(G, R), (B, B), (R, G)])
    assert outcome == OUTCOME_SUBSTITUTED and pair == (G, R)


def test_correct_recolor_still_applied() -> None:
    """queue[-2] が観測と整合 (順序違いも可) → 従来どおりその対で補正。"""
    pair, outcome = select_recolor_pair([R, G], [(B, B), (G, R), (B, R)])
    assert outcome == OUTCOME_ACCEPTED and pair == (G, R)


def test_unknown_observation_still_recolored() -> None:
    """不明セルは無視して整合判定。片方不明でも、両方不明でも従来の対を採用。"""
    queue = [(B, B), (G, R), (B, R)]
    assert select_recolor_pair([COLOR_UNKNOWN, R], queue) == ((G, R), OUTCOME_ACCEPTED)
    assert select_recolor_pair([COLOR_UNKNOWN, COLOR_UNKNOWN], queue) == (
        (G, R), OUTCOME_ACCEPTED)


def test_single_entry_queue_uses_last_and_empty_queue_keeps() -> None:
    """キュー 1 件なら末尾が従来の採用対。空なら補正なし。"""
    assert select_recolor_pair([G, R], [(G, R)]) == ((G, R), OUTCOME_ACCEPTED)
    assert select_recolor_pair([G, R], []) == (None, OUTCOME_KEPT_OBSERVED)


def test_lookback_is_bounded() -> None:
    """探索は直近 RECOLOR_HISTORY_LOOKBACK 件まで。それより古い一致は使わない。"""
    queue = [(G, R)] + [(B, B)] * RECOLOR_HISTORY_LOOKBACK
    assert select_recolor_pair([G, R], queue) == (None, OUTCOME_KEPT_OBSERVED)


def test_multiset_semantics() -> None:
    """同色2個の観測は同色対にだけ整合し、片色対とは整合しない。"""
    assert pair_consistent([B, B], (B, B))
    assert not pair_consistent([B, B], (B, R))
    assert pair_consistent([R, B], (B, R))


# ---------------- 配線 (cycle65 経路) ----------------

def _cycle65_pipe(enable: bool | None, queue: list[tuple[int, int]]):
    """STABLE→STABLE で新規 2 セル (緑赤縦置き) が確定する 1P を作り、infer_placement の
    引数を記録するパイプラインを返す。"""
    base = Board()
    base.set(12, 0, B)
    observed = base.copy()
    observed.set(11, 2, G)
    observed.set(12, 2, R)
    kwargs = {} if enable is None else dict(enable_next_recolor_pair_guard=enable)
    pipe = rp.RecognitionPipeline(
        image_reader=_StubImageReader(observed, Board()),  # type: ignore[arg-type]
        match_state_detector=_StubMatchDetector(True),  # type: ignore[arg-type]
        score_ocr=None, chain_tracker_1p=None, chain_tracker_2p=None,
        stable_frame_count=1, **kwargs,
    )
    ctx = pipe._sm_1p.context
    ctx.state = BoardState.STABLE
    ctx.confirmed_board = base.copy()
    ctx.pending_board = base.copy()
    ctx.next_queue = list(queue)
    real_update = pipe._sm_1p.update

    def stable_update(frame_idx, signals):  # noqa: ANN001, ANN202
        """TSUMO_FALL を経由せず、STABLE のまま新規 2 セルが確定した状況を作る。"""
        ctx_after = real_update(frame_idx, signals)
        ctx_after.state = BoardState.STABLE
        ctx_after.confirmed_board = observed.copy()
        return ctx_after

    pipe._sm_1p.update = stable_update  # type: ignore[method-assign]
    return pipe


@pytest.fixture
def recorded_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """cycle65 経路の infer_placement 呼出しの対を記録し、推論は行わない (None)。"""
    calls: list[tuple[int, int]] = []

    def fake(prev, cur, pair, *args, **kwargs):  # noqa: ANN001
        calls.append(tuple(pair))
        return None

    monkeypatch.setattr(rp, "infer_placement", fake)
    return calls


def _drive(pipe) -> None:
    pipe.update(0, 100.0, _dummy_frame())


def test_wiring_flag_defaults_off() -> None:
    """既定OFF・監査カウンタは全 0。"""
    pipe = _cycle65_pipe(None, [])
    assert pipe._enable_next_recolor_pair_guard is False
    assert set(pipe.next_recolor_guard_counts.values()) == {0}


def test_wiring_off_is_unchanged_and_writes_no_audit(recorded_calls) -> None:
    """OFF: 古い対でも従来どおり queue[-2] を渡し、監査は何も記録しない。"""
    pipe = _cycle65_pipe(False, [(G, R), (B, B), (B, R)])
    _drive(pipe)
    assert recorded_calls, "cycle65 経路に入っていること (空通過防止)"
    assert recorded_calls[-1] == (B, B)  # 従来の採用対 = queue[-2]
    assert sum(pipe.next_recolor_guard_counts.values()) == 0
    assert pipe.next_recolor_guard_log == []


def test_wiring_on_stale_queue_substitutes(recorded_calls) -> None:
    """ON: queue[-2] が青青でも履歴の緑赤へ差し替えて渡す。"""
    pipe = _cycle65_pipe(True, [(G, R), (B, B), (B, R)])
    _drive(pipe)
    assert pipe.next_recolor_guard_counts[OUTCOME_SUBSTITUTED] >= 1
    assert recorded_calls and recorded_calls[-1] == (G, R)


def test_wiring_on_no_match_keeps_observed(recorded_calls) -> None:
    """ON: 整合する対が履歴に無ければ infer_placement を呼ばず観測色のまま。"""
    pipe = _cycle65_pipe(True, [(B, B), (B, B), (B, R)])
    _drive(pipe)
    assert pipe.next_recolor_guard_counts[OUTCOME_KEPT_OBSERVED] >= 1
    assert recorded_calls == []
    board = pipe._sm_1p.context.confirmed_board
    assert (int(board.get(11, 2)), int(board.get(12, 2))) == (G, R)


def test_wiring_on_correct_pair_accepted(recorded_calls) -> None:
    """ON: queue[-2] が観測と整合すれば従来と同じ対を渡す (accepted)。"""
    pipe = _cycle65_pipe(True, [(B, B), (R, G), (B, R)])
    _drive(pipe)
    assert pipe.next_recolor_guard_counts[OUTCOME_ACCEPTED] >= 1
    assert recorded_calls[-1] == (R, G)


def test_ojama_observation_is_informative() -> None:
    """おじゃま観測は証拠として扱う (実測: 9 件中 8 件は本物のおじゃま)。

    色ぷよの対とは整合しないので、対を当てず観測のまま (kept_observed) にする。
    """
    assert select_recolor_pair([9, 2], [(B, G)]) == (None, OUTCOME_KEPT_OBSERVED)
    assert select_recolor_pair([9, 9], [(B, B)]) == (None, OUTCOME_KEPT_OBSERVED)
