"""R37 (2026-10-23): 試合終了後の試合開始を「得点 0 → 増加」で判定する解除の試験。

user の決まり (8/19「必ず試合前スコアは 0」、10/23「0 同士なら試合前、動いた瞬間試合開始」):
  試合終了 → 両者 0 点 (試合前) → どちらかの得点が増え、増えた値が 2 回続けて読めたら開始。
  盤面の画素は使わない。試合終了の固定 (match_end_locked) の最中でも開始してよい。
  前の試合の最終得点が残る画面 (0 を経ていない) は試合として数えない。
既定 OFF では何も変わらない (bit-identical)。
"""

from __future__ import annotations

import pytest

from src.board_state_machine import BoardState
from src.recognition_pipeline import PipelineResult, RecognitionPipeline
from tests.test_recognition_pipeline import (
    _dummy_frame,
    _empty_board,
    _FakeScoreTrackerSeq,
    _high_variance_frame,
    _StubImageReader,
    _StubMatchDetector,
    _StubScoreZeroDetector,
)

# 前の試合の最終得点 (やった!/ばたんきゅー の画面に残る値、c18 の実例)
FINAL = (13441, 9536)
ZERO = (0, 0)
# 最初の設置の得点 (video_30 試合 2: 14/25)
FIRST = (14, 25)
FRAME_DT = 0.05


class _ScriptedMatchEnd:
    """update() の戻り (match_end_locked) を外から切り替えられる MatchEndDetector スタブ。"""

    def __init__(self, locked: bool = True) -> None:
        self.locked = locked

    def update(self, frame, time_sec) -> bool:  # noqa: ANN001
        return self.locked

    def is_locked(self, time_sec) -> bool:  # noqa: ANN001
        return self.locked


def _make_pipe(
    pairs: list[tuple[int, int]],
    zero_then_move: bool,
    **extra: bool,
) -> tuple[RecognitionPipeline, _ScriptedMatchEnd]:
    """試合終了の固定が True のままの pipeline。得点は pairs を 1 コマずつ読む。"""
    end = _ScriptedMatchEnd(locked=True)
    pipe = RecognitionPipeline(
        image_reader=_StubImageReader(_empty_board(), _empty_board()),  # type: ignore[arg-type]
        match_state_detector=_StubMatchDetector(in_match=False),  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        match_end_detector=end,  # type: ignore[arg-type]
        score_zero_detector=_StubScoreZeroDetector(both_zero=False),  # type: ignore[arg-type]
        enable_large_roi_throttle=False,
        enable_post_match_lockdown_latch=True,
        enable_lockdown_score_zero_then_move_release=zero_then_move,
        # 隔離: 両者ほぼ 0 で pipeline 全体を reset() する別機構を切る
        enable_match_start_full_clear=False,
        # 本番の採用済み構成 (得点が動いても、盤面の画素が無ければ固定・ラッチ中は信じない)
        enable_result_screen_hardening=True,
        **extra,
    )
    pipe._score_tracker_1p = _FakeScoreTrackerSeq([p[0] for p in pairs])  # type: ignore[assignment]
    pipe._score_tracker_2p = _FakeScoreTrackerSeq([p[1] for p in pairs])  # type: ignore[assignment]
    return pipe, end


def _run(
    pipe: RecognitionPipeline, n: int, frame_fn=_dummy_frame, start: int = 0,
) -> list[PipelineResult]:
    """n コマ進める (0.05 秒刻み)。返り値は各コマの結果。"""
    return [
        pipe.update(start + i, 5.0 + FRAME_DT * (start + i), frame_fn())
        for i in range(n)
    ]


# 試合終了 (最終得点 3 コマ) → 0 同士 4 コマ → 最初の設置 14/25 が 2 コマ → さらに 3 コマ
SEQ_ZERO_THEN_MOVE = [FINAL] * 3 + [ZERO] * 4 + [FIRST] * 5
# 0 までは同じ。最初の設置 (14/25) の読取りが 2 コマ目 (添字 8) で確定する。
CONFIRM_INDEX = 3 + 4 + 1


def test_zero_then_move_starts_after_two_same_reads_even_during_end_lock() -> None:
    """0 を経た後の増加が 2 回続いたら開始。試合終了の固定の最中・盤面の画素なしでも開始する。"""
    pipe, _ = _make_pipe(SEQ_ZERO_THEN_MOVE, zero_then_move=True)
    results = _run(pipe, len(SEQ_ZERO_THEN_MOVE))
    # 確定したコマの次のコマ (CONFIRM_INDEX + 1) からラッチが外れ、試合中になる。
    assert pipe._lockdown_score_start_confirmed is True
    assert [r.is_match_active for r in results[: CONFIRM_INDEX + 1]] == [False] * (CONFIRM_INDEX + 1)
    assert results[CONFIRM_INDEX + 1].is_match_active is True
    assert results[CONFIRM_INDEX + 1].match_end_locked is True, "固定の最中に始まったことの確認"
    assert pipe._post_match_lockdown_active is False
    assert pipe._board_shows_real_gameplay(_dummy_frame()) is False, "画素の確認に頼っていない"


def test_zero_then_move_does_not_trap_state_machine_in_effect() -> None:
    """固定の残りで EFFECT に入らない (確定盤面が無いまま EFFECT に入ると連鎖まで STABLE へ戻れない)。

    最初の設置の得点が動いている間は別の経路 (cycle 71g) で演出扱いが外れるため、
    得点の窓 (1 秒) が過ぎるまで流してから確かめる。
    """
    seq = SEQ_ZERO_THEN_MOVE + [FIRST] * 40
    pipe, _ = _make_pipe(seq, zero_then_move=True)
    results = _run(pipe, len(seq))
    after = results[CONFIRM_INDEX + 1:]
    assert all(r.match_end_locked for r in after), "前提: 固定は最後まで True のまま"
    assert all(r.is_match_active for r in after)
    assert all(r.p1.state != BoardState.EFFECT for r in after)


def test_zero_then_move_off_keeps_latch_and_state_untouched() -> None:
    """既定 OFF: 同じ入力でもラッチは外れず、試合中にならず、R37 の状態は動かない。"""
    pipe, _ = _make_pipe(SEQ_ZERO_THEN_MOVE, zero_then_move=False)
    results = _run(pipe, len(SEQ_ZERO_THEN_MOVE))
    assert all(r.is_match_active is False for r in results)
    assert pipe._post_match_lockdown_active is True
    assert pipe._lockdown_zero_seen is False
    assert pipe._lockdown_score_start_confirmed is False


def test_zero_then_move_ignores_increase_without_passing_zero() -> None:
    """0 を経ていない増加 (前の試合の最終得点の読み違い・残り) は、何回続けても開始にしない。"""
    jitter = [FINAL] + [(13447, 9536)] * 6 + [(13447, 9541)] * 6
    pipe, _ = _make_pipe(jitter, zero_then_move=True)
    results = _run(pipe, len(jitter))
    assert pipe._lockdown_zero_seen is False
    assert pipe._lockdown_score_start_confirmed is False
    assert pipe._post_match_lockdown_active is True
    assert all(r.is_match_active is False for r in results)


def test_zero_then_move_single_misread_does_not_start() -> None:
    """増加が 1 回だけ・違う値・途切れた値では開始しない。同じ値が連続して初めて開始する。"""
    seq = (
        [FINAL] + [ZERO] * 3
        + [(14, 0)] + [ZERO] * 2          # 1 回だけの読み違い → 0 に戻る
        + [(14, 0), (15, 0)]               # 違う値
        + [(14, 0)] + [ZERO] + [(14, 0)]   # 間に 0 が挟まる (連続でない)
    )
    pipe, _ = _make_pipe(seq, zero_then_move=True)
    _run(pipe, len(seq))
    assert pipe._lockdown_zero_seen is True
    assert pipe._lockdown_score_start_confirmed is False
    assert pipe._post_match_lockdown_active is True
    # 続けて同じ値が 2 回読めたら開始する (2P の側だけの増加でもよい)。
    seq2 = seq + [(0, 25), (0, 25)]
    pipe2, _ = _make_pipe(seq2, zero_then_move=True)
    _run(pipe2, len(seq2))
    assert pipe2._lockdown_score_start_confirmed is True


def test_count_lockdown_increase_none_does_not_break_run() -> None:
    """読めなかったコマ (None) は連続を切らない。0 以下は数え直し、違う値は 1 から数え直す。"""
    pipe, _ = _make_pipe([ZERO], zero_then_move=True)
    count = pipe._count_lockdown_increase
    assert [count("1P", v) for v in (14, None, 14)] == [1, 1, 2]
    assert [count("1P", v) for v in (15, 15, 0, 15)] == [1, 2, 0, 1]
    assert count("2P", 25) == 1, "側ごとに独立"


def test_zero_then_move_requires_latch_flag() -> None:
    """ラッチが無いのに解除規則だけ ON は、黙って無効にせず ValueError。"""
    with pytest.raises(ValueError):
        RecognitionPipeline(
            image_reader=_StubImageReader(_empty_board(), _empty_board()),  # type: ignore[arg-type]
            match_state_detector=_StubMatchDetector(in_match=False),  # type: ignore[arg-type]
            score_ocr=None,
            chain_tracker_1p=None,
            chain_tracker_2p=None,
            enable_post_match_lockdown_latch=False,
            enable_lockdown_score_zero_then_move_release=True,
        )


def test_zero_then_move_reset_clears_state() -> None:
    """reset() (試合切替) で「0 を経たか」「増加の確認」を最初に戻す。"""
    pipe, _ = _make_pipe(SEQ_ZERO_THEN_MOVE, zero_then_move=True)
    _run(pipe, len(SEQ_ZERO_THEN_MOVE))
    assert pipe._lockdown_score_start_confirmed is True
    pipe.reset()
    assert pipe._lockdown_zero_seen is False
    assert pipe._lockdown_score_start_confirmed is False
    assert pipe._lockdown_inc_run == {"1P": 0, "2P": 0}


def test_zero_then_move_new_match_end_edge_rearms_the_wait() -> None:
    """開始の確定後に次の試合終了の立ち上がりが来たら、また「0 を経るまで試合外」に戻る。"""
    seq = SEQ_ZERO_THEN_MOVE + [(300, 200)] * 2 + [(48200, 30100)] * 6
    pipe, end = _make_pipe(seq, zero_then_move=True)
    _run(pipe, len(SEQ_ZERO_THEN_MOVE))
    assert pipe._lockdown_score_start_confirmed is True
    end.locked = False
    _run(pipe, 2, start=len(SEQ_ZERO_THEN_MOVE))
    end.locked = True   # 次の試合の終了 (最終得点が残る)
    results = _run(pipe, 6, start=len(SEQ_ZERO_THEN_MOVE) + 2)
    assert pipe._post_match_lockdown_active is True
    assert pipe._lockdown_score_start_confirmed is False
    assert all(r.is_match_active is False for r in results)


def test_zero_then_move_keeps_legacy_release_as_fallback() -> None:
    """ON でも従来の解除 (スコアが動く + 盤面の画素) は残る = ON は OFF より遅くならない。

    試合中に出た単発の誤爆 (0 を経ていない) のラッチを従来どおり短く外すための保険。
    """
    moving = [(100 + 10 * i, 100 + 10 * i) for i in range(40)]
    pipe, end = _make_pipe(
        moving, zero_then_move=True, enable_lockdown_score_moving_release=True,
    )
    _run(pipe, 1)
    assert pipe._post_match_lockdown_active is True
    end.locked = False
    _run(pipe, 30, frame_fn=_high_variance_frame, start=1)
    assert pipe._post_match_lockdown_active is False
    assert pipe._lockdown_score_start_confirmed is False, "R37 の経路ではなく従来の経路で外れた"
