"""Phase I: HiddenRowValidator のテスト."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.board import (
    COLOR_BLUE,
    COLOR_RED,
    Board,
)
from src.board_state_machine import BoardState
from src.self_supervised.hidden_row_validator import (
    COMPONENT_HIDDEN_ROW,
    HiddenRowValidator,
)


# ============================
# Test fixtures (PipelineResult / SideResult の最小スタブ)
# ============================


@dataclass
class _FakeSide:
    side: str = "1P"
    state: BoardState = BoardState.STABLE
    confirmed_board: Board | None = None
    next_pair: tuple[int, int] | None = None
    dnext_pair: tuple[int, int] | None = None


@dataclass
class _FakeResult:
    is_match_active: bool = True
    p1: _FakeSide = field(default_factory=_FakeSide)
    p2: _FakeSide = field(
        default_factory=lambda: _FakeSide(side="2P"),
    )


def _make_board(cells: dict[tuple[int, int], int]) -> Board:
    b = Board()
    for (r, c), color in cells.items():
        b.set(r, c, color)
    return b


def _stable_result(
    board_1p: Board,
    next_pair_1p: tuple[int, int] | None = (COLOR_RED, COLOR_BLUE),
    state_1p: BoardState = BoardState.STABLE,
    board_2p: Board | None = None,
    state_2p: BoardState = BoardState.STABLE,
) -> _FakeResult:
    p1 = _FakeSide(
        side="1P",
        state=state_1p,
        confirmed_board=board_1p,
        next_pair=next_pair_1p,
    )
    p2 = _FakeSide(
        side="2P",
        state=state_2p,
        confirmed_board=board_2p if board_2p is not None else _make_board({}),
        next_pair=None,
    )
    return _FakeResult(is_match_active=True, p1=p1, p2=p2)


# ============================
# tests
# ============================


def test_init_no_buffer() -> None:
    v = HiddenRowValidator()
    assert v.collect() == []


def test_match_inactive_no_update() -> None:
    """is_match_active=False の場合は内部更新せず."""
    v = HiddenRowValidator()
    res = _stable_result(_make_board({}))
    res.is_match_active = False
    v.update(0, 0.0, res, None)
    assert v.collect() == []


def test_first_stable_no_inference() -> None:
    """初回 STABLE では prev_board が無いため pending に追加されない."""
    v = HiddenRowValidator()
    res = _stable_result(_make_board({}))
    v.update(0, 0.0, res, None)
    assert v.collect() == []


def test_two_stables_normal_drop_no_pending() -> None:
    """2 セル新規出現 → 隠し段 EMPTY 確定 → 色付き pending は登録されず."""
    v = HiddenRowValidator()
    # frame 0: 空盤面 STABLE
    v.update(0, 0.0, _stable_result(_make_board({})), None)
    # frame 1: ペア 2 個落下 (row 11/12 col 2 に RED/BLUE)
    cells = {(11, 2): COLOR_RED, (12, 2): COLOR_BLUE}
    v.update(1, 0.5, _stable_result(_make_board(cells)), None)
    # 隠し段は EMPTY 確定 → tracker は pending を保持しない
    # 何も emit されないこと
    assert v.collect() == []


def test_one_new_cell_creates_pending_and_reveal() -> None:
    """row 1 に 1 個出現 → 隠し段に推論 puyo が乗る → 後続の連鎖で reveal."""
    v = HiddenRowValidator()
    # frame 0: 空盤面 STABLE
    v.update(0, 0.0, _stable_result(_make_board({})), None)
    # frame 1: row 1 (= HIDDEN_ROWS) に RED 1 個 (隠し段に BLUE がある推論)
    board1 = _make_board({(1, 2): COLOR_RED})
    v.update(1, 0.5, _stable_result(board1), None)
    # まだ reveal 起きていない → buffer 空
    assert v.collect() == []
    # frame 2: 連鎖 (CHAIN state) を経由
    res2 = _stable_result(board1, state_1p=BoardState.CHAIN)
    v.update(2, 1.0, res2, None)
    # frame 3: STABLE で row 1 col 2 に隠し段 BLUE が落ちて出現
    board3 = _make_board({(1, 2): COLOR_BLUE})
    v.update(3, 1.5, _stable_result(board3, next_pair_1p=None), None)
    samples = v.collect()
    assert len(samples) >= 1
    s = samples[0]
    assert s.component == COMPONENT_HIDDEN_ROW
    # 隠し段推論 BLUE / 観測 BLUE → match=True
    assert s.metadata["match"] is True
    assert s.label == COLOR_BLUE


def test_one_new_cell_mismatch_recorded() -> None:
    """推論色 != 観測色 でも擬似ラベルとして記録される (confidence=0)."""
    v = HiddenRowValidator()
    v.update(0, 0.0, _stable_result(_make_board({})), None)
    # frame 1: row 1 に RED 1 個 (next_pair=RED,BLUE → 推論色は BLUE)
    board1 = _make_board({(1, 2): COLOR_RED})
    v.update(1, 0.5, _stable_result(board1), None)
    # frame 2: TSUMO_FALL を経由
    v.update(
        2, 1.0,
        _stable_result(board1, state_1p=BoardState.TSUMO_FALL),
        None,
    )
    # frame 3: row 1 col 2 に GREEN が出現 (推論 BLUE と不一致)
    # 連鎖で RED が消え、隠し段の puyo (推論 BLUE) が落ちてくると期待したが、
    # 実際は GREEN だった (隠し段推論誤り)
    from src.board import COLOR_GREEN
    board3 = _make_board({(1, 2): COLOR_GREEN})
    v.update(3, 1.5, _stable_result(board3, next_pair_1p=None), None)
    samples = v.collect()
    assert len(samples) >= 1
    s = samples[0]
    assert s.metadata["match"] is False
    assert s.label == COLOR_GREEN
    assert s.confidence == 0.0


def test_no_action_state_no_reveal_check_strict() -> None:
    """厳格モード: STABLE → STABLE (アクション系を経由しない) では reveal 判定しない."""
    v = HiddenRowValidator(enable_lenient_reveal=False)
    v.update(0, 0.0, _stable_result(_make_board({})), None)
    board1 = _make_board({(1, 2): COLOR_RED})
    v.update(1, 0.5, _stable_result(board1), None)
    # ※ アクション系を経由せず直接 STABLE
    board2 = _make_board({(1, 2): COLOR_BLUE})
    v.update(2, 1.0, _stable_result(board2), None)
    # action_seen=False のため reveal 判定スキップ → emit 0
    assert v.collect() == []


def test_no_action_state_lenient_emit() -> None:
    """lenient モード ON (default): action 経由なしでも row 1 色変化で emit."""
    v = HiddenRowValidator(enable_lenient_reveal=True)
    v.update(0, 0.0, _stable_result(_make_board({})), None)
    # frame 1: 推論する基となる prev_board を確定
    board1 = _make_board({(1, 2): COLOR_RED})
    v.update(1, 0.5, _stable_result(board1), None)
    # frame 2: action 経由なし、row 1 col 2 が RED → BLUE に変化
    board2 = _make_board({(1, 2): COLOR_BLUE})
    v.update(2, 1.0, _stable_result(board2), None)
    samples = v.collect()
    # lenient 経路で emit されるはず
    lenient = [
        s for s in samples
        if s.metadata.get("source") == "reveal_track_lenient"
    ]
    assert len(lenient) >= 1
    # lenient confidence
    from src.self_supervised.hidden_row_validator import (
        LENIENT_REVEAL_CONFIDENCE,
    )
    assert lenient[0].confidence == LENIENT_REVEAL_CONFIDENCE


def test_lenient_no_emit_without_color_change() -> None:
    """lenient モードでも row 1 に色変化が無ければ emit しない."""
    v = HiddenRowValidator(enable_lenient_reveal=True)
    v.update(0, 0.0, _stable_result(_make_board({})), None)
    # frame 1: row 1 RED
    board1 = _make_board({(1, 2): COLOR_RED})
    v.update(1, 0.5, _stable_result(board1), None)
    # frame 2: 同じ盤面 (色変化なし)
    v.update(2, 1.0, _stable_result(board1), None)
    # row 1 が変化していないので lenient でも emit せず
    samples = v.collect()
    lenient = [
        s for s in samples
        if s.metadata.get("source") == "reveal_track_lenient"
    ]
    assert len(lenient) == 0


def test_reset_clears_state_and_buffer() -> None:
    v = HiddenRowValidator()
    v.update(0, 0.0, _stable_result(_make_board({})), None)
    board1 = _make_board({(1, 2): COLOR_RED})
    v.update(1, 0.5, _stable_result(board1), None)
    v.reset()
    # reset 後は前 frame 情報が消えて初回扱い
    v.update(0, 2.0, _stable_result(_make_board({})), None)
    assert v.collect() == []


def test_collect_empties_buffer() -> None:
    """collect() を 2 回呼ぶと 2 回目は空."""
    v = HiddenRowValidator()
    # reveal を発生させる
    v.update(0, 0.0, _stable_result(_make_board({})), None)
    board1 = _make_board({(1, 2): COLOR_RED})
    v.update(1, 0.5, _stable_result(board1), None)
    v.update(
        2, 1.0,
        _stable_result(board1, state_1p=BoardState.CHAIN),
        None,
    )
    board3 = _make_board({(1, 2): COLOR_BLUE})
    v.update(3, 1.5, _stable_result(board3, next_pair_1p=None), None)
    first = v.collect()
    second = v.collect()
    assert len(first) >= 1
    assert second == []


def test_silent_skip_on_inferrer_failure() -> None:
    """infer_hidden_row が失敗してもエラーにせず continue."""
    # board 系統に異常状態 (None confirmed_board) を渡しても落ちない
    v = HiddenRowValidator()
    res = _stable_result(_make_board({}))
    # confirmed_board が None
    res.p1.confirmed_board = None
    v.update(0, 0.0, res, None)
    # buffer が空のまま
    assert v.collect() == []


def test_2p_side_also_processed() -> None:
    """2P 側も同様に処理される."""
    v = HiddenRowValidator()
    p1_empty = _make_board({})
    p2_empty = _make_board({})
    res0 = _FakeResult(
        is_match_active=True,
        p1=_FakeSide(
            side="1P", state=BoardState.STABLE,
            confirmed_board=p1_empty, next_pair=(COLOR_RED, COLOR_BLUE),
        ),
        p2=_FakeSide(
            side="2P", state=BoardState.STABLE,
            confirmed_board=p2_empty, next_pair=(COLOR_RED, COLOR_BLUE),
        ),
    )
    v.update(0, 0.0, res0, None)
    # 2P 側 row 1 に RED 1 個
    p2_board1 = _make_board({(1, 2): COLOR_RED})
    res1 = _FakeResult(
        is_match_active=True,
        p1=_FakeSide(
            side="1P", state=BoardState.STABLE,
            confirmed_board=p1_empty, next_pair=None,
        ),
        p2=_FakeSide(
            side="2P", state=BoardState.STABLE,
            confirmed_board=p2_board1, next_pair=(COLOR_RED, COLOR_BLUE),
        ),
    )
    v.update(1, 0.5, res1, None)
    # OJAMA 経由
    res2 = _FakeResult(
        is_match_active=True,
        p1=_FakeSide(
            side="1P", state=BoardState.STABLE,
            confirmed_board=p1_empty, next_pair=None,
        ),
        p2=_FakeSide(
            side="2P", state=BoardState.OJAMA_FALL,
            confirmed_board=p2_board1, next_pair=(COLOR_RED, COLOR_BLUE),
        ),
    )
    v.update(2, 1.0, res2, None)
    # 2P 側に reveal
    p2_board3 = _make_board({(1, 2): COLOR_BLUE})
    res3 = _FakeResult(
        is_match_active=True,
        p1=_FakeSide(
            side="1P", state=BoardState.STABLE,
            confirmed_board=p1_empty, next_pair=None,
        ),
        p2=_FakeSide(
            side="2P", state=BoardState.STABLE,
            confirmed_board=p2_board3, next_pair=None,
        ),
    )
    v.update(3, 1.5, res3, None)
    samples = v.collect()
    assert any(s.input_data["side"] == "2P" for s in samples)


def test_sample_metadata_contains_frame_idx() -> None:
    """擬似ラベルに frame_idx が含まれる."""
    v = HiddenRowValidator()
    v.update(0, 0.0, _stable_result(_make_board({})), None)
    board1 = _make_board({(1, 2): COLOR_RED})
    v.update(1, 0.5, _stable_result(board1), None)
    v.update(
        2, 1.0,
        _stable_result(board1, state_1p=BoardState.CHAIN),
        None,
    )
    board3 = _make_board({(1, 2): COLOR_BLUE})
    v.update(3, 1.5, _stable_result(board3, next_pair_1p=None), None)
    samples = v.collect()
    assert len(samples) >= 1
    assert "frame_idx" in samples[0].metadata
    assert "original_inference_t" in samples[0].metadata
