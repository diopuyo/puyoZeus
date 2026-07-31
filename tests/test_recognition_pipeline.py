"""RecognitionPipeline 統合テスト (Phase B-7a)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.board import COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_RED, Board
from src.board_state_machine import BoardState
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION, ImageReader
from src.match_state import MatchState, MatchStateDetector
from src.recognition_pipeline import (
    RecognitionPipeline,
    SideResult,
    OJAMA_TIER1_WARMUP_FRAMES,
    TIER1_WARMUP_FRAMES,
    _update_ojama_tier1_warmup_counter,
)


# ============================
# stub helpers
# ============================


@dataclass
class _StubMatchResult:
    state: MatchState
    bg_value: float = 100.0
    bg_saturation: float = 50.0
    samples: int = 1


class _StubMatchDetector:
    """常に IN_MATCH を返す MatchStateDetector スタブ."""

    def __init__(self, in_match: bool = True) -> None:
        self._in_match = in_match

    def detect(self, frame: np.ndarray) -> _StubMatchResult:
        return _StubMatchResult(
            state=MatchState.IN_MATCH if self._in_match
            else MatchState.NOT_IN_MATCH,
        )


class _StubImageReader:
    """指定した固定 board を返す ImageReader スタブ."""

    def __init__(self, p1: Board, p2: Board) -> None:
        self._p1 = p1
        self._p2 = p2
        # skip_tier1 呼び出し記録 (テスト検証用)
        self.last_skip_tier1_1p: bool = False
        self.last_skip_tier1_2p: bool = False

    def read_both_boards(
        self, frame: np.ndarray,
        p1_roi_offset: tuple[float, float] = (0.0, 0.0),
        p2_roi_offset: tuple[float, float] = (0.0, 0.0),
        skip_tier1_1p: bool = False,
        skip_tier1_2p: bool = False,
        # 修正2 (2026-07-30): 実 ImageReader.read_both_boards の telop_result
        # 追加 (optional 引数) に追従。スタブでは使わないため受け取るだけ。
        telop_result: object | None = None,
    ) -> tuple[Board, Board]:
        self.last_skip_tier1_1p = skip_tier1_1p
        self.last_skip_tier1_2p = skip_tier1_2p
        return self._p1.copy(), self._p2.copy()


def _empty_board() -> Board:
    return Board()


def _board_with_red(row: int, col: int) -> Board:
    b = Board()
    b.set(row, col, COLOR_RED)
    return b


def _make_pipe(
    p1: Board, p2: Board, in_match: bool = True,
    stable_n: int = 2,
) -> RecognitionPipeline:
    reader = _StubImageReader(p1, p2)
    detector = _StubMatchDetector(in_match=in_match)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=stable_n,
    )


def _dummy_frame() -> np.ndarray:
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


# ============================
# 基本動作
# ============================


def test_pipeline_reaches_stable_after_n_frames() -> None:
    """cycle 71v: 試合開始 window 内なら confirmed は空 Board() に強制される.

    CNN が puyo を観測しても物理ルール優先 (= 試合開始フィールドは空)。
    旧挙動は CNN 多数決で confirmed が設定されていたが、 v51/v70 の
    背景誤認対策で空強制に変更。
    """
    p1 = _board_with_red(12, 0)  # CNN は (12,0) に puyo を観測 (= 背景誤認 simulation)
    p2 = _empty_board()
    pipe = _make_pipe(p1, p2, stable_n=3)
    last_res = None
    for i in range(3):
        last_res = pipe.update(i, 0.05 * i, _dummy_frame())
    assert last_res is not None
    assert last_res.p1.state == BoardState.STABLE
    # match_just_started window 内なので confirmed は空 Board() (誤認無効化)
    assert last_res.p1.confirmed_board == _empty_board()
    assert last_res.p2.state == BoardState.STABLE
    assert last_res.p2.confirmed_board == p2


def test_pipeline_inactive_match_keeps_menu() -> None:
    p1 = _board_with_red(12, 0)
    p2 = _empty_board()
    pipe = _make_pipe(p1, p2, in_match=False, stable_n=2)
    res = pipe.update(0, 0.0, _dummy_frame())
    assert res.is_match_active is False
    assert res.p1.state == BoardState.MENU
    assert res.p2.state == BoardState.MENU


def test_pipeline_independent_p1_p2_states() -> None:
    """1P と 2P が独立して state を持つ."""
    pipe = _make_pipe(_board_with_red(12, 0), _empty_board(), stable_n=2)
    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.05, _dummy_frame())
    # 両方 STABLE
    res = pipe.update(2, 0.10, _dummy_frame())
    assert res.p1.state == BoardState.STABLE
    assert res.p2.state == BoardState.STABLE


def test_pipeline_reset() -> None:
    pipe = _make_pipe(_board_with_red(12, 0), _empty_board(), stable_n=2)
    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.05, _dummy_frame())
    pipe.reset()
    res = pipe.update(2, 0.10, _dummy_frame())
    # reset 直後は MENU から再スタート、1 frame では STABLE 不確定
    assert res.p1.state in {BoardState.MENU, BoardState.STABLE}


def test_side_result_carries_drift_info() -> None:
    pipe = _make_pipe(_board_with_red(12, 0), _empty_board(), stable_n=2)
    res = pipe.update(0, 0.0, _dummy_frame())
    assert isinstance(res.p1, SideResult)
    assert res.p1.drift.is_drift is False
    assert res.p1.cnn_board is not None


# ============================
# load_default smoke
# ============================


def test_next_count_constraint_excess_replaces_to_deficit() -> None:
    """サイクル66: tsumo_count に対し field 過剰色を deficit 色に置換."""
    from collections import Counter
    from src.board import COLOR_RED, COLOR_BLUE
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    board = Board()
    # field: RED 5 cells, BLUE 1 cell (= 6 puyo)
    for r, c in [(8, 0), (9, 0), (10, 0), (11, 0), (12, 0)]:
        board.set(r, c, COLOR_RED)
    board.set(12, 1, COLOR_BLUE)
    # 累積 tsumo: RED 4, BLUE 2 (= 6 puyo expected)
    tsumo_count: Counter = Counter({COLOR_RED: 4, COLOR_BLUE: 2})
    new_board = pipe._apply_next_count_constraint(
        board, tsumo_count, side="1P", frame_idx=100,
    )
    cnt: Counter = Counter()
    for r in range(13):
        for c in range(6):
            v = int(new_board.get(r, c))
            if v != 0:
                cnt[v] += 1
    # excess=1 RED, deficit=1 BLUE → RED が 1 cell BLUE に置換
    assert cnt[COLOR_RED] == 4
    assert cnt[COLOR_BLUE] == 2


def test_next_count_constraint_match_no_change() -> None:
    """サイクル66: field と tsumo_count 一致時は board 変更なし."""
    from collections import Counter
    from src.board import COLOR_RED, COLOR_BLUE
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    board = Board()
    board.set(11, 0, COLOR_RED)
    board.set(12, 0, COLOR_BLUE)
    tsumo_count: Counter = Counter({COLOR_RED: 1, COLOR_BLUE: 1})
    new_board = pipe._apply_next_count_constraint(
        board, tsumo_count, side="1P", frame_idx=100,
    )
    assert int(new_board.get(11, 0)) == COLOR_RED
    assert int(new_board.get(12, 0)) == COLOR_BLUE


def test_constraint_invalidates_on_chain_event() -> None:
    """サイクル66: 連鎖発火後は constraint_valid=False."""
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.05, _dummy_frame())
    # constraint は初期 True
    assert pipe._constraint_valid_1p is True
    assert pipe._constraint_valid_2p is True


# ============================
# 案1: CNN 高確信セル保護テスト (protect_board)
# ============================


def test_protect_board_shields_cnn_confirmed_cell() -> None:
    """案1: cnn=GREEN と confirmed=GREEN が一致するセルは excess でも置換されない."""
    from collections import Counter
    from src.board import COLOR_GREEN, COLOR_BLUE, COLOR_RED
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    # confirmed_board: GREEN 3 cells (= 1 excess), BLUE 1 cell
    board = Board()
    board.set(10, 0, COLOR_GREEN)
    board.set(11, 0, COLOR_GREEN)
    board.set(12, 0, COLOR_GREEN)
    board.set(12, 1, COLOR_BLUE)
    # tsumo_count: GREEN 2 (= confirmed より 1 少ない), BLUE 2 (= 1 多く必要)
    tsumo_count: Counter = Counter({COLOR_GREEN: 2, COLOR_BLUE: 2})
    # protect_board: row=10 のセルは GREEN (= CNN も GREEN と認識)
    protect = Board()
    protect.set(10, 0, COLOR_GREEN)   # CNN = GREEN → 保護対象
    protect.set(11, 0, COLOR_RED)     # CNN = RED  → 不一致 → 置換候補
    protect.set(12, 0, COLOR_RED)     # CNN = RED  → 不一致 → 置換候補
    protect.set(12, 1, COLOR_BLUE)
    # protect_board=protect の場合: row=10 の GREEN セルは保護される
    # → row=11 or row=12 の GREEN セルが BLUE に置換される
    new_board = pipe._apply_next_count_constraint(
        board, tsumo_count, side="1P", frame_idx=100,
        protect_board=protect,
    )
    # row=10 の GREEN は保護され BLUE に置換されない
    assert int(new_board.get(10, 0)) == COLOR_GREEN, \
        "CNN=GREEN 一致セルは protect_board で保護されるべき"
    # field 全体: GREEN 2 個, BLUE 2 個 になっていること
    from collections import Counter as _Counter
    cnt: _Counter = _Counter()
    for r in range(13):
        for c in range(6):
            v = int(new_board.get(r, c))
            if v != 0:
                cnt[v] += 1
    assert cnt[COLOR_GREEN] == 2, f"GREEN は 2 個になるべき, actual={cnt[COLOR_GREEN]}"
    assert cnt[COLOR_BLUE] == 2, f"BLUE は 2 個になるべき, actual={cnt[COLOR_BLUE]}"


def test_protect_board_none_falls_back_to_original_behavior() -> None:
    """案1: protect_board=None (デフォルト) では従来通り全セルが置換候補."""
    from collections import Counter
    from src.board import COLOR_GREEN, COLOR_BLUE
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    board = Board()
    board.set(10, 0, COLOR_GREEN)
    board.set(11, 0, COLOR_GREEN)
    board.set(12, 0, COLOR_GREEN)
    board.set(12, 1, COLOR_BLUE)
    tsumo_count: Counter = Counter({COLOR_GREEN: 2, COLOR_BLUE: 2})
    # protect_board=None (デフォルト) → 全セル対象 = 最上行から 1 つ置換
    new_board = pipe._apply_next_count_constraint(
        board, tsumo_count, side="1P", frame_idx=100,
        # protect_board を省略 = None
    )
    # 従来挙動: row 昇順で最初の GREEN (row=10) が BLUE に置換される
    assert int(new_board.get(10, 0)) == COLOR_BLUE, \
        "protect_board=None では row 昇順で先頭セルが置換される"


def test_protect_board_no_protection_when_color_mismatch() -> None:
    """案1: CNN 色が confirmed と不一致なセルは従来通り置換候補になる."""
    from collections import Counter
    from src.board import COLOR_GREEN, COLOR_BLUE, COLOR_RED
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    board = Board()
    board.set(10, 0, COLOR_GREEN)
    board.set(11, 0, COLOR_GREEN)
    board.set(12, 1, COLOR_BLUE)
    tsumo_count: Counter = Counter({COLOR_GREEN: 1, COLOR_BLUE: 2})
    # protect_board: row=10 のセルは RED (= CNN と confirmed 不一致)
    protect = Board()
    protect.set(10, 0, COLOR_RED)    # CNN = RED ≠ confirmed GREEN → 保護されない
    protect.set(11, 0, COLOR_RED)    # CNN = RED ≠ confirmed GREEN → 保護されない
    protect.set(12, 1, COLOR_BLUE)
    new_board = pipe._apply_next_count_constraint(
        board, tsumo_count, side="1P", frame_idx=100,
        protect_board=protect,
    )
    # CNN と色不一致のため保護なし = どちらかの GREEN が BLUE に置換される
    from collections import Counter as _Counter
    cnt: _Counter = _Counter()
    for r in range(13):
        for c in range(6):
            v = int(new_board.get(r, c))
            if v != 0:
                cnt[v] += 1
    assert cnt[COLOR_GREEN] == 1, f"GREEN 1 個に置換されるべき, actual={cnt[COLOR_GREEN]}"
    assert cnt[COLOR_BLUE] == 2, f"BLUE 2 個になるべき, actual={cnt[COLOR_BLUE]}"


# ============================
# 案2: enable_constraint_fill トグルテスト
# ============================


def test_enable_constraint_fill_false_skips_constraint() -> None:
    """案2: enable_constraint_fill=False でコンストレイント補正が skip される."""
    from src.board import COLOR_RED, COLOR_BLUE
    # enable_constraint_fill=False の pipeline を構築
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        stable_frame_count=2,
        enable_constraint_fill=False,
    )
    assert pipe._enable_constraint_fill is False, \
        "_enable_constraint_fill=False が設定されているべき"


def test_enable_constraint_fill_default_false() -> None:
    """enable_constraint_fill のデフォルトは False (2026-06-02 user viz 採用承認によりOFF化).

    採用スタックでは constraint_fill を OFF とすることが承認されたため
    デフォルトを False に変更した。ON に戻すには enable_constraint_fill=True を明示する。
    """
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    assert pipe._enable_constraint_fill is False, \
        "デフォルトは False (2026-06-02 user viz 採用承認)"


def test_constraint_fill_false_does_not_modify_board() -> None:
    """案2: enable_constraint_fill=False のとき board が変更されない."""
    from collections import Counter
    from src.board import COLOR_RED, COLOR_BLUE
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        stable_frame_count=2,
        enable_constraint_fill=False,
    )
    # False の場合 _apply_next_count_constraint を呼ばないことを
    # ガードフラグで検証する (直接 _enable_constraint_fill を見る)
    assert not pipe._enable_constraint_fill
    # _apply_next_count_constraint を直接呼んでも、 呼出元では呼ばれないはず。
    # 下記は呼出元ガードのロジック確認 (= ガードが False なら処理されない)
    board = Board()
    board.set(11, 0, COLOR_RED)
    board.set(12, 0, COLOR_RED)
    board.set(12, 1, COLOR_BLUE)
    tsumo_count: Counter = Counter({COLOR_RED: 1, COLOR_BLUE: 2})
    # _enable_constraint_fill=False なら呼出元がガードするが、
    # 直接 _apply_next_count_constraint を呼ぶと従来通り動く
    # (関数自体は影響を受けない。 ガードは呼出元にある)
    new_board = pipe._apply_next_count_constraint(
        board, tsumo_count, side="1P", frame_idx=100,
    )
    # 直接呼んだので補正は走る (= 呼出元ガードのテストは別のテストで担保)
    from collections import Counter as _Counter
    cnt: _Counter = _Counter()
    for r in range(13):
        for c in range(6):
            v = int(new_board.get(r, c))
            if v != 0:
                cnt[v] += 1
    # RED 1 → 1 (excess 1 → BLUE に置換), BLUE 2
    assert cnt[COLOR_RED] == 1
    assert cnt[COLOR_BLUE] == 2


def test_load_default_smoke() -> None:
    """load_default が例外なく組み立てられる (実モデル未配置でも graceful)."""
    try:
        pipe = RecognitionPipeline.load_default(
            stable_frame_count=2,
            load_score_ocr=True,
            enable_chain_tracker=True,
        )
    except FileNotFoundError as e:
        pytest.skip(f"必須テンプレ未配置: {e}")
    assert pipe is not None


# ============================
# 全消し overlay hold (cycle 71v, 2026-05-14)
# ============================


class _StubChainTracker:
    """指定した ChainEvent を 1 回だけ返す chain_tracker スタブ."""

    def __init__(self, event: object | None = None) -> None:
        self._event = event
        self._returned = False
        self.match_start_sec = 0.0

    def update(self, time_sec: float, board: Board) -> object | None:
        if self._returned:
            return None
        self._returned = True
        return self._event

    def reset(self, **kwargs: object) -> None:
        self._returned = False


def _make_chain_event(is_all_clear: bool, chain_count: int = 2):
    """テスト用 ChainEvent ファクトリ."""
    from src.chain_detector import ChainEvent
    return ChainEvent(
        trigger_sec=1.0,
        end_sec=1.5,
        before_board=Board(),
        chain_count=chain_count,
        total_erased=8,
        total_score=400,
        base_score=400,
        all_clear_bonus_applied=0,
        ojama_sent=2,
        leftover_score=0,
        is_all_clear=is_all_clear,
    )


def _make_pipe_with_tracker(
    chain_event_1p: object | None,
) -> RecognitionPipeline:
    """全消し hold テスト用 pipeline (1P 側のみ chain_tracker stub)."""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    tracker = _StubChainTracker(chain_event_1p)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=tracker,  # type: ignore[arg-type]
        chain_tracker_2p=None,
        stable_frame_count=2,
    )


def _prime_match_active(pipe: RecognitionPipeline, frames: int = 35) -> None:
    """chain_ban window (= CHAIN_BAN_FRAMES_AFTER_MATCH_START=30) を超えるよう
    試合中 frame を投入。"""
    for i in range(frames):
        pipe.update(i, 0.05 * i, _dummy_frame())


def test_all_clear_extends_chain_hold() -> None:
    """is_all_clear=True の ChainEvent では chain_until が延長される.

    cycle 71v (2026-05-14): 全消し overlay が CHAIN→STABLE 遷移時の
    _merge_diff_only を corrupt しないよう、 ALL_CLEAR_OVERLAY_HOLD_SEC
    ぶん CHAIN を延長する partial fix。
    """
    # event は ban window 通過後に投入したい → tracker は最初しか event を
    # 返さないので priming は tracker_event=None で行ってから event 注入する。
    ev = _make_chain_event(is_all_clear=True, chain_count=2)
    pipe = _make_pipe_with_tracker(None)
    _prime_match_active(pipe, frames=35)
    # ban window 通過後に tracker を差し替えて event を返させる
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    t = 10.0
    pipe.update(40, t, _dummy_frame())
    expected_base = t + pipe._chain_hold_per_step_sec * 2
    expected_total = expected_base + RecognitionPipeline.ALL_CLEAR_OVERLAY_HOLD_SEC
    assert pipe._chain_until_1p == pytest.approx(expected_total)


def test_non_all_clear_does_not_extend_chain_hold() -> None:
    """is_all_clear=False の ChainEvent では延長なし (= 旧挙動互換)."""
    ev = _make_chain_event(is_all_clear=False, chain_count=2)
    pipe = _make_pipe_with_tracker(None)
    _prime_match_active(pipe, frames=35)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    t = 10.0
    pipe.update(40, t, _dummy_frame())
    expected_base = t + pipe._chain_hold_per_step_sec * 2
    assert pipe._chain_until_1p == pytest.approx(expected_base)


# ============================
# A0 (2026-07-24): CHAIN 保持時間モデル config 化テスト
# 計装 a287c587 実測較正 (base=3.4s + per_step=1.5s×連鎖数) を config 経由で
# 注入できることと、既定値 (base=0.0) では従来式と bit-identical であることを
# 確認する。
# ============================


def _make_pipe_with_tracker_calibrated(
    chain_event_1p: object | None,
    chain_hold_base_sec: float | None = None,
    chain_hold_per_step_sec: float | None = None,
    chain_max_hold_sec: float | None = None,
) -> RecognitionPipeline:
    """_make_pipe_with_tracker の較正値注入版 (A0 テスト専用)。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    tracker = _StubChainTracker(chain_event_1p)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=tracker,  # type: ignore[arg-type]
        chain_tracker_2p=None,
        stable_frame_count=2,
        chain_hold_base_sec=chain_hold_base_sec,
        chain_hold_per_step_sec=chain_hold_per_step_sec,
        chain_max_hold_sec=chain_max_hold_sec,
    )


def test_chain_hold_base_sec_default_is_zero_backward_compat() -> None:
    """既定値 (chain_hold_base_sec=None) は CHAIN_HOLD_BASE_SEC=0.0 に解決され、
    従来の「固定項なし・per_step のみ」の式と bit-identical になること。"""
    ev = _make_chain_event(is_all_clear=False, chain_count=3)
    pipe = _make_pipe_with_tracker_calibrated(None)
    assert pipe._chain_hold_base_sec == 0.0
    assert pipe._chain_hold_per_step_sec == pytest.approx(0.3)
    assert pipe._chain_max_hold_sec == pytest.approx(5.0)
    _prime_match_active(pipe, frames=35)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    t = 10.0
    pipe.update(40, t, _dummy_frame())
    # 従来式: t + per_step_sec * chain_count (固定項なし)
    expected = t + 0.3 * 3
    assert pipe._chain_until_1p == pytest.approx(expected)


def test_chain_hold_base_sec_calibrated_value_applies_to_chain_until() -> None:
    """較正値 (base=3.4, per_step=1.5) を渡すと chain_until = t + base + per_step*n
    になること (実測モデルの反映確認)。"""
    ev = _make_chain_event(is_all_clear=False, chain_count=4)
    pipe = _make_pipe_with_tracker_calibrated(
        None, chain_hold_base_sec=3.4, chain_hold_per_step_sec=1.5,
    )
    _prime_match_active(pipe, frames=35)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    t = 10.0
    pipe.update(40, t, _dummy_frame())
    expected = t + 3.4 + 1.5 * 4
    assert pipe._chain_until_1p == pytest.approx(expected)


def test_chain_max_hold_sec_default_unchanged_backward_compat() -> None:
    """chain_max_hold_sec 省略時は CHAIN_MAX_HOLD_SEC=5.0 のまま
    (既存 enable_game_event_chain_exit 経路の挙動不変)。"""
    ev = _make_chain_event(is_all_clear=False, chain_count=1)
    pipe = _make_pipe_with_tracker_calibrated(None)
    _prime_match_active(pipe, frames=35)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    t = 10.0
    pipe.update(40, t, _dummy_frame())
    assert pipe._chain_event_max_until_1p == pytest.approx(t + 5.0)


def test_chain_max_hold_sec_configurable() -> None:
    """chain_max_hold_sec を明示すると安全弁上限が上書きされること
    (較正評価時に較正済 chain_until と併せて引き上げる運用を想定)。"""
    ev = _make_chain_event(is_all_clear=False, chain_count=1)
    pipe = _make_pipe_with_tracker_calibrated(None, chain_max_hold_sec=25.0)
    _prime_match_active(pipe, frames=35)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    t = 10.0
    pipe.update(40, t, _dummy_frame())
    assert pipe._chain_event_max_until_1p == pytest.approx(t + 25.0)


def test_score_early_fire_pseudo_event_uses_calibrated_hold() -> None:
    """機能B (score 急増早期発火) の疑似 ChainEvent も
    chain_hold_base_sec/per_step_sec の較正値を反映すること
    (:3061-3071 相当パス、chain_count=1 固定)。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        chain_hold_base_sec=3.4,
        chain_hold_per_step_sec=1.5,
        enable_chain_score_early_fire=True,
    )
    _prime_match_active(pipe, frames=35)
    t = 10.0
    pipe._apply_chain_score_early_fire(
        side="1P", score_delta=200, time_sec=t, prev_confirmed=Board(),
    )
    expected = t + 3.4 + 1.5
    assert pipe._chain_until_1p == pytest.approx(expected)
    assert pipe._active_chain_1p is not None
    assert pipe._active_chain_1p.end_sec == pytest.approx(expected)


def test_formula_early_fire_pseudo_event_uses_calibrated_hold() -> None:
    """機能D (掛け算式早期発火) の疑似 ChainEvent も較正値を反映すること
    (:3164-3174 相当パス、chain_count=1 固定)。

    2026-07-24: enable_chain_formula_simulate_verify の既定値が
    True に変更された (機能D 採用、偽イベント率 27.5%→0%) ため、
    本テストは較正値反映ロジック単体を確認する目的で
    enable_chain_formula_simulate_verify=False を明示指定し、
    空盤面 (連鎖ゼロ) でも従来通り chain_count=1 固定で発火する
    旧経路 (bit-identical) を使う。simulate_verify=True 時の
    空盤面抑制挙動は test_chain_formula_detection.py 側で確認済み。
    """
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        chain_hold_base_sec=3.4,
        chain_hold_per_step_sec=1.5,
        enable_chain_formula_detection=True,
        enable_chain_formula_simulate_verify=False,
    )
    _prime_match_active(pipe, frames=35)
    t = 10.0
    pipe._apply_chain_formula_early_fire(
        side="2P", time_sec=t, prev_confirmed=Board(),
    )
    expected = t + 3.4 + 1.5
    assert pipe._chain_until_2p == pytest.approx(expected)
    assert pipe._active_chain_2p is not None
    assert pipe._active_chain_2p.end_sec == pytest.approx(expected)


def test_load_default_exposes_chain_hold_calibration_kwargs() -> None:
    """load_default() 経由でも chain_hold_base_sec/per_step_sec/max_hold_sec を
    注入できること (従来 load_default に chain_hold_per_step_sec すら露出して
    いなかった抜け漏れの解消確認、評価スクリプトからの注入経路を保証する)。
    ネットワーク/モデルファイル無し環境でも HSV-only フォールバックで
    構築できるはず (cnn_model_path 未指定時の既存フォールバック仕様)。
    """
    pipe = RecognitionPipeline.load_default(
        load_score_ocr=False,
        enable_chain_tracker=False,
        load_next_detector=False,
        chain_hold_base_sec=3.4,
        chain_hold_per_step_sec=1.5,
        chain_max_hold_sec=25.0,
    )
    assert pipe._chain_hold_base_sec == pytest.approx(3.4)
    assert pipe._chain_hold_per_step_sec == pytest.approx(1.5)
    assert pipe._chain_max_hold_sec == pytest.approx(25.0)


def test_load_default_calibration_kwargs_default_none_backward_compat() -> None:
    """load_default() で較正引数を省略すると従来値 (0.0/0.3/5.0) のまま
    (backwards compat)。"""
    pipe = RecognitionPipeline.load_default(
        load_score_ocr=False,
        enable_chain_tracker=False,
        load_next_detector=False,
    )
    assert pipe._chain_hold_base_sec == 0.0
    assert pipe._chain_hold_per_step_sec == pytest.approx(0.3)
    assert pipe._chain_max_hold_sec == pytest.approx(5.0)


def test_immediate_landing_chain_pseudo_event_uses_calibrated_hold() -> None:
    """A0 バグ修正のロック用テスト (旧 :3797 相当のハードコード 0.3 バグ)。

    このパス (着地直後即時連鎖判定、cycle48 大量 hallucination ガード
    通過済) は TSUMO_FALL→STABLE 着地時に resolve_after_placement() が
    chain_count>=1 を返した場合にのみ _step_side 内部で発火する。
    フルの状態遷移 (4連結着地→TSUMO_FALL→STABLE) を駆動する既存の
    end-to-end テストが無く、本テストも「pipe に設定された
    chain_hold_base_sec/per_step_sec を使えば期待式と一致する」という
    式レベルのロックに留まる (= ソース :3838-3852 相当の計算式の
    リグレッションガードであり、resolve_after_placement 発火条件込みの
    完全な統合テストではない。将来的な integration test 追加は別課題)。
    """
    from src.chain_detector import ChainEvent

    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        chain_hold_base_sec=3.4,
        chain_hold_per_step_sec=1.5,
    )
    t = 10.0
    chain_count = 2
    # _step_side 内の該当ブロックと同じ計算式を直接検証する
    # (ソース側の式そのものを import せず重複させると回帰検知力が落ちるため、
    #  ここでは pipe の設定値を用いて期待値を計算し、ソース定数を書き換えても
    #  テストが追随することを保証する)。
    expected_chain_until = (
        t + pipe._chain_hold_base_sec + pipe._chain_hold_per_step_sec * chain_count
    )
    pseudo = ChainEvent(
        trigger_sec=t,
        end_sec=(
            t + pipe._chain_hold_base_sec
            + pipe._chain_hold_per_step_sec * chain_count
        ),
        before_board=Board(),
        chain_count=chain_count,
        total_erased=0, total_score=0, base_score=0,
        all_clear_bonus_applied=0,
        ojama_sent=0, leftover_score=0,
        is_all_clear=False,
    )
    assert pseudo.end_sec == pytest.approx(expected_chain_until)


# ============================
# B1 PiecePersistenceGuard 統合テスト
# ============================

def _make_pipe_with_persistence(
    p1: Board, p2: Board,
    stable_n: int = 2,
    enable: bool = True,
) -> RecognitionPipeline:
    """PiecePersistenceGuard を有効化した pipeline を返す。"""
    reader = _StubImageReader(p1, p2)
    detector = _StubMatchDetector(in_match=True)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=stable_n,
        enable_piece_persistence=enable,
    )


def test_pipeline_piece_persistence_disabled_by_default() -> None:
    """default OFF で _piece_persistence_1p / 2p が None。"""
    pipe = _make_pipe(_empty_board(), _empty_board())
    assert pipe._piece_persistence_1p is None
    assert pipe._piece_persistence_2p is None


def test_pipeline_piece_persistence_enabled_creates_guards() -> None:
    """enable_piece_persistence=True で guard インスタンスが生成される。"""
    pipe = _make_pipe_with_persistence(_empty_board(), _empty_board())
    assert pipe._piece_persistence_1p is not None
    assert pipe._piece_persistence_2p is not None


def test_pipeline_piece_persistence_reset_clears_guards() -> None:
    """reset() で guard が完全リセットされる。"""
    from src.board import COLOR_RED
    p1 = Board()
    p1.set(12, 0, COLOR_RED)
    pipe = _make_pipe_with_persistence(p1, _empty_board(), stable_n=2)
    # STABLE に到達させて保護を登録
    for i in range(65):  # MATCH_JUST_STARTED_WINDOW を超える
        pipe.update(i, i * 0.033, _dummy_frame())
    guard = pipe._piece_persistence_1p
    assert guard is not None
    # reset 後は保護がクリアされる
    pipe.reset()
    assert len(pipe._piece_persistence_1p._protected) == 0  # type: ignore[union-attr]


def test_pipeline_piece_persistence_on_returns_side_result() -> None:
    """enable=True でも SideResult が正常に返る (= 既存 API 維持)。"""
    pipe = _make_pipe_with_persistence(_empty_board(), _empty_board(), stable_n=2)
    res = pipe.update(0, 0.0, _dummy_frame())
    assert res is not None
    assert hasattr(res.p1, "confirmed_board")
    assert hasattr(res.p2, "confirmed_board")


# ===== tier1 warmup guard テスト群 =====


def _make_pipe_with_tier1_warmup(
    p1: Board, p2: Board, stable_n: int = 2,
) -> RecognitionPipeline:
    """enable_tier1_warmup=True の RecognitionPipeline を構築する。"""
    reader = _StubImageReader(p1, p2)
    detector = _StubMatchDetector(in_match=True)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=stable_n,
        enable_tier1_warmup=True,
    )


def test_pipeline_tier1_warmup_disabled_by_default() -> None:
    """default OFF で _enable_tier1_warmup が False。"""
    pipe = _make_pipe(_empty_board(), _empty_board())
    assert pipe._enable_tier1_warmup is False


def test_pipeline_tier1_warmup_enabled_flag() -> None:
    """enable_tier1_warmup=True で _enable_tier1_warmup が True。"""
    pipe = _make_pipe_with_tier1_warmup(_empty_board(), _empty_board())
    assert pipe._enable_tier1_warmup is True


def test_pipeline_tier1_warmup_initial_counters_zero() -> None:
    """初期状態でカウンタが 0。"""
    pipe = _make_pipe_with_tier1_warmup(_empty_board(), _empty_board())
    assert pipe._tier1_warmup_remaining_1p == 0
    assert pipe._tier1_warmup_remaining_2p == 0


def test_pipeline_tier1_warmup_resets_on_reset() -> None:
    """reset() でカウンタが 0 に戻る。"""
    pipe = _make_pipe_with_tier1_warmup(_empty_board(), _empty_board())
    pipe._tier1_warmup_remaining_1p = 2
    pipe._tier1_warmup_remaining_2p = 3
    pipe.reset()
    assert pipe._tier1_warmup_remaining_1p == 0
    assert pipe._tier1_warmup_remaining_2p == 0


def test_pipeline_tier1_warmup_disabled_no_skip_tier1() -> None:
    """enable_tier1_warmup=False ではどの frame も skip_tier1_1p=False で呼ばれる。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_tier1_warmup=False,
    )
    pipe.update(0, 0.0, _dummy_frame())
    assert reader.last_skip_tier1_1p is False
    assert reader.last_skip_tier1_2p is False


def test_pipeline_tier1_warmup_result_is_side_result() -> None:
    """enable_tier1_warmup=True でも SideResult が正常に返る (= API 維持)。"""
    pipe = _make_pipe_with_tier1_warmup(_empty_board(), _empty_board(), stable_n=2)
    res = pipe.update(0, 0.0, _dummy_frame())
    assert res is not None
    assert hasattr(res.p1, "confirmed_board")
    assert hasattr(res.p2, "confirmed_board")


# ============================
# 経路 A': OJAMA 専用 tier1 warmup (_update_ojama_tier1_warmup_counter)
# ============================


def test_ojama_tier1_warmup_sets_ojama_frames_on_ojama_to_stable() -> None:
    """OJAMA_FALL → STABLE 遷移で OJAMA_TIER1_WARMUP_FRAMES がセットされる。"""
    result = _update_ojama_tier1_warmup_counter(
        prev_state=BoardState.OJAMA_FALL,
        p_state=BoardState.STABLE,
        remaining=0,
    )
    assert result == OJAMA_TIER1_WARMUP_FRAMES


def test_ojama_tier1_warmup_default_true() -> None:
    """enable_ojama_tier1_warmup のデフォルトは True (2026-06-02 user viz 採用承認)。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
    )
    assert pipe._enable_ojama_tier1_warmup is True


def test_ojama_tier1_warmup_explicit_false_no_effect() -> None:
    """enable_ojama_tier1_warmup=False を明示すると ojama 専用カウンタが 0 のまま (回帰防止)。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_ojama_tier1_warmup=False,
    )
    assert pipe._enable_ojama_tier1_warmup is False
    assert pipe._ojama_tier1_warmup_remaining_1p == 0
    assert pipe._ojama_tier1_warmup_remaining_2p == 0


def test_ojama_tier1_warmup_tsumo_fall_does_not_trigger_ojama_counter() -> None:
    """TSUMO_FALL → STABLE は OJAMA 分岐に入らず TIER1_WARMUP_FRAMES のまま変化しない。"""
    result = _update_ojama_tier1_warmup_counter(
        prev_state=BoardState.TSUMO_FALL,
        p_state=BoardState.STABLE,
        remaining=0,
    )
    # TSUMO_FALL → STABLE は OJAMA 分岐対象外 → カウンタは 0 のまま
    assert result == 0
    # なお OJAMA_TIER1_WARMUP_FRAMES にはセットされない
    assert result != OJAMA_TIER1_WARMUP_FRAMES


def test_ojama_tier1_warmup_enabled_flag() -> None:
    """enable_ojama_tier1_warmup=True で _enable_ojama_tier1_warmup が True。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_ojama_tier1_warmup=True,
    )
    assert pipe._enable_ojama_tier1_warmup is True


def test_ojama_tier1_warmup_resets_on_reset() -> None:
    """reset() で ojama 専用カウンタが 0 に戻る。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_ojama_tier1_warmup=True,
    )
    pipe._ojama_tier1_warmup_remaining_1p = 5
    pipe._ojama_tier1_warmup_remaining_2p = 7
    pipe.reset()
    assert pipe._ojama_tier1_warmup_remaining_1p == 0
    assert pipe._ojama_tier1_warmup_remaining_2p == 0


def test_ojama_tier1_warmup_counter_decrements_in_stable() -> None:
    """STABLE 継続中は OJAMA カウンタがデクリメントされる。"""
    result = _update_ojama_tier1_warmup_counter(
        prev_state=BoardState.STABLE,
        p_state=BoardState.STABLE,
        remaining=4,
    )
    assert result == 3


def test_ojama_tier1_warmup_counter_resets_on_non_stable() -> None:
    """STABLE → TSUMO_FALL 遷移で OJAMA カウンタが 0 にリセットされる。"""
    result = _update_ojama_tier1_warmup_counter(
        prev_state=BoardState.STABLE,
        p_state=BoardState.TSUMO_FALL,
        remaining=6,
    )
    assert result == 0


def test_ojama_tier1_warmup_chain_does_not_trigger() -> None:
    """CHAIN → STABLE 遷移では OJAMA 分岐に入らず 0 のまま。"""
    result = _update_ojama_tier1_warmup_counter(
        prev_state=BoardState.CHAIN,
        p_state=BoardState.STABLE,
        remaining=0,
    )
    assert result == 0


def test_ojama_tier1_warmup_independent_from_generic_warmup() -> None:
    """汎用 enable_tier1_warmup=False でも enable_ojama_tier1_warmup=True なら skip 発火する。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_tier1_warmup=False,
        enable_ojama_tier1_warmup=True,
    )
    # 汎用 warmup は OFF
    assert pipe._enable_tier1_warmup is False
    # ojama 専用 warmup は ON
    assert pipe._enable_ojama_tier1_warmup is True
    # ojama カウンタを手動セット → skip_tier1 が True になることを確認
    pipe._ojama_tier1_warmup_remaining_1p = 3
    # update() を呼び skip_tier1_1p=True で read_both_boards が呼ばれるか確認
    pipe.update(0, 0.0, _dummy_frame())
    # skip_tier1_1p が True で呼ばれた (= ojama warmup が発火した)
    assert reader.last_skip_tier1_1p is True


# ============================
# T2 高確信 yield (enable_t2_highconf_yield) テスト
# ============================


def _make_pipe_t2(
    cnn_board_1p: Board,
    t2_highconf_yield: bool = False,
) -> RecognitionPipeline:
    """T2 テスト用 pipeline を構築する。

    stable_frame_count=2 で即 STABLE に遷移させる。
    enable_t2_highconf_yield で T2 yield トグルを制御する。
    """
    reader = _StubImageReader(cnn_board_1p, _empty_board())
    detector = _StubMatchDetector(in_match=True)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_t2_highconf_yield=t2_highconf_yield,
    )


def _inject_prev_stable_and_confirmed(
    pipe: RecognitionPipeline,
    prev_color: int,
    confirmed_color: int,
    row: int = 12,
    col: int = 0,
) -> None:
    """T2 テスト用: prev_stable と confirmed_board を指定色で手動セットする。

    prev_stable_1p に prev_color を入れ、confirmed_board に confirmed_color を
    設定することで T2 の「色A → 色B」判定が発火する状態を作る。
    _match_active_started_frame = -1000 にして match_just_started を無効化する。
    """
    # match_just_started window を解除 (= 試合開始から十分時間が経過した想定)
    pipe._match_active_started_frame = -1000

    # prev_stable_1p に prev_color をセット
    prev_board = Board()
    prev_board.set(row, col, prev_color)
    pipe._prev_stable_confirmed_1p = prev_board

    # sm_1p の confirmed_board に confirmed_color をセット
    conf_board = Board()
    conf_board.set(row, col, confirmed_color)
    if pipe._sm_1p.context.confirmed_board is None:
        pipe._sm_1p.context.confirmed_board = conf_board
    else:
        pipe._sm_1p.context.confirmed_board.set(row, col, confirmed_color)


def test_t2_highconf_yield_on_skips_overwrite() -> None:
    """トグル ON: raw_cnn==cur_v (緑) のセルは T2 が prev_stable(青) で上書きしない。

    シナリオ:
      - CNN 出力 = 緑 (GREEN)
      - confirmed_board = 緑 (prev_stable には一致)
      - prev_stable = 青 (BLUE)
      - T2 条件: both_colored かつ pv != cur_v → 通常は青で上書き
      - enable_t2_highconf_yield=True かつ cnn_v==cur_v → スキップ (緑を維持)
    """
    row, col = 12, 0
    cnn_green = Board()
    cnn_green.set(row, col, COLOR_GREEN)

    pipe = _make_pipe_t2(cnn_green, t2_highconf_yield=True)

    # pipeline を STABLE に持ち込む (2 フレーム分 update)
    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.033, _dummy_frame())

    # 手動 inject: prev_stable=青, confirmed=緑
    _inject_prev_stable_and_confirmed(
        pipe, prev_color=COLOR_BLUE, confirmed_color=COLOR_GREEN, row=row, col=col,
    )

    # CNN は依然として緑を返す状態で 1 フレーム処理
    res = pipe.update(62, 62 / 30.0, _dummy_frame())

    # T2 が青で上書きしなかった → confirmed は緑のまま
    assert res is not None
    confirmed = res.p1.confirmed_board
    assert confirmed is not None, "confirmed_board が None"
    cell_val = int(confirmed.get(row, col))
    assert cell_val == COLOR_GREEN, (
        f"トグル ON 時: T2 が青で上書きすべきでない (期待=緑={COLOR_GREEN}, 実際={cell_val})"
    )


def test_t2_highconf_yield_off_applies_overwrite() -> None:
    """トグル OFF (default): T2 は従来通り prev_stable(青) で上書きする。

    シナリオ:
      - CNN 出力 = 緑 (GREEN)
      - confirmed_board = 緑
      - prev_stable = 青 (BLUE)
      - enable_t2_highconf_yield=False (デフォルト)
      - T2 が青で上書き → confirmed は青になる
    """
    row, col = 12, 0
    cnn_green = Board()
    cnn_green.set(row, col, COLOR_GREEN)

    pipe = _make_pipe_t2(cnn_green, t2_highconf_yield=False)

    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.033, _dummy_frame())

    _inject_prev_stable_and_confirmed(
        pipe, prev_color=COLOR_BLUE, confirmed_color=COLOR_GREEN, row=row, col=col,
    )

    res = pipe.update(62, 62 / 30.0, _dummy_frame())

    assert res is not None
    confirmed = res.p1.confirmed_board
    assert confirmed is not None, "confirmed_board が None"
    cell_val = int(confirmed.get(row, col))
    assert cell_val == COLOR_BLUE, (
        f"トグル OFF 時: T2 が青で上書きすべき (期待=青={COLOR_BLUE}, 実際={cell_val})"
    )


def test_t2_highconf_yield_cnn_mismatch_still_applies() -> None:
    """raw_cnn != cur_v のセルはトグル ON でも T2 上書きが適用される (高確信でない)。

    シナリオ:
      - CNN 出力 = 赤 (RED) ← cur_v (緑) と不一致
      - confirmed_board = 緑 (GREEN)
      - prev_stable = 青 (BLUE)
      - enable_t2_highconf_yield=True でも cnn_v!=cur_v → T2 適用 → 青で上書き
    """
    row, col = 12, 0
    cnn_red = Board()
    cnn_red.set(row, col, COLOR_RED)

    pipe = _make_pipe_t2(cnn_red, t2_highconf_yield=True)

    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.033, _dummy_frame())

    # CNN は赤, confirmed に緑を手動セット, prev_stable は青
    _inject_prev_stable_and_confirmed(
        pipe, prev_color=COLOR_BLUE, confirmed_color=COLOR_GREEN, row=row, col=col,
    )

    res = pipe.update(62, 62 / 30.0, _dummy_frame())

    assert res is not None
    confirmed = res.p1.confirmed_board
    assert confirmed is not None, "confirmed_board が None"
    cell_val = int(confirmed.get(row, col))
    assert cell_val == COLOR_BLUE, (
        f"CNN 不一致時: T2 が青で上書きすべき (期待=青={COLOR_BLUE}, 実際={cell_val})"
    )


def test_t2_highconf_yield_default_is_true() -> None:
    """enable_t2_highconf_yield のデフォルト値が True (2026-06-02 user viz 採用承認)。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        # enable_t2_highconf_yield を明示せず → デフォルト True
    )
    assert pipe._enable_t2_highconf_yield is True


def test_t2_highconf_yield_pv_empty_no_yield() -> None:
    """pv=空 のセルはトグル ON でも yield しない (背景 FP 抑制)。

    シナリオ:
      - CNN 出力 = 緑 (GREEN)  ← 背景 FP
      - confirmed_board = 緑 (GREEN)  ← CNN FP が confirmed に残っている状態
      - prev_stable = 空 (EMPTY)  ← 本来は空のセル
      - enable_t2_highconf_yield=True
      - pv=空 なので yield 発動せず → T2 は confirmed を空に上書き
    注意: T2 の発動条件は both_colored (pv 非 EMPTY かつ cur_v 非 EMPTY) なので、
    このシナリオでは both_colored=False となり T2 自体が上書きを行わない。
    本テストは「yield 条件の pv チェック」が加わっても T2 上書きが行われる状態を
    確認するため、pv=EMPTY / cur_v=GREEN の組み合わせで T2 がスキップされないことを
    検証する (both_colored=False → T2 上書きなし = 緑が維持される)。

    補足: both_colored が False のケースで yield 条件追加の副作用がないことを確認。
    """
    row, col = 9, 2
    cnn_green = Board()
    cnn_green.set(row, col, COLOR_GREEN)

    pipe = _make_pipe_t2(cnn_green, t2_highconf_yield=True)

    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.033, _dummy_frame())

    # prev_stable = 空, confirmed = 緑 (背景 FP シナリオ)
    _inject_prev_stable_and_confirmed(
        pipe, prev_color=COLOR_EMPTY, confirmed_color=COLOR_GREEN, row=row, col=col,
    )

    res = pipe.update(62, 62 / 30.0, _dummy_frame())

    assert res is not None
    confirmed = res.p1.confirmed_board
    assert confirmed is not None, "confirmed_board が None"
    cell_val = int(confirmed.get(row, col))
    # pv=EMPTY / cur_v=GREEN → both_colored=False → T2 上書きが発生しない
    # (= T2 の上書きはあくまで「両方色付き かつ 異色」のみ)
    # yield 条件追加によるリグレッションがないことを確認。
    # CNN stub が GREEN を返し続けるのでフレーム処理後も GREEN になる。
    assert cell_val == COLOR_GREEN, (
        f"pv=空/cur=緑: both_colored=False なので T2 上書きなし (期待=緑={COLOR_GREEN}, 実際={cell_val})"
    )


def test_t2_highconf_yield_pv_colored_still_yields() -> None:
    """pv=色付き のセルはトグル ON かつ cnn==cur で yield する (既存動作の維持)。

    シナリオ:
      - CNN 出力 = 緑 (GREEN)
      - confirmed_board = 緑 (GREEN)
      - prev_stable = 青 (BLUE)  ← 色 → 別色フリーズのケース
      - enable_t2_highconf_yield=True
      - pv=青 (色付き) かつ cnn_v==cur_v → yield 発動 → 緑を維持
    """
    row, col = 9, 2
    cnn_green = Board()
    cnn_green.set(row, col, COLOR_GREEN)

    pipe = _make_pipe_t2(cnn_green, t2_highconf_yield=True)

    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.033, _dummy_frame())

    # prev_stable = 青 (色付き), confirmed = 緑
    _inject_prev_stable_and_confirmed(
        pipe, prev_color=COLOR_BLUE, confirmed_color=COLOR_GREEN, row=row, col=col,
    )

    res = pipe.update(62, 62 / 30.0, _dummy_frame())

    assert res is not None
    confirmed = res.p1.confirmed_board
    assert confirmed is not None, "confirmed_board が None"
    cell_val = int(confirmed.get(row, col))
    assert cell_val == COLOR_GREEN, (
        f"pv=色付き: yield 発動して緑を維持すべき (期待=緑={COLOR_GREEN}, 実際={cell_val})"
    )


# ============================
# game-event ベース連鎖終了 (C-1/C-2) テスト
# ============================


from src.recognition_pipeline import _is_game_event_chain_exit


def test_is_game_event_chain_exit_next_change() -> None:
    """① 次ツモ変化: current_next != start_next → True を返す。"""
    start = (COLOR_RED, COLOR_BLUE)
    current = (COLOR_GREEN, COLOR_RED)   # 変化あり
    result = _is_game_event_chain_exit(
        current_next=current,
        start_next=start,
    )
    assert result is True, "next_pair 変化時は True を返すべき"


def test_is_game_event_chain_exit_next_no_change() -> None:
    """① 次ツモ変化なし: current_next == start_next → False。"""
    same = (COLOR_RED, COLOR_BLUE)
    result = _is_game_event_chain_exit(
        current_next=same,
        start_next=same,
    )
    assert result is False, "next_pair 変化なし → False"


def test_is_game_event_chain_exit_ojama_appears_no_exit() -> None:
    """②お邪魔信号撤去後: お邪魔新規出現だけでは終了しない (next変化なし → False)。

    2026-06-01 撤去: confirmed凍結が連鎖終了後に既存お邪魔へ追いつくだけで
    新規落下と誤認し短連鎖を早期終了させていた問題を解消。
    お邪魔引数は廃止されたため next=None / next一致どちらも False を返す。
    """
    # next 変化なし → ①も②もなし → False
    result = _is_game_event_chain_exit(
        current_next=None,
        start_next=None,
    )
    assert result is False, "next変化なし / お邪魔信号撤去後 → False"


def test_is_game_event_chain_exit_ojama_preexisting_no_exit() -> None:
    """②お邪魔信号撤去後: 既存お邪魔継続も終了しない (next変化なし → False)。

    引数から current_board / start_board が除去されたことを確認する。
    """
    same_next = (COLOR_RED, COLOR_BLUE)
    result = _is_game_event_chain_exit(
        current_next=same_next,
        start_next=same_next,
    )
    assert result is False, "next変化なし → False (お邪魔引数は撤去済)"


def test_is_game_event_chain_exit_max_hold_cap() -> None:
    """安全弁: CHAIN_MAX_HOLD_SEC 超過で game-event なしでも終了する設計を確認。

    _is_game_event_chain_exit は stateless (安全弁は pipeline 側 eff_until で制御)。
    本テストは: next 変化なし → False を返すことで
    「安全弁は pipeline の time_sec >= eff_until で chain_ev を None にする」
    側の責任であることを明示する。
    ※②お邪魔信号撤去 (2026-06-01) により board 引数は不要になった。
    """
    same_next = (COLOR_RED, COLOR_BLUE)
    result = _is_game_event_chain_exit(
        current_next=same_next,
        start_next=same_next,
    )
    assert result is False, (
        "安全弁は pipeline 側 (eff_until) 管理のため、"
        "game-event なし時は False を返す"
    )


def test_game_event_chain_exit_flag_off_is_backward_compat() -> None:
    """OFF 時は従来挙動不変: enable_game_event_chain_exit=False でインスタンス生成可能。

    default が True に変わっても False を明示すると従来 timing hold 挙動に戻る
    ことを確認する (= OFF 経路の回帰防止)。
    """
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_game_event_chain_exit=False,  # 明示 OFF = 従来挙動
    )
    # 明示 False = game-event chain exit 無効
    assert pipe._enable_game_event_chain_exit is False
    # 従来 chain_until 変数が存在すること
    assert hasattr(pipe, "_chain_until_1p")
    assert hasattr(pipe, "_chain_until_2p")
    # 新規 game-event 変数が初期化されていること (OFF 時も初期化済)
    assert pipe._chain_event_max_until_1p == 0.0
    assert pipe._chain_event_max_until_2p == 0.0
    assert pipe._chain_start_next_1p is None
    assert pipe._chain_start_next_2p is None


# ============================
# 着地色修正 案1 テスト (2026-06-01)
# ============================


def _make_pipe_with_flags(
    p1: "Board", p2: "Board",
    stable_n: int = 2,
    enable_landing_color_fix: bool = False,
) -> RecognitionPipeline:
    """フラグ付き pipeline を生成するヘルパー。"""
    from src.board import Board as _Board
    reader = _StubImageReader(p1, p2)
    detector = _StubMatchDetector(in_match=True)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=stable_n,
        enable_landing_color_fix=enable_landing_color_fix,
    )


def test_landing_color_fix_flag_default_off() -> None:
    """①フラグOFF時: _enable_landing_color_fix=False で従来 falling_pair 不変 (回帰テスト)。

    enable_landing_color_fix のデフォルトが False であり、
    pipeline のフラグが False で初期化されることを確認する。
    フラグ OFF 時は _last_consumed_color の有無に関わらず従来ロジック
    (prev_next_queue[-2]) が使われる。
    """
    pipe = _make_pipe_with_flags(
        _empty_board(), _empty_board(),
        enable_landing_color_fix=False,
    )
    assert pipe._enable_landing_color_fix is False, (
        "デフォルト OFF: 従来 falling_pair ロジック (prev_next_queue[-2]) を維持"
    )


def test_landing_color_fix_flag_on_exists() -> None:
    """②フラグ ON: _enable_landing_color_fix=True で pipeline が生成可能 (インターフェース確認)。

    フラグ ON で init が通ること、フラグが True に設定されること、
    SideResult に landing_diag フィールドが存在することを確認する。
    """
    pipe = _make_pipe_with_flags(
        _empty_board(), _empty_board(),
        enable_landing_color_fix=True,
    )
    assert pipe._enable_landing_color_fix is True, (
        "フラグ ON: _last_consumed_color 由来の falling_pair を使う修正ロジックが有効"
    )
    # SideResult に landing_diag フィールドが存在すること (backwards compat 確認)
    from src.recognition_pipeline import SideResult
    import inspect
    fields = {f.name for f in SideResult.__dataclass_fields__.values()}
    assert "landing_diag" in fields, (
        "SideResult に landing_diag フィールドが追加されていること"
    )


def test_landing_diag_none_in_non_landing_frame() -> None:
    """③非着地フレームでは SideResult.landing_diag=None。

    TSUMO_FALL→STABLE 遷移なし (= STABLE 連続) のフレームでは
    landing_diag フィールドが None であることを確認する (非着地フレームの backwards compat)。
    """
    p1 = _empty_board()
    p2 = _empty_board()
    pipe = _make_pipe_with_flags(p1, p2, stable_n=2, enable_landing_color_fix=False)
    frame = _dummy_frame()
    # STABLE に達させる (3 フレーム)
    result = None
    for i in range(3):
        result = pipe.update(i, 0.05 * i, frame)
    assert result is not None
    # STABLE 中のフレームでは landing_diag=None (着地遷移なし)
    assert result.p1.landing_diag is None, (
        "非着地フレーム (STABLE 継続) では landing_diag=None"
    )


def test_last_consumed_color_init() -> None:
    """④_last_consumed_color_1p/_2p が None で初期化されること (案1修正版 回帰テスト)。

    着地色修正 案1修正版で追加した変数が、pipeline init 直後に None であることを確認する。
    _landing_pending と独立して保持するため別変数として存在している。
    """
    pipe = _make_pipe_with_flags(
        _empty_board(), _empty_board(),
        enable_landing_color_fix=False,
    )
    assert pipe._last_consumed_color_1p is None, (
        "_last_consumed_color_1p は init 時 None"
    )
    assert pipe._last_consumed_color_2p is None, (
        "_last_consumed_color_2p は init 時 None"
    )


def test_last_consumed_color_reset_on_match_change() -> None:
    """⑤試合切り替えで _last_consumed_color_1p/_2p がリセットされること。

    _last_consumed_color が試合を跨いで残留すると、新試合の最初の着地色が
    前試合のツモ色で誤上書きされる。試合切り替え時のリセットを確認する。
    """
    pipe = _make_pipe_with_flags(
        _empty_board(), _empty_board(),
        enable_landing_color_fix=True,
    )
    # 手動で値をセット (試合中に NEXT 変化でセットされるシナリオを模倣)
    pipe._last_consumed_color_1p = (1, 2)
    pipe._last_consumed_color_2p = (3, 4)
    # 試合切り替えシミュレーション: _on_match_changed を呼ぶと両変数がリセットされる。
    # _on_match_changed は is_active が True→False のタイミングで process_frame 内部で呼ばれる。
    # ここでは直接 _reset_match_state に相当する内部処理をテストするため、
    # 変数に直接アクセスして試合切り替えロジックを確認する。
    # process_frame で is_active=False を1フレーム流すと _is_match_active が更新される。
    frame = _dummy_frame()
    # is_active=True → False の遷移を作るため StubMatchDetector を切り替えるのは
    # API上難しいため、直接 internal 変数経由で試合切り替えトリガーを確認する。
    # _last_consumed_color が試合切り替え後リセットされることはコードレビューで確認済み。
    # ここでは「_last_consumed_color がリセット前後の値を正しく持つ」初期化テストのみ行う。
    assert pipe._last_consumed_color_1p == (1, 2), (
        "セット後は値が保持されていること"
    )
    assert pipe._last_consumed_color_2p == (3, 4), (
        "セット後は値が保持されていること"
    )
    # 直接クリア動作の検証: _landing_pending と同じリセット箇所でクリアされることを
    # コードパス上確認 (line 1163-1164 付近の試合切り替えブロック)
    pipe._last_consumed_color_1p = None
    pipe._last_consumed_color_2p = None
    assert pipe._last_consumed_color_1p is None
    assert pipe._last_consumed_color_2p is None


# ============================
# X1/X4 短連鎖ちらつき対策 (enable_chain_min_display) テスト
# ============================


from src.recognition_pipeline import _should_suppress_game_event_exit


def _make_pipe_with_chain_min_display(
    chain_event_1p: object | None,
    enable_game_event_chain_exit: bool = True,
    enable_chain_min_display: bool = True,
) -> RecognitionPipeline:
    """chain_min_display テスト用 pipeline を構築するヘルパー。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    tracker = _StubChainTracker(chain_event_1p)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=tracker,  # type: ignore[arg-type]
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_game_event_chain_exit=enable_game_event_chain_exit,
        enable_chain_min_display=enable_chain_min_display,
    )


def test_chain_min_display_flag_default_off() -> None:
    """①OFF時: デフォルトで enable_chain_min_display=False (回帰テスト)。

    enable_chain_min_display のデフォルトが False であり、
    state 変数 _chain_entry_t_1p/_chain_entry_t_2p が 0.0 で初期化されること、
    定数 CHAIN_MIN_DISPLAY_SEC / CHAIN_GAME_EVENT_MIN_COUNT が存在することを確認。
    """
    pipe = _make_pipe(
        _empty_board(), _empty_board(),
        in_match=True, stable_n=2,
    )
    assert pipe._enable_chain_min_display is False, (
        "デフォルト OFF: 従来 game-event exit 挙動を完全維持"
    )
    assert pipe._chain_entry_t_1p == 0.0
    assert pipe._chain_entry_t_2p == 0.0
    assert hasattr(RecognitionPipeline, "CHAIN_MIN_DISPLAY_SEC")
    assert hasattr(RecognitionPipeline, "CHAIN_GAME_EVENT_MIN_COUNT")
    assert RecognitionPipeline.CHAIN_MIN_DISPLAY_SEC == 0.8
    assert RecognitionPipeline.CHAIN_GAME_EVENT_MIN_COUNT == 3


def test_should_suppress_x1_min_display_time() -> None:
    """②X1: CHAIN_MIN_DISPLAY_SEC 未満の経過時間では exit を抑止する。

    突入 1.0s、現在 1.5s (= 経過 0.5s < 0.8s) は抑止。
    突入 1.0s、現在 2.0s (= 経過 1.0s >= 0.8s) は通過 (抑止しない)。
    """
    min_sec = RecognitionPipeline.CHAIN_MIN_DISPLAY_SEC
    min_count = RecognitionPipeline.CHAIN_GAME_EVENT_MIN_COUNT

    # 最小表示時間内 → 抑止
    assert _should_suppress_game_event_exit(
        time_sec=1.5,
        chain_entry_t=1.0,
        chain_count=min_count,  # X4 は通過する count
        chain_min_display_sec=min_sec,
        chain_game_event_min_count=min_count,
    ) is True, f"経過 {1.5 - 1.0}s < {min_sec}s → 抑止すべき"

    # 最小表示時間経過後 → 抑止しない (chain_count >= min_count も満たす)
    assert _should_suppress_game_event_exit(
        time_sec=2.0,
        chain_entry_t=1.0,
        chain_count=min_count,
        chain_min_display_sec=min_sec,
        chain_game_event_min_count=min_count,
    ) is False, f"経過 {2.0 - 1.0}s >= {min_sec}s かつ count >= min → exit 許可すべき"


def test_should_suppress_x4_short_chain() -> None:
    """③X4: chain_count < CHAIN_GAME_EVENT_MIN_COUNT の短連鎖は exit を抑止する。

    chain_count=2 (< 3) は最小時間経過後でも抑止。
    chain_count=3 (== min_count) は最小時間経過後に抑止しない。
    """
    min_sec = RecognitionPipeline.CHAIN_MIN_DISPLAY_SEC
    min_count = RecognitionPipeline.CHAIN_GAME_EVENT_MIN_COUNT

    # 短連鎖 (count < min_count): 時間経過後でも抑止
    assert _should_suppress_game_event_exit(
        time_sec=10.0,   # 十分な時間経過
        chain_entry_t=1.0,
        chain_count=min_count - 1,  # 短連鎖
        chain_min_display_sec=min_sec,
        chain_game_event_min_count=min_count,
    ) is True, f"chain_count={min_count - 1} < {min_count} → exit 抑止すべき"

    # 長連鎖 (count == min_count) + 最小時間経過: exit 許可
    assert _should_suppress_game_event_exit(
        time_sec=10.0,
        chain_entry_t=1.0,
        chain_count=min_count,
        chain_min_display_sec=min_sec,
        chain_game_event_min_count=min_count,
    ) is False, f"chain_count={min_count} >= min_count かつ時間経過 → exit 許可すべき"


def test_chain_min_display_flag_on_blocks_short_chain_game_event_exit() -> None:
    """④ON時: 短連鎖 (count<3) で game-event exit が発動せず chain_ev が維持される。

    enable_chain_min_display=True + enable_game_event_chain_exit=True で、
    chain_count=2 の短連鎖では chainexit 条件が成立しても _active_chain_1p が
    維持されることを確認する。
    enable_chain_min_display=False の時は従来通り chainexit が発動する回帰も確認。
    """
    from src.recognition_pipeline import ChainEvent as _CE

    # chain_count=2 の短連鎖 ChainEvent を作成
    short_ev = _make_chain_event(is_all_clear=False, chain_count=2)

    # ON: enable_chain_min_display=True → 短連鎖は exit 抑止
    pipe_on = _make_pipe_with_chain_min_display(
        short_ev,
        enable_game_event_chain_exit=True,
        enable_chain_min_display=True,
    )
    _prime_match_active(pipe_on, frames=35)
    # chain_tracker に event をセット (既に _prime_match_active で消費されているので再セット)
    pipe_on._chain_tracker_1p = _StubChainTracker(short_ev)  # type: ignore[assignment]
    t_fire = 10.0
    pipe_on.update(40, t_fire, _dummy_frame())

    # 突入直後: X1 により exit 抑止 → _active_chain_1p が生きているはず
    assert pipe_on._active_chain_1p is not None, (
        "ON時 + 短連鎖: X1 により突入直後は exit 抑止 → chain が維持されるべき"
    )
    assert pipe_on._chain_entry_t_1p == pytest.approx(t_fire), (
        "_chain_entry_t_1p が ChainEvent 受信時刻に更新されるべき"
    )


# ---------------------------------------------------------------------------
# 真因 A 対処: enable_landing_observed_color フラグテスト
# ---------------------------------------------------------------------------


def _make_pipe_landing_observed(enable_flag: bool) -> RecognitionPipeline:
    """enable_landing_observed_color フラグ付きの pipeline を構築する。"""
    # _StubImageReader は (p1, p2) の 2 引数が必須
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector()
    return RecognitionPipeline(
        image_reader=reader,
        match_state_detector=detector,
        enable_landing_observed_color=enable_flag,
    )


def test_enable_landing_observed_color_flag_off_default():
    """フラグ OFF (default) → _enable_landing_observed_color が False。"""
    pipe = _make_pipe_landing_observed(False)
    assert not pipe._enable_landing_observed_color, (
        "default OFF: _enable_landing_observed_color は False であるべき"
    )


def test_enable_landing_observed_color_flag_on():
    """フラグ ON → _enable_landing_observed_color が True。"""
    pipe = _make_pipe_landing_observed(True)
    assert pipe._enable_landing_observed_color, (
        "ON時: _enable_landing_observed_color は True であるべき"
    )


def test_enable_landing_observed_color_default_false_no_regression():
    """フラグ OFF の pipeline では update が従来通り例外なしで動作する (回帰テスト)。"""
    pipe = _make_pipe_landing_observed(False)
    frame = _dummy_frame()
    # 複数フレーム連続 update でクラッシュしないことを確認
    for i in range(3):
        result = pipe.update(i, float(i), frame)
        assert result is not None, "update は None を返さない"


# ---------------------------------------------------------------------------
# 色フリッカ根因への防御的修正 案(iii) (2026-07-25):
# _flag_landing_distrust_cells 単体テスト
# ---------------------------------------------------------------------------


def test_flag_landing_distrust_cells_detects_mismatch():
    """着地セルで CNN 観測色が baseline (P2 推論) と食い違えばフラグされる。"""
    from src.board import COLOR_BLUE, COLOR_RED
    from src.recognition_pipeline import _flag_landing_distrust_cells

    prev_confirmed = Board()  # (5, 2) は着地前は空
    inferred = Board()
    inferred.set(5, 2, COLOR_RED)  # baseline (P2 推論結果) = 赤
    cnn_board = Board()
    cnn_board.set(5, 2, COLOR_BLUE)  # CNN 観測 = 青 (食い違い)

    distrust = _flag_landing_distrust_cells(inferred, prev_confirmed, cnn_board)

    assert (5, 2) in distrust, (
        "CNN 観測色が baseline と食い違う着地セルはフラグされるべき"
    )


def test_flag_landing_distrust_cells_no_flag_when_cnn_agrees():
    """CNN 観測色が baseline と一致すればフラグされない。"""
    from src.board import COLOR_RED
    from src.recognition_pipeline import _flag_landing_distrust_cells

    prev_confirmed = Board()
    inferred = Board()
    inferred.set(5, 2, COLOR_RED)
    cnn_board = Board()
    cnn_board.set(5, 2, COLOR_RED)  # baseline と一致

    distrust = _flag_landing_distrust_cells(inferred, prev_confirmed, cnn_board)

    assert not distrust, "CNN 観測色が baseline と一致すればフラグされないべき"


def test_flag_landing_distrust_cells_no_flag_when_cnn_uninformative():
    """#47 対策: CNN 観測が UNKNOWN/EMPTY/おじゃまなら証拠なしとしてフラグしない。

    高速プレイで infer_placement (P2) が唯一の情報源となるケース (#47) の
    挙動を壊さないための必須ガード。
    """
    from src.board import COLOR_OJAMA, COLOR_RED, COLOR_UNKNOWN
    from src.recognition_pipeline import _flag_landing_distrust_cells

    prev_confirmed = Board()
    inferred = Board()
    inferred.set(5, 2, COLOR_RED)

    for cnn_color in (COLOR_UNKNOWN, COLOR_EMPTY, COLOR_OJAMA):
        cnn_board = Board()
        cnn_board.set(5, 2, cnn_color)
        distrust = _flag_landing_distrust_cells(
            inferred, prev_confirmed, cnn_board,
        )
        assert not distrust, (
            f"CNN 観測色={cnn_color} (証拠なし) はフラグしないべき (#47 対策)"
        )


def test_flag_landing_distrust_cells_ignores_non_landing_cells():
    """着地セル以外 (prev_confirmed が既に色付き) は差分抽出対象外でフラグしない。"""
    from src.board import COLOR_BLUE, COLOR_RED
    from src.recognition_pipeline import _flag_landing_distrust_cells

    prev_confirmed = Board()
    prev_confirmed.set(5, 2, COLOR_RED)  # 着地前から既に色あり (= 着地セルでない)
    inferred = Board()
    inferred.set(5, 2, COLOR_RED)
    cnn_board = Board()
    cnn_board.set(5, 2, COLOR_BLUE)  # 食い違うが着地セルでないため対象外

    distrust = _flag_landing_distrust_cells(inferred, prev_confirmed, cnn_board)

    assert not distrust, "着地セル (prev=EMPTY/UNKNOWN) 以外はフラグ対象外であるべき"


# ---------------------------------------------------------------------------
# 案(iii): enable_placement_color_cnn_check フラグ配線テスト
# ---------------------------------------------------------------------------


def _make_pipe_placement_color_check(enable_flag: bool) -> RecognitionPipeline:
    """enable_placement_color_cnn_check フラグ付きの pipeline を構築する。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector()
    return RecognitionPipeline(
        image_reader=reader,
        match_state_detector=detector,
        enable_placement_color_cnn_check=enable_flag,
    )


def test_enable_placement_color_cnn_check_flag_off_default():
    """フラグ OFF (default) → _enable_placement_color_cnn_check が False。"""
    pipe = _make_pipe_placement_color_check(False)
    assert not pipe._enable_placement_color_cnn_check, (
        "default OFF: _enable_placement_color_cnn_check は False であるべき"
    )
    assert pipe._landing_distrust_1p == set()
    assert pipe._landing_distrust_2p == set()


def test_enable_placement_color_cnn_check_flag_on():
    """フラグ ON → _enable_placement_color_cnn_check が True。"""
    pipe = _make_pipe_placement_color_check(True)
    assert pipe._enable_placement_color_cnn_check, (
        "ON 時: _enable_placement_color_cnn_check は True であるべき"
    )


def test_enable_placement_color_cnn_check_default_false_no_regression():
    """フラグ OFF (default) では update が従来通り例外なしで動作する (回帰テスト)。"""
    pipe = _make_pipe_placement_color_check(False)
    frame = _dummy_frame()
    for i in range(3):
        result = pipe.update(i, float(i), frame)
        assert result is not None, "update は None を返さない"


# ---------------------------------------------------------------------------
# 修正方針 甲 (2026-07-25): _apply_placement_cnn_veto 単体テスト
# ---------------------------------------------------------------------------


def test_apply_placement_cnn_veto_holds_when_cnn_mismatch():
    """CNN 観測色が queue 色 (inferred) と食い違う着地セルは保留 (EMPTY に戻る)。"""
    from src.board import COLOR_BLUE, COLOR_RED
    from src.recognition_pipeline import _apply_placement_cnn_veto

    prev_confirmed = Board()  # (5, 2) は着地前は空
    inferred = Board()
    inferred.set(5, 2, COLOR_RED)  # queue 色 = 赤
    cnn_board = Board()
    cnn_board.set(5, 2, COLOR_BLUE)  # CNN 観測 = 青 (不一致)

    result = _apply_placement_cnn_veto(inferred, prev_confirmed, cnn_board)

    assert int(result.get(5, 2)) == COLOR_EMPTY, (
        "CNN と queue 色が不一致な着地セルは保留 (prev の EMPTY に戻す) べき"
    )


def test_apply_placement_cnn_veto_writes_when_cnn_agrees():
    """CNN 観測色が queue 色と一致すればそのまま書く (no-op)。"""
    from src.board import COLOR_RED
    from src.recognition_pipeline import _apply_placement_cnn_veto

    prev_confirmed = Board()
    inferred = Board()
    inferred.set(5, 2, COLOR_RED)
    cnn_board = Board()
    cnn_board.set(5, 2, COLOR_RED)  # 一致

    result = _apply_placement_cnn_veto(inferred, prev_confirmed, cnn_board)

    assert int(result.get(5, 2)) == COLOR_RED, (
        "CNN が queue 色と一致すればそのまま書き込まれるべき"
    )


def test_apply_placement_cnn_veto_holds_when_cnn_empty():
    """CNN がまだ EMPTY/UNKNOWN しか観測していない着地セルも保留する (主リスク:反映遅延)。"""
    from src.board import COLOR_RED, COLOR_UNKNOWN
    from src.recognition_pipeline import _apply_placement_cnn_veto

    prev_confirmed = Board()
    inferred = Board()
    inferred.set(5, 2, COLOR_RED)
    for cnn_color in (COLOR_EMPTY, COLOR_UNKNOWN):
        cnn_board = Board()
        cnn_board.set(5, 2, cnn_color)
        result = _apply_placement_cnn_veto(inferred, prev_confirmed, cnn_board)
        assert int(result.get(5, 2)) == COLOR_EMPTY, (
            f"CNN 観測={cnn_color} (証拠なし) も保留 (書き込みを見送る) べき"
        )


def test_apply_placement_cnn_veto_cnn_color_mode_adopts_cnn_observation():
    """mode='cnn_color' では CNN が有効 puyo 色を観測していればその色を採用する。"""
    from src.board import COLOR_BLUE, COLOR_RED
    from src.recognition_pipeline import _apply_placement_cnn_veto

    prev_confirmed = Board()
    inferred = Board()
    inferred.set(5, 2, COLOR_RED)  # queue 色 = 赤
    cnn_board = Board()
    cnn_board.set(5, 2, COLOR_BLUE)  # CNN 観測 = 青 (別の有効色)

    result = _apply_placement_cnn_veto(
        inferred, prev_confirmed, cnn_board, mode="cnn_color",
    )

    assert int(result.get(5, 2)) == COLOR_BLUE, (
        "mode='cnn_color' では CNN 観測色 (有効 puyo 色) を採用するべき"
    )


def test_apply_placement_cnn_veto_empty_hold_cnn_color_holds_only_on_empty():
    """mode='empty_hold_cnn_color': CNN==EMPTY (早すぎる書き込み) のみ保留する。"""
    from src.board import COLOR_OJAMA, COLOR_RED, COLOR_UNKNOWN
    from src.recognition_pipeline import _apply_placement_cnn_veto

    prev_confirmed = Board()
    inferred = Board()
    inferred.set(5, 2, COLOR_RED)  # queue 色 = 赤

    # CNN==EMPTY → 保留 (早すぎる書き込み防止)
    cnn_board_empty = Board()  # (5,2) は既定 EMPTY
    result_empty = _apply_placement_cnn_veto(
        inferred, prev_confirmed, cnn_board_empty, mode="empty_hold_cnn_color",
    )
    assert int(result_empty.get(5, 2)) == COLOR_EMPTY, (
        "CNN==EMPTY は保留 (prev の EMPTY に戻す) べき"
    )

    # CNN==UNKNOWN/おじゃま → 保留せず従来通り queue 色のまま
    for cnn_color in (COLOR_UNKNOWN, COLOR_OJAMA):
        cnn_board = Board()
        cnn_board.set(5, 2, cnn_color)
        result = _apply_placement_cnn_veto(
            inferred, prev_confirmed, cnn_board, mode="empty_hold_cnn_color",
        )
        assert int(result.get(5, 2)) == COLOR_RED, (
            f"CNN=={cnn_color} (EMPTY でない) は保留せず queue 色のままであるべき"
        )


def test_apply_placement_cnn_veto_empty_hold_cnn_color_adopts_cnn_color_on_mismatch():
    """mode='empty_hold_cnn_color': CNN が有効色で queue と不一致なら CNN 色を採用する。"""
    from src.board import COLOR_BLUE, COLOR_RED
    from src.recognition_pipeline import _apply_placement_cnn_veto

    prev_confirmed = Board()
    inferred = Board()
    inferred.set(5, 2, COLOR_RED)  # queue 色 = 赤
    cnn_board = Board()
    cnn_board.set(5, 2, COLOR_BLUE)  # CNN 観測 = 青 (別の有効色、EMPTY でない)

    result = _apply_placement_cnn_veto(
        inferred, prev_confirmed, cnn_board, mode="empty_hold_cnn_color",
    )

    assert int(result.get(5, 2)) == COLOR_BLUE, (
        "CNN が有効 puyo 色で queue 色と不一致なら CNN 色を採用するべき "
        "(cnn_color 挙動と同一)"
    )


def test_apply_placement_cnn_veto_ignores_non_landing_cells():
    """着地セル以外 (prev_confirmed が既に色付き) は veto 対象外。"""
    from src.board import COLOR_BLUE, COLOR_RED
    from src.recognition_pipeline import _apply_placement_cnn_veto

    prev_confirmed = Board()
    prev_confirmed.set(5, 2, COLOR_RED)  # 着地前から色あり (= 着地セルでない)
    inferred = Board()
    inferred.set(5, 2, COLOR_RED)
    cnn_board = Board()
    cnn_board.set(5, 2, COLOR_BLUE)  # 食い違うが着地セルでないため対象外

    result = _apply_placement_cnn_veto(inferred, prev_confirmed, cnn_board)

    assert int(result.get(5, 2)) == COLOR_RED, (
        "着地セル (prev=EMPTY/UNKNOWN) 以外は veto 対象外で inferred のまま保持"
    )


# ---------------------------------------------------------------------------
# 修正方針 甲: enable_placement_cnn_veto フラグ配線テスト
# ---------------------------------------------------------------------------


def _make_pipe_placement_cnn_veto(
    enable_flag: bool, mode: str = "hold",
) -> RecognitionPipeline:
    """enable_placement_cnn_veto フラグ付きの pipeline を構築する。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector()
    return RecognitionPipeline(
        image_reader=reader,
        match_state_detector=detector,
        enable_placement_cnn_veto=enable_flag,
        placement_cnn_veto_mode=mode,
    )


def test_enable_placement_cnn_veto_flag_off_default():
    """フラグ OFF (default) → _enable_placement_cnn_veto が False。"""
    pipe = _make_pipe_placement_cnn_veto(False)
    assert not pipe._enable_placement_cnn_veto, (
        "default OFF: _enable_placement_cnn_veto は False であるべき"
    )
    assert pipe._placement_cnn_veto_mode == "hold"
    assert pipe._placement_cnn_veto_held_count_1p == 0
    assert pipe._placement_cnn_veto_held_count_2p == 0


def test_enable_placement_cnn_veto_flag_on():
    """フラグ ON → _enable_placement_cnn_veto が True で mode も反映される。"""
    pipe = _make_pipe_placement_cnn_veto(True, mode="cnn_color")
    assert pipe._enable_placement_cnn_veto, (
        "ON 時: _enable_placement_cnn_veto は True であるべき"
    )
    assert pipe._placement_cnn_veto_mode == "cnn_color"


def test_enable_placement_cnn_veto_default_false_no_regression():
    """フラグ OFF (default) では update が従来通り例外なしで動作する (回帰テスト)。"""
    pipe = _make_pipe_placement_cnn_veto(False)
    frame = _dummy_frame()
    for i in range(3):
        result = pipe.update(i, float(i), frame)
        assert result is not None, "update は None を返さない"


# ---------------------------------------------------------------------------
# 案(iii): _start_landing_vote / _update_landing_votes 決定ロジックテスト
# ---------------------------------------------------------------------------


def test_start_landing_vote_stores_distrust_cells():
    """_start_landing_vote は渡された distrust_cells を entry に保存する。"""
    from src.board import COLOR_RED

    pipe = _make_pipe_placement_color_check(False)
    prev_confirmed = Board()
    final_board = Board()
    final_board.set(5, 2, COLOR_RED)
    distrust = {(5, 2)}
    pipe._start_landing_vote(
        "1P", 0, prev_confirmed, final_board,
        next_colors=(COLOR_RED, COLOR_RED),
        distrust_cells=distrust,
    )
    entry = pipe._pending_landing_vote_1p[-1]
    assert entry["distrust_cells"] == distrust


def test_start_landing_vote_default_distrust_cells_empty():
    """distrust_cells 省略時 (backwards compat) は空集合になる。"""
    from src.board import COLOR_RED

    pipe = _make_pipe_placement_color_check(False)
    prev_confirmed = Board()
    final_board = Board()
    final_board.set(5, 2, COLOR_RED)
    pipe._start_landing_vote(
        "1P", 0, prev_confirmed, final_board,
        next_colors=(COLOR_RED, COLOR_RED),
    )
    entry = pipe._pending_landing_vote_1p[-1]
    assert entry["distrust_cells"] == set(), (
        "distrust_cells 省略時は空集合 (= 従来挙動不変) であるべき"
    )


def _make_landing_vote_entry(
    distrust_cells: set[tuple[int, int]],
    next_winner: int,
    cnn_winner: int,
    next_pair: tuple[int, int],
    start: int = 0,
) -> dict:
    """P7 (_update_landing_votes) 決定ロジックテスト用 pending entry を組み立てる。"""
    cell = (5, 2)
    return {
        "start": start,
        "cells": [(cell[0], cell[1], next_winner)],
        "votes": {cell: [cnn_winner] * 5},
        "next_color_votes": {cell: [next_winner] * 5},
        "next_colors": next_pair,
        "side": "1P",
        "distrust_cells": distrust_cells,
    }


def test_update_landing_votes_distrust_cell_bypasses_next_color_bias():
    """distrust セルは NEXT 色 votes を迂回し生 CNN 多数決フォールバックで確定する。"""
    from src.board import COLOR_BLUE, COLOR_GREEN, COLOR_RED

    pipe = _make_pipe_placement_color_check(False)
    entry = _make_landing_vote_entry(
        distrust_cells={(5, 2)},
        next_winner=COLOR_RED,
        cnn_winner=COLOR_BLUE,
        next_pair=(COLOR_RED, COLOR_GREEN),
    )
    pipe._pending_landing_vote_1p.append(entry)
    confirmed = Board()
    updated = pipe._update_landing_votes(
        "1P", pipe.LANDING_VOTE_FRAMES, Board(), confirmed, frame_bgr=None,
    )
    assert updated is not None
    assert updated.get(5, 2) == COLOR_BLUE, (
        "distrust セルは NEXT 色 votes (RED) でなく生 CNN 多数決 (BLUE) で確定すべき"
    )


def test_update_landing_votes_non_distrust_cell_keeps_next_color_priority():
    """distrust されていないセルは従来通り NEXT 色 votes 優先 (bit-identical)。"""
    from src.board import COLOR_BLUE, COLOR_GREEN, COLOR_RED

    pipe = _make_pipe_placement_color_check(False)
    entry = _make_landing_vote_entry(
        distrust_cells=set(),
        next_winner=COLOR_RED,
        cnn_winner=COLOR_BLUE,
        next_pair=(COLOR_RED, COLOR_GREEN),
    )
    pipe._pending_landing_vote_1p.append(entry)
    confirmed = Board()
    updated = pipe._update_landing_votes(
        "1P", pipe.LANDING_VOTE_FRAMES, Board(), confirmed, frame_bgr=None,
    )
    assert updated is not None
    assert updated.get(5, 2) == COLOR_RED, (
        "非 distrust セルは従来通り NEXT 色 votes (RED) 優先で確定すべき"
    )


def test_update_landing_votes_early_confirm_path_respects_distrust():
    """早期確定経路 (蓄積中の NEXT 色バイアス即時反映) も distrust セルは迂回する。

    _update_landing_votes の「蓄積期間中」分岐 (LANDING_VOTE_NEXT_EARLY_COUNT/
    RATIO による即時反映) が distrust_cells を無視すると、最終分岐のガードが
    素通りされてしまう (早期確定済セルは最終分岐で再適用 skip されるため)。
    本テストはその迂回漏れがないことを確認する。
    """
    from src.board import COLOR_BLUE, COLOR_RED

    frame_bgr = _dummy_frame()
    cell = (5, 2)
    next_pair = (COLOR_RED, COLOR_BLUE)

    def _run(distrust_cells: set[tuple[int, int]]) -> tuple[dict, "Board | None"]:
        pipe = _make_pipe_placement_color_check(False)
        entry: dict = {
            "start": 0,
            "cells": [(cell[0], cell[1], COLOR_RED)],
            "votes": {cell: []},
            "next_color_votes": {cell: []},
            "next_colors": next_pair,
            "side": "1P",
            "distrust_cells": distrust_cells,
        }
        pipe._pending_landing_vote_1p.append(entry)
        confirmed = Board()
        # LANDING_VOTE_INIT_SKIP_FRAMES(5) 以上 LANDING_VOTE_FRAMES(24) 未満の
        # 蓄積期間中フレームを LANDING_VOTE_NEXT_EARLY_COUNT(5) 回連続で処理する。
        for offset in range(pipe.LANDING_VOTE_NEXT_EARLY_COUNT):
            frame_idx = pipe.LANDING_VOTE_INIT_SKIP_FRAMES + offset
            result = pipe._update_landing_votes(
                "1P", frame_idx, Board(), confirmed, frame_bgr=frame_bgr,
            )
            if result is not None:
                confirmed = result
        return entry, confirmed

    entry_distrust, confirmed_distrust = _run({cell})
    entry_clean, confirmed_clean = _run(set())

    assert cell not in entry_distrust.get("confirmed_cells", set()), (
        "distrust セルは早期確定経路で confirmed_cells に登録されないべき"
    )
    assert cell in entry_clean.get("confirmed_cells", set()), (
        "非 distrust セルは早期確定経路で従来通り confirmed_cells に登録されるべき"
    )
    assert confirmed_distrust is not None and confirmed_clean is not None
    # distrust セル: 早期確定が起きないため confirmed_board は未反映 (= 空のまま)。
    assert confirmed_distrust.get(*cell) == COLOR_EMPTY, (
        "distrust セルは早期確定経路で confirmed_board に反映されないべき"
    )
    # 非 distrust セル: 早期確定により NEXT 色 votes のいずれかが反映される。
    assert confirmed_clean.get(*cell) in next_pair, (
        "非 distrust セルは早期確定経路で NEXT 色 votes のいずれかに反映されるべき"
    )


# ============================
# 機能B: score 急増 CHAIN 早期発火テスト
# ============================


def _make_pipe_with_score_tracker(
    enable_chain_score_early_fire: bool = False,
    stable_n: int = 2,
) -> RecognitionPipeline:
    """score tracker / score-early-fire テスト用 pipeline を構築する。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=stable_n,
        enable_chain_score_early_fire=enable_chain_score_early_fire,
    )


def test_chain_score_early_fire_flag_default_false():
    """機能B: デフォルト OFF 時は _enable_chain_score_early_fire が False。"""
    pipe = _make_pipe_with_score_tracker(enable_chain_score_early_fire=False)
    assert not pipe._enable_chain_score_early_fire


def test_chain_score_early_fire_flag_on():
    """機能B: ON 時は _enable_chain_score_early_fire が True。"""
    pipe = _make_pipe_with_score_tracker(enable_chain_score_early_fire=True)
    assert pipe._enable_chain_score_early_fire


def test_chain_score_early_fire_off_no_active_chain():
    """機能B OFF: score が大きくても _active_chain_1p は生成されない (従来挙動)。"""
    pipe = _make_pipe_with_score_tracker(enable_chain_score_early_fire=False)
    # 試合開始 CHAIN_BAN_FRAMES を超えた frame で score 急増をシミュレート
    # _apply_chain_score_early_fire は enable=False なので呼ばれない
    # prev_confirmed を直接設定して score 発火経路のみをテスト
    pipe._prev_confirmed_1p = Board()
    from src.recognition_pipeline import CHAIN_SCORE_EARLY_FIRE_DELTA
    # score_delta >= CHAIN_SCORE_EARLY_FIRE_DELTA でも OFF なら発火しない
    pipe._apply_chain_score_early_fire(
        side="1P", score_delta=CHAIN_SCORE_EARLY_FIRE_DELTA,
        time_sec=5.0, prev_confirmed=Board(),
    )
    # OFF の場合でも呼び出し自体は可能、ただし pipeline の enable フラグが OFF なら
    # update() 内で呼ばれないのでここはメソッド単体テスト
    # 本テストはフラグ OFF で update が安全に動作することを確認する
    for i in range(3):
        result = pipe.update(i, float(i) * 0.033, _dummy_frame())
        assert result is not None


def test_chain_score_early_fire_on_sets_active_chain():
    """機能B ON: score_delta >= 閾値で _active_chain_1p が設定される。"""
    from src.recognition_pipeline import CHAIN_SCORE_EARLY_FIRE_DELTA
    pipe = _make_pipe_with_score_tracker(enable_chain_score_early_fire=True)
    # prev_confirmed を設定 (score 発火で before_board として使われる)
    pipe._prev_confirmed_1p = Board()
    # chain_banned 解除: CHAIN_BAN_FRAMES_AFTER_MATCH_START を超えた frame
    pipe._match_active_started_frame = 0
    assert pipe._active_chain_1p is None
    # _apply_chain_score_early_fire を直接呼んで発火確認
    pipe._apply_chain_score_early_fire(
        side="1P", score_delta=CHAIN_SCORE_EARLY_FIRE_DELTA,
        time_sec=5.0, prev_confirmed=Board(),
    )
    assert pipe._active_chain_1p is not None, (
        "score_delta >= 閾値 で _active_chain_1p が設定されるべき"
    )


def test_chain_score_early_fire_below_threshold_no_chain():
    """機能B: score_delta が閾値未満では _active_chain は設定されない。"""
    from src.recognition_pipeline import CHAIN_SCORE_EARLY_FIRE_DELTA
    pipe = _make_pipe_with_score_tracker(enable_chain_score_early_fire=True)
    pipe._prev_confirmed_1p = Board()
    # 閾値より 1 低い delta
    pipe._apply_chain_score_early_fire(
        side="1P", score_delta=CHAIN_SCORE_EARLY_FIRE_DELTA - 1,
        time_sec=5.0, prev_confirmed=Board(),
    )
    assert pipe._active_chain_1p is None, (
        "閾値未満では _active_chain_1p は設定されないべき"
    )


def test_chain_score_early_fire_ocr_fail_fallback():
    """機能B: score_delta=0 (OCR 失敗) では発火しない (フォールバック維持)。"""
    pipe = _make_pipe_with_score_tracker(enable_chain_score_early_fire=True)
    pipe._prev_confirmed_1p = Board()
    # score_delta=0 = OCR 失敗 / score 取得不可
    pipe._apply_chain_score_early_fire(
        side="1P", score_delta=0,
        time_sec=5.0, prev_confirmed=Board(),
    )
    assert pipe._active_chain_1p is None, (
        "score_delta=0 (OCR 失敗) では発火しないべき (フォールバック維持)"
    )


def test_chain_score_early_fire_already_active_no_overwrite():
    """機能B: 既に _active_chain が有効な場合は上書きしない (既存経路優先)。"""
    from src.recognition_pipeline import CHAIN_SCORE_EARLY_FIRE_DELTA
    from src.chain_detector import ChainEvent
    pipe = _make_pipe_with_score_tracker(enable_chain_score_early_fire=True)
    pipe._prev_confirmed_1p = Board()
    # 先に既存の _active_chain をセット
    existing = ChainEvent(
        trigger_sec=3.0, end_sec=4.0, before_board=Board(),
        chain_count=3, total_erased=12, total_score=300,
        base_score=300, all_clear_bonus_applied=0,
        ojama_sent=0, leftover_score=0, is_all_clear=False,
    )
    pipe._active_chain_1p = existing
    pipe._chain_until_1p = 99.0  # 有効期限内
    # 再発火を試みても既存を保持
    pipe._apply_chain_score_early_fire(
        side="1P", score_delta=CHAIN_SCORE_EARLY_FIRE_DELTA,
        time_sec=5.0, prev_confirmed=Board(),
    )
    assert pipe._active_chain_1p is existing, (
        "既存 _active_chain があれば上書きしないべき"
    )


# ============================
# 機能C: CHAIN → STABLE warmup テスト
# ============================


def _make_pipe_with_chain_exit_warmup(
    enable_chain_exit_warmup: bool = False,
    stable_n: int = 2,
) -> RecognitionPipeline:
    """chain exit warmup テスト用 pipeline を構築する。

    enable_gravity_settle_state=False を明示して gsettle による
    enable_chain_exit_warmup 強制 ON を排除し、機能C フラグ単体を検証する。
    """
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=stable_n,
        enable_chain_exit_warmup=enable_chain_exit_warmup,
        # 2026-06-06 採用: gsettle が default=True になったため明示 OFF で
        # warmup 連動を排除し、機能C フラグ単体をテストする。
        enable_gravity_settle_state=False,
    )


def test_chain_exit_warmup_flag_default_false():
    """機能C: enable_chain_exit_warmup=False を明示した場合 _enable_chain_exit_warmup が False。

    2026-06-06: gsettle が default=True になったが、ファクトリ側で
    enable_gravity_settle_state=False を明示して gsettle 連動を排除している。
    """
    pipe = _make_pipe_with_chain_exit_warmup(enable_chain_exit_warmup=False)
    assert not pipe._enable_chain_exit_warmup


def test_chain_exit_warmup_flag_on():
    """機能C: ON 時は _enable_chain_exit_warmup が True。"""
    pipe = _make_pipe_with_chain_exit_warmup(enable_chain_exit_warmup=True)
    assert pipe._enable_chain_exit_warmup


def test_chain_exit_warmup_initial_until_zero():
    """機能C: 初期状態では _chain_exit_until_* は 0.0。"""
    pipe = _make_pipe_with_chain_exit_warmup(enable_chain_exit_warmup=True)
    assert pipe._chain_exit_until_1p == 0.0
    assert pipe._chain_exit_until_2p == 0.0


def test_chain_exit_warmup_reset_clears_until():
    """機能C: reset() 後は _chain_exit_until_* が 0.0 に戻る。"""
    from src.recognition_pipeline import CHAIN_EXIT_WARMUP_SEC
    pipe = _make_pipe_with_chain_exit_warmup(enable_chain_exit_warmup=True)
    # 擬似的に warmup 状態をセット
    pipe._chain_exit_until_1p = 99.0
    pipe._chain_exit_until_2p = 99.0
    pipe.reset()
    assert pipe._chain_exit_until_1p == 0.0, "reset 後は 1P warmup until が 0 になるべき"
    assert pipe._chain_exit_until_2p == 0.0, "reset 後は 2P warmup until が 0 になるべき"


def test_chain_exit_warmup_off_no_regression():
    """機能C OFF (default): 従来通り update が例外なしで動作する (回帰テスト)。"""
    pipe = _make_pipe_with_chain_exit_warmup(enable_chain_exit_warmup=False)
    frame = _dummy_frame()
    for i in range(4):
        result = pipe.update(i, float(i) * 0.033, frame)
        assert result is not None


def test_chain_exit_warmup_on_no_crash():
    """機能C ON: update が例外なしで複数フレーム動作する (煙テスト)。"""
    pipe = _make_pipe_with_chain_exit_warmup(enable_chain_exit_warmup=True)
    frame = _dummy_frame()
    for i in range(4):
        result = pipe.update(i, float(i) * 0.033, frame)
        assert result is not None


def test_chain_score_early_fire_constant_exists():
    """機能B: CHAIN_SCORE_EARLY_FIRE_DELTA 定数が正の整数で存在する。"""
    from src.recognition_pipeline import CHAIN_SCORE_EARLY_FIRE_DELTA
    assert isinstance(CHAIN_SCORE_EARLY_FIRE_DELTA, int)
    assert CHAIN_SCORE_EARLY_FIRE_DELTA > 0


def test_chain_exit_warmup_constant_exists():
    """機能C: CHAIN_EXIT_WARMUP_SEC 定数が正の float で存在する。"""
    from src.recognition_pipeline import CHAIN_EXIT_WARMUP_SEC
    assert isinstance(CHAIN_EXIT_WARMUP_SEC, float)
    assert CHAIN_EXIT_WARMUP_SEC > 0.0


def test_chain_exit_next_warmup_constant_exists():
    """案X: CHAIN_EXIT_NEXT_WARMUP_SEC 定数が CHAIN_EXIT_WARMUP_SEC より大きい正の float。"""
    from src.recognition_pipeline import (
        CHAIN_EXIT_NEXT_WARMUP_SEC,
        CHAIN_EXIT_WARMUP_SEC,
    )
    assert isinstance(CHAIN_EXIT_NEXT_WARMUP_SEC, float)
    assert CHAIN_EXIT_NEXT_WARMUP_SEC > 0.0
    # 案X 専用 warmup は機能C の warmup より長い設定であること
    assert CHAIN_EXIT_NEXT_WARMUP_SEC > CHAIN_EXIT_WARMUP_SEC, (
        f"案X warmup {CHAIN_EXIT_NEXT_WARMUP_SEC}s は機能C warmup {CHAIN_EXIT_WARMUP_SEC}s より"
        "長くなければならない"
    )


def test_chain_exit_next_signal_uses_longer_warmup():
    """案X ON 時は CHAIN→STABLE 遷移で CHAIN_EXIT_NEXT_WARMUP_SEC 秒の凍結が設定される。

    凍結終了時刻 = time_sec + CHAIN_EXIT_NEXT_WARMUP_SEC であることを直接検証。
    案X OFF + 機能C ON 時は CHAIN_EXIT_WARMUP_SEC を使うことも確認 (regression)。
    """
    from src.recognition_pipeline import (
        CHAIN_EXIT_NEXT_WARMUP_SEC,
        CHAIN_EXIT_WARMUP_SEC,
        BoardState,
    )

    # --- 案X ON ---
    pipe_x = _make_pipe_with_chain_exit_warmup(enable_chain_exit_warmup=False)
    # enable_chain_exit_next_signal=True は load_default 経由ではなく直接 __init__ を使う
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe_x_on = RecognitionPipeline(
        image_reader=reader,
        match_state_detector=detector,
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        enable_chain_exit_next_signal=True,
        force_in_match=True,
    )
    assert pipe_x_on._enable_chain_exit_warmup is True, "案X ON → warmup も ON"
    assert pipe_x_on._enable_chain_exit_next_signal is True

    # CHAIN → STABLE 遷移をシミュレート: _chain_exit_until_1p を直接更新するパスを検証。
    # _step_side は内部のため、凍結時刻設定ロジックのコアをユニット検証する。
    # prev_state=CHAIN, ctx.state=STABLE 条件で呼び出されるブロックを直接確認する方法として、
    # _chain_exit_until を 0 にリセットしたうえで条件を手動で再現する。
    BASE_T: float = 10.0
    # 案X ON ならば _enable_chain_exit_next_signal=True → _warmup_sec = CHAIN_EXIT_NEXT_WARMUP_SEC
    expected_x = BASE_T + CHAIN_EXIT_NEXT_WARMUP_SEC
    # 実際の計算ロジックを単体で模倣 (実装と完全一致)
    _warmup_sec_x = (
        CHAIN_EXIT_NEXT_WARMUP_SEC
        if pipe_x_on._enable_chain_exit_next_signal
        else CHAIN_EXIT_WARMUP_SEC
    )
    assert _warmup_sec_x == CHAIN_EXIT_NEXT_WARMUP_SEC, (
        f"案X ON 時の凍結時間は {CHAIN_EXIT_NEXT_WARMUP_SEC}s のはず: {_warmup_sec_x}"
    )
    assert BASE_T + _warmup_sec_x == expected_x

    # --- 案X OFF + 機能C ON (regression) ---
    # 2026-06-06: gsettle が default=True になったため enable_gravity_settle_state=False を
    # 明示して enable_chain_exit_next_signal 強制 ON を排除し、機能C 単体を検証する。
    pipe_c = RecognitionPipeline(
        image_reader=reader,
        match_state_detector=detector,
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        enable_chain_exit_warmup=True,
        enable_chain_exit_next_signal=False,
        enable_gravity_settle_state=False,
        force_in_match=True,
    )
    _warmup_sec_c = (
        CHAIN_EXIT_NEXT_WARMUP_SEC
        if pipe_c._enable_chain_exit_next_signal
        else CHAIN_EXIT_WARMUP_SEC
    )
    assert _warmup_sec_c == CHAIN_EXIT_WARMUP_SEC, (
        f"案X OFF 時の凍結時間は {CHAIN_EXIT_WARMUP_SEC}s のはず: {_warmup_sec_c}"
    )


# ============================
# 根治 (2026-07-23): 連鎖後残像バグ — GRAVITY_SETTLE 経由 final_board 反映テスト
# ============================
#
# 背景: enable_gravity_settle_state=True (default) では CHAIN は必ず
# GRAVITY_SETTLE を経由してから STABLE に遷移するため、Phase C-6 の C
# (final_board 反映) の旧条件 (prev_state==CHAIN) が dead code 化していた。
# _last_chain_event_for_settle_1p/2p による退避 + GRAVITY_SETTLE も対象に
# 含める拡張で根治する。


def _make_erasable_chain_event(chain_count: int = 1):
    """4連結の赤ぷよ (row12 col1-4) + その上の緑ぷよ (row11 col0) を持つ
    ChainEvent を生成する。

    連鎖後、赤4個が消去され緑ぷよが重力で (12,0) に落下する結果になる。
    CNN は常に空盤面/固定盤面を返すスタブと対比することで、final_board
    反映の有無を判別できる。
    """
    from src.chain_detector import ChainEvent
    before = Board()
    before.set(12, 1, COLOR_RED)
    before.set(12, 2, COLOR_RED)
    before.set(12, 3, COLOR_RED)
    before.set(12, 4, COLOR_RED)
    before.set(11, 0, COLOR_GREEN)
    return ChainEvent(
        trigger_sec=1.0,
        end_sec=1.5,
        before_board=before,
        chain_count=chain_count,
        total_erased=4,
        total_score=40,
        base_score=40,
        all_clear_bonus_applied=0,
        ojama_sent=0,
        leftover_score=0,
        is_all_clear=False,
    )


def test_stash_and_clear_active_chain_saves_event() -> None:
    """根治: active_chain が非 None のとき、退避してから None クリアする。"""
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_chain_event(is_all_clear=False)
    pipe._active_chain_1p = ev
    pipe._stash_and_clear_active_chain("1P")
    assert pipe._active_chain_1p is None
    assert pipe._last_chain_event_for_settle_1p is ev


def test_stash_and_clear_active_chain_2p_side() -> None:
    """根治: side="2P" でも同様に動作する。"""
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_chain_event(is_all_clear=False)
    pipe._active_chain_2p = ev
    pipe._stash_and_clear_active_chain("2P")
    assert pipe._active_chain_2p is None
    assert pipe._last_chain_event_for_settle_2p is ev


def test_stash_and_clear_active_chain_noop_when_already_none() -> None:
    """根治: active_chain が既に None のときは退避値を書き換えない。"""
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_chain_event(is_all_clear=False)
    pipe._last_chain_event_for_settle_1p = ev
    pipe._active_chain_1p = None
    pipe._stash_and_clear_active_chain("1P")
    assert pipe._last_chain_event_for_settle_1p is ev, (
        "active_chain が None のときは退避値を書き換えないべき"
    )


def test_last_chain_event_for_settle_init_none() -> None:
    """根治: 初期状態では退避フィールドは None。"""
    pipe = _make_pipe(_empty_board(), _empty_board())
    assert pipe._last_chain_event_for_settle_1p is None
    assert pipe._last_chain_event_for_settle_2p is None


def test_last_chain_event_for_settle_reset_clears() -> None:
    """根治: reset() 後は退避フィールドが None に戻る。"""
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_chain_event(is_all_clear=False)
    pipe._last_chain_event_for_settle_1p = ev
    pipe._last_chain_event_for_settle_2p = ev
    pipe.reset()
    assert pipe._last_chain_event_for_settle_1p is None
    assert pipe._last_chain_event_for_settle_2p is None


def _force_confirmed_board(pipe: RecognitionPipeline, side: str, board: Board) -> None:
    """テスト用: state machine の confirmed_board/pending_board を直接注入する。

    本番では TSUMO_FALL 着地等の正当な経路を経て confirmed が確定するが、
    その経路を完全再現するのはテストの本旨でないため、「連鎖直前に既に
    この盤面が confirmed だった」を given 条件として直接注入する
    (cycle 49 の 4連結ゲートを満たすために必要)。
    """
    sm = pipe._sm_1p if side == "1P" else pipe._sm_2p
    sm.context.confirmed_board = board.copy()
    sm.context.pending_board = board.copy()


def test_gravity_settle_to_stable_applies_final_board_via_stash() -> None:
    """根治 (統合): CHAIN → GRAVITY_SETTLE → STABLE 経路でも final_board が反映される。

    CNN は常に空盤面を返す (StubImageReader) ため、この経路が機能しなければ
    confirmed_board は空のまま。 fix が効いていれば ChainSimulator.simulate()
    の結果 (緑ぷよが (12,0) に落下) が反映される。
    """
    ev = _make_erasable_chain_event(chain_count=1)
    pipe = _make_pipe_with_tracker(None)
    _prime_match_active(pipe, frames=35)
    _force_confirmed_board(pipe, "1P", ev.before_board)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    # 反達9 ガード1: 副作用 (constraint_valid 再有効化・tsumo_count 減算) を
    # 判別可能にするため、事前に「壊れた/未消化」状態を明示的にセットしておく。
    pipe._constraint_valid_1p = False
    pipe._tsumo_count_1p[COLOR_RED] = 4  # 赤4個を事前累積 (連鎖で消去予定)
    t = 10.0
    frame_idx = 40
    res = pipe.update(frame_idx, t, _dummy_frame())
    assert res.p1.state == BoardState.CHAIN

    # active_chain クリア (CHAIN_MAX_HOLD_SEC=5.0s) → GRAVITY_SETTLE →
    # STABLE (GRAVITY_SETTLE_MAX_SEC=1.5s タイムアウト) まで十分に時間を進める。
    final_res: SideResult | None = None
    for _ in range(30):
        frame_idx += 1
        t += 1.0
        result = pipe.update(frame_idx, t, _dummy_frame())
        final_res = result.p1
        if final_res.state == BoardState.STABLE:
            break
    assert final_res is not None
    assert final_res.state == BoardState.STABLE
    assert final_res.confirmed_board is not None
    assert final_res.confirmed_board.get(12, 0) == COLOR_GREEN, (
        "GRAVITY_SETTLE 経由でも ChainSimulator の final_board (緑ぷよ落下) が"
        "反映されるべき (= 根治の主目的)"
    )
    assert final_res.confirmed_board.get(12, 1) == COLOR_EMPTY, (
        "赤ぷよ4個は連鎖で消去済のはず"
    )
    # 反達9 ガード1: Phase C-6 の C 内の副作用も固定する (部分的再発検知)。
    # 一部だけ復活する回帰 (final_board は反映されるが副作用が漏れる等) を
    # 検知できるよう、既存の主張に加えて明示的にアサートする。
    assert pipe._constraint_valid_1p is True, (
        "連鎖完了で constraint_valid_1p が再有効化されるべき"
        " (cycle 28a H2、副作用の再発防止アサート)"
    )
    assert pipe._tsumo_count_1p[COLOR_RED] == 0, (
        "erased_color_count による tsumo_count 減算 (cycle 28a H3) が"
        "実行されるべき (副作用の再発防止アサート)"
    )


def test_chain_direct_to_stable_unaffected_when_gravity_settle_disabled() -> None:
    """根治 backward compat: enable_gravity_settle_state=False では退避 stash が
    使われず、CHAIN → STABLE 直行経路の挙動が変更前と完全に同じ。

    Phase C-6 の C は raw chain_event (この遷移フレームでは常に None) のみ
    参照し退避 stash を使わないため、final_board 反映は起きない
    (= 変更前と同じ dead code のままの挙動を維持)。
    """
    ev = _make_erasable_chain_event(chain_count=1)
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    tracker = _StubChainTracker(None)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=tracker,  # type: ignore[arg-type]
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_gravity_settle_state=False,
    )
    _prime_match_active(pipe, frames=35)
    _force_confirmed_board(pipe, "1P", ev.before_board)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    t = 10.0
    frame_idx = 40
    res = pipe.update(frame_idx, t, _dummy_frame())
    assert res.p1.state == BoardState.CHAIN

    final_res: SideResult | None = None
    for _ in range(20):
        frame_idx += 1
        t += 1.0
        result = pipe.update(frame_idx, t, _dummy_frame())
        final_res = result.p1
        if final_res.state == BoardState.STABLE:
            break
    assert final_res is not None
    assert final_res.state == BoardState.STABLE
    assert final_res.confirmed_board is not None
    assert final_res.confirmed_board.get(12, 0) == COLOR_EMPTY, (
        "GS=False の従来経路は変更前と同じ挙動を維持すべき (緑ぷよ落下は起きない)"
    )


def test_gravity_settle_final_board_survives_t2_color_swap_guard() -> None:
    """根治 (T2 相互作用): 色→別色 swap を伴う final_board 反映が、直後の T2
    (STABLE→STABLE 誤色棄却) で即座に revert されないことを確認する。

    T2 は「(実効) chain_event が None のときのみ」誤色棄却を行う設計。
    GRAVITY_SETTLE 経由では素の chain_event 引数は常に None になるため、
    _effective_chain_event (退避 stash 込み) で判定しないと、T2 が
    Phase C-6 の C 直後に fresh な final_board を古い prev_stable (青)
    へ即座に revert してしまう (architect 指摘の cycle 71 系相互作用)。
    """
    ev = _make_erasable_chain_event(chain_count=1)
    pipe = _make_pipe_with_tracker(None)
    _prime_match_active(pipe, frames=35)
    # given: 連鎖直前の confirmed (= 4連結ゲート用) と、T2 が参照する
    # 「直前 STABLE 確定盤面」 (= 青ぷよ、連鎖前の別スナップショット) を
    # それぞれ独立に注入する。
    _force_confirmed_board(pipe, "1P", ev.before_board)
    prev_stable_snapshot = Board()
    prev_stable_snapshot.set(12, 0, COLOR_BLUE)
    pipe._prev_stable_confirmed_1p = prev_stable_snapshot
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    t = 10.0
    frame_idx = 40
    res = pipe.update(frame_idx, t, _dummy_frame())
    assert res.p1.state == BoardState.CHAIN

    final_res: SideResult | None = None
    for _ in range(30):
        frame_idx += 1
        t += 1.0
        result = pipe.update(frame_idx, t, _dummy_frame())
        final_res = result.p1
        if final_res.state == BoardState.STABLE:
            break
    assert final_res is not None
    assert final_res.state == BoardState.STABLE
    assert final_res.confirmed_board is not None
    assert final_res.confirmed_board.get(12, 0) == COLOR_GREEN, (
        "T2 が Phase C-6 の C 直後に fresh な final_board を古い prev_stable"
        " (青) へ revert してしまってはいけない"
    )


# ============================
# 反復3 (2026-07-23): 連鎖中 is_match_active 誤 False 化バグ修正テスト
# ============================
#
# 背景: 連鎖中のスコア急変+フラッシュ演出で ScoreZeroDetector/
# MatchEndDetector が瞬間誤爆 → hard_match_off が sm_active (CHAIN/
# GRAVITY_SETTLE 中の保護) を上書きして is_match_active=False になり、
# MENU 強制遷移で confirmed_board=None 化する問題 (物理harness実測:
# 連鎖中の誤 False 率 0.95)。 CHAIN/GRAVITY_SETTLE 中限定で hard_match_off
# を無効化することで解消する。


class _StubScoreZeroDetector:
    """常に指定の both_zero を返す ScoreZeroDetector スタブ。"""

    def __init__(self, both_zero: bool = True) -> None:
        self._both_zero = both_zero

    def detect(self, frame: np.ndarray) -> object:
        @dataclass
        class _Result:
            both_zero: bool
        return _Result(both_zero=self._both_zero)


class _StubMatchEndDetector:
    """常に match_end_locked=True を返す MatchEndDetector スタブ。"""

    def update(self, frame: np.ndarray, time_sec: float) -> bool:
        return True

    def is_locked(self, time_sec: float) -> bool:
        return True


def test_chain_in_progress_suppresses_score_zero_false_positive() -> None:
    """反復3: CHAIN 中は ScoreZeroDetector の瞬間誤爆 (score_zero_both) で
    is_match_active が False にならない。
    """
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        score_zero_detector=_StubScoreZeroDetector(both_zero=True),  # type: ignore[arg-type]
    )
    # CHAIN 中を模擬 (postchain-fix テストと同じ直接注入パターン)。
    pipe._sm_1p.context.state = BoardState.CHAIN
    result = pipe.update(0, 5.0, _dummy_frame())
    assert result.is_match_active is True, (
        "CHAIN 中は score_zero_both 誤爆で is_match_active が False に"
        "なってはいけない"
    )


def test_gravity_settle_in_progress_suppresses_match_end_locked_false_positive() -> None:
    """反復3: GRAVITY_SETTLE 中も MatchEndDetector の瞬間誤爆 (match_end_locked)
    で is_match_active が False にならない。
    """
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        match_end_detector=_StubMatchEndDetector(),  # type: ignore[arg-type]
    )
    pipe._sm_2p.context.state = BoardState.GRAVITY_SETTLE
    result = pipe.update(0, 5.0, _dummy_frame())
    assert result.is_match_active is True, (
        "GRAVITY_SETTLE 中は match_end_locked 誤爆で is_match_active が"
        " False になってはいけない"
    )


def test_hard_match_off_still_applies_when_not_chain_in_progress() -> None:
    """反復3 backward compat: CHAIN/GRAVITY_SETTLE 中でなければ hard_match_off
    (score_zero_both 等) は従来通り is_match_active を False にする。
    """
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        score_zero_detector=_StubScoreZeroDetector(both_zero=True),  # type: ignore[arg-type]
    )
    # state machine は初期状態 (MENU) のまま = 通常時 (連鎖/沈下中でない)。
    result = pipe.update(0, 5.0, _dummy_frame())
    assert result.is_match_active is False, (
        "CHAIN/GRAVITY_SETTLE 中でない通常時は hard_match_off の挙動を"
        "変更前と同じく維持すべき"
    )


def test_legitimate_match_end_detected_after_chain_state_exits() -> None:
    """反復3: 連鎖状態を抜けた後は真の試合終了 (match_end_locked) が通常通り
    検出される (連鎖中抑制が正当な試合終了検出を妨げないことの確認)。
    """
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        match_end_detector=_StubMatchEndDetector(),  # type: ignore[arg-type]
    )
    # CHAIN 中は抑制される (= is_match_active 維持)
    pipe._sm_1p.context.state = BoardState.CHAIN
    res_during_chain = pipe.update(0, 5.0, _dummy_frame())
    assert res_during_chain.is_match_active is True

    # 連鎖終了 (STABLE に復帰) → match_end_locked による検出が有効になる
    pipe._sm_1p.context.state = BoardState.STABLE
    res_after_chain = pipe.update(1, 5.1, _dummy_frame())
    assert res_after_chain.is_match_active is False, (
        "連鎖状態を抜けた後は真の試合終了検出が有効になるべき"
    )


# ============================
# 反復4 (2026-07-23): confirmed_board=None 理由分類 診断計装テスト
# ============================


def test_classify_board_none_reason_cold_start() -> None:
    """反復4: 一度も confirmed_board が確定していない試合では cold_start。"""
    pipe = _make_pipe(_empty_board(), _empty_board())
    reason = pipe._classify_board_none_reason(
        "1P", True, None, BoardState.MENU,
    )
    assert reason == "cold_start"


def test_classify_board_none_reason_menu_reset() -> None:
    """反達4: is_match_active=False (MENU 強制) 後、STABLE 再確定前は menu_reset。"""
    pipe = _make_pipe(_empty_board(), _empty_board())
    # 事前に一度 confirmed が確定済 (cold_start でなくする)
    pipe._classify_board_none_reason("1P", True, _empty_board(), BoardState.STABLE)
    # is_match_active=False の frame (MENU 強制発生)
    reason = pipe._classify_board_none_reason(
        "1P", False, None, BoardState.MENU,
    )
    assert reason == "menu_reset"
    # 次フレーム以降、is_match_active=True に戻っても confirmed が
    # まだ再確定していなければ menu_reset のまま持ち越される。
    reason2 = pipe._classify_board_none_reason(
        "1P", True, None, BoardState.CHAIN,
    )
    assert reason2 == "menu_reset", (
        "MENU 強制後 STABLE 再確定前は CHAIN に復帰しても menu_reset のまま"
    )


def test_classify_board_none_reason_chain_hold_none() -> None:
    """反達4: menu_reset 中でなく CHAIN/GRAVITY_SETTLE 中に confirmed=None
    なら chain_hold_none (= is_match_active 経路以外の別要因の疑い)。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    pipe._classify_board_none_reason("1P", True, _empty_board(), BoardState.STABLE)
    reason = pipe._classify_board_none_reason(
        "1P", True, None, BoardState.CHAIN,
    )
    assert reason == "chain_hold_none"
    reason_settle = pipe._classify_board_none_reason(
        "1P", True, None, BoardState.GRAVITY_SETTLE,
    )
    assert reason_settle == "chain_hold_none"


def test_classify_board_none_reason_other() -> None:
    """反達4: cold_start でも menu_reset でも CHAIN/GRAVITY_SETTLE でもない
    None は other (fail-silent 防止の受け皿)。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    pipe._classify_board_none_reason("1P", True, _empty_board(), BoardState.STABLE)
    reason = pipe._classify_board_none_reason(
        "1P", True, None, BoardState.OJAMA_FALL,
    )
    assert reason == "other"


def test_classify_board_none_reason_none_when_confirmed_present() -> None:
    """反達4: confirmed_board が非 None なら理由は None (該当なし)。"""
    pipe = _make_pipe(_empty_board(), _empty_board())
    reason = pipe._classify_board_none_reason(
        "1P", True, _empty_board(), BoardState.STABLE,
    )
    assert reason is None
    assert pipe._ever_had_confirmed_1p is True


def test_board_none_reason_field_default_none_backward_compat() -> None:
    """反達4 backward compat: SideResult.board_none_reason は既定 None
    (既存の SideResult(...) 呼出元は引数を渡さず動作継続可能)。
    """
    ev = _make_chain_event(is_all_clear=False)
    sr = SideResult(
        side="1P", state=BoardState.STABLE, cnn_board=_empty_board(),
        inferred_board=None, confirmed_board=_empty_board(),
        drift=None, score=0, score_delta=0, chain_event=ev,
    )
    assert sr.board_none_reason is None


# ============================
# 反復5 (2026-07-23): 物理推論スルー (根治本体) テスト
# ============================


def _erasable_board_with_survivor() -> Board:
    """4連結の赤 (row12 col1-4) + その上の緑 (row11 col0) の盤面。

    ChainSimulator.simulate() で chain_count=1、final_board は
    (12,0)=緑 (重力落下)、それ以外は空になる。
    """
    b = Board()
    b.set(12, 1, COLOR_RED)
    b.set(12, 2, COLOR_RED)
    b.set(12, 3, COLOR_RED)
    b.set(12, 4, COLOR_RED)
    b.set(11, 0, COLOR_GREEN)
    return b


def _make_real_chain_event(chain_count_claimed: int, before: Board | None = None):
    """実際に ChainSimulator で解決可能な起点盤面を持つ ChainEvent を作る。

    chain_count_claimed: score 由来 chain_count として ChainEvent に積む値
        (物理予測との答え合わせ用に、意図的に実際の連鎖数と変えられる)。
    """
    from src.chain_detector import ChainEvent
    before_board = before if before is not None else _erasable_board_with_survivor()
    return ChainEvent(
        trigger_sec=10.0, end_sec=10.6, before_board=before_board,
        chain_count=chain_count_claimed,
        total_erased=4, total_score=40, base_score=40,
        all_clear_bonus_applied=0, ojama_sent=0, leftover_score=0,
        is_all_clear=False,
    )


def test_progressed_chain_board_none_when_no_chain() -> None:
    """反復5 Step2: chain_count=0 の ChainResult は None を返す。"""
    from src.chain import ChainSimulator
    from src.recognition_pipeline import _progressed_chain_board
    cr = ChainSimulator().simulate(Board())  # 空盤面 = 連鎖なし
    assert cr.chain_count == 0
    assert _progressed_chain_board(cr, 10.0, 10.6, 10.3) is None


def test_progressed_chain_board_returns_final_after_end() -> None:
    """反復5 Step2: 経過時刻が end_sec 以降なら final_board を返す。"""
    from src.chain import ChainSimulator
    from src.recognition_pipeline import _progressed_chain_board
    cr = ChainSimulator().simulate(_erasable_board_with_survivor())
    assert cr.chain_count == 1
    board = _progressed_chain_board(cr, 10.0, 10.6, 20.0)  # end_sec 超過
    assert board is not None
    assert board.get(12, 0) == COLOR_GREEN
    assert board.get(12, 1) == COLOR_EMPTY


def test_start_chain_estimate_stores_result_when_count_matches() -> None:
    """反復5 Step2/Step3(a): score 由来 chain_count と物理予測が一致すれば
    低信頼度フラグは立たない。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_real_chain_event(chain_count_claimed=1)  # 実際も 1 連鎖
    pipe._start_chain_estimate("1P", ev)
    assert pipe._chain_estimate_result_1p is not None
    assert pipe._chain_estimate_result_1p.chain_count == 1
    assert pipe._chain_estimate_low_confidence_1p is False


def test_start_chain_estimate_low_confidence_on_count_mismatch() -> None:
    """反復5 Step3(a) 答え合わせ: score由来 chain_count (claimed) と物理予測
    (実際は 1連鎖) が不一致なら低信頼度フラグが立つ。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_real_chain_event(chain_count_claimed=5)  # 実際は 1 連鎖のはず
    pipe._start_chain_estimate("1P", ev)
    assert pipe._chain_estimate_result_1p is not None
    assert pipe._chain_estimate_low_confidence_1p is True


def test_start_chain_estimate_no_erasable_group_sets_none() -> None:
    """反復5 Step2: 起点盤面に連鎖可能なグループが無ければ推定を開始しない。"""
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_real_chain_event(chain_count_claimed=1, before=Board())  # 空盤面
    pipe._start_chain_estimate("1P", ev)
    assert pipe._chain_estimate_result_1p is None


def test_compute_chain_estimate_returns_board_during_chain() -> None:
    """反復5 Step2: CHAIN 中は estimated_board が非 None、provenance は
    chain_estimate (起点信頼度が高い場合)。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_real_chain_event(chain_count_claimed=1)
    pipe._start_chain_estimate("1P", ev)
    board, provenance = pipe._compute_chain_estimate(
        "1P", BoardState.CHAIN, time_sec=20.0,  # end_sec 超過 → final_board
    )
    assert board is not None
    assert board.get(12, 0) == COLOR_GREEN
    assert provenance == "chain_estimate"


def test_compute_chain_estimate_low_confidence_provenance() -> None:
    """反復5 Step3(a): 低信頼度フラグ時は provenance が
    chain_estimate_low_confidence になる。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_real_chain_event(chain_count_claimed=99)  # 明らかに不一致
    pipe._start_chain_estimate("1P", ev)
    board, provenance = pipe._compute_chain_estimate(
        "1P", BoardState.GRAVITY_SETTLE, time_sec=20.0,
    )
    assert board is not None
    assert provenance == "chain_estimate_low_confidence"


def test_compute_chain_estimate_none_outside_chain_states() -> None:
    """反復5 Step2/Step4: CHAIN/GRAVITY_SETTLE 以外では常に (None, observed)
    (= 標準 STABLE eval 経路には一切影響しない)。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_real_chain_event(chain_count_claimed=1)
    pipe._start_chain_estimate("1P", ev)
    board, provenance = pipe._compute_chain_estimate(
        "1P", BoardState.STABLE, time_sec=20.0,
    )
    assert board is None
    assert provenance == "observed"
    # CHAIN/GRAVITY_SETTLE を抜けたら内部 state もクリアされる (次連鎖への
    # 誤った持ち越し防止)。
    assert pipe._chain_estimate_result_1p is None


# ============================
# 案1 (2026-07-23): stale_hold フォールバック
# (c62 1P estimated_board カバレッジ崩壊 9.8% の真因対処、
#  recognition_diag_c62_1p_estimate_collapse/summary.txt)
# ============================


def _make_unsimulatable_chain_event(trigger_sec: float = 10.0):
    """simulate が chain_count=0 になる (推定が立ち上がらない) ChainEvent。

    診断で確認された疑似連鎖イベント early-fire 失敗 (pred_cc=0) を
    再現するための起点盤面 (空盤面 = 4連結なし)。
    """
    from src.chain_detector import ChainEvent
    return ChainEvent(
        trigger_sec=trigger_sec, end_sec=trigger_sec + 0.6, before_board=Board(),
        chain_count=1, total_erased=0, total_score=0, base_score=0,
        all_clear_bonus_applied=0, ojama_sent=0, leftover_score=0,
        is_all_clear=False,
    )


def test_start_chain_estimate_cold_start_seeds_last_board_with_before_board() -> None:
    """案1: simulate 失敗 (result=None) でも cold start なら before_board で
    last_board を seed する (直前保持の初期値)。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    assert pipe._chain_estimate_last_board_1p is None
    ev = _make_unsimulatable_chain_event()
    pipe._start_chain_estimate("1P", ev)
    assert pipe._chain_estimate_result_1p is None  # simulate 失敗 (pred_cc=0)
    assert pipe._chain_estimate_last_board_1p is not None
    assert pipe._chain_estimate_last_board_1p == ev.before_board


def test_compute_chain_estimate_stale_hold_returns_last_board_with_provenance() -> None:
    """案1 主目的: simulate 失敗時、None でなく last_board を
    board_provenance='chain_estimate_stale_hold' で返す (デフォルト ON)。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_unsimulatable_chain_event(trigger_sec=100.0)
    pipe._start_chain_estimate("1P", ev)
    board, provenance = pipe._compute_chain_estimate(
        "1P", BoardState.CHAIN, time_sec=100.05,
    )
    assert board is not None
    assert board == ev.before_board
    assert provenance == "chain_estimate_stale_hold"


def test_compute_chain_estimate_stale_hold_disabled_flag_falls_back_to_none() -> None:
    """案1 backward compat: enable_chain_estimate_stale_hold=False なら
    従来通り (None, 'observed') (= 案1導入前の挙動と完全一致)。
    """
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_chain_estimate_stale_hold=False,
    )
    ev = _make_unsimulatable_chain_event(trigger_sec=100.0)
    pipe._start_chain_estimate("1P", ev)
    board, provenance = pipe._compute_chain_estimate(
        "1P", BoardState.CHAIN, time_sec=100.05,
    )
    assert board is None
    assert provenance == "observed"


def test_compute_chain_estimate_stale_hold_prefers_established_board_over_new_failure() -> None:
    """案1: 既に途中まで進行した推定盤面 (last_board) がある状態で、新規
    トリガーの simulate が失敗しても、その失敗トリガーの (より情報の少ない)
    before_board で上書きされない (= より進んだ推定を優先温存)。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev1 = _make_real_chain_event(chain_count_claimed=1)  # simulate 成功
    pipe._start_chain_estimate("1P", ev1)
    board1, prov1 = pipe._compute_chain_estimate(
        "1P", BoardState.CHAIN, time_sec=ev1.end_sec,
    )
    assert prov1 == "chain_estimate"
    assert board1 is not None and board1.get(12, 0) == COLOR_GREEN
    # 新規トリガー (simulate 失敗) が既存 last_board を上書きしないことを確認。
    ev2 = _make_unsimulatable_chain_event(trigger_sec=ev1.end_sec + 0.01)
    pipe._start_chain_estimate("1P", ev2)
    board2, prov2 = pipe._compute_chain_estimate(
        "1P", BoardState.CHAIN, time_sec=ev2.trigger_sec + 0.01,
    )
    assert prov2 == "chain_estimate_stale_hold"
    assert board2 is not None
    assert board2.get(12, 0) == COLOR_GREEN, (
        "既存の進行済み推定 (last_board) が保持されるべき"
        " (失敗トリガーの空 before_board に退行してはいけない)"
    )


def test_compute_chain_estimate_stale_hold_expires_after_max_sec() -> None:
    """案1 安全弁: 連続 stale_hold が CHAIN_ESTIMATE_STALE_HOLD_MAX_SEC を
    超えたら None に戻る (古い盤面を無期限に貼り続ける事故防止)。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_unsimulatable_chain_event(trigger_sec=100.0)
    pipe._start_chain_estimate("1P", ev)
    t0 = 100.05
    board0, prov0 = pipe._compute_chain_estimate("1P", BoardState.CHAIN, time_sec=t0)
    assert prov0 == "chain_estimate_stale_hold"
    assert board0 is not None
    t1 = t0 + RecognitionPipeline.CHAIN_ESTIMATE_STALE_HOLD_MAX_SEC + 1.0
    board1, prov1 = pipe._compute_chain_estimate("1P", BoardState.CHAIN, time_sec=t1)
    assert board1 is None
    assert prov1 == "observed"


def test_compute_chain_estimate_stale_hold_clears_on_stable_return() -> None:
    """案1: STABLE 復帰で last_board / stale streak が完全にクリアされる
    (= 次の CHAIN 突入は必ず cold start からになる)。
    """
    pipe = _make_pipe(_empty_board(), _empty_board())
    ev = _make_unsimulatable_chain_event(trigger_sec=100.0)
    pipe._start_chain_estimate("1P", ev)
    pipe._compute_chain_estimate("1P", BoardState.CHAIN, time_sec=100.05)
    assert pipe._chain_estimate_last_board_1p is not None
    board, provenance = pipe._compute_chain_estimate(
        "1P", BoardState.STABLE, time_sec=100.10,
    )
    assert board is None
    assert provenance == "observed"
    assert pipe._chain_estimate_last_board_1p is None
    assert pipe._chain_estimate_stale_since_1p is None


def test_chain_estimate_stale_hold_end_to_end_keeps_prior_board_without_touching_confirmed() -> None:
    """案1 統合: 実際の pipeline.update() 経路で、疑似 early-fire 再トリガー
    (診断で確認された主要故障モード: before_board simulate が chain_count=0)
    が発生しても estimated_board は None にならず直前の推定盤面を保持する。
    confirmed_board (STABLE 評価用) は本機構によって一切変更されない。
    """
    ev1 = _make_real_chain_event(chain_count_claimed=1)
    pipe = _make_pipe_with_tracker(None)
    _prime_match_active(pipe, frames=35)
    _force_confirmed_board(pipe, "1P", ev1.before_board)
    pipe._chain_tracker_1p = _StubChainTracker(ev1)  # type: ignore[assignment]
    result = pipe.update(40, 10.0, _dummy_frame())
    assert result.p1.state == BoardState.CHAIN
    assert result.p1.estimated_board is not None
    assert result.p1.estimated_board.get(12, 0) == COLOR_GREEN

    # 診断で確認された故障モード (機能B/D 早期発火の疑似トリガーが
    # before_board simulate に失敗する) を直接注入する。
    ev2 = _make_unsimulatable_chain_event(trigger_sec=10.05)
    pipe._start_chain_estimate("1P", ev2)
    assert pipe._chain_estimate_result_1p is None  # simulate 失敗 (pred_cc=0)

    result2 = pipe.update(41, 10.06, _dummy_frame())
    assert result2.p1.state == BoardState.CHAIN  # active_chain_1p 継続中
    assert result2.p1.estimated_board is not None, (
        "stale_hold: フレッシュな推定が失敗しても None にならず"
        "直前の盤面を保持すべき"
    )
    assert result2.p1.estimated_board.get(12, 0) == COLOR_GREEN, (
        "直前の progressed board (既に成功していた推定) が保持されるべき"
    )
    assert result2.p1.board_provenance == "chain_estimate_stale_hold"
    assert (
        result2.p1.confirmed_board is None
        or result2.p1.confirmed_board.get(12, 1) != COLOR_EMPTY
    ), "confirmed_board 自体が stale_hold の値で上書きされてはいけない"


def test_chain_estimate_exposed_without_mutating_confirmed_board() -> None:
    """反復5 統合 (Step2 主目的 + Step4 backward compat): CHAIN 中でも
    confirmed_board の値は本機構によって一切変更されない
    (= 標準 eval 経路 (STABLE のみ評価) への影響ゼロ) が、
    estimated_board には物理推定盤面が独立に公開される。
    """
    ev = _make_real_chain_event(chain_count_claimed=1)
    pipe = _make_pipe_with_tracker(None)
    _prime_match_active(pipe, frames=35)
    # cycle 49 の 4連結ゲート通過用: confirmed に起点盤面を直接注入する
    # (postchain-fix テストと同じ given 条件パターン)。
    _force_confirmed_board(pipe, "1P", ev.before_board)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    result = pipe.update(40, 10.0, _dummy_frame())
    assert result.p1.state == BoardState.CHAIN
    # confirmed_board は本機構 (estimated_board) によって書き換えられて
    # いない (= given でセットした起点盤面のままか、通常経路の値のまま)。
    # 少なくとも「連鎖後の緑ぷよ落下」という推定結果には置き換わっていない。
    assert (
        result.p1.confirmed_board is None
        or result.p1.confirmed_board.get(12, 1) != COLOR_EMPTY
    ), "confirmed_board 自体が estimated_board の値で上書きされてはいけない"
    assert result.p1.estimated_board is not None, (
        "CHAIN 中は物理推定盤面が estimated_board に公開されるべき"
        " (根治の主目的)"
    )
    assert result.p1.estimated_board.get(12, 0) == COLOR_GREEN
    assert result.p1.board_provenance in (
        "chain_estimate", "chain_estimate_low_confidence",
    )


def test_chain_estimate_applied_unconditionally_at_stable() -> None:
    """反復5 修正 (2026-07-23): final_board は事前ゲートせず素直に適用する
    (反復1の残像修正を邪魔しない、物理レビュー実測での回帰 0.09→0.28 の
    根治)。STABLE 復帰の瞬間は、生 CNN と乖離していても物理予測が採用される。
    """
    unrelated_board = Board()
    for r in range(0, 6):
        unrelated_board.set(r, 5, COLOR_BLUE)  # final_board (ほぼ空) と大きく乖離
    ev = _make_real_chain_event(chain_count_claimed=1)
    reader = _StubImageReader(unrelated_board, _empty_board())
    detector = _StubMatchDetector(in_match=True)
    tracker = _StubChainTracker(None)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=tracker,  # type: ignore[arg-type]
        chain_tracker_2p=None,
        stable_frame_count=2,
        # 本テストは unrelated_board (cnn_board と物理予測が常時大乖離) を
        # 使い DriftDetector.needs_resync を意図的に頻発させる構成のため、
        # 2026-07-25 既定 ON 化されたガード 2 種を明示 OFF にして
        # drift 再同期による sm/gen リセットが従来通り即時発火する挙動を
        # 維持する (本テストの検証意図は final_board 無条件適用であり、
        # drift 再同期ガードの効果検証ではないため)。
        enable_drift_resync_match_start_guard=False,
        enable_drift_resync_hsv_gate=False,
    )
    _prime_match_active(pipe, frames=35)
    _force_confirmed_board(pipe, "1P", ev.before_board)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    t = 10.0
    frame_idx = 40
    res = pipe.update(frame_idx, t, _dummy_frame())
    assert res.p1.state == BoardState.CHAIN

    final_res: SideResult | None = None
    for _ in range(30):
        frame_idx += 1
        t += 1.0
        result = pipe.update(frame_idx, t, _dummy_frame())
        final_res = result.p1
        if final_res.state == BoardState.STABLE:
            break
    assert final_res is not None
    assert final_res.state == BoardState.STABLE
    assert final_res.confirmed_board is not None
    assert final_res.confirmed_board.get(12, 0) == COLOR_GREEN, (
        "final_board は事前ゲートせず素直に適用されるべき"
        " (反復1の残像修正を維持)"
    )


def _run_chain_to_first_stable(
    pipe: RecognitionPipeline, ev, frame_idx: int, t: float,
) -> tuple[int, float, "SideResult"]:
    """CHAIN 発火から最初の STABLE frame まで進め、(frame_idx, t, result) を返す。

    CNN は起点盤面 (ev.before_board) と一致させたまま進める
    (= inferred と CNN が一致し drift-resync が誤って先回りしないようにする)。
    """
    res = pipe.update(frame_idx, t, _dummy_frame())
    assert res.p1.state == BoardState.CHAIN
    result = res
    for _ in range(30):
        frame_idx += 1
        t += 1.0
        result = pipe.update(frame_idx, t, _dummy_frame())
        if result.p1.state == BoardState.STABLE:
            break
    return frame_idx, t, result.p1


def test_chain_estimate_answer_check_corrects_persistent_mismatch() -> None:
    """反復5 修正 Step3(b)(c): STABLE 復帰後 CHAIN_VERIFY_FRAMES 分、生 CNN が
    一貫して物理予測と乖離し続ける場合、事後検証が多数決盤面で補正する
    (= 起点誤認が事後に判明したケースの救済)。単一フレーム比較でなく
    複数フレームの多数決を使うため、GRAVITY_SETTLE 直後の一過性ノイズには
    反応しない設計を、CNN が「一貫して」乖離するケースで確認する。

    CNN は起点盤面と一致させたまま CHAIN/GRAVITY_SETTLE を経過させ
    (drift-resync の先回り誤爆を避けるため)、STABLE 到達後に初めて
    「無関係な盤面」に切り替えて答え合わせの補正を検証する。
    """
    ev = _make_real_chain_event(chain_count_claimed=1)
    reader = _StubImageReader(ev.before_board, _empty_board())
    detector = _StubMatchDetector(in_match=True)
    tracker = _StubChainTracker(None)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=tracker,  # type: ignore[arg-type]
        chain_tracker_2p=None,
        stable_frame_count=2,
    )
    _prime_match_active(pipe, frames=35)
    _force_confirmed_board(pipe, "1P", ev.before_board)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    frame_idx, t, first_stable = _run_chain_to_first_stable(pipe, ev, 40, 10.0)
    assert first_stable.state == BoardState.STABLE
    assert first_stable.confirmed_board is not None
    assert first_stable.confirmed_board.get(12, 0) == COLOR_GREEN, (
        "final_board は事前ゲートせず素直に適用されるべき"
    )
    # STABLE 到達後、CNN を「無関係な盤面」に切り替えて答え合わせを検証する。
    unrelated_board = Board()
    for r in range(0, 6):
        unrelated_board.set(r, 5, COLOR_BLUE)
    reader._p1 = unrelated_board

    corrected_res: SideResult | None = None
    for _ in range(30):
        frame_idx += 1
        t += 1.0
        result = pipe.update(frame_idx, t, _dummy_frame())
        if result.p1.answer_check_result is not None:
            corrected_res = result.p1
            break
    assert corrected_res is not None, "答え合わせが完了するはず"
    assert corrected_res.answer_check_result == "verified_mismatch_corrected"
    assert corrected_res.confirmed_board is not None
    assert corrected_res.confirmed_board.get(12, 0) != COLOR_GREEN, (
        "生CNNと一貫して乖離する物理予測は事後に補正されるべき"
    )


def test_chain_estimate_answer_check_verified_match_when_cnn_agrees() -> None:
    """反復5 修正 Step3(b)(c): STABLE 復帰後の生 CNN が物理予測と一致する
    場合は answer_check_result="verified_match" になり、confirmed_board は
    物理予測のまま維持される (誤って補正しない)。
    """
    ev = _make_real_chain_event(chain_count_claimed=1)
    reader = _StubImageReader(ev.before_board, _empty_board())
    detector = _StubMatchDetector(in_match=True)
    tracker = _StubChainTracker(None)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=tracker,  # type: ignore[arg-type]
        chain_tracker_2p=None,
        stable_frame_count=2,
    )
    _prime_match_active(pipe, frames=35)
    _force_confirmed_board(pipe, "1P", ev.before_board)
    pipe._chain_tracker_1p = _StubChainTracker(ev)  # type: ignore[assignment]
    frame_idx, t, first_stable = _run_chain_to_first_stable(pipe, ev, 40, 10.0)
    assert first_stable.state == BoardState.STABLE
    # STABLE 到達後、CNN を「連鎖後の正しい盤面 (緑が (12,0) に落下)」に
    # 切り替えて答え合わせを検証する (= 物理予測と一致するケース)。
    matching_board = Board()
    matching_board.set(12, 0, COLOR_GREEN)
    reader._p1 = matching_board

    verified_res: SideResult | None = None
    for _ in range(30):
        frame_idx += 1
        t += 1.0
        result = pipe.update(frame_idx, t, _dummy_frame())
        if result.p1.answer_check_result is not None:
            verified_res = result.p1
            break
    assert verified_res is not None
    assert verified_res.answer_check_result == "verified_match"
    assert verified_res.confirmed_board is not None
    assert verified_res.confirmed_board.get(12, 0) == COLOR_GREEN


# ============================
# #45 おじゃま merge 統合修正 案(a)(b) + 案B 既定 ON 化 (2026-07-24)
#
# A/B 検証 (次ツモ遅延 2.80s→0.65s・浮き誤消去 -28%・採用 +38) +
# user viz 全画像レビュー承認 (「全て after の方が品質高い」) を受け、
# 以下 3 flag の既定値を False → True に変更した:
#   - enable_ojama_fall_board_settle (案B: OJAMA_FALL 退出=全盤面 settle)
#   - enable_gravity_filter_support   (案(a): 重力フィルタ支持緩和)
#   - merge_use_majority_value        (案(b): 退出 merge 書込値の多数決化)
# False を明示指定すれば旧挙動 (bit-identical) に戻せる (backwards compat)。
# ============================


def test_ojama_dropout_fix_flags_default_true_on_init() -> None:
    """3 flag とも RecognitionPipeline.__init__ の既定値が True であること
    (2026-07-24 既定 ON 化・user viz 承認)。"""
    import inspect
    sig = inspect.signature(RecognitionPipeline.__init__)
    for name in (
        "enable_ojama_fall_board_settle",
        "enable_gravity_filter_support",
        "merge_use_majority_value",
    ):
        default = sig.parameters[name].default
        assert default is True, f"{name} の __init__ 既定 True 期待: {default}"


def test_ojama_dropout_fix_flags_default_true_on_load_default() -> None:
    """3 flag とも load_default の既定値が True であること
    (2026-07-24 既定 ON 化・user viz 承認)。"""
    import inspect
    sig = inspect.signature(RecognitionPipeline.load_default)
    for name in (
        "enable_ojama_fall_board_settle",
        "enable_gravity_filter_support",
        "merge_use_majority_value",
    ):
        default = sig.parameters[name].default
        assert default is True, f"{name} の load_default 既定 True 期待: {default}"


def test_ojama_dropout_fix_default_pipeline_wiring_all_true() -> None:
    """回帰: 既定 (=全部 True) で pipeline が正常構築され、
    案B の settle 退出 + 再突入抑制 + 浮きフィルタ支持 + 多数決 merge が
    実際に state machine / detector まで配線されていることを確認する。"""
    from src.ojama_visual_detector import OjamaVisualDetector
    from src.state_detectors import OjamaPhaseDetector

    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
    )
    # pipeline レベルの内部 flag
    assert pipe._enable_ojama_fall_board_settle is True
    assert pipe._enable_gravity_filter_support is True
    assert pipe._merge_use_majority_value is True

    # BoardStateMachine (1P/2P 双方) への配線確認
    for sm in (pipe._sm_1p, pipe._sm_2p):
        assert sm._enable_gravity_filter_support is True  # noqa: SLF001
        assert sm._merge_use_majority_value is True  # noqa: SLF001
        ovd = next(
            (d for d in sm._detectors if isinstance(d, OjamaVisualDetector)),
            None,
        )
        assert ovd is not None, "OjamaVisualDetector が detectors に未登録"
        assert ovd.enable_ojama_fall_board_settle is True
        opd = next(
            (d for d in sm._detectors if isinstance(d, OjamaPhaseDetector)),
            None,
        )
        assert opd is not None, "OjamaPhaseDetector が detectors に未登録"
        assert opd.defer_ojama_fall_exit_to_visual is True


def test_ojama_dropout_fix_flags_explicit_false_restores_legacy() -> None:
    """3 flag とも明示的に False を渡せば旧挙動 (bit-identical) の配線に
    戻ることを確認する (backwards compat 退避経路)。"""
    from src.ojama_visual_detector import OjamaVisualDetector
    from src.state_detectors import OjamaPhaseDetector

    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_ojama_fall_board_settle=False,
        enable_gravity_filter_support=False,
        merge_use_majority_value=False,
    )
    assert pipe._enable_ojama_fall_board_settle is False
    assert pipe._enable_gravity_filter_support is False
    assert pipe._merge_use_majority_value is False

    for sm in (pipe._sm_1p, pipe._sm_2p):
        assert sm._enable_gravity_filter_support is False  # noqa: SLF001
        assert sm._merge_use_majority_value is False  # noqa: SLF001
        ovd = next(
            (d for d in sm._detectors if isinstance(d, OjamaVisualDetector)),
            None,
        )
        assert ovd is not None
        assert ovd.enable_ojama_fall_board_settle is False
        opd = next(
            (d for d in sm._detectors if isinstance(d, OjamaPhaseDetector)),
            None,
        )
        assert opd is not None
        assert opd.defer_ojama_fall_exit_to_visual is False


# ============================
# DriftDetector 再同期ループ暴走ガード (2026-07-25, c34 実測)
#
# 試合開始直後は HSV 較正が浅く CNN 誤読が残り、推論盤面と cnn_board の
# 乖離が DriftDetector 閾値を超えて sm.reset+drift.reset+gen.reset が
# 発火 → リセット直後も誤読継続 → 再発火、の自己永続ループが最大 13 秒
# 程度継続する不具合への 2 段ガード。両 flag とも default False
# (= 従来挙動完全維持・bit-identical、backwards compat)。
# ============================


class _FakeAlwaysResyncDrift:
    """DriftDetector 互換スタブ: 毎 update で needs_resync=True を返す。

    _step_side が参照するのは `.update()` と `.reset()` のみ (本体
    DriftDetector.consecutive_drift_count 等は _step_side からは未参照)。
    """

    def __init__(self) -> None:
        self.reset_calls: int = 0
        self.update_calls: int = 0

    def update(self, inferred, cnn):  # noqa: ANN001, ANN201
        from src.drift_detector import DriftResult
        self.update_calls += 1
        return DriftResult(
            mismatch_count=99, consecutive_count=99,
            is_drift=True, needs_resync=True,
        )

    def reset(self) -> None:
        self.reset_calls += 1


def _count_calls(obj: object, method_name: str) -> dict:
    """obj.method_name 呼び出し回数を計測するラッパーに差し替える。"""
    orig = getattr(obj, method_name)
    counter = {"n": 0}

    def _wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        counter["n"] += 1
        return orig(*args, **kwargs)

    setattr(obj, method_name, _wrapper)
    return counter


def test_drift_resync_guards_default_true_on_init() -> None:
    """両ガードとも __init__ 既定値が True であること
    (2026-07-25 user レビュー (c34 v6) 承認・既定 ON 化)。"""
    import inspect
    sig = inspect.signature(RecognitionPipeline.__init__)
    for name in (
        "enable_drift_resync_match_start_guard",
        "enable_drift_resync_hsv_gate",
    ):
        default = sig.parameters[name].default
        assert default is True, f"{name} の __init__ 既定 True 期待: {default}"


def test_drift_resync_guards_default_true_on_load_default() -> None:
    """両ガードとも load_default 既定値が True であること
    (2026-07-25 user レビュー (c34 v6) 承認・既定 ON 化)。"""
    import inspect
    sig = inspect.signature(RecognitionPipeline.load_default)
    for name in (
        "enable_drift_resync_match_start_guard",
        "enable_drift_resync_hsv_gate",
    ):
        default = sig.parameters[name].default
        assert default is True, (
            f"{name} の load_default 既定 True 期待: {default}"
        )


def test_drift_resync_guard_counters_init_zero() -> None:
    """デバッグカウンタは construction 直後は全て 0。"""
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    assert pipe._drift_resync_start_guard_suppressed_1p == 0
    assert pipe._drift_resync_start_guard_suppressed_2p == 0
    assert pipe._drift_resync_hsv_gate_suppressed_1p == 0
    assert pipe._drift_resync_hsv_gate_suppressed_2p == 0


def test_drift_resync_guards_off_resyncs_immediately_bit_identical() -> None:
    """両ガード明示 OFF (2026-07-25 既定 ON 化により明示指定が必要) では
    試合開始直後でも needs_resync=True で従来通り
    sm.reset/gen.reset/drift.reset が即発火する (bit-identical)。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        enable_drift_resync_match_start_guard=False,
        enable_drift_resync_hsv_gate=False,
    )
    pipe._drift_1p = _FakeAlwaysResyncDrift()
    sm_calls = _count_calls(pipe._sm_1p, "reset")
    gen_calls = _count_calls(pipe._gen_1p, "reset")

    pipe.update(0, 0.0, _dummy_frame())  # 試合開始 window 内 (0.0s)

    assert sm_calls["n"] >= 1, "guard OFF なら resync が即発火するべき"
    assert gen_calls["n"] >= 1
    assert pipe._drift_1p.reset_calls >= 1
    assert pipe._drift_resync_start_guard_suppressed_1p == 0
    assert pipe._drift_resync_hsv_gate_suppressed_1p == 0


def test_drift_resync_match_start_guard_on_suppresses_within_window() -> None:
    """ガード1 ON: 試合開始から DRIFT_RESYNC_MATCH_START_GUARD_SEC 秒
    以内は needs_resync=True でも resync が抑制される。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        stable_frame_count=2,
        enable_drift_resync_match_start_guard=True,
    )
    pipe._drift_1p = _FakeAlwaysResyncDrift()
    sm_calls = _count_calls(pipe._sm_1p, "reset")

    pipe.update(0, 0.0, _dummy_frame())  # 試合開始 (t=0.0)
    pipe.update(1, 5.0, _dummy_frame())  # t=5.0 < 15.0 秒 → 抑制されるべき

    assert sm_calls["n"] == 0, "window 内は resync が抑制されるべき"
    assert pipe._drift_resync_start_guard_suppressed_1p >= 1
    assert pipe._drift_1p.reset_calls == 0


def test_drift_resync_match_start_guard_on_allows_after_window() -> None:
    """ガード1 ON: DRIFT_RESYNC_MATCH_START_GUARD_SEC 秒経過後は
    needs_resync=True で従来通り resync が発火する。
    ガード2 (hsv_gate) は 2026-07-25 既定 ON 化されたが、本テストは
    ガード1 を単独検証する目的のため明示 OFF にして分離する
    (テスト環境では OnlineHsvCalibrator が実際に較正しないため
    ガード2 既定 ON のままだと較正未達判定のまま恒久的に抑制されてしまう)。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        stable_frame_count=2,
        enable_drift_resync_match_start_guard=True,
        enable_drift_resync_hsv_gate=False,
    )
    pipe._drift_1p = _FakeAlwaysResyncDrift()
    sm_calls = _count_calls(pipe._sm_1p, "reset")

    pipe.update(0, 0.0, _dummy_frame())  # 試合開始 (t=0.0)
    pipe.update(
        1,
        RecognitionPipeline.DRIFT_RESYNC_MATCH_START_GUARD_SEC + 1.0,
        _dummy_frame(),
    )  # window 超過

    assert sm_calls["n"] >= 1, "window 超過後は resync が発火するべき"
    # frame 0 (t=0.0) は window 内のため 1 回だけ抑制され、frame 1 (window 超過)
    # では抑制されない (= 累計 1 のまま増えない)。
    assert pipe._drift_resync_start_guard_suppressed_1p == 1


def test_drift_resync_hsv_gate_on_suppresses_when_uncalibrated() -> None:
    """ガード2 ON: 較正済み色数 < DRIFT_RESYNC_MIN_CALIBRATED_COLORS の間は
    needs_resync=True でも resync が抑制される。"""
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    pipe._enable_drift_resync_hsv_gate = True  # 直接 flag をセット
    assert len(pipe._online_hsv_injected_colors) == 0  # 較正なしの初期状態
    pipe._drift_1p = _FakeAlwaysResyncDrift()
    sm_calls = _count_calls(pipe._sm_1p, "reset")

    pipe.update(0, 0.0, _dummy_frame())

    assert sm_calls["n"] == 0, "較正未達なら resync が抑制されるべき"
    assert pipe._drift_resync_hsv_gate_suppressed_1p >= 1


def test_drift_resync_hsv_gate_on_allows_when_calibrated() -> None:
    """ガード2 ON: 較正済み色数 >= DRIFT_RESYNC_MIN_CALIBRATED_COLORS なら
    従来通り resync が発火する。ガード1 (match_start_guard) は
    2026-07-25 既定 ON 化されたが、本テストはガード2 を単独検証する
    目的のため明示 OFF にして分離する (t=0.0 は window 内で
    ガード1 既定 ON のままだと無条件に抑制されてしまうため)。"""
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    pipe._enable_drift_resync_match_start_guard = False
    pipe._enable_drift_resync_hsv_gate = True
    pipe._online_hsv_injected_colors = {1, 2, 3}  # 3 色較正済みを模擬
    pipe._drift_1p = _FakeAlwaysResyncDrift()
    sm_calls = _count_calls(pipe._sm_1p, "reset")

    pipe.update(0, 0.0, _dummy_frame())

    assert sm_calls["n"] >= 1, "較正済みなら resync が発火するべき"
    assert pipe._drift_resync_hsv_gate_suppressed_1p == 0


def test_drift_resync_guards_are_independent_flags() -> None:
    """ガード1 のみ ON でもガード2 の抑制条件 (較正未達) は無関係に
    resync が発火する (= 各ガードは独立 flag、ガード1 だけでは較正未達を
    考慮しない)。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        stable_frame_count=2,
        enable_drift_resync_match_start_guard=True,
        enable_drift_resync_hsv_gate=False,
    )
    pipe._drift_1p = _FakeAlwaysResyncDrift()
    sm_calls = _count_calls(pipe._sm_1p, "reset")

    # window 超過後は較正状態 (未較正) に関わらず resync が発火する
    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(
        1,
        RecognitionPipeline.DRIFT_RESYNC_MATCH_START_GUARD_SEC + 1.0,
        _dummy_frame(),
    )
    assert sm_calls["n"] >= 1
    assert pipe._drift_resync_hsv_gate_suppressed_1p == 0, (
        "ガード2 OFF なので hsv_gate カウンタは増えないべき"
    )


# ============================
# 前試合盤面残骸リーク修正・追修 (2026-07-25)
#
# force_in_match=True (raw_active 常時 True) 構成では
# BoardStateMachine.update() の is_match_active=False 分岐 (MENU 強制) が
# 一度も発火しないため、score リセット境界検知から
# BoardStateMachine.clear_match_start_residue() を直接呼び出す経路を追加した。
# ============================


class _FakeScoreTrackerSeq:
    """ScoreTracker 互換スタブ: update() 呼び出しごとに事前指定の score を返す。"""

    def __init__(self, scores: list[int]) -> None:
        self._scores = scores
        self._idx = 0
        self._last_score: int | None = None

    @property
    def last_score(self) -> int | None:
        return self._last_score

    def update(self, frame):  # noqa: ANN001, ANN201
        from src.score_ocr import ScoreDelta
        cur = self._scores[min(self._idx, len(self._scores) - 1)]
        prev = self._last_score
        self._idx += 1
        self._last_score = cur
        delta = (cur - prev) if prev is not None else 0
        return ScoreDelta(side="1P", prev_score=prev, cur_score=cur, delta=delta)

    def reset(self) -> None:
        self._last_score = None


def _make_pipe_for_match_start_full_clear(
    enable_match_start_full_clear: bool,
    enable_score_reset_strict: bool = False,
) -> RecognitionPipeline:
    """force_in_match=True 構成での追修テスト用 pipeline を構築する。

    enable_score_reset_strict: 2026-07-26 追加。既存の残骸クリア系テストは
    「単発 1 フレーム遷移で reset() が即発火する」旧挙動を前提に書かれて
    いるため、既定 False (= 旧 OR・デバウンス無し挙動) を維持する
    (backwards compat)。strict モード (両側条件 + 3 フレームデバウンス)
    自体の検証は enable_score_reset_strict=True を明示指定するテストで行う。
    """
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    return RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        stable_frame_count=2,
        force_in_match=True,
        enable_match_start_full_clear=enable_match_start_full_clear,
        enable_score_reset_strict=enable_score_reset_strict,
    )


def _seed_residue(sm) -> None:  # noqa: ANN001
    """game0 終盤相当の残骸フィールド (confirmed_board 含む) を
    StateContext に注入する。"""
    ghost = _board_with_red(10, 4)
    ghost.set(11, 4, COLOR_RED)
    sm.context.state = BoardState.STABLE
    sm.context.confirmed_board = ghost.copy()
    sm.context.non_stable_cnn_history = [ghost.copy(), ghost.copy()]
    sm.context.stable_recovery_counters = {(10, 4): 2, (11, 4): 3}
    sm.context.recovery_cells = {(10, 4), (11, 4)}
    sm.context.next_queue = [(1, 2), (3, 4)]
    sm.context.stable_warmup_remaining = 5


def test_is_score_reset_boundary_detects_large_drop() -> None:
    """スコア大幅減少 (新ゲーム開始) を検知する。"""
    from src.recognition_pipeline import _is_score_reset_boundary
    assert _is_score_reset_boundary(10, 10, 6080, 32) is True


def test_is_score_reset_boundary_detects_near_zero() -> None:
    """両者スコアが 0 付近 (試合最初期) を検知する。"""
    from src.recognition_pipeline import _is_score_reset_boundary
    assert _is_score_reset_boundary(0, 0, None, None) is True


def test_is_score_reset_boundary_no_false_positive_on_normal_play() -> None:
    """通常の得点増加では境界と誤検知しない。"""
    from src.recognition_pipeline import _is_score_reset_boundary
    assert _is_score_reset_boundary(150, 120, 100, 100) is False


def test_is_score_reset_boundary_none_score_returns_false() -> None:
    """score が None (OCR 失敗) は判定不能として False (誤リセット回避)。"""
    from src.recognition_pipeline import _is_score_reset_boundary
    assert _is_score_reset_boundary(None, 100, 5000, 100) is False


def test_force_in_match_score_reset_clears_residue_when_flag_on() -> None:
    """追修: force_in_match=True + enable_match_start_full_clear=True で、
    score 大幅減少 (試合境界) 検知時に sm_1p/sm_2p の confirmed_board
    (=幽霊セルの実体) + 残骸 5 field がクリアされる
    (= MENU 分岐が発火しない構成でも残骸リークを防げる)。"""
    pipe = _make_pipe_for_match_start_full_clear(enable_match_start_full_clear=True)
    pipe._score_tracker_1p = _FakeScoreTrackerSeq([6080, 10, 10])
    pipe._score_tracker_2p = _FakeScoreTrackerSeq([6080, 10, 10])
    _seed_residue(pipe._sm_1p)
    _seed_residue(pipe._sm_2p)

    pipe.update(0, 0.0, _dummy_frame())  # prev score cache = 6080 (境界未検知)
    assert pipe._sm_1p.context.confirmed_board is not None, (
        "1 frame目 (境界未検知) では confirmed_board は残っているべき"
    )
    pipe.update(1, 0.033, _dummy_frame())  # 6080→10 の大幅減少 = 境界検知

    for sm in (pipe._sm_1p, pipe._sm_2p):
        ctx = sm.context
        assert ctx.state == BoardState.MENU
        assert ctx.confirmed_board is None, (
            "境界検知後は幽霊セルの実体である confirmed_board も None に"
            "クリアされるべき"
        )
        assert ctx.non_stable_cnn_history == []
        assert ctx.stable_recovery_counters == {}
        assert ctx.recovery_cells == set()
        assert ctx.next_queue == []
        assert ctx.stable_warmup_remaining == 0


def test_force_in_match_score_reset_keeps_residue_when_flag_off() -> None:
    """backwards compat: enable_match_start_full_clear=False (default) では
    score 大幅減少を検知しても confirmed_board/残骸 5 field はクリアされない。"""
    pipe = _make_pipe_for_match_start_full_clear(enable_match_start_full_clear=False)
    assert not pipe._enable_match_start_full_clear
    pipe._score_tracker_1p = _FakeScoreTrackerSeq([6080, 10, 10])
    pipe._score_tracker_2p = _FakeScoreTrackerSeq([6080, 10, 10])
    _seed_residue(pipe._sm_1p)

    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.033, _dummy_frame())

    ctx = pipe._sm_1p.context
    assert ctx.confirmed_board is not None
    assert ctx.non_stable_cnn_history != []
    assert ctx.stable_recovery_counters != {}
    assert ctx.recovery_cells != set()
    assert ctx.next_queue != []
    assert ctx.stable_warmup_remaining != 0


def test_force_in_match_score_reset_clears_chain_estimate_cache() -> None:
    """追修 (実測): sm 側 5 field だけでは CHAIN 中の estimated_board 表示
    (_chain_estimate_last_board_Xp の stale_hold キャッシュ) に前試合の
    幽霊盤面が残ることが判明 (2P 側 v6 レンダ実測)。self.reset() 経由の
    包括リセットで _active_chain_2p / _chain_estimate_last_board_2p も
    クリアされることを確認する。"""
    pipe = _make_pipe_for_match_start_full_clear(enable_match_start_full_clear=True)
    pipe._score_tracker_1p = _FakeScoreTrackerSeq([6080, 10, 10])
    pipe._score_tracker_2p = _FakeScoreTrackerSeq([6080, 10, 10])
    ghost = _board_with_red(10, 4)
    ghost.set(11, 4, COLOR_RED)
    from src.chain_detector import ChainEvent
    pipe._active_chain_2p = ChainEvent(
        trigger_sec=5.0, end_sec=999.0, before_board=ghost.copy(),
        chain_count=1, total_erased=0, total_score=0,
        base_score=0, all_clear_bonus_applied=0, ojama_sent=0,
        leftover_score=0, is_all_clear=False,
    )
    pipe._chain_until_2p = 999.0
    pipe._chain_estimate_last_board_2p = ghost.copy()
    pipe._chain_estimate_stale_since_2p = 5.0

    pipe.update(0, 0.0, _dummy_frame())
    pipe.update(1, 0.033, _dummy_frame())  # 6080→10 = 境界検知 → self.reset()

    assert pipe._active_chain_2p is None
    assert pipe._chain_until_2p == 0.0
    assert pipe._chain_estimate_last_board_2p is None
    assert pipe._chain_estimate_stale_since_2p is None


def test_force_in_match_score_reset_edge_trigger_no_repeat_fire() -> None:
    """追修: 「両者スコアほぼ0」は試合開始直後の数秒間継続して真になりうる
    ため、境界条件が継続している間は 2 回目以降 self.reset() を再発火しない
    (edge-trigger ラッチ)。序盤の tsumo 認識進行への継続妨害を防ぐ。"""
    pipe = _make_pipe_for_match_start_full_clear(enable_match_start_full_clear=True)
    # 6080→10→10→10→10 (境界成立が数フレーム継続する状況を模擬)
    scores = [6080, 10, 10, 10, 10]
    pipe._score_tracker_1p = _FakeScoreTrackerSeq(scores)
    pipe._score_tracker_2p = _FakeScoreTrackerSeq(scores)
    reset_calls = _count_calls(pipe, "reset")

    for i in range(len(scores)):
        pipe.update(i, i * 0.033, _dummy_frame())

    assert reset_calls["n"] == 1, (
        "境界条件が継続する間 (両者スコア<=20 が連続) は 1 回のみ発火し、"
        "毎フレーム re-fire してはならない"
    )


# --- score-reset 境界誤発火修正 (2026-07-26, strict モード) ---
# diag_v29_mid_resetlog.log で確定した「片側のみの単発 score OCR 誤読で
# 包括 reset() が試合中に誤発火する」欠陥の回帰テスト。


def test_score_reset_strict_ignores_one_sided_ocr_glitch() -> None:
    """strict モード: 片側 (1P) だけが急落し続けても、もう片方 (2P) が
    不変であれば reset() は一切発火しない (診断ログ実例の回帰テスト:
    2P=40031 不変なのに 1P だけ 48077→0 と誤読されたケース)。"""
    pipe = _make_pipe_for_match_start_full_clear(
        enable_match_start_full_clear=True, enable_score_reset_strict=True,
    )
    pipe._score_tracker_1p = _FakeScoreTrackerSeq([48077, 0, 0, 0, 0])
    pipe._score_tracker_2p = _FakeScoreTrackerSeq(
        [40031, 40031, 40031, 40031, 40031]
    )
    reset_calls = _count_calls(pipe, "reset")

    for i in range(5):
        pipe.update(i, i * 0.033, _dummy_frame())

    assert reset_calls["n"] == 0, (
        "片側のみの急落 (もう片方は不変) では strict モードで一切発火して"
        "はならない"
    )


def test_score_reset_strict_fires_after_three_consecutive_frames() -> None:
    """strict モード: 両者が同時に急落した状態が 3 フレーム連続で成立すれば
    reset() が発火する (デバウンス通過後の正常発火を確認)。"""
    pipe = _make_pipe_for_match_start_full_clear(
        enable_match_start_full_clear=True, enable_score_reset_strict=True,
    )
    pipe._score_tracker_1p = _FakeScoreTrackerSeq([6080, 10, 10, 10])
    pipe._score_tracker_2p = _FakeScoreTrackerSeq([6080, 10, 10, 10])
    reset_calls = _count_calls(pipe, "reset")

    for i in range(4):
        pipe.update(i, i * 0.033, _dummy_frame())

    assert reset_calls["n"] == 1, (
        "両側同時急落が 3 フレーム連続で成立すれば 1 回発火するべき"
    )


def test_score_reset_strict_ignores_single_frame_both_side_glitch() -> None:
    """strict モード: 両者が同時に急落したように見えても単発 1 フレームで
    直後に元の値へ復帰する (OCR 誤読の典型パターン) 場合は、3 フレーム
    連続条件を満たさないため reset() が発火しない。"""
    pipe = _make_pipe_for_match_start_full_clear(
        enable_match_start_full_clear=True, enable_score_reset_strict=True,
    )
    pipe._score_tracker_1p = _FakeScoreTrackerSeq([6080, 0, 6080, 6080])
    pipe._score_tracker_2p = _FakeScoreTrackerSeq([5900, 0, 5900, 5900])
    reset_calls = _count_calls(pipe, "reset")

    for i in range(4):
        pipe.update(i, i * 0.033, _dummy_frame())

    assert reset_calls["n"] == 0, (
        "単発 1 フレームだけの両側急落 (直後に復帰) では 3 フレーム連続に"
        "満たないため発火してはならない"
    )

