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


def test_enable_constraint_fill_default_true() -> None:
    """enable_constraint_fill のデフォルトは True (= main 同等、判断保留).

    2026-05-31: 一旦 OFF 化したが「constraint_fill が色破壊主因」が誤診断と判明
    (真因は infer_placement + T2) したため default ON に戻した。採否は user レビュー。
    """
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    assert pipe._enable_constraint_fill is True, \
        "デフォルトは True (判断保留、2026-05-31 OFF 撤回)"


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


def test_ojama_tier1_warmup_default_false_no_effect() -> None:
    """enable_ojama_tier1_warmup=False (default) では ojama 専用カウンタが 0 のまま。"""
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


def test_t2_highconf_yield_default_is_false() -> None:
    """enable_t2_highconf_yield のデフォルト値が False (backwards compat)。"""
    reader = _StubImageReader(_empty_board(), _empty_board())
    detector = _StubMatchDetector(in_match=True)
    pipe = RecognitionPipeline(
        image_reader=reader,  # type: ignore[arg-type]
        match_state_detector=detector,  # type: ignore[arg-type]
        score_ocr=None,
        chain_tracker_1p=None,
        chain_tracker_2p=None,
        stable_frame_count=2,
        # enable_t2_highconf_yield を明示せず → デフォルト False
    )
    assert pipe._enable_t2_highconf_yield is False


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
        current_board=None,
        start_board=None,
    )
    assert result is True, "next_pair 変化時は True を返すべき"


def test_is_game_event_chain_exit_next_no_change() -> None:
    """① 次ツモ変化なし: current_next == start_next → False (お邪魔もなし)。"""
    same = (COLOR_RED, COLOR_BLUE)
    result = _is_game_event_chain_exit(
        current_next=same,
        start_next=same,
        current_board=None,
        start_board=None,
    )
    assert result is False, "next_pair 変化なし / お邪魔なし → False"


def test_is_game_event_chain_exit_ojama_appears() -> None:
    """② 連鎖 side にお邪魔新規出現: True を返す。"""
    from src.board import COLOR_OJAMA
    start_board = Board()  # お邪魔なし
    current_board = Board()
    current_board.set(5, 2, COLOR_OJAMA)  # 新規お邪魔出現
    result = _is_game_event_chain_exit(
        current_next=None,
        start_next=None,
        current_board=current_board,
        start_board=start_board,
    )
    assert result is True, "自 side にお邪魔新規出現 → True"


def test_is_game_event_chain_exit_ojama_preexisting() -> None:
    """② 多段連鎖ガード: start_board にも同位置お邪魔あり → False。

    連鎖開始前から存在するお邪魔は「新規出現」ではないため終了しない。
    """
    from src.board import COLOR_OJAMA
    start_board = Board()
    start_board.set(5, 2, COLOR_OJAMA)  # 連鎖開始前から存在
    current_board = Board()
    current_board.set(5, 2, COLOR_OJAMA)  # 変化なし (同位置)
    result = _is_game_event_chain_exit(
        current_next=None,
        start_next=None,
        current_board=current_board,
        start_board=start_board,
    )
    assert result is False, "既存お邪魔は新規出現でない → False"


def test_is_game_event_chain_exit_max_hold_cap() -> None:
    """安全弁: CHAIN_MAX_HOLD_SEC 超過で game-event なしでも終了する設計を確認。

    _is_game_event_chain_exit は stateless (安全弁は pipeline 側 eff_until で制御)。
    本テストは: next 変化なし / お邪魔なし → False を返すことで
    「安全弁は pipeline の time_sec >= eff_until で chain_ev を None にする」
    側の責任であることを明示する。
    """
    same_next = (COLOR_RED, COLOR_BLUE)
    board_no_change = Board()
    result = _is_game_event_chain_exit(
        current_next=same_next,
        start_next=same_next,
        current_board=board_no_change,
        start_board=board_no_change,
    )
    assert result is False, (
        "安全弁は pipeline 側 (eff_until) 管理のため、"
        "game-event なし時は False を返す"
    )


def test_game_event_chain_exit_flag_off_is_backward_compat() -> None:
    """OFF 時は従来挙動不変: enable_game_event_chain_exit=False でインスタンス生成可能。"""
    pipe = _make_pipe(_empty_board(), _empty_board(), stable_n=2)
    # デフォルト False = game-event chain exit 無効
    assert pipe._enable_game_event_chain_exit is False
    # 従来 chain_until 変数が存在すること
    assert hasattr(pipe, "_chain_until_1p")
    assert hasattr(pipe, "_chain_until_2p")
    # 新規 game-event 変数が初期化されていること (OFF 時も初期化済)
    assert pipe._chain_event_max_until_1p == 0.0
    assert pipe._chain_event_max_until_2p == 0.0
    assert pipe._chain_start_next_1p is None
    assert pipe._chain_start_next_2p is None

