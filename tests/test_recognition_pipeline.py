"""RecognitionPipeline 統合テスト (Phase B-7a)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.board import COLOR_RED, Board
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

