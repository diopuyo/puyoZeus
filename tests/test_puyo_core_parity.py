"""puyo_core (Rust ネイティブ拡張) と src/chain_bitboard.py のパリティテスト。

user確定指示 (2026-08-12): ビームサーチ Rust 拡張は
`src/chain_bitboard.py` とビット一致のパリティが正解基準。
実盤面サンプル (data/indicators_v2/boards_lean_phase_l_2026-08-11/*.npz、
複数動画から500盤面以上) で連鎖数・消去数・スコア・最終盤面の完全一致を確認する。

拡張が未ビルドの環境ではフォールバック実装同士の自明な一致になってしまうため、
`NATIVE_AVAILABLE` が False の場合は skip する (拡張の正当性を検証する目的の
テストであり、フォールバックのテストではない)。

設置列挙は `_enumerate_placement_boards` (src/indicators_v2.py:3389) と
盤面一致を確認する (task 指示。indicators_v2.py は読み取りのみ、編集しない)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN, Board
from src.chain_bitboard import (
    batch_from_boards,
    planes_to_board,
    simulate_batch_with_approx_score,
)
from src.puyo_core_bridge import NATIVE_AVAILABLE, enumerate_placements, simulate_chain

_DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-11"
)
# 複数動画から広くサンプルするための対象本数上限 (task指示: 500盤面以上)。
_TARGET_TOTAL_BOARDS: int = 600
_VIDEOS_TO_SAMPLE: int = 12
_BOARDS_PER_VIDEO: int = 60
_RNG_SEED: int = 20260812

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE, reason="puyo_core ネイティブ拡張が未ビルド (maturin develop 要)",
)


def _load_sample_boards() -> "list[Board]":
    """複数 npz から盤面をサンプルして Board リストを返す (UNKNOWN含む盤面は除外)。"""
    npz_files = sorted(_DATA_DIR.glob("*.npz"))
    if not npz_files:
        pytest.skip(f"評価データが見つからない: {_DATA_DIR}")
    rng = np.random.RandomState(_RNG_SEED)
    chosen_files = rng.choice(
        npz_files, size=min(_VIDEOS_TO_SAMPLE, len(npz_files)), replace=False,
    )
    boards: "list[Board]" = []
    for path in chosen_files:
        data = np.load(path, allow_pickle=True)
        grids = data["grids"]
        n = grids.shape[0]
        if n == 0:
            continue
        idxs = rng.choice(n, size=min(_BOARDS_PER_VIDEO, n), replace=False)
        for i in idxs:
            grid = grids[i].astype(np.uint8)
            if np.any(grid == COLOR_UNKNOWN):
                continue  # UNKNOWN はエンジンに入れない仕様 (task指示)
            board = Board()
            board._grid = grid
            boards.append(board)
        if len(boards) >= _TARGET_TOTAL_BOARDS:
            break
    return boards


@pytest.fixture(scope="module")
def sample_boards() -> "list[Board]":
    boards = _load_sample_boards()
    if len(boards) < 500:
        pytest.skip(
            f"実盤面サンプルが500未満 ({len(boards)}件)。データ不足のためスキップ",
        )
    return boards


def test_simulate_chain_parity_with_chain_bitboard(sample_boards: "list[Board]") -> None:
    """連鎖数・消去数・お邪魔数・近似スコア・最終盤面が完全一致するか。"""
    mismatches: "list[str]" = []
    for i, board in enumerate(sample_boards):
        native = simulate_chain(board, exclude_hidden_row_from_pop=False)

        planes = batch_from_boards([board])
        py_result = simulate_batch_with_approx_score(planes)[0]

        if native.chain_count != py_result.chain_count:
            mismatches.append(
                f"[{i}] chain_count: native={native.chain_count} py={py_result.chain_count}",
            )
            continue
        if native.total_erased != py_result.total_erased:
            mismatches.append(
                f"[{i}] total_erased: native={native.total_erased} py={py_result.total_erased}",
            )
        if native.total_ojama != py_result.total_ojama:
            mismatches.append(
                f"[{i}] total_ojama: native={native.total_ojama} py={py_result.total_ojama}",
            )
        if native.score_approx != py_result.score_approx:
            mismatches.append(
                f"[{i}] score_approx: native={native.score_approx} py={py_result.score_approx}",
            )
        py_final_board = planes_to_board(py_result.final_planes)
        if not np.array_equal(native.final_board._grid, py_final_board._grid):
            mismatches.append(f"[{i}] final_board grid mismatch")

    n = len(sample_boards)
    n_bad = len(mismatches)
    assert not mismatches, (
        f"{n_bad}/{n} 件で chain_bitboard と不一致:\n" + "\n".join(mismatches[:20])
    )


def test_simulate_chain_parity_ghost_rule(sample_boards: "list[Board]") -> None:
    """幽霊連鎖ルール (exclude_hidden_row_from_pop=True) も simulate_batch とパリティ確認。

    近似得点版 (simulate_batch_with_approx_score) は幽霊連鎖ルール未対応
    (src/chain_bitboard.py 時点の実装、2026-07-29のまま) のため、厳密版
    simulate_batch (連鎖数・消去数のみ) と比較する (score_approx は比較しない)。
    """
    from src.chain_bitboard import simulate_batch

    mismatches: "list[str]" = []
    for i, board in enumerate(sample_boards):
        native = simulate_chain(board, exclude_hidden_row_from_pop=True)
        planes = batch_from_boards([board])
        py_result = simulate_batch(planes, exclude_hidden_row_from_pop=True)[0]

        if native.chain_count != py_result.chain_count:
            mismatches.append(
                f"[{i}] chain_count: native={native.chain_count} py={py_result.chain_count}",
            )
        if native.total_erased != py_result.total_erased:
            mismatches.append(
                f"[{i}] total_erased: native={native.total_erased} py={py_result.total_erased}",
            )
        if native.total_ojama != py_result.total_ojama:
            mismatches.append(
                f"[{i}] total_ojama: native={native.total_ojama} py={py_result.total_ojama}",
            )

    n = len(sample_boards)
    n_bad = len(mismatches)
    assert not mismatches, (
        f"{n_bad}/{n} 件で幽霊連鎖ルール下の chain_bitboard と不一致:\n"
        + "\n".join(mismatches[:20])
    )


def test_enumerate_placements_parity_with_indicators_v2(
    sample_boards: "list[Board]",
) -> None:
    """設置列挙が `_enumerate_placement_boards` (indicators_v2.py:3389) と一致するか。"""
    import src.indicators_v2 as iv

    pairs = ((1, 2), (3, 4), (1, 1))
    # 全盤面 x 全ペアは高コストなので先頭サブセットのみ (時間予算優先)。
    subset = sample_boards[:120]
    mismatches: "list[str]" = []
    for i, board in enumerate(subset):
        for pair in pairs:
            native_results = enumerate_placements(board, pair, filter_dead=True)
            native_boards = sorted(
                (b._grid.tobytes() for _c, _r, b in native_results),
            )
            py_boards = sorted(
                b._grid.tobytes() for b in iv._enumerate_placement_boards(board, pair)
            )
            if native_boards != py_boards:
                mismatches.append(
                    f"[{i}] pair={pair}: native={len(native_boards)}件 "
                    f"py={len(py_boards)}件 (盤面集合不一致)",
                )

    assert not mismatches, (
        f"{len(mismatches)} 件で設置列挙が不一致:\n" + "\n".join(mismatches[:20])
    )
