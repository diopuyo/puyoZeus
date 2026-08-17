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
from src.puyo_core_bridge import (
    NATIVE_AVAILABLE,
    beam_search,
    chain_metrics_after_drops,
    enumerate_and_simulate_placements,
    enumerate_placements,
    max_chain_after_drops_for_boards,
    potential_fire_power_raw_for_boards,
    simulate_after_drops,
    simulate_chain,
    simulate_chain_with_steps,
)

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


def test_exact_score_parity_with_chain_simulator(sample_boards: "list[Board]") -> None:
    """厳密得点 `exact_score` (2026-08-13追加、連結ボーナス反映) が
    `src.chain.ChainSimulator` + `src.scoring.calculate_chain_score` (本番の
    正解経路) と完全一致するか (scripts/mc_counter_estimator.py Rust載せ替え
    タスクの前提となるパリティ)。score_approx (連結ボーナス0近似) とは
    異なる値になり得る点に注意 (連結ボーナスが乗る盤面でのみ差が出る)。
    """
    from src.chain import ChainSimulator
    from src.scoring import calculate_chain_score

    sim = ChainSimulator()
    mismatches: "list[str]" = []
    for i, board in enumerate(sample_boards):
        native_exact = simulate_chain(board, exclude_hidden_row_from_pop=False).exact_score
        py_exact = calculate_chain_score(sim.simulate(board)).total_score
        if native_exact != py_exact:
            mismatches.append(
                f"[{i}] exact_score: native={native_exact} py={py_exact}",
            )

    n = len(sample_boards)
    n_bad = len(mismatches)
    assert not mismatches, (
        f"{n_bad}/{n} 件で ChainSimulator の厳密得点と不一致:\n"
        + "\n".join(mismatches[:20])
    )


def test_simulate_chain_with_steps_parity_with_chain_simulator(
    sample_boards: "list[Board]",
) -> None:
    """`simulate_chain_with_steps` (2026-08-13追加、XII-5 同時消しリッチネス
    native対応用) のステップごと詳細が `src.chain.ChainSimulator` と
    完全一致するか (`simultaneous_pop_richness` が使う
    `len(step.erased_groups)`/`erased_count`/`erased_ojama` と同一土俵)。
    """
    from src.chain import ChainSimulator

    sim = ChainSimulator()
    mismatches: "list[str]" = []
    for i, board in enumerate(sample_boards):
        native_result, native_steps = simulate_chain_with_steps(
            board, exclude_hidden_row_from_pop=False,
        )
        py_result = sim.simulate(board)
        if len(native_steps) != len(py_result.steps):
            mismatches.append(
                f"[{i}] ステップ数不一致: native={len(native_steps)} py={len(py_result.steps)}",
            )
            continue
        if native_result.chain_count != py_result.chain_count:
            mismatches.append(
                f"[{i}] chain_count: native={native_result.chain_count} py={py_result.chain_count}",
            )
        for s_idx, (n_step, py_step) in enumerate(zip(native_steps, py_result.steps)):
            expected_groups = len(py_step.erased_groups)
            expected_colors = len({g.color for g in py_step.erased_groups})
            if n_step.num_groups != expected_groups:
                mismatches.append(
                    f"[{i}] step{s_idx} num_groups: native={n_step.num_groups} py={expected_groups}",
                )
            if n_step.num_colors != expected_colors:
                mismatches.append(
                    f"[{i}] step{s_idx} num_colors: native={n_step.num_colors} py={expected_colors}",
                )
            if n_step.erased_count != py_step.erased_count:
                mismatches.append(
                    f"[{i}] step{s_idx} erased_count: native={n_step.erased_count} "
                    f"py={py_step.erased_count}",
                )
            if n_step.ojama_count != py_step.erased_ojama:
                mismatches.append(
                    f"[{i}] step{s_idx} ojama_count: native={n_step.ojama_count} "
                    f"py={py_step.erased_ojama}",
                )

    n = len(sample_boards)
    n_bad = len(mismatches)
    assert not mismatches, (
        f"{n_bad}件で ChainSimulator のステップ詳細と不一致 (対象{n}件):\n"
        + "\n".join(mismatches[:20])
    )


def test_simulate_after_drops_parity_with_individual_calls(
    sample_boards: "list[Board]",
) -> None:
    """バッチ版 `simulate_after_drops` (2026-08-13追加) が、同じ候補を1件ずつ
    `simulate_chain` で個別に呼んだ結果と完全一致するか
    (`scripts/mc_counter_estimator.py` の native載せ替えタスクの前提)。
    """
    from src.board import BOARD_COLS

    drops = [(col, color) for col in range(BOARD_COLS) for color in (1, 2, 3, 4, 5)]
    subset = sample_boards[:80]
    mismatches: "list[str]" = []
    for i, board in enumerate(subset):
        batch_results = simulate_after_drops(board, drops)
        for (col, color), batch_r in zip(drops, batch_results):
            dropped = _drop_one_reference(board, col, color)
            if dropped is None:
                if batch_r is not None:
                    mismatches.append(f"[{i}] col={col} color={color}: 満杯なのにNoneでない")
                continue
            individual_r = simulate_chain(dropped)
            if batch_r is None:
                mismatches.append(f"[{i}] col={col} color={color}: 置けるはずがNone")
                continue
            if not np.array_equal(batch_r.dropped_board._grid, dropped._grid):
                mismatches.append(f"[{i}] col={col} color={color}: dropped_board不一致")
            if batch_r.chain_result.chain_count != individual_r.chain_count:
                mismatches.append(
                    f"[{i}] col={col} color={color} chain_count不一致: "
                    f"batch={batch_r.chain_result.chain_count} individual={individual_r.chain_count}",
                )
            if batch_r.chain_result.exact_score != individual_r.exact_score:
                mismatches.append(
                    f"[{i}] col={col} color={color} exact_score不一致: "
                    f"batch={batch_r.chain_result.exact_score} individual={individual_r.exact_score}",
                )

    assert not mismatches, (
        f"{len(mismatches)} 件でバッチ版が個別呼び出しと不一致:\n"
        + "\n".join(mismatches[:20])
    )


def test_chain_metrics_after_drops_parity_with_simulate_after_drops(
    sample_boards: "list[Board]",
) -> None:
    """軽量版 `chain_metrics_after_drops` (2026-08-13追加、盤面を返さない版) が
    `simulate_after_drops` の (chain_count, exact_score) と完全一致するか。
    """
    from src.board import BOARD_COLS

    drops = [(col, color) for col in range(BOARD_COLS) for color in (1, 2, 3, 4, 5)]
    subset = sample_boards[:80]
    mismatches: "list[str]" = []
    for i, board in enumerate(subset):
        lean = chain_metrics_after_drops(board, drops)
        full = simulate_after_drops(board, drops)
        for (col, color), lean_r, full_r in zip(drops, lean, full):
            if (lean_r is None) != (full_r is None):
                mismatches.append(f"[{i}] col={col} color={color}: None判定が不一致")
                continue
            if lean_r is None:
                continue
            expected = (full_r.chain_result.chain_count, full_r.chain_result.exact_score)
            if lean_r != expected:
                mismatches.append(
                    f"[{i}] col={col} color={color}: lean={lean_r} full={expected}",
                )

    assert not mismatches, (
        f"{len(mismatches)} 件で軽量版が不一致:\n" + "\n".join(mismatches[:20])
    )


def _drop_one_reference(board: Board, col: int, color: int) -> "Board | None":
    """`simulate_after_drops_parity` テスト専用の1個落下リファレンス実装
    (`src.puyo_core_bridge._drop_row_fallback` と同一仕様、テストの
    独立性を保つため複製)。"""
    height = board.height_of(col)
    if height >= BOARD_ROWS:
        return None
    row = BOARD_ROWS - 1 - height
    work = board.copy()
    work.set(row, col, color)
    return work


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


def test_enumerate_and_simulate_placements_parity_with_individual_calls(
    sample_boards: "list[Board]",
) -> None:
    """バッチ版 `enumerate_and_simulate_placements` (2026-08-13追加、
    `scripts/mc_counter_estimator.py` の境界コスト削減用) が、
    `enumerate_placements` + 配置ごとの `simulate_chain` 個別呼び出しと
    完全一致するか (`v3.2 選択ロジックの境界コスト削減` の前提パリティ)。
    """
    pairs = ((1, 2), (3, 4), (1, 1))
    subset = sample_boards[:120]
    mismatches: "list[str]" = []
    for i, board in enumerate(subset):
        for pair in pairs:
            batch = enumerate_and_simulate_placements(board, pair, filter_dead=False)
            individual = enumerate_placements(board, pair, filter_dead=False)
            if len(batch) != len(individual):
                mismatches.append(
                    f"[{i}] pair={pair}: 件数不一致 batch={len(batch)} individual={len(individual)}",
                )
                continue
            for (col, rotation, placed) in individual:
                matches = [
                    r for r in batch
                    if r.col == col and r.rotation == rotation
                ]
                if len(matches) != 1:
                    mismatches.append(f"[{i}] pair={pair} col={col} rot={rotation}: 対応なし")
                    continue
                r = matches[0]
                if not np.array_equal(r.placed_board._grid, placed._grid):
                    mismatches.append(f"[{i}] pair={pair} col={col} rot={rotation}: placed_board不一致")
                if r.is_dead != placed.is_dead():
                    mismatches.append(f"[{i}] pair={pair} col={col} rot={rotation}: is_dead不一致")
                individual_sim = simulate_chain(placed)
                if r.chain_result.chain_count != individual_sim.chain_count:
                    mismatches.append(
                        f"[{i}] pair={pair} col={col} rot={rotation}: chain_count不一致",
                    )
                if r.chain_result.exact_score != individual_sim.exact_score:
                    mismatches.append(
                        f"[{i}] pair={pair} col={col} rot={rotation}: exact_score不一致",
                    )

    assert not mismatches, (
        f"{len(mismatches)} 件でバッチ版が個別呼び出しと不一致:\n" + "\n".join(mismatches[:20])
    )


def test_max_chain_after_drops_for_boards_parity_with_individual_calls(
    sample_boards: "list[Board]",
) -> None:
    """バッチ版 `max_chain_after_drops_for_boards` (2026-08-13追加、
    `scripts/mc_counter_estimator.py::_select_build_placement` の tie-break
    境界コスト削減用) が、盤面ごとに `chain_metrics_after_drops` を個別
    呼び出しして最大 chain_count を取った場合と完全一致するか。
    """
    drops = [(col, color) for col in range(BOARD_COLS) for color in (1, 2, 3, 4, 5)]
    subset = sample_boards[:80]
    batch_results = max_chain_after_drops_for_boards(subset, drops)
    mismatches: "list[str]" = []
    for i, (board, batch_max) in enumerate(zip(subset, batch_results)):
        metrics = chain_metrics_after_drops(board, drops)
        chain_counts = [m[0] for m in metrics if m is not None]
        expected = max(chain_counts) if chain_counts else 0
        if batch_max != expected:
            mismatches.append(f"[{i}] batch={batch_max} individual={expected}")

    assert not mismatches, (
        f"{len(mismatches)} 件でバッチ版が個別呼び出しと不一致:\n" + "\n".join(mismatches[:20])
    )


def _potential_fire_power_raw_reference(
    board: Board, drops: "list[tuple[int, int]]", beam_k: int,
) -> int:
    """`potential_fire_power_raw_for_boards` の正解基準となる素朴なリファレンス
    実装 (`simulate_after_drops`+`chain_metrics_after_drops` を素直に2段
    ネストして呼ぶだけ、`_pfp_first_pass`/`_pfp_second_pass` と同一意味論)。
    """
    first_pass = [
        (r.chain_result.chain_count, r.dropped_board)
        for r in simulate_after_drops(board, drops)
        if r is not None
    ]
    first_pass.sort(key=lambda x: x[0], reverse=True)
    best_score = 0
    for _chain, board1 in first_pass[:beam_k]:
        for m in chain_metrics_after_drops(board1, drops):
            if m is None:
                continue
            _chain_count, exact_score = m
            if exact_score > best_score:
                best_score = exact_score
    return best_score


def test_potential_fire_power_raw_for_boards_parity_with_reference(
    sample_boards: "list[Board]",
) -> None:
    """バッチ版 `potential_fire_power_raw_for_boards` (2026-08-13追加、
    `scripts/mc_counter_estimator.py::_select_build_placement` の tie-break
    境界コスト削減用) が、素朴な2段ネスト参照実装と完全一致するか。
    """
    drops = [(col, color) for col in range(BOARD_COLS) for color in (1, 2, 3, 4, 5)]
    beam_k = 5
    subset = sample_boards[:40]
    batch_results = potential_fire_power_raw_for_boards(subset, drops, beam_k)
    mismatches: "list[str]" = []
    for i, (board, batch_val) in enumerate(zip(subset, batch_results)):
        expected = _potential_fire_power_raw_reference(board, drops, beam_k)
        if batch_val != expected:
            mismatches.append(f"[{i}] batch={batch_val} reference={expected}")

    assert not mismatches, (
        f"{len(mismatches)} 件でバッチ版が参照実装と不一致:\n" + "\n".join(mismatches[:20])
    )


# ビームサーチが枝刈りされない幅 (22配置 × 22配置 = 484通り以下) で使う値。
# `scripts/_verify_beam_miss_2026-08-09.py` と同じ全探索比較の考え方
# (task指示: 接続可能な形にしておく)。
_EXHAUSTIVE_BEAM_WIDTH: int = 500


def _brute_force_best_score(
    board: Board, pairs: "list[tuple[int, int]]",
) -> int:
    """ツモ列を全探索して到達できる最大 score_approx (running max) を返す (真値)。"""
    frontier = [board]
    best = 0
    for pair in pairs:
        nxt: "list[Board]" = []
        for b in frontier:
            for _col, _rot, placed in enumerate_placements(b, pair, filter_dead=True):
                sim = simulate_chain(placed, exclude_hidden_row_from_pop=True)
                best = max(best, sim.score_approx)
                nxt.append(sim.final_board)
        frontier = nxt
        if not frontier:
            break
    return best


def test_beam_search_matches_brute_force_at_shallow_depth(
    sample_boards: "list[Board]",
) -> None:
    """幅を枝刈りが起きないほど広くすれば、ビームサーチ=全探索の真値と一致するか。

    深さ1 (22通り) ・深さ2 (最大484通り) は全探索が現実的なので、
    `beam_search` (running-max方式) が真値を取りこぼさないことを確認する
    (`beam.rs` のアルゴリズム自体の正しさの検証、simulate/enumerate の
    単体パリティだけでは検出できないバグ — 手順復元・running-max更新の
    ロジック誤り — を捕捉する目的)。
    """
    subset = sample_boards[:30]
    pairs_1 = [(1, 2)]
    pairs_2 = [(1, 2), (3, 4)]
    mismatches: "list[str]" = []
    for i, board in enumerate(subset):
        for pairs in (pairs_1, pairs_2):
            truth = _brute_force_best_score(board, pairs)
            result = beam_search(
                board, pairs, beam_width=_EXHAUSTIVE_BEAM_WIDTH,
                exclude_hidden_row_from_pop=True,
            )
            if result.best_score != truth:
                mismatches.append(
                    f"[{i}] depth={len(pairs)}: beam={result.best_score} "
                    f"truth={truth}",
                )

    assert not mismatches, (
        f"{len(mismatches)} 件でビームサーチが全探索真値と不一致:\n"
        + "\n".join(mismatches[:20])
    )
