"""Rust ネイティブ拡張 `puyo_core` への薄い Python ブリッジ (2026-08-12 新規)。

user確定指示 (2026-08-12): リアルタイム応手探索 (深さ13〜16手/幅250のビーム
サーチを1探索100ms以内) を実現するための PyO3 拡張 `native/puyo_core/`。

**backwards compat 原則 (CLAUDE.md)**: 拡張が未ビルドの環境でも本モジュールの
import 自体は失敗しない (`try/except ImportError` で optional import)。
拡張が無い環境向けに Python フォールバック実装も提供する (低速だが動作する。
`src/chain_bitboard.py` の numpy バッチ処理を単発呼び出しで使う)。

**正解基準**: 拡張・フォールバックともに `src/chain_bitboard.py` とビット
一致のパリティを取る (`tests/test_puyo_core_parity.py` で担保)。

**stateless 実装原則**: 本モジュールの関数はすべて引数を破壊せず、内部状態を
持たない (state-holding が必要な場合は呼び出し側の外部 wrapper が持つ)。

## 2026-08-13 追加 (scripts/mc_counter_estimator.py Rust載せ替えタスク)

- `exact_score`: `ChainSimResult` に厳密得点 (連結ボーナス反映、
  `src.scoring.calculate_chain_score` と同一土俵) を追加。既存
  `score_approx` (連結ボーナス0近似) では値が異なる用途向け。
- `simulate_after_drops`/`chain_metrics_after_drops`: 1個ぷよを複数パターン
  落として連鎖シミュレートするバッチAPI (takapt定石探索・潜在火力ビーム
  探索の高速化用、Python<->Rust 境界を跨ぐ変換コストを候補数で償却)。
- `simulate_chain_with_steps`: `simulate_chain` の拡張版、ステップごとの
  同時消し詳細 (`ChainStepDetail`) も返す。XII-5 同時消しリッチネス
  (`src.indicators_v2.simultaneous_pop_richness`) の native 対応用
  (2026-08-13 追加、既存 `simulate_chain` は無変更)。
- `enumerate_and_simulate_placements`: `enumerate_placements` + 配置ごとの
  `simulate_chain` 個別呼び出しを1回のバッチ呼び出しに統合した結果型。
  `scripts/mc_counter_estimator.py` の `_select_best_placement`/
  `_select_build_placement` 用 (Python<->Rust 境界の往復回数削減、
  2026-08-13 追加、既存2関数は無変更)。
- `max_chain_after_drops_for_boards`: 複数盤面それぞれの最大連鎖数
  (列×色候補を試した上限) を1回のバッチ呼び出しで返す。
  `_select_build_placement` の tie-break (候補盤面ごとの
  `_current_max_chain_value` 個別呼び出し) を1回に統合するための API
  (2026-08-13 追加)。
- `potential_fire_power_raw_for_boards`: 複数盤面それぞれの
  potential_fire_power (2手先ビーム) 生値を1回のバッチ呼び出しで返す。
  `_select_build_placement` の tie-break で「潜在連鎖が同値タイの候補
  全件」に個別呼び出しされていた `_potential_fire_power_value`
  (実測: tied件数中央値10・最大22件、往復の主要コストと判明) を1回に
  統合するための API (2026-08-13 追加)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, Board

try:
    import puyo_core as _native  # type: ignore[import-not-found]
    NATIVE_AVAILABLE: bool = True
except ImportError:
    _native = None
    NATIVE_AVAILABLE = False


# ============================
# 配置列挙定数 (`src/indicators_v2.py::_enumerate_placements` 準拠、
# マジックナンバー禁止のため名前付き定数として複製する。既存関数は無変更)
# ============================
_ROTATION_VERTICAL_TOP_UP: int = 0
_ROTATION_HORIZONTAL_TOP_LEFT: int = 1
_ROTATION_VERTICAL_BOT_UP: int = 2
_ROTATION_HORIZONTAL_BOT_LEFT: int = 3
_NUM_ROTATIONS: int = 4


@dataclass(frozen=True)
class ChainSimResult:
    """1 盤面の連鎖シミュレーション結果 (拡張/フォールバック共通の返り値型)。

    Attributes:
        chain_count: 総連鎖数。
        total_erased: 通常ぷよ消去数合計。
        total_ojama: お邪魔消去数合計。
        score_approx: 近似得点 (連結ボーナス0近似、
            `chain_bitboard.BitboardChainScoreResult.score_approx` と同一土俵)。
        exact_score: 厳密得点 (連結ボーナス反映、`src/scoring.py::
            calculate_chain_score` と同一土俵。2026-08-13 追加、既存フィールドは
            無変更のため backwards compat)。
        final_board: 連鎖終了後の盤面。
    """
    chain_count: int
    total_erased: int
    total_ojama: int
    score_approx: int
    exact_score: int
    final_board: Board


@dataclass(frozen=True)
class DropSimResult:
    """`simulate_after_drops` の1候補分の結果 (2026-08-13 追加)。

    Attributes:
        dropped_board: 落下直後 (連鎖解決前) の盤面。呼び出し側の2手目探索
            (`scripts/mc_counter_estimator.py` 潜在火力ビーム等) が
            「未解決のまま次の1個を積む」という既存意味論を維持するために
            必要 (`chain_result.final_board` [連鎖解決後] とは異なる)。
        chain_result: dropped_board を連鎖シミュレートした結果。
    """
    dropped_board: Board
    chain_result: ChainSimResult


@dataclass(frozen=True)
class BeamSearchResult:
    """ビームサーチ結果 (拡張/フォールバック共通の返り値型)。

    Attributes:
        best_score: 探索中に発火した最大スコア (running max)。
        best_path: 最良手順 `[(col, rotation), ...]` (深さ順)。
        best_score_per_depth: 深さごとの running-max best スコア配列。
    """
    best_score: int
    best_path: "list[tuple[int, int]]"
    best_score_per_depth: "list[int]"


def _grid_to_flat_list(board: Board) -> "list[int]":
    """Board -> flatten した長さ78の色コードリスト (行優先)。"""
    return board._grid.astype(np.uint8).flatten().tolist()


def _flat_list_to_board(flat: "list[int]") -> Board:
    """flatten した長さ78の色コードリスト -> Board。"""
    grid = np.array(flat, dtype=np.uint8).reshape(BOARD_ROWS, BOARD_COLS)
    board = Board()
    board._grid = grid
    return board


# ============================
# 連鎖シミュレーション
# ============================


def simulate_chain(
    board: Board, exclude_hidden_row_from_pop: bool = False,
) -> ChainSimResult:
    """1 盤面を連鎖シミュレートする (拡張があれば使用、無ければ Python フォールバック)。

    Args:
        board: 判定対象の盤面 (破壊しない)。
        exclude_hidden_row_from_pop: 幽霊連鎖ルール (既定 False = 従来挙動、
            本番採用値は `src.production_config.GHOST_CHAIN_RULE_ENABLED`
            を呼び出し側が明示的に渡すこと。backwards compat のため本関数
            自体の既定値は変更しない)。
    """
    if NATIVE_AVAILABLE:
        flat = _grid_to_flat_list(board)
        r = _native.simulate_chain_py(flat, exclude_hidden_row_from_pop)
        return ChainSimResult(
            chain_count=r.chain_count,
            total_erased=r.total_erased,
            total_ojama=r.total_ojama,
            score_approx=r.score_approx,
            exact_score=r.exact_score,
            final_board=_flat_list_to_board(r.final_grid),
        )
    return _simulate_chain_fallback(board, exclude_hidden_row_from_pop)


def _simulate_chain_fallback(
    board: Board, exclude_hidden_row_from_pop: bool,
) -> ChainSimResult:
    """拡張未導入環境向け Python フォールバック (`src.chain_bitboard` 使用)。"""
    from src.chain_bitboard import (
        batch_from_boards,
        planes_to_board,
        simulate_batch_with_approx_score,
    )
    planes = batch_from_boards([board])
    result = simulate_batch_with_approx_score(planes)[0]
    # simulate_batch_with_approx_score は exclude_hidden_row_from_pop 未対応
    # (2026-07-29 時点の実装、simulate_batch のみ対応)。フォールバック経路で
    # 幽霊連鎖ルールを要求された場合は明示的に未対応として例外を出す
    # (無言で挙動が変わる=誤判定の温床になるため、CLAUDE.md fail-silent 禁止)。
    if exclude_hidden_row_from_pop:
        raise NotImplementedError(
            "Python フォールバックは exclude_hidden_row_from_pop=True 未対応です。"
            "native puyo_core 拡張をビルドしてください "
            "(native/puyo_core/, maturin develop)。"
        )
    # exact_score (連結ボーナス反映) は chain_bitboard 側に相当計算が無いため、
    # フォールバック限定で ChainSimulator (低速だが正解) を再実行して求める
    # (フォールバック自体が「拡張未導入時の低速経路」である前提であり、
    # 追加の再計算コストは許容する設計判断)。
    from src.chain import ChainSimulator as _FallbackSimulator
    from src.scoring import calculate_chain_score as _fallback_calculate_score
    exact_score = _fallback_calculate_score(
        _FallbackSimulator().simulate(board),
    ).total_score
    return ChainSimResult(
        chain_count=result.chain_count,
        total_erased=result.total_erased,
        total_ojama=result.total_ojama,
        score_approx=result.score_approx,
        exact_score=exact_score,
        final_board=planes_to_board(result.final_planes),
    )


@dataclass(frozen=True)
class ChainStepDetail:
    """1 ステップ (1 消し) の同時消し詳細 (2026-08-13 追加)。

    `src.chain.ChainStep` と同一の意味論 (`num_groups`=`len(erased_groups)`、
    `num_colors`=`erased_groups` 内の異なり色数)。XII-5 同時消しリッチネス
    (`src.indicators_v2.simultaneous_pop_richness`) の native 対応用。

    Attributes:
        num_groups: このステップで同時に消えたグループ数 (色をまたいで合計)。
        num_colors: このステップで消えた色の種類数。
        erased_count: このステップで消えた通常ぷよ数 (おじゃま除く)。
        ojama_count: このステップで消えたおじゃま数。
    """
    num_groups: int
    num_colors: int
    erased_count: int
    ojama_count: int


def simulate_chain_with_steps(
    board: Board, exclude_hidden_row_from_pop: bool = False,
) -> "tuple[ChainSimResult, list[ChainStepDetail]]":
    """`simulate_chain` の拡張版: 連鎖シミュレーション結果に加え、ステップごと
    の同時消し詳細 (`ChainStepDetail`) も返す (2026-08-13 追加)。

    XII-5 同時消しリッチネス (`src.indicators_v2.simultaneous_pop_richness`)
    が必要とする「各ステップの同時消しグループ数・色数・消去個数」を、
    連鎖シミュレーションを1回実行するだけで得るための拡張版
    (既存 `simulate_chain`/`ChainSimResult` は無変更)。

    Args:
        board: 判定対象の盤面 (破壊しない)。
        exclude_hidden_row_from_pop: 幽霊連鎖ルール (既定 False、
            backwards compat。`simulate_chain` と同じ既定値方針)。

    Returns:
        `(ChainSimResult, ChainStepDetail のリスト)`。`ChainStepDetail`
        のリストは `chain_count` と同じ長さ (ステップ順)。値は
        `src.chain.ChainSimulator.simulate(board).steps` の
        `erased_groups`/`erased_count`/`erased_ojama` と完全一致する
        (`tests/test_puyo_core_parity.py` で確認)。
    """
    if NATIVE_AVAILABLE:
        flat = _grid_to_flat_list(board)
        r, steps_raw = _native.simulate_chain_with_steps_py(flat, exclude_hidden_row_from_pop)
        result = ChainSimResult(
            chain_count=r.chain_count,
            total_erased=r.total_erased,
            total_ojama=r.total_ojama,
            score_approx=r.score_approx,
            exact_score=r.exact_score,
            final_board=_flat_list_to_board(r.final_grid),
        )
        steps = [
            ChainStepDetail(
                num_groups=s.num_groups, num_colors=s.num_colors,
                erased_count=s.erased_count, ojama_count=s.ojama_count,
            )
            for s in steps_raw
        ]
        return result, steps
    return _simulate_chain_with_steps_fallback(board, exclude_hidden_row_from_pop)


def _simulate_chain_with_steps_fallback(
    board: Board, exclude_hidden_row_from_pop: bool,
) -> "tuple[ChainSimResult, list[ChainStepDetail]]":
    """`simulate_chain_with_steps` の Python フォールバック。

    `_simulate_chain_fallback` (既存) で通常の連鎖シミュレーション結果を
    得た上で、ステップ詳細のみ `src.chain.ChainSimulator` を再実行して求める
    (`_simulate_chain_fallback` 自体も内部で ChainSimulator を1回実行して
    いるため2回実行になるが、フォールバック経路 [低速だが正解] 限定の
    冗長性として許容する)。exclude_hidden_row_from_pop=True の未対応は
    `_simulate_chain_fallback` 側の例外にそのまま委ねる (fail-silent回避)。
    """
    result = _simulate_chain_fallback(board, exclude_hidden_row_from_pop)
    from src.chain import ChainSimulator as _FallbackSimulator
    chain_result = _FallbackSimulator().simulate(board)
    steps = [
        ChainStepDetail(
            num_groups=len(step.erased_groups),
            num_colors=len({g.color for g in step.erased_groups}),
            erased_count=step.erased_count,
            ojama_count=step.erased_ojama,
        )
        for step in chain_result.steps
    ]
    return result, steps


def simulate_after_drops(
    board: Board,
    drops: "list[tuple[int, int]]",
    exclude_hidden_row_from_pop: bool = False,
) -> "list[DropSimResult | None]":
    """1 個ぷよを複数パターン (col, color) 落として、それぞれ連鎖シミュレート
    した結果をまとめて返す (2026-08-13 追加)。

    `scripts/mc_counter_estimator.py` の takapt定石探索 (列×色30通り) や
    潜在火力ビーム探索のように「同一の基準盤面に対して多数の1手先候補を
    評価する」場面向け。1 回の呼び出しで複数候補をまとめて処理することで
    Python<->Rust 境界を跨ぐ変換コスト (グリッド flatten/reshape 等) を
    候補数で償却する (`simulate_chain` を候補ごとに個別呼び出しするより
    高速、ただし値は完全に同一になる設計、意味論保存)。

    Args:
        board: 落とす前の基準盤面 (破壊しない)。
        drops: `[(col, color), ...]` の候補リスト。
        exclude_hidden_row_from_pop: 幽霊連鎖ルール (既定 False)。

    Returns:
        `drops` と同じ長さの `DropSimResult | None` リスト
        (None=その列が満杯で置けなかった候補、`simulate_chain` 単体呼び出し
        との完全一致は `tests/test_puyo_core_parity.py` で確認)。
    """
    if NATIVE_AVAILABLE:
        flat = _grid_to_flat_list(board)
        raw = _native.simulate_after_drops_py(flat, drops, exclude_hidden_row_from_pop)
        return [
            None if r is None else DropSimResult(
                dropped_board=_flat_list_to_board(r.dropped_grid),
                chain_result=ChainSimResult(
                    chain_count=r.chain_count,
                    total_erased=r.total_erased,
                    total_ojama=r.total_ojama,
                    score_approx=r.score_approx,
                    exact_score=r.exact_score,
                    final_board=_flat_list_to_board(r.final_grid),
                ),
            )
            for r in raw
        ]
    return [
        _drop_one_fallback_then_simulate(board, col, color, exclude_hidden_row_from_pop)
        for col, color in drops
    ]


def chain_metrics_after_drops(
    board: Board,
    drops: "list[tuple[int, int]]",
    exclude_hidden_row_from_pop: bool = False,
) -> "list[tuple[int, int] | None]":
    """`simulate_after_drops` の軽量版: 盤面 (grid) を一切返さず
    `(chain_count, exact_score)` のみ返す (2026-08-13 追加)。

    呼び出し側が盤面そのものを必要としない場面
    (`scripts/mc_counter_estimator.py` の current_max_chain 相当・
    potential_fire_power 2手目のお邪魔換算等) 向けの速度最適化
    (グリッドの Python<->Rust シリアライズコストを完全に無くす)。

    Returns:
        `drops` と同じ長さの `(chain_count, exact_score) | None` リスト
        (None=その列が満杯で置けなかった候補)。値は `simulate_after_drops`
        の対応要素と完全一致する (`tests/test_puyo_core_parity.py` で確認)。
    """
    if NATIVE_AVAILABLE:
        flat = _grid_to_flat_list(board)
        raw = _native.chain_metrics_after_drops_py(flat, drops, exclude_hidden_row_from_pop)
        return list(raw)
    results = simulate_after_drops(board, drops, exclude_hidden_row_from_pop)
    return [
        None if r is None else (r.chain_result.chain_count, r.chain_result.exact_score)
        for r in results
    ]


def max_chain_after_drops_for_boards(
    boards: "list[Board]",
    drops: "list[tuple[int, int]]",
    exclude_hidden_row_from_pop: bool = False,
) -> "list[int]":
    """複数盤面それぞれに対し、列×色候補 (`drops`) を試して到達できる最大
    連鎖数を返す (2026-08-13 追加)。

    `scripts/mc_counter_estimator.py` の `_select_build_placement` の
    tie-break (候補盤面ごとに `chain_metrics_after_drops` を個別呼び出しして
    最大連鎖数を取る) を1回のバッチ呼び出しに統合するための API
    (候補数分の Python<->Rust 往復を1回に削減)。

    Args:
        boards: 評価対象の盤面リスト (各破壊しない)。
        drops: `[(col, color), ...]` の候補リスト (通常は列×色30通り)。
        exclude_hidden_row_from_pop: 幽霊連鎖ルール (既定 False)。

    Returns:
        `boards` と同じ長さの最大連鎖数リスト (置ける候補が1つも無い盤面は0)。
        値は `chain_metrics_after_drops` を盤面ごとに個別呼び出しして
        `chain_count` の最大を取った場合と完全一致する
        (`tests/test_puyo_core_parity.py` で確認)。
    """
    if NATIVE_AVAILABLE:
        flats = [_grid_to_flat_list(b) for b in boards]
        raw = _native.max_chain_after_drops_for_boards_py(
            flats, drops, exclude_hidden_row_from_pop,
        )
        return list(raw)
    results: "list[int]" = []
    for board in boards:
        metrics = chain_metrics_after_drops(board, drops, exclude_hidden_row_from_pop)
        chain_counts = [m[0] for m in metrics if m is not None]
        results.append(max(chain_counts) if chain_counts else 0)
    return results


def potential_fire_power_raw_for_boards(
    boards: "list[Board]",
    drops: "list[tuple[int, int]]",
    beam_k: int,
    exclude_hidden_row_from_pop: bool = False,
) -> "list[int]":
    """複数盤面それぞれについて、potential_fire_power (2手先ビーム beam_k)
    の生値 (お邪魔換算前の最良2手先 exact_score) をまとめて返す
    (2026-08-13 追加)。

    `scripts/mc_counter_estimator.py::_select_build_placement` の tie-break
    (潜在連鎖が同値タイの候補全件に対して1件ずつ `_potential_fire_power_value`
    を呼ぶ、実測で tied 件数は中央値10件・最大22件) を1回のバッチ呼び出しに
    統合するための API (候補数分の Python<->Rust 往復を1回に削減)。

    アルゴリズム (`_pfp_first_pass`/`_pfp_second_pass` と同一意味論):
        1手目: `drops` を試し、シミュレート後 chain_count 降順で上位
            `beam_k` 件の「落下直後 (連鎖解決前)」盤面を残す。
        2手目: 残した各盤面にさらに `drops` を試し、到達できる最大
            `exact_score` を求める。

    Args:
        boards: 評価対象の盤面リスト (各破壊しない)。
        drops: `[(col, color), ...]` の候補リスト (通常は列×色30通り)。
        beam_k: 1手目で残す上位候補数。
        exclude_hidden_row_from_pop: 幽霊連鎖ルール (既定 False)。

    Returns:
        `boards` と同じ長さの最良2手先 exact_score リスト (候補が無い盤面は0)。
        値は `simulate_after_drops`+`chain_metrics_after_drops` で盤面ごとに
        個別に2段探索した場合と完全一致する (`tests/test_puyo_core_parity.py`
        で確認)。
    """
    if NATIVE_AVAILABLE:
        flats = [_grid_to_flat_list(b) for b in boards]
        raw = _native.potential_fire_power_raw_for_boards_py(
            flats, drops, beam_k, exclude_hidden_row_from_pop,
        )
        return list(raw)
    return [
        _potential_fire_power_raw_one_fallback(b, drops, beam_k, exclude_hidden_row_from_pop)
        for b in boards
    ]


def _potential_fire_power_raw_one_fallback(
    board: Board,
    drops: "list[tuple[int, int]]",
    beam_k: int,
    exclude_hidden_row_from_pop: bool,
) -> int:
    """`potential_fire_power_raw_for_boards` の Python フォールバック、1盤面分。"""
    first_pass = [
        (r.chain_result.chain_count, r.dropped_board)
        for r in simulate_after_drops(board, drops, exclude_hidden_row_from_pop)
        if r is not None
    ]
    first_pass.sort(key=lambda x: x[0], reverse=True)
    best_score = 0
    for _chain, board1 in first_pass[:beam_k]:
        for m in chain_metrics_after_drops(board1, drops, exclude_hidden_row_from_pop):
            if m is None:
                continue
            _chain_count, exact_score = m
            if exact_score > best_score:
                best_score = exact_score
    return best_score


def _drop_one_fallback_then_simulate(
    board: Board, col: int, color: int, exclude_hidden_row_from_pop: bool,
) -> "DropSimResult | None":
    """`simulate_after_drops` の Python フォールバック用、1候補分の処理。"""
    row = _drop_row_fallback(board, col)
    if row is None:
        return None
    dropped = board.copy()
    dropped.set(row, col, color)
    return DropSimResult(
        dropped_board=dropped,
        chain_result=simulate_chain(dropped, exclude_hidden_row_from_pop),
    )


# ============================
# 配置列挙
# ============================


def _drop_row_fallback(board: Board, col: int) -> "int | None":
    """列 col の落下先行 (可視+隠し段、height_of 基準)。満杯なら None。"""
    height = board.height_of(col)
    if height >= BOARD_ROWS:
        return None
    return BOARD_ROWS - 1 - height


def _place_pair_fallback(
    board: Board, pair: "tuple[int, int]", col: int, rotation: int,
) -> "Board | None":
    """`src/indicators_v2.py::_place_pair_to_board` と同一仕様の複製
    (indicators_v2.py は編集禁止のため、フォールバック側で独立に再実装する。
    重複だが「他エージェント作業中ファイルに触れない」制約を優先する)。
    """
    top, bot = pair
    if top == COLOR_EMPTY or bot == COLOR_EMPTY:
        return None
    work = board.copy()
    if rotation in (_ROTATION_VERTICAL_TOP_UP, _ROTATION_VERTICAL_BOT_UP):
        if not (0 <= col < BOARD_COLS):
            return None
        upper, lower = (top, bot) if rotation == _ROTATION_VERTICAL_TOP_UP else (bot, top)
        if work.height_of(col) > BOARD_ROWS - 2:
            return None
        row_lower = _drop_row_fallback(work, col)
        if row_lower is None:
            return None
        work.set(row_lower, col, lower)
        row_upper = _drop_row_fallback(work, col)
        if row_upper is None:
            return None
        work.set(row_upper, col, upper)
        return work
    if not (0 <= col < BOARD_COLS - 1):
        return None
    left, right = (top, bot) if rotation == _ROTATION_HORIZONTAL_TOP_LEFT else (bot, top)
    row_left = _drop_row_fallback(work, col)
    if row_left is None:
        return None
    work.set(row_left, col, left)
    row_right = _drop_row_fallback(work, col + 1)
    if row_right is None:
        return None
    work.set(row_right, col + 1, right)
    return work


def enumerate_placements(
    board: Board, pair: "tuple[int, int]", filter_dead: bool = True,
) -> "list[tuple[int, int, Board]]":
    """22 配置を列挙し `[(col, rotation, placed_board), ...]` を返す (発火前)。

    拡張があれば使用、無ければ Python フォールバック
    (`src/indicators_v2.py::_enumerate_placement_boards` と同一仕様の複製、
    そちらは編集禁止ファイルのため import せず独立実装)。

    Args:
        filter_dead: True (既定) で窒息する配置を除外する。
    """
    if NATIVE_AVAILABLE:
        flat = _grid_to_flat_list(board)
        raw = _native.enumerate_placements_py(flat, pair[0], pair[1], filter_dead)
        return [
            (col, rotation, _flat_list_to_board(grid))
            for col, rotation, grid, _is_dead in raw
        ]
    results: "list[tuple[int, int, Board]]" = []
    for rotation in range(_NUM_ROTATIONS):
        max_col = (
            BOARD_COLS
            if rotation in (_ROTATION_VERTICAL_TOP_UP, _ROTATION_VERTICAL_BOT_UP)
            else BOARD_COLS - 1
        )
        for col in range(max_col):
            placed = _place_pair_fallback(board, pair, col, rotation)
            if placed is None:
                continue
            if filter_dead and placed.is_dead():
                continue
            results.append((col, rotation, placed))
    return results


@dataclass(frozen=True)
class PlacementSimResult:
    """1配置 (設置直後盤面+連鎖シミュレーション結果) の統合結果 (2026-08-13追加)。

    `enumerate_placements` + 配置ごとの `simulate_chain` 個別呼び出しを1回の
    バッチ呼び出しに統合した結果型 (`scripts/mc_counter_estimator.py` の
    `_select_best_placement`/`_select_build_placement` 用、Python<->Rust
    境界の往復回数削減が目的、値は個別呼び出しした場合と完全一致)。

    Attributes:
        col: 設置列。
        rotation: 回転種別 (0-3、`_ROTATION_*` 定数参照)。
        placed_board: 設置直後 (連鎖解決前) の盤面。
        is_dead: 設置直後盤面が窒息しているか。
        chain_result: placed_board を連鎖シミュレートした結果。
    """
    col: int
    rotation: int
    placed_board: Board
    is_dead: bool
    chain_result: ChainSimResult


def enumerate_and_simulate_placements(
    board: Board,
    pair: "tuple[int, int]",
    filter_dead: bool = False,
    exclude_hidden_row_from_pop: bool = False,
) -> "list[PlacementSimResult]":
    """22配置の列挙+各配置の連鎖シミュレーションを1回にまとめて返す (2026-08-13追加)。

    `enumerate_placements` (1回) + 配置ごとの `simulate_chain` (最大22回) を
    個別に呼び出す場合と値は完全一致するが、拡張導入時は1回の
    Python<->Rust往復に統合される (境界コスト削減目的)。

    Args:
        board: 判定対象の盤面 (破壊しない)。
        pair: 設置するペア (top, bot)。
        filter_dead: True で設置直後 (連鎖解決前) が窒息する配置を除外
            (`enumerate_placements` と同一意味論、既定 False =
            `_select_build_placement` の「消去済み含め全候補見る」用途に合わせる)。
        exclude_hidden_row_from_pop: 幽霊連鎖ルール (既定 False)。

    Returns:
        `PlacementSimResult` のリスト (列挙順=`enumerate_placements` と同一、
        rotation昇順→col昇順)。
    """
    if NATIVE_AVAILABLE:
        flat = _grid_to_flat_list(board)
        raw = _native.enumerate_and_simulate_placements_py(
            flat, pair[0], pair[1], filter_dead, exclude_hidden_row_from_pop,
        )
        return [
            PlacementSimResult(
                col=r.col,
                rotation=r.rotation,
                placed_board=_flat_list_to_board(r.placed_grid),
                is_dead=r.is_dead,
                chain_result=ChainSimResult(
                    chain_count=r.chain_count,
                    total_erased=r.total_erased,
                    total_ojama=r.total_ojama,
                    score_approx=r.score_approx,
                    exact_score=r.exact_score,
                    final_board=_flat_list_to_board(r.final_grid),
                ),
            )
            for r in raw
        ]
    results: "list[PlacementSimResult]" = []
    for col, rotation, placed in enumerate_placements(board, pair, filter_dead=filter_dead):
        results.append(
            PlacementSimResult(
                col=col, rotation=rotation, placed_board=placed,
                is_dead=placed.is_dead(),
                chain_result=simulate_chain(placed, exclude_hidden_row_from_pop),
            )
        )
    return results


# ============================
# ビームサーチ
# ============================


def beam_search(
    board: Board,
    pairs: "list[tuple[int, int]]",
    beam_width: int,
    exclude_hidden_row_from_pop: bool = False,
    num_threads: "int | None" = None,
) -> BeamSearchResult:
    """リアルタイム応手探索ビームサーチ (深さ=len(pairs)、幅=beam_width)。

    Args:
        board: 探索開始盤面。
        pairs: 既知ツモ列 `[(top, bot), ...]`。
        beam_width: 各深さで残す候補数上限。
        exclude_hidden_row_from_pop: 幽霊連鎖ルール (既定 False、
            backwards compat。本番相当で使うなら呼び出し側が
            `src.production_config.GHOST_CHAIN_RULE_ENABLED` を明示的に渡す)。
        num_threads: 拡張利用時の rayon 並列スレッド数 (None=単スレッド)。
            フォールバック経路では無視する (常に単スレッド)。
    """
    if NATIVE_AVAILABLE:
        flat = _grid_to_flat_list(board)
        r = _native.beam_search_py(
            flat, list(pairs), beam_width, exclude_hidden_row_from_pop, num_threads,
        )
        return BeamSearchResult(
            best_score=r.best_score,
            best_path=list(r.best_path),
            best_score_per_depth=list(r.best_score_per_depth),
        )
    return _beam_search_fallback(board, pairs, beam_width, exclude_hidden_row_from_pop)


def _beam_search_fallback(
    board: Board,
    pairs: "list[tuple[int, int]]",
    beam_width: int,
    exclude_hidden_row_from_pop: bool,
) -> BeamSearchResult:
    """拡張未導入環境向け Python フォールバック (低速、正当性優先)。

    ネイティブ側 `native/puyo_core/src/beam.rs::beam_search` と同一の
    アルゴリズム (running max 方式) を Python で複製する。
    """
    frontier: "list[tuple[Board, int, int, tuple[int,int] | None]]" = [
        (board, 0, -1, None),
    ]
    # history[d] = そのフロンティア (親indexは1つ前の history[d-1] を指す)
    history: "list[list[tuple[Board, int, int, tuple[int,int] | None]]]" = [frontier]
    best_score_per_depth: "list[int]" = []
    global_best = 0

    for pair in pairs:
        prev = history[-1]
        candidates: "list[tuple[Board, int, int, tuple[int,int] | None]]" = []
        if prev:
            for idx, (b, running_best, _parent, _placement) in enumerate(prev):
                for col, rotation, placed in enumerate_placements(b, pair, filter_dead=True):
                    sim = simulate_chain(placed, exclude_hidden_row_from_pop)
                    new_best = max(running_best, sim.score_approx)
                    candidates.append((sim.final_board, new_best, idx, (col, rotation)))
        if not candidates:
            best_score_per_depth.append(global_best)
            history.append([])
            continue
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:beam_width]
        depth_best = max(c[1] for c in candidates)
        global_best = max(global_best, depth_best)
        best_score_per_depth.append(global_best)
        history.append(candidates)

    best_depth = len(history) - 1
    while not history[best_depth] and best_depth > 0:
        best_depth -= 1
    last = history[best_depth]
    if not last:
        return BeamSearchResult(best_score=0, best_path=[], best_score_per_depth=best_score_per_depth)
    best_local_idx = max(range(len(last)), key=lambda i: last[i][1])
    best_score = last[best_local_idx][1]

    path: "list[tuple[int, int]]" = []
    cur_depth, cur_idx = best_depth, best_local_idx
    while cur_depth > 0:
        _b, _rb, parent_idx, placement = history[cur_depth][cur_idx]
        path.append(placement)
        cur_idx = parent_idx
        cur_depth -= 1
    path.reverse()

    return BeamSearchResult(
        best_score=best_score, best_path=path, best_score_per_depth=best_score_per_depth,
    )
