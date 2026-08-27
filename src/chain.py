"""
連鎖シミュレーションモジュール

Board を受け取り、消去→重力→消去の繰り返しをシミュレートする。
indicators.py が必要とする連鎖情報（連鎖数・参加ぷよ数・発火後盤面）を提供する。
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    Board,
)

# ============================
# 定数定義
# ============================

# 消去に必要な最小グループサイズ (ぷよぷよルール)
MIN_ERASE_COUNT: int = 4

# 1行分のおじゃま落下数 = 列数
OJAMA_ROW_SIZE: int = BOARD_COLS

# フラッドフィル用の上下左右デルタ
NEIGHBOR_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),  # 上
    (1, 0),   # 下
    (0, -1),  # 左
    (0, 1),   # 右
)

# Phase G: Monte Carlo サンプル数のデフォルト (確率版 simulate)
PROBABILISTIC_DEFAULT_SAMPLES: int = 10
# mean_score 算出時の消去数→スコア換算の除数 (簡易代理)
PROBABILISTIC_SCORE_ERASE_DIVISOR: float = 10.0


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class PuyoGroup:
    """
    連続した同色ぷよのグループ。

    Attributes:
        color: グループの色コード (COLOR_RED など)。おじゃまは含まない。
        cells: (row, col) の集合。
        size: グループのサイズ (len(cells) のキャッシュ)。
        ojama_adjacent: グループに隣接するおじゃまセルの集合。
    """
    color: int
    cells: frozenset[tuple[int, int]]
    size: int
    ojama_adjacent: frozenset[tuple[int, int]]


@dataclass
class ChainStep:
    """
    連鎖の1ステップ（1消し）の情報。

    Attributes:
        chain_index: 何連鎖目か (1-indexed)。
        erased_groups: このステップで消えたグループ群。
        erased_ojama: このステップで消えたおじゃまの数。
        erased_count: 消えた通常ぷよの合計 (おじゃま除く)。
        board_before: 消去前の盤面スナップショット。
        board_after: 重力適用後の盤面スナップショット。
    """
    chain_index: int
    erased_groups: list[PuyoGroup]
    erased_ojama: int
    erased_count: int
    board_before: Board
    board_after: Board


@dataclass
class ChainResult:
    """
    simulate() が返す連鎖シミュレーション結果。

    Attributes:
        steps: 各連鎖ステップのリスト。空なら連鎖なし。
        chain_count: 総連鎖数 (= len(steps))。
        total_erased: 全ステップの erased_count 合計 (通常ぷよのみ)。
        total_ojama: 全ステップの erased_ojama 合計。
        final_board: 連鎖終了後の盤面。
        participating_cells: 連鎖参加ぷよ数 (= total_erased、indicators.py 用エイリアス)。
    """
    steps: list[ChainStep]
    chain_count: int
    total_erased: int
    total_ojama: int
    final_board: Board
    participating_cells: int


# ============================
# ChainSimulator
# ============================


class ChainSimulator:
    """
    ぷよぷよの連鎖をシミュレートするクラス。

    全メソッドはステートレスに設計されており、引数の Board を破壊しない。

    Usage:
        sim = ChainSimulator()
        result = sim.simulate(board)
        print(result.chain_count)

    高速化 (2026-05-06):
        Tier B 指標 (planning_entropy 等) で 1 frame あたり数十回
        simulate が呼ばれるため、盤面 grid をキーとした LRU キャッシュ
        を導入。同一盤面の simulate を高速化 (~10 倍)。

    幽霊連鎖ルール (2026-08-09 user伝授):
        ぷよぷよeスポーツでは 13段目 (隠し段、row=0) のぷよは 4つ繋がっても
        消えない ("幽霊連鎖" として上級者が利用する公式仕様)。
        `exclude_hidden_row_from_pop=True` でこの正しいルールを有効化できる。
        既定は False (従来挙動維持、backwards compat)。
        参考実装: `src/chain_bitboard.py` の `exclude_hidden_row_from_pop`
        (`simulate_batch` / `simulate_single`)。
    """

    # キャッシュサイズ上限 (盤面パターンの種類)
    _CACHE_MAX_SIZE: int = 50_000

    def __init__(
        self,
        cache_enabled: bool = True,
        exclude_hidden_row_from_pop: bool = False,
    ) -> None:
        # bytes キー (board._grid.tobytes()) → ChainResult のメモ化
        self._cache: dict[bytes, "ChainResult"] = {}
        self._cache_enabled = cache_enabled
        # 幽霊連鎖ルール切替 (2026-08-09、既定 False = 従来挙動維持)。
        # 既存指標・学習済み重み・過去 cycle の再現性を壊さないための
        # backwards compat フラグ (呼び出し元での切替判断は別途行う)。
        self._exclude_hidden_row_from_pop = exclude_hidden_row_from_pop

    # ============================
    # 公開メソッド
    # ============================

    def simulate(self, board: Board) -> ChainResult:
        """
        盤面の連鎖をシミュレートする (キャッシュ対応版)。

        引数の board は変更しない (内部で copy() して操作)。

        Args:
            board: シミュレート対象の盤面。

        Returns:
            ChainResult: 連鎖の結果情報。
            キャッシュ取得時も final_board を copy して返すため
            呼び出し側が安全に board を改変できる。
        """
        import dataclasses
        if not self._cache_enabled:
            return self._simulate_uncached(board)
        cache_key = board._grid.tobytes()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return dataclasses.replace(
                cached, final_board=cached.final_board.copy(),
            )
        result = self._simulate_uncached(board)
        if len(self._cache) < self._CACHE_MAX_SIZE:
            self._cache[cache_key] = dataclasses.replace(
                result, final_board=result.final_board.copy(),
            )
        return result

    def _simulate_uncached(self, board: Board) -> ChainResult:
        """キャッシュ無し本体. simulate() から呼び出される。"""
        work_board = board.copy()
        steps: list[ChainStep] = []
        total_erased = 0
        total_ojama = 0
        chain_index = 1

        while True:
            erasable = self.find_erasable_groups(work_board)
            if not erasable:
                break

            board_before = work_board.copy()
            erased_count = sum(g.size for g in erasable)
            erased_ojama = self._erase_groups(work_board, erasable)
            self.apply_gravity(work_board)
            board_after = work_board.copy()

            steps.append(ChainStep(
                chain_index=chain_index,
                erased_groups=erasable,
                erased_ojama=erased_ojama,
                erased_count=erased_count,
                board_before=board_before,
                board_after=board_after,
            ))
            total_erased += erased_count
            total_ojama += erased_ojama
            chain_index += 1

        return ChainResult(
            steps=steps,
            chain_count=len(steps),
            total_erased=total_erased,
            total_ojama=total_ojama,
            final_board=work_board,
            participating_cells=total_erased,
        )

    def find_groups(self, board: Board) -> list[PuyoGroup]:
        """
        盤面上の全グループを検出する (サイズ問わず)。

        おじゃまはグループを形成しない。隣接おじゃまは
        PuyoGroup.ojama_adjacent に収集される。

        幽霊連鎖ルール (self._exclude_hidden_row_from_pop=True 時):
        13段目 (隠し段、row < HIDDEN_ROWS) のセルはグループを形成しない
        (このセルを起点にグループを作らない。可視セル側からの取り込み
        防止は `_flood_fill` 側で行う)。

        Args:
            board: 検索対象の盤面。

        Returns:
            list[PuyoGroup]: 検出した全グループのリスト。
        """
        visited: list[list[bool]] = [
            [False] * BOARD_COLS for _ in range(BOARD_ROWS)
        ]
        groups: list[PuyoGroup] = []

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                color = board.get(row, col)
                # UNKNOWN (隠し段・量子状態) はグループ対象外
                if color in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN):
                    continue
                # 幽霊連鎖ルール: 13段目は連結判定の対象外 (ON時のみ)
                if self._exclude_hidden_row_from_pop and row < HIDDEN_ROWS:
                    continue
                if visited[row][col]:
                    continue

                group = self._flood_fill(board, row, col, color, visited)
                groups.append(group)

        return groups

    def find_erasable_groups(self, board: Board) -> list[PuyoGroup]:
        """
        消去可能なグループ (size >= MIN_ERASE_COUNT) を返す。

        Args:
            board: 検索対象の盤面。

        Returns:
            list[PuyoGroup]: 消去対象グループのリスト。
        """
        return [g for g in self.find_groups(board) if g.size >= MIN_ERASE_COUNT]

    def apply_gravity(self, board: Board) -> Board:
        """
        盤面に重力を適用し、ぷよを下詰めする (in-place)。

        UNKNOWN セルは壁として固定され、その上下を独立したセグメントとして
        処理する (UNKNOWN を跨いで落下しない)。

        Args:
            board: 重力を適用する盤面。

        Returns:
            Board: 同じ board インスタンス (in-place 変更)。
        """
        for col in range(BOARD_COLS):
            self._apply_gravity_column(board, col)
        return board

    @staticmethod
    def _apply_gravity_column(board: Board, col: int) -> None:
        """1 列に対する重力処理。UNKNOWN をセグメント境界として扱う。"""
        # UNKNOWN 位置を境界としてセグメントに分割
        segments: list[tuple[int, int]] = []
        seg_start = 0
        for row in range(BOARD_ROWS):
            if board.get(row, col) == COLOR_UNKNOWN:
                if seg_start < row:
                    segments.append((seg_start, row))
                seg_start = row + 1
        if seg_start < BOARD_ROWS:
            segments.append((seg_start, BOARD_ROWS))

        for (start, end) in segments:
            # セグメント内の非空 puyo を下詰め
            puyos = [
                board.get(r, col)
                for r in range(start, end)
                if board.get(r, col) != COLOR_EMPTY
            ]
            for r in range(start, end):
                board.set(r, col, COLOR_EMPTY)
            for i, color in enumerate(reversed(puyos)):
                board.set(end - 1 - i, col, color)

    def drop_ojama(
        self, board: Board, ojama_count: int, seed: int | None = None,
    ) -> Board:
        """
        指定個数のおじゃまを盤面に落とす。

        均等行分 (floor(N/6) 行) + 端数 (N mod 6) はランダムな列へ 1 個ずつ。
        再現性が必要な場合は seed を指定すること。
        引数の board は変更しない (内部で copy())。

        Args:
            board: おじゃまを落とす前の盤面。
            ojama_count: 落とすおじゃまの数。
            seed: 端数列選択の乱数シード (None = 毎回ランダム)。後方互換 optional。
                端数 (ojama_count % 6 > 0) があるのに None のままだと
                RuntimeWarning が出る (W39 再発防止、挙動は不変)。

        Returns:
            Board: おじゃまを落とした後の新しい盤面。

        Raises:
            ValueError: ojama_count が負の場合。
        """
        if ojama_count < 0:
            raise ValueError(f"おじゃま数が負の値: {ojama_count}")

        work_board = board.copy()

        # 各列に落とすおじゃまの数を計算 (端数はランダム列)
        drop_counts = self._calc_ojama_drop_counts(ojama_count, seed=seed)

        # 各列の下から空きセルを探しておじゃまを配置 (重力で下に落ちる)
        for col in range(BOARD_COLS):
            remaining = drop_counts[col]
            if remaining == 0:
                continue
            for row in range(BOARD_ROWS - 1, -1, -1):
                if work_board.get(row, col) == COLOR_EMPTY:
                    work_board.set(row, col, COLOR_OJAMA)
                    remaining -= 1
                    if remaining == 0:
                        break
            # 置けなかった分はスキップ (is_dead() で窒息を検出)

        return work_board

    # ============================
    # 内部メソッド
    # ============================

    def _flood_fill(
        self,
        board: Board,
        start_row: int,
        start_col: int,
        color: int,
        visited: list[list[bool]],
    ) -> PuyoGroup:
        """
        BFS フラッドフィルで同色グループを検出する。

        幽霊連鎖ルール (self._exclude_hidden_row_from_pop=True 時):
        隣接セルが13段目 (row < HIDDEN_ROWS) の場合、同色であっても
        グループに取り込まない (連結対象外)。ここで除外しないと
        「13段目1個+可視3個」を4連結と誤判定し、判定後に13段目分を
        差し引いて可視3個だけ消してしまう (誤り)。連結判定の前に
        除外することで、可視3個だけでは4連結不成立=何も消えない、
        という正しい挙動になる。隣接おじゃまの収集 (ojama_adjacent) は
        13段目かどうかに関わらず従来通り行う (`src/chain_bitboard.py`
        の `_expand` が全13行を対象にする実装と挙動を揃えるため)。

        Args:
            board: 対象の盤面。
            start_row: 開始行。
            start_col: 開始列。
            color: 収集する色。
            visited: 訪問済みフラグ (in-place で更新)。

        Returns:
            PuyoGroup: 検出したグループ。
        """
        cells: set[tuple[int, int]] = set()
        ojama_adjacent: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
        visited[start_row][start_col] = True

        while queue:
            row, col = queue.popleft()
            cells.add((row, col))

            for dr, dc in NEIGHBOR_DELTAS:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS):
                    continue
                is_hidden_neighbor = (
                    self._exclude_hidden_row_from_pop and nr < HIDDEN_ROWS
                )
                neighbor_color = board.get(nr, nc)
                if (
                    neighbor_color == color
                    and not visited[nr][nc]
                    and not is_hidden_neighbor
                ):
                    visited[nr][nc] = True
                    queue.append((nr, nc))
                elif neighbor_color == COLOR_OJAMA:
                    # おじゃまは visited に入れない (複数グループが同じおじゃまに隣接できる)
                    ojama_adjacent.add((nr, nc))

        return PuyoGroup(
            color=color,
            cells=frozenset(cells),
            size=len(cells),
            ojama_adjacent=frozenset(ojama_adjacent),
        )

    def _erase_groups(
        self, board: Board, groups: list[PuyoGroup]
    ) -> int:
        """
        グループのぷよを盤面から消去し、隣接おじゃまも消去する。

        複数グループが同一おじゃまに隣接する場合の重複を set で排除する。
        引数の board を in-place で変更する。

        Args:
            board: 消去対象の盤面。
            groups: 消去するグループのリスト。

        Returns:
            int: 消去したおじゃまの数 (重複排除後)。
        """
        # 通常ぷよの消去
        for group in groups:
            for row, col in group.cells:
                board.set(row, col, COLOR_EMPTY)

        # 隣接おじゃまを set で重複排除してから消去
        all_ojama: set[tuple[int, int]] = set()
        for group in groups:
            all_ojama |= group.ojama_adjacent
        for row, col in all_ojama:
            board.set(row, col, COLOR_EMPTY)

        return len(all_ojama)

    def _get_neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """
        盤内の上下左右の有効座標リストを返す。

        Args:
            row: 基準行。
            col: 基準列。

        Returns:
            list[tuple[int, int]]: (row, col) の有効隣接座標リスト。
        """
        neighbors: list[tuple[int, int]] = []
        for dr, dc in NEIGHBOR_DELTAS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS:
                neighbors.append((nr, nc))
        return neighbors

    def _calc_ojama_drop_counts(
        self, ojama_count: int, seed: int | None = None,
    ) -> list[int]:
        """
        おじゃま個数から各列への落下数を計算する。

        仕様: N 個を 6 列に均等配分 (各 floor(N/6)) + 端数 (N mod 6) は
        ランダムに選んだ remainder 列へ 1 個ずつ追加する。
        (user 伝授ルール: 端数はランダム列。旧実装の「左から順」は誤り。)

        dig_resistance 等の精度にも影響するため、呼び出し元が再現性を
        必要とする場合は seed を指定すること。seed 未指定のまま端数抽選に
        入った場合は RuntimeWarning を出す (2026-08-25 W39 再発防止ガード、
        挙動自体は従来通り OS乱数のまま = 既定挙動は不変)。

        Args:
            ojama_count: 落とすおじゃまの総数。
            seed: 端数列選択の乱数シード (None = 毎回ランダム)。

        Returns:
            list[int]: 各列 (index=0〜5) の落下数。
        """
        import random as _random
        drop_counts = [0] * BOARD_COLS
        full_rows, remainder = divmod(ojama_count, OJAMA_ROW_SIZE)

        for col in range(BOARD_COLS):
            drop_counts[col] += full_rows

        if remainder > 0:
            if seed is None:
                # 端数抽選が非決定 (OS乱数) 経路に入った = 呼び出しごとに
                # 結果が変わる。指標経路で踏むと実行間で値が揺れる
                # (docs/KNOWN_WEAKNESSES.md W39 の根因そのもの) ため警告する。
                # 挙動は変えない (backwards compat、警告のみ)。
                warnings.warn(
                    "drop_ojama: 端数列の抽選が seed 未指定 (OS乱数) で呼ばれた。"
                    "再現性が必要な経路では盤面由来の決定論的 seed を渡すこと"
                    " (docs/KNOWN_WEAKNESSES.md W39)。",
                    RuntimeWarning,
                    stacklevel=3,  # drop_ojama の呼び出し元を指す
                )
            rng = _random.Random(seed)
            cols_for_remainder = rng.sample(range(BOARD_COLS), remainder)
            for col in cols_for_remainder:
                drop_counts[col] += 1

        return drop_counts

    # ============================
    # Phase G: 確率版シミュレーション
    # ============================

    def simulate_probabilistic(
        self,
        prob_board: "ProbabilisticBoard",
        n_samples: int = PROBABILISTIC_DEFAULT_SAMPLES,
        seed: int | None = None,
    ) -> "ProbabilisticChainResult":
        """確率盤面を N 個サンプリングし、それぞれを simulate して結果を集約する.

        Args:
            prob_board: ProbabilisticBoard。
            n_samples: Monte Carlo サンプル数。
            seed: 再現性確保用シード (None ならランダム)。

        Returns:
            ProbabilisticChainResult: 集約結果。
                mean_chain_count / mean_erased_puyos / mean_score は平均値。
                mle_final_board は各セル各色の出現頻度を集計し最尤色で構築。
        """
        # 遅延 import で循環参照を避ける
        from src.probabilistic_board import ProbabilisticBoard
        if not isinstance(prob_board, ProbabilisticBoard):
            raise TypeError("prob_board must be ProbabilisticBoard")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0: {n_samples}")
        rng = np.random.default_rng(seed)
        samples: list[ChainResult] = []
        # 各 (row, col) で各色の出現回数を集計し、最尤色で最終盤面を構築
        color_counts: list[list[dict[int, int]]] = [
            [dict() for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)
        ]
        sum_chain = 0.0
        sum_erased = 0.0
        sum_score = 0.0
        for _ in range(n_samples):
            sampled = prob_board.sample_board(rng=rng)
            result = self.simulate(sampled)
            samples.append(result)
            sum_chain += float(result.chain_count)
            sum_erased += float(result.total_erased)
            # PROBABILISTIC_SCORE_PROXY: 連鎖数 + 消去数 / 10 を簡易代理
            # (本格的得点は OffsetPower で計算するためここでは簡素化)
            sum_score += (
                float(result.chain_count)
                + float(result.total_erased) / PROBABILISTIC_SCORE_ERASE_DIVISOR
            )
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    color = int(result.final_board.get(r, c))
                    color_counts[r][c][color] = (
                        color_counts[r][c].get(color, 0) + 1
                    )
        n = float(n_samples)
        mle_final = Board()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                counts = color_counts[r][c]
                if counts:
                    color = max(counts.items(), key=lambda kv: kv[1])[0]
                    mle_final.set(r, c, color)
        return ProbabilisticChainResult(
            mean_chain_count=sum_chain / n,
            mean_erased_puyos=sum_erased / n,
            mean_score=sum_score / n,
            samples=samples,
            mle_final_board=mle_final,
        )


# ============================
# Phase G: 確率版データクラス
# ============================


@dataclass
class ProbabilisticChainResult:
    """確率版 simulate の結果集約.

    Attributes:
        mean_chain_count: N サンプルの chain_count 平均.
        mean_erased_puyos: N サンプルの total_erased 平均.
        mean_score: N サンプルの簡易スコア平均 (chain + erased/10).
        samples: 個別 ChainResult のリスト (N 個).
        mle_final_board: 各セル最頻出色で構築した代表 final_board.
    """
    mean_chain_count: float
    mean_erased_puyos: float
    mean_score: float
    samples: list[ChainResult]
    mle_final_board: Board

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def std_chain_count(self) -> float:
        """chain_count 標準偏差 (サンプル数 0 で 0)."""
        if not self.samples:
            return 0.0
        arr = np.array([s.chain_count for s in self.samples], dtype=np.float64)
        return float(arr.std())
