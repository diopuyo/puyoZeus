"""ama(citrus610/ama, MIT)構成ループ B案プロトタイプ。

目的:
    「今の盤面から理想ツモ(その試合の4色限定)を自由に置いて最大火力(=得点)に
    する」ロジックを、ama の quiet::dfs (ai/search/dfs/quiet.cpp) が使う
    「候補生成 → 実シミュ検証 → 成長ガード → 採用/棄却 → 反復」の構成ループで
    プロトタイプ実装し、難所盤面 (current_max_chain が低いまま伸びなかった盤面)
    を実際に割れるかを検証する。

前回失敗の真因 (user診断、src/indicators_v2.py の _sat_expand_step 実装参照):
    旧来の自前ビルダーは「連結グループサイズ・列高さ」という静的近似ヒューリス
    ティックのみで手を採用しており、「その手を足したら実際に連鎖(火力)が
    伸びるか」を一度も実シミュで検証していなかった。結果、骨格を壊す手を
    誤採用していた。

本プロトタイプの核心 (ama からの移植箇所):
    - `_generate_candidates`: ai/search/dfs/quiet.cpp:271-365 `quiet::generate` の移植。
      列×色×個数×方向(垂直/右拡張/左拡張)で候補手を列挙する。
    - `ama_build` 内の growth guard: ai/search/dfs/quiet.cpp:200
      `if (sim_mask.get_size() > pre_chain)` の移植。「この候補を足した盤面を
      実際に発火させたら、直前より火力(得点)が伸びるか」を実シミュ検証し、
      伸びた候補のみ採用する。これが骨格破壊防止の核心。
    - `_eval_score`: ai/search/dfs/eval.cpp:11-92 `eval::evaluate` の部分移植
      (freestyle プロファイル、config.json の重みをそのまま使用)。
    - root/field 二状態管理: ai/search/dfs/quiet.cpp の dfs() は「候補生成の
      基準にする盤面 (発火後の残骸=field)」と「実際に積み上げる生盤面
      (発火させない root)」を分けて持つ。両者を同一視すると、生成した
      トリガー候補がそのまま root に適用した瞬間に必ず自己矛盾で棄却される
      バグを踏むため (実装中に実際に踏んで気付いた)、本プロトタイプも
      field_board (直前採用候補の発火後残骸) と root (蓄積生盤面) を
      分離して保持する。

user伝授のドメイン修正 (2026-07-22、実装中に2件反映):
    1. 1試合ごとに5色中1色がランダムに除外され、その試合は4色しか降らない。
       理想ツモの候補色は「試合(video_id, game_idx)で実際に出現した色」に
       限定する (`_compute_active_colors_by_game`)。実データでは除外色も
       認識ノイズで少数セル混入するため、単純な出現有無ではなく出現頻度の
       下位1色を除外する頻度ベース判定を採用する (詳細はコメント参照)。
    2. 飽和の本質は連鎖数ではなく火力(=得点、送りお邪魔量の元になる値)。
       目的関数・成長ガード・返り値のすべてを連鎖数ベースから得点(score)
       ベースに変更した (`_simulate_with_score`, `full_scan_best_score`)。
       連鎖数は参考値として併記する。

出典・帰属 (MIT License):
    本モジュールの候補生成ロジック (`_generate_candidates`, `_get_bound`,
    `_is_reachable`) および成長ガード付き構成ループ設計、評価式の一部
    (`_eval_score`, `_get_chi`, `_get_shape`, `_get_well`, `_get_bump`,
    `_get_link_23`) は citrus610/ama (https://github.com/citrus610/ama,
    MIT License, Copyright (c) 2023 citrus610) の
    `ai/search/dfs/quiet.cpp` / `ai/search/dfs/eval.cpp` / `config.json`
    (freestyle プロファイル) のロジック移植である (逐語コピーではない。
    SIMD(__m128i) 部分は意味論のみ理解し、本プロジェクト既存の
    `src/chain_bitboard.py` (これも ama 移植・別モジュール) の numpy バッチ
    シミュレータを一部流用する)。

    得点計算の式・ボーナステーブル (`chain_power`/`connection_bonus`/
    `color_bonus`) は src/scoring.py (公式得点式、既存実装) をそのまま
    import して使う (通ルール公式仕様、ama とは無関係の既存資産)。

スコープ・制約 (厳守):
    - src/ 本体・既存指標は一切変更しない (本ファイルは scripts/ 配下の使い捨て
      プロトタイプ)。
    - src/chain_bitboard.py は import して使うのみ (変更しない)。
    - src/indicators_v2.py・src/chain.py は一切編集しない。src/chain.py は
      src/scoring.py が内部で import しているが (`from src.chain import
      ChainResult, ChainStep`)、これは既存コードの既存依存であり本ファイルが
      新たに作る依存ではない。本ファイル自身は src.chain を直接 import せず、
      連鎖シミュレーション (`_simulate_with_score`) は Board のみを使う
      独立実装とする (別コーダの飽和AUC検証との実行時カップリングを避ける)。

正直な位置づけ:
    ama の平均11連鎖は「多数の実対局ターンの反復」で創発するものであり、
    静的な1発評価では出せない。本プロトタイプが「反復構成ループ」で
    難所盤面を割れるかどうかが検証の焦点であり、割れない/時間が非現実的
    (1盤面が分オーダーを大きく超える) 場合はその事実をそのまま報告する。
"""
from __future__ import annotations

import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, Board
from src.chain_bitboard import TRACKED_COLORS, board_to_planes
from src.scoring import (
    BASE_SCORE_PER_PUYO,
    MAX_BONUS_MULTIPLIER,
    MIN_BONUS_MULTIPLIER,
    chain_power,
    color_bonus,
    connection_bonus,
)

# ============================
# 定数 (マジックナンバー禁止規約に準拠、すべて定数化)
# ============================

# 盤面全体セル数 (6列×13行=78)。user確定の飽和定義 (約90-95%充填) の分母。
# src/indicators_v2.py の FULL_BOARD_CAP と同じ値だが、依存排除のためここで
# 独立して再定義する (import はしない)。
FULL_BOARD_CAP: int = BOARD_ROWS * BOARD_COLS  # = 78

# 目標充填率 (user確定 2026-07-22: 88-98%範囲、既定93%)。
FILL_RATIO_DEFAULT: float = 0.93

# ama quiet.cpp:232,240,256,262 の「heights[x] > 11」境界。
# ama は 12 visible行 + 1 buffer行 = 13bit 表現 (我々の BOARD_ROWS と同じ13)。
# 「height > 11」= 「可視12行が埋まり、あと1行(buffer/隠し段)しか残っていない」
# を意味するため、閾値の数値 11 はそのまま移植できる (BOARD_ROWS-2 = 11)。
HEIGHT_OVERFLOW_GUARD: int = BOARD_ROWS - 2  # = 11

# ama の x=2 (スポーン列起点)。我々の DEATH_COL (窒息判定列, board.py) と一致。
SPAWN_COL: int = 2

# 4連結で消去成立とみなす閾値 (MIN_ERASE_COUNT 相当、chain.py 非依存のため独立定義)。
GROUP4_TRIGGER: int = 4

# quiet::search が top-level で使う drop 個数 (eval.cpp:25 `quiet::search(field, 16, 3, ...)`)。
GENERATE_DROP_BUDGET: int = 3

# 構成ループの安全弁 (無限ループ防止、盤面全体セル数を上限とする)。
MAX_CONSTRUCT_ITERS: int = FULL_BOARD_CAP

# 得点シミュレーションの連鎖ステップ安全弁 (chain_bitboard.MAX_CHAIN_STEPS と同じ値)。
MAX_SCORE_SIM_STEPS: int = 19

# 全5色プール (chain_bitboard.TRACKED_COLORS: 赤青緑黄紫)。
# ⚠️ user伝授のドメイン修正 (2026-07-22): 実際の対局は1試合ごとに5色中1色が
# 除外され、その試合では4色しか降らない。理想ツモの候補色は5色固定ではなく
# 「その試合 (video_id, game_idx) で実際に出現頻度が高い上位4色 (=active_colors)」
# に限定しなければならない (5色で試すと来ない色まで使って天井を過大評価
# してしまう)。SCAN_COLORS は「全5色プール定数 (物理シミュレーションの
# 連結判定対象色)」として維持し、候補生成・before/after計測には必ず
# active_colors (試合別4色) を渡す。
SCAN_COLORS: "tuple[int, ...]" = TRACKED_COLORS

# 試合別 active_colors 判定: 出現頻度の上位何色を採用するか (通常4色)。
ACTIVE_COLOR_KEEP_COUNT: int = 4


@dataclass(frozen=True)
class Weight:
    """config.json の freestyle プロファイルをそのまま移植した評価重み。

    ama は build/ac/fast/freestyle の4プロファイルを持つが、本プロトタイプは
    「形テンプレなし・機能評価のみ」の freestyle (config.json 内 "freestyle" キー)
    を採用する (user指示: form=0=形テンプレなし)。再調整なしでこの実運用値を
    そのまま使う。

    注記 (得点ベース化に伴う変更): `chain` 重み (元 500、chain_count 用) は
    本プロトタイプでは使わない。第一項は実測得点 (total_score) をそのまま
    使う (user伝授のドメイン修正: 飽和の本質は連鎖数でなく得点)。
    """
    key: int = -200
    chi: int = 100
    y: int = 100
    link_2: int = 50
    link_3: int = 150
    shape: int = -50
    well: int = -50
    bump: int = -50


FREESTYLE_WEIGHT = Weight()


@dataclass(frozen=True)
class Candidate:
    """ama quiet::generate のコールバック引数 (x, p, need, dir) に対応する候補手。"""
    x: int
    color: int
    need: int
    direction: int  # 0=垂直積み, 1=右拡張, -1=左拡張


@dataclass(frozen=True)
class ScoreSimResult:
    """1盤面分の得点シミュレーション結果 (chain_bitboard.BitboardChainResult の得点版)。"""
    chain_count: int
    total_score: int
    final_board: Board


@dataclass
class BuildResult:
    """構成ループ1回分の結果 (難所盤面の before/after 対比用)。

    得点(score)が主目的、連鎖数(chain_ref)は参考値 (user伝授のドメイン修正)。
    """
    seed_puyo_count: int
    before_score: int
    before_chain_ref: int
    final_score: int
    final_chain_ref: int
    root: Board
    iterations: int
    n_sim_calls: int
    elapsed_sec: float
    trace: "list[tuple[int, Candidate, int, int, int]]" = field(default_factory=list)


# ============================
# 盤面ユーティリティ (src.board.Board のみ使用、chain.py/indicators_v2.py非依存)
# ============================


def _heights(board: Board) -> "list[int]":
    """全列の高さ配列を返す (Board.height_of の薄いラッパー)。"""
    return [board.height_of(c) for c in range(BOARD_COLS)]


def _drop_row(height: int) -> "int | None":
    """指定の高さから1個落としたときの着地行 (満杯なら None)。"""
    if height >= BOARD_ROWS:
        return None
    return BOARD_ROWS - 1 - height


def _group_size_after_drop(
    board: Board, row: int, col: int, color: int, cap: int = GROUP4_TRIGGER,
) -> int:
    """(row, col) に color を置いた後の同色連結グループサイズを返す (cap で早期打切り)。

    ama `get_mask_group_4` (bitboard flood-fill) の意味論を素朴な BFS で再現する。
    盤面は13行全体を連結判定に使う (docs/PUYO_RULES_CONFIRMED_2026-07-22.md、
    src/chain_bitboard.py の docstring と同じ方針)。
    """
    visited: "set[tuple[int, int]]" = {(row, col)}
    stack: "list[tuple[int, int]]" = [(row, col)]
    size = 0
    while stack:
        r, c = stack.pop()
        size += 1
        if size >= cap:
            return size
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS):
                continue
            if (nr, nc) in visited:
                continue
            if board.get(nr, nc) == color:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return size


def _drop_one_color(board: Board, col: int, color: int) -> "Board | None":
    """col 列の積み上がり最上段に color を1個置いた新 Board を返す (満杯なら None)。"""
    row = _drop_row(board.height_of(col))
    if row is None:
        return None
    work = board.copy()
    work.set(row, col, color)
    return work


# ============================
# 得点シミュレーション (pure-Python、chain.py 非依存の独立実装)
# ============================


def _find_erasable_groups(grid: np.ndarray) -> "list[tuple[int, set[tuple[int, int]]]]":
    """盤面全体 (13行) を BFS し、4連結以上の同色グループを列挙する。

    SCAN_COLORS (5色) のみ対象 (空・おじゃま・UNKNOWN は対象外)。
    """
    visited: "set[tuple[int, int]]" = set()
    groups: "list[tuple[int, set[tuple[int, int]]]]" = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if (r, c) in visited:
                continue
            color = int(grid[r, c])
            if color not in SCAN_COLORS:
                visited.add((r, c))
                continue
            stack = [(r, c)]
            cell_set: "set[tuple[int, int]]" = {(r, c)}
            visited.add((r, c))
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS):
                        continue
                    if (nr, nc) in visited:
                        continue
                    if int(grid[nr, nc]) == color:
                        visited.add((nr, nc))
                        cell_set.add((nr, nc))
                        stack.append((nr, nc))
            if len(cell_set) >= GROUP4_TRIGGER:
                groups.append((color, cell_set))
    return groups


def _compact_column_values(values: "list[int]") -> "list[int]":
    """重力コンパクション: 空セルを除いた値を下詰めする (相対順序は保持)。"""
    non_empty = [v for v in values if v != COLOR_EMPTY]
    pad = [COLOR_EMPTY] * (len(values) - len(non_empty))
    return pad + non_empty


def _simulate_with_score(board: Board) -> ScoreSimResult:
    """連鎖を最後まで解決し、公式得点式 (src/scoring.py) で得点を積算する。

    chain.py の ChainSimulator には依存しない独立実装 (BFS + 重力コンパクション)。
    得点式 (chain_power/connection_bonus/color_bonus) のみ src/scoring.py を
    再利用する (user伝授のドメイン修正: 飽和の本質は連鎖数でなく得点)。

    正当性は scripts/_tmp_ama_builder.py の対話的検証で chain_bitboard.simulate_single
    の chain_count と照合済み (実装ノート、README的な位置付けの docstring 内で明記)。
    """
    grid = board._grid.copy()
    chain_idx = 0
    total_score = 0

    for _ in range(MAX_SCORE_SIM_STEPS):
        groups = _find_erasable_groups(grid)
        if not groups:
            break
        chain_idx += 1

        erase_mask = np.zeros_like(grid, dtype=bool)
        for _color, cells in groups:
            for (r, c) in cells:
                erase_mask[r, c] = True

        # おじゃま隣接消去 (4連結消去グループに隣接するおじゃまも消える)。
        ojama_mask = np.zeros_like(grid, dtype=bool)
        erased_rc = np.argwhere(erase_mask)
        for r, c in erased_rc:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS and grid[nr, nc] == COLOR_OJAMA:
                    ojama_mask[nr, nc] = True

        erased_count = int(erase_mask.sum())
        chain_bonus_val = chain_power(chain_idx)
        conn_bonus_total = sum(connection_bonus(len(cells)) for _c, cells in groups)
        distinct_colors = len({c for c, _cells in groups})
        col_bonus = color_bonus(distinct_colors)
        raw_bonus = chain_bonus_val + conn_bonus_total + col_bonus
        total_bonus = max(MIN_BONUS_MULTIPLIER, min(MAX_BONUS_MULTIPLIER, raw_bonus))
        total_score += erased_count * BASE_SCORE_PER_PUYO * total_bonus

        grid[erase_mask | ojama_mask] = COLOR_EMPTY
        for c in range(BOARD_COLS):
            grid[:, c] = _compact_column_values(list(grid[:, c]))

    final_board = Board()
    final_board._grid = grid
    return ScoreSimResult(chain_count=chain_idx, total_score=total_score, final_board=final_board)


# ============================
# ama quiet.cpp 移植: 候補生成
# ============================


def _get_bound(heights: "list[int]") -> "tuple[int, int]":
    """ama quiet.cpp:226-248 `quiet::get_bound` の移植。

    スポーン列 (SPAWN_COL=2) を起点に、HEIGHT_OVERFLOW_GUARD (=11) を超える
    (=可視12行が埋まりほぼ満杯の) 列を境界として探索範囲を狭める。
    """
    x_min = x_max = SPAWN_COL
    for x in range(SPAWN_COL + 1, BOARD_COLS):
        if heights[x] > HEIGHT_OVERFLOW_GUARD:
            break
        x_max += 1
    for x in range(SPAWN_COL - 1, -1, -1):
        if heights[x] > HEIGHT_OVERFLOW_GUARD:
            break
        x_min -= 1
    return x_min, x_max


def _is_reachable(heights: "list[int]", x_ban: int) -> bool:
    """ama quiet.cpp:251-268 `quiet::is_reachable` の移植。

    x_ban 列へ実際に到達可能か (間の列がすべて HEIGHT_OVERFLOW_GUARD 以下か) を
    チェックする。x_ban=-1 (禁止列なし) のときは常に True。
    """
    if x_ban < 0:
        return True
    for i in range(SPAWN_COL + 1, x_ban):
        if heights[i] > HEIGHT_OVERFLOW_GUARD:
            return False
    for i in range(SPAWN_COL - 1, x_ban, -1):
        if heights[i] > HEIGHT_OVERFLOW_GUARD:
            return False
    return True


def _generate_candidates(
    board: Board,
    heights: "list[int]",
    x_min: int,
    x_max: int,
    x_ban: int,
    drop_budget: int,
    colors: "tuple[int, ...]",
) -> "list[Candidate]":
    """ama quiet.cpp:271-365 `quiet::generate` の移植。

    列×色×個数×方向 (垂直積み/右拡張/左拡張) で候補手を列挙する。
    理想ツモ=色自由なので ama の2個1組(ペア)制約は課さず、単色ブロブ
    (1個〜drop_budget個) を1列に積む形で候補化する (user指示通り)。

    Args:
        board: 候補生成の「基準にする盤面」(ama の `field` 相当、通常は
            直前に採用した候補の発火後残骸 = field_board)。root (蓄積生盤面)
            とは別物である点に注意 (2状態管理、モジュール docstring参照)。
        colors: 候補色 (呼び出し側で必ず「その試合の active_colors (通常4色)」
            を渡すこと。全5色プール SCAN_COLORS をそのまま渡すと、来ない色まで
            使って天井を過大評価してしまう=user伝授のドメイン修正)。
    """
    candidates: "list[Candidate]" = []
    # quiet.cpp:298 の水平重複チェック用マップ (列×色)。
    horizontal_checked: "set[tuple[int, int]]" = set()

    for x in range(x_min, x_max + 1):
        if x == x_ban:
            continue
        expand_r = (
            x < BOARD_COLS - 1 and x + 1 != x_ban and heights[x] == heights[x + 1]
            and not (x == SPAWN_COL - 1 and heights[SPAWN_COL] > HEIGHT_OVERFLOW_GUARD)
        )
        expand_l = (
            x > 0 and x - 1 != x_ban and heights[x] == heights[x - 1]
            and not (x == SPAWN_COL + 1 and heights[SPAWN_COL] > HEIGHT_OVERFLOW_GUARD)
        )
        drop_max = min(drop_budget, BOARD_ROWS - heights[x])
        if drop_max <= 0:
            continue

        for color in colors:
            dropped_n = _vertical_trigger_need(board, heights, x, color, drop_max)
            if dropped_n is not None:
                candidates.append(Candidate(x=x, color=color, need=dropped_n, direction=0))
                if dropped_n > 1:
                    if expand_r and (x, color) not in horizontal_checked:
                        candidates.append(Candidate(x=x, color=color, need=drop_max, direction=1))
                        horizontal_checked.add((x, color))
                    if expand_l and (x - 1, color) not in horizontal_checked:
                        candidates.append(Candidate(x=x, color=color, need=drop_max, direction=-1))
                        horizontal_checked.add((x - 1, color))
    return candidates


def _vertical_trigger_need(
    board: Board, heights: "list[int]", x: int, color: int, drop_max: int,
) -> "int | None":
    """quiet.cpp:339-347 の内側ループ移植: 1列に color を積み、4連結に達する

    最小個数を返す (達しなければ None)。既存の同色ぷよとの連結も含めて判定する
    (`_group_size_after_drop` が盤面全体を BFS するため)。
    """
    work = board.copy()
    for i in range(drop_max):
        row = _drop_row(heights[x] + i)
        if row is None:
            break
        work.set(row, x, color)
        if _group_size_after_drop(work, row, x, color) >= GROUP4_TRIGGER:
            return i + 1
    return None


# ============================
# 候補適用 + 妥当性チェック
# ============================


def _apply_candidate(
    root: Board, heights: "list[int]", cand: Candidate, skip_cut_check: bool = False,
) -> "Board | None":
    """quiet.cpp:140-170 (dfs内) の移植: 候補を root (蓄積生盤面) に適用した plan を返す。

    オーバーフロー・チェイン切断・窒息チェックで無効なら None。
    heights は root 自身の高さ (候補生成に使った field_board の高さとは別物)。

    Args:
        skip_cut_check: True のとき、チェイン切断チェック (quiet.cpp:168-170) を
            スキップする。ama では、この cut チェックは `dfs()` (2手目以降の
            延長) にのみ存在し、`search()` の初手 (quiet.cpp:12-115) には無い。
            初手は field_board == root であるため、`_vertical_trigger_need` が
            「field 上でちょうど4連結に達する」よう選んだ need を root
            (=field と同一) に再適用すると必ず4連結に達し、cut チェックを
            適用すると初手が恒久的に自己棄却されてしまう
            (実装中に実際に踏んで気付いたバグ)。呼び出し側 (ama_build) は
            x_ban<0 (=まだ何も採用していない=初手) のときに True を渡す。
    """
    overflow_pad = 1 if cand.x == SPAWN_COL else 0
    if heights[cand.x] + cand.need + overflow_pad > BOARD_ROWS - 1:
        return None

    plan = root.copy()
    if cand.direction == 0:
        for i in range(cand.need):
            row = _drop_row(heights[cand.x] + i)
            if row is None:
                return None
            plan.set(row, cand.x, cand.color)
    else:
        other = cand.x + cand.direction
        if not (0 <= other < BOARD_COLS):
            return None
        row_x = _drop_row(heights[cand.x])
        row_other = _drop_row(heights[other])
        if row_x is None or row_other is None:
            return None
        plan.set(row_x, cand.x, cand.color)
        plan.set(row_other, other, cand.color)

    if plan.is_dead():
        return None

    # quiet.cpp:168-170 のチェイン切断チェック (最初のセルの連結が3連結以下であること)。
    # NOTE: root と field_board では構造が異なるため、field_board 上で
    # 「ちょうど4連結に達する」よう選ばれた need も、root へ適用すると
    # 4連結に達しない (=このチェックを素通りする) のが正常系である
    # (2手目以降。初手は skip_cut_check=True で本チェック自体を行わない)。
    if not skip_cut_check:
        first_row = _drop_row(heights[cand.x])
        if first_row is not None:
            cut_group = _group_size_after_drop(plan, first_row, cand.x, cand.color)
            if cut_group > 3:
                return None
    return plan


# ============================
# ama eval.cpp 移植: 評価式 (freestyle プロファイル部分移植)
# ============================


def _get_chi(heights: "list[int]", x: int) -> int:
    """eval.cpp:127-168 `eval::get_chi` の移植 (発火点の横方向拡張余地)。"""
    chi = 0
    if x < BOARD_COLS - 1:
        for i in range(x + 1, BOARD_COLS):
            if heights[i] > heights[x]:
                break
            chi += 1
        for i in range(x + 1, BOARD_COLS):
            if heights[i] >= heights[x]:
                break
            chi += 1
    if x > 0:
        for i in range(x - 1, -1, -1):
            if heights[i] > heights[x]:
                break
            chi += 1
        for i in range(x - 1, -1, -1):
            if heights[i] >= heights[x]:
                break
            chi += 1
    return chi


def _get_shape(heights: "list[int]") -> int:
    """eval.cpp:171-188 `eval::get_shape` の移植 (freestyle: coef=[2,2,2,-2,-2,-2])。"""
    coef = (2, 2, 2, -2, -2, -2)
    avg = sum(heights) // BOARD_COLS
    return sum(abs(h - avg - c) for h, c in zip(heights, coef))


def _get_well(heights: "list[int]") -> int:
    """eval.cpp:192-211 `eval::get_well` の移植。"""
    well = 0
    if heights[0] < heights[1]:
        well += heights[1] - heights[0]
    if heights[5] < heights[4]:
        well += heights[4] - heights[5]
    for i in range(1, 5):
        if heights[i] < heights[i - 1] and heights[i] < heights[i + 1]:
            well += min(heights[i - 1], heights[i + 1]) - heights[i]
    return well


def _get_bump(heights: "list[int]") -> int:
    """eval.cpp:215-226 `eval::get_bump` の移植。"""
    bump = 0
    for i in range(1, 5):
        if heights[i] > heights[i - 1] and heights[i] > heights[i + 1]:
            bump += heights[i] - max(heights[i - 1], heights[i + 1])
    return bump


def _get_link_23(final_planes: "dict[int, np.ndarray]") -> "tuple[int, int]":
    """eval.cpp:266-295 `eval::get_link_23` の移植 (numpy版、SIMD不使用)。

    `src.chain_bitboard.board_to_planes` (公開API) の出力形式 (色別 uint16
    配列 shape=(6,)) に対し、2連結・3連結セル数をビットシフト+AND/ORで計算する
    (chain_bitboard.py 自体は変更せず、公開関数を呼ぶだけ)。
    """
    mask13 = np.uint16((1 << BOARD_ROWS) - 1)
    link_2 = 0
    link_3 = 0
    for color in SCAN_COLORS:
        m = final_planes[color].astype(np.uint16) & mask13
        u = (m >> np.uint16(1)) & mask13 & m
        d = (m << np.uint16(1)) & mask13 & m
        r = np.zeros_like(m)
        r[:-1] = m[1:]
        r = r & m
        left = np.zeros_like(m)
        left[1:] = m[:-1]
        left = left & m

        ud_and = u & d
        lr_and = left & r
        ud_or = u | d
        lr_or = left | r

        l3 = (ud_or & lr_or) | ud_and | lr_and
        # get_expand(l3) の代用 (自身+上下左右1マス膨張) をビット単位で計算。
        l3_exp = l3.copy()
        l3_exp |= (l3 >> np.uint16(1)) & mask13
        l3_exp |= (l3 << np.uint16(1)) & mask13
        l3_exp_r = np.zeros_like(l3)
        l3_exp_r[:-1] = l3[1:]
        l3_exp_l = np.zeros_like(l3)
        l3_exp_l[1:] = l3[:-1]
        l3_exp |= l3_exp_r
        l3_exp |= l3_exp_l

        l2 = (~l3_exp) & (u | left) & mask13
        link_2 += int(sum(bin(int(v)).count("1") for v in l2))
        link_3 += int(sum(bin(int(v)).count("1") for v in l3))
    return link_2, link_3


def _eval_score(
    plan: Board,
    plan_heights: "list[int]",
    cand: Candidate,
    original_height_x: int,
    sim_score: ScoreSimResult,
    key_count: int,
    weight: Weight,
) -> float:
    """eval.cpp:11-92 `eval::evaluate` の部分移植 (freestyle プロファイル)。

    user伝授のドメイン修正: 第一項は元 `chain.count * weight.chain` だったが、
    「飽和の本質は連鎖数でなく得点」との指示により、実測得点 (sim_score.total_score)
    そのものを第一項とする (火力=得点を直接最大化する)。残りの構造的副項
    (key/chi/y/link_2/link_3/shape/well/bump) は freestyle 重みのまま維持し、
    同点時のタイブレーク・構築方向のガイドとして使う。

    scope縮小 (正直な注記): 本プロトタイプは「盤面構成」の評価であり、ama原典の
    tear/waste (ツモ配置由来の手の非効率)・nuisance/side (お邪魔・左右偏り)・
    waste_14/u (14段目・u字形) は「1手の配置」文脈固有のため対象外とする。
    """
    heights_for_chi = list(plan_heights)
    heights_for_chi[cand.x] = original_height_x

    score = float(sim_score.total_score)
    score += original_height_x * weight.y
    score += key_count * weight.key
    score += _get_chi(heights_for_chi, cand.x) * weight.chi

    final_planes = board_to_planes(sim_score.final_board)
    link_2, link_3 = _get_link_23(final_planes)
    score += link_2 * weight.link_2
    score += link_3 * weight.link_3

    score += _get_shape(plan_heights) * weight.shape
    score += _get_well(plan_heights) * weight.well
    score += _get_bump(plan_heights) * weight.bump
    return score


# ============================
# 発火点フルスキャン (before/終端計測の共通処理)
# ============================


def full_scan_best_score(
    board: Board, colors: "tuple[int, ...]" = SCAN_COLORS,
) -> "tuple[int, int, Board | None]":
    """色数×6列の1個追加発火を全探索し、最大得点(火力)を返す。

    「現在盤面が1手で到達できる最大得点」= before値、および構成ループ終端の
    計測に共通利用する。連鎖数は参考値として併記する
    (user伝授のドメイン修正: 目的は得点、連鎖数は参考)。

    Args:
        colors: スキャン対象色。呼び出し側で「その試合の active_colors
            (通常4色)」を渡すこと。

    Returns:
        (best_score, chain_count_at_best_score, best_board)。
    """
    if board.is_dead():
        return 0, 0, None
    best_score = -1
    best_chain = 0
    best_board: "Board | None" = None
    for col in range(BOARD_COLS):
        for color in colors:
            dropped = _drop_one_color(board, col, color)
            if dropped is None:
                continue
            sim = _simulate_with_score(dropped)
            if sim.total_score > best_score:
                best_score = sim.total_score
                best_chain = sim.chain_count
                best_board = dropped
    if best_board is None:
        return 0, 0, None
    return best_score, best_chain, best_board


# ============================
# 構成ループ本体 (growth guard 付き / 無し 切替可能)
# ============================


def ama_build(
    seed: Board,
    active_colors: "tuple[int, ...]",
    guard_enabled: bool = True,
    drop_budget: int = GENERATE_DROP_BUDGET,
    fill_ratio: float = FILL_RATIO_DEFAULT,
    max_iters: int = MAX_CONSTRUCT_ITERS,
) -> BuildResult:
    """ama quiet::dfs 移植の構成ループ (単線greedy版、得点ベース)。

    Args:
        active_colors: この盤面が属する試合で実際に降る色の集合 (通常4色、
            `_compute_active_colors_by_game` で試合内出現頻度上位4色として求める)。
            候補生成・before/after計測のすべてでこの色集合のみを使う
            (user伝授のドメイン修正: 1試合につき5色中1色が除外される)。

    毎イテレーション、field_board (直前採用候補の発火後残骸、初期値は root と
    同じ) を基準に候補手を全列挙し、各候補を root (蓄積生盤面) に適用した plan
    に対して「実際に今発火させたら何点になるか」を実シミュ検証する。
    growth guard (quiet.cpp:200 の移植、得点版) が有効な場合、直前の pre_score
    を超える候補のみが採用対象になる (最良候補を1つ採用)。候補が尽きる
    (デッドロック) か飽和率に達したら終了し、最後に full_scan_best_score で
    最終到達得点を計測する。

    簡略化の明記 (正直な注記): ama の quiet::dfs は複数分岐を持つ再帰探索
    (DFS) だが、本プロトタイプは「各深さで最良候補を1つ採用する単線greedy」に
    簡略化している (計算コストを1盤面あたり現実的な時間に抑えるため)。
    growth guard という骨格破壊防止の核心ロジック、および root/field 二状態
    管理は完全に保持している。
    """
    t0 = time.perf_counter()
    root = seed.copy()
    field_board = root.copy()  # 初期状態: まだ何も発火していないので field=root
    before_score, before_chain_ref, _ = full_scan_best_score(root, colors=active_colors)
    pre_score = 0
    pre_chain_ref = 0
    x_ban = -1
    trace: "list[tuple[int, Candidate, int, int, int]]" = []
    n_sim_calls = 0
    target_cells = round(fill_ratio * FULL_BOARD_CAP)

    for it in range(max_iters):
        if root.count_puyos() >= target_cells or root.is_dead():
            break
        field_heights = _heights(field_board)
        root_heights = _heights(root)
        x_min, x_max = _get_bound(field_heights)
        candidates = _generate_candidates(
            field_board, field_heights, x_min, x_max, x_ban, drop_budget, colors=active_colors,
        )

        is_first_move = x_ban < 0
        best_score = None
        best_plan = None
        best_sim: "ScoreSimResult | None" = None
        best_cand = None
        for cand in candidates:
            plan = _apply_candidate(root, root_heights, cand, skip_cut_check=is_first_move)
            if plan is None:
                continue
            plan_heights = _heights(plan)
            if not _is_reachable(plan_heights, x_ban):
                continue
            sim = _simulate_with_score(plan)
            n_sim_calls += 1
            if guard_enabled and not (sim.total_score > pre_score):
                continue
            key_count = plan.count_puyos() - root.count_puyos()
            score = _eval_score(
                plan, plan_heights, cand, root_heights[cand.x], sim, key_count, FREESTYLE_WEIGHT,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_plan = plan
                best_sim = sim
                best_cand = cand

        if best_plan is None or best_sim is None:
            break  # growth guard を満たす手が尽きた (デッドロック終端)

        root = best_plan
        field_board = best_sim.final_board
        pre_score = best_sim.total_score
        pre_chain_ref = best_sim.chain_count
        if x_ban < 0:
            x_ban = best_cand.x
        trace.append((it, best_cand, pre_score, pre_chain_ref, root.count_puyos()))

    final_score, final_chain_ref, _ = full_scan_best_score(root, colors=active_colors)
    elapsed = time.perf_counter() - t0

    if pre_score >= final_score:
        result_score, result_chain_ref = pre_score, pre_chain_ref
    else:
        result_score, result_chain_ref = final_score, final_chain_ref

    return BuildResult(
        seed_puyo_count=seed.count_puyos(),
        before_score=before_score,
        before_chain_ref=before_chain_ref,
        final_score=result_score,
        final_chain_ref=result_chain_ref,
        root=root,
        iterations=len(trace),
        n_sim_calls=n_sim_calls,
        elapsed_sec=elapsed,
        trace=trace,
    )


# ============================
# 難所盤面の収集 (5盤面、参考連鎖数に幅を持たせて選定)
# ============================

# ⚠️ 実データ検証で判明 (2026-07-22): npz は「1動画=1試合」ではなく
# 「1動画=複数試合 (game_idx で区別)」だった (v29.npz は game_idx 0/1/2 の
# 3試合を含む)。さらに「出現の有無」だけで active_colors を決めると5色とも
# "出現あり" になってしまう: 除外色も認識ノイズ (誤認識) で数%のフレームに
# 数十〜百セル程度だけ紛れ込む (実測: v29 game0 で色1が85セル/全体4369セル・
# 出現フレーム20%なのに対し、色2-5は600-1400セル・60-80%出現)。よって
# 「出現有無」ではなく「出現頻度で下位1色を除外」する頻度ベース判定に修正する。


def _compute_active_colors_by_game(
    npz_path: Path,
) -> "dict[tuple[str, int], tuple[int, ...]]":
    """(video_id, game_idx) ごとに、出現頻度上位4色を active_colors として返す。

    user伝授のドメイン修正 (2026-07-22): 1試合ごとに5色中1色がランダムに除外
    される。実データでは除外色も認識ノイズで少数セル出現するため、単純な
    「出現の有無」では判定できず、出現セル数の下位1色を除外する
    (ACTIVE_COLOR_KEEP_COUNT=4 で上位4色を採用)。
    """
    data = np.load(str(npz_path), allow_pickle=True)
    grids = data["grids"]
    video_ids = data["video_id"]
    game_idxs = data["game_idx"]

    result: "dict[tuple[str, int], tuple[int, ...]]" = {}
    keys = {(str(v), int(g)) for v, g in zip(video_ids, game_idxs)}
    for video_id, game_idx in keys:
        mask = (video_ids == video_id) & (game_idxs == game_idx)
        sub_grids = grids[mask]
        counts = {c: int(np.sum(sub_grids == c)) for c in TRACKED_COLORS}
        ranked = sorted(TRACKED_COLORS, key=lambda c: counts[c], reverse=True)
        active = tuple(sorted(ranked[:ACTIVE_COLOR_KEEP_COUNT]))
        result[(video_id, game_idx)] = active
    return result


def _load_candidate_boards(
    npz_paths: "list[Path]", sample_per_file: int = 400,
) -> "list[tuple[Board, tuple[str, int]]]":
    """複数 npz からランダムサンプルした盤面を集める (窒息盤面は除外)。

    各盤面に (video_id, game_idx) を紐付けて返す。active_colors
    (試合別4色) の解決キーとして使う (user伝授のドメイン修正)。
    """
    boards: "list[tuple[Board, tuple[str, int]]]" = []
    for p in npz_paths:
        if not p.exists():
            continue
        data = np.load(str(p), allow_pickle=True)
        grids = data["grids"]
        video_ids = data["video_id"]
        game_idxs = data["game_idx"]
        # 注意: 組込み hash() は文字列に対しプロセスごとにランダム化される
        # (PYTHONHASHSEED)。再現性確保のため zlib.crc32 の決定的ハッシュを使う。
        rng = np.random.default_rng(zlib.crc32(p.name.encode()) & 0xFFFF)
        n = min(sample_per_file, len(grids))
        idxs = rng.choice(len(grids), size=n, replace=False)
        for i in idxs:
            b = Board.from_list(grids[i].tolist())
            if not b.is_dead():
                game_key = (str(video_ids[i]), int(game_idxs[i]))
                boards.append((b, game_key))
    return boards


def _select_hard_case_boards(
    boards_with_key: "list[tuple[Board, tuple[str, int]]]",
    active_colors_by_game: "dict[tuple[str, int], tuple[int, ...]]",
    n_pick: int = 5,
) -> "list[tuple[Board, tuple[int, ...]]]":
    """参考連鎖数 (試合別4色限定の1手到達最大連鎖) に幅を持たせて n_pick 盤面を選ぶ。

    注記 (正直さ): user言及の「前回の5盤面 (current=2/11/0/2/4)」の元ファイルを
    リポジトリ内で特定できなかったため、本プロトタイプでは同じ考え方
    (連鎖数=0前後・低連鎖・11連鎖の難所を含む幅) で v29系サンプルから
    独自に5盤面を選定する (前回と厳密に同一の盤面ではない点を明記する)。
    選定基準は連鎖数 (chain_ref) のままでよい (「11連鎖の難所」を見つけるための
    ラベルであり、構成ループ自体の目的関数=得点とは独立)。
    """
    scored: "list[tuple[int, Board, tuple[int, ...]]]" = []
    seen_keys: "set[bytes]" = set()
    for b, game_key in boards_with_key:
        key = b._grid.tobytes()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        colors = active_colors_by_game[game_key]
        _score, chain_ref, _board = full_scan_best_score(b, colors=colors)
        scored.append((chain_ref, b, colors))

    # 優先度: 11連鎖 (難所本命) → 低連鎖(0-4) を幅広く → それ以外。
    target_values = [11, 0, 2, 4, 6]
    picked: "list[tuple[Board, tuple[int, ...]]]" = []
    used_idx: "set[int]" = set()
    for target in target_values:
        best_i, best_diff = None, None
        for i, (chain_ref, _b, _c) in enumerate(scored):
            if i in used_idx:
                continue
            diff = abs(chain_ref - target)
            if best_diff is None or diff < best_diff:
                best_diff, best_i = diff, i
        if best_i is not None:
            used_idx.add(best_i)
            _, b, colors = scored[best_i]
            picked.append((b, colors))
        if len(picked) >= n_pick:
            break
    return picked


def main() -> None:
    print("=== ama構成ループ プロトタイプ (難所盤面 before/after 検証、得点ベース) ===\n")
    npz_dir = Path("data/indicators_v2/boards")
    npz_paths = [npz_dir / "v29.npz", npz_dir / "v30.npz", npz_dir / "v31.npz"]

    active_colors_by_game: "dict[tuple[str, int], tuple[int, ...]]" = {}
    for p in npz_paths:
        if not p.exists():
            continue
        per_game = _compute_active_colors_by_game(p)
        active_colors_by_game.update(per_game)
        for (video_id, game_idx), colors in sorted(per_game.items()):
            print(f"試合 {video_id} game{game_idx}: active_colors={colors}")
    print("")

    boards = _load_candidate_boards(npz_paths)
    print(f"サンプル読み込み: {len(boards)} 盤面 (重複除去前)\n")

    hard_cases = _select_hard_case_boards(boards, active_colors_by_game, n_pick=5)
    print(f"選定した難所盤面: {len(hard_cases)} 件\n")

    for idx, (board, colors) in enumerate(hard_cases):
        print(f"--- 盤面 #{idx} (active_colors={colors}) ---")
        board.display()

        print("[成長ガード有り (本命)]")
        result_guard = ama_build(board, active_colors=colors, guard_enabled=True)
        _print_result(result_guard)

        print("[成長ガード無し (アブレーション: 評価式のみ)]")
        result_noguard = ama_build(board, active_colors=colors, guard_enabled=False)
        _print_result(result_noguard)
        print("")

    print("=== 完了 ===")


def _print_result(r: BuildResult) -> None:
    print(
        f"  得点: before={r.before_score} -> after={r.final_score} "
        f"(delta={r.final_score - r.before_score:+d}) "
        f"[参考連鎖数: before={r.before_chain_ref} -> after={r.final_chain_ref}] "
        f"iterations={r.iterations} sim_calls={r.n_sim_calls} "
        f"elapsed={r.elapsed_sec:.2f}s "
        f"final_puyo_count={r.root.count_puyos()}/{FULL_BOARD_CAP}"
    )


if __name__ == "__main__":
    main()
