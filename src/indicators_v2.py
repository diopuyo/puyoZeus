"""第1バッチ 有利不利判定指標 (指標 v2) — チャンク1。

`docs/INDICATOR_V2_MEASUREMENT_SPEC_2026-06-18_ESTABLISHED.md` の実装仕様に従う。
セイバー流ボトムアップ方針: 単純な観測指標を多数提供し、重要度・閾値は学習で発見する。

設計方針 (CLAUDE.md 準拠):
    - 各指標は stateless 関数として実装 (state は外部 wrapper の責務)。
    - 指標値は 0〜1 正規化を必須とする。生値も併せて保持できるよう
      `IndicatorV2Value(score, raw)` で返す。
    - 正規化定数は仕様書の暫定値。実データ分布から後決定する
      (各定数に「データ後決定」コメントを付す)。
    - 既存資産 (chain.py / scoring.py / ojama_accounting.py / old/indicators.py /
      board.py) を流用する。

チャンク1の範囲 (既存流用中心・低リスク):
    ① 手数 / 盤面ぷよ総数 / マージンタイムrate
    ② 最大列高さ / 列凸凹 / 窒息余裕 (近接3列max併設)
    ③ 現在の最大連鎖数 / 即発火火力 / 連鎖効率 / 発火までの最短手数 /
      連結観測 (2連結数/3連結数/最大連結サイズ・色別+合計) / セカンド潜在
    ④ net収支 / forecast / 盤面お邪魔数
    ⑤ 連鎖所要時間 (観測=chain_event / 推定)
    ⑥ 受け力: 掘り耐性 / 吸収余地

除外 (チャンク2): 到達火力 (III-3, place_pair新規 + プロファイル要) /
                 WC変種 (III-8) / 連鎖所要時間定数 FRAMES_PER_CHAIN の実測更新。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    Board,
    VISIBLE_ROWS,
)
from src.chain import ChainResult, ChainSimulator
from src.scoring import (
    OJAMA_RATE_STANDARD,
    calculate_chain_score,
    compute_effective_rate,
    score_to_ojama,
)

# ============================
# 正規化定数 (暫定: 実データ分布から後決定)
# ============================

# I-1 手数: 1 試合の総手数概算 (上級者は 1 試合 ~50-100 手)。データ後決定。
NORM_TSUMO_COUNT: float = 100.0
# I-2 盤面ぷよ総数: 可視 12 行 × 6 列 = 72 セルが上限 (確定値)。
ON_FIELD_CAP: int = VISIBLE_ROWS * BOARD_COLS  # = 72

# II-1 最大列高さ / II-3 窒息余裕: 可視高さ 12 が上限 (確定値)。
MAX_COL_HEIGHT: int = VISIBLE_ROWS  # = 12
# II-2 列凸凹: 隣接列差の総和。最大 = 60 (5 隣接 × 12)。暫定。データ後決定。
NORM_BUMPINESS: float = 60.0
# II-3 窒息余裕: 近接 3 列 = 1,2,3 列目 (0-indexed)。仕様書準拠。
DEATH_NEIGHBOR_COLS: tuple[int, ...] = (1, 2, 3)

# III-1 現在の最大連鎖数: eスポーツ上級者の実用上限 ~19 連鎖。暫定。データ後決定。
NORM_MAX_CHAIN: float = 19.0
# III-4 連鎖効率 (即発火お邪魔 / 色ぷよ数) 正規化分母。暫定 2.0。データ後決定。
CHAIN_EFF_MAX: float = 2.0
# III-5 発火までの最短手数: 1〜N 個。N/MAX で楽観評価 (1個単位)。
IGNITION_MAX_PUYOS: int = 6
# III-5 探索する追加ぷよ数の上限 (N=1: 30通り, N=2: 900通り)。
IGNITION_TRIAL_LIMIT: int = 2
# III-5 探索対象色 (eスポ標準5色)。
from src.board import (  # noqa: E402  (定数群を末尾 import で整理)
    COLOR_RED,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_PURPLE,
)

IGNITION_TRIAL_COLORS: tuple[int, ...] = (
    COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE,
)

# IV: お邪魔系正規化 (盤面容量 72 で割る)。net収支は (x+72)/144。
OJAMA_NET_NORM_HALF: int = ON_FIELD_CAP  # 72
OJAMA_NET_NORM_FULL: int = ON_FIELD_CAP * 2  # 144

# V 連鎖所要時間: 観測秒の正規化分母 (暫定 14 秒)。データ後決定。
NORM_CHAIN_DURATION_SEC: float = 14.0
# V-2 推定: 1 連鎖あたり frame 数。
# ⚠️ TODO(チャンク2): CHAIN_DURATION_FRAMES_PER_CHAIN=84 はぷよ通由来で
#   eスポーツ用に未検証 (eスポはアニメ短縮で 30-50 frame/連鎖の可能性)。
#   実動画の chain_event start/end_sec を連鎖数で割って実測し定数更新が必須。
FRAMES_PER_CHAIN: float = 84.0
# V-2 推定 FPS (60fps 想定)。
ASSUMED_FPS: float = 60.0

# VI-1 掘り耐性: 仮想お邪魔テスト個数。
OJAMA_DEFENSE_TEST_COUNTS: tuple[int, ...] = (10, 20, 30)
# VI-1 掘り耐性: 掘削可能と判定する最小連鎖数。
OJAMA_DEFENSE_DIG_MIN_CHAIN: int = 1
# VI-1 掘り耐性: 生存率 / 掘削可能性の重み。
OJAMA_DEFENSE_SURVIVAL_WEIGHT: float = 0.7
OJAMA_DEFENSE_DIG_WEIGHT: float = 0.3

# 連結観測対象の最小サイズ (2連結 / 3連結)。
GROUP_OBSERVE_MIN_SIZE: int = 2

# 共有 simulator (LRU キャッシュ 5万件で高速化)。
_SHARED_SIMULATOR: ChainSimulator = ChainSimulator()


def _clamp01(value: float) -> float:
    """0〜1 にクランプする。"""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


@dataclass(frozen=True)
class IndicatorV2Value:
    """1 指標の算出結果 (正規化スコア + 生値)。

    Attributes:
        score: 0〜1 正規化値。
        raw: 正規化前の生値 (分布確認・デバッグ用)。
    """
    score: float
    raw: float


# ============================
# ① 進行度
# ============================


def tsumo_count_rate(tsumo_count: int) -> IndicatorV2Value:
    """I-1 手数。RecognitionPipeline.tsumo_count(side) の値を /NORM で正規化。

    Args:
        tsumo_count: 試合開始からの確定ツモ設置数 (手数)。

    Returns:
        IndicatorV2Value: score=手数/NORM_TSUMO_COUNT, raw=手数。
    """
    raw = float(max(0, tsumo_count))
    return IndicatorV2Value(score=_clamp01(raw / NORM_TSUMO_COUNT), raw=raw)


def board_puyo_total(board: Board) -> IndicatorV2Value:
    """I-2 盤面ぷよ総数 (お邪魔含む)。count_puyos()/72。"""
    raw = float(board.count_puyos())
    return IndicatorV2Value(score=_clamp01(raw / ON_FIELD_CAP), raw=raw)


def board_color_puyo_total(board: Board) -> IndicatorV2Value:
    """I-2 派生: 色ぷよ版 (お邪魔・UNKNOWN除く) /72。"""
    raw = float(_count_color_puyos(board))
    return IndicatorV2Value(score=_clamp01(raw / ON_FIELD_CAP), raw=raw)


def margin_time_rate(elapsed_sec: float) -> IndicatorV2Value:
    """I-3 経過時間 = マージンタイムrate。1 - effective_rate/70。

    0=序盤 (減衰なし) 〜 1=マージン最大 (レート最小)。

    Args:
        elapsed_sec: 試合開始からの経過秒 (OjamaAccountingTracker._elapsed 相当)。

    Returns:
        IndicatorV2Value: score=rate, raw=有効レート(個/70点)。
    """
    rate = compute_effective_rate(elapsed_sec, OJAMA_RATE_STANDARD)
    score = 1.0 - (rate / float(OJAMA_RATE_STANDARD))
    return IndicatorV2Value(score=_clamp01(score), raw=float(rate))


# ============================
# ② 占有・危険
# ============================


def max_column_height(board: Board) -> IndicatorV2Value:
    """II-1 最大列高さ。max(height_of(c))/12 (お邪魔も算入)。"""
    raw = float(max(board.height_of(c) for c in range(BOARD_COLS)))
    return IndicatorV2Value(score=_clamp01(raw / MAX_COL_HEIGHT), raw=raw)


def column_bumpiness(board: Board) -> IndicatorV2Value:
    """II-2 列凸凹。Σ|h[c+1]-h[c]|/60 (meatfighter 評価関数準拠)。"""
    heights = [board.height_of(c) for c in range(BOARD_COLS)]
    raw = float(sum(abs(heights[c + 1] - heights[c]) for c in range(BOARD_COLS - 1)))
    return IndicatorV2Value(score=_clamp01(raw / NORM_BUMPINESS), raw=raw)


def death_margin(board: Board) -> IndicatorV2Value:
    """II-3 窒息余裕 (メイン)。(12 - height_of(2))/12。

    値が大きいほど窒息列に余裕がある = 安全。
    """
    from src.board import DEATH_COL
    raw = float(MAX_COL_HEIGHT - board.height_of(DEATH_COL))
    return IndicatorV2Value(score=_clamp01(raw / MAX_COL_HEIGHT), raw=raw)


def death_margin_neighbor(board: Board) -> IndicatorV2Value:
    """II-3 窒息余裕 (近接3列補助)。(12 - max(h[1],h[2],h[3]))/12。"""
    max_h = max(board.height_of(c) for c in DEATH_NEIGHBOR_COLS)
    raw = float(MAX_COL_HEIGHT - max_h)
    return IndicatorV2Value(score=_clamp01(raw / MAX_COL_HEIGHT), raw=raw)


# ============================
# ③ 火力・潜在
# ============================


def current_max_chain(
    board: Board, simulator: ChainSimulator | None = None,
) -> IndicatorV2Value:
    """III-1 現在の最大連鎖数 (takapt 定石)。

    各列×各色 最大 30 通り 1 個落として simulate し、最大連鎖数を返す。
    `simulate(静止盤面)` 直接呼び出しは消せる 4 連結が無くほぼ 0 のため
    (takapt 定石: 各列に 1 色 1 個落とした時の最大連鎖) に変更。
    正規化: best_chain / NORM_MAX_CHAIN (暫定 19)。

    Args:
        board: 評価対象の確定盤面 (STABLE 時のみ)。
        simulator: ChainSimulator インスタンス (省略時は共有インスタンス)。
    """
    sim = simulator or _SHARED_SIMULATOR
    best_chain, _ = _takapt_best_drop(board, sim)
    raw = float(best_chain)
    return IndicatorV2Value(score=_clamp01(raw / NORM_MAX_CHAIN), raw=raw)


def immediate_fire_power(
    board: Board,
    elapsed_sec: float = 0.0,
    simulator: ChainSimulator | None = None,
) -> IndicatorV2Value:
    """III-2 即発火火力 (takapt 定石)。

    III-1 の takapt 探索で得た最良配置盤面に calculate_chain_score を適用し
    score_to_ojama でお邪魔換算する。/72 正規化。
    best_board が None (全列満杯 等) の場合は 0。

    Args:
        board: 評価対象の確定盤面 (STABLE 時のみ)。
        elapsed_sec: 試合相対経過秒 (マージンタイム用)。
        simulator: ChainSimulator インスタンス (省略時は共有インスタンス)。
    """
    sim = simulator or _SHARED_SIMULATOR
    _, best_board = _takapt_best_drop(board, sim)
    if best_board is None:
        return IndicatorV2Value(score=0.0, raw=0.0)
    ojama = _board_fire_ojama(best_board, elapsed_sec, sim)
    return IndicatorV2Value(
        score=_clamp01(float(ojama) / ON_FIELD_CAP), raw=float(ojama),
    )


def chain_efficiency(
    board: Board,
    elapsed_sec: float = 0.0,
    simulator: ChainSimulator | None = None,
) -> IndicatorV2Value:
    """III-4 連鎖効率 (takapt 定石の副産物)。

    即発火お邪魔 (III-2 相当) ÷ 色ぷよ総数 (密度)。/CHAIN_EFF_MAX 正規化。
    III-2 と同じ takapt 探索結果を再利用するため追加コスト < キャッシュヒット。
    色ぷよが 0 個なら 0。

    Args:
        board: 評価対象の確定盤面 (STABLE 時のみ)。
        elapsed_sec: 試合相対経過秒 (マージンタイム用)。
        simulator: ChainSimulator インスタンス (省略時は共有インスタンス)。
    """
    sim = simulator or _SHARED_SIMULATOR
    _, best_board = _takapt_best_drop(board, sim)
    if best_board is None:
        ojama = 0
    else:
        ojama = _board_fire_ojama(best_board, elapsed_sec, sim)
    color_count = _count_color_puyos(board)
    if color_count <= 0:
        raw = 0.0
    else:
        raw = float(ojama) / float(color_count)
    return IndicatorV2Value(score=_clamp01(raw / CHAIN_EFF_MAX), raw=raw)


def min_puyos_to_ignite(
    board: Board, simulator: ChainSimulator | None = None,
) -> IndicatorV2Value:
    """III-5 発火までの最短手数。N 個落として連鎖が伸びる最小 N。

    `1 - N/IGNITION_MAX_PUYOS` (近いほど高スコア)。1 個単位なので楽観的。
    src/old/indicators.py の `_min_puyos_to_ignite` を流用移植。

    Returns:
        IndicatorV2Value: score=1-N/MAX, raw=N (発火不能なら TRIAL_LIMIT+1)。
    """
    sim = simulator or _SHARED_SIMULATOR
    base_chain = sim.simulate(board).chain_count
    n = _search_min_ignite(board, sim, base_chain, IGNITION_TRIAL_LIMIT)
    score = 1.0 - (float(n) / float(IGNITION_MAX_PUYOS))
    return IndicatorV2Value(score=_clamp01(score), raw=float(n))


@dataclass(frozen=True)
class GroupObservation:
    """III-6 連結観測の生値 (色別 + 合計)。

    Attributes:
        pair_count: 2 連結 (size==2) のグループ数。
        triple_count: 3 連結 (size==3) のグループ数 = ポップ寸前。
        max_group_size: 最大連結サイズ。
    """
    pair_count: int
    triple_count: int
    max_group_size: int


def connectivity_observation(
    board: Board, simulator: ChainSimulator | None = None,
) -> tuple[GroupObservation, dict[int, GroupObservation]]:
    """III-6 連結観測。find_groups から 2連結数/3連結数/最大連結サイズ。

    色別 + 合計の両方を算出する (採否はデータ後)。お邪魔/UNKNOWN は対象外
    (find_groups が既に除外済み)。

    Returns:
        (合計の GroupObservation, 色 -> GroupObservation の dict)。
    """
    sim = simulator or _SHARED_SIMULATOR
    groups = sim.find_groups(board)
    total_pair = total_triple = total_max = 0
    per_color: dict[int, list[int]] = {}
    for g in groups:
        if g.size == 2:
            total_pair += 1
        elif g.size == 3:
            total_triple += 1
        total_max = max(total_max, g.size)
        per_color.setdefault(g.color, []).append(g.size)
    per_color_obs: dict[int, GroupObservation] = {}
    for color, sizes in per_color.items():
        per_color_obs[color] = GroupObservation(
            pair_count=sum(1 for s in sizes if s == 2),
            triple_count=sum(1 for s in sizes if s == 3),
            max_group_size=max(sizes),
        )
    return GroupObservation(total_pair, total_triple, total_max), per_color_obs


def second_chain_potential(
    board: Board, simulator: ChainSimulator | None = None,
) -> IndicatorV2Value:
    """III-7 セカンド潜在 / 副砲分離率。

    色ぷよ総数 − simulate(board).participating_cells (本線非参加=副砲/受け材料)。
    正規化は色ぷよ総数で割った比率 (本線に使われていない色ぷよの割合)。
    """
    sim = simulator or _SHARED_SIMULATOR
    color_count = _count_color_puyos(board)
    participating = sim.simulate(board).participating_cells
    non_participating = max(0, color_count - participating)
    raw = float(non_participating)
    if color_count <= 0:
        score = 0.0
    else:
        score = float(non_participating) / float(color_count)
    return IndicatorV2Value(score=_clamp01(score), raw=raw)


# ============================
# ④ お邪魔
# ============================


def ojama_net_balance(net_balance_capped: int) -> IndicatorV2Value:
    """IV-1 net収支。OjamaAccountSnapshot.net_balance_capped → (x+72)/144。

    0.5 が均衡、>0.5 が 1P 有利方向 (net 正)。

    Args:
        net_balance_capped: snapshot.net_balance_capped (= p2_capped - p1_capped)。
    """
    raw = float(net_balance_capped)
    score = (raw + OJAMA_NET_NORM_HALF) / OJAMA_NET_NORM_FULL
    return IndicatorV2Value(score=_clamp01(score), raw=raw)


def ojama_forecast(forecast: int) -> IndicatorV2Value:
    """IV-2 forecast。forecast_p1/p2 を /72 正規化。

    Args:
        forecast: snapshot.forecast_p1 または forecast_p2 (自分に向かう予告個数)。
    """
    raw = float(max(0, forecast))
    return IndicatorV2Value(score=_clamp01(raw / ON_FIELD_CAP), raw=raw)


def board_ojama_count(board: Board) -> IndicatorV2Value:
    """IV-3 盤面お邪魔数。可視領域のお邪魔ぷよ数/72。"""
    raw = float(_count_visible_ojama(board))
    return IndicatorV2Value(score=_clamp01(raw / ON_FIELD_CAP), raw=raw)


# ============================
# ⑤ テンポ
# ============================


def chain_duration_observed(
    start_sec: float | None, end_sec: float | None,
) -> IndicatorV2Value | None:
    """V-1 連鎖所要時間 (観測)。chain_event.end_sec - start_sec。

    = 相手に与える猶予時間。実測値がなければ None を返し、呼出側が
    chain_duration_estimated にフォールバックする。

    Args:
        start_sec: ChainEvent.trigger_sec (連鎖発火直前時刻)。
        end_sec: ChainEvent.end_sec (連鎖が落ち着いた時刻)。

    Returns:
        IndicatorV2Value (score=秒/14, raw=秒) または None (実測不能)。
    """
    if start_sec is None or end_sec is None:
        return None
    duration = float(end_sec) - float(start_sec)
    if duration < 0.0:
        return None
    return IndicatorV2Value(
        score=_clamp01(duration / NORM_CHAIN_DURATION_SEC), raw=duration,
    )


def chain_duration_estimated(chain_count: int) -> IndicatorV2Value:
    """V-2 連鎖所要時間 (推定)。chain_count × FRAMES_PER_CHAIN / FPS。

    ⚠️ FRAMES_PER_CHAIN=84 はぷよ通由来で eスポーツ用に未検証
    (チャンク2 で実測更新予定)。
    """
    frames = float(max(0, chain_count)) * FRAMES_PER_CHAIN
    duration = frames / ASSUMED_FPS
    return IndicatorV2Value(
        score=_clamp01(duration / NORM_CHAIN_DURATION_SEC), raw=duration,
    )


# ============================
# ⑥ 受け力 (守備)
# ============================


def dig_resistance(
    board: Board, simulator: ChainSimulator | None = None,
) -> IndicatorV2Value:
    """VI-1 掘り耐性。お邪魔 10/20/30 個落下後の本線生存度 + 掘削可能性。

    src/old/indicators.py の OjamaDefenseCapacityIndicator を流用移植。
    score = 平均 (0.7×survival + 0.3×dig)。0-1 スケール済み。

    限界: drop_ojama は毎ターン「左から6個ずつ均等」でお邪魔落下オフセット
    未反映 + 載りきらない分サイレントスキップ (窒息寸前の評価が甘い)。
    """
    sim = simulator or _SHARED_SIMULATOR
    if board.is_dead():
        return IndicatorV2Value(score=0.0, raw=0.0)
    base_chain = max(1, sim.simulate(board).chain_count)
    scores: list[float] = []
    for n_ojama in OJAMA_DEFENSE_TEST_COUNTS:
        scores.append(_dig_resistance_one(board, sim, base_chain, n_ojama))
    avg = sum(scores) / len(scores) if scores else 0.0
    return IndicatorV2Value(score=_clamp01(avg), raw=float(avg))


def absorption_capacity(board: Board) -> IndicatorV2Value:
    """VI-2 吸収余地 = (72 − 盤面ぷよ数)/72 = 空きセル容量 (位置は無視)。"""
    free = ON_FIELD_CAP - board.count_puyos()
    raw = float(max(0, free))
    return IndicatorV2Value(score=_clamp01(raw / ON_FIELD_CAP), raw=raw)


# ============================
# 内部ヘルパー
# ============================


def _count_color_puyos(board: Board) -> int:
    """色ぷよ数 (お邪魔・空・UNKNOWN 除く)。"""
    grid = board._grid
    mask = (
        (grid != COLOR_EMPTY)
        & (grid != COLOR_UNKNOWN)
        & (grid != COLOR_OJAMA)
    )
    return int(mask.sum())


def _count_visible_ojama(board: Board) -> int:
    """可視領域 (隠し段除く) のお邪魔ぷよ数。"""
    from src.board import HIDDEN_ROWS
    count = 0
    for row in range(HIDDEN_ROWS, BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board.get(row, col) == COLOR_OJAMA:
                count += 1
    return count


def _board_fire_ojama(
    board: Board,
    elapsed_sec: float,
    simulator: ChainSimulator | None,
) -> int:
    """指定盤面を発火した場合の送出お邪魔数を計算する (III-2/III-4 共通)。

    旧名 `_immediate_fire_ojama` を汎用化 (任意 board を渡せる版)。
    """
    sim = simulator or _SHARED_SIMULATOR
    result = sim.simulate(board)
    score = calculate_chain_score(result).total_score
    ojama = score_to_ojama(
        score=score, prev_leftover=0, elapsed_sec=elapsed_sec,
        rate_base=OJAMA_RATE_STANDARD,
    )
    return int(ojama.ojama_count)


def _takapt_best_drop(
    board: Board, sim: ChainSimulator,
) -> tuple[int, Board | None]:
    """takapt 定石: 各列×各色 (最大 30 通り) 1 個落として simulate し最良を返す。

    Args:
        board: 評価対象の盤面 (破壊しない)。
        sim: ChainSimulator インスタンス。

    Returns:
        (best_chain_count, best_board_after_drop)
        best_chain_count: 1 個追加で達成できる最大連鎖数。0 = 全列満杯 or 連鎖なし。
        best_board_after_drop: 最大連鎖を達成した 1 個追加後の盤面。
            全列満杯 等で 1 個も置けない場合は None。
    """
    best_chain: int = 0
    best_board: Board | None = None
    for col in range(BOARD_COLS):
        for color in IGNITION_TRIAL_COLORS:
            dropped = _drop_one_color(board, col, color)
            if dropped is None:
                continue  # 列満杯
            result = sim.simulate(dropped)
            if result.chain_count > best_chain:
                best_chain = result.chain_count
                best_board = dropped
    return best_chain, best_board


def _drop_one_color(board: Board, col: int, color: int) -> Board | None:
    """col 列の積み上がり最上段に color を 1 個置いた新 Board を返す。

    列が満杯 (height >= BOARD_ROWS) なら None を返す。
    board は破壊しない (copy を返す)。
    takapt 定石の基本操作。
    """
    row = _drop_row(board, col)
    if row is None:
        return None
    work = board.copy()
    work.set(row, col, color)
    return work


def _drop_row(board: Board, col: int) -> int | None:
    """指定列に 1 ぷよ落としたときの着地 row (埋まっていれば None)。"""
    height = board.height_of(col)
    if height >= BOARD_ROWS:
        return None
    return BOARD_ROWS - 1 - height


def _simulate_with_one(
    sim: ChainSimulator, board: Board, col: int, color: int,
) -> "ChainResult | None":
    """1 ぷよを col に落とした盤面で連鎖シミュレーション (置けなければ None)。

    _drop_one_color + sim.simulate の組み合わせ。
    _search_min_ignite から利用 (シミュレータをキャッシュ経由で呼ぶ)。
    """
    dropped = _drop_one_color(board, col, color)
    if dropped is None:
        return None
    return sim.simulate(dropped)


def _search_min_ignite(
    board: Board, sim: ChainSimulator, base_chain: int, trial_limit: int,
) -> int:
    """N 個追加で base_chain を超える発火が可能な最小 N を探索 (old 移植)。

    見つからなければ trial_limit + 1 を返す (= 発火困難)。
    """
    # N=1 (色 × 列 = 5 × 6 = 30 通り)
    for color in IGNITION_TRIAL_COLORS:
        for col in range(BOARD_COLS):
            res = _simulate_with_one(sim, board, col, color)
            if res is not None and res.chain_count > base_chain:
                return 1
    # N=2 (5×6 × 5×6 = 900 通り, trial_limit>=2 のときのみ)
    if trial_limit >= 2:
        for color1 in IGNITION_TRIAL_COLORS:
            for col1 in range(BOARD_COLS):
                board1 = _drop_one_color(board, col1, color1)
                if board1 is None or board1.is_dead():
                    continue
                for color2 in IGNITION_TRIAL_COLORS:
                    for col2 in range(BOARD_COLS):
                        res = _simulate_with_one(sim, board1, col2, color2)
                        if res is not None and res.chain_count > base_chain:
                            return 2
    return trial_limit + 1


def _dig_resistance_one(
    board: Board, sim: ChainSimulator, base_chain: int, n_ojama: int,
) -> float:
    """掘り耐性: お邪魔 n_ojama 個落下後の (0.7×survival + 0.3×dig)。"""
    try:
        ojama_board = sim.drop_ojama(board, n_ojama)
    except Exception:
        return 0.0
    if ojama_board.is_dead():
        return 0.0
    try:
        post = sim.simulate(ojama_board)
    except Exception:
        return 0.0
    survival = min(1.0, post.chain_count / float(base_chain))
    dig = 1.0 if post.chain_count >= OJAMA_DEFENSE_DIG_MIN_CHAIN else 0.0
    return OJAMA_DEFENSE_SURVIVAL_WEIGHT * survival + OJAMA_DEFENSE_DIG_WEIGHT * dig


__all__ = [
    "IndicatorV2Value",
    "GroupObservation",
    # ①
    "tsumo_count_rate",
    "board_puyo_total",
    "board_color_puyo_total",
    "margin_time_rate",
    # ②
    "max_column_height",
    "column_bumpiness",
    "death_margin",
    "death_margin_neighbor",
    # ③
    "current_max_chain",
    "immediate_fire_power",
    "chain_efficiency",
    "min_puyos_to_ignite",
    "connectivity_observation",
    "second_chain_potential",
    # ④
    "ojama_net_balance",
    "ojama_forecast",
    "board_ojama_count",
    # ⑤
    "chain_duration_observed",
    "chain_duration_estimated",
    # ⑥
    "dig_resistance",
    "absorption_capacity",
]
