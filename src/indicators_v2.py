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

# III-3 到達火力: early pruning の 1 手目上位選択数。
# 最大 22 配置のうちスコア上位 k のみ 2 手目展開 → 最大 k×22 sim。
REACH_FIRE_POWER_BEST_K: int = 5
# III-3 到達火力: 正規化分母 (お邪魔個数上限 = 72)。
REACH_FIRE_POWER_NORM: int = ON_FIELD_CAP  # = 72

# III-8 潜在火力: greedy ビームの 1 手目保持数。
# 1 手目 = 5×6=30 通り → 上位 K のみ 2 手目展開。最大 sim = 30 + K×30 = 180。
POTENTIAL_FIRE_POWER_BEAM_K: int = 5
# III-8 潜在火力: 最大追加ぷよ数 (デフォルト 2)。
POTENTIAL_FIRE_POWER_MAX_ADD: int = 2
# III-8 潜在火力: 正規化分母 (REACH_FIRE_POWER_NORM と統一)。
POTENTIAL_FIRE_POWER_NORM: int = ON_FIELD_CAP  # = 72

# VI-1 掘り耐性: 仮想お邪魔テスト個数。
OJAMA_DEFENSE_TEST_COUNTS: tuple[int, ...] = (10, 20, 30)
# VI-1 掘り耐性: 掘削可能と判定する最小連鎖数。
OJAMA_DEFENSE_DIG_MIN_CHAIN: int = 1
# VI-1 掘り耐性: 生存率 / 掘削可能性の重み。
OJAMA_DEFENSE_SURVIVAL_WEIGHT: float = 0.7
OJAMA_DEFENSE_DIG_WEIGHT: float = 0.3

# 連結観測対象の最小サイズ (2連結 / 3連結)。
GROUP_OBSERVE_MIN_SIZE: int = 2

# IX 形・組み品質 (connected_pair_quality)
# 「主連鎖に接続された2連結」と「孤立2連結」の閾値。
# ※隣接判定は静的な上下左右1マス接触で近似 (連鎖伝播とは異なる)。
# 同色の size >= MAIN_GROUP_MIN_SIZE を「主連鎖グループ候補」と見なす。
MAIN_GROUP_MIN_SIZE: int = 3
# 正規化分母: 上級者盤面の同色2連結数は最大でも 10 程度。暫定。データ後決定。
NORM_LINKED_PAIR: float = 10.0

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


# ============================
# III-3 到達火力 ヘルパー
# ============================


def _drop_one_inplace(board: Board, col: int, color: int) -> bool:
    """board に color を col 列の最下空セルに落下 (in-place)。

    Returns:
        True = 置けた、False = 列満杯。
    """
    row = _drop_row(board, col)
    if row is None:
        return False
    board.set(row, col, color)
    return True


def _drop_two_in_column(board: Board, col: int, upper: int, lower: int) -> bool:
    """同一列に 2 puyo を積む (lower が下・upper が上、in-place)。

    Returns:
        True = 両方置けた、False = 不可。
    """
    if board.height_of(col) > BOARD_ROWS - 2:
        return False
    if not _drop_one_inplace(board, col, lower):
        return False
    return _drop_one_inplace(board, col, upper)


def _place_pair_to_board(
    board: Board,
    pair: tuple[int, int],
    col: int,
    rotation: int,
) -> "Board | None":
    """指定列・回転で puyo ペアを盤面に配置し新しい Board を返す (板面を破壊しない)。

    22 配置の対応:
        縦: rotation=0 (TOP上/BOT下) col=0-5 → 6通り
            rotation=2 (BOT上/TOP下) col=0-5 → 6通り  = 計12通り
        横: rotation=1 (TOP左/BOT右) col=0-4 → 5通り
            rotation=3 (BOT左/TOP右) col=0-4 → 5通り  = 計10通り
    合計 22 通り。col の上限は回転に依存。

    Args:
        board: 元盤面 (破壊しない)。
        pair: (TOP色, BOT色) の組。
        col: 縦配置では落下列、横配置では左 puyo の列。
        rotation: 0=縦TOP上, 1=横TOP左, 2=縦BOT上, 3=横BOT左。

    Returns:
        Board | None: 不可なら None。
    """
    top, bot = pair
    if top == COLOR_EMPTY or bot == COLOR_EMPTY:
        return None
    work = board.copy()
    if rotation in (0, 2):
        # 縦配置 (同列に 2 puyo)
        if not (0 <= col < BOARD_COLS):
            return None
        upper, lower = (top, bot) if rotation == 0 else (bot, top)
        if not _drop_two_in_column(work, col, upper, lower):
            return None
        return work
    # 横配置: col と col+1 に各 1 puyo
    if not (0 <= col < BOARD_COLS - 1):
        return None
    left, right = (top, bot) if rotation == 1 else (bot, top)
    if not _drop_one_inplace(work, col, left):
        return None
    if not _drop_one_inplace(work, col + 1, right):
        return None
    return work


def _enumerate_placements(
    board: Board,
    pair: tuple[int, int],
    sim: ChainSimulator,
) -> "list[tuple[int, Board]]":
    """22 配置すべてを試し (chain_count, 配置後盤面) リストを返す。

    置けない組み合わせは除外 (空盤面なら最大 22 要素)。
    重複盤面はキャッシュによるシミュレーションで自動吸収。

    Returns:
        list[(chain_count, placed_board)] — chain_count 降順でソート済み。
    """
    results: list[tuple[int, Board]] = []
    for rotation in range(4):
        max_col = BOARD_COLS if rotation in (0, 2) else BOARD_COLS - 1
        for col in range(max_col):
            placed = _place_pair_to_board(board, pair, col, rotation)
            if placed is None:
                continue
            chain_result = sim.simulate(placed)
            results.append((chain_result.chain_count, placed))
    results.sort(key=lambda x: x[0], reverse=True)
    return results


@dataclass(frozen=True)
class ReachFirePowerResult:
    """III-3 到達火力の算出結果。

    Attributes:
        value: 正規化済み IndicatorV2Value (score=/72, raw=お邪魔数)。
        source: "reach" (next+dnext 両方) または "fallback_immediate" (片方 None)。
        max_chain: 到達できた最大連鎖数 (デバッグ/verify 用)。
    """
    value: IndicatorV2Value
    source: str
    max_chain: int


def reach_fire_power(
    board: Board,
    next_pair: "tuple[int, int] | None",
    dnext_pair: "tuple[int, int] | None",
    elapsed_sec: float = 0.0,
    simulator: "ChainSimulator | None" = None,
    best_k: int = REACH_FIRE_POWER_BEST_K,
) -> ReachFirePowerResult:
    """III-3 到達火力 (2 手先読み)。

    実 next/dnext ペアを 22 配置ずつ探索し、最大スコアをお邪魔換算する。
    early pruning: 1 手目の chain_count 上位 best_k (デフォルト 5) のみ 2 手目展開。
    最大 sim 数 = 22 (1 手目) + 5 × 22 (2 手目) = 132 sim/STABLE。

    next/dnext のいずれか None → immediate_fire_power (III-2) にフォールバック。
    source フィールドで区別可能。

    下界性 (検証): next+dnext 揃い時 raw >= III-2 の raw (>=III-1)。

    Args:
        board: STABLE 確定盤面。
        next_pair: (TOP色, BOT色) または None (未検知)。
        dnext_pair: (TOP色, BOT色) または None (未検知)。
        elapsed_sec: 試合相対経過秒 (マージンタイム用)。
        simulator: ChainSimulator インスタンス (省略時は共有)。
        best_k: early pruning の 1 手目保留数 (デフォルト 5)。

    Returns:
        ReachFirePowerResult: value(score/raw) + source + max_chain。
    """
    sim = simulator or _SHARED_SIMULATOR
    # next/dnext 片方でも None → フォールバック
    if next_pair is None or dnext_pair is None:
        fb = immediate_fire_power(board, elapsed_sec, sim)
        return ReachFirePowerResult(
            value=fb, source="fallback_immediate", max_chain=0,
        )
    return _reach_fire_power_core(board, next_pair, dnext_pair, elapsed_sec, sim, best_k)


def _reach_fire_power_core(
    board: Board,
    next_pair: "tuple[int, int]",
    dnext_pair: "tuple[int, int]",
    elapsed_sec: float,
    sim: "ChainSimulator",
    best_k: int,
) -> ReachFirePowerResult:
    """reach_fire_power の本体 (next/dnext 確定時のみ呼ばれる)。

    1 手目 next_pair 22 配置 → chain_count 上位 best_k 選択 →
    2 手目 dnext_pair 22 配置 → max chain_score。
    """
    # 1 手目: 22 配置 (chain_count 降順ソート済み)
    first_candidates = _enumerate_placements(board, next_pair, sim)
    if not first_candidates:
        # 1 手目配置が 0 = 全列満杯 → フォールバック
        fb = immediate_fire_power(board, elapsed_sec, sim)
        return ReachFirePowerResult(
            value=fb, source="fallback_immediate", max_chain=0,
        )
    # early pruning: 上位 best_k のみ残す
    pruned = first_candidates[:best_k]

    best_ojama = 0
    best_max_chain = 0

    for _, board_after_first in pruned:
        # 2 手目: 22 配置
        second_candidates = _enumerate_placements(board_after_first, dnext_pair, sim)
        for chain_count, board_after_second in second_candidates:
            # 2 手目配置後の連鎖スコアを計算
            result2 = sim.simulate(board_after_second)
            score2 = calculate_chain_score(result2).total_score
            ojama2 = score_to_ojama(
                score=score2, prev_leftover=0,
                elapsed_sec=elapsed_sec, rate_base=OJAMA_RATE_STANDARD,
            )
            ojama_count = int(ojama2.ojama_count)
            if ojama_count > best_ojama:
                best_ojama = ojama_count
                best_max_chain = result2.chain_count

    value = IndicatorV2Value(
        score=_clamp01(float(best_ojama) / REACH_FIRE_POWER_NORM),
        raw=float(best_ojama),
    )
    return ReachFirePowerResult(
        value=value, source="reach", max_chain=best_max_chain,
    )


# ============================
# III-8 潜在火力
# ============================


def _pfp_first_pass(
    board: Board,
    sim: ChainSimulator,
    beam_k: int,
) -> "list[tuple[int, Board]]":
    """潜在火力 1 手目 greedy: 5色×6列=30通り simulate → chain 上位 beam_k を返す。

    Args:
        board: 評価対象の確定盤面。
        sim: ChainSimulator インスタンス。
        beam_k: 保持する上位候補数。

    Returns:
        (chain_count, dropped_board) を chain_count 降順に最大 beam_k 個。
    """
    candidates: list[tuple[int, Board]] = []
    for col in range(BOARD_COLS):
        for color in IGNITION_TRIAL_COLORS:
            dropped = _drop_one_color(board, col, color)
            if dropped is None:
                continue
            chain = sim.simulate(dropped).chain_count
            candidates.append((chain, dropped))
    # chain_count 降順で上位 beam_k を返す
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:beam_k]


def _pfp_second_pass(
    candidates: "list[tuple[int, Board]]",
    sim: ChainSimulator,
    elapsed_sec: float,
) -> int:
    """潜在火力 2 手目: 各候補盤面から再度 30 通り探索し最大お邪魔数を返す。

    Args:
        candidates: 1 手目 top-K の (chain_count, board) リスト。
        sim: ChainSimulator インスタンス。
        elapsed_sec: マージンタイム計算用経過秒。

    Returns:
        2 手まで追加した場合の最大お邪魔数 (int)。
    """
    best_ojama: int = 0
    for _, board1 in candidates:
        for col in range(BOARD_COLS):
            for color in IGNITION_TRIAL_COLORS:
                board2 = _drop_one_color(board1, col, color)
                if board2 is None:
                    continue
                ojama = _board_fire_ojama(board2, elapsed_sec, sim)
                if ojama > best_ojama:
                    best_ojama = ojama
    return best_ojama


def potential_fire_power(
    board: Board,
    elapsed_sec: float = 0.0,
    simulator: "ChainSimulator | None" = None,
    max_add: int = POTENTIAL_FIRE_POWER_MAX_ADD,
) -> IndicatorV2Value:
    """III-8 潜在火力 (ツモ依存なし)。

    実ツモに依存せず「この盤面に仕込まれている最大連鎖ポテンシャル」を測る。
    任意色ぷよを最大 max_add 個・任意列に greedy ビーム探索で追加し、
    simulate して得られる最大お邪魔数を返す。

    探索 (max_add=2 デフォルト):
        1 手目: 5色×6列=30 通り simulate → chain 上位 BEAM_K=5 を保持。
        2 手目: top-K 各盤面から再度 30 通り → 最大お邪魔数。
        最大 sim 数 = 30 + 5×30 = 180 (全探索 900 の 1/5)。

    正規化: raw / POTENTIAL_FIRE_POWER_NORM (=72 =ON_FIELD_CAP)。

    Args:
        board: STABLE 確定盤面 (stateless: 破壊しない)。
        elapsed_sec: 試合相対経過秒 (マージンタイム用)。
        simulator: ChainSimulator インスタンス (省略時は共有インスタンス)。
        max_add: 最大追加ぷよ数 (デフォルト 2。1 なら 30 sim のみ)。

    Returns:
        IndicatorV2Value: score=raw/72 (0〜1), raw=最大お邪魔数。
    """
    sim = simulator or _SHARED_SIMULATOR
    # 1 手目: 30 通り探索 (chain_count 基準で top-K 候補を選択)
    top_k = _pfp_first_pass(board, sim, POTENTIAL_FIRE_POWER_BEAM_K)
    if not top_k:
        # 全列満杯 → 0
        return IndicatorV2Value(score=0.0, raw=0.0)
    if max_add == 1:
        # 1 手のみ: top_k から直接お邪魔数を計算
        best_ojama = max(
            _board_fire_ojama(b, elapsed_sec, sim) for _, b in top_k
        )
    else:
        # 2 手目 (max_add >= 2): top-K 各盤面を起点に再度 30 通り
        best_ojama = _pfp_second_pass(top_k, sim, elapsed_sec)
    raw = float(best_ojama)
    return IndicatorV2Value(
        score=_clamp01(raw / POTENTIAL_FIRE_POWER_NORM), raw=raw,
    )


# ============================
# VII 打ち合い収支 (条件1)
# ============================

# 11 本対戦較正: 連鎖数 → 生成お邪魔数 ≈ CHAIN_OJAMA_A * exp(CHAIN_OJAMA_B * n)。
CHAIN_OJAMA_A: float = 30.13
CHAIN_OJAMA_B: float = 0.297

# 時間推定: 1 連鎖あたりの目安秒数 (~0.30 秒/連鎖、ノイジーなので平均フィット値使用)。
TIME_PER_CHAIN_SEC: float = 0.30

# honsen_output 正規化分母: お邪魔換算が 72 (盤面容量) を超えることも多いため
# 余裕を持たせて 144 (=ON_FIELD_CAP * 2) を上限とする。
HONSEN_OUTPUT_NORM: float = float(ON_FIELD_CAP * 2)  # = 144.0

# ---- テンポ核 (時間窓つき打ち合い収支) ----
# 1手あたりの秒数: labeled_win.csv 40112行の手入れイベント間隔中央値 (実測)。
# tsumo_count_raw 増加タイミングの差分を計算。中央値=0.733秒/手。
SEC_PER_HAND: float = 0.733  # 実測中央値 (データ後決定済)

# current→achievable の連鎖数差 1 あたりに要する手数の概算。
# reach_fire_power_max_chain の gap 分布: 中央値=0, 75%=1, 90%=3。
# 1連鎖の伸びに 2 手程度かかる (ぷよ 2 個 = 1 組) という直感と整合。
HANDS_PER_CHAIN_GAP: float = 2.0  # データ後決定済 (gap×2手で到達を概算)


def chain_to_ojama(n: float) -> float:
    """連鎖数 n から生成されるお邪魔個数を較正カーブで推定する。

    カーブ: CHAIN_OJAMA_A * exp(CHAIN_OJAMA_B * n)  (11 本較正済)。
    n <= 0 の場合は 0.0 を返す。

    Args:
        n: 連鎖数 (0 以下で 0.0)。

    Returns:
        推定お邪魔個数 (float)。
    """
    import math
    if n <= 0.0:
        return 0.0
    return CHAIN_OJAMA_A * math.exp(CHAIN_OJAMA_B * n)


def chain_to_time(n: float) -> float:
    """連鎖数 n の発火に要する推定秒数を返す。

    推定: TIME_PER_CHAIN_SEC * n (時間はノイジーなので平均フィット使用)。
    n <= 0 は 0.0 を返す。

    Args:
        n: 連鎖数 (0 以下で 0.0)。

    Returns:
        推定所要秒数 (float, >= 0)。
    """
    return max(0.0, TIME_PER_CHAIN_SEC * n)


def honsen_output(
    board: Board,
    simulator: ChainSimulator | None = None,
) -> IndicatorV2Value:
    """VII-1 打ち合い出力 (本線 + セカンド合算お邪魔量)。

    片側の「打ち合い出力」を測る。相対指標(収支)は eval 側で
    「1P.honsen_output.raw - 2P.honsen_output.raw」の差として使う想定。

    算出方法:
        chain_to_ojama(current_max_chain.raw) を本線出力として使用。

        ⚠️ second_chain_potential.raw は「本線非参加の色ぷよ数」であり
        連鎖数ではない。副砲の連鎖数を直接観測できる指標が現時点で存在しないため、
        本実装では current_max_chain 単体で打ち合い出力を推定する。
        副砲連鎖数が取得可能になった段階で「chain_to_ojama(副砲連鎖数)」を加算する
        拡張が容易な設計にしている (コメント参照)。

    正規化: raw / HONSEN_OUTPUT_NORM (=144=ON_FIELD_CAP*2)。
    上限クランプあり (19連鎖想定上限 ≒ 1170個、1.0 にクランプ)。

    Args:
        board: STABLE 確定盤面。
        simulator: ChainSimulator (省略時は共有インスタンス)。

    Returns:
        IndicatorV2Value: score=raw/144 (0〜1), raw=推定お邪魔数。
    """
    sim = simulator or _SHARED_SIMULATOR
    # 本線の連鎖数 (III-1 current_max_chain と同じロジック)
    main_chain_result = sim.simulate(board)
    main_chains = float(main_chain_result.chain_count)
    # 本線お邪魔量
    main_ojama = chain_to_ojama(main_chains)
    # ── 副砲加算 (拡張ポイント) ──
    # 副砲の連鎖数が得られる場合は以下を有効化:
    #   sub_chains = <副砲連鎖数を観測する関数>
    #   sub_ojama = chain_to_ojama(sub_chains)
    # 現状は 0.0 (= current_max_chain 単体)。
    sub_ojama = 0.0
    raw = main_ojama + sub_ojama
    return IndicatorV2Value(
        score=_clamp01(raw / HONSEN_OUTPUT_NORM),
        raw=raw,
    )


def honsen_tempo_output(
    current_chain: float,
    achievable_chain: float,
    opp_chain: float,
) -> IndicatorV2Value:
    """VII-2 テンポ核: 相手本線の窓内で自分が伸ばせる打ち合い出力。

    モデル:
        window = chain_to_time(opp_chain)  ... 相手本線の所要秒数
        hands  = window / SEC_PER_HAND      ... その窓で自分が置ける手数
        frac   = min(1.0, hands / (chain_gap * HANDS_PER_CHAIN_GAP))
                   ... achievable まで伸ばすのに要する手数の達成率
        my_built = current_chain + frac * (achievable_chain - current_chain)
        raw    = chain_to_ojama(my_built)  ... 自分の打ち合い出力

    achievable_chain: reach_fire_power_max_chain が有効なら使用。
        0 (無効) の場合は current_chain + 2 で近似 (フォールバック)。
    opp_chain が 0 の場合: window=0 → hands=0 → frac=0 → raw=chain_to_ojama(current) と等価。

    テンポ核の意図: 相手本線が大きい(長い)ほど窓が広がり自分の my_built が増える。
        → 大連鎖ビルド中の優位性が反映される。

    正規化: raw / HONSEN_OUTPUT_NORM (=144)。

    Args:
        current_chain: 自分の現在最大連鎖数 (current_max_chain_raw)。
        achievable_chain: 到達目標連鎖数 (reach_fire_power_max_chain 等)。
                          0 以下の場合は current_chain + 2 にフォールバック。
        opp_chain: 相手の現在最大連鎖数 (current_max_chain_raw)。

    Returns:
        IndicatorV2Value: score=raw/144 (0〜1), raw=推定お邪魔数。
    """
    import math
    # achievable が無効 (0 以下 or 不明) の場合のフォールバック: current + 2
    _FALLBACK_CHAIN_ADD: float = 2.0
    ach = achievable_chain if achievable_chain > 0.0 else (current_chain + _FALLBACK_CHAIN_ADD)
    # 相手本線の窓
    window = chain_to_time(opp_chain)
    # 窓内で自分が置ける手数
    hands = window / SEC_PER_HAND
    # current → achievable に要する総手数
    chain_gap = max(0.0, ach - current_chain)
    hands_needed = chain_gap * HANDS_PER_CHAIN_GAP
    # 達成率 (window が短ければ部分的にしか伸ばせない)
    frac = min(1.0, hands / hands_needed) if hands_needed > 0.0 else 1.0
    # 窓内で到達できる連鎖数
    my_built = current_chain + frac * chain_gap
    raw = chain_to_ojama(my_built)
    return IndicatorV2Value(
        score=_clamp01(raw / HONSEN_OUTPUT_NORM),
        raw=raw,
    )


# ============================
# VIII 催促耐性: ojama_disruption (条件2「潰し」)
# ============================

# 催促量の代表値 (≒2段 = 12個)。実際の催促量は可変だが観測代表値として固定。
# データ後決定で変更可能。
OJAMA_DISRUPTION_DEFAULT_N: int = 12
# Monte Carlo サンプル数 (軽量化のため小さく設定)。
OJAMA_DISRUPTION_DEFAULT_SAMPLES: int = 4


def ojama_disruption(
    board: Board,
    ojama_n: int = OJAMA_DISRUPTION_DEFAULT_N,
    n_samples: int = OJAMA_DISRUPTION_DEFAULT_SAMPLES,
    simulator: "ChainSimulator | None" = None,
) -> IndicatorV2Value:
    """VIII 催促潰し度 (disruptability): お邪魔 ojama_n 個着弾で連鎖が壊れる割合。

    設計意図:
        「相手のこの盤面は催促で潰されやすいか (disruptability)」を表す。
        相手側に対して高い = こちらが催促を通せる = 有利 の文脈で使用する。
        符号の扱い (1P視点/2P視点の反転) は eval/collect 側の責務。

    計算手順:
        1. before = simulate(board).chain_count  (現在の到達連鎖)。
        2. n_samples 回、drop_ojama(board, ojama_n, seed=i) で端数列をランダム化して
           落下後盤面の after_i を計算。
        3. reduction = mean_i( max(0, (before - after_i)) / max(1, before) )
           = 連鎖が壊れた割合 (0=無傷, 1=全壊)。
        4. before <= 0 (そもそも連鎖なし) は 0.0 を返す。

    潰し成立の定義: 到達連鎖の低下割合 (別定義が必要な場合は関数を差し替え可)。

    正規化: reduction は 0〜1 に自然に収まるため clamp のみ適用。

    Args:
        board: 評価対象の盤面 (STABLE 確定盤面想定)。
        ojama_n: 代表催促量 (個数)。既定 12 (≒2段)。
        n_samples: Monte Carlo サンプル数。既定 8。
        simulator: ChainSimulator インスタンス (None = 共有 _SHARED_SIMULATOR)。

    Returns:
        IndicatorV2Value: score=reduction (0〜1), raw=reduction。
    """
    sim = simulator if simulator is not None else _SHARED_SIMULATOR
    before_result = sim.simulate(board)
    before = before_result.chain_count
    if before <= 0:
        return IndicatorV2Value(score=0.0, raw=0.0)
    total_reduction = 0.0
    for i in range(n_samples):
        try:
            ojama_board = sim.drop_ojama(board, ojama_n, seed=i)
        except Exception:
            continue
        after_result = sim.simulate(ojama_board)
        after = after_result.chain_count
        total_reduction += max(0.0, (before - after)) / max(1, before)
    raw = total_reduction / max(1, n_samples)
    return IndicatorV2Value(score=_clamp01(raw), raw=raw)


# ============================
# IX 形・組み品質 (connected_pair_quality) — EXTRA_INDICATOR_NAMES 末尾
# ============================


def _neighbors_of(cells: "frozenset[tuple[int, int]]") -> "set[tuple[int, int]]":
    """cells の全セルの上下左右1マスを返す (cells 自身は除く)。

    盤面範囲チェックは行わない (範囲外の座標もセットに含まれるが、
    グループ cells との交差判定には影響なし)。
    """
    nbrs: set[tuple[int, int]] = set()
    for r, c in cells:
        nbrs.add((r - 1, c))
        nbrs.add((r + 1, c))
        nbrs.add((r, c - 1))
        nbrs.add((r, c + 1))
    return nbrs - cells


def _connected_pairs_classify(
    board: Board,
    simulator: "ChainSimulator | None" = None,
) -> "tuple[int, int]":
    """2連結を「主連鎖隣接」と「孤立」に分類してカウントする。

    ※ 近似定義: 色を問わず size >= MAIN_GROUP_MIN_SIZE (=3) のグループに
    静的に1マス隣接しているかどうかで判定する。
    隣接する「大きな塊の脇にある2連結」=主連鎖構造に組み込まれやすい形、
    と見なす近似。連鎖伝播の実際の経路とは異なる静的な空間的隣接による近似。

    注: 同色の隣接2連結は find_groups が1つのグループに統合するため
    「同色 size>=3 に隣接する同色 size=2」は構造上発生しない。
    本指標では色を問わず大きなグループへの近接を「主連鎖貢献」の代理として扱う。

    Args:
        board: 評価対象の確定盤面 (STABLE 時のみ)。
        simulator: ChainSimulator インスタンス (省略時は共有インスタンス)。

    Returns:
        (main_linked_count, isolated_count):
            main_linked_count: (任意色の) size>=3 グループに隣接する2連結の数。
            isolated_count: どの size>=3 グループにも隣接しない孤立2連結の数。
    """
    sim = simulator or _SHARED_SIMULATOR
    groups = sim.find_groups(board)

    # 全色を合わせた size >= MAIN_GROUP_MIN_SIZE グループの cells を収集
    main_cells_all: set[tuple[int, int]] = set()
    for g in groups:
        if g.size >= MAIN_GROUP_MIN_SIZE:
            main_cells_all.update(g.cells)

    main_linked = 0
    isolated = 0
    for g in groups:
        if g.size != 2:
            continue
        if _neighbors_of(g.cells) & main_cells_all:
            main_linked += 1
        else:
            isolated += 1
    return main_linked, isolated


def main_linked_pair_count(
    board: Board, simulator: "ChainSimulator | None" = None,
) -> IndicatorV2Value:
    """IX-1 主連鎖隣接2連結数。

    同色2連結のうち、同色 size>=MAIN_GROUP_MIN_SIZE(=3) グループに
    静的に1マス隣接しているものの数。
    「あと少しで主連鎖に組み込める "生きた" 連結」を近似で数える。
    正規化: raw / NORM_LINKED_PAIR (暫定 10.0。データ後決定)。

    ※ 近似: 連鎖伝播経路でなく静的空間隣接で判定。

    Args:
        board: STABLE 確定盤面。
        simulator: ChainSimulator インスタンス (省略時は共有インスタンス)。

    Returns:
        IndicatorV2Value: score=raw/10, raw=主連鎖隣接2連結の個数。
    """
    main_linked, _ = _connected_pairs_classify(board, simulator)
    raw = float(main_linked)
    return IndicatorV2Value(score=_clamp01(raw / NORM_LINKED_PAIR), raw=raw)


def isolated_pair_count(
    board: Board, simulator: "ChainSimulator | None" = None,
) -> IndicatorV2Value:
    """IX-2 孤立2連結数。

    同色2連結のうち、どの同色 size>=MAIN_GROUP_MIN_SIZE(=3) グループにも
    静的に隣接しないもの (= 主連鎖に貢献しにくい"死に連結")。
    正規化: raw / NORM_LINKED_PAIR (暫定 10.0。データ後決定)。

    ※ 近似: 連鎖伝播経路でなく静的空間隣接で判定。

    Args:
        board: STABLE 確定盤面。
        simulator: ChainSimulator インスタンス (省略時は共有インスタンス)。

    Returns:
        IndicatorV2Value: score=raw/10, raw=孤立2連結の個数。
    """
    _, isolated = _connected_pairs_classify(board, simulator)
    raw = float(isolated)
    return IndicatorV2Value(score=_clamp01(raw / NORM_LINKED_PAIR), raw=raw)


def main_linked_ratio(
    board: Board, simulator: "ChainSimulator | None" = None,
) -> IndicatorV2Value:
    """IX-3 主連鎖隣接2連結率 = main_linked / (main_linked + isolated)。

    2連結がゼロの場合は 0.0 を返す (= 情報なし)。
    score = raw (すでに 0〜1 に自然に収まる)。

    ※ 近似: 連鎖伝播経路でなく静的空間隣接で判定。

    Args:
        board: STABLE 確定盤面。
        simulator: ChainSimulator インスタンス (省略時は共有インスタンス)。

    Returns:
        IndicatorV2Value: score=ratio (0〜1), raw=ratio。
    """
    main_linked, isolated = _connected_pairs_classify(board, simulator)
    total = main_linked + isolated
    if total == 0:
        return IndicatorV2Value(score=0.0, raw=0.0)
    raw = float(main_linked) / float(total)
    return IndicatorV2Value(score=_clamp01(raw), raw=raw)


__all__ = [
    "IndicatorV2Value",
    "GroupObservation",
    "ReachFirePowerResult",
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
    "reach_fire_power",
    "potential_fire_power",
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
    # VII 打ち合い収支 (条件1)
    "chain_to_ojama",
    "chain_to_time",
    "honsen_output",
    "CHAIN_OJAMA_A",
    "CHAIN_OJAMA_B",
    "TIME_PER_CHAIN_SEC",
    "HONSEN_OUTPUT_NORM",
    # VII-2 テンポ核 (時間窓つき打ち合い収支)
    "honsen_tempo_output",
    "SEC_PER_HAND",
    "HANDS_PER_CHAIN_GAP",
    # VIII 催促潰し度 (条件2「潰し」)
    "ojama_disruption",
    "OJAMA_DISRUPTION_DEFAULT_N",
    "OJAMA_DISRUPTION_DEFAULT_SAMPLES",
    # IX 形・組み品質 — EXTRA_INDICATOR_NAMES 末尾
    "main_linked_pair_count",
    "isolated_pair_count",
    "main_linked_ratio",
    "MAIN_GROUP_MIN_SIZE",
    "NORM_LINKED_PAIR",
]
