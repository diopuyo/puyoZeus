"""#24 打ち合い計測器 K拡張: MCロールアウトによる反撃力推定器 (2026-08-04, v2)。

背景・課題:
    scripts.measure_exchange_effectiveness.MAX_SUPPORTED_K_HANDS=4 のため、
    src.indicators_v2.expected_fire_power / counter_reach_probability は
    「相手が着弾までに実際何手打てるか」に関わらず K<=4 で飽和する。実際の
    大型連鎖 (8連鎖等) の着弾遅延では相手は実時間で最大約13手打てる
    (user承認方針: 「Kは近似値として出すのが正しい」、K上限を実時間手数まで
    モンテカルロ近似で拡張する)。

## v2 ポリシー修正 (2026-08-04、main発注): 「積んで、期限に発火」
v1 (コミットd1fa032) は既存 expected_fire_power/counter_reach_probability の
選択則 (毎手、素点最大=即時発火優先) をそのまま流用したが、検収match_04で
「溜めて大きく発火する」構図を再現できず系統的過小評価 (35 vs 実測319) と
判明した (v1コミットログ参照)。反撃は「飛来お邪魔の着弾直前に撃つもの」
なので、意味論として正しいのは **時間予算の終わりまで組み (発火しない)、
期限に達したら最良の1手で発火する** モデルである。v2はこれに合わせて
ロールアウトの内部ポリシーのみ変更する (公開API `estimate_counter_
distribution` の引数・返り値の型は無変更、後方互換)。

v2 設計方針 (既存資産の再利用優先、CLAUDE.md準拠、新規ロジック最小):
    - 「組む」フェーズ: 各手、22配置 (既存 _enumerate_placements、再実装
      しない) のうち **消去が起きない (chain_count==0) 配置だけを候補**に
      絞り (除外方式、理由は _select_build_placement docstring 参照)、その
      中で盤面の潜在連鎖 (既存指標III-1 current_max_chain) が最大のものを
      選ぶ。タイブレークは既存指標III-8 potential_fire_power (深い2手先
      ビーム)。全22配置が消去を伴う (強制発火、板がほぼ発火待ちのみで
      構成される極端なケース) 場合のみ、最小連鎖の配置に後退する。
    - 「発火」フェーズ (期限到達時、1回のみ): 未消費の既知ツモが残って
      いればその実ペアで22配置探索 (v1 _select_best_placement を再利用)
      した最良得点、無ければ既存指標III-2 immediate_fire_power (「任意
      1色の最良トリガー」を探す既存機構、再実装しない) にフォールバック
      する。この1回だけが「発火」であり、ロールアウト中の他の手は一切
      発火しない。

⚠️ 正直な注記 (v1→v2の教訓の裏返し): v1のH2改由来の過大評価バイアス
(既知ツモ単独完成値の理論値が実測を超える) と、v1自体が持っていた
過小評価バイアス (即時発火優先で大型連鎖を再現できない) は逆方向だった。
v2は「組んで最後に撃つ」ことで過小評価を緩和する狙いだが、逆に
current_max_chain (=takapt定石、色を選べる前提の潜在力) を毎手のタイブレークに
使うため、v1同様「色を自由に選べたら」という理論寄りの過大評価リスクを
再び持ち込む可能性がある。検収 (match_01: P(>=416) が低いままか) で
上振れの有無を必ず確認する。

本モジュールは stateless (盤面を破壊しない、内部状態を持たない)。
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

import numpy as np

from src.board import COLOR_OJAMA, Board
from src.chain import ChainSimulator
from src.indicators_v2 import (
    _SHARED_SIMULATOR,
    _enumerate_placements,
    _near_future_active_colors,
    _near_future_is_valid_pair,
    _score_to_ojama_count,
    current_max_chain,
    immediate_fire_power,
    potential_fire_power,
)
from src.scoring import calculate_chain_score

# ============================
# 段別最速設置時間テーブル (2026-08-03 実測較正、再フィット禁止)
# ============================
# 出典: scripts/measure_placement_speed_by_row_2026-08-03.py の本走行結果
# (70動画・246,298件の通常設置イベント、物理下限0.05秒除外→段別
# 最速25%→IQR頑健化、logs/placement_speed_2026-08-03.log)。project convention
# (NORMAL_PLACEMENT_GAP_P9999_SEC等と同じ「測定値をそのまま使う、シーン
# 逆算禁止」) に従い、この dict を唯一の物差しとする。
#
# key=row_index (盤面座標そのまま、0=最上段/隠し段側、12=最下段)。
# value=その段に新規2セルの上端 (topmost) が乗った通常設置イベントの
# 最速設置時間 (秒)。段が高い (=盤面が埋まっている状態) ほど落下距離が
# 短く速い、という user伝授の物理と整合する実測値。
PLACEMENT_SPEED_BY_ROW_SEC: "dict[int, float]" = {
    0: 0.134, 1: 0.184, 2: 0.254, 3: 0.272, 4: 0.302, 5: 0.328,
    6: 0.363, 7: 0.406, 8: 0.436, 9: 0.489, 10: 0.496, 11: 0.426, 12: 0.431,
}

# テーブル範囲外 (0-12) の row_index を渡された場合の防御的フォールバック。
# 新規にチューニングした値ではなく、テーブル内の最大値 (最も遅い側、安全側)
# をそのまま採用する。
PLACEMENT_SPEED_FALLBACK_SEC: float = max(PLACEMENT_SPEED_BY_ROW_SEC.values())

# ロールアウトの安全弁 (無限ループ防止)。時間予算が尽きる方が通常先に効くため
# (段別最速でも0.134秒/手はかかる)、この上限に到達するのは時間予算が極端に
# 大きい異常系のみ。user指示「実時間手数(〜13手)」に安全マージンを取った値。
MC_COUNTER_MAX_HANDS_HARD_CAP: int = 20

# 既定ロールアウト本数 (引数で上書き可)。
MC_COUNTER_DEFAULT_N_ROLLOUTS: int = 200

# 二重チャネルの分位点 (task定義: p25=実践値控除用の保守的下側、
# p75=理論値決着判定用の上側)。
MC_COUNTER_PRACTICAL_PERCENTILE: float = 25.0
MC_COUNTER_THEORETICAL_PERCENTILE: float = 75.0


def _clamp_row_index(row_index: int) -> int:
    """テーブル範囲 (0-12) 外の row_index を防御的にクランプする。"""
    return max(0, min(12, row_index))


def _placement_row_index(before_grid: np.ndarray, after_grid: np.ndarray) -> int:
    """設置直後 (発火前) の新規2セルのうち最も上 (row_indexが小さい) の段を返す。

    scripts/measure_placement_speed_by_row_2026-08-03.py の
    _new_color_positions + min(row) と同一の定義 (再実装だが3行のみ、
    ハイフン入り日付モジュール名の importlib 依存を避けるための最小限
    インライン化。ロジック自体 [新規色セル検出+最小行] は再考案していない)。
    新規セルが見つからない (=満杯で配置できない防御的ケース) 場合は
    最下段相当 (12、最も遅い側) を安全側の既定にする。
    """
    rows, _cols = np.where((before_grid == 0) & (after_grid != 0) & (after_grid != COLOR_OJAMA))
    if len(rows) == 0:
        return 12
    return _clamp_row_index(int(rows.min()))


def _mc_counter_seed(board: Board, time_budget_sec: float) -> int:
    """盤面+時間予算から決定論的シードを導出する (stateless、
    src.indicators_v2._expected_fire_seed と同思想: 同一入力には常に同一
    結果)。既知ツモが与えられている手数は乱数を使わないため、シードは
    盤面+時間予算のみに依存させる (仕様通り)。
    """
    grid_crc = zlib.crc32(board._grid.tobytes())
    budget_component = int(round(time_budget_sec * 1000.0))
    return (grid_crc ^ budget_component) & 0xFFFFFFFF


@dataclass(frozen=True)
class McRolloutOutcome:
    """1本のロールアウト結果 (デバッグ/検証用の内部値も保持)。"""
    achieved_ojama: float   # このロールアウトで到達した最大お邪魔換算値
    hands_used: int         # 時間予算内で実際に打てた手数
    time_used_sec: float    # 消費した時間 (段別テーブルの積分値)


def _select_best_placement(
    current: Board, pair: "tuple[int, int]", sim: ChainSimulator,
) -> "tuple[float, Board, Board] | None":
    """22配置 (既存 _enumerate_placements、再実装しない) のうち、
    calculate_chain_score (既存) の素点が最大の配置を選ぶ (=即時発火優先、
    v1の選択則そのもの)。v2では「発火フェーズ」(期限到達時に1回だけ撃つ、
    _deadline_trigger_value) 専用に温存する (v2の「組むフェーズ」には
    使わない、_select_build_placement を使うこと)。

    既存 _near_future_known_expand と同じ選択則。置き場所が無い
    (満杯・全滅) 場合は None を返す。

    Returns:
        (素点, 設置直後[発火前]の盤面, 発火後の最終盤面) または None。
    """
    best: "tuple[float, Board, Board] | None" = None
    for _chain_count, placed in _enumerate_placements(current, pair, sim):
        if placed.is_dead():
            continue
        result = sim.simulate(placed)
        score = float(calculate_chain_score(result).total_score)
        if best is None or score > best[0]:
            best = (score, placed, result.final_board)
    return best


def _select_build_placement(
    current: Board, pair: "tuple[int, int]", sim: ChainSimulator,
) -> "Board | None":
    """v2「組むフェーズ」の配置選択: 消去を起こさず、潜在連鎖が最大の配置を選ぶ。

    22配置 (既存 _enumerate_placements、再実装しない) のうち、消去が起きない
    (chain_count==0) ものだけを候補とし (除外方式)、盤面の潜在連鎖 (既存
    指標III-1 current_max_chain) が最大の配置を選ぶ。タイブレークは既存
    指標III-8 potential_fire_power (より深い2手先ビーム探索)。

    「除外」(ペナルティでなく候補から外す) を採る理由: 発火後の盤面で
    current_max_chain を評価すると、消えたぶんだけ潜在力が失われた
    **別の**盤面 (=もう積んでいない構造) を評価することになり、「組んで
    まだ撃たない」という意味論と矛盾する。ペナルティ (スコアを下げて選び
    にくくする) でも同じ順位付けの破綻は避けられないため、素直に候補から
    除く方式を採用する。

    全22配置が消去を伴う (強制発火、板がほぼ発火待ちのみで構成される極端
    なケース) 場合のみ、最小連鎖の配置に後退し、実際に解決させる
    (sim.simulate の final_board を返す。盤面状態を物理的に正しく保つため
    であり、この事故的な発火分の得点はどこにも加算しない = 過小評価の
    可能性を承知の上での単純化、正直な注記)。置き場所が全く無い場合は
    None。
    """
    candidates = _enumerate_placements(current, pair, sim)
    build_only = [(c, p) for c, p in candidates if c == 0 and not p.is_dead()]
    if not build_only:
        non_dead = [(c, p) for c, p in candidates if not p.is_dead()]
        if not non_dead:
            return None
        _chain_count, placed = min(non_dead, key=lambda cp: cp[0])
        return sim.simulate(placed).final_board
    if len(build_only) == 1:
        return build_only[0][1]
    scored = [(float(current_max_chain(p, simulator=sim).raw), p) for _c, p in build_only]
    best_potential = max(potential for potential, _p in scored)
    tied = [p for potential, p in scored if potential == best_potential]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda p: float(potential_fire_power(p, simulator=sim).raw))


def _deadline_trigger_value(
    final_board: Board,
    known_pairs: "tuple[tuple[int, int], ...]",
    known_used: int,
    sim: ChainSimulator,
    elapsed_sec: float,
) -> float:
    """v2「発火フェーズ」: 期限到達時に「最良のトリガー1手」を撃った場合の
    反撃値 (お邪魔換算) を返す。ロールアウト全体でこの1回だけが発火。

    未消費の既知ツモ (known_pairs[known_used]) が有効なら、その実ペアを
    22配置探索 (v1 _select_best_placement を再利用、実際に来る2色が分かって
    いるのでこれを使う)。既知ツモが無い/使い切っている場合は、既存指標
    III-2 immediate_fire_power (「任意1色の最良トリガー」を探す既存機構、
    再実装しない) にフォールバックする。
    """
    if final_board.is_dead():
        return 0.0
    if known_used < len(known_pairs) and _near_future_is_valid_pair(known_pairs[known_used]):
        best = _select_best_placement(final_board, known_pairs[known_used], sim)
        score = best[0] if best is not None else 0.0
        return float(_score_to_ojama_count(score, elapsed_sec))
    return float(immediate_fire_power(final_board, elapsed_sec=elapsed_sec, simulator=sim).raw)


def _rollout_once(
    board: Board,
    time_budget_sec: float,
    colors: "tuple[int, ...]",
    known_pairs: "tuple[tuple[int, int], ...]",
    sim: ChainSimulator,
    rng: "random.Random",
    elapsed_sec: float,
) -> McRolloutOutcome:
    """1本のロールアウト (v2: 「積んで、期限に発火」)。既知ツモ→以降ランダム
    4色で、時間予算を段別テーブルで動的に消費しながら**発火せず組み続け**、
    予算を使い切った盤面に対して最後に1回だけ最良のトリガーを撃つ。

    手数予算を事前に1回だけ計算するのではなく、各手ごとに「選んだ配置の
    段」を実測テーブルで引いて時間を消費する (盤面の埋まり具合に応じて
    段が変わり、それに応じて1手の時間も変わるため、静的な事前計算より
    物理的に正確、というv1からの設計判断を継承)。
    """
    current = board
    elapsed = 0.0
    hands_used = 0
    known_used = 0
    for _hand_index in range(MC_COUNTER_MAX_HANDS_HARD_CAP):
        if current.is_dead():
            break
        if known_used < len(known_pairs) and _near_future_is_valid_pair(known_pairs[known_used]):
            pair = known_pairs[known_used]
            using_known = True
        else:
            pair = (rng.choice(colors), rng.choice(colors))
            using_known = False

        placed = _select_build_placement(current, pair, sim)
        if placed is None:
            break  # 置き場所が無い (満杯)
        row_index = _placement_row_index(current._grid, placed._grid)
        step_time = PLACEMENT_SPEED_BY_ROW_SEC.get(row_index, PLACEMENT_SPEED_FALLBACK_SEC)
        if elapsed + step_time > time_budget_sec:
            break  # 時間予算超過 (この手は打てない)
        elapsed += step_time
        hands_used += 1
        if using_known:
            known_used += 1
        current = placed

    achieved_ojama = _deadline_trigger_value(current, known_pairs, known_used, sim, elapsed_sec)
    return McRolloutOutcome(achieved_ojama=achieved_ojama, hands_used=hands_used, time_used_sec=elapsed)


@dataclass(frozen=True)
class McCounterDistribution:
    """反撃力 (お邪魔換算) のMC分布。

    Attributes:
        mean/p25/p75: お邪魔換算個数のロールアウト分布の代表値。
            p25=実践値控除用 (保守的下側、相手が上振れしなくても届く量)。
            p75=理論値決着判定用 (上側、相手が上振れしても届く量、
            「受け切れないか」の判定に使う想定、二重チャネルの意味論)。
        prob_at_least: {threshold_ojama: P(到達値>=threshold)}
            (呼び出し側が渡した閾値のみ計算、既存 counter_reach_probability
            の到達確率と同じ意味論)。
        n_rollouts: 実施したロールアウト本数 (窒息盤面/n_rollouts<=0では0)。
        mean_hands_used: ロールアウト平均の実打手数 (時間予算内で打てた手数)。
        time_budget_sec: 入力した時間予算 (デバッグ用、そのまま保持)。
    """
    mean: float
    p25: float
    p75: float
    prob_at_least: "dict[float, float]"
    n_rollouts: int
    mean_hands_used: float
    time_budget_sec: float


def _empty_distribution(
    time_budget_sec: float, thresholds_ojama: "tuple[float, ...]",
) -> McCounterDistribution:
    """窒息盤面/ロールアウト0本用の 0 埋め結果 (応手不能)。"""
    return McCounterDistribution(
        mean=0.0, p25=0.0, p75=0.0,
        prob_at_least={float(th): 0.0 for th in thresholds_ojama},
        n_rollouts=0, mean_hands_used=0.0, time_budget_sec=time_budget_sec,
    )


def estimate_counter_distribution(
    board: Board,
    time_budget_sec: float,
    known_pairs: "tuple[tuple[int, int], ...]" = (),
    thresholds_ojama: "tuple[float, ...]" = (),
    n_rollouts: int = MC_COUNTER_DEFAULT_N_ROLLOUTS,
    active_colors: "tuple[int, ...] | None" = None,
    simulator: "ChainSimulator | None" = None,
    elapsed_sec: float = 0.0,
) -> McCounterDistribution:
    """時間予算 (秒) 内で応手側が実現できる反撃力 (お邪魔換算) の分布をMCで
    推定する (#24 K拡張、K=4飽和の代わりに実時間手数まで近似する)。

    Args:
        board: 応手側 (受け手) の STABLE 確定盤面 (破壊しない)。
        time_budget_sec: 着弾までの時間予算 (秒、呼び出し側が
            scripts.measure_exchange_effectiveness.estimate_landing_delay_sec
            等で見積もった値を渡す想定、本関数はその見積もり方法に依存しない)。
        known_pairs: 既知ネクスト (次・次々の順、無効なペアは
            (-1, -1) 等で渡せば自動的に無視される)。先頭から手数分だけ
            強制適用し、以降 (または最初から無効なら全手) は
            active_colors から一様ランダムサンプルする。
        thresholds_ojama: 到達確率を計算したい閾値 (お邪魔換算) の一覧。
        n_rollouts: ロールアウト本数 (既定200)。
        active_colors: 試合別4色 (省略時は盤面出現色フォールバック、既存
            _near_future_active_colors と同じ)。
        simulator: ChainSimulator (省略時は共有インスタンス)。
        elapsed_sec: 試合相対経過秒 (お邪魔換算のマージンタイム補正用、
            既存 score_to_ojama 系と同じ意味)。

    Returns:
        McCounterDistribution: mean/p25/p75/到達確率/平均打手数。

    シードは盤面+時間予算から決定論的に導出する (_mc_counter_seed、
    stateless: 同一入力には常に同一結果)。
    """
    sim = simulator or _SHARED_SIMULATOR
    if board.is_dead() or n_rollouts <= 0:
        return _empty_distribution(time_budget_sec, thresholds_ojama)
    colors = active_colors if active_colors is not None else _near_future_active_colors(board)
    rng = random.Random(_mc_counter_seed(board, time_budget_sec))

    ojama_values = np.empty(n_rollouts, dtype=float)
    hands_values = np.empty(n_rollouts, dtype=float)
    for i in range(n_rollouts):
        outcome = _rollout_once(board, time_budget_sec, colors, known_pairs, sim, rng, elapsed_sec)
        ojama_values[i] = outcome.achieved_ojama
        hands_values[i] = float(outcome.hands_used)

    prob_at_least = {float(th): float(np.mean(ojama_values >= th)) for th in thresholds_ojama}
    return McCounterDistribution(
        mean=float(np.mean(ojama_values)),
        p25=float(np.percentile(ojama_values, MC_COUNTER_PRACTICAL_PERCENTILE)),
        p75=float(np.percentile(ojama_values, MC_COUNTER_THEORETICAL_PERCENTILE)),
        prob_at_least=prob_at_least,
        n_rollouts=n_rollouts,
        mean_hands_used=float(np.mean(hands_values)),
        time_budget_sec=time_budget_sec,
    )
