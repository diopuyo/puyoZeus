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

## v3 Rust ネイティブ拡張載せ替え (2026-08-13、意味論保存)

大連鎖時 (持ち時間5秒超→7〜8手の先読み) にロールアウト内側の連鎖
シミュレーションが1回あたり数秒かかることが実測で判明した
(内訳は `scripts/_profile_mc_counter_estimator_2026-08-13.py` 参照、
`ChainSimulator.simulate` 呼び出しが総時間の9割超、その大半は「組む
フェーズ」の tie-break 由来: `current_max_chain` 約12%・
`potential_fire_power` 約87%)。**推定ロジック・乱数系列は一切変えず**、
内側の連鎖シミュレーション呼び出しのみ `native/puyo_core` (Rust拡張)
の等価APIに置換する (`use_native: bool = True` 引数、拡張未導入環境では
自動的に純Pythonへフォールバック、backwards compat)。

厳密得点 (`calculate_chain_score` 相当) が必要な箇所
(`potential_fire_power`/`_select_best_placement` の tie-break) のために、
`native/puyo_core/src/bitboard.rs` に `exact_score` (連結ボーナス反映) を
2026-08-13 に新規追加した (既存 `score_approx` は連結ボーナス0近似で
値が異なるため使えなかった)。`tests/test_puyo_core_parity.py::
test_exact_score_parity_with_chain_simulator` で実盤面600件、
`ChainSimulator`+`calculate_chain_score` との完全一致を確認済み。
`src.indicators_v2` 自体は一切変更していない (そちらの呼び出し元は
全て従来どおり Python 実装を使い続ける)。

さらに、列×色30通り (takapt定石探索・潜在火力ビーム探索の1手先候補) を
1候補ずつ native 呼び出しすると Python<->Rust 境界の変換コストが支配的に
なることが実測で判明したため (プロファイル詳細は上記スクリプト参照)、
`simulate_after_drops`/`chain_metrics_after_drops` (`src/puyo_core_bridge.py`
2026-08-13 追加) で30通りを1回のバッチ呼び出しにまとめている
(`_current_max_chain_value`/`_pfp_first_pass_native`/
`_pfp_second_pass_native` 参照)。

## v3.1 重力違反盤面の安全弁 (2026-08-13 追加)

`scripts/build_labeled_win_from_npz.py` の `_board_is_gravity_consistent`
(同日発見) と同根の既知 native 制約: puyo_core の重力実装は入力盤面が
既に重力一貫であることを前提にした定数時間最適化であり、認識由来の
浮きぷよ盤面 (実測0.28%、`project_gravity_violation_regen_lead_2026-07-30`
系の既知欠陥) を渡すと消去後の重力適用が不完全になり、2手目以降の
連鎖判定がズレる (実例: 2連鎖と判定すべき盤面が1連鎖と誤判定、Pythonが
正)。`estimate_counter_distribution` が受け取る「実盤面」(呼び出し元が
渡す STABLE 確定盤面) は認識由来のためこの違反を持ち得るが、ロールアウト
内部で `_select_build_placement`/`_deadline_trigger_value` 等が生成する
盤面は全てシミュレーション産 (`ChainSimulator.simulate`/native
`simulate_chain` の出力) であり、重力一貫は連鎖シミュレーションの後処理
として保証される (シミュレータ自身が重力を適用してから返す) ため
再チェック不要 (`build_labeled_win_from_npz.py` の全 native 呼び出し直前
チェックとは異なり、本モジュールはロールアウトの**入口 1 箇所のみ**で
チェックすれば十分)。よって `estimate_counter_distribution` の入口で
`board` を1回だけ `_board_is_gravity_consistent` で判定し、違反時は
そのロールアウト呼び出し全体 (`n_rollouts` 本すべて) を `use_native=False`
(純Python経路) で実行することで「完全一致」を保証する (native の恒久修正
は別課題、native/puyo_core 自体は本タスクで変更しない)。
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

import numpy as np

from src.board import BOARD_COLS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN, Board
from src.chain import ChainSimulator
from src.indicators_v2 import (
    IGNITION_TRIAL_COLORS,
    POTENTIAL_FIRE_POWER_BEAM_K,
    POTENTIAL_FIRE_POWER_MAX_ADD,
    _SHARED_SIMULATOR,
    _enumerate_placements,
    _near_future_active_colors,
    _near_future_is_valid_pair,
    _score_to_ojama_count,
    current_max_chain,
    immediate_fire_power,
    potential_fire_power,
)
from src.puyo_core_bridge import NATIVE_AVAILABLE
from src.puyo_core_bridge import chain_metrics_after_drops as _native_chain_metrics_after_drops
from src.puyo_core_bridge import enumerate_placements as _native_enumerate_placements
from src.puyo_core_bridge import simulate_after_drops as _native_simulate_after_drops
from src.puyo_core_bridge import simulate_chain as _native_simulate_chain
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


# ============================
# Rust ネイティブ拡張 (puyo_core) 載せ替えヘルパー (2026-08-13)
# ============================
# モジュール docstring の「v3 Rust ネイティブ拡張載せ替え」参照。
# 各関数は use_native=False (または NATIVE_AVAILABLE=False) で既存
# src.indicators_v2 の実装にそのまま委譲する (完全一致・fail-safe)。


def _board_is_gravity_consistent(board: Board) -> bool:
    """各列に「浮きぷよ」由来のギャップが無いか判定する (native 載せ替えの
    安全弁、モジュール docstring「v3.1 重力違反盤面の安全弁」参照)。

    `scripts/build_labeled_win_from_npz.py::_board_is_gravity_consistent`
    と全く同一の意味論・実装 (相互参照コメント: 両ファイルとも
    scripts/ 直下のスクリプトであり、片方をもう片方から import すると
    npz変換CLI一式 (argparse・pandas等の重い依存) が本番オーバーレイ経路
    [`scripts/visualize_advantage_overlay.py` 等] に引き込まれてしまうため
    複製している。ロジックを変える場合は両ファイル両方を修正すること)。
    UNKNOWN セルは占有扱いしない (`Board.height_of`/Rust `occ` と同じ意味論)。
    """
    grid = board._grid
    unoccupied = (grid == COLOR_EMPTY) | (grid == COLOR_UNKNOWN)
    for col in range(BOARD_COLS):
        occupied_rows = np.where(~unoccupied[:, col])[0]
        if len(occupied_rows) == 0:
            continue
        top_row = int(occupied_rows[0])
        if np.any(unoccupied[top_row:, col]):
            return False
    return True


def _native_chain_count(board: Board) -> int:
    """native puyo_core で連鎖数のみ求める (exact_score不要な箇所用)。"""
    return _native_simulate_chain(board).chain_count


# 列×色30通り (takapt定石/潜在火力探索の1手先候補) の (col, color) 一覧。
# `_takapt_best_drop`/`_pfp_first_pass` と同一探索順 (col昇順→色昇順)。
_DROP_CANDIDATES_30: "tuple[tuple[int, int], ...]" = tuple(
    (col, color) for col in range(BOARD_COLS) for color in IGNITION_TRIAL_COLORS
)


def _current_max_chain_value(
    board: Board, sim: ChainSimulator, use_native: bool,
) -> int:
    """既存指標 III-1 current_max_chain の raw 値 (takapt定石、列×色30通り
    探索の最大連鎖数) を native/Python 切替可能な形で返す。

    use_native=False (または拡張未導入) の場合は既存
    `src.indicators_v2.current_max_chain` にそのまま委譲する (完全一致、
    indicators_v2.py 自体は無変更)。use_native=True の場合は同じ30通りを
    `chain_metrics_after_drops` (盤面を返さない軽量バッチAPI、2026-08-13
    追加) で1回の呼び出しにまとめて評価する (chain_count のみ必要なため
    盤面のシリアライズを完全に省く、`_takapt_best_drop` と同一探索順・
    同一比較則 [`>` で更新])。
    """
    if not (use_native and NATIVE_AVAILABLE):
        return int(current_max_chain(board, simulator=sim).raw)
    best_chain = 0
    for r in _native_chain_metrics_after_drops(board, _DROP_CANDIDATES_30):
        if r is None:
            continue
        chain_count, _exact_score = r
        if chain_count > best_chain:
            best_chain = chain_count
    return best_chain


def _pfp_first_pass_native(board: Board, beam_k: int) -> "list[tuple[int, Board]]":
    """潜在火力1手目 (native版、`indicators_v2._pfp_first_pass` と同一探索順)。

    バッチAPI `simulate_after_drops` で30通りを1回にまとめて評価する。
    `dropped_board` (連鎖解決前の落下直後盤面) を候補として保持する点が
    `_pfp_first_pass` の意味論そのもの (2手目探索は未解決のまま積む)。
    """
    candidates: "list[tuple[int, Board]]" = [
        (r.chain_result.chain_count, r.dropped_board)
        for r in _native_simulate_after_drops(board, _DROP_CANDIDATES_30)
        if r is not None
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:beam_k]


def _pfp_second_pass_native(candidates: "list[tuple[int, Board]]") -> int:
    """潜在火力2手目 (native版、`indicators_v2._pfp_second_pass` と同一)。

    候補ごとに `chain_metrics_after_drops` (盤面を返さない軽量バッチAPI) を
    1回呼び (計 beam_k 回、90通りを1手先候補ごとにまとめて評価)、
    exact_score をお邪魔換算する。盤面 (final_board) はこの2手目探索では
    使わないため軽量版で十分 (意味論に影響しない、速度目的の選択)。
    elapsed_sec は呼び出し元 (`_select_build_placement` の tie-break) が
    常に 0.0 で呼んでいた既存挙動をそのまま踏襲する (意味論保存のため本関数
    自体は elapsed_sec を受け取らない設計にして呼び間違いを防ぐ)。
    """
    best_ojama = 0
    for _chain, board1 in candidates:
        for r in _native_chain_metrics_after_drops(board1, _DROP_CANDIDATES_30):
            if r is None:
                continue
            _chain_count, exact_score = r
            ojama = _score_to_ojama_count(float(exact_score), 0.0)
            if ojama > best_ojama:
                best_ojama = ojama
    return best_ojama


def _potential_fire_power_value(
    board: Board, sim: ChainSimulator, use_native: bool,
) -> float:
    """既存指標 III-8 potential_fire_power の raw 値 (elapsed_sec=0.0固定、
    既存 `_select_build_placement` の呼び出し方に合わせる) を native/Python
    切替可能な形で返す。use_native=False の場合は既存
    `src.indicators_v2.potential_fire_power` にそのまま委譲する。
    """
    if not (use_native and NATIVE_AVAILABLE):
        return float(potential_fire_power(board, elapsed_sec=0.0, simulator=sim).raw)
    top_k = _pfp_first_pass_native(board, POTENTIAL_FIRE_POWER_BEAM_K)
    if not top_k:
        return 0.0
    if POTENTIAL_FIRE_POWER_MAX_ADD == 1:
        best_ojama = max(
            _score_to_ojama_count(float(_native_simulate_chain(b).exact_score), 0.0)
            for _, b in top_k
        )
    else:
        best_ojama = _pfp_second_pass_native(top_k)
    return float(best_ojama)


def _enumerate_placements_dispatch(
    current: Board, pair: "tuple[int, int]", sim: ChainSimulator, use_native: bool,
) -> "list[tuple[int, Board]]":
    """既存 `_enumerate_placements` (indicators_v2) と同一の
    (chain_count, placed_board) 集合を native/Python 切替可能な形で返す。

    native 経路は元実装の「chain_count 降順ソート」を行わない (元の列挙順
    =rotation昇順→col昇順のみを返す)。呼び出し元 `_select_build_placement`
    の tie-break は常に「同じ chain_count 値の候補群」内での min/max であり、
    Python の安定ソートは同値キー内の相対順序を変えないため、この省略は
    選択結果に影響しない (報告書の設計判断参照、意味論保存)。
    """
    if not (use_native and NATIVE_AVAILABLE):
        return _enumerate_placements(current, pair, sim)
    raw = _native_enumerate_placements(current, pair, filter_dead=False)
    return [(_native_chain_count(placed), placed) for _col, _rotation, placed in raw]


def _select_best_placement(
    current: Board, pair: "tuple[int, int]", sim: ChainSimulator,
    use_native: bool = True,
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

    use_native=True (既定) かつ拡張導入済みの場合、native puyo_core の
    `exact_score` (連結ボーナス反映、`calculate_chain_score` とパリティ
    確認済み) を使う。1配置あたりの simulate 呼び出しを1回に共有する
    (元実装は `_enumerate_placements` 内と本関数内で同一盤面を2回
    simulate していたが、simulate は盤面のみに依存する純粋関数のため
    結果再利用は意味論に影響しない、速度目的の最適化)。
    """
    if use_native and NATIVE_AVAILABLE:
        raw = _native_enumerate_placements(current, pair, filter_dead=False)
        scored = [(placed, _native_simulate_chain(placed)) for _c, _r, placed in raw]
        # 元の _enumerate_placements と同じ安定ソート (chain_count降順)。
        scored.sort(key=lambda x: x[1].chain_count, reverse=True)
        best_native: "tuple[float, Board, Board] | None" = None
        for placed, sim_result in scored:
            if placed.is_dead():
                continue
            score = float(sim_result.exact_score)
            if best_native is None or score > best_native[0]:
                best_native = (score, placed, sim_result.final_board)
        return best_native
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
    use_native: bool = True,
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

    use_native=True (既定) かつ拡張導入済みの場合、内側の連鎖評価
    (chain_count/current_max_chain/potential_fire_power 相当) を native
    puyo_core に置換する (`_enumerate_placements_dispatch`/
    `_current_max_chain_value`/`_potential_fire_power_value` 参照、
    推定ロジック自体はここでは一切変えない)。
    """
    candidates = _enumerate_placements_dispatch(current, pair, sim, use_native)
    build_only = [(c, p) for c, p in candidates if c == 0 and not p.is_dead()]
    if not build_only:
        non_dead = [(c, p) for c, p in candidates if not p.is_dead()]
        if not non_dead:
            return None
        _chain_count, placed = min(non_dead, key=lambda cp: cp[0])
        if use_native and NATIVE_AVAILABLE:
            return _native_simulate_chain(placed).final_board
        return sim.simulate(placed).final_board
    if len(build_only) == 1:
        return build_only[0][1]
    scored = [
        (float(_current_max_chain_value(p, sim, use_native)), p) for _c, p in build_only
    ]
    best_potential = max(potential for potential, _p in scored)
    tied = [p for potential, p in scored if potential == best_potential]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda p: _potential_fire_power_value(p, sim, use_native))


def _deadline_trigger_value(
    final_board: Board,
    known_pairs: "tuple[tuple[int, int], ...]",
    known_used: int,
    sim: ChainSimulator,
    elapsed_sec: float,
    use_native: bool = True,
) -> float:
    """v2「発火フェーズ」: 期限到達時に「最良のトリガー1手」を撃った場合の
    反撃値 (お邪魔換算) を返す。ロールアウト全体でこの1回だけが発火。

    未消費の既知ツモ (known_pairs[known_used]) が有効なら、その実ペアを
    22配置探索 (v1 _select_best_placement を再利用、実際に来る2色が分かって
    いるのでこれを使う)。既知ツモが無い/使い切っている場合は、既存指標
    III-2 immediate_fire_power (「任意1色の最良トリガー」を探す既存機構、
    再実装しない) にフォールバックする (immediate_fire_power はロールアウト
    1回あたり最大1回しか呼ばれないため native 化の優先度は低く、
    indicators_v2 の既存 Python 実装をそのまま使う)。
    """
    if final_board.is_dead():
        return 0.0
    if known_used < len(known_pairs) and _near_future_is_valid_pair(known_pairs[known_used]):
        best = _select_best_placement(final_board, known_pairs[known_used], sim, use_native)
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
    use_native: bool = True,
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

        placed = _select_build_placement(current, pair, sim, use_native)
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

    achieved_ojama = _deadline_trigger_value(
        current, known_pairs, known_used, sim, elapsed_sec, use_native,
    )
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
    use_native: bool = True,
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
        use_native: True (既定) で内側の連鎖シミュレーションに native
            puyo_core 拡張を使う (2026-08-13 追加、意味論保存の載せ替え)。
            拡張未導入環境では自動的に純Python実装へフォールバックする
            (`src.puyo_core_bridge.NATIVE_AVAILABLE` 判定、fail-safe)。
            False を明示すれば拡張の有無に関わらず常に純Python経路を使う
            (パリティ検証用)。**`board` が重力違反 (認識由来の浮きぷよ)
            を含む場合、この値に関わらずこの呼び出し全体が自動的に純
            Python経路に固定される** (モジュール docstring「v3.1 重力違反
            盤面の安全弁」参照、`_board_is_gravity_consistent` で入口1回
            のみ判定)。

    Returns:
        McCounterDistribution: mean/p25/p75/到達確率/平均打手数。

    シードは盤面+時間予算から決定論的に導出する (_mc_counter_seed、
    stateless: 同一入力には常に同一結果)。use_native の値は乱数系列
    (rng の使い方) に一切影響しない (内側の連鎖評価バックエンドのみが
    変わる)。
    """
    sim = simulator or _SHARED_SIMULATOR
    if board.is_dead() or n_rollouts <= 0:
        return _empty_distribution(time_budget_sec, thresholds_ojama)
    colors = active_colors if active_colors is not None else _near_future_active_colors(board)
    rng = random.Random(_mc_counter_seed(board, time_budget_sec))

    # 重力違反盤面の安全弁 (モジュール docstring「v3.1」参照): 入口の実盤面
    # (認識由来、認識起因の浮きぷよを持ち得る) をここで1回だけ判定する。
    # ロールアウト内部で生成される盤面は全てシミュレーション産で重力一貫が
    # 保証されるため再チェック不要 (再チェックすると全ロールアウトで
    # O(BOARD_COLS)判定を毎手繰り返す無駄が生じる)。違反時はこの呼び出し
    # 全体を純Python経路に固定し、native/Python混在による不整合を防ぐ。
    effective_use_native = use_native and _board_is_gravity_consistent(board)

    ojama_values = np.empty(n_rollouts, dtype=float)
    hands_values = np.empty(n_rollouts, dtype=float)
    for i in range(n_rollouts):
        outcome = _rollout_once(
            board, time_budget_sec, colors, known_pairs, sim, rng, elapsed_sec,
            effective_use_native,
        )
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
