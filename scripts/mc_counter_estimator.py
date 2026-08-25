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

## v3.2 選択ロジックの境界コスト削減 (2026-08-13 追加)

v3 (`simulate_after_drops`/`chain_metrics_after_drops` 経由のバッチ化) の後も
`_select_best_placement`/`_select_build_placement` (内部の
`_enumerate_placements_dispatch`) は「22配置列挙 (1回) + 配置ごとの
`simulate_chain` (最大22回)」という計最大23回の Python<->Rust 往復を毎手
1回 (組むフェーズは毎手、発火フェーズはロールアウト末尾に1回) 行っていた
(実測 0.24s/rollout の主要因)。`src/puyo_core_bridge.py` に
`enumerate_and_simulate_placements` (2026-08-13 追加、`native/puyo_core/
src/lib.rs::enumerate_and_simulate_placements_py` を1回呼ぶだけで22配置+
シミュレーション結果一式を取得するバッチAPI) を新設し、両関数の native
分岐をこれ1回の呼び出しに置き換えた (推定ロジック・選択則は一切変えず、
往復回数のみ削減、値は旧実装と完全一致。速度計測は
`scripts/_bench_mc_counter_native_2026-08-13.py` で新旧比較する)。

さらに実測すると、上記だけでは効果が限定的だった (「1手分の列挙+選択」の
往復は1回に減ったが、`_select_build_placement` の tie-break
[潜在連鎖 `current_max_chain` 評価] が build_only 候補盤面ごとに
`_current_max_chain_value` を個別呼び出ししており [最大22回]、これが
実際の主要コストだった)。`native/puyo_core/src/lib.rs::
max_chain_after_drops_for_boards_py` (複数盤面×列×色30通りを1回で評価) と
`src/puyo_core_bridge.max_chain_after_drops_for_boards` を追加し、
`_current_max_chain_values_batch` 経由でこの tie-break も1回のバッチ
呼び出しに統合した (値は完全一致)。

それでもなお速度改善が体感できなかったため、一時診断スクリプトで
`_select_build_placement` の候補構成を実測したところ、**`current_max_chain`
が同値タイになる候補数 (`tied`) が中央値10件・最大22件** と判明した
(60盤面サンプル、`build_only` 自体も中央値19件)。`tied` 全件に対して
`max(tied, key=_potential_fire_power_value)` を呼ぶと、候補1件あたり
最大 `1+POTENTIAL_FIRE_POWER_BEAM_K` (=6) 回の native往復が発生するため、
これが実測上のロールアウト主要コストだった (前段の enumerate+simulate
統合・current_max_chain バッチ化はいずれも0.3〜0.5ms/手規模で、この
tie-break の方が桁違いに大きかった)。`native/puyo_core/src/lib.rs::
potential_fire_power_raw_for_boards_py` (複数盤面×2手先ビームを1回で評価)
と `src/puyo_core_bridge.potential_fire_power_raw_for_boards` を追加し、
`_potential_fire_power_values_batch` 経由でこの tie-break も1回のバッチ
呼び出しに統合した (`POTENTIAL_FIRE_POWER_MAX_ADD==2` [現行値] の場合のみ、
値は完全一致)。

## v3.3 選択ロジック全体のRust融合+既知ツモ重複計算排除 (2026-08-21 追加)

上記3回のバッチ呼び出し (列挙+シミュレーション/current_max_chain/
potential_fire_power) をさらに1回に統合した。`native/puyo_core/src/
lib.rs::select_build_placement_py`+`src.puyo_core_bridge.select_build_
placement` が `_select_build_placement` の選択ロジック全体 (22配置列挙+
連鎖判定+2段tie-break) をRust内で完結させ、途中候補のBoardオブジェクト
生成 (従来最大44個/手) をゼロにする (memory
`project_counter_reach_cost_breakdown_2026-08-21` の実測根拠: 200本×15手
規模のロールアウトで約700万盤面のシミュレーションを要求しており、その
75%がこのtie-break経路だった)。`POTENTIAL_FIRE_POWER_MAX_ADD==2` (現行値)
の場合のみこの融合経路を使う (それ以外は従来の個別バッチ呼び出しに
フォールバック、値は完全一致)。

さらに、既知ツモ (次・次々) が渡されている場合、その区間は乱数を消費
しないため `n_rollouts` 本全てで厳密に同一の計算結果になる
(`_rollout_once` 内で `rng.choice` が呼ばれるのは非既知区間のみ)。
`_compute_known_prefix_state` でこの既知区間を1回だけ計算し、全
ロールアウトの起点として共有する (`estimate_counter_distribution` の
`enable_prefix_dedup` 引数、既定True。値は毎回フルに再計算した場合と
完全一致)。短い時間予算 (相手1〜2連鎖想定) では既知2手が実時間手数の
大部分を占めるため、この区間の重複計算排除が有効に働く。

## v4 ビームロールアウト方式 (2026-08-21 追加、実験的・既定OFF)

user決定 (`project_counter_beam_rollout_design_2026-08-21`、コーディネータ発注):
「ツモのみランダム・置き方はビームサーチ」への方式変更を検証する。
v2/v3の「1手ずつ2段tie-breakで選ぶ greedy (=幅1相当)」が「今は損だが後で
大きい」手を捨ててしまい、溜めて大きく発火する構図を再現できない
(v1由来の過小評価、モジュール冒頭 v1/v2 セクション参照) ことへの対策。

構成 (`_rollout_once_beam` 参照): 見えているツモ (`known_pairs`) は確定、
その先を先にまとめてランダムに引き、そのツモ列全体に対して
`native/puyo_core` のビームサーチ (`src.puyo_core_bridge.beam_search`) で
置き方を決める。到達できた最大火力 (running max, `BeamSearchResult.
best_score`) を反撃値として使う。深さ (=ツモ列の長さ) は時間予算を
`PLACEMENT_SPEED_BY_ROW_SEC` の平均値で割った近似手数 (段別の動的な時間
消費は追跡しない、experimental な近似、既存 greedy 方式の厳密な段別追跡
とは異なる点に注意)。

**既定は現行方式 (greedy) のまま** (`estimate_counter_distribution` の
`rollout_mode="greedy"` が既定、backwards compat)。`rollout_mode="beam"` を
明示した場合のみこの方式に切り替わる。`beam_width` は未検証のため既定値を
持たない (呼び出し側が明示的に指定する。幅の飽和点の実測は別途
`scripts/_bench_counter_beam_rollout_2026-08-21.py` で行う、シーンからの
逆算禁止 [`feedback_overfitting_awareness_2026-08-04`])。採否は user 判断
(`project_indicator_reorg_process_2026-08-12`)。
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    DEATH_ROW,
    Board,
)
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
from src.puyo_core_bridge import beam_search as _native_beam_search
from src.puyo_core_bridge import beam_search_continue as _native_beam_search_continue
from src.puyo_core_bridge import exact_shallow_search as _native_exact_shallow_search
from src.puyo_core_bridge import chain_metrics_after_drops as _native_chain_metrics_after_drops
from src.puyo_core_bridge import (
    enumerate_and_simulate_placements as _native_enumerate_and_simulate_placements,
)
from src.puyo_core_bridge import (
    max_chain_after_drops_for_boards as _native_max_chain_after_drops_for_boards,
)
from src.puyo_core_bridge import (
    potential_fire_power_raw_for_boards as _native_potential_fire_power_raw_for_boards,
)
from src.puyo_core_bridge import select_build_placement as _native_select_build_placement
from src.puyo_core_bridge import simulate_after_drops as _native_simulate_after_drops
from src.puyo_core_bridge import simulate_chain as _native_simulate_chain
from src.scoring import OJAMA_RATE_STANDARD, calculate_chain_score, compute_effective_rate

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

# v4 ビームロールアウト (2026-08-21 追加) の深さ換算用平均設置時間。
# ビームサーチは探索前にツモ列の長さ (深さ) を固定する必要があり、greedy方式
# のような「置いた段に応じて動的に時間を消費する」追跡はできない (分岐ごとに
# 段が異なるため)。既存の実測テーブル `PLACEMENT_SPEED_BY_ROW_SEC` (物理量)
# の単純平均を使う (シーンからの逆算ではなく、既存の物理実測値からの導出、
# feedback_overfitting_awareness_2026-08-04 準拠)。
BEAM_ROLLOUT_AVG_STEP_TIME_SEC: float = (
    sum(PLACEMENT_SPEED_BY_ROW_SEC.values()) / len(PLACEMENT_SPEED_BY_ROW_SEC)
)

# ============================
# v5 ama方式 浅い完全探索 (exact_shallow) + auto振り分け (2026-08-21 追加)
# ============================
# user決定「短い連鎖はama方式(深さ2完全探索)で決定的に、長い連鎖はビーム
# サーチで」への対応。設計記録: coordinator発注 (project_counter_beam_
# rollout_design_2026-08-21 の後続、ama [citrus610/ama、MIT] の dfs::attack
# 調査に基づく)。

# ama の防御的枝刈り高さ閾値 (`ai/search/dfs/attack.cpp` の `height > 11`)
# を、このプロジェクトの盤面表現から導出した値 (物理量からの導出、11を
# そのまま持ち込まない)。このプロジェクトは6列×13行 (row0=隠し段)、窒息
# 判定は可視最上段 (DEATH_ROW=1) の窒息列 (DEATH_COL) が埋まった時点。
# 窒息列の高さが `BOARD_ROWS - DEATH_ROW` (=12) に達すると窒息する
# (=height>=12 で既に窒息、この状態の候補は既存の `filter_dead=True` で
# 別途除外済み)。ama と同じ「危険な高さ」の防御的閾値として、この値
# (12、すなわち height>=12 で枝刈り = ama の `height>11` と同じ判定) を
# 全列共通の防御的しきい値として採用する。
# **注記 (2026-08-21、正直な注記)**: 窒息列 (DEATH_COL) 単体ではこの閾値は
# 既存の `filter_dead=True` と重複する (height>=12 の窒息列は既にその
# placement自体が death 判定で除外されているため無害)。しかし ama 同様
# **全列**にこの閾値を適用するため、窒息していない他列がこの高さに達した
# 場合にも枝刈りする — これは「捨てても答えが変わらない」保証がない
# **ヒューリスティック** (ama自身の設計も同様、`exact_shallow` 専用の
# 内部設計選択として受け入れる。bit-identical であることは主張しない。
# `tests/test_mc_counter_estimator.py::TestExactShallowHeightPruning` で
# 「実際にどの程度結果を変えるか」を実盤面で定量化している)。
EXACT_SHALLOW_PRUNE_HEIGHT: int = BOARD_ROWS - DEATH_ROW  # = 12

# exact_shallow 単独モードの深さ安全弁 (2026-08-21 追加)。22^depth の組合せ
# 爆発を避けるための上限 (22^3=10,648 は数msで完了する規模、22^4=234,256
# は1ロールアウトあたり無視できないコストになる — 実測は
# `scripts/_bench_counter_beam_rollout_2026-08-21.py` 参照)。見えている
# ツモ (NEXT+ダブルNEXT=2手) 相当+安全マージン1手、というuser指示の
# 「概ね2〜3手」に対応する値。
EXACT_SHALLOW_MAX_DEPTH: int = 3

# ビームロールアウトの初期集団として exact_shallow の完全探索結果を使う
# 深さ (2026-08-21 追加、user指示①「完全探索の結果をビームサーチの初期
# 集団にする」)。「見えているツモ = NEXT+ダブルNEXT」に対応する2手固定
# (`EXACT_SHALLOW_MAX_DEPTH` とは別の定数: そちらはexact_shallow単独モード
# の安全弁、これはビーム継続の種として使う深さで、ama の
# dfs::attack (深さ2固定) に厳密に対応する)。
EXACT_SHALLOW_SEED_DEPTH: int = 2


def _ojama_threshold_to_score_threshold(ojama_threshold: float, elapsed_sec: float) -> int:
    """お邪魔換算閾値を素点 (score) の閾値に厳密変換する (2026-08-21 追加、
    user指示②「答えを変えない打ち切り」用)。

    `_score_to_ojama_count` (= `score_to_ojama`) は固定 `elapsed_sec` の下で
    `ojama = (score + 0) // rate` (`rate = compute_effective_rate(elapsed_sec)`
    は正の整数、`src/scoring.py::score_to_ojama` 参照) という floor除算の
    単調非減少関数なので、厳密な逆変換が閉形式で求まる: `ojama >= k` を
    満たす最小の score は `score = ceil(k) * rate` (floor除算の性質上、
    `score // rate >= n ⟺ score >= n * rate`、n=ceil(k) は「ojamaは整数
    なので k 以上になる最小の整数」)。近似・二分探索は使わない。
    """
    rate = compute_effective_rate(elapsed_sec, OJAMA_RATE_STANDARD)
    return math.ceil(ojama_threshold) * rate


def _canonical_pair(pair: "tuple[int, int]") -> "tuple[int, int]":
    """(top,bot) と (bot,top) は到達可能な盤面集合が同一なので同一視する
    (2026-08-21 追加、深さ1〜2限定の部分木共有キャッシュ用キー正規化)。

    `native/puyo_core/src/bitboard.rs::place_pair` の回転対称性 (縦置き
    TOP_UP/BOT_UP が top/bot を入れ替えるだけの組) により、pair=(a,b) と
    pair=(b,a) は22配置の結果盤面集合が完全に同一になる (数学的に保証、
    ヒューリスティックではない)。よってキャッシュキーとして安全に統合できる。
    """
    return (min(pair), max(pair))

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


def _current_max_chain_values_batch(
    boards: "list[Board]", sim: ChainSimulator, use_native: bool,
) -> "list[float]":
    """`_current_max_chain_value` を複数候補盤面にわたって評価する
    (`_select_build_placement` の tie-break 専用、2026-08-13 追加)。

    use_native=True かつ拡張導入済みの場合、候補盤面ごとに個別呼び出しして
    いた `_current_max_chain_value` (盤面数 = 最大22回の native 往復) を
    `max_chain_after_drops_for_boards` (1回のバッチ呼び出しで全候補盤面×
    列×色30通りをまとめて評価) に置き換え、往復を1回に削減する
    (`v3.2 選択ロジックの境界コスト削減` docstring参照、値は
    `_current_max_chain_value` を1件ずつ呼んだ場合と完全一致)。
    """
    if not (use_native and NATIVE_AVAILABLE):
        return [float(_current_max_chain_value(b, sim, use_native)) for b in boards]
    return [
        float(v) for v in _native_max_chain_after_drops_for_boards(boards, _DROP_CANDIDATES_30)
    ]


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


def _potential_fire_power_values_batch(
    boards: "list[Board]", sim: ChainSimulator, use_native: bool,
) -> "list[float]":
    """`_potential_fire_power_value` を複数候補盤面にわたって評価する
    (`_select_build_placement` の tie-break 専用、2026-08-13 追加)。

    実測診断で `_select_build_placement` の tied 候補数は中央値10件・
    最大22件と判明し (60盤面サンプル)、候補ごとに `_potential_fire_power_
    value` (最大 1+beam_k 回の native往復) を個別呼び出しすることが
    ロールアウトの残存コストの主因だった (`v3.2 選択ロジックの境界コスト
    削減` docstring参照)。native経路かつ `POTENTIAL_FIRE_POWER_MAX_ADD==2`
    (現行値) の場合のみ `potential_fire_power_raw_for_boards`
    (1回のバッチ呼び出し) に置き換え、それ以外 (将来 MAX_ADD が変わった
    場合・フォールバック) は既存 `_potential_fire_power_value` の個別
    呼び出しをそのまま維持する (値は完全一致)。
    """
    if not (use_native and NATIVE_AVAILABLE) or POTENTIAL_FIRE_POWER_MAX_ADD != 2:
        return [_potential_fire_power_value(b, sim, use_native) for b in boards]
    raw = _native_potential_fire_power_raw_for_boards(
        boards, _DROP_CANDIDATES_30, POTENTIAL_FIRE_POWER_BEAM_K,
    )
    return [float(_score_to_ojama_count(float(v), 0.0)) for v in raw]


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

    2026-08-13: native経路は `enumerate_and_simulate_placements` (1回の
    バッチ呼び出しで22配置+連鎖シミュレーション結果一式を取得) に置き換え、
    従来「列挙1回+配置ごとに simulate_chain 最大22回」だった Python<->Rust
    往復を1回に削減した (`v3.2 選択ロジックの境界コスト削減` docstring参照、
    値は完全一致)。
    """
    if not (use_native and NATIVE_AVAILABLE):
        return _enumerate_placements(current, pair, sim)
    results = _native_enumerate_and_simulate_placements(current, pair, filter_dead=False)
    return [(r.chain_result.chain_count, r.placed_board) for r in results]


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
    確認済み) を使う。2026-08-13: 22配置列挙+配置ごとの連鎖シミュレーション
    (最大22回の個別 native 呼び出し) を `enumerate_and_simulate_placements`
    (1回のバッチ呼び出し) に統合し、Python<->Rust 往復を削減した
    (`v3.2 選択ロジックの境界コスト削減` docstring参照、選択則・値は完全一致)。
    """
    if use_native and NATIVE_AVAILABLE:
        results = _native_enumerate_and_simulate_placements(current, pair, filter_dead=False)
        # 元の _enumerate_placements と同じ安定ソート (chain_count降順)。
        results.sort(key=lambda r: r.chain_result.chain_count, reverse=True)
        best_native: "tuple[float, Board, Board] | None" = None
        for r in results:
            if r.is_dead:
                continue
            score = float(r.chain_result.exact_score)
            if best_native is None or score > best_native[0]:
                best_native = (score, r.placed_board, r.chain_result.final_board)
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
    `_current_max_chain_values_batch`/`_potential_fire_power_value` 参照、
    推定ロジック自体はここでは一切変えない)。2026-08-13: tie-break の
    current_max_chain 評価は候補盤面ごとの個別呼び出し (最大22回の native
    往復) から `_current_max_chain_values_batch` (1回のバッチ呼び出し) に
    置き換えた (`v3.2 選択ロジックの境界コスト削減` docstring参照、値は
    従来と完全一致)。

    2026-08-21 (v3.3): 上記3回 (列挙+シミュレーション/current_max_chain/
    potential_fire_power) の個別バッチ呼び出しを、native側の
    `select_build_placement` (1回の呼び出しで選択ロジック全体を完結、
    途中候補の Board オブジェクト生成をゼロにする) にさらに統合した。
    `POTENTIAL_FIRE_POWER_MAX_ADD == 2` (現行値) の場合のみこの融合経路を
    使う (native側の固定アルゴリズムが MAX_ADD==2 の意味論のみ実装している
    ため、それ以外の値では以下の従来経路にフォールバックする)。値は
    従来の3回個別呼び出しと完全一致する
    (`tests/test_puyo_core_parity.py::
    test_select_build_placement_parity_with_reference` で確認)。
    """
    if use_native and NATIVE_AVAILABLE and POTENTIAL_FIRE_POWER_MAX_ADD == 2:
        return _native_select_build_placement(
            current, pair, list(_DROP_CANDIDATES_30), POTENTIAL_FIRE_POWER_BEAM_K,
            exclude_hidden_row_from_pop=False,  # 既存呼び出し元は全て既定Falseで統一
        )
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
    build_boards = [p for _c, p in build_only]
    chain_values = _current_max_chain_values_batch(build_boards, sim, use_native)
    scored = list(zip(chain_values, build_boards))
    best_potential = max(potential for potential, _p in scored)
    tied = [p for potential, p in scored if potential == best_potential]
    if len(tied) == 1:
        return tied[0]
    # 2026-08-13: tied 全件への _potential_fire_power_value 個別呼び出し
    # (実測でロールアウトの主要コスト) を1回のバッチ呼び出しに統合
    # (`_potential_fire_power_values_batch` 参照、値は完全一致)。
    pfp_values = _potential_fire_power_values_batch(tied, sim, use_native)
    best_idx = max(range(len(tied)), key=lambda i: pfp_values[i])
    return tied[best_idx]


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


@dataclass(frozen=True)
class _KnownPrefixState:
    """既知ツモ区間 (乱数を消費しない先頭手) の共有ロールアウト状態
    (2026-08-21 追加、v3.3 既知ツモ重複計算排除)。

    既知ツモの処理は `rng` を一切消費しないため (`_rollout_once` 参照)、
    `n_rollouts` 本すべてで厳密に同一の計算結果になる。1回だけ計算して
    この状態を全ロールアウトの起点として共有することで、ロールアウトの
    たびに同じ既知手を再計算する無駄を省く (値は毎回フルに再計算した
    場合と完全一致、`tests/test_mc_counter_estimator.py::
    TestKnownPrefixDedup` で確認)。

    Attributes:
        current/elapsed/hands_used/known_used: 対応する `_rollout_once`
            ループ変数の、既知区間終了時点の値。
        consumed_iterations: 既知区間で消費した `_hand_index` の回数
            (`early_stop=False` の場合のみ、続行ループの開始位置として使う)。
        early_stop: True の場合、既知区間中に窒息/置き場所無し/時間予算
            超過/ハードキャップ到達でロールアウトが打ち切られている
            (以降の手を打つ余地が無く、続行ループは不要)。
    """
    current: Board
    elapsed: float
    hands_used: int
    known_used: int
    consumed_iterations: int
    early_stop: bool


def _compute_known_prefix_state(
    board: Board,
    time_budget_sec: float,
    known_pairs: "tuple[tuple[int, int], ...]",
    sim: ChainSimulator,
    use_native: bool,
) -> _KnownPrefixState:
    """既知ツモ区間 (乱数不使用) を1回だけ処理する
    (`_rollout_once` の先頭ループと同一ロジック、乱数を使わない点のみ異なる)。
    """
    current = board
    elapsed = 0.0
    hands_used = 0
    known_used = 0
    for hand_index in range(MC_COUNTER_MAX_HANDS_HARD_CAP):
        if current.is_dead():
            return _KnownPrefixState(current, elapsed, hands_used, known_used, hand_index, True)
        if not (
            known_used < len(known_pairs) and _near_future_is_valid_pair(known_pairs[known_used])
        ):
            # 乱数区間 (ロールアウトごとに異なる) に入る手前で打ち切り、
            # 続行ループがこの hand_index からランダム手を試す。
            return _KnownPrefixState(current, elapsed, hands_used, known_used, hand_index, False)
        placed = _select_build_placement(current, known_pairs[known_used], sim, use_native)
        if placed is None:
            return _KnownPrefixState(current, elapsed, hands_used, known_used, hand_index, True)
        row_index = _placement_row_index(current._grid, placed._grid)
        step_time = PLACEMENT_SPEED_BY_ROW_SEC.get(row_index, PLACEMENT_SPEED_FALLBACK_SEC)
        if elapsed + step_time > time_budget_sec:
            return _KnownPrefixState(current, elapsed, hands_used, known_used, hand_index, True)
        elapsed += step_time
        hands_used += 1
        known_used += 1
        current = placed
    return _KnownPrefixState(
        current, elapsed, hands_used, known_used, MC_COUNTER_MAX_HANDS_HARD_CAP, True,
    )


def _rollout_once(
    board: Board,
    time_budget_sec: float,
    colors: "tuple[int, ...]",
    known_pairs: "tuple[tuple[int, int], ...]",
    sim: ChainSimulator,
    rng: "random.Random",
    elapsed_sec: float,
    use_native: bool = True,
    prefix: "_KnownPrefixState | None" = None,
) -> McRolloutOutcome:
    """1本のロールアウト (v2: 「積んで、期限に発火」)。既知ツモ→以降ランダム
    4色で、時間予算を段別テーブルで動的に消費しながら**発火せず組み続け**、
    予算を使い切った盤面に対して最後に1回だけ最良のトリガーを撃つ。

    手数予算を事前に1回だけ計算するのではなく、各手ごとに「選んだ配置の
    段」を実測テーブルで引いて時間を消費する (盤面の埋まり具合に応じて
    段が変わり、それに応じて1手の時間も変わるため、静的な事前計算より
    物理的に正確、というv1からの設計判断を継承)。

    2026-08-21 (v3.3): `prefix` (既知ツモ区間の共有計算結果、
    `_compute_known_prefix_state` 参照) を渡すと、乱数を消費しない先頭手を
    再計算せずその状態から続行する (値は毎回フルに計算した場合と完全一致)。
    省略時 (None、既定) は従来どおり `board` から全て計算する
    (backwards compat、既存呼び出し元は無変更で動く)。
    """
    if prefix is not None and prefix.early_stop:
        achieved_ojama = _deadline_trigger_value(
            prefix.current, known_pairs, prefix.known_used, sim, elapsed_sec, use_native,
        )
        return McRolloutOutcome(achieved_ojama, prefix.hands_used, prefix.elapsed)
    if prefix is not None:
        current, elapsed, hands_used, known_used = (
            prefix.current, prefix.elapsed, prefix.hands_used, prefix.known_used,
        )
        start_hand = prefix.consumed_iterations
    else:
        current, elapsed, hands_used, known_used, start_hand = board, 0.0, 0, 0, 0

    for _hand_index in range(start_hand, MC_COUNTER_MAX_HANDS_HARD_CAP):
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


def _time_budget_to_beam_depth(time_budget_sec: float) -> int:
    """時間予算をビームロールアウトの探索深さ (ツモ列の長さ) に換算する
    (v4、`BEAM_ROLLOUT_AVG_STEP_TIME_SEC` 参照)。負・ゼロ予算は深さ0。
    """
    if time_budget_sec <= 0.0:
        return 0
    depth = round(time_budget_sec / BEAM_ROLLOUT_AVG_STEP_TIME_SEC)
    return int(min(MC_COUNTER_MAX_HANDS_HARD_CAP, max(0, depth)))


def _draw_beam_tsumo_sequence(
    depth: int,
    known_pairs: "tuple[tuple[int, int], ...]",
    colors: "tuple[int, ...]",
    rng: "random.Random",
) -> "list[tuple[int, int]]":
    """深さ分のツモ列を作る: 先頭は有効な既知ツモをそのまま使い、以降
    (既知ツモを使い切った/無効な場合含む) は `colors` から一様ランダムに
    まとめて引く (v4、「ツモのみランダム」の実装本体)。
    """
    pairs: "list[tuple[int, int]]" = []
    for i in range(depth):
        if i < len(known_pairs) and _near_future_is_valid_pair(known_pairs[i]):
            pairs.append(known_pairs[i])
        else:
            pairs.append((rng.choice(colors), rng.choice(colors)))
    return pairs


def _rollout_once_exact_shallow(
    board: Board,
    time_budget_sec: float,
    colors: "tuple[int, ...]",
    known_pairs: "tuple[tuple[int, int], ...]",
    rng: "random.Random",
    elapsed_sec: float,
    early_exit_score: "int | None" = None,
    _max_depth_override: "int | None" = None,
) -> McRolloutOutcome:
    """ama方式 (`ai/search/dfs/attack.cpp`) の浅い完全探索1本 (v5、
    2026-08-21 追加)。見えているツモ (NEXT+ダブルNEXT、最大
    `EXACT_SHALLOW_MAX_DEPTH` 手) を全列挙し、amaと同じ防御的高さ枝刈り
    (`EXACT_SHALLOW_PRUNE_HEIGHT`、ヒューリスティック、モジュール
    docstring参照) を適用する。時間予算が既知ツモの範囲・安全弁を超える
    場合、超えた分だけランダムに引く (`_draw_beam_tsumo_sequence` 参照、
    乱数消費はその分のみ)。

    **ama との差分**: ama は `TRIGGER=100000` で「返せる/返せない」を
    二値判定するが、本関数は連続値 (お邪魔換算) をそのまま返し、二値化
    しない (`reference_ojama_damage_nonlinear_2026-07-29`「返せるか二値は
    設計として誤り」準拠)。`early_exit_score` は打ち切りタイミングの
    最適化にのみ使い、返り値自体を二値化するものではない。

    Args:
        early_exit_score: user指示②。Some(t) で running_best が t 以上に
            達した時点で打ち切る (「閾値到達確率」用途限定、呼び出し元
            `estimate_counter_distribution` の `early_exit_at_threshold`
            docstring参照)。
        _max_depth_override: テスト専用 (陽性対照: 深さ1に落とすと過小
            評価が悪化することの確認用)。通常は None (既定の
            `EXACT_SHALLOW_MAX_DEPTH` を使う)。
    """
    if board.is_dead():
        return McRolloutOutcome(0.0, 0, 0.0)
    max_depth = EXACT_SHALLOW_MAX_DEPTH if _max_depth_override is None else _max_depth_override
    depth = min(_time_budget_to_beam_depth(time_budget_sec), max_depth)
    if depth <= 0:
        return McRolloutOutcome(0.0, 0, 0.0)
    pairs = _draw_beam_tsumo_sequence(depth, known_pairs, colors, rng)
    result = _native_exact_shallow_search(
        board, pairs, exclude_hidden_row_from_pop=False, use_exact_score=True,
        max_height=EXACT_SHALLOW_PRUNE_HEIGHT, early_exit_score=early_exit_score,
    )
    achieved_ojama = float(_score_to_ojama_count(float(result.best_score), elapsed_sec))
    return McRolloutOutcome(
        achieved_ojama=achieved_ojama, hands_used=depth,
        time_used_sec=depth * BEAM_ROLLOUT_AVG_STEP_TIME_SEC,
    )


def _truncate_frontier_by_running_best(
    frontier: "list", beam_width: int,
) -> "list":
    """フロンティアを running_best 降順で上位 `beam_width` 件に絞る
    (2026-08-21 追加、user指示①「初期フロンティアは上位beam_width個」の
    仕様通りの実装、`_rollout_once_beam` 参照)。

    Python の `sorted` は安定ソート (同値は元の順序を保つ) であり、
    native `beam::expand_one_depth` の `sort_by`(こちらも安定ソート) と
    同じタイブレーク規則になる (両者の入力順序が同じ場合に限る、本関数の
    入力は native 側が返した順序のまま渡される想定)。
    """
    return sorted(frontier, key=lambda f: f.running_best, reverse=True)[:beam_width]


def _lookup_or_compute_exact_shallow_seed(
    board: Board,
    seed_pairs: "tuple[tuple[int, int], ...]",
    early_exit_score: "int | None",
    seed_cache: "dict[tuple[tuple[int, int], ...], object] | None",
):
    """深さ1〜2限定の部分木共有 (2026-08-21 追加、user指示④/③)。

    `seed_pairs` の正規化キー (`_canonical_pair`、盤面到達集合が同一なので
    数学的に安全) で `seed_cache` を引き、無ければ `exact_shallow_search`
    を計算してキャッシュする。`seed_cache=None` なら毎回計算する
    (呼び出し元が渡さなければ従来通り、backwards compat)。

    実測 (`tests/test_mc_counter_estimator.py::TestKnownPrefixDedup` 系と
    同じ考え方): n_rollouts=60 で深さ1の重複率83.3%・深さ2で35.0%
    (`project_counter_beam_rollout_design_2026-08-21` 系の実測)。
    """
    key = tuple(_canonical_pair(p) for p in seed_pairs)
    if seed_cache is not None and key in seed_cache:
        return seed_cache[key]
    result = _native_exact_shallow_search(
        board, list(seed_pairs), exclude_hidden_row_from_pop=False, use_exact_score=True,
        max_height=EXACT_SHALLOW_PRUNE_HEIGHT, early_exit_score=early_exit_score,
    )
    if seed_cache is not None:
        seed_cache[key] = result
    return result


def _rollout_once_beam(
    board: Board,
    time_budget_sec: float,
    colors: "tuple[int, ...]",
    known_pairs: "tuple[tuple[int, int], ...]",
    rng: "random.Random",
    elapsed_sec: float,
    beam_width: int,
    use_exact_score: bool = True,
    early_exit_score: "int | None" = None,
    seed_cache: "dict[tuple[tuple[int, int], ...], object] | None" = None,
) -> McRolloutOutcome:
    """v4/v5 ビームロールアウト1本 (モジュール docstring「v4」「v5」参照)。

    見えているツモ (`known_pairs`) 確定+その先ランダムのツモ列を1本まとめて
    引く。**v5 (2026-08-21 追加)**: 先頭 `EXACT_SHALLOW_SEED_DEPTH` 手は
    ama方式の完全探索 (`exact_shallow_search`、深さ2なら22+22²=506通り) で
    処理し、その `final_frontier` (完全探索終了時点の全候補) を**ビーム
    サーチの初期集団**として使う (user指示①「初期集団の質を上げる」、
    コストは増えない — 深さ2までの完全探索はどちらにしろ行う)。残り
    (深さ3以降) は従来通りビームサーチで継続する。

    到達できた最大火力 (running max) を反撃値として返す。

    Args:
        beam_width: ビーム幅 (呼び出し側が明示的に指定、未検証のため既定値
            は持たない)。
        use_exact_score: True (既定) で評価に厳密得点 `exact_score`
            (連結ボーナス反映) を使う。`simulate_chain` は両方を常に計算
            するため速度上のトレードオフは無い (実測で確認済み、
            `scripts/_bench_counter_beam_rollout_2026-08-21.py` 参照)。
        early_exit_score: user指示②、`_rollout_once_exact_shallow` と同じ
            意味論 (「閾値到達確率」用途限定)。
        seed_cache: user指示④、深さ1〜2限定の部分木共有キャッシュ
            (`_lookup_or_compute_exact_shallow_seed` 参照)。呼び出し元
            `estimate_counter_distribution` が1回の呼び出しの中で共有する
            (呼び出しを跨いでは保持しない、stateless原則)。
    """
    if board.is_dead():
        return McRolloutOutcome(0.0, 0, 0.0)
    depth = _time_budget_to_beam_depth(time_budget_sec)
    if depth <= 0:
        return McRolloutOutcome(0.0, 0, 0.0)
    pairs = _draw_beam_tsumo_sequence(depth, known_pairs, colors, rng)

    seed_depth = min(depth, EXACT_SHALLOW_SEED_DEPTH)
    seed_pairs = tuple(pairs[:seed_depth])
    seed_result = _lookup_or_compute_exact_shallow_seed(
        board, seed_pairs, early_exit_score, seed_cache,
    )
    remaining_pairs = pairs[seed_depth:]

    if not remaining_pairs or (
        early_exit_score is not None and seed_result.best_score >= early_exit_score
    ):
        # 残り手が無い、または既に閾値到達が確定済み (seed_result は
        # 打ち切りなしで完全探索を終えている場合のみこの分岐に入る、
        # `_lookup_or_compute_exact_shallow_seed` docstring参照)。
        best_score = seed_result.best_score
    else:
        # user指示①の仕様通り「484通りから上位beam_width個」に絞ってから
        # 継続する (2026-08-21 実測で判明: 絞らずに全件[最大484件] を
        # そのまま継続に渡すと、幅が狭い場面ではむしろ従来より遅くなる
        # ["コストは増えない" という前提が崩れる、`_bench_counter_beam_
        # speedups_2026-08-21.py` で確認] — 絞ることで幅が保証する計算量
        # 上限を守りつつ、探索空間全体から選んだ質の高い種を使う、という
        # 意図通りの動作になる)。
        seeded_frontier = _truncate_frontier_by_running_best(seed_result.final_frontier, beam_width)
        continued = _native_beam_search_continue(
            seeded_frontier, seed_result.best_score, remaining_pairs, beam_width,
            exclude_hidden_row_from_pop=False, use_exact_score=use_exact_score,
            early_exit_score=early_exit_score,
        )
        best_score = continued.best_score

    achieved_ojama = float(_score_to_ojama_count(float(best_score), elapsed_sec))
    return McRolloutOutcome(
        achieved_ojama=achieved_ojama, hands_used=depth,
        time_used_sec=depth * BEAM_ROLLOUT_AVG_STEP_TIME_SEC,
    )


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


_ROLLOUT_MODE_GREEDY: str = "greedy"
_ROLLOUT_MODE_BEAM: str = "beam"
_ROLLOUT_MODE_EXACT_SHALLOW: str = "exact_shallow"
_ROLLOUT_MODE_AUTO: str = "auto"
_VALID_ROLLOUT_MODES: "tuple[str, ...]" = (
    _ROLLOUT_MODE_GREEDY, _ROLLOUT_MODE_BEAM, _ROLLOUT_MODE_EXACT_SHALLOW, _ROLLOUT_MODE_AUTO,
)


def _resolve_auto_rollout_mode(time_budget_sec: float) -> str:
    """`rollout_mode="auto"` の振り分け (2026-08-21 追加、v5)。

    境界は**手数から物理的に**決める (`_time_budget_to_beam_depth`、
    `PLACEMENT_SPEED_BY_ROW_SEC` の実測平均由来。シーンからの逆算ではない、
    `feedback_overfitting_awareness_2026-08-04` 準拠)。打てる手数が
    `EXACT_SHALLOW_MAX_DEPTH` (見えているツモの範囲+安全マージン、概ね
    2〜3手) 以内なら決定的な `exact_shallow` (乱数無し・確実な下界)、
    それを超えるなら `beam` に切り替える。この判定は `time_budget_sec`
    のみに依存し盤面には依存しないため、1回の `estimate_counter_
    distribution` 呼び出し内で全ロールアウト共通 (ロールアウトごとに
    モードが変わることはない)。
    """
    depth = _time_budget_to_beam_depth(time_budget_sec)
    return _ROLLOUT_MODE_EXACT_SHALLOW if depth <= EXACT_SHALLOW_MAX_DEPTH else _ROLLOUT_MODE_BEAM


def _run_rollouts(
    rollout_mode: str,
    board: Board,
    time_budget_sec: float,
    colors: "tuple[int, ...]",
    known_pairs: "tuple[tuple[int, int], ...]",
    sim: ChainSimulator,
    rng: "random.Random",
    elapsed_sec: float,
    effective_use_native: bool,
    prefix: "_KnownPrefixState | None",
    n_rollouts: int,
    beam_width: "int | None",
    beam_use_exact_score: bool,
    early_exit_score: "int | None",
) -> "tuple[np.ndarray, np.ndarray]":
    """`n_rollouts` 本を実行し (お邪魔換算値, 実打手数) の配列対を返す
    (`estimate_counter_distribution` から抽出、v4/v5 のモード分岐用)。
    """
    resolved_mode = (
        _resolve_auto_rollout_mode(time_budget_sec) if rollout_mode == _ROLLOUT_MODE_AUTO
        else rollout_mode
    )
    # 深さ1〜2限定の部分木共有キャッシュ (2026-08-21 追加、v5 user指示④):
    # 1回の estimate_counter_distribution 呼び出し内 (=このロールアウト
    # ループ) でのみ共有し、呼び出しを跨いでは保持しない (stateless原則)。
    seed_cache: "dict[tuple[tuple[int, int], ...], object]" = {}

    ojama_values = np.empty(n_rollouts, dtype=float)
    hands_values = np.empty(n_rollouts, dtype=float)
    for i in range(n_rollouts):
        if resolved_mode == _ROLLOUT_MODE_BEAM:
            assert beam_width is not None  # 呼び出し元で検証済み
            outcome = _rollout_once_beam(
                board, time_budget_sec, colors, known_pairs, rng, elapsed_sec,
                beam_width, beam_use_exact_score, early_exit_score, seed_cache,
            )
        elif resolved_mode == _ROLLOUT_MODE_EXACT_SHALLOW:
            outcome = _rollout_once_exact_shallow(
                board, time_budget_sec, colors, known_pairs, rng, elapsed_sec,
                early_exit_score,
            )
        else:
            outcome = _rollout_once(
                board, time_budget_sec, colors, known_pairs, sim, rng, elapsed_sec,
                effective_use_native, prefix,
            )
        ojama_values[i] = outcome.achieved_ojama
        hands_values[i] = float(outcome.hands_used)
    return ojama_values, hands_values


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
    enable_prefix_dedup: bool = True,
    rollout_mode: str = _ROLLOUT_MODE_GREEDY,
    beam_width: "int | None" = None,
    beam_use_exact_score: bool = True,
    early_exit_at_threshold: bool = False,
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
        enable_prefix_dedup: True (既定) で既知ツモ区間 (乱数不使用、
            `_compute_known_prefix_state` 参照) を1回だけ計算して全
            ロールアウトで共有する (2026-08-21 追加、v3.3)。False にすると
            従来どおりロールアウトごとにフルで再計算する (パリティ検証用、
            値は enable_prefix_dedup の値に関わらず完全一致する)。
        rollout_mode: "greedy" (既定、既存挙動そのまま) / "beam" (v4) /
            "exact_shallow" (v5、ama方式の浅い完全探索、モジュール
            docstring「v5」参照) / "auto" (v5、時間予算で "exact_shallow"
            と "beam" を自動振り分け、`_resolve_auto_rollout_mode` 参照)。
            "beam"/"auto" 選択時は `beam_width` を明示的に指定すること
            (未検証のため既定値を持たない、"auto" が exact_shallow に
            振り分けられた場合は使われないが、事前に必須として要求する)。
        beam_width: "beam"/"auto" モード時のビーム幅 (必須、他では無視)。
        beam_use_exact_score: "beam" モード時に厳密得点 `exact_score` を
            使うか (既定True、`_rollout_once_beam` 参照)。
        early_exit_at_threshold: user指示②「答えを変えない打ち切り」
            (2026-08-21 追加、既定False)。True にすると、ロールアウトの
            running-max が `max(thresholds_ojama)` に相当する素点閾値を
            超えた時点で以降の手を計算せず打ち切る (`beam`/`exact_shallow`
            /`auto` モードのみ有効、`greedy` では無視)。running_best は
            深さに対して単調非減少なので **`prob_at_least` は打ち切りの
            有無に関わらず完全一致する** (`tests/test_mc_counter_
            estimator.py::TestEarlyExitAtThreshold` で確認)。**`mean`/
            `p25`/`p75` は打ち切り時点の値 [下限] になり、打ち切り無しの
            場合の最終値より低く出る可能性がある** — 確率チャネル
            (`prob_at_least`、学習用) にのみ使うこと。`thresholds_ojama`
            が空の場合は打ち切り基準が無いため無効化される (無視)。

    Returns:
        McCounterDistribution: mean/p25/p75/到達確率/平均打手数。

    シードは盤面+時間予算から決定論的に導出する (_mc_counter_seed、
    stateless: 同一入力には常に同一結果)。use_native の値は乱数系列
    (rng の使い方) に一切影響しない (内側の連鎖評価バックエンドのみが
    変わる)。
    """
    if rollout_mode not in _VALID_ROLLOUT_MODES:
        raise ValueError(f"rollout_mode は {_VALID_ROLLOUT_MODES} のいずれか: {rollout_mode!r}")
    if rollout_mode in (_ROLLOUT_MODE_BEAM, _ROLLOUT_MODE_AUTO) and beam_width is None:
        raise ValueError(
            f"rollout_mode={rollout_mode!r} 使用時は beam_width を明示的に指定してください "
            "(未検証のため既定値を持たない、幅の飽和点は実測して呼び出し側が決めること。"
            "'auto' が beam に振り分けられる場合に備え、'auto' でも必須にしている)。",
        )
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

    # 既知ツモ区間 (乱数不使用) の重複計算排除 (2026-08-21 追加、v3.3、
    # greedyモードのみ。beamモードは既知区間もビーム探索の一部として毎回
    # 評価するため対象外、モジュール docstring「v4」参照): 全ロールアウトで
    # 厳密に同一の結果になるため1回だけ計算して共有する (`_KnownPrefixState`
    # docstring参照、値は毎回再計算した場合と完全一致)。
    prefix = (
        _compute_known_prefix_state(board, time_budget_sec, known_pairs, sim, effective_use_native)
        if enable_prefix_dedup and rollout_mode == _ROLLOUT_MODE_GREEDY else None
    )

    # 答えを変えない打ち切り (2026-08-21 追加、v5 user指示②): 複数閾値が
    # あれば最大値を使う (低い閾値を超えても高い閾値はまだ分からないため、
    # 全閾値を超えた時点でしか打ち切れない、コーディネータ指摘の通り)。
    # お邪魔換算閾値→素点閾値の変換は厳密な逆変換
    # (`_ojama_threshold_to_score_threshold`、二分探索や近似ではない)。
    early_exit_score = (
        _ojama_threshold_to_score_threshold(max(thresholds_ojama), elapsed_sec)
        if early_exit_at_threshold and thresholds_ojama else None
    )

    ojama_values, hands_values = _run_rollouts(
        rollout_mode, board, time_budget_sec, colors, known_pairs, sim, rng, elapsed_sec,
        effective_use_native, prefix, n_rollouts, beam_width, beam_use_exact_score,
        early_exit_score,
    )

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
