"""boards_lean 系 npz (盤面グリッド原情報) → labeled_win 形式 CSV 変換ツール。

## 背景 (2026-08-12 user選択肢C確定)
CSV は使い捨ての派生物、npz (boards_lean_phase_l_2026-08-11/*.npz 等) が
恒久資産という前提で設計する。指標セットが増減しても npz 群を再収集せず
(動画も不要)、本ツールで CSV を安く再生成できるようにする。

## 薄い委譲構造 (指標が増減してもこのファイルは触らない)
指標の実計算は 100% `src/indicators_v2.py` の関数群に委譲する。本ツールが
持つのは「npz の 1 行 → Board 再構築 → レジストリ内の関数を呼んで列を書く」
という薄いループだけ。**新指標を追加したい場合は `GRID_ONLY_INDICATORS`
(または `GRID_ONLY_HEAVY_INDICATORS`) に 1 行足すだけで済む** (INDICATOR_
COLUMNS 末尾追加ルールと同じ精神)。

## 指標大整理 (2026-08-12 user確定、docs/INDICATOR_REORG_PROPOSAL_2026-08-12.md
「決定記録」節) の反映
- **a-1 完全重複の削除**: `*_raw` 列 (score と定数倍の完全重複) は CSV に
  一切出力しない。`saturated_chain_count` (current_max_chain と全19万場面で
  完全一致) は full profile のレジストリから削除。
  `absorption_capacity` (board_puyo_total と完全重複) は本ツールでは元々
  未収集のため対応不要 (既に GRID_ONLY_INDICATORS/HEAVY のいずれにも無い)。
- **b-1 center_bulge の分解**: 合成版 `center_bulge` (indicators_v2.py側は
  backwards compat のため関数自体は残す) の代わりに `center_bulge_color`
  (色ぷよのみ) / `center_bulge_ojama` (おじゃまのみ) の2列を出力する。
  新指標のため own/diff 両方を出す (DIFF_KEEP_OWN_NEW_COLUMNS)。
- **b-2 「相手との差」列**: `_analyze_reorg_diff_2026-08-12.py` の
  merge_asof(direction="backward") 方式を移植 (自分の各時刻に「相手の直近
  確定値」を対応付ける、対応成功率99.3%実測)。対象列の分類は下記
  DIFF_* 定数群を参照 (b-2決定記録の4分類 + 例外1分類、一括変換時の
  「diff化してよい列」明示リスト化 = 2026-08-10 user恒久指示に従う)。
  色ぷよ×おじゃまの関係は非単調 (おじゃま≈0なら色ぷよ多い=有利/材料律、
  おじゃま多いかつ色ぷよ多いなら「潰しが刺さった」不利シグナルの可能性、
  2026-08-12 user伝授) なため、own+diff+比率+交互作用の4列で表現する
  (単一の向きを決め打ちしない、詳細は DIFF_KEEP_OWN_PAIR_COLUMNS 直上の
  コメント)。
- **全消しボーナス予約中フラグ (`all_clear_bonus_pending`)**: 2026-08-12
  user伝授 (設計訂正版)。本質は「盤面が空かどうか」という瞬間状態ではなく、
  「全消しボーナス (2100点、おじゃま約30個相当) が未消費で残っている」
  という**持続状態**。npz の score 列 (post-hoc 復元可能) を使い、通常の
  落下ボーナス (最大 MAX_DROP_BONUS_SCORE=250) を大きく超え、かつ連鎖に
  紐付かない (chain_mechanism 未タグ) スコア増分を全消しボーナス計上と
  みなして ON にし、次の連鎖 (ボーナス消費) で OFF にする状態機械として
  実装する (詳細は `_compute_all_clear_bonus_pending` docstring)。相手の
  直近own値をそのまま carry する `opp_all_clear_bonus_pending` も出力する
  (diff ではない。フラグの差分は無意味で、相手が今ボーナス保持中かという
  生の状態そのものが受け側の判断材料になるため)。score OCR が信頼できない
  動画 (既知3本: c26/c58/c69) では NaN になる。
  **既知の精度限界**: `src/chain_detector.py` の `VideoChainTracker.
  all_clear_pending` が実運用 (認識パイプライン) で全く同じ状態を厳密に
  追跡済みだが、この値は npz に保存されていない (再収集が必要)。本関数は
  npz に既にある score/chain_mechanism からの post-hoc 近似であり、
  chain_mechanism のタグ網羅率が低いデータでは中型連鎖を誤って全消し
  ボーナスと判定する既知の過検出がある (実測: video_c143 で該当行約6.7%、
  詳細コメントは ALL_CLEAR_JUMP_UPPER_BOUND_SCORE 直上)。精度が要件を
  満たすかは要review、将来的な収集強化 (ChainEvent.is_all_clear の
  npz保存) が正確な解。

## 現状カバー範囲 (正直な記録、詳細は調査報告参照)
- npz に入っている原情報: grids / video_id / side / t_sec / game_idx /
  frame_idx / won / score / next1_a,b / dnext_a,b (--with-next 収集時) /
  chain_trigger_sec / chain_mechanism (--enable-chain-tracker 収集時)。
- **本ツールが計算するのは「盤面グリッドのみ」から求まる指標のみ**
  (GRID_ONLY_INDICATORS / GRID_ONLY_HEAVY_INDICATORS)。
- **未対応 (既知のギャップ、意図的に列を出力しない)**:
  ojama_net_balance / ojama_forecast — OjamaAccountingTracker は
  「毎フレームの BoardState 遷移 + tsumo_settled タイミング」を要求するが
  npz は STABLE 重複除去済みスナップショットのみで、この遷移列を保持して
  いない。score 列から近似復元する経路は別途検討 (調査報告参照)。
  列を出力しないことで `_resolve_features()` の列存在ガードが自動的に
  除外する (既存の「未収集列」と同じ扱い、後方互換)。
- next_pair 依存指標 (near_future_fire_power 等) は本ツールでは未実装
  (next1_a/b はレジストリの外で読み取り可能、拡張ポイントとしてコメントで
  明示するに留める)。
- 1 npz = 1 動画 (video_id 定数、side="1P"/"2P" 混在) を前提に diff 列を
  計算する (2026-08-12 実データで確認済み、data/indicators_v2/
  boards_lean_phase_l_2026-08-11/c143.npz 等)。

## 使い方
    python -m scripts.build_labeled_win_from_npz \\
        --npz-dir data/indicators_v2/boards_lean_phase_l_2026-08-11 \\
        --out data/indicators_v2/study/labeled_win_from_npz_2026-08-12.csv \\
        --profile light

--profile light: sub-ms 指標のみ (高速、反復開発向け)。
--profile full : current_max_chain 等の重い連鎖シミュ系も含む。既定で
    Rust拡張 puyo_core (`src/puyo_core_bridge.py`) に自動載せ替えられる
    (2026-08-13 追加、未ビルド環境では自動的に既存 Python 実装へ
    フォールバックする)。`--no-native` でこの載せ替えを強制的に無効化できる
    (パリティ検証・デバッグ用)。
"""
from __future__ import annotations

import argparse
import csv
import functools
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import BOARD_COLS, COLOR_EMPTY, COLOR_UNKNOWN, Board  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402
from src.production_config import GHOST_CHAIN_RULE_ENABLED  # noqa: E402
from src.puyo_core_bridge import NATIVE_AVAILABLE as _PUYO_CORE_AVAILABLE  # noqa: E402
from src.puyo_core_bridge import (  # noqa: E402
    chain_metrics_after_drops as _native_chain_metrics_after_drops,
    simulate_after_drops as _native_simulate_after_drops,
    simulate_chain as _native_simulate_chain,
)
from src.scoring import ALL_CLEAR_BONUS  # noqa: E402

# ============================
# 指標レジストリ (薄い委譲構造の核心)
# ============================
# 値は IndicatorV2Value を返す Board -> value の関数のみ (score 列を自動で
# 書く。*_raw は a-1 決定 (2026-08-12) により出力しない)。新指標を増やす/
# 減らす場合はこの2つの dict を編集するだけで良く、変換ループ本体
# (_compute_row) は触らない。

# light: 実測 <0.15ms/行 (ベンチ scripts/_verify... 2026-08-12、n=500)。
# center_bulge (合成版) は b-1 決定により center_bulge_color/_ojama の
# 2列に分解済み (indicators_v2.py 側の center_bulge() 関数自体は
# backwards compat のため残っている)。
GRID_ONLY_INDICATORS: dict[str, Callable[[Board], "iv.IndicatorV2Value"]] = {
    "board_color_puyo_total": iv.board_color_puyo_total,
    "board_puyo_total": iv.board_puyo_total,
    "max_column_height": iv.max_column_height,
    "column_bumpiness": iv.column_bumpiness,
    "death_margin": iv.death_margin,
    "death_margin_neighbor": iv.death_margin_neighbor,
    "center_bulge_color": iv.center_bulge_color,
    "center_bulge_ojama": iv.center_bulge_ojama,
    "board_ojama_count": iv.board_ojama_count,
}
# full: 連鎖シミュレーションを要する重い指標 (実測 1〜19ms/行)。
# --profile full 指定時のみ計算する。
# saturated_chain_count は a-1 決定 (2026-08-12) により削除済み
# (current_max_chain と19万場面で完全一致、src/production_config.py の
# ATTRIBUTION_EXCLUDED_INDICATORS にも同根拠で既に記録あり)。
GRID_ONLY_HEAVY_INDICATORS: dict[str, Callable[[Board], "iv.IndicatorV2Value"]] = {
    "current_max_chain": iv.current_max_chain,
    "dig_resistance": iv.dig_resistance,
    "ukeyasusa": iv.ukeyasusa,
    "sub_chain_count": iv.sub_chain_count,
}

VALID_PROFILES: tuple[str, ...] = ("light", "full")

# ============================
# Rust ネイティブ拡張 (puyo_core) 載せ替え (2026-08-13 追加)
# ============================
# GRID_ONLY_HEAVY_INDICATORS (上記4列) は ChainSimulator.simulate の繰り返し
# 呼び出しが支配的コスト (current_max_chain: 30回、dig_resistance: 4回、
# ukeyasusa: dig_resistance丸ごと再利用、sub_chain_count: 最大61回、実測
# 1〜19ms/行)。scripts/mc_counter_estimator.py と同じ「呼び出し側に native
# 分岐を足す」パターンで src/puyo_core_bridge.py 経由の Rust 実装に載せ替える
# (indicators_v2.py 自体は無変更、完全一致は
# tests/test_build_labeled_win_from_npz.py::TestNativeHeavyIndicatorParity
# で担保)。幽霊連鎖ルール (`_SHARED_SIMULATOR` の設定) と揃えるため、native
# 呼び出しは必ず exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED を
# 明示する。dig_resistance のおじゃま落下 (`ChainSimulator.drop_ojama`) は
# 乱数 (端数列のランダム選択) を含むため native化せず既存 Python 実装のまま
# 保持する (載せ替え対象は連鎖シミュレーション部分のみ、既存の非決定性は
# この載せ替えでは変えない・直さない)。
#
# **既知の native 制約 (2026-08-13 発見、`_board_is_gravity_consistent`
# 参照)**: puyo_core の重力実装は入力盤面が既に gravity一貫であることを
# 前提にした最適化であり、認識由来の浮きぷよ (既存の重力違反、既知欠陥) を
# 含む盤面では連鎖数がズレる場合がある (実測: 実盤面1,097件中1件で
# current_max_chain の chain_count が 2→1 に食い違った)。全 native
# 呼び出しの直前で `_board_is_gravity_consistent(board)` を確認し、違反時は
# 既存 Python 実装へフォールバックすることで「完全一致」を保証する
# (native の恒久修正は別課題、native/puyo_core 自体は本タスクで変更しない)。

# takapt定石 (列×色30通り) の (col, color) 一覧。`iv._takapt_best_drop` と
# 同一探索順 (col昇順→色昇順、scripts/mc_counter_estimator.py の
# `_DROP_CANDIDATES_30` と同一定義を独立複製 — 両ファイルとも編集しない
# 制約のため)。
_NATIVE_DROP_CANDIDATES_30: tuple[tuple[int, int], ...] = tuple(
    (col, color) for col in range(BOARD_COLS) for color in iv.IGNITION_TRIAL_COLORS
)


def _board_is_gravity_consistent(board: Board) -> bool:
    """各列に「浮きぷよ」由来のギャップが無いか判定する (native 載せ替えの安全弁)。

    **2026-08-13 実データ調査で発見した既知の native 制約**: puyo_core
    (Rust) の重力実装は、消去後の各列シフト量を「その列で消えたセル数」
    として計算するビット圧縮の定数時間実装であり、入力盤面が既に
    gravity一貫 (各列で占有セルの下に空きが無い) であることを前提に
    最適化されている。既に重力違反 (認識由来の浮きぷよ、
    `project_gravity_violation_regen_lead_2026-07-30` 系の既知欠陥) を含む
    盤面を渡すと、2手目以降の連鎖判定が食い違う場合がある (実測:
    1,097件の実盤面サンプル中1件で current_max_chain の chain_count が
    2→1 に食い違うことを確認済み。手動トレースで Python
    ChainSimulator の chain_count=2 が正しいゲーム挙動と確認済み)。
    この関数で事前検知し、違反時は呼び出し側が既存 Python 実装
    (`indicators_v2.py`、常に正しい) へフォールバックする
    (「完全一致」要件を守るための安全弁、native の恒久修正は別課題)。
    UNKNOWN セルは占有扱いしない (`height_of`/Rust `occ` と同じ意味論)。
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


def _native_takapt_best_drop(
    board: Board,
) -> "tuple[int, Board | None, object | None]":
    """takapt定石探索の native 版 (`iv._takapt_best_drop` と同一意味論)。

    `sub_chain_count` が「本線発火後の final_board」を必要とするため、
    盤面付きバッチAPI `simulate_after_drops` で候補ごとの ChainSimResult も
    保持して返す (`iv._takapt_best_drop` 内で最良候補の simulate 結果を
    使い捨てていた分の再計算を省く最適化)。

    Returns:
        (最大連鎖数, 1個追加後の盤面 [連鎖解決前] または None, その
        ChainSimResult または None)。`>` による厳密な大小比較のため同値は
        先に見つかった (col昇順→色昇順) 候補を保持する
        (`iv._takapt_best_drop` の tie-break と同一)。
    """
    best_chain = 0
    best_board: "Board | None" = None
    best_result = None
    for r in _native_simulate_after_drops(
        board, _NATIVE_DROP_CANDIDATES_30,
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    ):
        if r is None:
            continue
        if r.chain_result.chain_count > best_chain:
            best_chain = r.chain_result.chain_count
            best_board = r.dropped_board
            best_result = r.chain_result
    return best_chain, best_board, best_result


def _native_current_max_chain(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 III-1 `current_max_chain` の native 分岐版。

    use_native=False または拡張未導入時は既存 `iv.current_max_chain` に
    そのまま委譲する (完全一致、indicators_v2.py 自体は無変更)。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.current_max_chain(board)
    best_chain = 0
    for r in _native_chain_metrics_after_drops(
        board, _NATIVE_DROP_CANDIDATES_30,
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    ):
        if r is None:
            continue
        chain_count, _exact_score = r
        if chain_count > best_chain:
            best_chain = chain_count
    raw = float(best_chain)
    return iv.IndicatorV2Value(score=iv._clamp01(raw / iv.NORM_MAX_CHAIN), raw=raw)


def _native_dig_resistance_one(
    board: Board, base_chain: int, n_ojama: int,
) -> float:
    """dig_resistance 1点分の native 版 (`iv._dig_resistance_one` と同一計算)。

    おじゃま落下 (`drop_ojama`) は乱数を含むため既存 Python 実装
    (`iv._SHARED_SIMULATOR`、上部コメント参照) をそのまま再利用し、連鎖
    シミュレーションのみ native に置き換える。
    """
    try:
        ojama_board = iv._SHARED_SIMULATOR.drop_ojama(board, n_ojama)
    except Exception:
        return 0.0
    if ojama_board.is_dead():
        return 0.0
    try:
        post_chain = _native_simulate_chain(
            ojama_board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
        ).chain_count
    except Exception:
        return 0.0
    survival = min(1.0, post_chain / float(base_chain))
    dig = 1.0 if post_chain >= iv.OJAMA_DEFENSE_DIG_MIN_CHAIN else 0.0
    return (
        iv.OJAMA_DEFENSE_SURVIVAL_WEIGHT * survival
        + iv.OJAMA_DEFENSE_DIG_WEIGHT * dig
    )


def _native_dig_resistance(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 VI-1 `dig_resistance` の native 分岐版 (`_native_dig_resistance_one`
    に3点分を委譲)。use_native=False または拡張未導入時は既存
    `iv.dig_resistance` にそのまま委譲する (完全一致)。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.dig_resistance(board)
    if board.is_dead():
        return iv.IndicatorV2Value(score=0.0, raw=0.0)
    base_chain = max(
        1, _native_simulate_chain(
            board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
        ).chain_count,
    )
    scores = [
        _native_dig_resistance_one(board, base_chain, n)
        for n in iv.OJAMA_DEFENSE_TEST_COUNTS
    ]
    avg = sum(scores) / len(scores) if scores else 0.0
    return iv.IndicatorV2Value(score=iv._clamp01(avg), raw=float(avg))


def _native_ukeyasusa(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 X-1 `ukeyasusa` の native 分岐版 (dig_resistance 部分のみ
    native化、absorption_capacity/death_margin は sim 不要のため既存関数を
    直接使う)。use_native=False または拡張未導入時は既存 `iv.ukeyasusa` に
    そのまま委譲する。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.ukeyasusa(board)
    s_abs = iv.absorption_capacity(board).score
    s_dig = _native_dig_resistance(board, use_native=True).score
    s_death = iv.death_margin(board).score
    score = (
        iv.UKEYASUSA_W_ABSORPTION * s_abs
        + iv.UKEYASUSA_W_DIG * s_dig
        + iv.UKEYASUSA_W_DEATH * s_death
    )
    raw_abs = iv.absorption_capacity(board).raw
    return iv.IndicatorV2Value(score=iv._clamp01(score), raw=raw_abs)


def _native_sub_chain_count(
    board: Board, use_native: bool = True,
) -> "iv.IndicatorV2Value":
    """既存 XII-4 `sub_chain_count` の native 分岐版。

    1手目探索の最良候補が保持する ChainSimResult (`_native_takapt_best_drop`)
    をそのまま「本線発火結果」として再利用する (`iv.sub_chain_count` の
    `sim.simulate(best_board)` 再計算と等価。ChainSimulator.simulate は
    決定的関数のため同一盤面には常に同一結果、再計算を省く最適化)。
    2手目探索は盤面を返さない軽量バッチAPIで行う。
    """
    if not (use_native and _PUYO_CORE_AVAILABLE and _board_is_gravity_consistent(board)):
        return iv.sub_chain_count(board)
    best_chain, best_board, best_result = _native_takapt_best_drop(board)
    if best_board is None or best_chain == 0 or best_result is None:
        return iv.IndicatorV2Value(score=0.0, raw=0.0)
    if best_result.chain_count == 0:
        return iv.IndicatorV2Value(score=0.0, raw=0.0)
    post_board = best_result.final_board
    sub_best_chain = 0
    for r in _native_chain_metrics_after_drops(
        post_board, _NATIVE_DROP_CANDIDATES_30,
        exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
    ):
        if r is None:
            continue
        chain_count, _exact_score = r
        if chain_count > sub_best_chain:
            sub_best_chain = chain_count
    raw = float(sub_best_chain)
    return iv.IndicatorV2Value(score=iv._clamp01(raw / iv.NORM_SUB_CHAIN), raw=raw)


# full profile 重い4列の native 版レジストリ (use_native は呼び出し側が
# functools.partial で bind する、`_resolve_indicator_registry` 参照)。
GRID_ONLY_HEAVY_INDICATORS_NATIVE: dict[str, Callable[..., "iv.IndicatorV2Value"]] = {
    "current_max_chain": _native_current_max_chain,
    "dig_resistance": _native_dig_resistance,
    "ukeyasusa": _native_ukeyasusa,
    "sub_chain_count": _native_sub_chain_count,
}

# ============================
# b-2: 「相手との差」列 (2026-08-12 user確定、決定記録参照)
# ============================
# 一括変換の恒久指示 (2026-08-10 user指示) に従い、diff化してよい列を
# 下記4分類 + 例外1分類で明示リスト化する (リストに無い列は安全側デフォルト
# として diff化しない=own のみ)。

# (1) own を削除して diff_ に完全置換する列。b-2実測で11項目中10項目が
# 「自分のみ」より「差」で当てやすさ改善 (連結最大サイズは終盤0.508→0.567)。
DIFF_REPLACE_OWN_COLUMNS: tuple[str, ...] = (
    "max_column_height",
    "column_bumpiness",
    "death_margin",
    "death_margin_neighbor",
    "conn_pair_count",
    "conn_max_group_size",
)

# (2) own+diff 両方 (full profile 限定・重い連鎖シミュ系)。b-2本文は
# grid-only 系のみ実測済みで、この4列は「同様にdiff化するがownも残すか
# 迷う場合は両方出して報告」という task 側の指示に従い own+diff 両方を
# 出力する (2026-08-12 コーダ判断、user確認待ち)。
DIFF_KEEP_OWN_HEAVY_COLUMNS: tuple[str, ...] = (
    "current_max_chain",
    "dig_resistance",
    "ukeyasusa",
    "sub_chain_count",
)

# (3) own+diff 両方 (user指示8/12最重要: 色ぷよ総数は単独指標でなく
# おじゃま総数とペアで見る。差にすると向きが逆転する謎の答えとして、
# 単純な own/diff置換対象から外し、両方+比率/積の交互作用列で表現する)。
#
# ## 色ぷよ×おじゃまの関係は非単調 (2026-08-12 user伝授、重要)
# 色ぷよ総数とおじゃま総数の「多いほど有利/不利」という単一の向きは無い。
# 条件によって意味が反転する:
#   - おじゃま≈0 のとき: 色ぷよが多いほど有利 (材料律、reference_
#     color_puyo_material_law_2026-08-04「色ぷよの多さは未構造でも構造化
#     に向かえば強い構造につながる」)。
#   - おじゃまが多い かつ 色ぷよも多いとき: 「潰しが刺さった」
#     (色ぷよを溜めて構えていたところに攻撃を受け、構築中の連鎖ごと
#     埋まった) 可能性があり、この場合は不利シグナルになりうる。
# この非単調性は単純な差分・単一係数では表現できないため、own両方+比率+
# 交互作用の4列 (下記 PAIR_INTERACTION_COLUMNS 含む) で学習側に条件分岐的な
# 判断余地を残す設計にしている (単一の「向き」を決め打ちしない)。
DIFF_KEEP_OWN_PAIR_COLUMNS: tuple[str, ...] = (
    "board_color_puyo_total",
    "board_puyo_total",
    "board_ojama_count",
)

# (4) own+diff 両方 (b-1 新設の分解列。新指標のため測定用に両方出す)。
DIFF_KEEP_OWN_NEW_COLUMNS: tuple[str, ...] = (
    "center_bulge_color",
    "center_bulge_ojama",
)

# (5) diff化しない例外 (own のみ残す)。b-2本文: 3個連結は差にすると終盤の
# 不可解な逆転がさらに悪化するため対象外。
DIFF_EXEMPT_OWN_ONLY_COLUMNS: tuple[str, ...] = (
    "conn_triple_count",
)

# connectivity_observation() は registry の外で個別に書く固定3列
# (このツール独自の慣習、GRID_ONLY_INDICATORS には乗らない)。
CONN_ALWAYS_PRESENT_COLUMNS: tuple[str, ...] = (
    "conn_pair_count", "conn_triple_count", "conn_max_group_size",
)

# ============================
# 全消しボーナス予約中フラグ (all_clear_bonus_pending)
# — 2026-08-12 user伝授 (設計訂正版)
# ============================
# 本質は「盤面が空かどうか」という瞬間状態ではなく、「全消しボーナス
# (2100点、おじゃま約30個相当 = 2100/OJAMA_RATE_STANDARD(70)) が未消費で
# 残っている」という持続状態。瞬間フラグだと学習が「空である」ことそのもの
# を意味だと誤解しかねないため、ボーナスが「予約中」の区間全体を ON にする
# 状態機械として実装する (_compute_all_clear_bonus_pending)。
# 全消しボーナスの固定値は src.scoring.ALL_CLEAR_BONUS が単一情報源
# (src/chain_detector.py VideoChainTracker が実運用でこの値を「次の連鎖
# 発火時に持ち越し加算」する仕様で実装済み、docs/PUYO_RULES_CONFIRMED_
# 2026-07-22.md 準拠)。ここでの再定義はしない (マジックナンバー重複回避)。
ALL_CLEAR_BONUS_SCORE: int = ALL_CLEAR_BONUS  # = 2100 (src.scoring と同一値)
MAX_DROP_BONUS_SCORE: int = 250  # 通常の落下ボーナス上限 (これ以下は通常運転)
# 連鎖外ジャンプ (ON トリガー) の判定帯。下限は通常の落下ボーナス上限と全消し
# ボーナスのちょうど中間 (双方から十分離してマージンを確保)。
#
# 【実データ検証 (2026-08-12、video_c143) で判明した既知の精度限界】
# chain_mechanism タグは実際の連鎖の一部にしか付かない (1P側の1175点超
# ジャンプ96件中、タグ付きはわずか4件、92件は untagged の実連鎖=数千〜
# 数万点)。下限のみだと数万点ジャンプまで誤検出するため上限も設ける
# (固定2100点から大きく外れる巨大ジャンプは明らかに普通の大型連鎖)。
# 上限を設けても [1175,3150] 帯だけで1動画中72件 (1P+2P) が候補になり、
# 現実的な全消し頻度より明らかに過大 → 帯に入るが実際は中型連鎖という
# 取りこぼしが残る (chain_mechanism のタグ網羅率向上が本質的な解、
# 現状は近似ヒューリスティックである旨をコメントで明示)。
ALL_CLEAR_JUMP_LOWER_BOUND_SCORE: float = (
    (MAX_DROP_BONUS_SCORE + ALL_CLEAR_BONUS_SCORE) / 2.0
)  # = 1175.0
ALL_CLEAR_JUMP_UPPER_BOUND_SCORE: float = ALL_CLEAR_BONUS_SCORE * 1.5  # = 3150.0

# 状態機械が own 列として生成する固定列 (registry の外、CONN_ALWAYS_
# PRESENT_COLUMNS と同じ位置付け)。
TEMPORAL_STATE_COLUMNS: tuple[str, ...] = (
    "all_clear_bonus_pending",
)

# chain_mechanism が「連鎖タグ無し」を意味する値の集合 (小文字化して比較)。
# 空文字が通常だが、npz 側の型変換で "nan"/"none" 文字列化する経路があっても
# 安全に「タグ無し」として扱う (誤って「連鎖あり」と誤認しない安全側デフォルト)。
_NO_CHAIN_TAG_VALUES: frozenset[str] = frozenset({"", "nan", "none"})

# 相手側の「直近own値そのもの」を carry する列 (diff = own-opp ではなく
# opp_<col> = 相手の直近own値、2026-08-12 user伝授)。all_clear_bonus_pending
# はフラグ (0/1) なので差分には意味が無く、「相手が今ボーナス保持中か」と
# いう相手側の生の状態そのものが受け側の判断材料になる (自分が攻撃を受ける
# 側かどうかの判断に相手のボーナス保持状態が効くため)。
CARRY_OPPONENT_COLUMNS: tuple[str, ...] = (
    "all_clear_bonus_pending",
)

# ペア交互作用列 (b-2 user指示8/12「色ぷよ×おじゃまの関係性を特徴として
# 捉える」の最小実装)。上記の非単調性コメント参照: color_ojama_ratio_own は
# 「今どれだけ色ぷよ優勢か」、color_diff_x_ojama_diff は「色ぷよを増やしつつ
# おじゃまも増えた (催促を受けながら構築) / 減らしつつ減った (受けて掘った)
# 等の同時変化」を捉える。どちらも単調な有利/不利の向きを仮定しない
# (向きの決定は学習に委ねる、CLAUDE.md「観測軸を提供→学習で重要度を発見」)。
# 0除算防止の epsilon はマジックナンバー化を避けて定数化。
COLOR_OJAMA_RATIO_EPS: float = 1e-6
PAIR_INTERACTION_COLUMNS: tuple[str, ...] = (
    "color_ojama_ratio_own", "color_diff_x_ojama_diff",
)


def _resolve_indicator_registry(
    profile: str, use_native: bool = True,
) -> dict[str, Callable[[Board], "iv.IndicatorV2Value"]]:
    """profile に応じて使う指標レジストリを確定する (light はheavy除外)。

    use_native (2026-08-13 追加、既定 True): full profile の重い4列を
    Rust拡張 puyo_core 経由で計算する native 版レジストリ
    (GRID_ONLY_HEAVY_INDICATORS_NATIVE) に切り替える。拡張未導入環境では
    各関数が自動的に既存 Python 実装 (GRID_ONLY_HEAVY_INDICATORS 相当) へ
    フォールバックするため、通常この引数は既定値のままでよい。False を
    渡すと native 分岐を無条件で無効化する (パリティ検証・デバッグ用)。
    """
    if profile != "full":
        return dict(GRID_ONLY_INDICATORS)
    heavy = {
        name: functools.partial(fn, use_native=use_native)
        for name, fn in GRID_ONLY_HEAVY_INDICATORS_NATIVE.items()
    }
    return {**GRID_ONLY_INDICATORS, **heavy}


def _resolve_diff_target_columns(
    registry: dict[str, Callable[[Board], "iv.IndicatorV2Value"]],
) -> list[str]:
    """レジストリに実在する列のうち diff_ 化対象を返す (b-2 決定記録の4分類)。

    DIFF_EXEMPT_OWN_ONLY_COLUMNS (例外) は候補に含めない。レジストリに無い
    列 (light profile での heavy 系等) は自動的に除外される。
    """
    candidates = (
        DIFF_REPLACE_OWN_COLUMNS + DIFF_KEEP_OWN_PAIR_COLUMNS
        + DIFF_KEEP_OWN_NEW_COLUMNS + DIFF_KEEP_OWN_HEAVY_COLUMNS
    )
    available = set(registry.keys()) | set(CONN_ALWAYS_PRESENT_COLUMNS)
    return [c for c in candidates if c in available]


def _resolve_carry_target_columns(
    registry: dict[str, Callable[[Board], "iv.IndicatorV2Value"]],
) -> list[str]:
    """実在する列のうち opp_<col> carry 対象を返す。

    CARRY_OPPONENT_COLUMNS (相手の直近own値をそのまま残す列、
    all_clear_bonus_pending 等のフラグ向け、2026-08-12 user伝授) の実在
    チェック。TEMPORAL_STATE_COLUMNS は registry の外で常に生成されるため
    別途 available に含める。
    """
    available = set(registry.keys()) | set(TEMPORAL_STATE_COLUMNS)
    return [c for c in CARRY_OPPONENT_COLUMNS if c in available]


# CSV メタ列 (labeled_win.csv 既存フォーマットと同じ、tsumo は近似値)
META_COLUMNS: tuple[str, ...] = (
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side", "won",
)


def _final_fieldnames(profile: str) -> list[str]:
    """出力CSVの最終列順を構築する (a-1 raw削除・b-1 分解・b-2 diff/carry化を反映)。

    own 側の出力対象は DIFF_REPLACE_OWN_COLUMNS を除いた列のみ (diff_ に
    完全置換した列は own を書かない、CSVには乗せず内部計算だけに使う)。
    """
    registry = _resolve_indicator_registry(profile)
    own_candidates = (
        list(registry.keys()) + list(CONN_ALWAYS_PRESENT_COLUMNS)
        + list(TEMPORAL_STATE_COLUMNS)
    )
    own_output_cols = [c for c in own_candidates if c not in DIFF_REPLACE_OWN_COLUMNS]
    diff_output_cols = [f"diff_{c}" for c in _resolve_diff_target_columns(registry)]
    carry_output_cols = [f"opp_{c}" for c in _resolve_carry_target_columns(registry)]
    return (
        list(META_COLUMNS) + own_output_cols + diff_output_cols
        + carry_output_cols + list(PAIR_INTERACTION_COLUMNS)
    )


def _compute_row(
    grid: np.ndarray,
    registry: dict[str, Callable[[Board], "iv.IndicatorV2Value"]],
) -> dict[str, float]:
    """1 スナップショットの grid からレジストリ内の全指標を計算する。

    Board 再構築を 1 回だけ行い、レジストリの全関数に共有する (npz→Board
    復元コストの重複を避ける)。connectivity_observation (score を持たない
    タプル戻り値) は個別に追加する (collect_indicators_v2.py と同じ流儀)。

    a-1 決定 (2026-08-12) により `*_raw` はここで一切書かない (score のみ、
    CSV出力から完全重複列を削除)。
    """
    board = Board.from_list(grid.tolist())
    row: dict[str, float] = {}
    for name, fn in registry.items():
        row[name] = fn(board).score
    total_conn, _ = iv.connectivity_observation(board)
    row["conn_pair_count"] = float(total_conn.pair_count)
    row["conn_triple_count"] = float(total_conn.triple_count)
    row["conn_max_group_size"] = float(total_conn.max_group_size)
    return row


def _grouped_side_frames(
    df: pd.DataFrame, side: str, group_cols: list[str], diff_cols: list[str],
) -> dict[tuple, pd.DataFrame]:
    """指定 side の行を (video_id, game_idx) 単位の t_sec 昇順 df に分割する。

    scripts/_analyze_reorg_diff_2026-08-12.py の `_grouped_opp_dict` を移植
    (相手側として使う際の下ごしらえ、b-2)。
    """
    src = df[df["side"] == side][group_cols + ["t_sec"] + diff_cols]
    out: dict[tuple, pd.DataFrame] = {}
    for key, g in src.groupby(group_cols, sort=False):
        out[key] = g.sort_values("t_sec").drop(columns=group_cols).reset_index(drop=True)
    return out


def _asof_attach_opponent(
    self_df: pd.DataFrame,
    opp_groups: dict[tuple, pd.DataFrame],
    group_cols: list[str],
    diff_cols: list[str],
) -> pd.DataFrame:
    """自分の各行に「相手の直近確定値」を merge_asof(backward) で対応付ける。

    scripts/_analyze_reorg_diff_2026-08-12.py の `_asof_join_by_game` を移植
    (b-2)。相手の同時刻以前の最新スナップショットが無い先頭区間は NaN に
    なる (実対応成功率は同スクリプトの検証で99.3%、b-2 決定記録参照)。
    """
    opp_col_names = [f"_opp_{c}" for c in diff_cols]
    rename_map = dict(zip(diff_cols, opp_col_names))
    parts: list[pd.DataFrame] = []
    for key, g in self_df.groupby(group_cols, sort=False):
        gs = g.sort_values("t_sec").reset_index(drop=True)
        opp_g = opp_groups.get(key)
        if opp_g is None or len(opp_g) == 0:
            gs2 = gs.copy()
            for c in opp_col_names:
                gs2[c] = float("nan")
            parts.append(gs2)
            continue
        opp_renamed = opp_g.rename(columns=rename_map)
        merged = pd.merge_asof(gs, opp_renamed, on="t_sec", direction="backward")
        parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else self_df


def _attach_opponent_diff_columns(
    rows: list[dict], diff_cols: list[str], carry_cols: tuple[str, ...] = (),
) -> list[dict]:
    """1P/2P 行に「相手の直近確定値」由来の列を追加する (b-2)。

    (video_id, game_idx) 単位で相手側の直近確定値を対応付け、
    diff_cols は diff_<col> = 自分の値 - 相手の直近値 を、carry_cols は
    opp_<col> = 相手の直近値そのもの (差分をとらない) を書く。carry は
    all_clear_bonus_pending 等の 0/1 フラグ向け (2026-08-12 user伝授:
    フラグの差分は無意味で、相手の生の状態そのものが判断材料になる)。
    1 npz = 1 動画である
    前提 (video_id は事実上定数だが、複数動画結合にも安全なよう group_cols
    に含める)。side が "1P"/"2P" 以外の行は対応不能として NaN で素通しする
    (サイレントにデータを消さない)。

    Args:
        rows: convert_one_npz が積んだ meta+indicator 行のリスト。
        diff_cols: diff_ 化する base 列名リスト (_resolve_diff_target_columns)。
        carry_cols: opp_ carry 化する base 列名リスト
            (_resolve_carry_target_columns、既定は空=後方互換)。

    Returns:
        list[dict]: 各行に diff_<col> / opp_<col> キーを追加したリスト。
    """
    pair_cols = list(diff_cols) + [c for c in carry_cols if c not in diff_cols]
    if not pair_cols or not rows:
        return rows
    df = pd.DataFrame(rows)
    group_cols = ["video_id", "game_idx"]
    known_sides = ("1P", "2P")
    parts: list[pd.DataFrame] = []
    for self_side, opp_side in (("1P", "2P"), ("2P", "1P")):
        self_df = df[df["side"] == self_side].reset_index(drop=True)
        if len(self_df) == 0:
            continue
        opp_groups = _grouped_side_frames(df, opp_side, group_cols, pair_cols)
        parts.append(_asof_attach_opponent(self_df, opp_groups, group_cols, pair_cols))
    other_df = df[~df["side"].isin(known_sides)].reset_index(drop=True)
    if len(other_df) > 0:
        for c in pair_cols:
            other_df[f"_opp_{c}"] = float("nan")
        parts.append(other_df)
    combined = pd.concat(parts, ignore_index=True) if parts else df
    for c in diff_cols:
        combined[f"diff_{c}"] = combined[c] - combined[f"_opp_{c}"]
        combined = combined.drop(columns=[f"_opp_{c}"])
    for c in carry_cols:
        combined[f"opp_{c}"] = combined[f"_opp_{c}"]
        combined = combined.drop(columns=[f"_opp_{c}"])
    return combined.to_dict("records")


def _add_pair_interaction_columns(rows: list[dict]) -> list[dict]:
    """色ぷよ×おじゃまのペア交互作用列を追加する (b-2 user指示8/12)。

    色ぷよ総数は単独指標でなくおじゃま総数とのペアで見る必要がある
    (2026-08-12 user指示)。own の比率 + diff の積の2列を新設する。
    diff_board_color_puyo_total / diff_board_ojama_count は常に
    DIFF_KEEP_OWN_PAIR_COLUMNS 経由で存在する前提 (存在しない場合は
    NaN を素通しする、サイレント失敗にしない)。
    """
    for row in rows:
        color = float(row.get("board_color_puyo_total", float("nan")))
        ojama = float(row.get("board_ojama_count", float("nan")))
        row["color_ojama_ratio_own"] = color / (color + ojama + COLOR_OJAMA_RATIO_EPS)
        diff_color = float(row.get("diff_board_color_puyo_total", float("nan")))
        diff_ojama = float(row.get("diff_board_ojama_count", float("nan")))
        row["color_diff_x_ojama_diff"] = diff_color * diff_ojama
    return rows


def _all_clear_next_state(state: float, delta: float, chain_tag: str) -> float:
    """1ステップ分の状態遷移 (ON/OFF 判定の核心。定数の意味は
    ALL_CLEAR_JUMP_LOWER/UPPER_BOUND_SCORE・MAX_DROP_BONUS_SCORE 直上の
    コメント、既知の精度限界は _all_clear_state_for_group docstring 参照)。
    """
    is_tagged = chain_tag.strip().lower() not in _NO_CHAIN_TAG_VALUES
    in_bonus_band = (
        ALL_CLEAR_JUMP_LOWER_BOUND_SCORE < delta <= ALL_CLEAR_JUMP_UPPER_BOUND_SCORE
    )
    if state == 0.0:
        return 1.0 if (not is_tagged and in_bonus_band) else 0.0
    return 0.0 if (is_tagged or delta > MAX_DROP_BONUS_SCORE) else 1.0


def _all_clear_state_for_group(
    scores: list[int], chain_tags: list[str],
) -> list[float]:
    """1つの (video_id, side, game_idx) 内、t_sec昇順の score/chain_mechanism
    列から all_clear_bonus_pending の状態遷移を計算する内部ヘルパー。

    状態機械の要点 (2026-08-12 user伝授、設計訂正版。全消しの本質は瞬間の
    空盤面でなく「ボーナス2100点が未消費で残っている」持続状態):
    新しい試合の最初の行は「ボーナス未保持」(0.0) から始まる (試合を
    またいでボーナスは持ち越されない)。ON/OFF 判定の実体は
    _all_clear_next_state() に委譲する。score が読めない行 (-1、または
    直前の既知scoreがまだ無い) は増分を計算できないため NaN を返し、
    状態自体は変更しない (score OCR破綻の既知3動画 c26/c58/c69 では、
    行0以外の列全体が実質NaNになる)。

    既知の精度限界 (要review): chain_mechanism のタグ網羅率が低いデータ
    では ON 判定帯に中型連鎖の得点も多数入り込み、全消し以外を誤って ON に
    する (実測: video_c143 1本で1P+2P合計72件が該当、現実的な全消し頻度
    より過大)。本実装は user 指定の検出方法をそのまま実装した近似
    ヒューリスティックであり、精度向上には chain_mechanism タグの網羅率
    向上または別の検出法が必要。
    """
    n = len(scores)
    out: list[float] = [float("nan")] * n
    state = 0.0
    last_valid_score: int | None = None
    for i in range(n):
        if i == 0:
            out[0] = state
            if scores[0] != -1:
                last_valid_score = scores[0]
            continue
        cur = scores[i]
        if cur == -1 or last_valid_score is None:
            if cur != -1:
                last_valid_score = cur
            continue
        delta = float(cur - last_valid_score)
        last_valid_score = cur
        state = _all_clear_next_state(state, delta, chain_tags[i])
        out[i] = state
    return out


def _compute_all_clear_bonus_pending(
    rows: list[dict], scores: list[int], chain_mechanisms: list[str],
) -> None:
    """(video_id, side, game_idx) 単位で全消しボーナス予約中フラグを計算し、
    rows[i]["all_clear_bonus_pending"] に in-place で書き込む。

    scores / chain_mechanisms は rows と同じ添字で対応する前提 (convert_
    one_npz が同じループで構築するため)。状態遷移の実体は
    _all_clear_state_for_group() 参照。
    """
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        key = (r["video_id"], r["side"], r["game_idx"])
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        idxs.sort(key=lambda i: rows[i]["t_sec"])
        group_scores = [scores[i] for i in idxs]
        group_tags = [chain_mechanisms[i] for i in idxs]
        states = _all_clear_state_for_group(group_scores, group_tags)
        for pos, i in enumerate(idxs):
            rows[i]["all_clear_bonus_pending"] = states[pos]


def _approx_tsumo(rows_meta: list[dict]) -> None:
    """(video_id, side, game_idx) 内の t_sec 順位で手数を近似する (in-place)。

    実際の tsumo_count (RecognitionPipeline.tsumo_count) とは異なり得る近似
    値 (STABLE スナップショット数 != 常に手数、ojama落下等でも新スナップ
    ショットが生まれるため)。現行 FEATURES/FEATURE_CANDIDATES はどの列も
    tsumo に依存しないため、メタ情報としての近似で実害はない
    (model_indicator_win.META_COLS が特徴量から除外する対象と同じ扱い)。
    """
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows_meta):
        key = (r["video_id"], r["side"], r["game_idx"])
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        idxs.sort(key=lambda i: rows_meta[i]["t_sec"])
        for rank, i in enumerate(idxs):
            rows_meta[i]["tsumo"] = rank


def _build_meta_rows(
    video_ids: np.ndarray, game_idxs: np.ndarray, t_secs: np.ndarray,
    frame_idxs: np.ndarray, sides: np.ndarray, wons: np.ndarray,
) -> list[dict]:
    """npz のメタ配列群から meta 列のみの行リストを組み立てる。"""
    return [
        {
            "video_id": str(video_ids[i]), "game_idx": int(game_idxs[i]),
            "t_sec": float(t_secs[i]), "frame": int(frame_idxs[i]),
            "side": str(sides[i]), "won": float(wons[i]),
        }
        for i in range(len(video_ids))
    ]


def convert_one_npz(
    npz_path: Path, registry: dict[str, Callable[[Board], "iv.IndicatorV2Value"]],
) -> list[dict]:
    """1 npz ファイルを labeled_win 形式の行リストに変換する。

    npz は zlib 圧縮 (np.savez_compressed) のため、`d[key]` は毎回アーカイブ
    メンバを再展開するコストがある。ループの外で各配列を 1 回だけ取り出す
    (2026-08-12 実測: これを怠ると 1本 6693行で 99.8s、修正後は同じ内容が
    数秒で終わる。原情報を安く再変換できることが選択肢C成立の前提のため
    この最適化は必須)。

    b-2/全消しボーナスフラグの詳細はモジュール docstring 参照
    (「指標大整理」節)。1 npz = 1 動画・1P/2P混在という前提で処理する
    (実データで確認済み)。
    """
    d = np.load(str(npz_path), allow_pickle=True)
    grids = d["grids"]
    n = len(grids)
    scores = d["score"] if "score" in d.files else np.full(n, -1, dtype=np.int64)
    chain_mechanisms = (
        d["chain_mechanism"] if "chain_mechanism" in d.files else np.array([""] * n)
    )
    rows = _build_meta_rows(
        d["video_id"], d["game_idx"], d["t_sec"], d["frame_idx"], d["side"], d["won"],
    )
    _approx_tsumo(rows)
    for i in range(n):
        rows[i].update(_compute_row(grids[i], registry))
    _compute_all_clear_bonus_pending(
        rows, [int(s) for s in scores], [str(m) for m in chain_mechanisms],
    )
    diff_cols = _resolve_diff_target_columns(registry)
    carry_cols = _resolve_carry_target_columns(registry)
    rows = _attach_opponent_diff_columns(rows, diff_cols, tuple(carry_cols))
    rows = _add_pair_interaction_columns(rows)
    return rows


def convert_dir(
    npz_dir: Path, out_csv: Path, profile: str = "light", use_native: bool = True,
) -> tuple[int, float]:
    """npz_dir 内の全 npz を変換し out_csv に書き出す。

    Args:
        use_native: full profile の重い4列を Rust拡張 puyo_core 経由で計算
            するか (2026-08-13 追加、既定 True、後方互換の optional 引数)。
            `_resolve_indicator_registry` 参照。

    Returns:
        (書き出し行数, 所要秒数)。
    """
    registry = _resolve_indicator_registry(profile, use_native=use_native)
    t0 = time.time()
    all_rows: list[dict] = []
    npz_files = sorted(npz_dir.glob("*.npz"))
    for i, p in enumerate(npz_files):
        rows = convert_one_npz(p, registry)
        all_rows.extend(rows)
        print(f"[{i+1}/{len(npz_files)}] {p.name}: {len(rows)} rows "
              f"(累計 {len(all_rows)}, {time.time()-t0:.1f}s)")
    if not all_rows:
        print("[WARN] 変換対象行が0件でした")
        return 0, time.time() - t0
    fieldnames = _final_fieldnames(profile)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        # extrasaction="ignore": diff_ に完全置換した own 列 (b-2) が行dict
        # には残っているが CSV には出さない (diff計算に own が必要なため
        # 削除せず保持している、a-1/b-2 の設計判断)。
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    elapsed = time.time() - t0
    print(f"[done] {len(all_rows)} 行 -> {out_csv} ({elapsed:.1f}s, "
          f"{len(npz_files)}本, profile={profile})")
    return len(all_rows), elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", choices=VALID_PROFILES, default="light")
    ap.add_argument(
        "--no-native", dest="use_native", action="store_false", default=True,
        help=(
            "full profile の重い4列で Rust拡張 puyo_core への載せ替えを"
            "無効化し既存Python実装のみで計算する (2026-08-13追加、"
            "パリティ検証・デバッグ用)。"
        ),
    )
    a = ap.parse_args()
    convert_dir(a.npz_dir, a.out, profile=a.profile, use_native=a.use_native)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
