"""指標 v2 (第1バッチ) 算出パイプライン — 1 動画 → dataset CSV。

`docs/INDICATOR_V2_MEASUREMENT_SPEC_2026-06-17.md` のパイプライン仕様に従う。

処理概要:
    - RecognitionPipeline.load_default (visualize_recognition と同じ load_default 経路)
      で 1 動画を frame 単位処理。--no-per-video-hsv 相当で自動 HSV のみ動作。
    - 両者 STABLE snapshot で指標を算出し dataset 行 (CSV) を出力。
    - OjamaAccountingTracker を viz 統合と同様に駆動して net収支/forecast snapshot を得る。

前処理 (仕様書 4):
    - STABLE 時のみ算出 (両者個別。各 side が STABLE のフレームのみ行を出力)。
    - 試合境界 (score 大幅減少) で game_idx を進め、手数をリセット (pipeline 内部で自動)。
    - 連続フレーム間引き: 同一 STABLE 区間 (盤面が変わらない連続フレーム) は 1 回のみ出力。
    - 全消し直後フレーム除外: 盤面ぷよ数 0 (= 全消し / 試合開始直後) の STABLE は除外。

各行メタ: video_id / game_idx / t_sec / frame / 手数(tsumo) / side(1P/2P)。

間引き方式 (2種類、独立):
    - --sample-interval / --sample-interval-frames: 認識 (pipeline.update) 自体を
      間引く既存方式。2026-07-30 実測でこれを使うと状態機械が遷移を取りこぼし
      current_max_chain 等が壊れることが確定した (memory
      `project_frame_sampling_corrupts_boards_2026-07-30`)。
    - --indicator-interval-frames: 認識は全フレーム実行したまま、指標計算・行の
      書き出しだけを間引く新方式 (2026-07-30 追加)。おじゃま会計 drain・
      試合境界検知はエッジ検出型のため間引きの対象外 (常に毎フレーム実行)。

使い方 (短尺検証):
    python -m scripts.collect_indicators_v2 \
        --video data/frames/video_124_4min.mp4 \
        --out data/indicators_v2/video_124_4min.csv --max-sec 60
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# プロジェクトルートを import path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import Board  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import (  # noqa: E402
    OjamaAccountingTracker,
    OjamaAccountSnapshot,
)
from src.recognition_pipeline import RecognitionPipeline, SideResult  # noqa: E402
import src.indicators_v2 as iv  # noqa: E402

# 出力解像度 (認識は 1920x1080 前提)
TARGET_W: int = 1920
TARGET_H: int = 1080
DEFAULT_FPS: float = 30.0
# 試合境界検知: score がこの値以上減少したら新しい試合とみなす (会計と同基準)
SCORE_RESET_THRESHOLD: int = 500
# サンプル間引き幅の下限 (0 以下指定は 1 フレームおき = 全フレームに丸める)
MIN_SAMPLE_INTERVAL_FRAMES: int = 1


def _resolve_sample_interval_frames(
    sample_interval_sec: float,
    fps: float,
    sample_interval_frames: Optional[int] = None,
) -> int:
    """認識サンプル間隔を「実際に使うフレーム数」として一意に確定する。

    2026-07-28 user指示: 学習データ収集は fps に依存しない「Nフレームに1回」を
    正確に指定したい。sample_interval_frames が指定された場合はそれを最優先し、
    fps に関係なくそのフレーム数ごとに 1 回だけ認識する。
    省略時 (None) は従来通り sample_interval_sec (秒) を fps 換算する
    (完全後方互換、既存呼出元は一切変わらない)。

    Args:
        sample_interval_sec: 認識サンプル間隔秒 (0 = 全フレーム)。
        fps: 動画の fps。
        sample_interval_frames: 認識サンプル間隔フレーム数 (優先指定、省略可)。
            0 以下が渡された場合も不正値として扱い、下限 1 に丸める
            (秒指定の既存 max(1, ...) と挙動を揃える)。

    Returns:
        実際に使うフレーム間引き幅 (最小 MIN_SAMPLE_INTERVAL_FRAMES)。
    """
    if sample_interval_frames is not None:
        resolved = sample_interval_frames
    else:
        resolved = int(round(sample_interval_sec * fps))
    return max(MIN_SAMPLE_INTERVAL_FRAMES, resolved)


def _resolve_indicator_interval_frames(
    indicator_interval_frames: Optional[int] = None,
) -> int:
    """指標計算・行出力の間引き幅を確定する (2026-07-30 追加)。

    2026-07-30 実測 (memory `project_frame_sampling_corrupts_boards_2026-07-30`)
    で、認識そのもの (pipeline.update) を間引くと状態機械が前後フレームの
    差分で遷移判定するため取りこぼしが起き、current_max_chain が 37.4% の
    盤面でズレる (過小評価に偏る) ことが確定した。これを避けるため、認識は
    常に全フレームで実行したまま「指標計算 + 行の書き出し」だけを N フレーム
    ごとに間引きたい場合に本関数の戻り値を使う。

    既存の --sample-interval / --sample-interval-frames (認識自体の間引き、
    `_resolve_sample_interval_frames` が担当) とは完全に独立しており、
    本関数はそちらの挙動に一切影響しない。

    省略時 (None) は 1 (間引きなし = 毎フレーム指標計算、従来挙動) を返し、
    既存呼出元の挙動を一切変えない (後方互換)。

    Args:
        indicator_interval_frames: 指標計算を行うフレーム間隔 (省略可)。
            0 以下が渡された場合は不正値として下限 1 に丸める
            (_resolve_sample_interval_frames と同じ丸めルール)。

    Returns:
        実際に使う間引き幅 (最小 MIN_SAMPLE_INTERVAL_FRAMES)。
    """
    if indicator_interval_frames is None:
        return MIN_SAMPLE_INTERVAL_FRAMES
    return max(MIN_SAMPLE_INTERVAL_FRAMES, indicator_interval_frames)


# XVI 平均ツモ期待火力 (expected_fire_power) の収集 opt-in フラグ (2026-07-22)。
# user判断: fire_stability/expected_fire は「観測軸として残す」(4動画の狭い
# 標本でnull判定が出たが、低ティア・データ増で再評価の余地がある)。
# ただし expected_fire_power は重い (実測1.7〜3.5秒/盤面、
# scripts/_tmp_bench_expected_fire.py 参照) ため、将来のデータ拡充
# (Phase L、動画数を大幅に増やす) で常時収集すると ~1fps律速の収集
# パイプラインが破綻する。既定 OFF の opt-in にし、必要な時だけ True にする
# (fire_stability は軽い(near_future_fire_power と同水準)ため既定収集のまま)。
COLLECT_EXPECTED_FIRE: bool = False


# ============================
# CSV 列定義 (順序固定)
# ============================

# メタ列
META_COLUMNS: tuple[str, ...] = (
    "video_id", "game_idx", "t_sec", "frame", "tsumo", "side",
)
# 指標列 (順序保持。新指標は末尾に追加)
INDICATOR_COLUMNS: tuple[str, ...] = (
    # ① 進行度
    "tsumo_count_rate", "tsumo_count_raw",
    "board_puyo_total", "board_puyo_total_raw",
    "board_color_puyo_total", "board_color_puyo_total_raw",
    "margin_time_rate", "margin_time_rate_raw",
    # ② 占有・危険
    "max_column_height", "max_column_height_raw",
    "column_bumpiness", "column_bumpiness_raw",
    "death_margin", "death_margin_raw",
    "death_margin_neighbor", "death_margin_neighbor_raw",
    # ③ 火力・潜在
    "current_max_chain", "current_max_chain_raw",
    "immediate_fire_power", "immediate_fire_power_raw",
    "reach_fire_power", "reach_fire_power_raw", "reach_fire_power_source",
    "reach_fire_power_max_chain",
    "chain_efficiency", "chain_efficiency_raw",
    "min_puyos_to_ignite", "min_puyos_to_ignite_raw",
    "conn_pair_count", "conn_triple_count", "conn_max_group_size",
    "second_chain_potential", "second_chain_potential_raw",
    # ④ お邪魔
    "ojama_net_balance", "ojama_net_balance_raw",
    "ojama_forecast", "ojama_forecast_raw",
    "board_ojama_count", "board_ojama_count_raw",
    # ⑤ テンポ
    "chain_duration_sec", "chain_duration_source",
    # ⑥ 受け力
    "dig_resistance", "dig_resistance_raw",
    "absorption_capacity", "absorption_capacity_raw",
    # VIII 催促潰し度 (条件2「潰し」)
    "ojama_disruption", "ojama_disruption_raw",
    # IX 形・組み品質
    "main_linked_pair_count", "main_linked_pair_count_raw",
    "isolated_pair_count", "isolated_pair_count_raw",
    "main_linked_ratio", "main_linked_ratio_raw",
    # X 受けやすさ
    "ukeyasusa", "ukeyasusa_raw",
    # XII board sim 本命指標 (飽和連鎖量・発火点・副砲・同時消しリッチネス)
    "saturated_chain_count", "saturated_chain_count_raw",
    "ignition_point_count", "ignition_point_count_raw",
    "multi_color_ignition", "multi_color_ignition_raw",
    "sub_chain_count", "sub_chain_count_raw",
    "simultaneous_pop_richness", "simultaneous_pop_richness_raw",
    # XIV 近未来最大火力 (near_future_fire_power, K=1..5)
    # — INDICATOR_COLUMNS 末尾 (新指標は常に末尾追加で順序保持、2026-07-22 本番統合)
    "near_future_fire_k1", "near_future_fire_k1_raw",
    "near_future_fire_k2", "near_future_fire_k2_raw",
    "near_future_fire_k3", "near_future_fire_k3_raw",
    "near_future_fire_k4", "near_future_fire_k4_raw",
    "near_future_fire_k5", "near_future_fire_k5_raw",
    # XV 火力の受けの多さ (fire_stability, K=2,4,6)
    # — INDICATOR_COLUMNS 末尾 (新指標は常に末尾追加で順序保持、2026-07-22 本番統合)
    "fire_stability_k2", "fire_stability_k2_raw",
    "fire_stability_k4", "fire_stability_k4_raw",
    "fire_stability_k6", "fire_stability_k6_raw",
    # XVI 平均ツモ期待火力 (expected_fire_power, K=1..4)
    # — INDICATOR_COLUMNS 末尾 (新指標は常に末尾追加で順序保持、2026-07-22 本番統合)
    # ⚠️ K=3,4 追加時に列定義の更新漏れがあった (正直な記録): 実装は
    # EXPECTED_FIRE_K_LEVELS=(1,2,3,4) を計算するのに列は k1,k2 のみだったため、
    # COLLECT_EXPECTED_FIRE=True で収集すると csv.DictWriter が未定義列
    # (expected_fire_k3/k4) で ValueError を起こす潜在バグだった。ここで是正する。
    "expected_fire_k1", "expected_fire_k1_raw",
    "expected_fire_k2", "expected_fire_k2_raw",
    "expected_fire_k3", "expected_fire_k3_raw",
    "expected_fire_k4", "expected_fire_k4_raw",
)
ALL_COLUMNS: tuple[str, ...] = META_COLUMNS + INDICATOR_COLUMNS


@dataclass
class _BoardNpzAccumulator:
    """盤面グリッド npz ダンプ用の蓄積バッファ。

    CSV の各行と 1 対 1 対応するよう、rows リストと同期して追記する。
    """
    grids: list[np.ndarray] = field(default_factory=list)    # (13,6) uint8 のリスト
    video_ids: list[str] = field(default_factory=list)
    sides: list[str] = field(default_factory=list)           # "1P" / "2P"
    t_secs: list[float] = field(default_factory=list)
    game_idxs: list[int] = field(default_factory=list)
    frame_idxs: list[int] = field(default_factory=list)

    def append(
        self, grid: np.ndarray, video_id: str, side: str,
        t_sec: float, game_idx: int, frame_idx: int,
    ) -> None:
        """スナップショット 1 件を追加する。"""
        self.grids.append(grid.copy())
        self.video_ids.append(video_id)
        self.sides.append(side)
        self.t_secs.append(t_sec)
        self.game_idxs.append(game_idx)
        self.frame_idxs.append(frame_idx)

    def save(self, path: Path) -> None:
        """npz 形式で保存する。grids は (N,13,6) int8。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            grids=np.array(self.grids, dtype=np.int8),
            video_id=np.array(self.video_ids),
            side=np.array(self.sides),
            t_sec=np.array(self.t_secs, dtype=np.float32),
            game_idx=np.array(self.game_idxs, dtype=np.int32),
            frame_idx=np.array(self.frame_idxs, dtype=np.int32),
        )


# ============================
# 試合単位 active_colors トラッカー (2026-07-22, stateless修正)
# ============================
# 背景 (正直な記録): near_future_fire_power の初回統合では「試合ごとに
# 5色中1色除外」というドメイン事実を1盤面のみから近似していたが、
# win-AUC再検証でプロト (試合全体の色頻度で判定) との乖離が中盤で最大-0.11
# に達し、原因は約27%の盤面でこの近似がプロトと食い違うことと特定した。
# CLAUDE.md「観測指標は stateless、state 保持は外部 wrapper」に従い、
# iv.near_future_fire_power 自体は純関数のまま維持し、試合単位の色頻度計算
# (プロトの _compute_active_colors_by_game と同じ「頻度上位N色採用」ロジック、
# scripts/_tmp_ama_builder.py 参照) をこの収集パイプライン側の外部トラッカー
# _GameColorTracker に移す。
#
# ⚠️ プロトとの相違点 (正直な注記): プロトはオフラインの完成済み試合データ
# 全体 (未来のフレームも含む) から頻度を求めたが、本トラッカーは動画を
# 1パスで逐次処理するため「その時点までの累積頻度」しか使えない
# (因果的・先読み無し)。試合序盤は データ不足で GAME_COLOR_MIN_DISTINCT
# 未満のことがあり、その場合は近似段階を1段落として
# iv.near_future_fire_power 自身の盤面出現色フォールバックに委ねる
# (二重フォールバック構成、後方互換)。

# 出現頻度上位何色を active_colors として採用するか (プロトと同じ4)。
GAME_COLOR_KEEP_COUNT: int = 4
# 累積で観測できた色の種類数がこれ未満なら None (盤面出現色フォールバックへ)。
GAME_COLOR_MIN_DISTINCT: int = 4


@dataclass
class _GameColorTracker:
    """1 (video_id, side, game_idx) 分の色出現頻度を累積するトラッカー。

    near_future_fire_power は stateless 純関数のまま、本クラスが
    「試合単位 active_colors」という state を外部で保持する
    (CLAUDE.md 準拠の wrapper)。試合境界 (game_idx 変化) で reset() する。
    """
    counts: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        """試合境界で呼ぶ (新しい試合の頻度をゼロから積み直す)。"""
        self.counts = {}

    def update(self, board: Board) -> None:
        """1 盤面分のセル色を累積する (色ぷよ5色のみ対象、お邪魔等は除外)。"""
        for row in board._grid:
            for cell in row:
                c = int(cell)
                if c in iv.IGNITION_TRIAL_COLORS:
                    self.counts[c] = self.counts.get(c, 0) + 1

    def active_colors(self) -> "tuple[int, ...] | None":
        """出現頻度上位 GAME_COLOR_KEEP_COUNT 色を返す (データ不足なら None)。

        観測できた色の種類数が GAME_COLOR_MIN_DISTINCT 未満なら、判別材料
        不足として None を返し、呼び出し側フォールバック (盤面出現色) に委ねる。
        """
        observed = {c for c, n in self.counts.items() if n > 0}
        if len(observed) < GAME_COLOR_MIN_DISTINCT:
            return None
        ranked = sorted(
            iv.IGNITION_TRIAL_COLORS, key=lambda c: self.counts.get(c, 0), reverse=True,
        )
        return tuple(sorted(ranked[:GAME_COLOR_KEEP_COUNT]))


@dataclass
class _SideTracker:
    """1 side の前処理状態 (間引き・全消し検知用)。"""
    game_idx: int = 0
    prev_score: int | None = None
    last_emitted_grid: bytes | None = None  # 直前に出力した盤面 (間引き)
    prev_tsumo: int = 0  # tsumo_count 駆動 drain 用: 前回の手数
    color_tracker: _GameColorTracker = field(default_factory=_GameColorTracker)


def _compute_row(
    video_id: str,
    side_label: str,
    side: SideResult,
    board: Board,
    t_sec: float,
    frame_idx: int,
    tsumo: int,
    elapsed_sec: float,
    snap: OjamaAccountSnapshot,
    active_colors: "tuple[int, ...] | None" = None,
) -> dict[str, object]:
    """1 STABLE snapshot から指標を算出し CSV 行 dict を返す。

    Args:
        active_colors: 試合単位 active_colors (_GameColorTracker 由来)。
            None なら near_future_fire_power 側の盤面出現色フォールバックに
            委ねる (後方互換維持、optional 追加のみ)。
    """
    is_p1 = side_label == "1P"
    net = snap.net_balance_capped if is_p1 else -snap.net_balance_capped
    forecast = snap.forecast_p1 if is_p1 else snap.forecast_p2
    # ⑤ 連鎖所要時間: chain_event があれば観測、無ければ推定。
    dur, dur_src = _chain_duration(side)
    total_conn, _ = iv.connectivity_observation(board)
    row: dict[str, object] = {
        "video_id": video_id,
        "side": side_label,
        "t_sec": round(t_sec, 3),
        "frame": frame_idx,
        "tsumo": tsumo,
    }
    _fill_indicator_columns(
        row, board, tsumo, elapsed_sec, net, forecast, total_conn,
        side.next_pair, side.dnext_pair, active_colors,
    )
    row["chain_duration_sec"] = round(dur.raw, 3) if dur is not None else 0.0
    row["chain_duration_source"] = dur_src
    return row


def _fill_indicator_columns(
    row: dict[str, object],
    board: Board,
    tsumo: int,
    elapsed_sec: float,
    net: int,
    forecast: int,
    total_conn: iv.GroupObservation,
    next_pair: "tuple[int, int] | None" = None,
    dnext_pair: "tuple[int, int] | None" = None,
    active_colors: "tuple[int, ...] | None" = None,
) -> None:
    """指標値を row dict に書き込む (chain_duration を除く)。

    Args:
        active_colors: near_future_fire_power に渡す試合単位4色
            (_GameColorTracker 由来、None なら盤面出現色フォールバック)。
    """
    tc = iv.tsumo_count_rate(tsumo)
    bp = iv.board_puyo_total(board)
    bc = iv.board_color_puyo_total(board)
    mt = iv.margin_time_rate(elapsed_sec)
    mh = iv.max_column_height(board)
    bm = iv.column_bumpiness(board)
    dm = iv.death_margin(board)
    dn = iv.death_margin_neighbor(board)
    cm = iv.current_max_chain(board)
    ifp = iv.immediate_fire_power(board, elapsed_sec)
    rfp = iv.reach_fire_power(board, next_pair, dnext_pair, elapsed_sec)
    ce = iv.chain_efficiency(board, elapsed_sec)
    mi = iv.min_puyos_to_ignite(board)
    sc = iv.second_chain_potential(board)
    nb = iv.ojama_net_balance(net)
    fc = iv.ojama_forecast(forecast)
    bo = iv.board_ojama_count(board)
    dr = iv.dig_resistance(board)
    ab = iv.absorption_capacity(board)
    od = iv.ojama_disruption(board)
    mlp = iv.main_linked_pair_count(board)
    ip = iv.isolated_pair_count(board)
    mlr = iv.main_linked_ratio(board)
    # X 受けやすさ: dig_resistance を内包するため連鎖シミュが走る。
    #   STABLE snapshot の都度算出で問題なし (毎フレームではない)。
    uk = iv.ukeyasusa(board)
    # XII board sim 本命指標: micro-benchmark 済み (scripts/_tmp_bench_xii.py)。
    #   1 snapshot あたり平均 3ms 程度 (200ms 予算比で十分小さい) のため
    #   共有キャッシュ実装なしのシンプル呼び出しで問題ない
    #   (ChainSimulator 内蔵の simulate キャッシュが自然に重複を吸収する)。
    sat = iv.saturated_chain_count(board)
    igp = iv.ignition_point_count(board)
    mci = iv.multi_color_ignition(board)
    sub = iv.sub_chain_count(board)
    spr = iv.simultaneous_pop_richness(board)
    row.update({
        "tsumo_count_rate": tc.score, "tsumo_count_raw": tc.raw,
        "board_puyo_total": bp.score, "board_puyo_total_raw": bp.raw,
        "board_color_puyo_total": bc.score, "board_color_puyo_total_raw": bc.raw,
        "margin_time_rate": mt.score, "margin_time_rate_raw": mt.raw,
        "max_column_height": mh.score, "max_column_height_raw": mh.raw,
        "column_bumpiness": bm.score, "column_bumpiness_raw": bm.raw,
        "death_margin": dm.score, "death_margin_raw": dm.raw,
        "death_margin_neighbor": dn.score, "death_margin_neighbor_raw": dn.raw,
        "current_max_chain": cm.score, "current_max_chain_raw": cm.raw,
        "immediate_fire_power": ifp.score, "immediate_fire_power_raw": ifp.raw,
        "reach_fire_power": rfp.value.score, "reach_fire_power_raw": rfp.value.raw,
        "reach_fire_power_source": rfp.source,
        "reach_fire_power_max_chain": rfp.max_chain,
        "chain_efficiency": ce.score, "chain_efficiency_raw": ce.raw,
        "min_puyos_to_ignite": mi.score, "min_puyos_to_ignite_raw": mi.raw,
        "conn_pair_count": total_conn.pair_count,
        "conn_triple_count": total_conn.triple_count,
        "conn_max_group_size": total_conn.max_group_size,
        "second_chain_potential": sc.score, "second_chain_potential_raw": sc.raw,
        "ojama_net_balance": nb.score, "ojama_net_balance_raw": nb.raw,
        "ojama_forecast": fc.score, "ojama_forecast_raw": fc.raw,
        "board_ojama_count": bo.score, "board_ojama_count_raw": bo.raw,
        "dig_resistance": dr.score, "dig_resistance_raw": dr.raw,
        "absorption_capacity": ab.score, "absorption_capacity_raw": ab.raw,
        "ojama_disruption": od.score, "ojama_disruption_raw": od.raw,
        "main_linked_pair_count": mlp.score, "main_linked_pair_count_raw": mlp.raw,
        "isolated_pair_count": ip.score, "isolated_pair_count_raw": ip.raw,
        "main_linked_ratio": mlr.score, "main_linked_ratio_raw": mlr.raw,
        # X 受けやすさ
        "ukeyasusa": uk.score, "ukeyasusa_raw": uk.raw,
        # XII board sim 本命指標 (新指標末尾追加)
        "saturated_chain_count": sat.score, "saturated_chain_count_raw": sat.raw,
        "ignition_point_count": igp.score, "ignition_point_count_raw": igp.raw,
        "multi_color_ignition": mci.score, "multi_color_ignition_raw": mci.raw,
        "sub_chain_count": sub.score, "sub_chain_count_raw": sub.raw,
        "simultaneous_pop_richness": spr.score,
        "simultaneous_pop_richness_raw": spr.raw,
    })
    _fill_near_future_columns(row, board, next_pair, dnext_pair, elapsed_sec, active_colors)
    _fill_fire_stability_columns(row, board, next_pair, dnext_pair, active_colors)
    _fill_expected_fire_columns(row, board, elapsed_sec, active_colors)


def _fill_near_future_columns(
    row: dict[str, object],
    board: Board,
    next_pair: "tuple[int, int] | None",
    dnext_pair: "tuple[int, int] | None",
    elapsed_sec: float,
    active_colors: "tuple[int, ...] | None" = None,
) -> None:
    """XIV 近未来最大火力 (near_future_fire_k1..k5) を row dict に書き込む。

    既知ネクスト・ダブルネクスト (next_pair/dnext_pair、SideResult 由来) を
    使い、無ければ iv.near_future_fire_power 側で理想ツモにフォールバックする。
    1回のビームサーチで K=1..5 を同時取得する (~40ms/盤面、プロト実測値。
    XII board sim 本命指標と同水準の予算内)。

    active_colors (_GameColorTracker 由来の試合単位4色) を渡すことで、
    プロト検証相当の色制限精度を再現する (2026-07-22 stateless修正)。
    """
    nf = iv.near_future_fire_power(
        board, next_pair, dnext_pair, elapsed_sec, active_colors=active_colors,
    )
    for k in iv.NEAR_FUTURE_K_LEVELS:
        row[f"near_future_fire_k{k}"] = nf.values[k].score
        row[f"near_future_fire_k{k}_raw"] = nf.values[k].raw


def _fill_fire_stability_columns(
    row: dict[str, object],
    board: Board,
    next_pair: "tuple[int, int] | None",
    dnext_pair: "tuple[int, int] | None",
    active_colors: "tuple[int, ...] | None" = None,
) -> None:
    """XV 火力の受けの多さ (fire_stability_k2/4/6) を row dict に書き込む。

    near_future_fire_power と同じビーム machinery (_GameColorTracker 由来の
    active_colors・next_pair/dnext_pair) を流用する副産物として安価に計算する
    (2026-07-22 本番統合、user提案#30)。
    """
    fs = iv.fire_stability(board, next_pair, dnext_pair, active_colors=active_colors)
    for k in iv.FIRE_STABILITY_K_LEVELS:
        row[f"fire_stability_k{k}"] = fs.values[k].score
        row[f"fire_stability_k{k}_raw"] = fs.values[k].raw


def _fill_expected_fire_columns(
    row: dict[str, object],
    board: Board,
    elapsed_sec: float,
    active_colors: "tuple[int, ...] | None" = None,
    enabled: "bool | None" = None,
) -> None:
    """XVI 平均ツモ期待火力 (expected_fire_k1..k4) を row dict に書き込む。

    ⚠️ opt-in 設計 (2026-07-22、user判断): expected_fire_power は重い
    (実測1.7〜3.5秒/盤面、scripts/_tmp_bench_expected_fire.py 参照。
    near_future_fire_power の ~40-70ms/盤面 の20-40倍) ため、既定 OFF の
    opt-in にする。Phase L (動画数を大幅に増やすデータ拡充) で常時収集すると
    ~1fps律速の収集パイプラインが破綻するため。

    enabled=None (既定) のときはモジュール定数 COLLECT_EXPECTED_FIRE を都度
    参照する (呼び出し時点の値を動的に見る。テストで monkeypatch する場合も
    正しく反映されるよう、関数のデフォルト引数に直接束縛しない設計)。
    False (既定) のときは計算せず row に何も追加しない
    (CSV列は INDICATOR_COLUMNS 定義に残ったまま、csv.DictWriter の
    restval=既定'' により空欄で出力される = 列存在ガードと整合する後方互換)。
    有効にしたい場合は COLLECT_EXPECTED_FIRE=True に変更するか、本関数の
    enabled 引数を明示的に True で呼ぶ。

    fire_stability (near_future_fire_power と同水準の軽さ) は対象外
    (既定収集のまま、opt-inガード不要)。
    """
    if enabled is None:
        enabled = COLLECT_EXPECTED_FIRE
    if not enabled:
        return
    ef = iv.expected_fire_power(board, elapsed_sec=elapsed_sec, active_colors=active_colors)
    for k in iv.EXPECTED_FIRE_K_LEVELS:
        row[f"expected_fire_k{k}"] = ef.values[k].score
        row[f"expected_fire_k{k}_raw"] = ef.values[k].raw


def _chain_duration(side: SideResult) -> tuple[iv.IndicatorV2Value, str]:
    """連鎖所要時間を観測優先・推定フォールバックで返す。

    Returns:
        (IndicatorV2Value, source) where source ∈ {"observed", "estimated", "none"}。
    """
    ev = side.chain_event
    if ev is not None:
        observed = iv.chain_duration_observed(ev.trigger_sec, ev.end_sec)
        if observed is not None:
            return observed, "observed"
        return iv.chain_duration_estimated(ev.chain_count), "estimated"
    return iv.IndicatorV2Value(score=0.0, raw=0.0), "none"


def _update_game_idx(
    tracker: _SideTracker, score: int | None,
) -> None:
    """score 大幅減少で game_idx を進める (試合境界分割)。

    試合境界では _GameColorTracker (試合単位 active_colors 用の色頻度) も
    reset し、新しい試合の頻度をゼロから積み直す。
    """
    if score is not None and tracker.prev_score is not None:
        if tracker.prev_score - score >= SCORE_RESET_THRESHOLD:
            tracker.game_idx += 1
            tracker.color_tracker.reset()
    if score is not None:
        tracker.prev_score = score


def _should_emit(
    tracker: _SideTracker, side: SideResult, board: Board,
) -> bool:
    """この STABLE snapshot を出力すべきか (間引き + 全消し除外)。"""
    if side.state != BoardState.STABLE or board is None:
        return False
    # 全消し直後 / 試合開始直後 (盤面ぷよ 0) は除外
    if board.count_puyos() == 0:
        return False
    # 連続フレーム間引き: 直前に出力した盤面と同一なら skip
    grid_bytes = board._grid.tobytes()
    if grid_bytes == tracker.last_emitted_grid:
        return False
    return True


def _process_side(
    video_id: str,
    side_label: str,
    side: SideResult,
    tracker: _SideTracker,
    pipeline: RecognitionPipeline,
    tsumo_tracker: OjamaAccountingTracker,
    t_sec: float,
    frame_idx: int,
    snap: OjamaAccountSnapshot,
    rows: list[dict[str, object]],
    npz_acc: Optional["_BoardNpzAccumulator"] = None,
) -> None:
    """1 side を処理し、出力対象なら rows に行を追加する。

    ⚠️ 2026-07-30 変更: 試合境界 (game_idx) 検知 `_update_game_idx` はここでは
    呼ばない。score 減少検知はエッジ検出型 (前回値との差分判定) なので、
    指標計算間引き (--indicator-interval-frames) の影響を受けないよう
    呼出元 `collect()` のループで毎フレーム呼ぶ設計に変更した
    (呼出元が `_update_game_idx` を毎フレーム呼び終えている前提で本関数を呼ぶこと)。

    Args:
        npz_acc: 盤面グリッドダンプ用バッファ (省略時はダンプしない)。
    """
    board = side.confirmed_board
    if board is None or not _should_emit(tracker, side, board):
        return
    # 試合単位 active_colors 用の色頻度を累積 (この盤面分も含めて確定させてから使う)。
    tracker.color_tracker.update(board)
    active_colors = tracker.color_tracker.active_colors()
    elapsed = tsumo_tracker._elapsed(t_sec)  # 試合相対経過秒 (マージンタイム用)
    tsumo = pipeline.tsumo_count(side_label)
    row = _compute_row(
        video_id, side_label, side, board, t_sec, frame_idx,
        tsumo, elapsed, snap, active_colors,
    )
    row["game_idx"] = tracker.game_idx
    rows.append(row)
    # 盤面グリッドダンプ: CSV 行と 1 対 1 対応で追加
    if npz_acc is not None:
        npz_acc.append(
            board._grid, video_id, side_label,
            round(t_sec, 3), tracker.game_idx, frame_idx,
        )
    tracker.last_emitted_grid = board._grid.tobytes()


def collect(
    video_path: Path,
    out_path: Path,
    max_sec: float = 0.0,
    sample_interval_sec: float = 0.0,
    start_sec: float = 0.0,
    board_npz_path: Optional[Path] = None,
    sample_interval_frames: Optional[int] = None,
    indicator_interval_frames: Optional[int] = None,
) -> int:
    """1 動画を処理して指標 dataset CSV を出力する。

    Args:
        video_path: 入力動画パス。
        out_path: 出力 CSV パス。
        max_sec: 処理する最大秒数 (0 = 全長)。start_sec との組み合わせで
            start_sec 〜 start_sec+max_sec の区間を処理する。
        sample_interval_sec: 認識サンプル間隔秒 (0 = 全フレーム)。
        start_sec: 処理開始オフセット秒 (デフォルト 0)。0 より大きい場合は
            cap.set で該当フレームにシークしてから処理を開始する。
            状態機械は連続フレームが要るため、シーク直後の数秒は MENU/非STABLE
            として扱われ既存の warmup バッファで吸収される。
            start_sec=0 のときの挙動は従来と完全に同一 (後方互換)。
        board_npz_path: 盤面グリッド npz 出力パス (省略時は保存しない)。
            grids=(N,13,6) int8 + メタ配列を保存する。CSV 行と 1 対 1 対応。
        sample_interval_frames: 認識サンプル間隔フレーム数 (省略可)。
            指定すると fps に関係なくそのフレーム数ごとに 1 回認識し、
            sample_interval_sec より優先される (2026-07-28 追加)。
            省略時 (None) は sample_interval_sec の従来挙動を完全維持する
            (後方互換)。実際に使われた間引き幅は標準出力にログされる。
        indicator_interval_frames: 指標計算・行出力のみの間引き幅 (省略可、
            2026-07-30 追加)。認識 (pipeline.update) は常に
            sample_interval_frames/sample_interval_sec の間隔通り実行したまま、
            「指標計算 + 行の書き出し」だけをこの値の倍数フレームに絞る。
            2026-07-29 実測 (memory
            `project_frame_sampling_corrupts_boards_2026-07-30`) で、認識
            自体を間引くと状態機械が遷移を取りこぼし current_max_chain が
            37.4%の盤面でズレる (過小評価に偏る) ことが確定したため、認識と
            指標計算の間引きを分離する目的で追加した。
            省略時 (None) は 1 (間引きなし) となり、既存呼出元の挙動を
            完全に維持する (後方互換)。おじゃま会計 drain (tsumo_count 増分
            駆動) と試合境界 (game_idx) 検知はエッジ検出型のため、本引数の
            値に関わらず常に毎フレーム実行する (指標計算・行出力のみが対象)。

    Returns:
        出力した行数。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {video_path}", file=sys.stderr)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 開始フレーム計算 + シーク
    start_frame = int(start_sec * fps) if start_sec > 0.0 else 0
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    # 処理フレーム数 = max_sec 相当に限定 (0 = 残り全体)
    if max_sec > 0:
        end_frame = min(total_frames, start_frame + int(max_sec * fps))
    else:
        end_frame = total_frames
    n_frames_to_process = max(0, end_frame - start_frame)

    video_id = video_path.stem

    # visualize_recognition と同じ load_default 経路 (自動 HSV のみ = per-video inject なし)
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )
    _vid_match = __import__("re").search(r"(v\d+|video_\d+)", video_path.name)
    if _vid_match and hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(_vid_match.group(1))

    ojama_tracker = OjamaAccountingTracker()
    ojama_tracker.reset()
    prev_state_p1 = BoardState.MENU
    prev_state_p2 = BoardState.MENU
    tracker_p1 = _SideTracker()
    tracker_p2 = _SideTracker()
    rows: list[dict[str, object]] = []
    # 盤面グリッドダンプバッファ (--board-npz 指定時のみ有効)
    npz_acc: Optional[_BoardNpzAccumulator] = (
        _BoardNpzAccumulator() if board_npz_path is not None else None
    )
    effective_interval_frames = _resolve_sample_interval_frames(
        sample_interval_sec, fps, sample_interval_frames,
    )
    # 指標計算・行出力のみの間引き幅 (認識間引きとは独立、2026-07-30 追加)。
    effective_indicator_interval_frames = _resolve_indicator_interval_frames(
        indicator_interval_frames,
    )
    # 実際に使われた間引き幅を明示ログ (fps 違いによる意図しない間引きの
    # 見落としを後から気付けるようにするため、2026-07-28 追加)。
    print(
        f"[collect] sample_interval: {effective_interval_frames} frames "
        f"(fps={fps:.3f}, sample_interval_sec={sample_interval_sec}, "
        f"sample_interval_frames_arg={sample_interval_frames})"
    )
    print(
        f"[collect] indicator_interval: {effective_indicator_interval_frames} "
        f"frames (indicator_interval_frames_arg={indicator_interval_frames}; "
        "認識は上記 sample_interval で毎回実行、指標計算・行出力のみ本間隔で間引く)"
    )

    for local_i in range(n_frames_to_process):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        # fi はビデオ全体での絶対フレーム番号。t_sec は絶対時刻
        fi = start_frame + local_i
        t_sec = fi / fps
        if local_i % effective_interval_frames != 0:
            continue
        # --- 認識 (state machine): sample_interval に従い実行 (従来通り) ---
        result = pipeline.update(fi, t_sec, frame)
        # ============================================================
        # 毎フレーム必須処理 (エッジ検出・状態保持型):
        # indicator_interval_frames による間引きの対象外にする。
        # 2026-07-30 実測で「間引くと取りこぼす」ことが確定した処理群。
        # ============================================================
        # おじゃま会計 drain (tsumo_count 増分駆動): 手数の増分を1つずつ
        # 消費するため、間引くと着地イベントそのものを取りこぼす
        # (2026-07-29 実測、15フレーム間引きで手数が実質100%消失した教訓)。
        snap = _drive_ojama(
            ojama_tracker, result.p1, result.p2,
            prev_state_p1, prev_state_p2, t_sec,
            tracker_p1=tracker_p1,
            tracker_p2=tracker_p2,
            pipeline=pipeline,
        )
        # 試合境界 (game_idx) 検知: score 大幅減少という「前回値との差分」で
        # 判定するエッジ検出のため、指標間引きと無関係に毎フレーム進める。
        _update_game_idx(tracker_p1, result.p1.score)
        _update_game_idx(tracker_p2, result.p2.score)
        # 状態遷移比較 (_drive_ojama の on_state_transition 用) も毎フレーム更新。
        prev_state_p1 = result.p1.state
        prev_state_p2 = result.p2.state
        # ============================================================
        # 指標計算・行出力 (stateless): ここだけ indicator_interval_frames
        # で間引く。既存列は現在盤面のみの純関数のため、間引いても値は
        # 壊れず出力行数が減るだけ (2026-07-30 追加)。
        # ============================================================
        if fi % effective_indicator_interval_frames != 0:
            continue
        _process_side(
            video_id, "1P", result.p1, tracker_p1, pipeline,
            ojama_tracker, t_sec, fi, snap, rows, npz_acc,
        )
        _process_side(
            video_id, "2P", result.p2, tracker_p2, pipeline,
            ojama_tracker, t_sec, fi, snap, rows, npz_acc,
        )
    cap.release()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ALL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # 盤面グリッド npz を保存 (--board-npz 指定時のみ)
    if npz_acc is not None and board_npz_path is not None:
        npz_acc.save(board_npz_path)
        print(f"[collect] board npz -> {board_npz_path} : {len(npz_acc.grids)} grids")

    return len(rows)


def _drive_ojama(
    tracker: OjamaAccountingTracker,
    p1: SideResult,
    p2: SideResult,
    prev_p1: BoardState,
    prev_p2: BoardState,
    t_sec: float,
    tracker_p1: "_SideTracker | None" = None,
    tracker_p2: "_SideTracker | None" = None,
    pipeline: "RecognitionPipeline | None" = None,
) -> OjamaAccountSnapshot:
    """OjamaAccountingTracker を on_state_transition / on_tsumo_settled で駆動。

    drain トリガーは tsumo_count 増分駆動 (手数ベース) を優先する。
    pipeline / tracker_Xp が渡された場合は tsumo_count の増分 delta 回
    on_tsumo_settled を呼ぶ。渡されない場合は旧トリガー (TSUMO_FALL→STABLE)
    で動作し後方互換を維持する (内部テスト等の呼出元を壊さない)。

    Args:
        tracker: お邪魔会計追跡器。
        p1, p2: 各 side の認識結果。
        prev_p1, prev_p2: 前フレームの各 side の状態。
        t_sec: 現在時刻 (秒)。
        tracker_p1, tracker_p2: 手数 prev_tsumo を保持する _SideTracker。
            None の場合は旧 TSUMO_FALL→STABLE トリガーにフォールバック。
        pipeline: tsumo_count(side) を提供する RecognitionPipeline。
            None の場合は旧 TSUMO_FALL→STABLE トリガーにフォールバック。
    """
    tracker.on_state_transition("p1", prev_p1, p1.state, p1.score, t_sec)
    tracker.on_state_transition("p2", prev_p2, p2.state, p2.score, t_sec)

    if pipeline is not None and tracker_p1 is not None and tracker_p2 is not None:
        # tsumo_count 増分駆動: delta 回 on_tsumo_settled を呼ぶ
        _drain_by_tsumo_delta(tracker, pipeline, tracker_p1, "p1", "1P", t_sec)
        _drain_by_tsumo_delta(tracker, pipeline, tracker_p2, "p2", "2P", t_sec)
    else:
        # フォールバック: 旧 TSUMO_FALL→STABLE トリガー (後方互換)
        if prev_p1 == BoardState.TSUMO_FALL and p1.state == BoardState.STABLE:
            tracker.on_tsumo_settled("p1", t_sec)
        if prev_p2 == BoardState.TSUMO_FALL and p2.state == BoardState.STABLE:
            tracker.on_tsumo_settled("p2", t_sec)

    return tracker.get_snapshot(t_sec)


# tsumo_count 増分 drain に使うサイドラベル対応定数
_SIDE_LABEL_TO_OJAMA_KEY: dict[str, str] = {"1P": "p1", "2P": "p2"}


def _drain_by_tsumo_delta(
    tracker: OjamaAccountingTracker,
    pipeline: RecognitionPipeline,
    side_tracker: "_SideTracker",
    ojama_key: str,
    pipeline_key: str,
    t_sec: float,
) -> None:
    """tsumo_count の増分 delta 回 on_tsumo_settled を呼ぶ。

    Args:
        tracker: お邪魔会計追跡器。
        pipeline: tsumo_count(side) を提供する RecognitionPipeline。
        side_tracker: prev_tsumo を保持する _SideTracker。
        ojama_key: "p1" または "p2" (OjamaAccountingTracker への key)。
        pipeline_key: "1P" または "2P" (pipeline.tsumo_count への key)。
        t_sec: 現在時刻 (秒)。
    """
    curr_tsumo = pipeline.tsumo_count(pipeline_key)
    delta = curr_tsumo - side_tracker.prev_tsumo
    # 試合境界 (手数リセット) では delta < 0 になるため skip (会計は
    # on_state_transition の MENU/score減少検知で既にリセット済み)
    if delta > 0:
        for _ in range(delta):
            tracker.on_tsumo_settled(ojama_key, t_sec)
    side_tracker.prev_tsumo = curr_tsumo


def main() -> int:
    parser = argparse.ArgumentParser(description="指標 v2 dataset 収集")
    parser.add_argument("--video", type=Path, required=True, help="入力動画")
    parser.add_argument("--out", type=Path, required=True, help="出力 CSV パス")
    parser.add_argument(
        "--max-sec", type=float, default=0.0,
        help="処理する最大秒数 (0 = 全長)。--start-sec と組み合わせて区間指定可能",
    )
    parser.add_argument(
        "--start-sec", type=float, default=0.0,
        help="処理開始オフセット秒 (デフォルト 0)。"
             "--start-sec S --max-sec D で S秒〜S+D秒を処理する。"
             "シーク後は状態機械 warmup のため序盤数秒は non-STABLE として扱われる",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=0.0,
        help="認識サンプル間隔秒 (0 = 全フレーム)。"
             "--sample-interval-frames 指定時はそちらが優先される",
    )
    parser.add_argument(
        "--sample-interval-frames", type=int, default=None,
        dest="sample_interval_frames",
        help=(
            "認識サンプル間隔フレーム数 (省略可、整数)。"
            "fps に関係なくこのフレーム数ごとに 1 回認識する。"
            "--sample-interval (秒) より優先される。"
            "例: 8フレームに1回 (60fps 想定) なら --sample-interval-frames 8"
        ),
    )
    parser.add_argument(
        "--board-npz", type=Path, default=None,
        help=(
            "盤面グリッド npz 出力パス。指定時のみ保存。"
            "grids=(N,13,6) int8 + video_id/side/t_sec/game_idx/frame_idx を含む。"
            "CSV 行と 1 対 1 対応 (同順)。"
        ),
    )
    parser.add_argument(
        "--indicator-interval-frames", type=int, default=None,
        dest="indicator_interval_frames",
        help=(
            "指標計算・行出力のみの間引き幅 (省略可、整数、2026-07-30 追加)。"
            "認識 (pipeline.update) は --sample-interval / --sample-interval-frames "
            "の間隔通り実行したまま、指標計算・行の書き出しだけをこのフレーム数"
            "ごとに絞る。省略時は 1 (間引きなし=毎フレーム、従来挙動)。"
            "認識自体を間引くと状態機械の遷移取りこぼしで current_max_chain 等が"
            "壊れることが実測済みのため、認識は全フレームのままにしたい場合に使う"
            "(例: --indicator-interval-frames 6 --sample-interval-frames 未指定)"
        ),
    )
    args = parser.parse_args()
    n = collect(
        args.video, args.out,
        max_sec=args.max_sec,
        sample_interval_sec=args.sample_interval,
        start_sec=args.start_sec,
        board_npz_path=args.board_npz,
        sample_interval_frames=args.sample_interval_frames,
        indicator_interval_frames=args.indicator_interval_frames,
    )
    print(f"[collect] {args.video.name} -> {args.out} : {n} rows")
    if args.start_sec > 0.0:
        print(f"[collect] start_sec={args.start_sec:.1f} max_sec={args.max_sec:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
