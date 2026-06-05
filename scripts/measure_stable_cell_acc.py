"""STABLE 確定盤面 cell-level 正解率測定スクリプト (Phase I 精度評価)。

3 者独立判定で STABLE 確定盤面の cell 単位正解率を測定する。
3 者:
  1. raw_cnn  : CNN+HSV hybrid の ImageReader 直出力 (物理推論 post-process 前)
  2. raw_hsv  : HSV-only pipeline の ImageReader 直出力 (CNN 完全無効化)
  3. confirmed: CNN+HSV hybrid + 全物理推論 post-process 後の確定盤面

合意 = 3 者のうち少なくとも 2 者が一致したセルを正解ラベル確定とみなす。
分裂 cell は JSON に出力し、人手チェック対象として flag する。

使い方:
    python scripts/measure_stable_cell_acc.py         --videos v89,v97,v29         --holdout v89,v97         --video-dir data/holdout_videos         --output data/verify/stable_cell_acc/2026-05-26.json

判定基準:
    PASS = holdout 全マス平均 >= 99.5% かつ 色別最低 >= 98%
    FAIL = 上記未達 (内訳出力)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import multiprocessing as _mp
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# プロジェクトルートを sys.path に追加 (script 直接実行時)
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
)
from src.board_state_machine import BoardState
from src.recognition_evaluator import compute_avg_puyo_count
from src.recognition_pipeline import RecognitionPipeline
# ============================
# 定数定義
# ============================

# 評価対象色 (UNKNOWN は正解ラベル対象外)
EVAL_COLORS: tuple[int, ...] = (
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)

COLOR_NAMES: dict[int, str] = {
    COLOR_EMPTY:   "empty",
    COLOR_RED:     "red",
    COLOR_BLUE:    "blue",
    COLOR_GREEN:   "green",
    COLOR_YELLOW:  "yellow",
    COLOR_PURPLE:  "purple",
    COLOR_OJAMA:   "ojama",
    COLOR_UNKNOWN: "unknown",
}

# 精度基準値
PASS_OVERALL_THRESHOLD: float = 0.995
PASS_PER_COLOR_THRESHOLD: float = 0.98

# 認識処理間隔 (秒)
DEFAULT_SAMPLE_INTERVAL_SEC: float = 1.0 / 30.0

# 1 動画あたり最大処理フレーム数 (0 = 制限なし)
DEFAULT_MAX_FRAMES: int = 0

# UNKNOWN 含む cell をスキップするか
SKIP_UNKNOWN_CELLS: bool = True

# 不一致 cell 出力上限
DISAGREEMENT_OUTPUT_LIMIT: int = 500
DISAGREEMENT_COLLECT_LIMIT: int = 2000

# I1 メトリクス: per_col_unknown_rate 閾値
# STABLE confirmed_board で col 別 COLOR_UNKNOWN 比率が高い = 認識崩壊 (v89 27-30s 相当)
# mismatch/replace が fail-silent でも col 別 UNKNOWN 率は明示的に上昇する
PER_COL_UNKNOWN_WARNING: float = 0.15  # 15% 超 = WARNING
PER_COL_UNKNOWN_CRITICAL: float = 0.30  # 30% 超 = CRITICAL

# I1 メトリクス: non_stable_consecutive_frames 閾値
# state が stable 以外の連続サンプリングフレーム数
# 試合序盤 15 秒バッファ後から計測する (= 初期化猶予)
NON_STABLE_CRITICAL_FRAMES: int = 180  # 180 sample frame = ~3 秒 @ 60fps
NON_STABLE_WARMUP_SEC: float = 15.0  # 試合開始から 15 秒は計測除外

# 改修3: non_stable chain中除外
# CHAIN 中の non-stable は大連鎖の正常動作であるため連続カウント対象外にする。
# MENU / TSUMO_FALL / EFFECT 等 CHAIN 以外の連続 non-stable のみ検知する。
NON_STABLE_CHAIN_EXCLUDE: bool = True  # True = CHAIN state を連続 non-stable 除外

# I1 メトリクス: per_col_empty_rate_by_game_phase 閾値
# 中盤 (= 30 秒以降) で特定 col が全 STABLE フレーム中 100% EMPTY なら CRITICAL
# v40_match01「1P col=1 全 EMPTY 誤判定」 を捕捉する
MIDGAME_START_SEC: float = 30.0  # 中盤開始時刻 (秒)
MIDGAME_COL_EMPTY_CRITICAL: float = 0.99  # 99% 以上 EMPTY = CRITICAL
MIDGAME_COL_MIN_FRAMES: int = 30  # 最低 30 STABLE frame が必要
# 改修2: avg_puyo per-side + CRITICAL閾値
# STABLE フレームの 1P / 2P 各サイド平均ぷよ数が本定数未満の場合は列崩壊疑いとして
# failures に WARNING を追加する (FAIL 化する)。
# 実測最小値 18 以上を前提に保守的閾値を設定する (誤発報防止)。
AVG_PUYO_COUNT_CRITICAL: float = 5.0  # 5.0 未満 = 列崩壊疑い WARNING

# クリップ末尾 N 秒を中盤評価から除外する (短クリップ偽陽性対策)
# 問題: v70_match01 (30.5s) は末尾 0.5 秒だけが中盤判定区間に入り、
# 連鎖後の正当な列空きを「中盤列崩壊」と誤 FAIL していた。
# 対策: クリップ末尾 MIDGAME_TRAIL_EXCLUDE_SEC を中盤評価から除外する。
# 真の中盤列崩壊 (v40_match01 等の 60s クリップ中盤) は影響を受けない。
# 後方互換のため既存定数は変更せず新定数として追加する。
MIDGAME_TRAIL_EXCLUDE_SEC: float = 10.0  # 末尾 10 秒は中盤評価から除外

# HSV DB ルート
_HSV_DB_ROOT = Path("data/per_video_hsv_ranges")
_HSV_MERGED_DEFAULT = _HSV_DB_ROOT / "_merged_default.json"

# 動画検索ディレクトリ (デフォルト)
_DEFAULT_VIDEO_DIRS: tuple[Path, ...] = (
    Path("data/evaluation_videos"),
    Path("data/holdout_videos"),
)

# ============================
# 後処理破壊検知 (postprocess_corruption) 定数
# ============================
# corruption ログ上限 (メモリ節約のため)
CORRUPTION_LOG_LIMIT: int = 100
# 0.1% 超で FAIL
POSTPROCESS_CORRUPTION_REJECT_RATE: float = 0.001
# 0.05% 超で WARNING
POSTPROCESS_CORRUPTION_WARNING_RATE: float = 0.0005
# 片側に 50% 以上集中したら side_bias あり
POSTPROCESS_SIDE_BIAS_THRESHOLD: float = 0.50
# side_bias 判定に必要な最低 corruption セル数
POSTPROCESS_SIDE_BIAS_MIN_CELLS: int = 3
# サブカテゴリ: empty_to_color の WARNING 閾値 (FAIL にはしない、情報提示のみ)
# raw_cnn==raw_hsv==EMPTY なのに confirmed が色になっているケース (空→色FP)
CORRUPTION_EMPTY_TO_COLOR_WARNING_RATE: float = 0.001  # 0.1% 超で WARNING

# ============================
# 持続 corruption 集計 (postprocess_corruption_persistent) 定数
# ============================
# N サンプルフレーム以上連続した corruption run のみを「持続」として計上する。
# 診断結果: 1-2fr が 67.5% (1fr 点滅が 49.7%) のため 3fr 以上で点滅を除外できる。
# --corruption-persist-frames N CLI 引数で上書き可能 (後方互換: default=3)。
CORRUPTION_PERSIST_MIN_FRAMES: int = 3
# ============================
# データクラス
# ============================

@dataclass
class VideoStats:
    """1 動画の集計結果。

    agreed_cells の判定方式:
      3 者独立モード (use_three_way=True): raw_cnn / raw_hsv / confirmed の
      うち 2 者以上が一致したセルを合意とみなす。
      2 者モード (後方互換): raw_cnn == raw_hsv の一致のみ (旧挙動)。
    """

    video_id: str
    is_holdout: bool
    total_cells: int = 0
    agreed_cells: int = 0
    correct_by_color: dict = field(default_factory=lambda: defaultdict(int))
    total_by_color: dict = field(default_factory=lambda: defaultdict(int))
    correct_by_row: dict = field(default_factory=lambda: defaultdict(int))
    total_by_row: dict = field(default_factory=lambda: defaultdict(int))
    disagreement_count: int = 0
    stable_frame_count: int = 0
    # 3 者独立メトリクス (追加フィールド、後方互換のため default=0)
    # physics_fix_count: raw_cnn != raw_hsv だが confirmed が正解ラベルと一致したセル数
    physics_fix_count: int = 0
    # all_three_agree_count: 3 者全員一致セル数
    all_three_agree_count: int = 0
    # ------------------------------------------------
    # I1 追加メトリクス (後方互換のため全て default 付き)
    # ------------------------------------------------
    # per_col_unknown_cells[col]: confirmed_board で COLOR_UNKNOWN だった cell 数 (col 別)
    per_col_unknown_cells: dict = field(default_factory=lambda: defaultdict(int))
    # per_col_stable_cells[col]: STABLE フレームで確認した cell 数 (col 別、分母)
    per_col_stable_cells: dict = field(default_factory=lambda: defaultdict(int))
    # non_stable_max_consecutive: warmup 後の最長連続 non-stable サンプルフレーム数
    non_stable_max_consecutive: int = 0
    # per_col_midgame_empty_cells[col]: 中盤 (>= MIDGAME_START_SEC) で EMPTY だった cell 数
    per_col_midgame_empty_cells: dict = field(default_factory=lambda: defaultdict(int))
    # per_col_midgame_cells[col]: 中盤 STABLE フレームで確認した cell 数 (col 別、分母)
    per_col_midgame_cells: dict = field(default_factory=lambda: defaultdict(int))
    # クリップ総時間 (秒)。0.0 = 不明 (末尾除外を適用しない)
    # 後方互換のため default=0.0。_open_capture で設定される。
    clip_duration_sec: float = 0.0
    # _non_stable_current_by_side: side 別 non-stable 連続カウンタ (内部、init 時 {}).
    # non_stable_max_consecutive の更新用。直接参照禁止。
    _non_stable_current_by_side: dict = field(default_factory=dict, repr=False, compare=False)
    # C1: avg_puyo_count メトリクス (後方互換のため default 付き)
    # STABLE フレームの 1P+2P 合算ぷよ数合計と frame 数
    _puyo_count_sum: int = 0
    _puyo_count_n_stable: int = 0
    # 改修2: per-side 別 puyo count (後方互換のため default 付き)
    # side → (count_sum, n_stable_frames) を保持する
    _puyo_count_sum_by_side: dict = field(default_factory=lambda: defaultdict(int))
    _puyo_count_n_stable_by_side: dict = field(default_factory=lambda: defaultdict(int))
    # 改修1: confirmed_majority_agree (後方互換のため default 付き)
    # STABLE セルで confirmed_val == majority_label(多数決) が一致した数と総 cell 数
    _confirmed_agree_cells: int = 0
    _confirmed_total_cells: int = 0
    # per_color 別 confirmed agree (後方互換のため default 付き)
    _confirmed_agree_by_color: dict = field(default_factory=lambda: defaultdict(int))
    _confirmed_total_by_color: dict = field(default_factory=lambda: defaultdict(int))
    # 並列ワーカが収集した不一致 cell リスト (後方互換のため default=[])
    # 逐次モードでは空リスト。並列モードではワーカ内で収集した値が入り、
    # 親プロセスで各動画分を統合する。pickle 可能な plain dict リスト。
    _local_disagreements: list = field(default_factory=list, repr=False, compare=False)
    # ------------------------------------------------
    # 後処理破壊検知 (postprocess_corruption) フィールド (後方互換 default 付き)
    # raw_cnn==raw_hsv で一致しているのに confirmed が異なるセルを検知する。
    # constraint_fill が正解を上書き破壊しているケースを定量化する。
    # ------------------------------------------------
    # 破壊 cell 総数
    postprocess_corruption_count: int = 0
    # side → count
    postprocess_corruption_by_side: dict = field(default_factory=lambda: defaultdict(int))
    # (from_color, to_color) → count
    postprocess_corruption_color_pairs: dict = field(default_factory=lambda: defaultdict(int))
    # 座標ログ (repr=False でデバッグ出力肥大化を防ぐ)
    postprocess_corruption_log: list = field(default_factory=list, repr=False)
    # サブカテゴリ別 count (後方互換のため全て default=0)
    # empty_to_color: raw_cnn==raw_hsv==EMPTY なのに confirmed が色 (空→色FP = 主課題)
    corruption_empty_to_color_count: int = 0
    # color_to_color: raw_cnn==raw_hsv==色A なのに confirmed が色B (色フリーズ系)
    corruption_color_to_color_count: int = 0
    # color_to_empty: raw_cnn==raw_hsv==色 なのに confirmed が EMPTY (色→空 消失)
    corruption_color_to_empty_count: int = 0

    # ------------------------------------------------
    # 持続 corruption 集計 (postprocess_corruption_persistent) フィールド
    # N サンプルフレーム以上連続した corruption run のみを別計上する。
    # 1fr 点滅 (49.7%) を除外し、真の持続誤認を可視化する。
    # 後方互換のため全て default 付き。
    # ------------------------------------------------
    # 持続 corruption セル-フレーム 総数 (run 長 >= CORRUPTION_PERSIST_MIN_FRAMES の frame)
    postprocess_corruption_persistent_count: int = 0
    # サブカテゴリ別 持続 count (後方互換のため全て default=0)
    corruption_persistent_empty_to_color_count: int = 0
    corruption_persistent_color_to_color_count: int = 0
    corruption_persistent_color_to_empty_count: int = 0
    # 内部: per-cell の進行中 corruption run を追跡する dict
    # キー: (side, row, col)  値: {"raw_val": int, "confirmed_val": int, "length": int, "subcategory": str}
    # 直接参照禁止 (repr=False でデバッグ出力肥大化防止)。
    _persist_run_state: dict = field(
        default_factory=dict, repr=False, compare=False,
    )


# ============================
# ユーティリティ
# ============================

def _resolve_video_path(video_id: str, video_dir: Optional[Path]) -> Optional[Path]:
    """動画 ID からファイルパスを解決する。

    video_dir 以下のサブディレクトリも再帰検索する (rglob)。
    これにより data/match_clips/v29/v29_match01.mp4 形式にも対応。
    """
    search_dirs: list[Path] = (
        [video_dir] if video_dir is not None else list(_DEFAULT_VIDEO_DIRS)
    )
    for d in search_dirs:
        if not d.exists():
            continue
        # rglob で再帰検索 (サブディレクトリ対応)
        for f in sorted(d.rglob("*")):
            if f.suffix in (".mp4", ".mkv", ".avi", ".mov") and video_id in f.name:
                return f
    return None


def _resolve_hsv_path(video_id: str) -> Path:
    """動画 ID から per-video HSV JSON を解決する。

    clip ID (例: v29_match01) の場合、先頭の v<NN> 部分を抽出して
    v29.json を探す。完全一致ファイルが優先。
    """
    # 完全一致を優先
    candidate = _HSV_DB_ROOT / f"{video_id}.json"
    if candidate.exists():
        return candidate
    # clip ID (v29_match01 など) から動画 ID を抽出して再試行
    import re
    m = re.match(r"^(v\d+)", video_id)
    if m:
        base_candidate = _HSV_DB_ROOT / f"{m.group(1)}.json"
        if base_candidate.exists():
            return base_candidate
    return _HSV_MERGED_DEFAULT


def _inject_hsv(pipe: RecognitionPipeline, hsv_path: Path) -> None:
    """pipeline に per-video HSV ranges を注入する。"""
    if not hsv_path.exists():
        return
    try:
        with hsv_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        ranges = {
            int(k): tuple(int(x) for x in v)
            for k, v in state.get("per_video_ranges", {}).items()
        }
        from src.hybrid_classifier import HybridClassifier
        hc = pipe._reader._classifier
        if (
            isinstance(hc, HybridClassifier)
            and hasattr(hc._hsv, "set_color_ranges_from_simple")
            and ranges
        ):
            hc._hsv.set_color_ranges_from_simple(ranges)
            if pipe._online_hsv is not None:
                pipe._online_hsv_injected = True
    except Exception as e:
        print(f"[measure] HSV inject 失敗 ({hsv_path}): {e}", file=sys.stderr)
def _make_pipeline_cnn(
    video_id: str,
    enable_constraint_fill: bool = True,
    enable_t2_highconf_yield: bool = False,
    enable_infer_empty_guard: bool = False,
    enable_game_event_chain_exit: bool = False,
    enable_landing_color_fix: bool = False,
    enable_chain_min_display: bool = False,
    enable_hsv_classify_fallback: bool = False,
    enable_landing_observed_color: bool = False,
    enable_red_hue_wrap_fix: bool = False,
    enable_specular_robust_saturation: bool = False,
    enable_stable_recovery_gate: bool = False,
    enable_ojama_visual_detection: bool = False,
    enable_ojama_visual_chain_exit: bool = False,
    enable_ojama_infer_guard: bool = False,
    enable_ojama_settle_detection: bool = False,
    enable_ojama_tier1_warmup: bool = False,
    enable_chain_score_early_fire: bool = False,
    enable_chain_exit_warmup: bool = False,
    enable_chain_formula_detection: bool = False,
    enable_hsv_deferred_consensus: bool = False,
    enable_ojama_warning_glow_guard: bool = False,
    enable_chain_max_hold_override: bool = False,
    # 案X*(A)(B)+warmup (2026-06-05): NextSlide signal による CHAIN 即終了。
    # default False = 従来挙動完全維持 (backwards compat)。
    enable_chain_exit_next_signal: bool = False,
    # feat/gravity-settle-2026-06-05: 連鎖終了直後 GRAVITY_SETTLE 状態を有効化。
    # default False = 従来挙動完全維持 (backwards compat)。
    enable_gravity_settle_state: bool = False,
    disable_per_video_hsv: bool = False,
) -> RecognitionPipeline:
    """CNN + HSV ハイブリッド pipeline を構築する。

    Args:
        video_id: 動画 ID (per-video HSV 解決に使用)。
        enable_constraint_fill: False にすると NEXT 累積制約による
            色 count 補正 (_apply_next_count_constraint) を無効化する。
            confirmed 経路 (CNN+物理推論 post-process) に効く。
            backwards compat: デフォルト True = 従来挙動。
        enable_t2_highconf_yield: True にすると T2 の prev_stable 上書きを
            CNN 支持セルでスキップする (infer_placement 誤推論 + T2 フリーズ修正)。
            backwards compat: デフォルト False = 従来挙動。
        enable_infer_empty_guard: True にすると infer_placement の空セル
            hallucination ガードを有効化する。
            backwards compat: デフォルト False = 従来挙動。
        enable_game_event_chain_exit: True にすると game-event ベース連鎖終了を
            有効化する (次ツモ変化 / お邪魔降下で CHAIN 終了)。
            backwards compat: デフォルト False = 従来挙動。
        enable_landing_color_fix: True にすると TSUMO_FALL→STABLE 着地時の
            falling_pair を _landing_pending (消費済みツモ色) に切り替える。
            backwards compat: デフォルト False = 従来挙動。
        enable_chain_min_display: True にすると X1/X4 短連鎖ちらつき対策を有効化。
            CHAIN 最小表示時間 (CHAIN_MIN_DISPLAY_SEC) + 短連鎖 game-event exit 抑止。
            backwards compat: デフォルト False = 従来挙動。
        enable_hsv_classify_fallback: True にすると HSV 分類 fallback を有効化。
            _classify_next_pair_by_hsv の 2 択強制確定を回避し、
            黄→赤誤分類 (~900 件) 発火点を修正する。
            backwards compat: デフォルト False = 従来挙動。
        enable_landing_observed_color: True にすると着地セルの CNN==HSV 一致色補正を有効化。
            falling_pair タイミングずれで生じる着地色誤りを上流で断つ (真因 A 対処)。
            backwards compat: デフォルト False = 従来挙動。
        enable_red_hue_wrap_fix: True にすると赤色相折り返し補正を有効化。
            HSV median で赤 2 峰 (H=0-4 と H=166-179) を 1 峰に collapse する。
            backwards compat: デフォルト False = 従来挙動。
        enable_specular_robust_saturation: True にすると光沢ハイライト除外彩度計算を有効化。
            白ハイライト画素を彩度 median 計算から除外して EMPTY 誤判定を防ぐ (案D)。
            backwards compat: デフォルト False = 従来挙動。
        enable_ojama_visual_detection: True にするとおじゃま視覚検知 (親フラグ) を有効化。
            backwards compat: デフォルト False = 従来挙動。
        enable_ojama_visual_chain_exit: True にすると CHAIN→STABLE 復帰をお邪魔視覚に委譲。
            backwards compat: デフォルト False = 従来挙動。
        enable_ojama_infer_guard: True にすると OJAMA_FALL 直後の infer_placement を抑止。
            backwards compat: デフォルト False = 従来挙動。
        enable_ojama_settle_detection: True にすると OJAMA_FALL 中 count 不変で STABLE 復帰。
            backwards compat: デフォルト False = 従来挙動。
        enable_ojama_tier1_warmup: True にすると OJAMA 専用 tier1 warmup を有効化。
            backwards compat: デフォルト False = 従来挙動。
        enable_chain_formula_detection: True にすると掛け算式検知 CHAIN 早期発火を有効化。
            backwards compat: デフォルト False = 従来挙動。
        disable_per_video_hsv: True にすると per-video 手調整 HSV inject をスキップする。
            OnlineHsvCalibrator は load_default で生成されるため自動 HSV 学習は継続する。
            「自動 HSV + 既定 merged レンジのみ」の汎用精度測定用。
            backwards compat: デフォルト False = 従来挙動と完全一致。
    """
    pipe = RecognitionPipeline.load_default(
        force_in_match=True,
        enable_constraint_fill=enable_constraint_fill,
        enable_t2_highconf_yield=enable_t2_highconf_yield,
        enable_infer_empty_guard=enable_infer_empty_guard,
        enable_game_event_chain_exit=enable_game_event_chain_exit,
        enable_landing_color_fix=enable_landing_color_fix,
        enable_chain_min_display=enable_chain_min_display,
        enable_hsv_classify_fallback=enable_hsv_classify_fallback,
        enable_landing_observed_color=enable_landing_observed_color,
        enable_red_hue_wrap_fix=enable_red_hue_wrap_fix,
        enable_specular_robust_saturation=enable_specular_robust_saturation,
        enable_stable_recovery_gate=enable_stable_recovery_gate,
        enable_ojama_visual_detection=enable_ojama_visual_detection,
        enable_ojama_visual_chain_exit=enable_ojama_visual_chain_exit,
        enable_ojama_infer_guard=enable_ojama_infer_guard,
        enable_ojama_settle_detection=enable_ojama_settle_detection,
        enable_ojama_tier1_warmup=enable_ojama_tier1_warmup,
        enable_chain_score_early_fire=enable_chain_score_early_fire,
        enable_chain_exit_warmup=enable_chain_exit_warmup,
        enable_chain_formula_detection=enable_chain_formula_detection,
        enable_hsv_deferred_consensus=enable_hsv_deferred_consensus,
        enable_ojama_warning_glow_guard=enable_ojama_warning_glow_guard,
        enable_chain_max_hold_override=enable_chain_max_hold_override,
        enable_chain_exit_next_signal=enable_chain_exit_next_signal,
        enable_gravity_settle_state=enable_gravity_settle_state,
    )
    # per-video 手調整 HSV inject: disable_per_video_hsv=True の場合はスキップする。
    # OnlineHsvCalibrator は load_default で生成済みのため自動 HSV 学習は継続する。
    if not disable_per_video_hsv:
        _inject_hsv(pipe, _resolve_hsv_path(video_id))
    return pipe


def _make_pipeline_hsv_only(
    video_id: str,
    disable_per_video_hsv: bool = False,
) -> RecognitionPipeline:
    """HSV-only pipeline を構築する。

    cnn_override_prob=2.0 で CNN 採用閾値を 1.0 超にし、
    HybridClassifier が常に HSV 側を採用するよう強制する。

    Args:
        video_id: 動画 ID (per-video HSV 解決に使用)。
        disable_per_video_hsv: True のとき per-video 手調整 HSV inject をスキップする。
            OnlineHsvCalibrator は load_default で生成済みのため自動 HSV 学習は継続する。
            「自動 HSV + 既定 merged レンジのみ」の全 3 軸整合測定用。
            backwards compat: デフォルト False = 従来挙動と完全一致。

    Note:
        disable_per_video_hsv=True 時は raw_cnn / raw_hsv / confirmed の全 3 軸が
        手調整 HSV なしの自動 HSV のみで動作する。これにより 3 者合意 metric が
        「完全自動条件での内部整合率」を測定する。
        ただし CNN==HSV 両方が同じ誤りに合意した場合 corruption に出ず acc が
        見かけ上保たれる fail-silent リスクが上がるため、viz 目視で補完すること。
    """
    pipe = RecognitionPipeline.load_default(
        cnn_override_prob=2.0,
        force_in_match=True,
    )
    # per-video 手調整 HSV inject: disable_per_video_hsv=True の場合はスキップする。
    # OnlineHsvCalibrator は load_default で生成済みのため自動 HSV 学習は継続する。
    if not disable_per_video_hsv:
        _inject_hsv(pipe, _resolve_hsv_path(video_id))
    return pipe


# ============================
# 1 動画処理
# ============================


def _open_capture(
    video_path: Path,
    max_frames: int,
    sample_interval_sec: float,
) -> tuple:
    """動画キャプチャを開き (cap, fps, n_target, interval_frames, clip_duration_sec) を返す。

    clip_duration_sec: クリップ全体の時間 (秒)。中盤末尾除外判定に使う。
    戻り値: (cap, fps, n_target, interval_frames, clip_duration_sec) または None。
    backwards compat: 既存呼出元は 4 要素 tuple を期待しているため、
    5 要素 tuple に拡張。呼出元 (_process_video) も同時に更新する。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_target = total_frames if max_frames <= 0 else min(total_frames, max_frames)
    interval_frames = max(1, int(round(sample_interval_sec * fps)))
    # クリップ全体の秒数 (= 実総フレーム数を使う。max_frames 制限前の値)
    clip_duration_sec = total_frames / fps if fps > 0 else 0.0
    return cap, fps, n_target, interval_frames, clip_duration_sec



def _eval_one_frame(
    video_id: str,
    fi: int,
    fps: float,
    interval_frames: int,
    frame: object,
    pipe_cnn: RecognitionPipeline,
    pipe_hsv: RecognitionPipeline,
    stats: VideoStats,
    disagreements: list[dict],
    persist_min_frames: int = CORRUPTION_PERSIST_MIN_FRAMES,
) -> None:
    """1 frame の認識・合意判定を行い stats を更新する。

    3 者独立方式:
      raw_cnn   = res_cnn.pX.cnn_board  (ImageReader 直出力、物理推論 post-process 前)
      raw_hsv   = res_hsv.pX.cnn_board  (HSV-only pipeline の ImageReader 直出力)
      confirmed = res_cnn.pX.confirmed_board (CNN+物理推論 post-process 後の確定盤面)

    Args:
        persist_min_frames: 持続 corruption と見なす最小連続サンプルフレーム数。
            デフォルト = CORRUPTION_PERSIST_MIN_FRAMES (= 3)。
            後方互換: オプション引数のため既存呼出元は変更不要。
    """
    t_sec = fi / fps
    if fi % interval_frames != 0:
        return
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    res_cnn = pipe_cnn.update(fi, t_sec, frame)
    res_hsv = pipe_hsv.update(fi, t_sec, frame)
    for side, sr_cnn, sr_hsv in [
        ("1P", res_cnn.p1, res_hsv.p1),
        ("2P", res_cnn.p2, res_hsv.p2),
    ]:
        if sr_cnn.state != BoardState.STABLE or sr_cnn.confirmed_board is None:
            # 改修3: CHAIN state は大連鎖の正常 non-stable のためカウント対象外にする。
            # MENU / TSUMO_FALL / EFFECT 等の異常系 non-stable のみを連続カウントする。
            # NON_STABLE_CHAIN_EXCLUDE=False で旧挙動 (chain もカウント) に戻せる。
            is_chain_state = (
                NON_STABLE_CHAIN_EXCLUDE
                and sr_cnn.state == BoardState.CHAIN
            )
            if is_chain_state:
                # 連鎖中はカウンタをリセットして連続検知を中断する
                stats._non_stable_current_by_side[side] = 0
            elif t_sec >= NON_STABLE_WARMUP_SEC:
                # warmup 後のみカウント (chain 以外の non-stable)
                stats._non_stable_current_by_side[side] = (
                    stats._non_stable_current_by_side.get(side, 0) + 1
                )
                cur = stats._non_stable_current_by_side[side]
                if cur > stats.non_stable_max_consecutive:
                    stats.non_stable_max_consecutive = cur
            # non-STABLE に遷移した = 進行中の corruption run を確定してリセット
            # non-STABLE を挟んだら「連続サンプルフレームの継続性」が途切れる
            _flush_corruption_persist_runs_for_side(side, persist_min_frames, stats)
            continue
        # STABLE フレームで non-stable カウントをリセット
        stats._non_stable_current_by_side[side] = 0
        stats.stable_frame_count += 1
        # C1: STABLE confirmed_board のぷよ数を集計 (= avg_puyo_count 計算用)
        # 改修2: side を渡して per-side 集計も有効化する
        _collect_puyo_count(sr_cnn.confirmed_board, stats, side=side)
        _eval_side_frame(
            side, fi, t_sec, video_id,
            raw_cnn_board=sr_cnn.cnn_board,
            raw_hsv_board=sr_hsv.cnn_board,
            confirmed_board=sr_cnn.confirmed_board,
            stats=stats,
            disagreements=disagreements,
            persist_min_frames=persist_min_frames,
        )


def _run_frame_loop(
    video_id: str,
    cap: object,
    fps: float,
    n_target: int,
    interval_frames: int,
    is_holdout: bool,
    pipe_cnn: RecognitionPipeline,
    pipe_hsv: RecognitionPipeline,
    disagreements: list[dict],
    clip_duration_sec: float = 0.0,
    persist_min_frames: int = CORRUPTION_PERSIST_MIN_FRAMES,
) -> VideoStats:
    """動画 frame ループを走らせ VideoStats を返す。

    clip_duration_sec: クリップ全体の秒数 (中盤末尾除外に使用)。
    0.0 = 不明 → 末尾除外を適用しない (後方互換)。
    persist_min_frames: 持続 corruption と見なす最小連続サンプルフレーム数。
        デフォルト = CORRUPTION_PERSIST_MIN_FRAMES (= 3)。後方互換: optional。
    """
    stats = VideoStats(video_id=video_id, is_holdout=is_holdout)
    stats.clip_duration_sec = clip_duration_sec
    for fi in range(n_target):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        _eval_one_frame(
            video_id, fi, fps, interval_frames, frame,
            pipe_cnn, pipe_hsv, stats, disagreements,
            persist_min_frames=persist_min_frames,
        )
        if fi % 300 == 0 and fi > 0:
            print(
                f"  [progress] {fi}/{n_target} ({fi*100/max(n_target,1):.0f}%) "
                f"agreed={stats.agreed_cells} total={stats.total_cells}"
            )
    # 動画末端 (eof): 残存する全 active run を確定する
    _flush_all_corruption_persist_runs(persist_min_frames, stats)
    return stats


def _process_video(
    video_id: str,
    video_path: Path,
    is_holdout: bool,
    max_frames: int,
    sample_interval_sec: float,
    disagreements: list[dict],
    enable_constraint_fill: bool = True,
    enable_t2_highconf_yield: bool = False,
    enable_infer_empty_guard: bool = False,
    enable_game_event_chain_exit: bool = False,
    enable_landing_color_fix: bool = False,
    enable_chain_min_display: bool = False,
    enable_hsv_classify_fallback: bool = False,
    enable_landing_observed_color: bool = False,
    enable_red_hue_wrap_fix: bool = False,
    enable_specular_robust_saturation: bool = False,
    enable_stable_recovery_gate: bool = False,
    enable_ojama_visual_detection: bool = False,
    enable_ojama_visual_chain_exit: bool = False,
    enable_ojama_infer_guard: bool = False,
    enable_ojama_settle_detection: bool = False,
    enable_ojama_tier1_warmup: bool = False,
    enable_chain_score_early_fire: bool = False,
    enable_chain_exit_warmup: bool = False,
    enable_chain_formula_detection: bool = False,
    enable_hsv_deferred_consensus: bool = False,
    enable_ojama_warning_glow_guard: bool = False,
    enable_chain_max_hold_override: bool = False,
    # 案X*(A)(B)+warmup (2026-06-05): NextSlide signal による CHAIN 即終了。
    enable_chain_exit_next_signal: bool = False,
    # feat/gravity-settle-2026-06-05: 連鎖終了直後 GRAVITY_SETTLE 状態を有効化。
    enable_gravity_settle_state: bool = False,
    persist_min_frames: int = CORRUPTION_PERSIST_MIN_FRAMES,
    disable_per_video_hsv: bool = False,
) -> VideoStats:
    """1 動画を処理し VideoStats を返す。

    Args:
        enable_constraint_fill: False にすると confirmed 経路の
            constraint_fill を無効化して測定する。
            backwards compat: デフォルト True = 従来挙動。
        enable_t2_highconf_yield: True にすると T2 の prev_stable 上書きを
            CNN 支持セルでスキップする (infer_placement 誤推論 + T2 フリーズ修正)。
            backwards compat: デフォルト False = 従来挙動。
        enable_infer_empty_guard: True にすると infer_placement 空セル
            hallucination ガードを有効化する。
            backwards compat: デフォルト False = 従来挙動。
        enable_game_event_chain_exit: True にすると game-event ベース連鎖終了を
            有効化する (次ツモ変化 / お邪魔降下で CHAIN 終了)。
            backwards compat: デフォルト False = 従来挙動。
        enable_landing_color_fix: True にすると TSUMO_FALL→STABLE 着地時の
            falling_pair を _landing_pending (消費済みツモ色) に切り替える。
            backwards compat: デフォルト False = 従来挙動。
        enable_chain_min_display: True にすると X1/X4 短連鎖ちらつき対策を有効化。
            backwards compat: デフォルト False = 従来挙動。
        enable_hsv_classify_fallback: True にすると HSV 分類 fallback を有効化。
            _classify_next_pair_by_hsv の 2 択強制確定を回避し、
            黄→赤誤分類 (~900 件) 発火点を修正する。
            backwards compat: デフォルト False = 従来挙動。
        enable_landing_observed_color: True にすると着地セルの CNN==HSV 一致色補正を有効化。
            falling_pair タイミングずれで生じる着地色誤りを上流で断つ (真因 A 対処)。
            backwards compat: デフォルト False = 従来挙動。
        enable_red_hue_wrap_fix: True にすると赤色相折り返し補正を有効化する。
            HSV median で赤 2 峰 (H=0-4 と H=166-179) を 1 峰に collapse する。
            backwards compat: デフォルト False = 従来挙動。
        enable_specular_robust_saturation: True にすると光沢ハイライト除外彩度計算を有効化。
            白ハイライト画素を彩度 median 計算から除外して EMPTY 誤判定を防ぐ (案D)。
            backwards compat: デフォルト False = 従来挙動。
    """
    cap_info = _open_capture(video_path, max_frames, sample_interval_sec)
    if cap_info is None:
        print(f"[measure] 動画を開けません: {video_path}", file=sys.stderr)
        return VideoStats(video_id=video_id, is_holdout=is_holdout)
    cap, fps, n_target, interval_frames, clip_duration_sec = cap_info
    # confirmed 経路 (CNN+物理推論) のみ各フラグを制御する。
    # raw_hsv 経路は constraint_fill を通らないため変更不要。
    pipe_cnn = _make_pipeline_cnn(
        video_id,
        enable_constraint_fill=enable_constraint_fill,
        enable_t2_highconf_yield=enable_t2_highconf_yield,
        enable_infer_empty_guard=enable_infer_empty_guard,
        enable_game_event_chain_exit=enable_game_event_chain_exit,
        enable_landing_color_fix=enable_landing_color_fix,
        enable_chain_min_display=enable_chain_min_display,
        enable_hsv_classify_fallback=enable_hsv_classify_fallback,
        enable_landing_observed_color=enable_landing_observed_color,
        enable_red_hue_wrap_fix=enable_red_hue_wrap_fix,
        enable_specular_robust_saturation=enable_specular_robust_saturation,
        enable_stable_recovery_gate=enable_stable_recovery_gate,
        enable_ojama_visual_detection=enable_ojama_visual_detection,
        enable_ojama_visual_chain_exit=enable_ojama_visual_chain_exit,
        enable_ojama_infer_guard=enable_ojama_infer_guard,
        enable_ojama_settle_detection=enable_ojama_settle_detection,
        enable_ojama_tier1_warmup=enable_ojama_tier1_warmup,
        enable_chain_score_early_fire=enable_chain_score_early_fire,
        enable_chain_exit_warmup=enable_chain_exit_warmup,
        enable_chain_formula_detection=enable_chain_formula_detection,
        enable_hsv_deferred_consensus=enable_hsv_deferred_consensus,
        enable_ojama_warning_glow_guard=enable_ojama_warning_glow_guard,
        enable_chain_max_hold_override=enable_chain_max_hold_override,
        enable_chain_exit_next_signal=enable_chain_exit_next_signal,
        enable_gravity_settle_state=enable_gravity_settle_state,
        disable_per_video_hsv=disable_per_video_hsv,
    )
    # disable_per_video_hsv=True のとき raw_hsv 軸も手調整 inject をスキップし、
    # 全 3 軸 (raw_cnn / raw_hsv / confirmed) を自動 HSV のみで動作させる。
    pipe_hsv = _make_pipeline_hsv_only(video_id, disable_per_video_hsv=disable_per_video_hsv)
    print(
        f"[measure] {video_id}: fps={fps:.1f} target={n_target} "
        f"holdout={is_holdout} clip_duration={clip_duration_sec:.1f}s"
        + (" [disable_per_video_hsv=ON: 全3軸自動HSVのみ]" if disable_per_video_hsv else "")
    )
    stats = _run_frame_loop(
        video_id, cap, fps, n_target, interval_frames,
        is_holdout, pipe_cnn, pipe_hsv, disagreements,
        clip_duration_sec=clip_duration_sec,
        persist_min_frames=persist_min_frames,
    )
    cap.release()
    rate = stats.agreed_cells / stats.total_cells if stats.total_cells > 0 else 0.0
    print(
        f"[measure] {video_id} 完了: stable={stats.stable_frame_count} "
        f"total={stats.total_cells} 合意率={rate:.4f} disagree={stats.disagreement_count}"
    )
    return stats
def _process_video_worker(
    video_id: str,
    video_path_str: str,
    is_holdout: bool,
    max_frames: int,
    sample_interval_sec: float,
    enable_constraint_fill: bool,
    enable_t2_highconf_yield: bool = False,
    enable_infer_empty_guard: bool = False,
    enable_game_event_chain_exit: bool = False,
    enable_landing_color_fix: bool = False,
    enable_chain_min_display: bool = False,
    enable_hsv_classify_fallback: bool = False,
    enable_landing_observed_color: bool = False,
    enable_red_hue_wrap_fix: bool = False,
    enable_specular_robust_saturation: bool = False,
    enable_stable_recovery_gate: bool = False,
    enable_ojama_visual_detection: bool = False,
    enable_ojama_visual_chain_exit: bool = False,
    enable_ojama_infer_guard: bool = False,
    enable_ojama_settle_detection: bool = False,
    enable_ojama_tier1_warmup: bool = False,
    enable_chain_score_early_fire: bool = False,
    enable_chain_exit_warmup: bool = False,
    enable_chain_formula_detection: bool = False,
    enable_hsv_deferred_consensus: bool = False,
    enable_ojama_warning_glow_guard: bool = False,
    enable_chain_max_hold_override: bool = False,
    # 案X*(A)(B)+warmup (2026-06-05): NextSlide signal による CHAIN 即終了。
    enable_chain_exit_next_signal: bool = False,
    # feat/gravity-settle-2026-06-05: 連鎖終了直後 GRAVITY_SETTLE 状態を有効化。
    enable_gravity_settle_state: bool = False,
    persist_min_frames: int = CORRUPTION_PERSIST_MIN_FRAMES,
    disable_per_video_hsv: bool = False,
) -> VideoStats:
    """並列ワーカ用: 1 動画を処理して VideoStats を返す。

    引数はすべて pickle 可能な軽量型のみ。
    pipeline は本関数内で load_default() してワーカ内ロードする (CUDA spawn 安全)。
    不一致 cell は stats._local_disagreements に格納して返す。
    """
    # ワーカ内で sys.path を復元する (spawn では親の path が引き継がれない場合がある)
    import sys
    from pathlib import Path as _Path
    _proj = str(_Path(__file__).resolve().parent.parent)
    if _proj not in sys.path:
        sys.path.insert(0, _proj)
    # ワーカ内でローカル disagreements を収集する
    local_disagrees: list[dict] = []
    stats = _process_video(
        video_id=video_id,
        video_path=_Path(video_path_str),
        is_holdout=is_holdout,
        max_frames=max_frames,
        sample_interval_sec=sample_interval_sec,
        disagreements=local_disagrees,
        enable_constraint_fill=enable_constraint_fill,
        enable_t2_highconf_yield=enable_t2_highconf_yield,
        enable_infer_empty_guard=enable_infer_empty_guard,
        enable_game_event_chain_exit=enable_game_event_chain_exit,
        enable_landing_color_fix=enable_landing_color_fix,
        enable_chain_min_display=enable_chain_min_display,
        enable_hsv_classify_fallback=enable_hsv_classify_fallback,
        enable_landing_observed_color=enable_landing_observed_color,
        enable_red_hue_wrap_fix=enable_red_hue_wrap_fix,
        enable_specular_robust_saturation=enable_specular_robust_saturation,
        enable_stable_recovery_gate=enable_stable_recovery_gate,
        enable_ojama_visual_detection=enable_ojama_visual_detection,
        enable_ojama_visual_chain_exit=enable_ojama_visual_chain_exit,
        enable_ojama_infer_guard=enable_ojama_infer_guard,
        enable_ojama_settle_detection=enable_ojama_settle_detection,
        enable_ojama_tier1_warmup=enable_ojama_tier1_warmup,
        enable_chain_score_early_fire=enable_chain_score_early_fire,
        enable_chain_exit_warmup=enable_chain_exit_warmup,
        enable_chain_formula_detection=enable_chain_formula_detection,
        enable_hsv_deferred_consensus=enable_hsv_deferred_consensus,
        enable_ojama_warning_glow_guard=enable_ojama_warning_glow_guard,
        enable_chain_max_hold_override=enable_chain_max_hold_override,
        enable_chain_exit_next_signal=enable_chain_exit_next_signal,
        enable_gravity_settle_state=enable_gravity_settle_state,
        persist_min_frames=persist_min_frames,
        disable_per_video_hsv=disable_per_video_hsv,
    )
    stats._local_disagreements = local_disagrees
    return stats


def _majority_vote(a: int, b: int, c: int) -> int:
    """3 値の多数決を返す。全員不一致の場合は a (raw_cnn) を返す。"""
    if a == b or a == c:
        return a
    if b == c:
        return b
    return a  # 全員不一致: raw_cnn を基準にする


def _classify_corruption_subcategory(
    raw_val: int,
    confirmed_val: int,
    stats: "VideoStats",
) -> None:
    """corruption のサブカテゴリを判定して stats に加算する。

    サブカテゴリ:
      empty_to_color: raw==EMPTY → confirmed==色 (空→色FP、infer_placement hallucination 主因)
      color_to_color: raw==色A  → confirmed==色B (色フリーズ系、T2 自己強化が主因)
      color_to_empty: raw==色   → confirmed==EMPTY (色→空 消失)

    Args:
        raw_val: raw_cnn == raw_hsv の合意値。
        confirmed_val: 後処理後の確定値。
        stats: 加算先の VideoStats インスタンス。
    """
    # EVAL_COLORS に含まれない値は呼び出し元でガード済みのため追加チェック不要
    if raw_val == COLOR_EMPTY and confirmed_val != COLOR_EMPTY:
        # 空→色 FP: infer_placement / constraint_fill deficit force-fill が主因
        stats.corruption_empty_to_color_count += 1
    elif raw_val != COLOR_EMPTY and confirmed_val != COLOR_EMPTY:
        # 色→別色: T2 フリーズ / constraint_fill replace が主因
        stats.corruption_color_to_color_count += 1
    elif raw_val != COLOR_EMPTY and confirmed_val == COLOR_EMPTY:
        # 色→空: infer_placement commit refuse 後に T2 が消す等の経路
        stats.corruption_color_to_empty_count += 1


def _check_postprocess_corruption(
    video_id: str,
    fi: int,
    t_sec: float,
    side: str,
    row: int,
    col: int,
    raw_cnn_val: int,
    raw_hsv_val: int,
    confirmed_val: int,
    stats: VideoStats,
    *,
    log_limit: int = CORRUPTION_LOG_LIMIT,
) -> None:
    """後処理破壊 (postprocess_corruption) を検知して stats に記録する。

    検知条件:
      - raw_cnn_val == raw_hsv_val  (CNN と HSV が一致: 正解候補として信頼できる)
      - confirmed_val != raw_cnn_val  (後処理が書き換えた)
      - raw_cnn_val と confirmed_val どちらも COLOR_UNKNOWN でない

    これは constraint_fill が CNN・HSV 一致セルを誤って上書きしたケースを捕捉する。
    NOTE: raw_cnn==raw_hsv==誤りの全列崩壊型は本検知の対象外。
    　　　その場合は per_col_midgame_empty (I1) / avg_puyo_count (C1) で別途検知。
    UNKNOWN 除外は per_col_unknown_rate (I1) が担当するためここでは除外のみ実施。
    """
    if (
        raw_cnn_val == raw_hsv_val
        and confirmed_val != raw_cnn_val
        and raw_cnn_val not in (COLOR_UNKNOWN,)
        and confirmed_val not in (COLOR_UNKNOWN,)
    ):
        stats.postprocess_corruption_count += 1
        stats.postprocess_corruption_by_side[side] += 1
        stats.postprocess_corruption_color_pairs[(raw_cnn_val, confirmed_val)] += 1
        # サブカテゴリ分類: 変化方向で分類する
        _classify_corruption_subcategory(
            raw_cnn_val, confirmed_val, stats,
        )
        if len(stats.postprocess_corruption_log) < log_limit:
            stats.postprocess_corruption_log.append({
                "video": video_id,
                "frame": fi,
                "t_sec": round(t_sec, 2),
                "side": side,
                "row": row,
                "col": col,
                "raw_cnn": COLOR_NAMES.get(raw_cnn_val, str(raw_cnn_val)),
                "confirmed": COLOR_NAMES.get(confirmed_val, str(confirmed_val)),
            })


def _get_persist_subcategory(raw_val: int, confirmed_val: int) -> str:
    """corruption の方向文字列を返す (持続 run 分類用)。

    戻り値: "empty_to_color" / "color_to_color" / "color_to_empty"
    """
    if raw_val == COLOR_EMPTY and confirmed_val != COLOR_EMPTY:
        return "empty_to_color"
    if raw_val != COLOR_EMPTY and confirmed_val == COLOR_EMPTY:
        return "color_to_empty"
    return "color_to_color"


def _update_corruption_persist_run(
    side: str,
    row: int,
    col: int,
    raw_val: int,
    confirmed_val: int,
    is_corruption: bool,
    stats: "VideoStats",
    persist_min_frames: int,
) -> None:
    """1 STABLE フレームの持続 corruption run を更新する。

    corruption が継続中なら run length を +1 する。
    run が終了した (corruption 解消 or 値変化) 場合は run を確定し、
    run 長 >= persist_min_frames なら postprocess_corruption_persistent_count に計上する。
    non-STABLE フレームは _flush_corruption_persist_runs_for_side で処理する。

    Args:
        is_corruption: 現フレームで raw==confirmed 違反か。
        persist_min_frames: 持続と見なす最小フレーム数。
    """
    key = (side, row, col)
    run = stats._persist_run_state.get(key)

    if is_corruption:
        subcat = _get_persist_subcategory(raw_val, confirmed_val)
        if run is None:
            # 新規 run 開始
            stats._persist_run_state[key] = {
                "raw_val": raw_val,
                "confirmed_val": confirmed_val,
                "length": 1,
                "subcategory": subcat,
            }
        elif run["raw_val"] == raw_val and run["confirmed_val"] == confirmed_val:
            # 同一 corruption が継続
            run["length"] += 1
        else:
            # 値が変化した = 前の run を確定してから新規開始
            _commit_persist_run(run, persist_min_frames, stats)
            stats._persist_run_state[key] = {
                "raw_val": raw_val,
                "confirmed_val": confirmed_val,
                "length": 1,
                "subcategory": subcat,
            }
    else:
        # corruption が解消された = run を確定
        if run is not None:
            _commit_persist_run(run, persist_min_frames, stats)
            del stats._persist_run_state[key]


def _commit_persist_run(
    run: dict,
    persist_min_frames: int,
    stats: "VideoStats",
) -> None:
    """run を確定して持続 corruption に計上する (run 長 >= persist_min_frames 時のみ)。

    1 run に含まれる全フレームを計上するのではなく、
    run 長が閾値以上かどうかのみで判定して run 長分を加算する。
    この方式により「N fr 以上連続した corruption のみ」を集計できる。
    """
    if run["length"] < persist_min_frames:
        return
    stats.postprocess_corruption_persistent_count += run["length"]
    subcat = run["subcategory"]
    if subcat == "empty_to_color":
        stats.corruption_persistent_empty_to_color_count += run["length"]
    elif subcat == "color_to_color":
        stats.corruption_persistent_color_to_color_count += run["length"]
    elif subcat == "color_to_empty":
        stats.corruption_persistent_color_to_empty_count += run["length"]


def _flush_corruption_persist_runs_for_side(
    side: str,
    persist_min_frames: int,
    stats: "VideoStats",
) -> None:
    """指定 side の全 active run を確定する (non-STABLE 遷移 or eof 時に呼ぶ)。

    non-STABLE フレームを挟んだ場合は run をリセットする。
    これにより「連続サンプルフレームでの持続」を保証する。
    """
    keys_to_remove = [k for k in stats._persist_run_state if k[0] == side]
    for key in keys_to_remove:
        run = stats._persist_run_state.pop(key)
        _commit_persist_run(run, persist_min_frames, stats)


def _flush_all_corruption_persist_runs(
    persist_min_frames: int,
    stats: "VideoStats",
) -> None:
    """全 side の全 active run を確定する (動画 eof 時に呼ぶ)。"""
    for run in list(stats._persist_run_state.values()):
        _commit_persist_run(run, persist_min_frames, stats)
    stats._persist_run_state.clear()


def _record_cell(
    video_id: str, fi: int, t_sec: float, side: str,
    row: int, col: int,
    raw_cnn_val: int, raw_hsv_val: int, confirmed_val: int,
    stats: VideoStats,
    disagreements: list[dict],
) -> None:
    """1 cell の 3 者独立合意判定結果を stats と disagreements に記録する。

    合意ラベル = raw_cnn / raw_hsv / confirmed の多数決 (2 者以上一致)。
    全員不一致時は raw_cnn を正解ラベルとして扱う (最保守的方針)。

    合意 = 多数決ラベルが raw_cnn と一致 (= 集計基準は raw_cnn 主軸)。
    """
    label = _majority_vote(raw_cnn_val, raw_hsv_val, confirmed_val)
    stats.total_cells += 1
    # total_by_color / total_by_row は label (多数決) を基準にする
    stats.total_by_color[label] += 1
    stats.total_by_row[row] += 1

    all_agree = (raw_cnn_val == raw_hsv_val == confirmed_val)
    if all_agree:
        stats.all_three_agree_count += 1

    # 改修1: confirmed_majority_agree_rate 集計
    # STABLE セルで confirmed_val == majority_label(多数決) の一致率を計算する。
    # agreed / disagreement 両ケースを含む全 STABLE セルで集計する。
    # 1.0 に近いほど confirmed が多数決と整合 (confirmed 精度の代理指標)。
    # FAIL 判定には使わず情報提示のみ。
    stats._confirmed_total_cells += 1
    stats._confirmed_total_by_color[label] += 1
    if confirmed_val == label:
        stats._confirmed_agree_cells += 1
        stats._confirmed_agree_by_color[label] += 1

    if raw_cnn_val == label:
        stats.agreed_cells += 1
        stats.correct_by_color[label] += 1
        stats.correct_by_row[row] += 1
    else:
        stats.disagreement_count += 1
        if len(disagreements) < DISAGREEMENT_COLLECT_LIMIT:
            disagreements.append({
                "video": video_id, "frame": fi, "t_sec": round(t_sec, 2),
                "side": side, "cell": [row, col],
                "predictions": {
                    "raw_cnn": COLOR_NAMES.get(raw_cnn_val, str(raw_cnn_val)),
                    "raw_hsv": COLOR_NAMES.get(raw_hsv_val, str(raw_hsv_val)),
                    "confirmed": COLOR_NAMES.get(confirmed_val, str(confirmed_val)),
                    "majority_label": COLOR_NAMES.get(label, str(label)),
                },
            })
        return

    # physics_fix_count: raw_cnn != raw_hsv だが confirmed が label と一致したケース
    if raw_cnn_val != raw_hsv_val and confirmed_val == label:
        stats.physics_fix_count += 1


def _eval_side_frame(
    side: str,
    fi: int,
    t_sec: float,
    video_id: str,
    raw_cnn_board: object,
    raw_hsv_board: object,
    confirmed_board: object,
    stats: VideoStats,
    disagreements: list[dict],
    persist_min_frames: int = CORRUPTION_PERSIST_MIN_FRAMES,
) -> None:
    """1 サイド × 1 frame の cell ごとに 3 者独立合意判定し stats を更新する。

    Args:
        raw_cnn_board: CNN+HSV hybrid ImageReader 直出力 (物理推論 post-process 前)。
        raw_hsv_board: HSV-only pipeline の ImageReader 直出力 (CNN 無効化済)。
        confirmed_board: CNN+物理推論 post-process 後の確定盤面 (STABLE 時のみ非 None)。
        persist_min_frames: 持続 corruption と見なす最小連続サンプルフレーム数。
            デフォルト = CORRUPTION_PERSIST_MIN_FRAMES (= 3)。
            後方互換: オプション引数のため既存呼出元は変更不要。
    """
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            raw_cnn_val = int(raw_cnn_board.get(row, col)) if raw_cnn_board is not None else COLOR_UNKNOWN
            raw_hsv_val = int(raw_hsv_board.get(row, col)) if raw_hsv_board is not None else COLOR_UNKNOWN
            confirmed_val = int(confirmed_board.get(row, col)) if confirmed_board is not None else COLOR_UNKNOWN
            if SKIP_UNKNOWN_CELLS and (
                raw_cnn_val == COLOR_UNKNOWN
                or raw_hsv_val == COLOR_UNKNOWN
                or confirmed_val == COLOR_UNKNOWN
            ):
                continue
            # 評価対象色フィルタ: 多数決ラベルで判定 (全員 UNKNOWN 等を除外)
            label = _majority_vote(raw_cnn_val, raw_hsv_val, confirmed_val)
            if label not in EVAL_COLORS:
                continue
            _record_cell(
                video_id, fi, t_sec, side, row, col,
                raw_cnn_val, raw_hsv_val, confirmed_val,
                stats, disagreements,
            )
            # 後処理破壊検知: raw_cnn==raw_hsv なのに confirmed が異なるセルを検知
            is_corruption = (
                raw_cnn_val == raw_hsv_val
                and confirmed_val != raw_cnn_val
                and raw_cnn_val not in (COLOR_UNKNOWN,)
                and confirmed_val not in (COLOR_UNKNOWN,)
            )
            _check_postprocess_corruption(
                video_id, fi, t_sec, side, row, col,
                raw_cnn_val, raw_hsv_val, confirmed_val,
                stats,
            )
            # 持続 corruption run 追跡: is_corruption を用いて run を更新する
            _update_corruption_persist_run(
                side, row, col, raw_cnn_val, confirmed_val,
                is_corruption, stats, persist_min_frames,
            )
    # I1 メトリクス集計: confirmed_board の col 別 UNKNOWN 率 + 中盤 EMPTY 率
    if confirmed_board is not None:
        _collect_col_metrics(fi, t_sec, confirmed_board, stats)


def _collect_puyo_count(
    confirmed_board: object,
    stats: VideoStats,
    side: Optional[str] = None,
) -> None:
    """STABLE confirmed_board の非 EMPTY・非 UNKNOWN cell 数を stats に加算する。

    C1 avg_puyo_count_per_stable_frame 計算用。
    1 サイド分のカウントを加算する (= frame ごとに 1P / 2P 別に呼ばれる)。

    Args:
        confirmed_board: STABLE 確定盤面。None なら何もしない。
        stats: 集計先の VideoStats インスタンス。
        side: "1P" or "2P"。改修2 per-side 集計用。None の場合は per-side 集計を行わない
              (backwards compat: 既存呼出元は side=None のまま動作)。
    """
    if confirmed_board is None:
        return
    count = 0
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            val = int(confirmed_board.get(row, col))
            if val not in (COLOR_EMPTY, COLOR_UNKNOWN):
                count += 1
    # 従来の 1P+2P 合算集計 (後方互換維持)
    stats._puyo_count_sum += count
    stats._puyo_count_n_stable += 1
    # 改修2: per-side 別集計 (side が指定された場合のみ)
    if side is not None:
        stats._puyo_count_sum_by_side[side] += count
        stats._puyo_count_n_stable_by_side[side] += 1


def _is_midgame_frame(t_sec: float, clip_duration_sec: float) -> bool:
    """フレームが中盤評価区間に属するかを返す。

    中盤評価区間の定義:
      [MIDGAME_START_SEC, clip_duration_sec - MIDGAME_TRAIL_EXCLUDE_SEC)

    clip_duration_sec=0.0 (不明) の場合は末尾除外を適用せず、
    従来通り t_sec >= MIDGAME_START_SEC のみで判定する (後方互換)。

    短クリップ偽陽性対策 (v70_match01 問題):
      30.5s クリップでは末尾 10s を除外すると有効中盤区間が
      [30.0, 20.5) となりフレームが含まれなくなる。
      MIDGAME_COL_MIN_FRAMES (=30) に満たないため CRITICAL 判定されない。
    """
    if t_sec < MIDGAME_START_SEC:
        return False
    if clip_duration_sec <= 0.0:
        # clip 長が不明のときは末尾除外を適用しない (後方互換)
        return True
    trail_cutoff = clip_duration_sec - MIDGAME_TRAIL_EXCLUDE_SEC
    return t_sec < trail_cutoff


def _collect_col_metrics(
    fi: int,
    t_sec: float,
    confirmed_board: object,
    stats: VideoStats,
) -> None:
    """STABLE confirmed_board から col 別 UNKNOWN 率と中盤 EMPTY 率を集計する。

    col 別 UNKNOWN 率が高い = STABLE 中の認識崩壊 (v89 27-30s 相当) を捕捉。
    中盤 EMPTY 率が 100% = col=1 全 EMPTY 誤判定 (v40_match01 相当) を捕捉。
    clip_duration_sec は stats.clip_duration_sec から取得する。
    """
    is_midgame = _is_midgame_frame(t_sec, stats.clip_duration_sec)
    for col in range(BOARD_COLS):
        col_unknown = 0
        col_cells = 0
        col_empty_mid = 0
        col_cells_mid = 0
        for row in range(BOARD_ROWS):
            val = int(confirmed_board.get(row, col))
            col_cells += 1
            if val == COLOR_UNKNOWN:
                col_unknown += 1
            if is_midgame:
                col_cells_mid += 1
                if val == COLOR_EMPTY:
                    col_empty_mid += 1
        stats.per_col_stable_cells[col] += col_cells
        stats.per_col_unknown_cells[col] += col_unknown
        if is_midgame:
            stats.per_col_midgame_cells[col] += col_cells_mid
            stats.per_col_midgame_empty_cells[col] += col_empty_mid


# ============================
# 集計・判定
# ============================


def _build_color_acc(stats_list: list[VideoStats]) -> dict[str, float]:
    """色別合意率 dict を生成する。"""
    total: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    for s in stats_list:
        for c in EVAL_COLORS:
            total[c] += s.total_by_color.get(c, 0)
            correct[c] += s.correct_by_color.get(c, 0)
    return {
        COLOR_NAMES[c]: correct[c] / total[c] if total[c] > 0 else 1.0
        for c in EVAL_COLORS
    }


def _build_row_acc(stats_list: list[VideoStats]) -> dict[str, float]:
    """行別合意率 dict を生成する。"""
    total: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    for s in stats_list:
        for r in range(BOARD_ROWS):
            total[r] += s.total_by_row.get(r, 0)
            correct[r] += s.correct_by_row.get(r, 0)
    return {
        f"row_{r}": correct[r] / total[r] if total[r] > 0 else 1.0
        for r in range(BOARD_ROWS)
    }


def _build_avg_puyo_by_side(s: VideoStats) -> dict[str, object]:
    """VideoStats から per-side avg_puyo_count を計算して返す。

    改修2: 1P / 2P 別の STABLE フレーム平均ぷよ数を集計する。
    データがない side は None を返す (後方互換)。
    Returns: {"1P": float|None, "2P": float|None}
    """
    result: dict[str, object] = {}
    for side in ("1P", "2P"):
        n = s._puyo_count_n_stable_by_side.get(side, 0)
        total = s._puyo_count_sum_by_side.get(side, 0)
        result[side] = total / n if n > 0 else None
    return result


def _build_confirmed_agree_rate(s: VideoStats) -> dict[str, object]:
    """VideoStats から confirmed_majority_agree_rate を計算して返す。

    改修1: confirmed_val == majority_label(多数決) の STABLE 全セル一致率。
    overall + per_color を返す。FAIL 判定には使わず情報提示のみ。
    Returns: {"overall": float|None, "per_color": {color_name: float|None}}
    """
    overall = (
        s._confirmed_agree_cells / s._confirmed_total_cells
        if s._confirmed_total_cells > 0 else None
    )
    per_color: dict[str, object] = {}
    for c in EVAL_COLORS:
        total = s._confirmed_total_by_color.get(c, 0)
        agree = s._confirmed_agree_by_color.get(c, 0)
        per_color[COLOR_NAMES[c]] = agree / total if total > 0 else None
    return {"overall": overall, "per_color": per_color}


def _build_video_acc(stats_list: list[VideoStats]) -> dict[str, dict]:
    """動画別集計 dict を生成する。"""
    return {
        s.video_id: {
            "acc": s.agreed_cells / s.total_cells if s.total_cells > 0 else 0.0,
            "total_cells": s.total_cells,
            "agreed_cells": s.agreed_cells,
            "disagreement_count": s.disagreement_count,
            "stable_frame_count": s.stable_frame_count,
            "is_holdout": s.is_holdout,
            # 3 者独立追加メトリクス
            "physics_fix_count": s.physics_fix_count,
            "all_three_agree_count": s.all_three_agree_count,
            # I1 追加メトリクス
            "non_stable_max_consecutive": s.non_stable_max_consecutive,
            "per_col_unknown_rate": {
                str(col): (
                    s.per_col_unknown_cells.get(col, 0)
                    / s.per_col_stable_cells.get(col, 1)
                )
                for col in range(6)
            },
            "per_col_midgame_empty_rate": {
                str(col): (
                    s.per_col_midgame_empty_cells.get(col, 0)
                    / s.per_col_midgame_cells.get(col, 1)
                    if s.per_col_midgame_cells.get(col, 0) > 0 else None
                )
                for col in range(6)
            },
            # C1 avg_puyo_count_per_stable_frame (= fail-silent 経路検知)
            "avg_puyo_count_per_stable_frame": (
                s._puyo_count_sum / s._puyo_count_n_stable
                if s._puyo_count_n_stable > 0 else None
            ),
            "n_stable_frames_puyo": s._puyo_count_n_stable,
            # 改修2: per-side 別 avg_puyo_count
            # 1P / 2P 各サイドの STABLE フレーム平均ぷよ数。
            # 一方のサイドのみ極端に低い場合は片側列崩壊を示す。
            "avg_puyo_count_per_side": _build_avg_puyo_by_side(s),
            # 改修1: confirmed_majority_agree_rate (confirmed 精度代理指標)
            # confirmed_val == majority_label の全 STABLE セル一致率。
            # FAIL 判定には使わず情報提示のみ。1.0 = confirmed が多数決と完全整合。
            "confirmed_majority_agree_rate": _build_confirmed_agree_rate(s),
        }
        for s in stats_list
    }


def _build_i1_summary(stats_list: list[VideoStats]) -> dict:
    """I1 メトリクスの全動画 worst-case サマリを返す。

    per_col_unknown_rate の worst col と worst video、
    non_stable_max_consecutive の max video、
    per_col_midgame_empty_rate の worst col を集計する。
    """
    worst_unknown: dict[int, float] = {col: 0.0 for col in range(6)}
    worst_unknown_vid: dict[int, str] = {col: "" for col in range(6)}
    max_non_stable: int = 0
    max_non_stable_vid: str = ""
    worst_midgame: dict[int, float] = {col: 0.0 for col in range(6)}
    worst_midgame_vid: dict[int, str] = {col: "" for col in range(6)}
    for s in stats_list:
        for col in range(6):
            stable = s.per_col_stable_cells.get(col, 0)
            if stable > 0:
                rate = s.per_col_unknown_cells.get(col, 0) / stable
                if rate > worst_unknown[col]:
                    worst_unknown[col] = rate
                    worst_unknown_vid[col] = s.video_id
        if s.non_stable_max_consecutive > max_non_stable:
            max_non_stable = s.non_stable_max_consecutive
            max_non_stable_vid = s.video_id
        for col in range(6):
            mid_cells = s.per_col_midgame_cells.get(col, 0)
            if mid_cells >= MIDGAME_COL_MIN_FRAMES:
                rate = s.per_col_midgame_empty_cells.get(col, 0) / mid_cells
                if rate > worst_midgame[col]:
                    worst_midgame[col] = rate
                    worst_midgame_vid[col] = s.video_id
    return {
        "per_col_unknown_worst": {
            str(col): {"rate": worst_unknown[col], "video": worst_unknown_vid[col]}
            for col in range(6)
        },
        "non_stable_max_consecutive": {
            "max": max_non_stable, "video": max_non_stable_vid
        },
        "per_col_midgame_empty_worst": {
            str(col): {"rate": worst_midgame[col], "video": worst_midgame_vid[col]}
            for col in range(6)
        },
        "thresholds": {
            "per_col_unknown_warning": PER_COL_UNKNOWN_WARNING,
            "per_col_unknown_critical": PER_COL_UNKNOWN_CRITICAL,
            "non_stable_critical_frames": NON_STABLE_CRITICAL_FRAMES,
            "midgame_col_empty_critical": MIDGAME_COL_EMPTY_CRITICAL,
            # 改修2: per-side avg_puyo CRITICAL 閾値を記録 (後方互換追加)
            "avg_puyo_count_critical": AVG_PUYO_COUNT_CRITICAL,
            # 改修3: chain 中 non-stable 除外フラグを記録 (後方互換追加)
            "non_stable_chain_exclude": NON_STABLE_CHAIN_EXCLUDE,
        },
    }


def _aggregate_stats(stats_list: list[VideoStats]) -> dict:
    """VideoStats リストから JSON 出力用 dict を生成する。"""
    total_cells = sum(s.total_cells for s in stats_list)
    correct = sum(s.agreed_cells for s in stats_list)
    overall_acc = correct / total_cells if total_cells > 0 else 0.0
    total_physics_fix = sum(s.physics_fix_count for s in stats_list)
    total_all_three = sum(s.all_three_agree_count for s in stats_list)
    return {
        "overall": {
            "acc": overall_acc,
            "total_cells": total_cells,
            "correct": correct,
            # 3 者独立追加メトリクス
            "physics_fix_count": total_physics_fix,
            "all_three_agree_count": total_all_three,
            "physics_fix_rate": (
                total_physics_fix / total_cells if total_cells > 0 else 0.0
            ),
            "all_three_agree_rate": (
                total_all_three / total_cells if total_cells > 0 else 0.0
            ),
        },
        "per_color": _build_color_acc(stats_list),
        "per_row": _build_row_acc(stats_list),
        "per_video": _build_video_acc(stats_list),
        # I1 メトリクス集計サマリ (全動画 worst-case)
        "i1_metrics_summary": _build_i1_summary(stats_list),
        # 後処理破壊検知 (postprocess_corruption)
        "postprocess_corruption": _aggregate_corruption(stats_list, total_cells),
    }


def _compute_holdout_summary(
    stats_list: list[VideoStats],
    holdout_ids: list[str],
) -> dict:
    """holdout 動画のみの集計結果を返す。"""
    ho_stats = [s for s in stats_list if s.video_id in holdout_ids]
    if not ho_stats:
        return {"acc": None, "videos": holdout_ids, "note": "holdout 動画なし"}
    total = sum(s.total_cells for s in ho_stats)
    correct = sum(s.agreed_cells for s in ho_stats)
    return {
        "acc": correct / total if total > 0 else 0.0,
        "total_cells": total,
        "correct": correct,
        "videos": holdout_ids,
    }


def _judge_pass_fail(
    overall_acc: float,
    per_color: dict[str, float],
    holdout_acc: Optional[float],
    stats_list: Optional[list] = None,
) -> tuple[str, list[str]]:
    """PASS/FAIL 判定と失敗理由リストを返す。

    I1 メトリクス (per_col_unknown_rate / non_stable_consecutive / per_col_midgame_empty)
    が NG ならも FAIL にする。stats_list=None なら従来通りの acc 判定のみ。
    backwards compat: stats_list は optional 引数。
    """
    target_acc = holdout_acc if holdout_acc is not None else overall_acc
    failures: list[str] = []

    if target_acc < PASS_OVERALL_THRESHOLD:
        failures.append(
            f"全マス平均 {target_acc:.4f} < 閾値 {PASS_OVERALL_THRESHOLD:.4f}"
        )

    for color_name, acc in per_color.items():
        if acc < PASS_PER_COLOR_THRESHOLD:
            failures.append(
                f"色別 {color_name}: {acc:.4f} < 閾値 {PASS_PER_COLOR_THRESHOLD:.4f}"
            )

    # I1 メトリクス判定
    if stats_list is not None:
        failures.extend(_judge_i1_metrics(stats_list))

    return ("PASS" if not failures else "FAIL"), failures


def _judge_pass_fail_with_corruption(
    overall_acc: float,
    per_color: dict[str, float],
    holdout_acc: Optional[float],
    stats_list: Optional[list] = None,
    corruption_section: Optional[dict] = None,
) -> tuple[str, list[str]]:
    """PASS/FAIL 判定に postprocess_corruption 判定を加えたラッパー。

    backwards compat: _judge_pass_fail の引数を踏襲しつつ corruption_section を追加。
    corruption_section=None なら corruption 判定をスキップする (従来挙動と同等)。

    _judge_corruption_metrics が返す "[WARNING]" プレフィックス付きメッセージは
    failures に追加するが FAIL 判定には使わない (情報提示のみ)。
    """
    verdict, failures = _judge_pass_fail(
        overall_acc, per_color, holdout_acc, stats_list
    )
    if corruption_section is not None:
        corruption_messages = _judge_corruption_metrics(corruption_section)
        # [WARNING] プレフィックス付きメッセージは FAIL 判定に使わず情報提示のみ
        fail_messages = [
            m for m in corruption_messages if not m.startswith("[WARNING]")
        ]
        warning_messages = [
            m for m in corruption_messages if m.startswith("[WARNING]")
        ]
        failures.extend(fail_messages)
        # WARNING は failures に追加するが verdict 計算前に済んでいない → 後で追加
        verdict = "PASS" if not failures else "FAIL"
        # WARNING を failures 末尾に追加 (verdict には影響しない)
        failures.extend(warning_messages)
    return verdict, failures


def _judge_i1_metrics(stats_list: list) -> list[str]:
    """I1 追加メトリクスの FAIL 判定を返す。

    per_col_unknown_rate / non_stable_max_consecutive / per_col_midgame_empty_rate
    の 3 メトリクスが閾値超なら FAIL 理由を追加する。
    mismatch/replace が fail-silent でも本メトリクスは発火する (= cycle 23/24 の反省)。
    """
    failures: list[str] = []
    for s in stats_list:
        # メトリクス 1: per_col_unknown_rate
        for col in range(6):
            stable = s.per_col_stable_cells.get(col, 0)
            if stable == 0:
                continue
            rate = s.per_col_unknown_cells.get(col, 0) / stable
            if rate >= PER_COL_UNKNOWN_CRITICAL:
                failures.append(
                    f"[{s.video_id}] col={col} UNKNOWN率 {rate:.1%} >= CRITICAL閾値 {PER_COL_UNKNOWN_CRITICAL:.0%}"
                    f" (fail-silent 対象: cycle 23/24 参照)"
                )
            elif rate >= PER_COL_UNKNOWN_WARNING:
                failures.append(
                    f"[{s.video_id}] col={col} UNKNOWN率 {rate:.1%} >= WARNING閾値 {PER_COL_UNKNOWN_WARNING:.0%}"
                )
        # メトリクス 2: non_stable_max_consecutive
        if s.non_stable_max_consecutive >= NON_STABLE_CRITICAL_FRAMES:
            failures.append(
                f"[{s.video_id}] non_stable 連続 {s.non_stable_max_consecutive} frames"
                f" >= CRITICAL閾値 {NON_STABLE_CRITICAL_FRAMES}"
            )
        # メトリクス 3: per_col_midgame_empty_rate
        for col in range(6):
            mid_cells = s.per_col_midgame_cells.get(col, 0)
            if mid_cells < MIDGAME_COL_MIN_FRAMES:
                continue
            empty_rate = s.per_col_midgame_empty_cells.get(col, 0) / mid_cells
            if empty_rate >= MIDGAME_COL_EMPTY_CRITICAL:
                failures.append(
                    f"[{s.video_id}] 中盤 col={col} EMPTY率 {empty_rate:.1%} >= CRITICAL閾値 {MIDGAME_COL_EMPTY_CRITICAL:.0%}"
                    f" (v40_match01 全EMPTY誤判定パターン)"
                )
        # 改修2: per-side avg_puyo_count CRITICAL 閾値チェック
        # AVG_PUYO_COUNT_CRITICAL 未満 = 列崩壊疑い (実測最小 18 以上が正常)
        for side in ("1P", "2P"):
            n = s._puyo_count_n_stable_by_side.get(side, 0)
            if n <= 0:
                continue
            avg = s._puyo_count_sum_by_side.get(side, 0) / n
            if avg < AVG_PUYO_COUNT_CRITICAL:
                failures.append(
                    f"[{s.video_id}] {side} avg_puyo_count {avg:.2f} < "
                    f"CRITICAL閾値 {AVG_PUYO_COUNT_CRITICAL:.1f}"
                    f" (列崩壊疑い: STABLE フレームのぷよ数が異常に少ない)"
                )
    return failures

def _aggregate_corruption_subcategory(
    stats_list: list["VideoStats"],
    total_stable_cells: int,
) -> dict:
    """corruption のサブカテゴリ別 count / rate を集計して返す。

    サブカテゴリ:
      empty_to_color: raw==EMPTY → confirmed==色 (空→色FP)
      color_to_color: raw==色A  → confirmed==色B (色フリーズ系)
      color_to_empty: raw==色   → confirmed==EMPTY (色→空 消失)

    Args:
        stats_list: 動画統計リスト。
        total_stable_cells: 分母 (STABLE 確定 cell 総数)。

    Returns:
        サブカテゴリ別 {count, rate} dict。後方互換のため既存フィールドとは別キー。
    """
    empty_to_color = sum(s.corruption_empty_to_color_count for s in stats_list)
    color_to_color = sum(s.corruption_color_to_color_count for s in stats_list)
    color_to_empty = sum(s.corruption_color_to_empty_count for s in stats_list)
    denom = total_stable_cells if total_stable_cells > 0 else 1

    def _rate(n: int) -> float:
        return n / denom

    return {
        "empty_to_color": {
            "count": empty_to_color,
            "rate": _rate(empty_to_color),
            "note": "infer_placement hallucination / constraint_fill deficit force-fill 主因",
        },
        "color_to_color": {
            "count": color_to_color,
            "rate": _rate(color_to_color),
            "note": "T2 自己強化フリーズ / constraint_fill replace 主因",
        },
        "color_to_empty": {
            "count": color_to_empty,
            "rate": _rate(color_to_empty),
            "note": "infer_placement commit refuse 後 T2 消去等",
        },
    }


def _aggregate_persistent_corruption(
    stats_list: list["VideoStats"],
    total_stable_cells: int,
) -> dict:
    """持続 corruption (N fr 以上連続) を集計して返す。

    per-frame 全件集計 (postprocess_corruption) とは独立した別集計で、
    1fr 点滅ノイズを除外した「真の持続誤認」のみを可視化する。
    既存フィールドには一切影響しない (後方互換追加)。

    Returns:
        count, rate, subcategory, persist_min_frames を含む dict。
    """
    total_persist = sum(s.postprocess_corruption_persistent_count for s in stats_list)
    denom = total_stable_cells if total_stable_cells > 0 else 1
    rate = total_persist / denom

    e2c = sum(s.corruption_persistent_empty_to_color_count for s in stats_list)
    c2c = sum(s.corruption_persistent_color_to_color_count for s in stats_list)
    c2e = sum(s.corruption_persistent_color_to_empty_count for s in stats_list)

    def _r(n: int) -> float:
        return n / denom

    return {
        "count": total_persist,
        "rate": rate,
        "persist_min_frames": CORRUPTION_PERSIST_MIN_FRAMES,
        "subcategory": {
            "empty_to_color": {"count": e2c, "rate": _r(e2c)},
            "color_to_color": {"count": c2c, "rate": _r(c2c)},
            "color_to_empty": {"count": c2e, "rate": _r(c2e)},
        },
        "note": (
            f"N>={CORRUPTION_PERSIST_MIN_FRAMES} fr 連続の corruption run のみ集計。"
            "1fr 点滅 (49.7%) 等の短期ノイズを除外した真の持続誤認を示す。"
        ),
    }


def _aggregate_corruption(
    stats_list: list["VideoStats"],
    total_stable_cells: int,
) -> dict:
    """全動画の postprocess_corruption 集計を返す。

    Args:
        stats_list: 動画統計リスト。
        total_stable_cells: 分母 (STABLE 確定 cell 総数)。

    Returns:
        count / rate / by_side / color_pairs / side_bias / corruption_ratio / log
        を含む dict。
    """
    total_corruption = sum(s.postprocess_corruption_count for s in stats_list)
    total_physics_fix = sum(s.physics_fix_count for s in stats_list)
    rate = total_corruption / total_stable_cells if total_stable_cells > 0 else 0.0

    # side 別集計
    by_side: dict[str, int] = defaultdict(int)
    for s in stats_list:
        for side, cnt in s.postprocess_corruption_by_side.items():
            by_side[side] += cnt

    # color_pair 集計 (JSON キーに変換)
    color_pairs: dict[str, int] = defaultdict(int)
    for s in stats_list:
        for (fc, tc), cnt in s.postprocess_corruption_color_pairs.items():
            key = f"{COLOR_NAMES.get(fc, str(fc))}->{COLOR_NAMES.get(tc, str(tc))}"
            color_pairs[key] += cnt

    # side_bias: 片側 & 1 色が POSTPROCESS_SIDE_BIAS_THRESHOLD 以上集中 & >=MIN セル
    side_bias = _detect_side_bias(stats_list, total_corruption)

    # corruption_ratio = corruption / (corruption + physics_fix)
    denom = total_corruption + total_physics_fix
    corruption_ratio = total_corruption / denom if denom > 0 else 0.0

    # ログ: 全動画分を CORRUPTION_LOG_LIMIT 件まで統合
    log: list[dict] = []
    for s in stats_list:
        log.extend(s.postprocess_corruption_log)
        if len(log) >= CORRUPTION_LOG_LIMIT:
            log = log[:CORRUPTION_LOG_LIMIT]
            break

    # サブカテゴリ別集計 (後方互換: 既存フィールドに追加)
    subcategory = _aggregate_corruption_subcategory(stats_list, total_stable_cells)

    # 持続 corruption 集計 (postprocess_corruption_persistent)
    # N サンプルフレーム以上連続した corruption run のみを別集計する。
    # 1fr 点滅 (49.7%) を除外して真の持続誤認を可視化する。
    persistent_section = _aggregate_persistent_corruption(stats_list, total_stable_cells)

    return {
        "count": total_corruption,
        "rate": rate,
        "by_side": dict(by_side),
        "color_pairs": dict(color_pairs),
        "side_bias": side_bias,
        "corruption_ratio": corruption_ratio,
        "log": log,
        # サブカテゴリ別 count / rate (後方互換: 追加フィールド)
        "subcategory": subcategory,
        # 持続 corruption 集計 (後方互換: 追加フィールド)
        "postprocess_corruption_persistent": persistent_section,
        # fail-silent 罠注記: raw_cnn==raw_hsv==誤りの全列崩壊型は本指標で検知不可
        "blind_spot_note": (
            "raw_cnn==raw_hsv==誤り の全列崩壊型は本指標では検知不可。"
            "per_col_midgame_empty (I1) / avg_puyo_count (C1) で別途確認・viz 目視必須。"
        ),
    }


def _detect_side_bias(
    stats_list: list["VideoStats"],
    total_corruption: int,
) -> dict:
    """side_bias: 片側 & 書換先 1 色が THRESHOLD 以上集中を検知する。

    Returns:
        {"detected": bool, "dominant_side": str|None, "dominant_color": str|None,
         "dominant_rate": float}
    """
    if total_corruption < POSTPROCESS_SIDE_BIAS_MIN_CELLS:
        return {
            "detected": False, "dominant_side": None,
            "dominant_color": None, "dominant_rate": 0.0,
        }

    # side 別 color_pair 集計
    side_color: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in stats_list:
        for (fc, tc), cnt in s.postprocess_corruption_color_pairs.items():
            # side 別は by_side から派生できないので per-stats の side 情報を利用
            pass
    # side_color は postprocess_corruption_by_side と color_pairs から近似判定
    # side 別合計が >= THRESHOLD * total_corruption かつ >= MIN_CELLS
    by_side_total: dict[str, int] = defaultdict(int)
    for s in stats_list:
        for side, cnt in s.postprocess_corruption_by_side.items():
            by_side_total[side] += cnt

    dominant_side = None
    dominant_rate = 0.0
    for side, cnt in by_side_total.items():
        if total_corruption > 0:
            r = cnt / total_corruption
            if r >= POSTPROCESS_SIDE_BIAS_THRESHOLD and cnt >= POSTPROCESS_SIDE_BIAS_MIN_CELLS:
                if r > dominant_rate:
                    dominant_rate = r
                    dominant_side = side

    # 書換先 1 色が >= THRESHOLD 集中を確認
    dominant_color = None
    if dominant_side is not None:
        color_pair_for_side: dict[str, int] = defaultdict(int)
        for s in stats_list:
            # side 別内訳は postprocess_corruption_log から再集計
            for entry in s.postprocess_corruption_log:
                if entry.get("side") == dominant_side:
                    color_pair_for_side[entry.get("confirmed", "")] += 1
        total_for_side = sum(color_pair_for_side.values())
        if total_for_side > 0:
            for color, cnt in color_pair_for_side.items():
                if cnt / total_for_side >= POSTPROCESS_SIDE_BIAS_THRESHOLD:
                    dominant_color = color
                    break

    return {
        "detected": dominant_side is not None,
        "dominant_side": dominant_side,
        "dominant_color": dominant_color,
        "dominant_rate": dominant_rate,
    }


def _judge_corruption_metrics(
    corruption_section: dict,
) -> list[str]:
    """postprocess_corruption セクションから FAIL / WARNING 判定を返す。

    FAIL 条件:
      - corruption_rate >= POSTPROCESS_CORRUPTION_REJECT_RATE
      - side_bias.detected == True

    WARNING 条件 (FAIL にはしない、情報提示のみ):
      - subcategory.empty_to_color.rate >= CORRUPTION_EMPTY_TO_COLOR_WARNING_RATE
        (空→色FP が多い = infer_placement hallucination / constraint_fill deficit 主因)
    """
    failures: list[str] = []
    rate = corruption_section.get("rate", 0.0)
    if rate >= POSTPROCESS_CORRUPTION_REJECT_RATE:
        failures.append(
            f"postprocess_corruption_rate {rate:.4%}"
            f" >= REJECT閾値 {POSTPROCESS_CORRUPTION_REJECT_RATE:.4%}"
            f" (constraint_fill が CNN・HSV 一致セルを破壊している可能性)"
        )
    side_bias = corruption_section.get("side_bias", {})
    if side_bias.get("detected", False):
        failures.append(
            f"postprocess_corruption side_bias 検知:"
            f" dominant_side={side_bias.get('dominant_side')}"
            f" dominant_color={side_bias.get('dominant_color')}"
            f" rate={side_bias.get('dominant_rate', 0.0):.1%}"
        )
    # サブカテゴリ WARNING: empty_to_color 高率 (FAIL にはしない)
    subcategory = corruption_section.get("subcategory", {})
    e2c_rate = subcategory.get("empty_to_color", {}).get("rate", 0.0)
    if e2c_rate >= CORRUPTION_EMPTY_TO_COLOR_WARNING_RATE:
        failures.append(
            f"[WARNING] corruption_empty_to_color_rate {e2c_rate:.4%}"
            f" >= WARNING閾値 {CORRUPTION_EMPTY_TO_COLOR_WARNING_RATE:.4%}"
            f" (空→色FP多発: infer_placement hallucination / constraint_fill deficit 主因。"
            f" FAIL にはしない = 情報提示)"
        )
    return failures


# ============================
# CLI
# ============================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STABLE 確定盤面 cell-level 正解率測定",
    )
    p.add_argument(
        "--videos", type=str, required=True,
        help="評価対象動画 ID リスト (カンマ区切り)。例: v89,v97,v29",
    )
    p.add_argument(
        "--holdout", type=str, default="",
        help="holdout 動画 ID リスト (カンマ区切り)。例: v89,v97",
    )
    p.add_argument(
        "--video-dir", type=Path, default=None,
        help="動画ファイル検索ルートディレクトリ。省略時は自動検索。",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="結果 JSON 出力パス。省略時は data/verify/stable_cell_acc/<timestamp>.json。",
    )
    p.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help="1 動画あたり最大処理フレーム数 (0=制限なし)。",
    )
    p.add_argument(
        "--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_SEC,
        help="認識処理間隔 (秒)。",
    )
    p.add_argument(
        "--constraint-fill",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_constraint_fill",
        help=(
            "NEXT 累積制約による色 count 補正 (constraint_fill) を制御する。 "
            "--constraint-fill で有効化、 --no-constraint-fill で無効化。 "
            "ライブラリ default=False (無効) に整合。 "
            "constraint_fill の net 効果測定: --constraint-fill で有効化して比較。 "
            "confirmed 経路 (CNN+物理推論 post-process) に効く。"
        ),
    )
    p.add_argument(
        "--t2-highconf-yield",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_t2_highconf_yield",
        help=(
            "T2 高確信 yield を制御する。 "
            "STABLE → STABLE 遷移時の prev_stable 上書き (T2) において、 "
            "CNN が現在の confirmed 色を支持しているセルはスキップする。 "
            "ライブラリ default=True (有効)。 "
            "--no-t2-highconf-yield で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--infer-empty-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_infer_empty_guard",
        help=(
            "infer_placement 空セル hallucination ガードを制御する。 "
            "pattern の非 diff セルが cnn_after で COLOR_EMPTY な候補をスキップし、 "
            "CNN が確信して空なセルへの NEXT 色書込 (hallucination) を防ぐ。 "
            "ライブラリ default=True (有効)。 "
            "--no-infer-empty-guard で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--game-event-chain-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_game_event_chain_exit",
        help=(
            "game-event ベース連鎖終了を制御する。 "
            "CHAIN 状態を timing hold だけでなく「次ツモ変化」または"
            "「連鎖側お邪魔降下」を検知するまで維持する。 "
            "安全弁として CHAIN_MAX_HOLD_SEC (5.0s) 超過で強制終了。 "
            "ライブラリ default=True (有効)。 "
            "--no-game-event-chain-exit で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--landing-color-fix",
        action="store_true",
        default=False,
        dest="enable_landing_color_fix",
        help=(
            "着地色修正 案1: TSUMO_FALL→STABLE 着地時の falling_pair を "
            "_landing_pending (消費済みツモ色) に切り替える。 "
            "slide_motion(R-7) 経由で 1 つ前のツモ色を指してしまう誤色問題の修正。 "
            "省略時は従来挙動 (prev_next_queue[-2] を使用)。"
        ),
    )
    p.add_argument(
        "--chain-min-display",
        action="store_true",
        default=False,
        dest="enable_chain_min_display",
        help=(
            "X1/X4 短連鎖ちらつき対策を有効化する。 "
            f"CHAIN 最小表示時間 (CHAIN_MIN_DISPLAY_SEC={RecognitionPipeline.CHAIN_MIN_DISPLAY_SEC}s) + "
            f"短連鎖 game-event exit 抑止 (chain_count < {RecognitionPipeline.CHAIN_GAME_EVENT_MIN_COUNT})。 "
            "enable_game_event_chain_exit と独立フラグ (効果分解のため)。 "
            "省略時は従来挙動 (game-event exit 無補正)。"
        ),
    )
    p.add_argument(
        "--hsv-classify-fallback",
        action="store_true",
        default=False,
        dest="enable_hsv_classify_fallback",
        help=(
            "HSV 分類 fallback を有効化する。 "
            "_classify_next_pair_by_hsv の 2 択強制確定を回避し、 "
            "両候補が拮抗・両候補とも遠い・低彩度 patch の場合は next_pair 素返しにする。 "
            "黄(H26)→赤(H7) 誤分類 (~900 件、 H 差 19) 発火点対策。 "
            "省略時は従来挙動 (2 択強制確定)。"
        ),
    )
    p.add_argument(
        "--landing-observed-color",
        action="store_true",
        default=False,
        dest="enable_landing_observed_color",
        help=(
            "真因 A 対処: 着地セルの CNN==HSV 一致色補正を有効化する。 "
            "TSUMO_FALL→STABLE 着地時に infer_placement 後の着地 2 cell で "
            "CNN 観測色と HSV-only 観測色が一致する場合は観測色を採用し、 "
            "falling_pair タイミングずれによる yellow→red 等の誤色を上流で断つ。 "
            "省略時は従来挙動 (infer_placement の結果をそのまま使用)。"
        ),
    )
    p.add_argument(
        "--red-hue-wrap-fix",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_red_hue_wrap_fix",
        help=(
            "赤色相折り返し補正を制御する。 "
            "赤ぷよの H 画素が 0-4 と 166-179 に 2 峰分布するため単純 median が "
            "赤/黄境界 (H=13/14) に乗り毎フレームちらつく問題を修正する。 "
            "ライブラリ default=True (有効)。 "
            "--no-red-hue-wrap-fix で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--specular-robust-saturation",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_specular_robust_saturation",
        help=(
            "案D: 光沢ハイライト除外彩度計算を制御する。 "
            "ぷよ表面の白ハイライト画素 (V>=" + str(210) + " かつ S<=" + str(60) + ") を "
            "彩度 median 計算から除外し、光沢球混入による EMPTY 誤判定を防ぐ。 "
            "ライブラリ default=True (有効)。 "
            "--no-specular-robust-saturation で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--stable-recovery-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_stable_recovery_gate",
        help=(
            "設計C 事後復旧ゲートを制御する。 "
            "STABLE 中に confirmed==EMPTY なのに CNN==HSV が同一有効色で "
            f"{8} フレーム継続したセルを confirmed に復旧する。 "
            "ライブラリ default=True (有効)。 "
            "--no-stable-recovery-gate で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--enable-ojama-visual-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_visual_detection",
        help=(
            "フェーズA: おじゃま視覚検知 (親フラグ) を制御する。 "
            "True にすると OjamaVisualDetector が BoardStateMachine に挿入され、 "
            "子フラグ (--enable-ojama-visual-chain-exit / --enable-ojama-settle-detection) も有効化。 "
            "ライブラリ default=True (有効)。 "
            "--no-enable-ojama-visual-detection で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--enable-ojama-visual-chain-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_visual_chain_exit",
        help=(
            "フェーズA: CHAIN → STABLE 復帰をお邪魔視覚検知に委譲する。 "
            "OjamaVisualDetector がお邪魔降下終了を検知したタイミングで CHAIN 終了判定する。 "
            "ライブラリ default=True (有効)。 "
            "--no-enable-ojama-visual-chain-exit で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--enable-ojama-infer-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_infer_guard",
        help=(
            "フェーズA: OJAMA_FALL → STABLE 直後の infer_placement を制御する。 "
            "ojama_tier1_warmup 期間中にツモが存在しないのに幽霊配置が走るのを防ぐ。 "
            "ライブラリ default=True (有効)。 "
            "--no-enable-ojama-infer-guard で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--enable-ojama-settle-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_settle_detection",
        help=(
            "フェーズA: OJAMA_FALL 中にお邪魔 count 不変フレームが続いたら STABLE 復帰する。 "
            "お邪魔落下完了後の長期 non-stable を短縮する。 "
            "ライブラリ default=True (有効)。 "
            "--no-enable-ojama-settle-detection で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--ojama-tier1-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_ojama_tier1_warmup",
        help=(
            "OJAMA 専用 tier1 warmup を制御する。 "
            "OJAMA_FALL → STABLE 遷移時に BG_FP tier1 を OJAMA_TIER1_WARMUP_FRAMES 間スキップし、 "
            "お邪魔消滅後の BG 距離による列崩壊 (v70 真因) を防ぐ。 "
            "ライブラリ default=True (有効)。 "
            "--no-ojama-tier1-warmup で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--chain-score-early-fire",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_chain_score_early_fire",
        help=(
            "機能B: score 急増 CHAIN 早期発火を制御する。 "
            f"True にすると自 side の score_delta >= {80} の frame で "
            "VideoChainTracker の puyo 減少検知を待たずに即 CHAIN state に突入する。 "
            "OCR 失敗 / score 取得不可時は従来の VideoChainTracker 経路を維持 (OR 追加)。 "
            "ライブラリ default=False (無効)。 "
            "--chain-score-early-fire で有効化、 --no-chain-score-early-fire で無効化。"
        ),
    )
    p.add_argument(
        "--chain-exit-warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_chain_exit_warmup",
        help=(
            "機能C: CHAIN → STABLE 遷移直後の confirmed 凍結を制御する。 "
            f"True にすると CHAIN→STABLE 復帰から {0.1}s 間 confirmed 更新を凍結し "
            "エフェクト残光色の混入を防ぐ。 時間ベース実装で fps 非依存。 "
            "ライブラリ default=False (無効)。 "
            "--chain-exit-warmup で有効化、 --no-chain-exit-warmup で無効化。"
        ),
    )
    p.add_argument(
        "--chain-formula-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_chain_formula_detection",
        help=(
            "機能D: 連鎖開始 掛け算式 検知を制御する。 "
            "True にすると score ROI の OCR が None (掛け算式表示で NCC conf 低下) かつ "
            "ink_ratio > CHAIN_FORMULA_INK_RATIO_MIN かつ last_score > 0 が "
            "CHAIN_FORMULA_CONSEC_FRAMES 連続で成立した frame で即 CHAIN state に突入する。 "
            "機能B (score 急増経路) と独立フラグ。 "
            "ライブラリ default=True (有効、 2026-06-03 採用)。 "
            "--no-chain-formula-detection で無効化 (旧挙動比較用)。"
        ),
    )
    p.add_argument(
        "--hsv-deferred-consensus",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="enable_hsv_deferred_consensus",
        help=(
            "案 Y-4: HSV-first commit + deferred consensus を制御する。 "
            "True にすると infer_placement が HSV 拮抗と判定した着地 2 候補を保留し、 "
            "後続 DEFERRED_MAX_FRAMES 内の CNN==HSV consensus 投票で確定させる。 "
            "ライブラリ default=False (無効)。 "
            "--hsv-deferred-consensus で有効化、 --no-hsv-deferred-consensus で無効化。"
        ),
    )
    # 不具合B 対処: 予告おじゃま発光ガード (2026-06-04)
    # store_true を使う (BooleanOptionalAction の --no- 接頭辞反転バグ回避)
    p.add_argument(
        "--ojama-warning-glow-guard",
        action="store_true",
        default=False,
        dest="enable_ojama_warning_glow_guard",
        help=(
            "不具合B 対処: 予告おじゃま発光ガードを有効化する。 "
            "相手連鎖の予告おじゃま演出による盤面上部多色発光を V_high_ratio で検知し、 "
            "STABLE 中の confirmed_board を frozen_board で保護する。 "
            "黄ぷよに発光が重なる黄(4)→おじゃま(9)誤認を防ぐ。 "
            "ライブラリ default=False (無効)。 --ojama-warning-glow-guard で有効化。"
        ),
    )
    p.add_argument(
        "--chain-max-hold-override",
        action="store_true",
        default=False,
        dest="enable_chain_max_hold_override",
        help=(
            "案P3: CHAIN_MAX_HOLD_SEC 超過後の ojama 保留を無効化する。 "
            "active_chain が CHAIN_MAX_HOLD_SEC 超過で強制クリアされた frame では "
            "ojama_top_positive による STABLE 復帰保留をスキップして強制 STABLE に遷移させる。 "
            "安全弁を本来機能させ連鎖過剰保持を解消する (v89 t34-40.87 の 6.87 秒超保持修正)。 "
            "ライブラリ default=False (無効)。 --chain-max-hold-override で有効化。"
        ),
    )
    # 案X*(A)(B)+warmup: NextSlide signal による CHAIN 即終了 (2026-06-05)
    # store_true を使う (BooleanOptionalAction の --no- 接頭辞反転バグ回避)
    p.add_argument(
        "--chain-exit-next-signal",
        action="store_true",
        default=False,
        dest="enable_chain_exit_next_signal",
        help=(
            "案X*: NextSlide signal による CHAIN 即終了を有効化する。 "
            "(A) 機能D 再点火抑制: 既に CHAIN 中なら 機能D (掛け算式) の発火をスキップし "
            "max_until 延長を止める。 "
            "(B) NextSlide signal (次ツモスライド) 検知で CHAIN を即終了させる。 "
            "warmup 連動: CHAIN_EXIT_WARMUP_SEC 秒間 confirmed 凍結を自動適用。 "
            "真因: ojama_top_positive 保留 + 機能D 再点火による 6.87 秒過剰保持 (v89 1P) を解消。 "
            "ライブラリ default=False (無効)。 --chain-exit-next-signal で有効化。"
        ),
    )
    # feat/gravity-settle-2026-06-05: 連鎖終了直後 GRAVITY_SETTLE 状態を有効化
    # 2026-06-06 採用: default=True。--no-gravity-settle-state で無効化可。
    p.add_argument(
        "--gravity-settle-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enable_gravity_settle_state",
        help=(
            "GRAVITY_SETTLE 状態を有効化する (feat/gravity-settle-2026-06-05)。 "
            "連鎖終了直後の重力 settle/着地中を採点外・confirmed 凍結として扱う。 "
            "CHAIN → GRAVITY_SETTLE → STABLE の遷移経路を有効化。 "
            "案X (--chain-exit-next-signal) との組み合わせを推奨 (内部で自動 ON)。 "
            "default=True (有効、2026-06-06 採用)。 --no-gravity-settle-state で無効化。"
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "動画単位の並列ワーカ数。1 (デフォルト) = 逐次実行 (backwards compat)。 "
            "2 以上で ProcessPoolExecutor (spawn) による並列処理を有効化。 "
            "推奨上限は CPU コア数 (GPU 使用率が低い場合に有効)。"
        ),
    )
    p.add_argument(
        "--corruption-persist-frames",
        type=int,
        default=CORRUPTION_PERSIST_MIN_FRAMES,
        dest="corruption_persist_frames",
        help=(
            "持続 corruption と見なす最小連続サンプルフレーム数。 "
            f"N フレーム以上連続した corruption run のみを postprocess_corruption_persistent に計上する。 "
            f"デフォルト {CORRUPTION_PERSIST_MIN_FRAMES} (1-2fr が 67.5%% のため 3fr 以上で点滅除外)。 "
            "既存 postprocess_corruption (per-frame 全件) は一切変更しない (後方互換)。"
        ),
    )
    p.add_argument(
        "--no-per-video-hsv",
        action="store_true",
        default=False,
        dest="disable_per_video_hsv",
        help=(
            "per-video 手調整 HSV inject をスキップする (完全自動 HSV 精度測定用)。 "
            "--no-per-video-hsv で inject を無効化し raw_cnn / raw_hsv / confirmed の "
            "全 3 軸を自動 HSV (OnlineHsvCalibrator) + 既定 merged レンジのみで動作させる。 "
            "3 者合意 metric が完全自動条件での内部整合率を測定する。 "
            "acc の手調整あり (99.87%%) との delta が手調整 HSV の寄与量を示す。 "
            "注意: CNN==HSV 両軸が同じ誤りに合意した場合 corruption に出ず "
            "acc が見かけ上保たれる fail-silent リスクが上がる。 "
            "viz 目視 + check_three_way_sudden_drop で補完すること。 "
            "デフォルト False = 従来挙動完全一致 (backwards compat)。"
        ),
    )
    return p.parse_args()



def _resolve_output_path(output: object) -> Path:
    """出力 JSON パスを決定する。"""
    if output is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
        return Path("data/verify/stable_cell_acc") / f"{ts}.json"
    return Path(output)


def _collect_results(
    video_ids: list[str],
    holdout_ids: list[str],
    video_dir: object,
    max_frames: int,
    sample_interval_sec: float,
    disagreements: list[dict],
    enable_constraint_fill: bool = True,
    workers: int = 1,
    enable_t2_highconf_yield: bool = False,
    enable_infer_empty_guard: bool = False,
    enable_game_event_chain_exit: bool = False,
    enable_landing_color_fix: bool = False,
    enable_chain_min_display: bool = False,
    enable_hsv_classify_fallback: bool = False,
    enable_landing_observed_color: bool = False,
    enable_red_hue_wrap_fix: bool = False,
    enable_specular_robust_saturation: bool = False,
    enable_stable_recovery_gate: bool = False,
    enable_ojama_visual_detection: bool = False,
    enable_ojama_visual_chain_exit: bool = False,
    enable_ojama_infer_guard: bool = False,
    enable_ojama_settle_detection: bool = False,
    enable_ojama_tier1_warmup: bool = False,
    enable_chain_score_early_fire: bool = False,
    enable_chain_exit_warmup: bool = False,
    enable_chain_formula_detection: bool = False,
    enable_hsv_deferred_consensus: bool = False,
    enable_ojama_warning_glow_guard: bool = False,
    enable_chain_max_hold_override: bool = False,
    # 案X*(A)(B)+warmup (2026-06-05): NextSlide signal による CHAIN 即終了。
    enable_chain_exit_next_signal: bool = False,
    # feat/gravity-settle-2026-06-05: 連鎖終了直後 GRAVITY_SETTLE 状態を有効化。
    enable_gravity_settle_state: bool = False,
    persist_min_frames: int = CORRUPTION_PERSIST_MIN_FRAMES,
    disable_per_video_hsv: bool = False,
) -> list[VideoStats]:
    """動画リストを走らせ VideoStats リストを返す。

    Args:
        enable_constraint_fill: False にすると confirmed 経路の
            constraint_fill を無効化して測定する。
            backwards compat: デフォルト True = 従来挙動。
        workers: 並列ワーカ数。1 (デフォルト) = 逐次実行 (backwards compat)。
            2 以上を指定すると ProcessPoolExecutor (spawn) で動画単位並列処理。
        enable_t2_highconf_yield: True にすると T2 の prev_stable 上書きを
            CNN 支持セルでスキップする。backwards compat: デフォルト False = 従来挙動。
        enable_infer_empty_guard: True にすると infer_placement 空セル
            hallucination ガードを有効化する。backwards compat: デフォルト False = 従来挙動。
        enable_game_event_chain_exit: True にすると game-event ベース連鎖終了を
            有効化する。backwards compat: デフォルト False = 従来挙動。
        enable_landing_color_fix: True にすると TSUMO_FALL→STABLE 着地時の
            falling_pair を _landing_pending (消費済みツモ色) に切り替える。
            backwards compat: デフォルト False = 従来挙動。
        enable_chain_min_display: True にすると X1/X4 短連鎖ちらつき対策を有効化。
            backwards compat: デフォルト False = 従来挙動。
        enable_hsv_classify_fallback: True にすると HSV 分類 fallback を有効化。
            _classify_next_pair_by_hsv の 2 択強制確定を回避する。
            backwards compat: デフォルト False = 従来挙動。
        enable_landing_observed_color: True にすると着地セルの CNN==HSV 一致色補正を有効化。
            backwards compat: デフォルト False = 従来挙動。
        enable_red_hue_wrap_fix: True にすると赤色相折り返し補正を有効化する。
            backwards compat: デフォルト False = 従来挙動。
        enable_specular_robust_saturation: True にすると光沢ハイライト除外彩度計算を有効化。
            backwards compat: デフォルト False = 従来挙動。
    """
    # 動画パスを事前解決 (並列化前に行うことでワーカに Path str を渡せる)
    video_tasks: list[tuple[str, Path]] = []
    for vid in video_ids:
        vpath = _resolve_video_path(vid, video_dir)
        if vpath is None:
            print(f"[measure] 動画ファイル未発見: {vid} → スキップ", file=sys.stderr)
            continue
        video_tasks.append((vid, vpath))

    if not video_tasks:
        return []

    effective_workers = min(workers, len(video_tasks))

    if effective_workers <= 1:
        return _collect_serial(
            video_tasks, holdout_ids, max_frames,
            sample_interval_sec, disagreements, enable_constraint_fill,
            enable_t2_highconf_yield=enable_t2_highconf_yield,
            enable_infer_empty_guard=enable_infer_empty_guard,
            enable_game_event_chain_exit=enable_game_event_chain_exit,
            enable_landing_color_fix=enable_landing_color_fix,
            enable_chain_min_display=enable_chain_min_display,
            enable_hsv_classify_fallback=enable_hsv_classify_fallback,
            enable_landing_observed_color=enable_landing_observed_color,
            enable_red_hue_wrap_fix=enable_red_hue_wrap_fix,
            enable_specular_robust_saturation=enable_specular_robust_saturation,
            enable_stable_recovery_gate=enable_stable_recovery_gate,
            enable_ojama_visual_detection=enable_ojama_visual_detection,
            enable_hsv_deferred_consensus=enable_hsv_deferred_consensus,
            enable_ojama_visual_chain_exit=enable_ojama_visual_chain_exit,
            enable_ojama_infer_guard=enable_ojama_infer_guard,
            enable_ojama_settle_detection=enable_ojama_settle_detection,
            enable_ojama_tier1_warmup=enable_ojama_tier1_warmup,
            enable_chain_score_early_fire=enable_chain_score_early_fire,
            enable_chain_exit_warmup=enable_chain_exit_warmup,
            enable_chain_formula_detection=enable_chain_formula_detection,
            enable_ojama_warning_glow_guard=enable_ojama_warning_glow_guard,
            enable_chain_max_hold_override=enable_chain_max_hold_override,
            enable_chain_exit_next_signal=enable_chain_exit_next_signal,
            enable_gravity_settle_state=enable_gravity_settle_state,
            persist_min_frames=persist_min_frames,
            disable_per_video_hsv=disable_per_video_hsv,
        )
    return _collect_parallel(
        video_tasks, holdout_ids, max_frames,
        sample_interval_sec, disagreements, enable_constraint_fill,
        effective_workers,
        enable_t2_highconf_yield=enable_t2_highconf_yield,
        enable_infer_empty_guard=enable_infer_empty_guard,
        enable_game_event_chain_exit=enable_game_event_chain_exit,
        enable_landing_color_fix=enable_landing_color_fix,
        enable_chain_min_display=enable_chain_min_display,
        enable_hsv_classify_fallback=enable_hsv_classify_fallback,
        enable_landing_observed_color=enable_landing_observed_color,
        enable_red_hue_wrap_fix=enable_red_hue_wrap_fix,
        enable_specular_robust_saturation=enable_specular_robust_saturation,
        enable_stable_recovery_gate=enable_stable_recovery_gate,
        enable_ojama_visual_detection=enable_ojama_visual_detection,
        enable_ojama_visual_chain_exit=enable_ojama_visual_chain_exit,
        enable_ojama_infer_guard=enable_ojama_infer_guard,
        enable_ojama_settle_detection=enable_ojama_settle_detection,
        enable_ojama_tier1_warmup=enable_ojama_tier1_warmup,
        enable_chain_score_early_fire=enable_chain_score_early_fire,
        enable_chain_exit_warmup=enable_chain_exit_warmup,
        enable_chain_formula_detection=enable_chain_formula_detection,
        enable_hsv_deferred_consensus=enable_hsv_deferred_consensus,
        enable_ojama_warning_glow_guard=enable_ojama_warning_glow_guard,
        enable_chain_max_hold_override=enable_chain_max_hold_override,
        enable_chain_exit_next_signal=enable_chain_exit_next_signal,
        enable_gravity_settle_state=enable_gravity_settle_state,
        persist_min_frames=persist_min_frames,
        disable_per_video_hsv=disable_per_video_hsv,
    )


def _collect_serial(
    video_tasks: list[tuple[str, Path]],
    holdout_ids: list[str],
    max_frames: int,
    sample_interval_sec: float,
    disagreements: list[dict],
    enable_constraint_fill: bool,
    enable_t2_highconf_yield: bool = False,
    enable_infer_empty_guard: bool = False,
    enable_game_event_chain_exit: bool = False,
    enable_landing_color_fix: bool = False,
    enable_chain_min_display: bool = False,
    enable_hsv_classify_fallback: bool = False,
    enable_landing_observed_color: bool = False,
    enable_red_hue_wrap_fix: bool = False,
    enable_specular_robust_saturation: bool = False,
    enable_stable_recovery_gate: bool = False,
    enable_ojama_visual_detection: bool = False,
    enable_ojama_visual_chain_exit: bool = False,
    enable_ojama_infer_guard: bool = False,
    enable_ojama_settle_detection: bool = False,
    enable_ojama_tier1_warmup: bool = False,
    enable_chain_score_early_fire: bool = False,
    enable_chain_exit_warmup: bool = False,
    enable_chain_formula_detection: bool = False,
    enable_hsv_deferred_consensus: bool = False,
    enable_ojama_warning_glow_guard: bool = False,
    enable_chain_max_hold_override: bool = False,
    # 案X*(A)(B)+warmup (2026-06-05): NextSlide signal による CHAIN 即終了。
    enable_chain_exit_next_signal: bool = False,
    # feat/gravity-settle-2026-06-05: 連鎖終了直後 GRAVITY_SETTLE 状態を有効化。
    enable_gravity_settle_state: bool = False,
    persist_min_frames: int = CORRUPTION_PERSIST_MIN_FRAMES,
    disable_per_video_hsv: bool = False,
) -> list[VideoStats]:
    """逐次実行で VideoStats リストを返す (workers=1 の従来挙動)。"""
    stats_list: list[VideoStats] = []
    for vid, vpath in video_tasks:
        vstats = _process_video(
            video_id=vid,
            video_path=vpath,
            is_holdout=(vid in holdout_ids),
            max_frames=max_frames,
            sample_interval_sec=sample_interval_sec,
            disagreements=disagreements,
            enable_constraint_fill=enable_constraint_fill,
            enable_t2_highconf_yield=enable_t2_highconf_yield,
            enable_infer_empty_guard=enable_infer_empty_guard,
            enable_game_event_chain_exit=enable_game_event_chain_exit,
            enable_landing_color_fix=enable_landing_color_fix,
            enable_chain_min_display=enable_chain_min_display,
            enable_hsv_classify_fallback=enable_hsv_classify_fallback,
            enable_landing_observed_color=enable_landing_observed_color,
            enable_red_hue_wrap_fix=enable_red_hue_wrap_fix,
            enable_specular_robust_saturation=enable_specular_robust_saturation,
            enable_stable_recovery_gate=enable_stable_recovery_gate,
            enable_ojama_visual_detection=enable_ojama_visual_detection,
            enable_ojama_visual_chain_exit=enable_ojama_visual_chain_exit,
            enable_ojama_infer_guard=enable_ojama_infer_guard,
            enable_ojama_settle_detection=enable_ojama_settle_detection,
            enable_ojama_tier1_warmup=enable_ojama_tier1_warmup,
            enable_chain_score_early_fire=enable_chain_score_early_fire,
            enable_chain_exit_warmup=enable_chain_exit_warmup,
            enable_chain_formula_detection=enable_chain_formula_detection,
            enable_hsv_deferred_consensus=enable_hsv_deferred_consensus,
            enable_ojama_warning_glow_guard=enable_ojama_warning_glow_guard,
            enable_chain_max_hold_override=enable_chain_max_hold_override,
            enable_chain_exit_next_signal=enable_chain_exit_next_signal,
            enable_gravity_settle_state=enable_gravity_settle_state,
            persist_min_frames=persist_min_frames,
            disable_per_video_hsv=disable_per_video_hsv,
        )
        stats_list.append(vstats)
    return stats_list


def _collect_parallel(
    video_tasks: list[tuple[str, Path]],
    holdout_ids: list[str],
    max_frames: int,
    sample_interval_sec: float,
    disagreements: list[dict],
    enable_constraint_fill: bool,
    workers: int,
    enable_t2_highconf_yield: bool = False,
    enable_infer_empty_guard: bool = False,
    enable_game_event_chain_exit: bool = False,
    enable_landing_color_fix: bool = False,
    enable_chain_min_display: bool = False,
    enable_hsv_classify_fallback: bool = False,
    enable_landing_observed_color: bool = False,
    enable_red_hue_wrap_fix: bool = False,
    enable_specular_robust_saturation: bool = False,
    enable_stable_recovery_gate: bool = False,
    enable_ojama_visual_detection: bool = False,
    enable_ojama_visual_chain_exit: bool = False,
    enable_ojama_infer_guard: bool = False,
    enable_ojama_settle_detection: bool = False,
    enable_ojama_tier1_warmup: bool = False,
    enable_chain_score_early_fire: bool = False,
    enable_chain_exit_warmup: bool = False,
    enable_chain_formula_detection: bool = False,
    enable_hsv_deferred_consensus: bool = False,
    enable_ojama_warning_glow_guard: bool = False,
    enable_chain_max_hold_override: bool = False,
    # 案X*(A)(B)+warmup (2026-06-05): NextSlide signal による CHAIN 即終了。
    enable_chain_exit_next_signal: bool = False,
    # feat/gravity-settle-2026-06-05: 連鎖終了直後 GRAVITY_SETTLE 状態を有効化。
    enable_gravity_settle_state: bool = False,
    persist_min_frames: int = CORRUPTION_PERSIST_MIN_FRAMES,
    disable_per_video_hsv: bool = False,
) -> list[VideoStats]:
    """ProcessPoolExecutor (spawn) で動画単位並列処理し VideoStats リストを返す。

    各ワーカは _process_video_worker 内で load_default() する (CUDA spawn 安全)。
    不一致 cell は stats._local_disagreements 経由で親プロセスに返却し、
    ここで disagreements リストに統合する。
    """
    # spawn コンテキストを明示 (fork だと CUDA が壊れる)
    mp_ctx = _mp.get_context("spawn")
    futures: dict = {}
    stats_list: list[VideoStats] = []

    print(f"[measure] 並列モード: workers={workers} 動画数={len(video_tasks)}")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=mp_ctx
    ) as executor:
        for vid, vpath in video_tasks:
            fut = executor.submit(
                _process_video_worker,
                vid,
                str(vpath),
                (vid in holdout_ids),
                max_frames,
                sample_interval_sec,
                enable_constraint_fill,
                enable_t2_highconf_yield,
                enable_infer_empty_guard,
                enable_game_event_chain_exit,
                enable_landing_color_fix,
                enable_chain_min_display,
                enable_hsv_classify_fallback,
                enable_landing_observed_color,
                enable_red_hue_wrap_fix,
                enable_specular_robust_saturation,
                enable_stable_recovery_gate,
                enable_ojama_visual_detection,
                enable_ojama_visual_chain_exit,
                enable_ojama_infer_guard,
                enable_ojama_settle_detection,
                enable_ojama_tier1_warmup,
                enable_chain_score_early_fire,
                enable_chain_exit_warmup,
                enable_chain_formula_detection,
                enable_hsv_deferred_consensus,
                enable_ojama_warning_glow_guard,
                enable_chain_max_hold_override,
                enable_chain_exit_next_signal,
                enable_gravity_settle_state,
                persist_min_frames,
                disable_per_video_hsv,
            )
            futures[fut] = vid

        for fut in concurrent.futures.as_completed(futures):
            vid = futures[fut]
            try:
                vstats = fut.result()
                # ワーカが収集した不一致 cell を親リストへ統合
                disagreements.extend(vstats._local_disagreements)
                vstats._local_disagreements = []  # メモリ節約
                stats_list.append(vstats)
            except Exception as exc:
                print(
                    f"[measure] {vid} 並列処理エラー: {exc!r}",
                    file=sys.stderr,
                )

    # 動画 ID 順でソートして決定論的な出力順を保つ
    id_order = {vid: i for i, (vid, _) in enumerate(video_tasks)}
    stats_list.sort(key=lambda s: id_order.get(s.video_id, 9999))
    return stats_list


def _print_summary(
    agg: dict,
    holdout_summary: dict,
    holdout_acc: object,
    holdout_ids: list,
    failures: list,
    n_disagreements: int,
    output_path: Path,
) -> None:
    """評価結果のサマリを標準出力に表示する。"""
    sep = "=" * 60
    verdict = "PASS" if not failures else "FAIL"
    print("")
    print(sep)
    print("判定: " + verdict)
    ov = agg["overall"]
    acc_str = "{:.4f} ({}/{})" .format(ov["acc"], ov["correct"], ov["total_cells"])
    print("全マス平均合意率: " + acc_str)
    if holdout_ids and holdout_acc is not None:
        ho_str = "{:.4f} ({}/{})".format(
            holdout_acc,
            holdout_summary.get("correct", 0),
            holdout_summary.get("total_cells", 0),
        )
        print("holdout 合意率:   " + ho_str)
    print("[色別合意率]")
    for cname, acc in sorted(agg["per_color"].items()):
        mark = "OK" if acc >= PASS_PER_COLOR_THRESHOLD else "NG"
        print("  {:8s}: {:.4f}  [{}]".format(cname, acc, mark))
    print("[不一致 cell 総数]: " + str(n_disagreements))
    print("[3 者独立メトリクス]")
    print("  全員一致率:         {:.4f} ({}/{})".format(
        ov.get("all_three_agree_rate", 0.0),
        ov.get("all_three_agree_count", 0),
        ov.get("total_cells", 0),
    ))
    print("  物理推論修正率:     {:.4f} ({}/{})".format(
        ov.get("physics_fix_rate", 0.0),
        ov.get("physics_fix_count", 0),
        ov.get("total_cells", 0),
    ))
    # I1 メトリクスサマリ出力
    per_vid = agg.get("per_video", {})
    has_i1 = any(
        "per_col_unknown_rate" in v for v in per_vid.values()
    )
    if has_i1:
        print("[I1 メトリクス: per_col_unknown_rate (STABLE 中 col 別 UNKNOWN 率)]")
        for vid_id, vid_data in per_vid.items():
            rates = vid_data.get("per_col_unknown_rate", {})
            for col_key, rate in sorted(rates.items()):
                if rate >= PER_COL_UNKNOWN_WARNING:
                    mark = "CRITICAL" if rate >= PER_COL_UNKNOWN_CRITICAL else "WARNING"
                    print(f"  [{vid_id}] col={col_key}: {rate:.1%}  [{mark}]")
        print("[I1 メトリクス: non_stable_max_consecutive (最長連続 non-STABLE フレーム数)]")
        for vid_id, vid_data in per_vid.items():
            n = vid_data.get("non_stable_max_consecutive", 0)
            mark = "CRITICAL" if n >= NON_STABLE_CRITICAL_FRAMES else "ok"
            if n > 0:
                print(f"  [{vid_id}] max={n}  [{mark}]")
        print("[I1 メトリクス: per_col_midgame_empty_rate (中盤 col 別 EMPTY 率)]")
        for vid_id, vid_data in per_vid.items():
            rates = vid_data.get("per_col_midgame_empty_rate", {})
            for col_key, rate in sorted(rates.items()):
                if rate is not None and rate >= MIDGAME_COL_EMPTY_CRITICAL:
                    print(f"  [{vid_id}] col={col_key}: {rate:.1%}  [CRITICAL]")
    # C1: avg_puyo_count_per_stable_frame 出力 (= fail-silent 経路検知)
    has_avg = any(
        v.get("avg_puyo_count_per_stable_frame") is not None
        for v in per_vid.values()
    )
    if has_avg:
        print("[C1 avg_puyo_count_per_stable_frame (STABLE フレームの 1P or 2P 平均ぷよ数)]")
        for vid_id, vid_data in per_vid.items():
            avg = vid_data.get("avg_puyo_count_per_stable_frame")
            n_st = vid_data.get("n_stable_frames_puyo", 0)
            if avg is not None:
                print(f"  [{vid_id}] avg={avg:.2f} (n_stable={n_st})")
                # 改修2: per-side avg_puyo 表示 (CRITICAL 閾値以下は [CRITICAL] 表示)
                per_side = vid_data.get("avg_puyo_count_per_side", {})
                for side_key, side_avg in sorted(per_side.items()):
                    if side_avg is None:
                        continue
                    mark = (
                        "CRITICAL"
                        if side_avg < AVG_PUYO_COUNT_CRITICAL
                        else "ok"
                    )
                    print(f"    {side_key}: avg={side_avg:.2f}  [{mark}]")
    # 改修1: confirmed_majority_agree_rate 表示 (情報提示のみ)
    has_cmar = any(
        "confirmed_majority_agree_rate" in v for v in per_vid.values()
    )
    if has_cmar:
        print(
            "[confirmed_majority_agree_rate "
            "(confirmed vs 多数決 一致率: 1.0=完全整合、情報提示のみ)]"
        )
        for vid_id, vid_data in per_vid.items():
            cmar = vid_data.get("confirmed_majority_agree_rate", {})
            overall_cmar = cmar.get("overall") if isinstance(cmar, dict) else None
            if overall_cmar is not None:
                print(f"  [{vid_id}] overall={overall_cmar:.4f}")
    if failures:
        print("[FAIL 理由]")
        for reason in failures:
            print("  - " + reason)
    print("[結果 JSON]: " + str(output_path))
    print(sep)
def main() -> int:
    """PASS なら 0, FAIL なら 1 を返す。"""
    args = _parse_args()
    video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
    holdout_ids = (
        [v.strip() for v in args.holdout.split(",") if v.strip()]
        if args.holdout else []
    )
    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 各フラグを args から直接取得する (BooleanOptionalAction で default=ライブラリ default と整合済)
    enable_constraint_fill: bool = bool(args.enable_constraint_fill)
    enable_t2_highconf_yield: bool = bool(args.enable_t2_highconf_yield)
    enable_infer_empty_guard: bool = bool(args.enable_infer_empty_guard)
    enable_game_event_chain_exit: bool = bool(args.enable_game_event_chain_exit)
    # landing_color_fix / chain_min_display / hsv_classify_fallback /
    # landing_observed_color はライブラリ default=False のまま (store_true 維持)
    enable_landing_color_fix: bool = bool(
        getattr(args, "enable_landing_color_fix", False)
    )
    enable_chain_min_display: bool = bool(
        getattr(args, "enable_chain_min_display", False)
    )
    enable_hsv_classify_fallback: bool = bool(
        getattr(args, "enable_hsv_classify_fallback", False)
    )
    enable_landing_observed_color: bool = bool(
        getattr(args, "enable_landing_observed_color", False)
    )
    enable_red_hue_wrap_fix: bool = bool(args.enable_red_hue_wrap_fix)
    enable_specular_robust_saturation: bool = bool(args.enable_specular_robust_saturation)
    enable_stable_recovery_gate: bool = bool(args.enable_stable_recovery_gate)
    enable_ojama_visual_detection: bool = bool(args.enable_ojama_visual_detection)
    enable_ojama_visual_chain_exit: bool = bool(args.enable_ojama_visual_chain_exit)
    enable_ojama_infer_guard: bool = bool(args.enable_ojama_infer_guard)
    enable_ojama_settle_detection: bool = bool(args.enable_ojama_settle_detection)
    enable_ojama_tier1_warmup: bool = bool(args.enable_ojama_tier1_warmup)
    enable_chain_score_early_fire: bool = bool(
        getattr(args, "enable_chain_score_early_fire", False)
    )
    enable_chain_exit_warmup: bool = bool(
        getattr(args, "enable_chain_exit_warmup", False)
    )
    enable_chain_formula_detection: bool = bool(
        getattr(args, "enable_chain_formula_detection", False)
    )
    enable_hsv_deferred_consensus: bool = bool(
        getattr(args, "enable_hsv_deferred_consensus", False)
    )
    enable_ojama_warning_glow_guard: bool = bool(
        getattr(args, "enable_ojama_warning_glow_guard", False)
    )
    enable_chain_max_hold_override: bool = bool(
        getattr(args, "enable_chain_max_hold_override", False)
    )
    enable_chain_exit_next_signal: bool = bool(
        getattr(args, "enable_chain_exit_next_signal", False)
    )
    # feat/gravity-settle-2026-06-05 (2026-06-06 採用: default=True)
    enable_gravity_settle_state: bool = bool(
        getattr(args, "enable_gravity_settle_state", True)
    )
    workers: int = max(1, args.workers)
    # --corruption-persist-frames: 1 以上であることを保証する
    persist_min_frames: int = max(1, int(getattr(args, "corruption_persist_frames", CORRUPTION_PERSIST_MIN_FRAMES)))
    # per-video 手調整 HSV inject 無効化フラグ (汎用精度測定用)
    disable_per_video_hsv: bool = bool(getattr(args, "disable_per_video_hsv", False))
    print(f"[measure] 評価開始: videos={video_ids} holdout={holdout_ids} workers={workers}")
    print(f"[measure] 出力先: {output_path}")
    print(f"[measure] constraint_fill={'ENABLED' if enable_constraint_fill else 'DISABLED'}")
    print(f"[measure] t2_highconf_yield={'ENABLED' if enable_t2_highconf_yield else 'DISABLED'}")
    print(f"[measure] infer_empty_guard={'ENABLED' if enable_infer_empty_guard else 'DISABLED'}")
    print(f"[measure] red_hue_wrap_fix={'ENABLED' if enable_red_hue_wrap_fix else 'DISABLED'}")
    print(f"[measure] specular_robust_saturation={'ENABLED' if enable_specular_robust_saturation else 'DISABLED'}")
    print(f"[measure] stable_recovery_gate={'ENABLED' if enable_stable_recovery_gate else 'DISABLED'}")
    print(f"[measure] ojama_visual_detection={'ENABLED' if enable_ojama_visual_detection else 'DISABLED'}")
    print(f"[measure] ojama_tier1_warmup={'ENABLED' if enable_ojama_tier1_warmup else 'DISABLED'}")
    if enable_game_event_chain_exit:
        print(
            "[measure] game_event_chain_exit ENABLED "
            "(--game-event-chain-exit 指定: game-event ベース連鎖終了)"
        )
    if enable_landing_color_fix:
        print(
            "[measure] landing_color_fix ENABLED "
            "(--landing-color-fix 指定: 着地色修正 案1 / falling_pair を _landing_pending に切り替え)"
        )
    if enable_chain_min_display:
        print(
            "[measure] chain_min_display ENABLED "
            f"(--chain-min-display 指定: X1 最小{RecognitionPipeline.CHAIN_MIN_DISPLAY_SEC}s + "
            f"X4 短連鎖 count<{RecognitionPipeline.CHAIN_GAME_EVENT_MIN_COUNT} exit 抑止)"
        )
    if enable_hsv_classify_fallback:
        print(
            "[measure] hsv_classify_fallback ENABLED "
            "(--hsv-classify-fallback 指定: 2 択強制確定回避 / 黄→赤誤分類発火点対策)"
        )
    if enable_landing_observed_color:
        print(
            "[measure] landing_observed_color ENABLED "
            "(--landing-observed-color 指定: 真因 A 対処 / CNN==HSV 一致色で着地補正)"
        )
    disagreements: list[dict] = []
    print(f"[measure] corruption_persist_frames={persist_min_frames} (persistent corruption 集計閾値)")
    # per-video HSV inject の状態をログ表示
    if disable_per_video_hsv:
        print(
            "[measure] disable_per_video_hsv=ON "
            "(手調整 per-video HSV inject スキップ: 自動 HSV + merged レンジのみで評価)"
        )
    stats_list = _collect_results(
        video_ids, holdout_ids, args.video_dir,
        args.max_frames, args.sample_interval, disagreements,
        enable_constraint_fill=enable_constraint_fill,
        workers=workers,
        enable_t2_highconf_yield=enable_t2_highconf_yield,
        enable_infer_empty_guard=enable_infer_empty_guard,
        enable_game_event_chain_exit=enable_game_event_chain_exit,
        enable_landing_color_fix=enable_landing_color_fix,
        enable_chain_min_display=enable_chain_min_display,
        enable_hsv_classify_fallback=enable_hsv_classify_fallback,
        enable_landing_observed_color=enable_landing_observed_color,
        enable_red_hue_wrap_fix=enable_red_hue_wrap_fix,
        enable_specular_robust_saturation=enable_specular_robust_saturation,
        enable_stable_recovery_gate=enable_stable_recovery_gate,
        enable_ojama_visual_detection=enable_ojama_visual_detection,
        enable_ojama_visual_chain_exit=enable_ojama_visual_chain_exit,
        enable_ojama_infer_guard=enable_ojama_infer_guard,
        enable_ojama_settle_detection=enable_ojama_settle_detection,
        enable_ojama_tier1_warmup=enable_ojama_tier1_warmup,
        enable_chain_score_early_fire=enable_chain_score_early_fire,
        enable_chain_exit_warmup=enable_chain_exit_warmup,
        enable_chain_formula_detection=enable_chain_formula_detection,
        enable_hsv_deferred_consensus=enable_hsv_deferred_consensus,
        enable_ojama_warning_glow_guard=enable_ojama_warning_glow_guard,
        enable_chain_max_hold_override=enable_chain_max_hold_override,
        enable_chain_exit_next_signal=enable_chain_exit_next_signal,
        persist_min_frames=persist_min_frames,
        disable_per_video_hsv=disable_per_video_hsv,
    )
    if not stats_list:
        print("[measure] 処理した動画がゼロ件。終了。", file=sys.stderr)
        return 2
    agg = _aggregate_stats(stats_list)
    holdout_summary = _compute_holdout_summary(stats_list, holdout_ids)
    holdout_acc = holdout_summary.get("acc") if holdout_ids else None
    corruption_section = agg.get("postprocess_corruption")
    verdict, failures = _judge_pass_fail_with_corruption(
        overall_acc=agg["overall"]["acc"],
        per_color=agg["per_color"],
        holdout_acc=holdout_acc,
        stats_list=stats_list,
        corruption_section=corruption_section,
    )
    # constraint_fill 無効時の注記: 本指標は 0 になりやすい
    postprocess_note: Optional[str] = None
    if not enable_constraint_fill:
        postprocess_note = (
            "constraint_fill 無効中のため本指標は 0 になりやすい。"
            "全列崩壊型は per_col_midgame_empty/avg_puyo_count で別途確認、viz 目視必須"
        )
    result = {
        **agg,
        "holdout_summary": holdout_summary,
        "disagreement_cells": disagreements[:DISAGREEMENT_OUTPUT_LIMIT],
        "disagreement_total": len(disagreements),
        "verdict": verdict, "failures": failures,
        "meta": {
            "videos": video_ids, "holdout": holdout_ids,
            "max_frames": args.max_frames,
            "sample_interval_sec": args.sample_interval,
            "pass_overall_threshold": PASS_OVERALL_THRESHOLD,
            "pass_per_color_threshold": PASS_PER_COLOR_THRESHOLD,
            # constraint_fill の on/off を記録 (後日比較用)
            "enable_constraint_fill": enable_constraint_fill,
            # t2_highconf_yield の on/off を記録 (後日比較用)
            "enable_t2_highconf_yield": enable_t2_highconf_yield,
            # game_event_chain_exit の on/off を記録 (後日比較用)
            "enable_game_event_chain_exit": enable_game_event_chain_exit,
            # landing_color_fix の on/off を記録 (後日比較用)
            "enable_landing_color_fix": enable_landing_color_fix,
            # chain_min_display の on/off を記録 (後日比較用)
            "enable_chain_min_display": enable_chain_min_display,
            # landing_observed_color の on/off を記録 (後日比較用)
            "enable_landing_observed_color": enable_landing_observed_color,
            # 並列ワーカ数を記録 (後日比較用)
            "workers": workers,
            # 改修3: non_stable chain 除外フラグ (後日比較用)
            "non_stable_chain_exclude": NON_STABLE_CHAIN_EXCLUDE,
            # フェーズA おじゃま視覚検知フラグ群 (後日比較用)
            "enable_ojama_visual_detection": enable_ojama_visual_detection,
            "enable_ojama_visual_chain_exit": enable_ojama_visual_chain_exit,
            "enable_ojama_infer_guard": enable_ojama_infer_guard,
            "enable_ojama_settle_detection": enable_ojama_settle_detection,
            "enable_ojama_tier1_warmup": enable_ojama_tier1_warmup,
            # 持続 corruption 集計閾値 (後日比較用)
            "corruption_persist_frames": persist_min_frames,
            # per-video 手調整 HSV inject 無効化フラグ (汎用精度測定用)
            # True = inject スキップ = 「自動 HSV + merged レンジのみ」での評価
            # False = inject 有効 = 従来挙動と完全一致 (backwards compat)
            "disable_per_video_hsv": disable_per_video_hsv,
        },
    }
    # constraint_fill 無効時の postprocess_corruption_note を追加
    if postprocess_note is not None:
        result["postprocess_corruption_note"] = postprocess_note
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _print_summary(agg, holdout_summary, holdout_acc, holdout_ids,
                   failures, len(disagreements), output_path)
    return 0 if verdict == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
