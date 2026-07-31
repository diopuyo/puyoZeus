"""統合 RecognitionPipeline (Phase B-7a).

B-1〜B-6 のコンポーネントを連結し、1 frame 投入で 1P/2P 両側の
state + 確定盤面 + drift 情報を返す。

Pipeline 構成:
    frame
      ├─ ImageReader.read_both_boards → (cnn_1p, cnn_2p)
      ├─ MatchStateDetector → is_match_active
      ├─ VideoChainTracker × 2     → ChainEvent (確定時のみ)
      ├─ ScoreTracker × 2          → ScoreDelta
      └─ side ごとに:
          ├─ BoardStateMachine.update(...)
          ├─ InferenceBoardGenerator.generate(...)
          └─ DriftDetector.update(inferred, cnn)
              └─ needs_resync なら state machine を reset + drift カウンタ reset

新方針 (project_recognition_strategy_pivot) の中核 pipeline。
state == STABLE 時の confirmed_board が「公式の確定盤面」として下流に流れる。
"""

from __future__ import annotations

import os
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN,
    HIDDEN_ROWS, Board,
)
from src.hidden_row_inferrer import infer_hidden_row
from src.probabilistic_board import ProbabilisticBoard
from src.board_state_machine import (
    BoardState,
    BoardStateMachine,
    DEFAULT_INITIAL_CONFIRM_MIN_VOTES,
    DetectorSignals,
    NON_STABLE_STATES,
    STABLE_RECOVERY_ADD_MIN_FRAMES,
    StateContext,
)

# W-α (Phase G C-1): STABLE 確定時に prob_board を埋めるかのフラグ。
# False にすると ProbabilisticBoard を計算せず prob_board=None のままになる。
# パフォーマンス上の逃げ道として用意するが、運用上は常に True で問題ない
# (隠し段推論は数十マイクロ秒オーダー)。
PROB_BOARD_PUBLISH_ON_STABLE: bool = True

# W-α: prob_board が None のときに ProbabilisticBoard.from_board(confirmed)
# でフォールバック生成して indicator 計算を継続するか。下流 (phase_e_collect)
# 側で扱う方が pipeline の単純さを保てるため、本モジュールでは生成しない。
PROBABILISTIC_FALLBACK_USE_FROM_BOARD: bool = True

# NON-STABLE → STABLE 遷移後、tier1 (bg_fp NCC による無条件 EMPTY 化) を
# スキップする frame 数。ツモ着地直後の cell を tier1 が誤 EMPTY 化するのを防ぐ。
# v95m15 分析: fi=540-565 で tier1 が着地直後の cell を誤 EMPTY 化 →
# false tsumo_fall 遷移 5 回連発 → multi-vote リセット → STABLE 復帰 0.4 秒遅延。
TIER1_WARMUP_FRAMES: int = 3

# OJAMA_FALL → STABLE 遷移後、tier1 をスキップする frame 数。
# お邪魔降下→消滅後にセル実画素が静的 BackgroundFingerprint に近づき
# (dist 138→3 等) tier1 が強制 EMPTY 化 → physics_fix が前 STABLE 色を上書き
# → 列崩壊が固定されるのを防ぐ。TSUMO_FALL 用 (3 frame) より長め。
# v70 frame=1674 分析: col=0,1,2 row7-12 が崩壊。
OJAMA_TIER1_WARMUP_FRAMES: int = 8

# 機能B: score 急増で即 CHAIN 突入する閾値 (enable_chain_score_early_fire 用)。
# ChainPhaseDetector.SCORE_DELTA_FIRE (=80) と同値に合わせる。
# 1 連鎖最小スコア≒40点、複数連鎖/ボーナスで 80+ が信頼できる発火閾値。
CHAIN_SCORE_EARLY_FIRE_DELTA: int = 80

# 機能C: CHAIN → STABLE 遷移直後の confirmed 凍結時間 (enable_chain_exit_warmup 用)。
# エフェクト残光 (~0.1s) を吸収し、その間 _merge_diff_only による誤色混入を防ぐ。
# 既存 STABLE_WARMUP_FRAMES=12 (0.4s @30fps) より短く設定 (0.1s で十分な残光吸収)。
CHAIN_EXIT_WARMUP_SEC: float = 0.1

# 機能D: 連鎖開始 掛け算式 検知 (enable_chain_formula_detection 用)。
# score ROI の OCR が None (掛け算式表示で NCC conf 低下) かつ ink_ratio が
# CHAIN_FORMULA_INK_RATIO_MIN より高い場合に連鎖開始とみなす。
# ink_ratio ガードはメニュー/試合外の真黒 ROI (ink_ratio≈0) を除外するためのもの。
# 実測: formula=0.975-1.000、通常スコア=1.000、真黒=0.000。
from src.score_ocr import SCORE_ROI_INK_RATIO_MIN as CHAIN_FORMULA_INK_RATIO_MIN  # noqa: E402
# 連続フレーム要件: 単発 OCR ノイズを除去するため N frame 連続で条件成立を要求する。
# 実データ(v70m2): formula が 2-3 frame 持続する。1 frame は偶発ノイズと区別しにくいため 2 を採用。
CHAIN_FORMULA_CONSEC_FRAMES: int = 2

# 大 ROI 走査 (MatchEndDetector 800x600 / TelopDetector 720x400) の間引き間隔。
# 2026-07-30 プロファイル実測: match_end 4.9ms x 2回/フレーム、telop 2.6ms x 2回/フレーム
# = 合計約15ms/フレームで、認識時間の約12%を占める。
# どちらも「勝敗演出」「全消しテロップ」という数秒間持続する事象の検出であり、
# 毎フレーム走査する必要がない (ゲートは一切無く無条件実行されていた)。
#
# リスク (有界): match_end の検出が遅れると lockdown (盤面凍結) の開始が遅れる。
# ただし MatchEndDetector の lockdown_sec=5.0 に対し遅延は最大 THROTTLE_FRAMES 分なので
# 相対的に小さい。また hard_match_off は score_zero_both との OR なので独立経路がある。
# bit-identical にはならないため既定 OFF (enable_large_roi_throttle)。
LARGE_ROI_THROTTLE_FRAMES: int = 8


class _ScoreValNotCached:
    """`_check_formula_detected` の cached_score_val 未指定を表す sentinel。

    修正1 (2026-07-30): スコア OCR 完全重複読み排除。
    score_ocr.read_side() の戻り値 (int | None) には「OCR 成功」も
    「OCR 失敗 (None)」も含まれるため、素の None をデフォルト値に
    使うと「キャッシュ未指定」と「キャッシュ済で OCR 失敗だった」を
    区別できない。専用 sentinel クラスで明確に分離する。
    """


_SCORE_VAL_NOT_CACHED = _ScoreValNotCached()

# 案X*(A)(B)+warmup: NextSlide signal による CHAIN 即終了 (enable_chain_exit_next_signal 用)。
# CHAIN_EXIT_NEXT_WARMUP_SEC: 案X 専用の warmup 凍結時間。
# 機能C の CHAIN_EXIT_WARMUP_SEC=0.1s はエフェクト残光のみを吸収する最小設定だが、
# 案X は連鎖を早く終わらせるため「置き直後・エフェクト残光」が STABLE 露出し誤認する。
# 0.5s に延長して遷移フレームをスキップし、連鎖終了直後の corruption を防ぐ。
# 機能C の CHAIN_EXIT_WARMUP_SEC=0.1 は不変 (backwards compat)。
CHAIN_EXIT_NEXT_WARMUP_SEC: float = 0.5
# (A) 機能D 再点火抑制: 既に CHAIN 中 (active_chain 有効) なら 機能D の発火をスキップ。
# (B) NextSlide signal での CHAIN 即終了: slide_motion=True が 1P/2P いずれかで確認
#     されたとき、その side の active_chain を即クリアして CHAIN 状態を解放する。
# warmup 連動: フラグ ON 時は enable_chain_exit_warmup を内部で自動有効化し、
#     凍結時間は CHAIN_EXIT_NEXT_WARMUP_SEC を使用 (CHAIN_EXIT_WARMUP_SEC より長い)。

# 前試合盤面残骸リーク修正・追修 (2026-07-25): force_in_match=True 構成では
# raw_active が常時 True になり、BoardStateMachine.update() の
# is_match_active=False 分岐 (MENU 強制、5 field クリアの発火点) が一度も
# 走らない。この構成での実際の試合境界は score リセット (新ゲーム開始で
# score が大幅減少 / 両者ほぼ0) でのみ検知できるため、その専用しきい値を
# ここで定義する (enable_match_start_full_clear=True 時のみ使用)。
# SCORE_RESET_THRESHOLD (=500) は ojama_accounting.py の既存定数を流用し重複させない。
from src.ojama_accounting import SCORE_RESET_THRESHOLD  # noqa: E402
# 両者スコアがこれ以下なら「0付近」とみなす (OCR ノイズ許容)。
# scripts/visualize_advantage_overlay.py の SCORE_NEAR_ZERO_THRESHOLD と同値。
MATCH_START_SCORE_NEAR_ZERO_THRESHOLD: int = 20

# 診断用 (2026-07-26, project_win_eval_regen_2026-07-26):
# enable_match_start_full_clear 由来の reset() 発火を計測する opt-in デバッグ出力。
# 環境変数未設定時は完全に no-op (既存挙動・性能に一切影響しない)。
_DEBUG_RESET_PROBE_ENV: str = "PUYO_DEBUG_RESET_PROBE"

# 誤発火修正 (2026-07-26, data/verify/win_eval_regen_2026-07-26/
# diag_v29_mid_resetlog.log で確定): 片側のみの単発 score OCR 誤読
# (例: 2P=40031 不変なのに 1P だけ 48077→0) で境界候補が単発 1 フレーム
# だけ真になっても、strict モードでは連続 SCORE_RESET_BOUNDARY_DEBOUNCE_FRAMES
# フレーム成立するまで実際の reset() 発火とみなさない (呼び出し側で使用)。
SCORE_RESET_BOUNDARY_DEBOUNCE_FRAMES: int = 3


def _is_score_reset_boundary(
    score1: int | None, score2: int | None,
    prev1: int | None, prev2: int | None,
    strict: bool = False,
) -> bool:
    """スコア推移から試合境界 (新ゲーム開始/全消しリセット) の候補フレームを
    検知する (純関数、1 フレーム単位の判定・デバウンスは呼び出し側の責務)。

    strict=False (default, backwards compat): 従来ロジック。
        scripts/visualize_advantage_overlay.py の `_detect_score_reset` と
        同一 (= 表示層の境界検知と内部 state clear の判定を一致させる)。
        片側のみの急落でも発火する (OR 条件)。
    strict=True (2026-07-26 誤発火修正): 片側のみの単発 OCR 誤読による
        誤発火を防ぐため、「両者ともに急落」または「両者ともにほぼ0」の
        場合のみ発火する (AND 条件)。片側のみの急落では発火しない。
    score が None (OCR 失敗) の場合は判定不能として False (誤リセット回避)。
    """
    if score1 is None or score2 is None:
        return False
    near_zero = (
        score1 <= MATCH_START_SCORE_NEAR_ZERO_THRESHOLD
        and score2 <= MATCH_START_SCORE_NEAR_ZERO_THRESHOLD
    )
    drop1 = prev1 is not None and prev1 - score1 >= SCORE_RESET_THRESHOLD
    drop2 = prev2 is not None and prev2 - score2 >= SCORE_RESET_THRESHOLD
    if not strict:
        return drop1 or drop2 or near_zero
    return (drop1 and drop2) or near_zero

from pathlib import Path

from src.background_fingerprint import (
    BackgroundFingerprint, capture_robust_fingerprint,
    capture_patch_pair_robust,
)
from src.chain_detector import DEBOUNCE_CONFIRM_FRAMES, ChainEvent, VideoChainTracker
from src.drift_detector import DriftDetector, DriftResult
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION, ImageReader
from src.inference_board import InferenceBoardGenerator
from src.match_end_detector import MatchEndDetector
from src.match_state import MatchState, MatchStateDetector
from src.next_detector import NextDetector
from src.online_hsv_calibrator import OnlineHsvCalibrator
from src.score_zero import ScoreZeroDetector
from src.telop_detector import TelopDetector, TelopResult
from src.next_slide_detector import (
    NextSlideDetector,
    SlideMotionResult,
    validate_tsumo_placement,
)
from src.placement_inferrer import (
    infer_placement, resolve_after_placement,
    correct_landing_cells_by_observed_color,
    DEFERRED_MAX_FRAMES,
)
from src.score_ocr import ScoreOcr, ScoreTracker
from src.ui_mask import UI_MASK_TARGET_CELLS
from src.state_detectors import (
    ChainPhaseDetector,
    EffectPhaseDetector,
    GravitySettleDetector,
    OjamaPhaseDetector,
    TsumoPhaseDetector,
)


# ============================
# 出力データ
# ============================


@dataclass(frozen=True)
class SideResult:
    """1 サイド分の pipeline 結果.

    Attributes:
        prob_board: STABLE 確定時に隠し段推論を反映した確率盤面
            (W-α、Phase G C-1)。STABLE 以外、または next_pair 不足等で
            推論できなかった場合は None。下流 (`compute_all_probabilistic`)
            は None なら `ProbabilisticBoard.from_board(confirmed_board)`
            にフォールバックして処理する。
    """

    side: str  # "1P" or "2P"
    state: BoardState
    cnn_board: Board       # ImageReader 直の出力 (drift detector 入力)
    inferred_board: Board | None  # 推論盤面 (state ごとに生成)
    confirmed_board: Board | None  # state machine が確定した STABLE 盤面
    drift: DriftResult
    score: int | None
    score_delta: int
    chain_event: ChainEvent | None
    prob_board: ProbabilisticBoard | None = None
    # Phase I: NextValidator 用に next_pair / dnext_pair を露出 (任意)
    # 既存 backward compat 維持のため default None
    next_pair: tuple[int, int] | None = None
    dnext_pair: tuple[int, int] | None = None
    # T4 PuyoErasureMonitor: STABLE 中の「色→EMPTY」 遷移 alert リスト。
    # fail-silent 自動検知用。eval スクリプトが p_to_e_count を集計する。
    # backwards compat のため default None。
    erasure_alerts: list[tuple[int, int]] | None = None
    # C2 StableTransitionMonitor: STABLE→STABLE 間で物理事由なき大幅ぷよ減少 alert。
    # [(frame_idx, t_sec, prev_count, curr_count, drop), ...] の形式。
    # backwards compat のため default None。
    transition_drop_alerts: list[tuple] | None = None
    # 着地色診断フィールド (2026-06-01 infer_placement 調査用)。
    # TSUMO_FALL→STABLE 着地フレームのみ非 None。非着地フレームは None。
    # dict キー:
    #   falling_pair_old: prev_next_queue[-2] 由来の (color1, color2) (従来ロジック)
    #   falling_pair_new: _last_consumed_color 由来の (color1, color2) (修正ロジック)
    #     ※ 案1初版は _landing_pending[1] を使用していたが、_landing_pending は
    #        NEXT変化フレームのgrace処理でクリアされるため着地フレームでは常にNone。
    #        修正版では _last_consumed_color (grace独立保持) を使用する。
    #   source: "last_consumed_color" | "next_queue_2" | "next_queue_1" | "none"
    # backwards compat のため default None。フラグ ON/OFF 問わず常に記録する。
    landing_diag: dict | None = None
    # 反復4 (2026-07-23): confirmed_board が None のとき、その理由を分類する
    # 診断計装フィールド (挙動は一切変えない optional 計装)。
    # confirmed_board is not None のフレームは常に None (該当なし)。
    # 値の種類:
    #   "cold_start": この試合でまだ一度も STABLE 確定していない (最初の未確定)
    #   "menu_reset": 直近で is_match_active=False → MENU 強制 (confirmed_board
    #                 =None 化) が起きて以降、STABLE で再確定していない
    #                 (board_state_machine.py:480-488 経路)
    #   "chain_hold_none": cold_start でも menu_reset でもないが CHAIN/
    #                 GRAVITY_SETTLE 中に confirmed_board が None
    #                 (= is_match_active 経路以外の別要因の疑い)
    #   "other": 上記以外 (通常起こらないはずだが fail-silent 防止用の受け皿)
    # backwards compat のため default None。
    board_none_reason: str | None = None
    # 反復5 (2026-07-23): 物理推論スルー (根治本体)。CHAIN/GRAVITY_SETTLE 中は
    # confirmed_board が None のままだが (= 標準 eval 経路は従来通り None を
    # 見て STABLE のみ評価、backward compat 完全維持)、起点盤面
    # (chain_event.before_board) から ChainSimulator で連鎖を前進させた
    # 「推定」盤面をここに公開する。相手盤面把握・打ち合い判定等の用途向け。
    # confirmed_board is not None のフレームは常に None (該当なし)。
    # backwards compat のため default None。
    estimated_board: "Board | None" = None
    # board_provenance の値:
    #   "observed": confirmed_board が実観測 (通常の STABLE 確定値)。
    #   "chain_estimate": CHAIN/GRAVITY_SETTLE 中の物理推定 (起点信頼度は高)。
    #   "chain_estimate_low_confidence": 起点盤面の物理予測 chain_count が
    #     score 由来 chain_count と不一致 (Step3(a) 答え合わせで検出、
    #     起点自体が誤認の疑いがあるため取り扱い注意)。
    #   "chain_estimate_stale_hold": 案1 (2026-07-23, c62 1P estimated_board
    #     カバレッジ崩壊 9.8% の真因診断 recognition_diag_c62_1p_estimate_collapse
    #     への対処)。CHAIN/GRAVITY_SETTLE 中に疑似連鎖イベントの early-fire 等で
    #     起点盤面 simulate が chain_count=0 (推定が立ち上がらない) となり
    #     新規推定を計算できないフレームで、直前に成功した推定盤面 (無ければ
    #     起点盤面) をそのまま保持して公開する。「推定が信頼できない保持中」
    #     を意味し、値そのものは古い可能性がある点に注意。
    #     CHAIN_ESTIMATE_STALE_HOLD_MAX_SEC 超過、または STABLE 復帰で
    #     自動的に None (= 従来同様 "observed") に戻る安全弁あり。
    # backwards compat のため default "observed"。
    board_provenance: str = "observed"
    # 反復5 修正 (2026-07-23): Step3(b)(c) 事後答え合わせの結果。
    # 連鎖後 final_board 適用直後から CHAIN_VERIFY_FRAMES 分の STABLE
    # cnn_board が集まったフレームでのみ非 None になる (それ以外は None =
    # 検証中/対象外)。
    #   "verified_match": 多数決盤面と物理予測が一致 (信頼度確認)。
    #   "verified_mismatch_corrected": 不一致のため多数決盤面で confirmed_board
    #     を補正した (起点誤認が事後に判明したケース)。
    # backwards compat のため default None。
    answer_check_result: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    """1 frame 投入結果."""

    frame_idx: int
    time_sec: float
    is_match_active: bool
    p1: SideResult
    p2: SideResult


# ============================
# 真因 A 対処: 着地補正ヘルパー
# ============================


def _apply_landing_observed_color_correction(
    inferred: "Board",
    prev_confirmed: "Board",
    cnn_board: "Board",
    reader: object,
    frame_bgr: "np.ndarray",
    region: object,
) -> "Board":
    """着地確定盤面の 2 cell を CNN==HSV 一致色で補正する.

    真因 A 対処 (fix/v70-zeropatch-redyellow, 2026-06-01):
    infer_placement は falling_pair に盲従するが、 falling_pair のタイミングずれで
    誤色が選ばれる (v89 54/72 不一致)。 着地 2 cell で CNN と HSV-only が
    一致する有効 puyo 色があれば、 2 つの独立認識器の合意を信頼して採用する。

    動作:
      1. inferred と prev_confirmed の差分 2 cell (= 着地セル) を抽出する。
      2. 各セルで cnn_board の色と hsv_classifier の観測色が一致し
         かつ有効 puyo 色 (空/UNKNOWN/お邪魔 以外) であれば inferred のその cell を上書き。
      3. 一致しない cell は inferred の色をそのまま保持する (保守的)。

    Args:
        inferred: infer_placement が返した着地後の確定盤面 (上書き元)。
        prev_confirmed: TSUMO_FALL 開始前の確定盤面 (= 差分取得用)。
        cnn_board: 着地後の CNN+HSV 融合観測盤面 (HybridClassifier 出力)。
        reader: ImageReader インスタンス。_classifier._hsv で ColorClassifier にアクセス。
        frame_bgr: 着地後の BGR フレーム画像。
        region: BoardRegion (cell_sample_rect を持つ)。

    Returns:
        補正後の確定盤面。
    """
    from src.board import (
        BOARD_ROWS, BOARD_COLS, COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA,
        COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE,
    )
    from src.placement_inferrer import (
        _extract_cell_patch_from_frame, _VALID_PUYO_COLORS,
    )

    # HSV-only 分類器を取得 (HybridClassifier._hsv or ColorClassifier 直接)
    classifier = getattr(reader, "_classifier", None)
    hsv_clf = getattr(classifier, "_hsv", classifier)
    if hsv_clf is None or not hasattr(hsv_clf, "classify"):
        return inferred  # 取得できなければ補正スキップ

    # 着地 2 cell を差分から抽出 (prev→inferred で新規に色が入ったセル)
    landing_cells: list[tuple[int, int]] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            pv = int(prev_confirmed.get(r, c))
            iv = int(inferred.get(r, c))
            if (
                pv in (COLOR_EMPTY, COLOR_UNKNOWN)
                and iv not in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA)
            ):
                landing_cells.append((r, c))

    if not landing_cells:
        return inferred

    from src.placement_inferrer import correct_landing_cells_by_observed_color
    from src.placement_inferrer import LandingPattern

    # LandingPattern を差分 2 cell から再構築 (補正関数に渡すため)
    if len(landing_cells) == 2:
        cells_tuple = (
            (landing_cells[0][0], landing_cells[0][1]),
            (landing_cells[1][0], landing_cells[1][1]),
        )
        # orientation は補正関数では参照しないが型合わせで設定
        (r1, c1), (r2, c2) = cells_tuple
        orientation = "vertical" if c1 == c2 else "horizontal"
        dummy_pattern = LandingPattern(cells=cells_tuple, orientation=orientation)
        return correct_landing_cells_by_observed_color(
            inferred, dummy_pattern, cnn_board, hsv_clf, frame_bgr, region,
        )

    # 着地 cell が 1 cell または 3+ cell の場合は個別に補正 (保守的)
    result = inferred.copy()
    for (r, c) in landing_cells:
        cnn_color = int(cnn_board.get(r, c))
        if cnn_color not in _VALID_PUYO_COLORS:
            continue
        patch = _extract_cell_patch_from_frame(frame_bgr, region, r, c)
        if patch is None or patch.size == 0:
            continue
        try:
            hsv_color = int(hsv_clf.classify(patch))
        except Exception:
            continue
        if hsv_color in _VALID_PUYO_COLORS and cnn_color == hsv_color:
            result.set(r, c, hsv_color)
    return result


# ============================
# 色フリッカ根因への防御的修正 案(iii) (2026-07-25)
# ============================


def _flag_landing_distrust_cells(
    inferred: "Board",
    prev_confirmed: "Board",
    cnn_board: "Board",
) -> set[tuple[int, int]]:
    """着地セルのうち CNN 観測色が baseline (P2 推論) と食い違う「疑わしいセル」を検出する.

    案(iii) の中核: 設置推論 (P2) が書いた色をこの関数自体は一切書き換えず
    (baseline 不変、地雷再発防止 = feedback_recognition_regression_prevention)、
    CNN 観測との不一致だけをフラグとして P7 (着地投票, _update_landing_votes) に
    伝播する。フラグされたセルだけが NEXT 色 2 択バイアスを迂回し、
    生 CNN 多数決フォールバックに必ず落ちる (呼び出し側で処理)。

    #47 対策: cnn_board の観測色が UNKNOWN/EMPTY/おじゃまの場合は
    「有効な反証なし」としてフラグしない。高速プレイで infer_placement が
    唯一の情報源となるケース (#47) の挙動を壊さないため。

    Args:
        inferred: infer_placement (+ 既存後処理) が返した着地後の確定盤面
            (baseline。この関数では変更しない)。
        prev_confirmed: TSUMO_FALL 開始前の確定盤面 (= 着地 cell 差分抽出用)。
        cnn_board: 着地後の CNN+HSV 融合観測盤面。

    Returns:
        疑わしい着地セル座標の集合 (r, c)。空集合 = 疑わしいセルなし。
    """
    from src.placement_inferrer import _VALID_PUYO_COLORS

    distrust: set[tuple[int, int]] = set()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            pv = int(prev_confirmed.get(r, c))
            iv = int(inferred.get(r, c))
            if not (
                pv in (COLOR_EMPTY, COLOR_UNKNOWN)
                and iv not in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA)
            ):
                continue
            cnn_v = int(cnn_board.get(r, c))
            if cnn_v in _VALID_PUYO_COLORS and cnn_v != iv:
                distrust.add((r, c))
    return distrust


# ============================
# 修正方針 甲: P2 設置推論の防御的 CNN 照合 (2026-07-25)
# ============================


def _apply_placement_cnn_veto(
    inferred: "Board",
    prev_confirmed: "Board",
    cnn_board: "Board",
    mode: str = "hold",
) -> "Board":
    """P2 (infer_placement) が着地セルへ色を書く前に、現フレーム CNN 観測と照合する.

    背景 (project_color_flicker_p2_root_cause_2026-07-25): P2 誤色の内訳は
    キューに正解無し 57% / 1 手先 24% / 1 手遅れ 14%。 案(iii)
    (_flag_landing_distrust_cells) は書いた後に P7 (着地投票) へフラグ伝播
    するが、本関数は書く前に止める「門番」として働く (併用可能、独立)。

    動作: 着地セル (= prev_confirmed が空/UNKNOWN → inferred で新規に有効色)
    のうち、cnn_board の観測色が inferred の色 (= NEXT キュー由来色) と
    一致しない場合、そのセルの書き込みを保留する。保留セルは
    prev_confirmed の値 (通常 EMPTY) に戻すだけで、新たな色は書かない。
    保留セルは既存の着地色補正 (_apply_landing_observed_color_correction) /
    P5 事後復旧ゲート (enable_stable_recovery_gate) / P7 3 票ゲート
    (_update_landing_votes) が後続フレームで正しい色を埋める
    (新規の復旧機構は作らない)。

    Args:
        inferred: infer_placement が返した着地後の確定盤面 (書き込み元)。
        prev_confirmed: TSUMO_FALL 開始前の確定盤面 (差分抽出用)。
        cnn_board: 着地後の CNN+HSV 融合観測盤面。
        mode: "hold" (既定) = 不一致セルを保留 (prev_confirmed の値に戻す)。
            "cnn_color" = CNN 観測色が有効 puyo 色ならその色を採用する
            (queue 色でなく CNN 色を書く)。CNN が EMPTY/UNKNOWN/おじゃま の
            場合は mode に関わらず保留する (無効色は書けないため)。
            A/B 計測 (8 フレーム反映基準) で "hold" が悪化する場合の代替案。
            "empty_hold_cnn_color" (追試, 2026-07-25): "hold" の副作用
            (色不一致セルまで保留 → P2 再発火疑いで書込件数倍増) を切り分ける
            ため、保留対象を「CNN が EMPTY (= 視覚的にまだ何も無い、早すぎる
            書き込みの証拠)」のケースのみに絞った複合変種。
            CNN==EMPTY → 保留 (prev_confirmed の値に戻す、早すぎる書き込み防止)。
            CNN が有効 puyo 色で queue 色と不一致 → CNN 色を採用 (cnn_color 挙動)。
            CNN==UNKNOWN/おじゃま、または CNN==queue 色 → 従来通り queue 色。

    Returns:
        veto 適用後の確定盤面 (該当なしなら inferred のコピーがそのまま返る)。
    """
    from src.placement_inferrer import _VALID_PUYO_COLORS

    result = inferred.copy()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            pv = int(prev_confirmed.get(r, c))
            iv = int(inferred.get(r, c))
            if not (
                pv in (COLOR_EMPTY, COLOR_UNKNOWN)
                and iv not in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA)
            ):
                continue  # 着地セルでない (対象外)
            cnn_v = int(cnn_board.get(r, c))
            if cnn_v == iv:
                continue  # CNN が queue 色と一致 → そのまま書く (no-op)
            if mode == "empty_hold_cnn_color":
                if cnn_v == COLOR_EMPTY:
                    result.set(r, c, pv)  # 早すぎる書き込みのみ保留
                elif cnn_v in _VALID_PUYO_COLORS:
                    result.set(r, c, cnn_v)  # CNN 観測の別色を採用
                # UNKNOWN/おじゃま は queue 色のまま (= 従来通り、no-op)
                continue
            if mode == "cnn_color" and cnn_v in _VALID_PUYO_COLORS:
                result.set(r, c, cnn_v)  # CNN 観測の別色を採用
                continue
            result.set(r, c, pv)  # 保留: prev_confirmed の値 (通常 EMPTY) に戻す
    return result


# ============================
# Pipeline 本体
# ============================


class RecognitionPipeline:
    """frame 入力で 1P/2P 確定盤面を返す統合 pipeline.

    Usage:
        pipe = RecognitionPipeline.load_default()
        for frame_idx, (frame, t) in enumerate(stream):
            res = pipe.update(frame_idx, t, frame)
            if res.p1.state == BoardState.STABLE:
                use(res.p1.confirmed_board)
    """

    # ChainEvent の有効期間: drop 観測時刻からこの秒数だけ CHAIN として
    # signals に乗せ続ける (state machine が CHAIN にロックされる)。
    # VideoChainTracker は drop 観測 1 frame だけ event を返すため、
    # pipeline 側で post-hoc に有効期間を伸ばす。
    CHAIN_HOLD_PER_STEP_SEC: float = 0.3

    # A0 (2026-07-24, 計装 a287c587 実測較正): CHAIN 保持時間モデルの固定項。
    # 実測 (23動画418イベント) では `固定項 + 係数×連鎖数` の線形モデルが
    # 原点通過モデルより有意に良く適合 (a≈2.61s, b≈1.17s/連鎖, R²0.356)。
    # 既定値 0.0 = 従来の「固定項なし・per_step のみ」の式と完全同一
    # (backwards compat, bit-identical)。較正値は呼び出し側 (評価 config) で
    # optional 引数 chain_hold_base_sec 経由で注入する (src 既定は変更しない)。
    CHAIN_HOLD_BASE_SEC: float = 0.0

    # game-event ベース連鎖終了: CHAIN 状態を timing hold だけでなく
    # 「次ツモ出現 (next_pair 変化)」または「連鎖した側の盤面にお邪魔新規出現」
    # を検知するまで維持する。安全弁として以下の秒数を上限とする。
    # 60fps × 5.0s = 300 frame。通常の長い連鎖 (= 11 連鎖 × 0.3s = 3.3s) を
    # 十分にカバーし、かつ異常時 (event 永続不達) での CHAIN 永続化を防ぐ。
    # A0 (2026-07-24): CHAIN_HOLD_BASE_SEC/PER_STEP_SEC の較正値を使うと
    # chain_until (本来の timing hold) がこの安全弁を上回るケースが生じうる
    # (eff_until = max(chain_until, chain_event_max_until) のため)。
    # その場合は呼び出し側で chain_max_hold_sec も併せて引き上げること
    # (既定値はこの定数のまま、backwards compat)。
    CHAIN_MAX_HOLD_SEC: float = 5.0

    # X1: CHAIN 最小表示時間 (enable_chain_min_display=True 時に有効)。
    # 一度 CHAIN に入ったら最低この秒数は game-event 終了を抑止する。
    # 短連鎖 (1-2 連鎖 × 0.3s = 0.3-0.6s) がお邪魔信号で 0.1-0.2s で即終了し
    # 「一瞬表示/ちらつき」になる問題を防ぐ。
    CHAIN_MIN_DISPLAY_SEC: float = 0.8

    # 反復5 Step3(b)(c) 修正版 (2026-07-23, 事後検証方式): Phase C-6 の C で
    # 物理予測 final_board を適用「した後」、直近 CHAIN_VERIFY_FRAMES 分の
    # STABLE cnn_board 多数決盤面と照合する答え合わせ。単一フレームの
    # 生 CNN (= GRAVITY_SETTLE 直後の残光ノイズが乗りやすい) との比較で
    # 正しい注入まで過剰棄却していた旧「事前ゲート」方式の回帰
    # (残像/連鎖後不一致率 0.09→0.28 悪化) を修正するため、適用は止めず
    # 事後の多数決比較で不一致なら補正する方式に変更。
    CHAIN_VERIFY_FRAMES: int = 5
    # 事後不一致とみなす cell 数閾値 (COLOR_UNKNOWN 除外)。
    # cycle 48 の大量 hallucination ガード基準 (6 cell) を流用。
    CHAIN_VERIFY_MISMATCH_CELLS: int = 6

    # 案1 (2026-07-23): stale_hold 安全弁。
    # `recognition_diag_c62_1p_estimate_collapse/summary.txt` の実測では、
    # 疑似連鎖イベント early-fire が連続発火し推定が立ち上がらない
    # (H2 contaminated=True) 崩壊区間は最長でも約 7.8 秒だった。
    # 本値はそれに安全マージンを持たせた「連続 stale_hold を許容する上限秒数」。
    # 超過したら None (= 従来同様 'observed') に戻し、state machine 側の
    # 未知バグで CHAIN が異常に長時間継続した場合に古い盤面を無期限に
    # 貼り続ける事故を防ぐ。
    CHAIN_ESTIMATE_STALE_HOLD_MAX_SEC: float = 12.0

    # X4: game-event 終了を発動する最小連鎖数。
    # chain_count < この値の連鎖は game-event 終了を発動せず timing hold のみ。
    # 短連鎖 (1-2 連鎖) では chainexit がノイズになるため除外する。
    CHAIN_GAME_EVENT_MIN_COUNT: int = 3

    # 全消し演出 overlay の chain_until 延長秒数 (2026-05-14, cycle 71v).
    # is_all_clear=True の ChainEvent では、 通常の chain hold に加えて
    # この秒数だけ CHAIN state を延長する。 これにより CHAIN→STABLE 遷移時の
    # _merge_diff_only が連鎖直後の overlay corrupted cnn_board で発火するのを防ぐ。
    #
    # 注意 (2026-05-14 ユーザー修正): ぷよぷよ eスポーツの全消しテロップは
    # 「次のぷよ消去 (= 次の連鎖) まで残る」 仕様、 かつ「ぷよより一つ下のレイヤー」
    # (puyo が telop に被さって描画される)。 したがって本延長はあくまで連鎖直後の
    # 強い overlay (カットイン + テロップフェードイン) 期間をカバーする partial fix
    # であり、 持続する telop に対する構造的解消は別途 telop 視覚 detector や
    # _all_clear_pending_X 経由の継続フィルタが必要 (TODO)。
    ALL_CLEAR_OVERLAY_HOLD_SEC: float = 1.5

    # 試合開始直後 window (cycle 71v, 2026-05-14, 汎用化対策).
    # IN_MATCH 開始から N frame 以内なら DetectorSignals.match_just_started=True を
    # 立て、 state_machine の初回 STABLE 確定で confirmed=空 Board() を強制する。
    # v51/v70 の 2 試合目開始時の背景誤認が confirmed に乗るのを構造的に解消。
    # 60fps 換算で 60 frame = 1.0s. ぷよぷよ eスポーツの試合開始は READY→GO の
    # 演出と第一ツモ落下前にこの window がほぼ収まる前提。
    MATCH_JUST_STARTED_WINDOW_FRAMES: int = 60
    # フレーム定数→時間定数化 Stage1 (2026-07-25): 実ロジックは秒定数を正として
    # 使う (frame 定数は既存 import 互換のため残置)。60fps 動画では
    # (frame_idx 差分)/60 == time_sec 差分 が恒等式のため bit-identical、
    # 30fps 動画では実秒基準になり体感の遅延 (旧: 実質 2 倍) を解消する。
    MATCH_JUST_STARTED_WINDOW_SEC: float = MATCH_JUST_STARTED_WINDOW_FRAMES / 60

    # 試合状態の hysteresis: 直前 N frame 内で 1P/2P がアクション中
    # (STABLE/TSUMO/CHAIN/OJAMA) なら、MatchStateDetector が NOT_IN_MATCH
    # を返しても is_match_active=True を強制。1 frame の単発誤判定を吸収し、
    # 「試合中に一瞬メニューに落ちる」現象を消す。物理的に試合中はメニュー
    # 状態にならない (ネットワーク切断など特殊状況除く) というユーザー仕様
    # に基づく。
    MATCH_ACTIVE_HOLD_FRAMES: int = 10
    # フレーム定数→時間定数化 Stage1 (2026-07-25): 実ロジックは秒定数を正とする。
    MATCH_ACTIVE_HOLD_SEC: float = MATCH_ACTIVE_HOLD_FRAMES / 60

    # 試合開始から N frame 内は CHAIN 禁止: 試合 active 開始直後は puyo が
    # 増える時期で連鎖発生はあり得ない。VideoChainTracker が「メニュー画面
    # 0 個 → 試合開始直後 puyo 出現」を「連鎖発火 = 急減」と誤検出する
    # 現象を ban する。最初の 1 手目から CHAIN state に遷移するのを防ぐ。
    CHAIN_BAN_FRAMES_AFTER_MATCH_START: int = 30
    # フレーム定数→時間定数化 Stage1 (2026-07-25): 実ロジックは秒定数を正とする。
    CHAIN_BAN_SEC_AFTER_MATCH_START: float = CHAIN_BAN_FRAMES_AFTER_MATCH_START / 60

    # cycle 71f (提案 A): score 動きで in_match 強制復帰判定の window と閾値.
    # SCORE_MOVE_WINDOW_FRAMES 内に SCORE_MOVE_MIN_DELTA 以上動いていれば
    # hard_match_off (= MatchEnd lockdown 等) を打ち消して試合中継続.
    # 60 frame = 1 秒. 連鎖発火 (= 100+ 点増加) でなくとも、 ツモ落下中の
    # +1 点増加が継続的にあれば検出可能.
    SCORE_MOVE_WINDOW_FRAMES: int = 60
    # フレーム定数→時間定数化 Stage1 (2026-07-25): 実ロジックは秒定数を正とする。
    # `_recent_scores_*` は「件数 N 件保持」から「直近 N/60 秒保持」に変更する
    # (score 取得タイムスタンプを併せて保持し、 経過秒数でトリムする)。
    SCORE_MOVE_WINDOW_SEC: float = SCORE_MOVE_WINDOW_FRAMES / 60
    SCORE_MOVE_MIN_DELTA: int = 5

    # cycle 71h: 試合切替直後の bg_fp 採取緩和ガード.
    # CNN が背景を puyo 誤認した場合でも、 試合切替後 N frame 内なら少量 (= <= 5)
    # puyo 観測まで bg_fp 採取を継続. v50 試合 2 開始時の鶏卵問題対策.
    # cycle 18 (2026-05-16, B3): cnn_phase_i_hsv_seed.pt (= empty 学習なし) では
    # 試合開始から puyo_count_total >> 5 で従来 ガードを満たさない (= bg_fp 永久に
    # 採取できない 鶏卵問題が再発). MAX_PUYO を 5→144 (全 cell) に緩和して、
    # 試合切替後 short window では puyo 数を問わず採取する。 短期間限定なので
    # 試合中の puyo 色を背景 fingerprint に取り込むリスクは小。
    BG_FP_FORCE_WINDOW_FRAMES: int = 120  # 2 秒
    # フレーム定数→時間定数化 Stage1 (2026-07-25): 実ロジックは秒定数を正とする。
    BG_FP_FORCE_WINDOW_SEC: float = BG_FP_FORCE_WINDOW_FRAMES / 60  # 2.0 秒
    # cycle 18 (2026-05-16, B3): cnn_phase_i_hsv_seed.pt (= empty 学習なし) では
    # bg_fp 鶏卵問題が再発するため 5→144 に緩和した定数。
    # B2 (A/B 対照実験): 144→4 に絞る仮説あり。__init__ で _bg_fp_force_max_puyo
    # instance 変数に上書きし、 load_default(bg_fp_force_max_puyo=4) で有効化。
    BG_FP_FORCE_MAX_PUYO: int = 144

    # cycle 71h: 着地後 vote 累積 refinement 設定.
    # 着地時に追加された cells の色を後続 N frame で蓄積し、 最頻値が
    # 「期待色 (NEXT 由来)」 または「他の puyo 色」 になれば confirmed を更新.
    # ユーザー要件「置いて 1 秒後には完璧な色を判別」 の実現.
    # cycle 71j (2026-05-12 案 1b): 60→30 frame に短縮し、 ratio 0.6→0.5 に緩和.
    # cycle 71k (2026-05-13 S3): さらに短縮 30→15 frame, ratio 0.5→0.4.
    # 着地後の修正速度を 0.5s→0.25s に短縮 (= ユーザー要件「気づくまで遅い」 対策).
    # cycle 71p (2026-05-13): 認識頻度 30fps、 判断時間 0.6 秒に調整.
    # LANDING_VOTE_FRAMES は動画 frame_idx 差分 (= 60fps).
    # 0.6 秒 × 60fps = 36 frame. 認識処理 30fps × 0.6s = 18 update 累積で多数決.
    # cycle 5 (2026-05-15, F8): 36 → 24 で 着地 vote 完了時間 0.6s → 0.4s.
    # 修正速度を更に上げる。 サンプル数減るが LANDING_VOTE_MIN_RATIO=0.3 と
    # 組み合わせて発火閾値を維持。
    LANDING_VOTE_FRAMES: int = 24
    # フレーム定数→時間定数化 Stage1 (2026-07-25): 実ロジックは秒定数を正とする
    # (frame 定数は既存 import / 白箱テスト互換のため残置)。
    LANDING_VOTE_SEC: float = LANDING_VOTE_FRAMES / 60  # 0.4 秒
    # cycle 4 (2026-05-15, F7): 0.4 → 0.3 に下げて landing_vote 補正速度↑.
    LANDING_VOTE_MIN_RATIO: float = 0.3  # 3 割で確定 (NEXT 色一致時)
    # cycle 26 (2026-05-18, A2): 着地直後 5 frame の CNN ぶれを除外。
    # この期間は raw CNN が着地フラッシュ・揺らぎで不安定なため vote 蓄積から除外。
    LANDING_VOTE_INIT_SKIP_FRAMES: int = 5
    # フレーム定数→時間定数化 Stage1 (2026-07-25): 実ロジックは秒定数を正とする。
    LANDING_VOTE_INIT_SKIP_SEC: float = LANDING_VOTE_INIT_SKIP_FRAMES / 60
    # cycle 26 (A2): NEXT 色不一致時の fallback ratio。NEXT pair 色と majority が
    # 不一致なら 0.3 では弱い → 0.5 必須に引上げて誤色採用を抑制。
    LANDING_VOTE_MISMATCH_MIN_RATIO: float = 0.5
    # cycle 26 (2026-05-18, A4): NEXT 色 votes (HSV 距離分類) 適用閾値。
    # 既存 cycle 71m β2'' は len>=3 のみで即適用 = 順序依存で不安定。
    # ratio チェック追加で誤分類を抑える (= 強い prior 化)。
    LANDING_VOTE_NEXT_MIN_COUNT: int = 3
    LANDING_VOTE_NEXT_MIN_RATIO: float = 0.7
    # cycle 26 (A4): 早期確定。vote 蓄積中でも len>=5, ratio>=0.8 で即適用。
    # 着地後 ~0.1s で正しい色に確定 → grace 終了直後の visual 安定化。
    LANDING_VOTE_NEXT_EARLY_COUNT: int = 5
    LANDING_VOTE_NEXT_EARLY_RATIO: float = 0.8

    # cycle 71n (案 θ, 2026-05-13): STABLE 中の長期不一致 vote.
    # 各 cell の CNN 観測色を直近 N update 呼出分保持し、 最頻値が confirmed と
    # 異なり一定 ratio 以上なら CNN 側で上書き. 「ずっと残る誤色」 を自動修正する.
    # cycle 71p (2026-05-13): 認識 30fps × 0.6 秒 = 18 update 呼出分の履歴.
    # 副作用回避のため ratio 0.9 維持 (= 16/18 一致で上書き、 ほぼ確実).
    STABLE_CNN_HISTORY_FRAMES: int = 18  # 約 0.6 秒 (= update 30fps × 0.6s)
    # cycle 2 (2026-05-15, F4): 0.9 → 0.75 で long-term vote 補正速度↑.
    # 18 frame 中 14 frame (= 78%) 一致で override 発火、 0.6s 以内に補正完了.
    STABLE_OVERRIDE_MIN_RATIO: float = 0.75

    # DriftDetector 再同期ループ暴走ガード (2026-07-25, c34 実測)。
    # 試合開始直後は HSV 較正が浅く (5色中1色) CNN 誤読が残り、推論盤面
    # (inferred) と cnn_board の乖離が DriftDetector 閾値 (cell 6 / frame 3) を
    # 超えて `sm.reset(keep_match_state=True) + drift.reset() + gen.reset()`
    # (本ファイル _step_side 内) が発火 → リセット直後も誤読継続 → 再発火、
    # の自己永続ループが最大 13 秒程度継続する不具合が確認された
    # (c34 実測: リセット 86 回中 85 回がこの経路)。
    # ガード1 (enable_drift_resync_match_start_guard) が有効な間、試合開始
    # (_match_active_started_time) からこの秒数以内は needs_resync を無視する
    # (観測された 13 秒暴走をカバーする値)。既存 MATCH_JUST_STARTED_WINDOW_SEC
    # (1.0 秒、確定盤面の空フィールド強制用) とは別目的の定数のため独立させる。
    DRIFT_RESYNC_MATCH_START_GUARD_SEC: float = 15.0
    # ガード2 (enable_drift_resync_hsv_gate) が有効な間、OnlineHsvCalibrator の
    # 較正済み色数 (_online_hsv_injected_colors) がこの値未満なら resync を
    # 抑制する。試合は 4 色構成 (reference_four_colors_per_match_2026-07-22) の
    # ため、3 色較正済みであれば概ね安定しているとみなす。
    DRIFT_RESYNC_MIN_CALIBRATED_COLORS: int = 3

    # baseline_broken リセット 限定緩和版 (2026-07-25, A/B 計測用)。
    # enable_baseline_broken_grace=True の場合、STABLE 突入からこの秒数
    # 経過するまではカウンタ加算を抑制する (着地直後の過渡的な puyo 数差で
    # 誤って自己修復 reset が発火するのを防ぐ目的)。
    BASELINE_BROKEN_STABLE_GRACE_SEC: float = 3.0

    def __init__(
        self,
        image_reader: ImageReader,
        match_state_detector: MatchStateDetector,
        score_ocr: ScoreOcr | None = None,
        chain_tracker_1p: VideoChainTracker | None = None,
        chain_tracker_2p: VideoChainTracker | None = None,
        stable_frame_count: int = 6,
        chain_hold_per_step_sec: float | None = None,
        # A0 (2026-07-24): CHAIN 保持時間モデルの固定項 base + per_step×連鎖数。
        # None (既定) なら CHAIN_HOLD_BASE_SEC=0.0 (= 従来式と bit-identical)。
        chain_hold_base_sec: float | None = None,
        # A0 (2026-07-24): CHAIN_MAX_HOLD_SEC (安全弁上限) の上書き。
        # None (既定) なら CHAIN_MAX_HOLD_SEC=5.0 (= 従来値、backwards compat)。
        # 較正済 chain_hold_base_sec/per_step_sec を使う場合は安全弁が
        # 事実上無効化されうるため (クラス定数コメント参照)、併せて引き上げること。
        chain_max_hold_sec: float | None = None,
        temporal_smoothing: int = 1,
        next_detector: NextDetector | None = None,
        force_in_match: bool = False,
        enable_pseudo_label: bool = False,
        pseudo_label_store: object | None = None,
        enable_next_slide_detector: bool = True,
        score_zero_detector: ScoreZeroDetector | None = None,
        match_end_detector: MatchEndDetector | None = None,
        telop_detector: TelopDetector | None = None,
        online_hsv: OnlineHsvCalibrator | None = None,
        enable_warmup_guard: bool = False,
        bg_fp_force_max_puyo: int | None = None,
        enable_piece_persistence: bool = False,
        enable_tier1_warmup: bool = False,
        enable_ojama_tier1_warmup: bool = True,
        # 2026-06-02: user viz 採用承認により default False (OFF) に変更。
        # 旧 default True から変更: constraint_fill は色破壊の主因でなく
        # (ON/OFF で corruption ほぼ不変)、採用スタックでは OFF が承認された。
        # --no-constraint-fill CLI フラグは既存互換のため維持 (冗長だが無害)。
        # True に戻すには enable_constraint_fill=True を明示する。
        enable_constraint_fill: bool = False,
        enable_t2_highconf_yield: bool = True,
        enable_infer_empty_guard: bool = True,
        # game-event ベース連鎖終了 (C-1/C-2 plan, 2026-06-01)。
        # True にすると CHAIN 状態を timing hold だけでなく、
        # 「次ツモ出現 (next_pair 変化)」または「連鎖した側の盤面にお邪魔新規出現」
        # を検知するまで維持する。True = default ON (2026-06-01 user 採用承認)。
        # False を明示すると従来 timing hold のみの挙動に戻る (backwards compat)。
        enable_game_event_chain_exit: bool = True,
        # 着地色修正 案1 (2026-06-01): TSUMO_FALL→STABLE 着地時の falling_pair を
        # prev_next_queue[-2] から _landing_pending (消費済みツモ色) に切り替える。
        # slide_motion(R-7) 経由で「1 つ前のツモ」を指してしまう誤色問題の修正。
        # デフォルト False = 従来挙動完全維持 (backwards compat)。
        enable_landing_color_fix: bool = False,
        # X1/X4 短連鎖ちらつき対策 (2026-06-01):
        # True で CHAIN 最小表示時間 (CHAIN_MIN_DISPLAY_SEC) および
        # 短連鎖 game-event exit 抑止 (CHAIN_GAME_EVENT_MIN_COUNT) を有効化。
        # enable_game_event_chain_exit と独立フラグ (効果分解のため)。
        # 実質 enable_game_event_chain_exit=True と併用される前提。
        # デフォルト False = 従来挙動完全維持 (backwards compat)。
        enable_chain_min_display: bool = False,
        # fix/v70-zeropatch-redyellow (2026-06-01): HSV 分類 fallback。
        # True にすると _classify_next_pair_by_hsv の 2 択強制確定を回避する。
        # 黄(H26)→赤(H7) 誤分類 (H 差 19) の発火点対策。
        # デフォルト False = 従来の 2 択強制確定 (完全不変、 backwards compat)。
        enable_hsv_classify_fallback: bool = False,
        # 真因 A 対処 (2026-06-01): 着地セルの CNN==HSV 一致色で falling_pair ズレを補正。
        # True にすると TSUMO_FALL→STABLE 着地時に着地 2 cell の CNN 観測色と
        # HSV-only 観測色が一致する場合、infer_placement 結果を観測色で上書きする。
        # 2026-07-25 user レビュー (c34 v6) 承認で既定 ON 化。
        # False を明示指定すれば旧挙動 (bit-identical) に戻せる (backwards compat)。
        enable_landing_observed_color: bool = True,
        # 色フリッカ根因への防御的修正 案(iii) (2026-07-25):
        # True にすると着地セルで CNN 観測色が baseline (P2 推論結果) と
        # 食い違う「疑わしいセル」を検出し、着地投票 (P7,
        # _update_landing_votes) に伝播する。フラグされたセルだけが
        # NEXT 色 2 択バイアスを迂回し、生 CNN 多数決フォールバックに
        # 必ず落ちる。baseline (P2 推論結果) 自体は書き換えない。
        # デフォルト False = 従来挙動完全維持・bit-identical (backwards compat)。
        # user 承認前の savepoint 実装のため default OFF 固定
        # (default ON 化 / main マージは別途 user 承認が必要)。
        enable_placement_color_cnn_check: bool = False,
        # 修正方針 甲: P2 設置推論の防御的 CNN 照合 (2026-07-25)。
        # True にすると infer_placement (P2) が着地セルへ色を書く時点で
        # 現フレーム CNN 観測と照合し、不一致 (queue 色と不一致、または
        # CNN が EMPTY/UNKNOWN/おじゃま) なら書き込みを保留する (そのセルは
        # 書かない = prev_confirmed の値のまま)。保留セルは既存の着地色補正 /
        # P5 事後復旧ゲート / P7 3 票ゲートが後続フレームで埋める
        # (新規の復旧機構は作らない)。案(iii) (enable_placement_color_cnn_check)
        # とは独立: 案(iii) は書いた後に P7 へフラグ伝播、本フラグは書く前に
        # 止める門番。両方 True でも安全に併用可能 (併用時は本フラグが先に
        # 適用されるため、案(iii) の distrust 判定対象セルは自然に減る)。
        # デフォルト False = 従来挙動完全維持・bit-identical (backwards compat)。
        # user 承認前の savepoint 実装のため default OFF 固定
        # (default ON 化 / main マージは別途 user 承認が必要)。
        enable_placement_cnn_veto: bool = False,
        # 上記 veto の不一致時挙動。"hold" (既定) = 保留 (書かない)。
        # "cnn_color" = CNN 観測色が有効 puyo 色ならその色を採用する
        # (queue 色でなく CNN 色を書く、EMPTY/UNKNOWN/おじゃま のみ保留)。
        # A/B 計測で "hold" が 8 フレーム反映基準を悪化させる場合の代替。
        # enable_placement_cnn_veto=False の間は完全に無視される (無害)。
        placement_cnn_veto_mode: str = "hold",
        # fix/v70-zeropatch-redyellow (2026-06-02): 赤色相折り返し補正。
        # True (default) にすると HSV 経路の median 計算で赤 2 峰を collapse し
        # 黄↔赤ちらつきを抑制する。ColorClassifier に enable_red_hue_wrap_fix を伝播。
        # user viz 採用承認済 (2026-06-02)。False = 従来の単純 median (後方互換が必要な場合のみ)。
        enable_red_hue_wrap_fix: bool = True,
        # 案D (fix/v70-zeropatch-redyellow): 光沢ハイライト除外彩度計算。
        # True にすると白ハイライト画素を彩度 median 計算から除外し、
        # ぷよ表面の光沢球混入による EMPTY 誤判定を防ぐ。
        # ColorClassifier に enable_specular_robust_saturation を伝播。
        # 2026-06-02: user viz 採用承認により default True に変更。
        enable_specular_robust_saturation: bool = True,
        # 設計C 事後復旧ゲート (2026-06-02):
        # True で STABLE 中の confirmed==EMPTY かつ CNN==HSV 持続合意セルを復旧する。
        # 2026-06-02: user viz 採用承認により default True に変更。
        # False に戻すには enable_stable_recovery_gate=False を明示する。
        enable_stable_recovery_gate: bool = True,
        # フェーズ A 精緻化 (2026-06-02): おじゃま視覚検知フラグ群。
        # enable_ojama_visual_detection: 親フラグ。True で全子フラグを有効化。
        # enable_ojama_visual_chain_exit: CHAIN → STABLE 復帰をお邪魔視覚検知に委譲。
        # enable_ojama_infer_guard: OJAMA_FALL → STABLE 直後に infer_placement を抑止。
        # enable_ojama_settle_detection: OJAMA_FALL 中お邪魔 count 不変で STABLE 復帰。
        # 2026-06-02: user viz 採用承認により全フラグ default True に変更。
        # 親フラグ=True により子フラグも全て有効になる (子個別は後方互換で True を明示)。
        enable_ojama_visual_detection: bool = True,
        enable_ojama_visual_chain_exit: bool = True,
        enable_ojama_infer_guard: bool = True,
        enable_ojama_settle_detection: bool = True,
        # 案B (第2の根本原因対処, 2026-07-24): OJAMA_FALL 退出条件を
        # 「全盤面ぷよ数が静止するまで待つ」方式 (GravitySettle と同型) に切替える。
        # True にすると OjamaVisualDetector.enable_ojama_fall_board_settle に加え、
        # OjamaPhaseDetector.defer_ojama_fall_exit_to_visual も同時に True になり、
        # 退出判定を視覚 settle 判定に一本化する (地雷=score 側の無条件 STABLE 復帰
        # を構造的に無効化)。
        # 2026-07-24 採用 (default ON): A/B 検証で次ツモ遅延 2.80s→0.65s・
        # 浮き誤消去 -28%・採用 +38 (user viz 全画像レビューで「全て after の方が
        # 品質高い」承認)。False を明示指定すれば旧挙動 (bit-identical) に戻せる
        # (backwards compat)。
        enable_ojama_fall_board_settle: bool = True,
        # 機能B: score 急増で即 CHAIN 突入する早期発火 (2026-06-02)。
        # True にすると自 side の score_delta >= CHAIN_SCORE_EARLY_FIRE_DELTA の frame で
        # VideoChainTracker の puyo 減少検知を待たずに即 CHAIN state に突入する。
        # score が取れない / OCR 失敗フレームでは従来の VideoChainTracker 経路が維持される
        # (OR 追加のため退行ゼロ)。 デフォルト False = 従来挙動完全維持 (backwards compat)。
        enable_chain_score_early_fire: bool = False,
        # 機能C: CHAIN → STABLE 遷移直後の confirmed 凍結 (2026-06-02)。
        # True にすると CHAIN→STABLE 復帰から CHAIN_EXIT_WARMUP_SEC 秒間 confirmed 更新を
        # 凍結し、エフェクト残光色が _merge_diff_only 経由で confirmed に混入するのを防ぐ。
        # 既存 enable_warmup_guard の CHAIN→STABLE 特化版。時間ベースで fps 非依存に実装。
        # デフォルト False = 従来挙動完全維持 (backwards compat)。
        enable_chain_exit_warmup: bool = False,
        # 機能D: 連鎖開始 掛け算式 検知 (2026-06-02)。
        # True にすると score ROI の OCR が None (掛け算式表示で NCC conf 低下) かつ
        # ink_ratio > CHAIN_FORMULA_INK_RATIO_MIN かつ last_score > 0 が
        # CHAIN_FORMULA_CONSEC_FRAMES 連続で成立した frame で即 CHAIN state に突入する。
        # 機能B (score 急増経路) と独立フラグ。どちらか一方だけ有効にすることも可能。
        # 誤発火ガード: ink_ratio (黒 ROI 除外) + 連続 2frame + last_score>0 の AND 条件。
        # 2026-06-03 採用 (default ON): v70 等 16 動画 A/B で連鎖突入 0.2-2.5s 早期化 +
        # baseline 黙過の小連鎖検知、non_stable最大 62→46 / corruption 1.57%→1.35% /
        # 空→色FP 0.27%→0.10% と全軸改善 (user viz 承認)。
        # False に戻すには enable_chain_formula_detection=False を明示する。
        enable_chain_formula_detection: bool = True,
        # 修正D (2026-07-24): 機能D 疑似発火の起点盤面を ChainSimulator で
        # 事前検証する。真因診断 (_diag_false_event_source_2026-07-24.py) で
        # 機能D 早期発火 77件中35件=45.5%が「連鎖ゼロの起点盤面」からの
        # 疑似発火 (偽イベント) と確定。True で before_board を simulate し、
        # chain_count==0 なら疑似発火を抑制、chain_count>0 なら固定1でなく
        # 実測値を使う。
        # 2026-07-24 採用 (default ON): 物理採点+独立診断で偽イベント率
        # 27.5%→0% と全面改善 (user viz 承認)。False に戻すには
        # enable_chain_formula_simulate_verify=False を明示する
        # (旧挙動・bit-identical、backwards compat のため維持)。
        enable_chain_formula_simulate_verify: bool = True,
        # 案 Y-4 HSV-first commit + deferred consensus (2026-06-03)。
        # True にすると infer_placement が HSV 拮抗と判定した着地 2 候補を保留し、
        # 後続フレームの CNN==HSV consensus 投票で確定させる。
        # default False = 従来挙動完全維持 (backwards compat)。
        enable_hsv_deferred_consensus: bool = False,
        # 不具合B 対処: 予告おじゃま発光ガード (2026-06-04)。
        # True にすると相手連鎖の予告おじゃま演出による盤面上部多色発光を検知し、
        # STABLE 中の confirmed_board を frozen_board で保護する。
        # 黄ぷよに発光が重なり黄(4)→おじゃま(9)誤認→連鎖誤消去を防ぐ。
        # default False = 従来挙動完全維持 (backwards compat)。
        enable_ojama_warning_glow_guard: bool = False,
        # 案P3: CHAIN_MAX_HOLD_SEC 超過後の ojama 保留を無効化 (2026-06-05)。
        # True にすると active_chain が CHAIN_MAX_HOLD_SEC 超過で強制クリアされた frame で
        # chain_max_hold_expired=True を DetectorSignals に乗せ、ChainPhaseDetector が
        # ojama_top_positive による STABLE 復帰保留をスキップして強制 STABLE に遷移する。
        # 安全弁 (CHAIN_MAX_HOLD_SEC) を本来機能させ「連鎖 6.87 秒過剰保持」 を解消する。
        # default False = 従来挙動完全維持 (backwards compat)。A/B 対照実験はフラグで行う。
        enable_chain_max_hold_override: bool = False,
        # 案X*(A)(B)+warmup: NextSlide signal による CHAIN 即終了 (2026-06-05)。
        # True にすると以下を一括有効化する:
        #   (A) 機能D (掛け算式) 再点火抑制: 既に CHAIN 中 (active_chain 有効) なら
        #       機能D の発火をスキップし、max_until の延長を止める。
        #   (B) NextSlide signal で CHAIN 即終了: slide_motion=True が確認された
        #       side の active_chain を即クリアし、timing hold や max_until に
        #       関係なく CHAIN 状態を解放する。次ツモスライド = 連鎖確実終了の証拠。
        #   warmup 連動: 内部で enable_chain_exit_warmup を自動有効化し、
        #       早期終了直後 CHAIN_EXIT_WARMUP_SEC 秒間 confirmed 凍結を適用して
        #       エフェクト残光色の混入を防ぐ。
        # NextDetector 精度 100% 確認済 (memory: project_next_detector_perfect_accuracy.md)。
        # default False = 従来挙動完全維持 (backwards compat)。
        enable_chain_exit_next_signal: bool = False,
        # feat/gravity-settle-2026-06-05: 連鎖終了直後の GRAVITY_SETTLE 状態を有効化。
        # True にすると CHAIN → GRAVITY_SETTLE → STABLE の遷移経路を有効化し、
        # 連鎖終了直後の重力 settle/着地中を採点外 (confirmed 凍結) として扱う。
        # 案X (enable_chain_exit_next_signal) と組み合わせて使う前提。
        # 本フラグ ON 時は内部で enable_chain_exit_next_signal も強制 ON にする
        # (連鎖が正確に終わらないと GRAVITY_SETTLE に入れないため)。
        # 2026-06-06 採用: 連鎖過剰保持の本質修正 = settle 状態。退行ゼロ + 連鎖境界正確化。
        # default True = 採用済み。False で無効化可 (backwards compat)。
        enable_gravity_settle_state: bool = True,
        # 案γ: CHAIN 中 slide_motion=True (次ツモスライド) が来た場合、
        # ojama_top_positive による CHAIN 過剰保持 (ojama-hold ガード) を上書きして終了。
        # enable_chain_ojama_exit=True かつ ojama_top_positive=True のときに slide が
        # None 返しを貫通する問題 (v89 t35.2-39.67 連鎖過剰保持) を解消する。
        # 2026-06-06 採用: corr +0.004% 誤差・v89 連鎖過剰保持完全解消・置き認識復活で
        # user 目視 OK + 退行なし → default True に昇格。False で無効化可。
        enable_slide_override_ojama_hold: bool = True,
        # 案1 (2026-07-23): estimated_board の stale_hold フォールバック。
        # True にすると CHAIN/GRAVITY_SETTLE 中に起点盤面 simulate が
        # chain_count=0 等で新規推定を計算できないフレームで、estimated_board
        # を None にせず直前の推定盤面 (無ければ起点盤面) を保持する
        # (board_provenance="chain_estimate_stale_hold")。
        # c62 1P estimated_board カバレッジ崩壊 (9.8%) の主因である疑似連鎖
        # イベント early-fire 連発への対処 (診断:
        # recognition_diag_c62_1p_estimate_collapse/summary.txt)。
        # confirmed_board (STABLE 評価用) は一切変更しない。
        # user viz 承認前の savepoint 実装のため default True だが、
        # False を渡せば従来挙動 (常に None) に完全に戻せる (backwards compat)。
        enable_chain_estimate_stale_hold: bool = True,
        # #45 おじゃま merge 統合修正 案(a) (2026-07-24): 重力フィルタ支持緩和。
        # 案B (enable_ojama_fall_board_settle) 適用後、 _merge_diff_only 内の
        # `_apply_gravity_filter` が F ガード (empty_to_color_guard) 起因で
        # EMPTY のまま残った cell を「浮き判定の gap」として誤扱いし、
        # 積もり中のおじゃまを浮きぷよ誤消去する副作用が判明。
        # True にすると `_apply_gravity_filter` に empty_to_color_guard
        # (多数決板) を support_board として渡し、そのガードで非 EMPTY と
        # 裏付けられる cell は gap 扱いしない。
        # (b) enable_ojama_fall_board_settle と独立に A/B 切り分け可能。
        # 2026-07-24 採用 (default ON): 案B (enable_ojama_fall_board_settle)
        # と併せた A/B 検証 (user viz 全画像レビュー承認) で採用。False を
        # 明示指定すれば旧挙動 (bit-identical) に戻せる (backwards compat)。
        enable_gravity_filter_support: bool = True,
        # #45 おじゃま merge 統合修正 案(b) (2026-07-24): 退出 merge 書込値の
        # 多数決化。 NON-STABLE → STABLE 復帰時の EMPTY→色 遷移ガード分岐で、
        # 従来は単一フレーム CNN 値 (cnn_v) と多数決値 (guard_v) が不一致だと
        # 却下していた (= 退出直前の単一フレームちらつきで正当な色復帰も却下)。
        # True にすると guard_v が EMPTY でない場合、 cnn_v でなく guard_v を
        # 書き込む (多数決値を信頼)。 (a) と独立に A/B 切り分け可能。
        # 2026-07-24 採用 (default ON): 案B と併せた A/B 検証 (user viz
        # 全画像レビュー承認) で採用。False を明示指定すれば旧挙動
        # (bit-identical) に戻せる (backwards compat)。
        merge_use_majority_value: bool = True,
        # DriftDetector 再同期ループ暴走ガード (2026-07-25, c34 実測)。
        # ガード1: True にすると試合開始から
        # DRIFT_RESYNC_MATCH_START_GUARD_SEC 秒以内は DriftDetector の
        # needs_resync を無視する (DriftDetector.update 自体は毎 frame 呼ぶため
        # 内部の連続乖離カウンタは追跡され続ける、reset だけ抑制される)。
        # 2026-07-25 user レビュー (c34 v6) 承認で既定 ON 化。
        # False を明示指定すれば旧挙動 (bit-identical) に戻せる (backwards compat)。
        enable_drift_resync_match_start_guard: bool = True,
        # ガード2: True にすると OnlineHsvCalibrator の較正済み色数
        # (_online_hsv_injected_colors) が DRIFT_RESYNC_MIN_CALIBRATED_COLORS
        # 未満の間、needs_resync を無視する。ガード1と独立に ON/OFF 可能。
        # 2026-07-25 user レビュー (c34 v6) 承認で既定 ON 化。
        # False を明示指定すれば旧挙動 (bit-identical) に戻せる (backwards compat)。
        enable_drift_resync_hsv_gate: bool = True,
        # cycle 31 baseline_broken 自己リセット 制御フラグ (2026-07-25,
        # A/B 計測用)。False にすると block 全体 (_check_baseline_broken_reset)
        # をスキップする。default True = 従来挙動完全維持 (backwards compat)。
        enable_baseline_broken_reset: bool = True,
        # 限定緩和版: True で STABLE 突入から BASELINE_BROKEN_STABLE_GRACE_SEC
        # 秒間はカウンタ加算を抑制する。enable_baseline_broken_reset=False の
        # 場合は無視される。default False = 従来挙動完全維持 (backwards compat)。
        enable_baseline_broken_grace: bool = False,
        # 列ゲート緩和 (enable_column_partial_support, 2026-07-25, A/B 計測用)。
        # True で設計C 事後復旧ゲートの安全弁C (浮き判定) が
        # stable_recovery_counters 進行中の支持セルを浮き扱いしなくなる。
        # 診断 (_diag_recovery_cell_timeseries_2026-07-25.py) で「8f 到達直前で
        # 毎回リセットされる」列ゲート型の未反映を救済する狙い。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        # user 承認前の savepoint 実装のため default OFF 固定。
        enable_column_partial_support: bool = False,
        # 前試合盤面残骸リーク修正 (feat/recognition-postchain-fix-2026-07-23,
        # A/B 計測用)。True で試合境界 (is_match_active False→MENU 強制) 時に
        # non_stable_cnn_history / stable_recovery_counters / recovery_cells /
        # stable_warmup_remaining / next_queue も完全クリアし、前試合終盤の
        # ぷよが次試合序盤に幽霊セルとして書き戻るのを防ぐ。
        # 2026-07-25 user レビュー (c34 v6) 承認で既定 ON 化。
        # False を明示指定すれば旧挙動 (bit-identical) に戻せる (backwards compat)。
        enable_match_start_full_clear: bool = True,
        # score-reset 境界誤発火修正 (2026-07-26, feat/recognition-postchain-fix-2026-07-23,
        # diag_v29_mid_resetlog.log で発見): 片側のみ score が急落して見える
        # 単発フレーム (例: 2P=40031 不変なのに 1P だけ 48077→0) が観測された。
        # 実フレーム精査 (score ROI 目視 + ScoreOcr 直接実行) の結果、これは
        # OCR 誤読ではなく「両者同時の本物の試合境界フェード演出」を、
        # 1P/2P 各々の OCR 信頼度ゲートが数フレームずれて通過したことによる
        # 見かけ上の片側化と判明 (掛け算式表示は avg_conf ゲートで既に正しく
        # None 扱いされており、0 が漏れ出すことはない)。
        # _is_score_reset_boundary を (a) 両側同時条件のみ許可 (b) 3 フレーム
        # 連続成立必須、に厳格化することで、この数フレームのずれを吸収し
        # 「片側だけ発火して見える」見かけ上の異常を解消する。ただし
        # video_29 該当窓の再検証では 5 件の reset は全て本物の試合境界と
        # 確認され、発火回数・収集行数は本修正前後で変化しなかった
        # (= labeled_win 行数 -30% の真因はこの修正だけでは説明できない、
        # 別途要調査)。既定 True (誤発火の疑いがあった箇所の防御的厳格化)。
        # False を明示指定すれば旧 (片側 OR・デバウンス無し) 挙動に戻せる
        # (backwards compat)。enable_match_start_full_clear=False の場合は
        # そもそも本ブロックが無効なので本フラグは参照されない。
        enable_score_reset_strict: bool = True,
        # 復旧カウンタ carryover (feat/recognition-postchain-fix-2026-07-23,
        # #51, 2026-07-26, A/B 計測用): True で STABLE→NON-STABLE 遷移時の
        # stable_recovery_counters/recovery_cells 即クリアを保留し、非
        # STABLE 滞在が RECOVERY_COUNTER_CARRYOVER_MAX_SEC (既定 2.0秒) 以内
        # なら STABLE 復帰後も引き継ぐ (8f 到達直前の未反映化への対処、
        # diag `recovery_cell_timeseries_2026-07-25`)。
        # 2026-07-27 user レビュー (video_84, #51系3修正全6観点OK) 承認で
        # 既定 ON 化。False を明示指定すれば旧挙動 (bit-identical) に戻せる
        # (backwards compat)。
        enable_recovery_counter_carryover: bool = True,
        # CNN 乱高下セル HSV フォールバック (#51 後半, 2026-07-26, A/B 計測用)。
        # True で深部セルの CNN 判定境界張り付き反転 (9↔1↔0↔4 等) を検出し、
        # その間 HSV 出力を復旧ゲートの合意値とみなす
        # (詳細は src/board_state_machine.py の定数定義部を参照)。
        # 2026-07-27 user レビュー (video_84, #51系3修正全6観点OK) 承認で
        # 既定 ON 化。False を明示指定すれば旧挙動 (bit-identical) に戻せる
        # (backwards compat)。
        enable_cnn_flicker_hsv_fallback: bool = True,
        # 色→空凍結の修正3点セット③ (feat/recognition-postchain-fix-2026-07-23,
        # 2026-07-27): 初回STABLE確定の多数決ガード。 True にすると
        # BoardStateMachine の baseline is None (初回確定) 時、直前
        # NON-STABLE滞在中に蓄積した non_stable_cnn_history の多数決で
        # 初回confirmedを構成する (fallback=new_cnn で観測不足セルはEMPTY化しない)。
        # 2026-07-27 user レビュー (video_84, #51系3修正全6観点OK) 承認で
        # 既定 ON 化。False を明示指定すれば旧挙動 (bit-identical) に戻せる
        # (backwards compat)。
        enable_initial_confirm_vote: bool = True,
        initial_confirm_min_votes: int = (
            DEFAULT_INITIAL_CONFIRM_MIN_VOTES
        ),
        # 大 ROI 走査 (match_end / telop) の間引き (2026-07-30、2026-07-31 既定ON)。
        # 走査を飛ばすので原理的に bit-identical にならないが、試合終了時刻を
        # またぐ窓での実測 (3動画 x 2イベント = 1800フレーム) で
        # **試合終了検出のずれ 0フレーム・盤面差分 0/1800** を確認した。
        # 遅延が伝播しないのは hard_match_off が score_zero_both との OR で、
        # score_zero / MatchStateDetector は間引き対象外のため独立経路が
        # 同一フレームで発火するから (設計時の有界性の主張が実証された)。
        # 速度 +19.5〜53.4%。False で従来の毎フレーム走査に戻る。
        enable_large_roi_throttle: bool = True,
        large_roi_throttle_frames: int = LARGE_ROI_THROTTLE_FRAMES,
        # 色→空 HSV 照合ガード (2026-07-30): True で NON-STABLE→STABLE 復帰
        # merge の色→空 遷移について HSV が色を保持する cell を消さない。
        # 光沢→空 の単一フレーム CNN 誤読が無投票消去され gravity filter で
        # 上のぷよまで連鎖消去される列デッドロック (c34 1P col=1, frame 14332
        # 実測) を根で止める。ただし 4動画測定 (c34/c58/c26/c69) で c58/c26 の 2P は
        # tail 悪化、c26/c69 の 1P は効果ゼロ、8フレーム達成率は OFF/ON 不変と判明
        # (data/verify/placement_confirm_frames_generalization_2026-07-30)。
        # 汎化未確認のため default OFF。True で有効化 (backwards compat)。
        enable_puyo_to_empty_hsv_guard: bool = False,
        # 復旧ゲート方向別しきい値 非対称化 (2026-07-30, A/B 計測用): True で
        # 方向1 (空→色) のみ recovery_add_min_frames (既定 4) で復旧させ、
        # 方向2/3 (色→空/色→色) は 8 frame 維持。「誤認が治るまでのラグ」短縮用。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        # user 承認前の savepoint 実装のため default OFF 固定。
        enable_asymmetric_recovery_min_frames: bool = False,
        recovery_add_min_frames: int = STABLE_RECOVERY_ADD_MIN_FRAMES,
    ) -> None:
        # B2 (A/B 対照実験): BG_FP_FORCE_MAX_PUYO を instance 変数で上書き可能に。
        # None なら class attribute 値 (= 144) を使う。
        self._bg_fp_force_max_puyo: int = (
            int(bg_fp_force_max_puyo)
            if bg_fp_force_max_puyo is not None
            else self.BG_FP_FORCE_MAX_PUYO
        )
        self._reader = image_reader
        # fix/v70-zeropatch-redyellow (2026-06-02): 赤色相折り返し補正を
        # ColorClassifier に伝播する (HybridClassifier._hsv 経由も含む)。
        if enable_red_hue_wrap_fix:
            clf = getattr(image_reader, "_classifier", None)
            hsv_clf = getattr(clf, "_hsv", clf)
            if hsv_clf is not None and hasattr(hsv_clf, "_enable_red_hue_wrap_fix"):
                hsv_clf._enable_red_hue_wrap_fix = True
        # 案D (fix/v70-zeropatch-redyellow): 光沢ハイライト除外彩度計算を
        # ColorClassifier に伝播する (HybridClassifier._hsv 経由も含む)。
        if enable_specular_robust_saturation:
            clf = getattr(image_reader, "_classifier", None)
            hsv_clf = getattr(clf, "_hsv", clf)
            if hsv_clf is not None and hasattr(
                hsv_clf, "_enable_specular_robust_saturation"
            ):
                hsv_clf._enable_specular_robust_saturation = True
        self._match_detector = match_state_detector
        # A: 試合境界補強 detector 群 (memory: 試合判定甘さ対策)。
        # 既存単独動作との backward compat 維持のため全て optional。
        self._score_zero_detector = score_zero_detector
        self._match_end_detector = match_end_detector
        self._telop_detector = telop_detector
        # Phase I.c: OnlineHsvCalibrator (動画別 HSV 自動学習)
        # STABLE 中に信頼サンプル蓄積 → ready 後に ColorClassifier ranges を上書き
        self._online_hsv = online_hsv
        self._online_hsv_injected: bool = False  # 任意の色が 1 つ以上 inject 済か
        # cycle 71v (2026-05-14): 段階的 inject 用. 既に inject 済の色集合.
        self._online_hsv_injected_colors: set[int] = set()
        # 着地直後 grace period: TSUMO_FALL→STABLE 遷移後 N frame は
        # CNN を信用せず inferred_landing で confirmed_board を hold。
        # ユーザー提案 (2026-05-10): 「置いた直後の誤認は推論で防げる」
        # フレーム定数→時間定数化 Stage1 (2026-07-25): tuple 第 3 要素に
        # time_sec 基準の満了時刻 (grace_until_time) を追加。実ロジックは
        # こちらを正として使い、第 1 要素 (frame 基準) は既存互換のため残置。
        self._landing_grace_1p: tuple[int, Board, float] | None = None
        self._landing_grace_2p: tuple[int, Board, float] | None = None
        # 2026-05-10 FIX-B: 5→10 frame に延長 (置いた直後の認識ぶれ抑制)
        # 着地直後の grace period. CNN/HSV の不安定 (= 着地直後の光・揺らぎ) で
        # confirmed が誤更新されるのを抑止. cycle 71h: 60→10 復帰 (= cycle 71f 互換).
        # cycle 71k (パターン 1 S2): 着地後の修正速度を上げるため grace を撤回.
        # cycle 71v (2026-05-14): v70 レビューで「置いた直後誤認からの修正に時間が
        # かかる」 報告。 grace=0 だと TSUMO_FALL→STABLE merge が着地直後の CNN
        # 雑音 (フラッシュ、 演出) を confirmed に取り込んでしまう。 物理推論された
        # final_board (= NEXT pair 色由来、 100% 精度) を hold するが、 長すぎると
        # grace が cell を上書きし続けて long-term vote (= 18 frame 0.9 ratio で発火)
        # の補正が「翌 frame で grace に消される」 ループになる。
        # 2026-05-15: 36 → 12 (= 0.2s @ 60fps) に短縮。 着地フラッシュ抑制には十分、
        # grace 終了後は long-term vote が CNN majority で 0.3s 以内に補正できる。
        # cycle 29 (2026-05-18): 12 → 5 frame に短縮 + 起動を NEXT 移動検知に変更。
        # NEXT 変化 = 着地確定 signal、 state machine 詰まり (v97 53 秒問題) を救済。
        self.LANDING_GRACE_FRAMES: int = 5
        # フレーム定数→時間定数化 Stage1 (2026-07-25): 実ロジックは秒定数を正とする
        # (frame 定数は既存互換のため残置)。60fps では bit-identical。
        self.LANDING_GRACE_SEC: float = self.LANDING_GRACE_FRAMES / 60
        # cycle 71h (ユーザー要件「1 秒後に完璧」 の本質対策):
        # TSUMO_FALL→STABLE 着地時に inferred_landing の追加 cells 位置を記録、
        # 後続 LANDING_VOTE_FRAMES の cnn_board で同位置 cell 色を蓄積.
        # 蓄積期間終了時に最頻値で confirmed を更新 = 1 秒後の正しい色を反映.
        # _pending_landing_vote: side -> list of (start_frame, list[(row, col)],
        #                                         vote_buffer: dict[(row,col), list[int]])
        self._pending_landing_vote_1p: list[dict] = []
        self._pending_landing_vote_2p: list[dict] = []
        self._chain_hold_per_step_sec = (
            chain_hold_per_step_sec
            if chain_hold_per_step_sec is not None
            else self.CHAIN_HOLD_PER_STEP_SEC
        )
        # A0 (2026-07-24): 固定項 (既定 0.0 = 従来式と bit-identical)。
        self._chain_hold_base_sec = (
            chain_hold_base_sec
            if chain_hold_base_sec is not None
            else self.CHAIN_HOLD_BASE_SEC
        )
        # A0 (2026-07-24): 安全弁上限の上書き (既定 5.0 = backwards compat)。
        self._chain_max_hold_sec = (
            chain_max_hold_sec
            if chain_max_hold_sec is not None
            else self.CHAIN_MAX_HOLD_SEC
        )
        # 時系列平均 (δ): 直近 N frame の cell 単位 majority vote。
        # CNN ぶれを抑え、state machine の安定化を狙う。
        self._smoothing_n = max(1, int(temporal_smoothing))
        self._cnn_history_1p: deque[Board] = deque(maxlen=self._smoothing_n)
        self._cnn_history_2p: deque[Board] = deque(maxlen=self._smoothing_n)
        # 直近受信 ChainEvent と有効期限 (1P/2P 別)。
        self._active_chain_1p: ChainEvent | None = None
        self._chain_until_1p: float = 0.0
        self._active_chain_2p: ChainEvent | None = None
        self._chain_until_2p: float = 0.0
        # 根治 (2026-07-23): CHAIN → GRAVITY_SETTLE → STABLE 経路でも連鎖後
        # final_board 反映 (Phase C-6 の C, _step_side 内) を機能させるための退避先。
        # enable_gravity_settle_state=True (default) では CHAIN は必ず
        # GRAVITY_SETTLE を経由してから STABLE に遷移するため、その遷移フレーム
        # では active_chain_* は既に None 化されている (= chain_event 引数も None)。
        # active_chain を None にする直前にこのフィールドへ退避しておき、
        # GRAVITY_SETTLE→STABLE 遷移時の fallback 参照元として使う。
        # enable_gravity_settle_state=False の場合は退避が消費される経路
        # (GRAVITY_SETTLE 遷移) 自体が発生しないため、既存動作に影響しない。
        self._last_chain_event_for_settle_1p: ChainEvent | None = None
        self._last_chain_event_for_settle_2p: ChainEvent | None = None
        # 反復4 (2026-07-23): confirmed_board=None 化の理由分類用 診断計装
        # (SideResult.board_none_reason)。挙動には一切影響しない。
        # _ever_had_confirmed_*: この試合で一度でも confirmed_board が
        #   非 None になったことがあるか (cold_start 判定用)。
        # _pending_menu_reset_*: is_match_active=False → MENU 強制
        #   (board_state_machine.py:480-488) が発生し、まだ STABLE で
        #   再確定していない状態かどうか (menu_reset 判定用)。
        self._ever_had_confirmed_1p: bool = False
        self._ever_had_confirmed_2p: bool = False
        self._pending_menu_reset_1p: bool = False
        self._pending_menu_reset_2p: bool = False
        # 反復5 (2026-07-23): 物理推論スルー機構 (根治本体)。
        # Step1 診断で「chain_event.before_board (= 起点盤面) は 85.7% の
        # ケースでそのまま使える」ことを確認済み。ctx.confirmed_board が
        # None 化 (drift resync 等) していても before_board は独立に
        # VideoChainTracker が捕捉するため影響を受けにくい。
        # CHAIN/GRAVITY_SETTLE 中、この起点から ChainSimulator で連鎖を
        # 前進させた盤面を SideResult.estimated_board として公開する
        # (confirmed_board 自体は一切変更しない = 標準 eval 経路への
        # 影響ゼロ、Step4 backward compat 要件)。
        self._chain_estimate_result_1p: "ChainResult | None" = None
        self._chain_estimate_result_2p: "ChainResult | None" = None
        self._chain_estimate_trigger_1p: float = 0.0
        self._chain_estimate_trigger_2p: float = 0.0
        self._chain_estimate_end_1p: float = 0.0
        self._chain_estimate_end_2p: float = 0.0
        # Step3(a) 答え合わせ: score 由来 chain_count (ev.chain_count) と
        # 物理予測 chain_count (before_board を simulate した実測値) が
        # 一致しない場合 True (= 起点盤面が疑わしい、低信頼度)。
        self._chain_estimate_low_confidence_1p: bool = False
        self._chain_estimate_low_confidence_2p: bool = False
        # 案1 (2026-07-23): stale_hold フォールバック用 state。
        # _chain_estimate_last_board_Xp: 直近に「成功した」推定盤面
        #   (result が非 None で計算できたもの)、または CHAIN 突入時点の
        #   起点盤面 (cold start フォールバック)。CHAIN/GRAVITY_SETTLE を
        #   抜けたら None にクリアする。
        # _chain_estimate_stale_since_Xp: 現在の stale_hold 連続適用が
        #   開始した time_sec (None = stale_hold 非適用中)。
        #   CHAIN_ESTIMATE_STALE_HOLD_MAX_SEC 超過判定に使う安全弁用。
        self._chain_estimate_last_board_1p: "Board | None" = None
        self._chain_estimate_last_board_2p: "Board | None" = None
        self._chain_estimate_stale_since_1p: float | None = None
        self._chain_estimate_stale_since_2p: float | None = None
        self._enable_chain_estimate_stale_hold: bool = bool(
            enable_chain_estimate_stale_hold,
        )
        # 反復5 修正 (2026-07-23): Step3(b)(c) 事後検証の進行 state。
        # {"expected": Board, "cnn_history": list[Board]} または None
        # (検証中でない)。Phase C-6 の C で final_board 適用直後にセットし、
        # 直近 CHAIN_VERIFY_FRAMES 分の STABLE cnn_board が集まったら
        # 多数決盤面と照合して補正する (_update_chain_estimate_verification)。
        self._chain_verify_pending_1p: dict | None = None
        self._chain_verify_pending_2p: dict | None = None
        # match active hysteresis 用
        self._last_active_frame_idx: int = -1
        self._match_active_started_frame: int = -1
        # フレーム定数→時間定数化 Stage1 (2026-07-25): time_sec 基準の対を追加。
        # 実ロジックはこちらを正として使う (frame 版は既存互換のため残置)。
        self._last_active_frame_time: float = -1.0
        self._match_active_started_time: float = -1.0
        # サイクル66 (2026-05-11): NEXT 累積色制約用 — 試合開始から最初の連鎖
        # 発火まで「累積 NEXT 色 = field 色 count」 制約を維持. 連鎖 / おじゃま
        # 落下発生時は invalidate (= 単純制約が崩れる).
        # 1P/2P 別.
        # tsumo_consumed: 色 → 累積数 (1P/2P 別、 NEXT 履歴遷移で +1)
        # last_next: 最後に観測した NEXT pair (履歴遷移検出用)
        # constraint_valid: True なら制約適用、 False なら無効 (連鎖後/おじゃま後)
        from collections import Counter as _Counter, deque as _deque
        # サイクル66: 累積確定 tsumo (= TSUMO_FALL→STABLE 後に commit)
        self._tsumo_count_1p: _Counter = _Counter()
        self._tsumo_count_2p: _Counter = _Counter()
        # サイクル67: in-flight tsumo (= NEXT 消費後、 着地完了前)
        # ペア (color1, color2) を queue 化、 TSUMO_FALL→STABLE で pop して commit
        self._pending_tsumo_1p: _deque = _deque()
        self._pending_tsumo_2p: _deque = _deque()
        self._last_seen_next_1p: tuple[int, int] | None = None
        self._last_seen_next_2p: tuple[int, int] | None = None
        self._constraint_valid_1p: bool = True
        self._constraint_valid_2p: bool = True
        # cycle 29 (2026-05-18): NEXT 変化 = 着地確定 signal として
        # (frame_idx, falling_pair) を保存。 次の _step_side で grace +
        # landing_vote を起動 (= state machine 詰まりを救済)。
        self._landing_pending_1p: tuple[int, tuple[int, int]] | None = None
        self._landing_pending_2p: tuple[int, tuple[int, int]] | None = None
        # 着地色修正 案1 修正版 (2026-06-01):
        # _landing_pending はNEXT変化フレームのgrace処理でクリアされるため
        # 実際の着地フレーム(TSUMO_FALL→STABLE)まで持続しない。
        # _last_consumed_color_Xp = 最後にNEXTで消費されたツモ色を保持し
        # TSUMO_FALL→STABLE 着地時の falling_pair 取得に使用する。
        # セット: NEXT変化検知時 / クリア: TSUMO_FALL→STABLE着地infer完了後。
        self._last_consumed_color_1p: tuple[int, int] | None = None
        self._last_consumed_color_2p: tuple[int, int] | None = None
        # 色フリッカ根因への防御的修正 案(iii) (2026-07-25):
        # TSUMO_FALL→STABLE 着地 infer 完了時に _flag_landing_distrust_cells で
        # 検出した「疑わしいセル」座標集合 (1P/2P 別)。
        # _last_consumed_color_Xp と同じ寿命管理パターン:
        # セット: 着地 infer 完了時 (enable_placement_color_cnn_check=True 時のみ) /
        # クリア: _start_landing_vote 呼出後 (P7 に伝播済み)。
        # enable_placement_color_cnn_check=False (default) では常に空集合。
        self._landing_distrust_1p: set[tuple[int, int]] = set()
        self._landing_distrust_2p: set[tuple[int, int]] = set()
        # cycle 31 (2026-05-18, B 軸): baseline 整合性 check + 自己修復。
        # STABLE 中に baseline と CNN 出力の diff が連続異常なら baseline 壊れ
        # 判定 → state reset (= 試合 active 再起動 + bg_fp 再採取)。
        # v97 53 秒 TSUMO_FALL 詰まり問題への救済策。
        self._baseline_broken_consec_1p: int = 0
        self._baseline_broken_consec_2p: int = 0
        # cycle 31 baseline_broken 自己リセット 制御フラグ (2026-07-25,
        # A/B 計測用)。_check_baseline_broken_reset で参照する。
        self._enable_baseline_broken_reset: bool = bool(enable_baseline_broken_reset)
        self._enable_baseline_broken_grace: bool = bool(enable_baseline_broken_grace)
        # STABLE 突入時刻 (1P/2P 別、grace 判定用。-1.0 = 未記録)。
        self._stable_entered_time_1p: float = -1.0
        self._stable_entered_time_2p: float = -1.0
        # grace 抑制回数カウンタ (1P/2P 別、効果測定用)。
        self._baseline_broken_grace_suppressed_1p: int = 0
        self._baseline_broken_grace_suppressed_2p: int = 0
        # baseline_broken reset 実発火回数カウンタ (1P/2P 別、A/B 効果測定用)。
        # _baseline_broken_consec_Xp は発火の度 0 に戻るため累計には使えない。
        self._baseline_broken_reset_count_1p: int = 0
        self._baseline_broken_reset_count_2p: int = 0
        # cycle 71v-B (2026-05-15): 試合中に観測した色を永続記録 (= NEXT 履歴 cap
        # 8 でスクロールアウトしても UNKNOWN 化しない)
        self._ever_seen_colors_1p: set[int] = set()
        self._ever_seen_colors_2p: set[int] = set()
        # 背景 FP 自動採取済フラグ + frame buffer (Phase C-5: robust 化)
        self._bg_fp_captured: bool = False
        self._bg_frame_buffer: deque[np.ndarray] = deque(maxlen=5)
        # MatchStateDetector が試合中なのに NOT_IN_MATCH を返す bug の対策で
        # is_match_active を常に True に強制する option (デバッグ・レビュー・用)
        self._force_in_match = bool(force_in_match)
        # 案 R3 改: 動画 ID (per-video プロファイル自動ロード用)。
        # set_video_id() で外部からセット可能。bg_fp 採取完了時に自動ロード。
        self._video_id: str | None = None
        # next_detector (任意): あれば毎 frame の next_pair を signals に渡す
        self._next_detector = next_detector
        # Phase I R-7: NEXT ROI スライド motion 検出器 (1P/2P 別 instance)。
        # 毎 frame の (prev, curr) BGR 比較で「ツモが置かれた」signal を取得し
        # TsumoPhaseDetector の TSUMO_FALL → STABLE 遷移を補強。
        self._enable_slide_detector = bool(enable_next_slide_detector)
        self._slide_detector_1p: NextSlideDetector | None = (
            NextSlideDetector(side="1P")
            if self._enable_slide_detector else None
        )
        self._slide_detector_2p: NextSlideDetector | None = (
            NextSlideDetector(side="2P")
            if self._enable_slide_detector else None
        )
        # 前 frame の BGR (slide motion 比較用)。最初の 1 frame は None。
        self._prev_frame: np.ndarray | None = None
        # Score tracker (任意): ScoreOcr が無ければ delta=0 固定
        self._score_ocr = score_ocr
        self._score_tracker_1p: ScoreTracker | None = (
            ScoreTracker("1P", score_ocr) if score_ocr else None
        )
        self._score_tracker_2p: ScoreTracker | None = (
            ScoreTracker("2P", score_ocr) if score_ocr else None
        )
        # 連鎖検出 (任意): 無ければ chain_event 常時 None で動作
        self._chain_tracker_1p = chain_tracker_1p
        self._chain_tracker_2p = chain_tracker_2p
        # 修正C (2026-07-24): reset() で VideoChainTracker を再構築する際、
        # debounce 設定を引き継ぐために保持しておく (backwards compat:
        # tracker 未設定 / debounce_confirm_frames 未実装の duck-typed
        # スタブ (テスト用) なら既定値 DEBOUNCE_CONFIRM_FRAMES=1 にフォールバック)。
        self._chain_debounce_confirm_frames: int = getattr(
            chain_tracker_1p, "debounce_confirm_frames", DEBOUNCE_CONFIRM_FRAMES,
        )
        # cycle 71d (案 D8): VideoChainTracker への入力盤面に「前 frame の confirmed_board」 を
        # 使う. raw CNN 振動 (= cnn 32↔27 単発スパイク) を投票後の confirmed で吸収できる.
        # 初回 frame は confirmed が無いため raw CNN にフォールバック.
        self._prev_confirmed_1p: Board | None = None
        self._prev_confirmed_2p: Board | None = None
        # cycle 71j (案 1a): 直前 STABLE confirmed_board を保持し、 現 STABLE で
        # UNKNOWN cell があれば直前 STABLE 値で埋める. エフェクト下で CNN が
        # UNKNOWN を出した後の凍結を補完し、 viz の「?」 を解消.
        self._prev_stable_confirmed_1p: Board | None = None
        self._prev_stable_confirmed_2p: Board | None = None
        # cycle 71n (案 θ, 2026-05-13): STABLE 中の cell ごとに CNN 観測色履歴を保持.
        # N frame の最頻値が confirmed と長期不一致なら CNN 側で上書き
        # (= 「ずっと残る誤色」 を自動修正).
        self._stable_cnn_history_1p: dict[tuple[int, int], list[int]] = {}
        self._stable_cnn_history_2p: dict[tuple[int, int], list[int]] = {}
        # cycle 71f (提案 A): score の直近履歴を保持し、 試合 2 開始直後の演出で
        # MatchStateDetector が not_in_match と判定しても、 score が継続的に
        # 動いていれば「試合中」 を強制復帰させる. 60 frame = 1 秒分.
        self._recent_scores_1p: list[int | None] = []
        self._recent_scores_2p: list[int | None] = []
        # フレーム定数→時間定数化 Stage1 (2026-07-25): 各 score 観測の time_sec を
        # 並行保持し、 「直近 N 件」保持を「直近 SCORE_MOVE_WINDOW_SEC 秒」保持に
        # 変換する (_trim_score_window 参照)。
        self._recent_score_times_1p: list[float] = []
        self._recent_score_times_2p: list[float] = []
        # 設計C 事後復旧ゲート: フラグを保持し _build_state_machine / _step_side に伝播。
        self._enable_stable_recovery_gate: bool = bool(enable_stable_recovery_gate)
        # フェーズ A 精緻化 (2026-06-02): おじゃま視覚検知フラグ群。
        # enable_ojama_visual_detection = 親フラグ (True で子フラグも有効化)。
        # 個別フラグは親フラグ OFF でも True 指定可能 (細粒度制御用)。
        # default False = 従来挙動完全維持 (backwards compat)。
        _ovd_parent = bool(enable_ojama_visual_detection)
        self._enable_ojama_visual_detection: bool = _ovd_parent
        self._enable_ojama_visual_chain_exit: bool = (
            _ovd_parent or bool(enable_ojama_visual_chain_exit)
        )
        self._enable_ojama_infer_guard: bool = (
            _ovd_parent or bool(enable_ojama_infer_guard)
        )
        self._enable_ojama_settle_detection: bool = (
            _ovd_parent or bool(enable_ojama_settle_detection)
        )
        # 案B (第2の根本原因対処, 2026-07-24): OJAMA_FALL 全盤面 settle 判定フラグ。
        # 独立公開フラグにせず OjamaVisualDetector / OjamaPhaseDetector 両方の
        # 切替に直結させる (「ovd だけ直して地雷 (OjamaPhaseDetector 側の無条件
        # STABLE 復帰) を直し忘れる」構成を構造的に不能化する設計)。
        # default False = 従来挙動完全維持 (backwards compat)。
        self._enable_ojama_fall_board_settle: bool = bool(
            enable_ojama_fall_board_settle
        )
        # 案P3: CHAIN_MAX_HOLD_SEC 超過 ojama 保留無効化フラグ。
        # _build_state_machine 呼び出し前に格納が必要 (self.* 参照のため)。
        self._enable_chain_max_hold_override: bool = bool(enable_chain_max_hold_override)
        # 案γ: slide_motion で ojama-hold を上書きするフラグ。
        # _build_state_machine 呼び出し前に格納が必要 (引数として渡すため)。
        self._enable_slide_override_ojama_hold: bool = bool(
            enable_slide_override_ojama_hold
        )
        # feat/gravity-settle-2026-06-05: GRAVITY_SETTLE 状態フラグ。
        # _build_state_machine 呼び出し前に設定が必要 (引数として渡すため)。
        # gravity_settle=True の場合は chain_exit_next_signal も強制 ON にする
        # (連鎖が正確に終わらないと GRAVITY_SETTLE に入れないため)。
        _gs = bool(enable_gravity_settle_state)
        if _gs:
            enable_chain_exit_next_signal = True  # 内部強制 ON
        self._enable_gravity_settle_state: bool = _gs
        # #45 おじゃま merge 統合修正 案(a)(b) (2026-07-24):
        # _build_state_machine 呼び出し前に格納が必要 (引数として渡すため)。
        self._enable_gravity_filter_support: bool = bool(
            enable_gravity_filter_support
        )
        self._merge_use_majority_value: bool = bool(merge_use_majority_value)
        # 列ゲート緩和 (enable_column_partial_support, 2026-07-25):
        # _build_state_machine 呼び出し前に格納が必要 (引数として渡すため)。
        self._enable_column_partial_support: bool = bool(
            enable_column_partial_support
        )
        # 前試合盤面残骸リーク修正 (2026-07-23):
        # _build_state_machine 呼び出し前に格納が必要 (引数として渡すため)。
        self._enable_match_start_full_clear: bool = bool(
            enable_match_start_full_clear
        )
        # 復旧カウンタ carryover (2026-07-26):
        # _build_state_machine 呼び出し前に格納が必要 (引数として渡すため)。
        self._enable_recovery_counter_carryover: bool = bool(
            enable_recovery_counter_carryover
        )
        # CNN 乱高下セル HSV フォールバック (#51 後半, 2026-07-26):
        # _build_state_machine 呼び出し前に格納が必要 (引数として渡すため)。
        self._enable_cnn_flicker_hsv_fallback: bool = bool(
            enable_cnn_flicker_hsv_fallback
        )
        # 色→空凍結の修正3点セット③ (2026-07-27): 初回STABLE確定の多数決ガード。
        # _build_state_machine 呼び出し前に格納が必要 (引数として渡すため)。
        self._enable_initial_confirm_vote: bool = bool(
            enable_initial_confirm_vote
        )
        self._initial_confirm_min_votes: int = int(initial_confirm_min_votes)
        # 大 ROI 走査 (match_end 800x600 / telop 720x400) の間引き (2026-07-30)。
        # 既定 OFF = 従来通り毎フレーム走査で bit-identical。
        self._enable_large_roi_throttle: bool = bool(enable_large_roi_throttle)
        self._large_roi_throttle_frames: int = int(large_roi_throttle_frames)
        # 間引き時に流用する前回結果 (従来は update() 内で毎フレーム初期化していた)
        self._last_match_end_locked: bool = False
        self._last_telop_visible: bool = False
        self._last_telop_result: "TelopResult | None" = None
        # 色→空 HSV 照合ガード (2026-07-30): _build_state_machine 呼び出し前に
        # 格納が必要 (引数として渡すため)。
        self._enable_puyo_to_empty_hsv_guard: bool = bool(
            enable_puyo_to_empty_hsv_guard
        )
        # 復旧ゲート方向別しきい値 非対称化 (2026-07-30): _build_state_machine
        # 呼び出し前に格納が必要 (引数として渡すため)。
        self._enable_asymmetric_recovery_min_frames: bool = bool(
            enable_asymmetric_recovery_min_frames
        )
        self._recovery_add_min_frames: int = int(recovery_add_min_frames)
        # 追修 (2026-07-25): force_in_match=True 構成用の score リセット境界
        # 検知に使う前フレームスコアキャッシュ (enable_match_start_full_clear
        # 時のみ参照)。
        self._prev_score_for_reset_1p: int | None = None
        self._prev_score_for_reset_2p: int | None = None
        # 追修: score リセット境界の edge-trigger ラッチ (継続 near-zero 期間中の
        # 連続 self.reset() 発火を防ぐ。境界条件が一旦 False に戻るまで再発火しない)。
        self._match_start_boundary_latched: bool = False
        # score-reset 境界誤発火修正 (2026-07-26): strict モードの厳格化条件を
        # 保持し、デバウンス用連続フレームカウンタを管理する。
        self._enable_score_reset_strict: bool = bool(enable_score_reset_strict)
        self._score_reset_boundary_streak: int = 0
        # DriftDetector 再同期ループ暴走ガード (2026-07-25, c34 実測)。
        # 個別 flag で独立 ON/OFF 可能 (_step_side の needs_resync 分岐で参照)。
        self._enable_drift_resync_match_start_guard: bool = bool(
            enable_drift_resync_match_start_guard,
        )
        self._enable_drift_resync_hsv_gate: bool = bool(
            enable_drift_resync_hsv_gate,
        )
        # デバッグカウンタ: 各ガードが needs_resync を抑制した回数 (1P/2P 別)。
        # 効果測定用 (_diag 系スクリプトから読み出す想定)。
        self._drift_resync_start_guard_suppressed_1p: int = 0
        self._drift_resync_start_guard_suppressed_2p: int = 0
        self._drift_resync_hsv_gate_suppressed_1p: int = 0
        self._drift_resync_hsv_gate_suppressed_2p: int = 0
        # 1P/2P state machine (独立)
        # B1 (M1 warmup guard): enable_warmup_guard=True で STABLE 遷移直後 N frame confirmed 凍結。
        # フェーズ A 精緻化: OjamaVisualDetector 登録フラグを伝播。
        self._sm_1p = self._build_state_machine(
            stable_frame_count, enable_warmup_guard=enable_warmup_guard,
            enable_stable_recovery_gate=enable_stable_recovery_gate,
            enable_ojama_visual_detection=self._enable_ojama_visual_detection,
            enable_ojama_visual_chain_exit=self._enable_ojama_visual_chain_exit,
            enable_ojama_settle_detection=self._enable_ojama_settle_detection,
            enable_ojama_fall_board_settle=self._enable_ojama_fall_board_settle,
            enable_chain_max_hold_override=self._enable_chain_max_hold_override,
            enable_gravity_settle_state=self._enable_gravity_settle_state,
            enable_slide_override_ojama_hold=self._enable_slide_override_ojama_hold,
            enable_gravity_filter_support=self._enable_gravity_filter_support,
            merge_use_majority_value=self._merge_use_majority_value,
            enable_column_partial_support=self._enable_column_partial_support,
            enable_match_start_full_clear=self._enable_match_start_full_clear,
            enable_recovery_counter_carryover=self._enable_recovery_counter_carryover,
            enable_cnn_flicker_hsv_fallback=self._enable_cnn_flicker_hsv_fallback,
            enable_initial_confirm_vote=self._enable_initial_confirm_vote,
            initial_confirm_min_votes=self._initial_confirm_min_votes,
            enable_puyo_to_empty_hsv_guard=(
                self._enable_puyo_to_empty_hsv_guard
            ),
            enable_asymmetric_recovery_min_frames=(
                self._enable_asymmetric_recovery_min_frames
            ),
            recovery_add_min_frames=self._recovery_add_min_frames,
        )
        self._sm_2p = self._build_state_machine(
            stable_frame_count, enable_warmup_guard=enable_warmup_guard,
            enable_stable_recovery_gate=enable_stable_recovery_gate,
            enable_ojama_visual_detection=self._enable_ojama_visual_detection,
            enable_ojama_visual_chain_exit=self._enable_ojama_visual_chain_exit,
            enable_ojama_settle_detection=self._enable_ojama_settle_detection,
            enable_ojama_fall_board_settle=self._enable_ojama_fall_board_settle,
            enable_chain_max_hold_override=self._enable_chain_max_hold_override,
            enable_gravity_settle_state=self._enable_gravity_settle_state,
            enable_slide_override_ojama_hold=self._enable_slide_override_ojama_hold,
            enable_gravity_filter_support=self._enable_gravity_filter_support,
            merge_use_majority_value=self._merge_use_majority_value,
            enable_column_partial_support=self._enable_column_partial_support,
            enable_match_start_full_clear=self._enable_match_start_full_clear,
            enable_recovery_counter_carryover=self._enable_recovery_counter_carryover,
            enable_cnn_flicker_hsv_fallback=self._enable_cnn_flicker_hsv_fallback,
            enable_initial_confirm_vote=self._enable_initial_confirm_vote,
            initial_confirm_min_votes=self._initial_confirm_min_votes,
            enable_puyo_to_empty_hsv_guard=(
                self._enable_puyo_to_empty_hsv_guard
            ),
            enable_asymmetric_recovery_min_frames=(
                self._enable_asymmetric_recovery_min_frames
            ),
            recovery_add_min_frames=self._recovery_add_min_frames,
        )
        # 推論 / drift
        self._gen_1p = InferenceBoardGenerator()
        self._gen_2p = InferenceBoardGenerator()
        self._drift_1p = DriftDetector()
        self._drift_2p = DriftDetector()
        # Phase I: 擬似ラベル抽出 hook (default off で既存挙動維持)
        self._enable_pseudo_label = bool(enable_pseudo_label)
        self._pseudo_label_store = pseudo_label_store
        self._pseudo_validators: list = []  # CrossValidator の list
        if self._enable_pseudo_label:
            self._init_pseudo_validators()
        # 2026-05-11 サイクル71 Phase 1a: 物理推論主軸用 ChainSimulator.
        # 旧実装は遅延初期化していたが、 着地時に毎回呼ぶため事前構築.
        from src.chain import ChainSimulator as _ChainSimulator
        self._chain_sim: _ChainSimulator = _ChainSimulator()
        # T4 PuyoErasureMonitor: STABLE 中「色→EMPTY」 遷移を自動検知。
        # 1P/2P 独立 instance。試合切替時は reset() で消去。
        from src.puyo_erasure_monitor import PuyoErasureMonitor as _PEM
        self._erasure_monitor_1p: _PEM = _PEM()
        self._erasure_monitor_2p: _PEM = _PEM()
        # T4: 静的背景マスク (per-video 保存用ディレクトリ)
        self._static_mask_captured: bool = False
        # C2 StableTransitionMonitor: STABLE→STABLE 間の物理事由なきぷよ減少検知。
        # 1P/2P 独立 instance。試合切替時は reset() で消去。
        from src.stable_transition_monitor import StableTransitionMonitor as _STM
        self._transition_monitor_1p: _STM = _STM()
        self._transition_monitor_2p: _STM = _STM()
        # B1 PiecePersistenceGuard: STABLE 中 cell 色の物理保護ガード。
        # default False で既存挙動を維持 (backwards compat)。
        # ⚠️ 2026-05-28 user 目視で v89m7/v40m7/v30_5min 等にて
        #   「連鎖中エフェクトをぷよと誤認 / お邪魔認識なし / 序盤誤認多い」 確定で撤回。
        #   PuyoErasureMonitor の「色→空」 検知では捕捉できない blind spot (= NON-STABLE
        #   → STABLE 遷移直後の誤確定値が物理保護で固定化)。
        #   True にしてはいけない。実装は参照用に残置。
        from src.piece_persistence_guard import PiecePersistenceGuard as _PPG
        self._piece_persistence_1p: "_PPG | None" = (
            _PPG() if enable_piece_persistence else None
        )
        self._piece_persistence_2p: "_PPG | None" = (
            _PPG() if enable_piece_persistence else None
        )
        # NON-STABLE → STABLE 遷移直後の tier1 warmup guard。
        # True で TIER1_WARMUP_FRAMES の間 tier1 (bg_fp NCC 無条件 EMPTY 化) をスキップ。
        # default False で既存テスト全て pass を維持 (backwards compat)。
        # v95m15: tier1 が着地直後 cell を誤 EMPTY 化 → false tsumo_fall 5 回連発の対策。
        self._enable_tier1_warmup: bool = bool(enable_tier1_warmup)
        self._tier1_warmup_remaining_1p: int = 0
        self._tier1_warmup_remaining_2p: int = 0
        # 経路 A': OJAMA_FALL → STABLE 遷移専用の tier1 warmup guard。
        # True で OJAMA_TIER1_WARMUP_FRAMES の間 tier1 をスキップ。
        # 汎用 tier1_warmup (v51m2 で +108 退行) と独立して、お邪魔消滅後の
        # セル背景化による列崩壊のみを対処する。default False で既存挙動不変。
        self._enable_ojama_tier1_warmup: bool = bool(enable_ojama_tier1_warmup)
        self._ojama_tier1_warmup_remaining_1p: int = 0
        self._ojama_tier1_warmup_remaining_2p: int = 0
        # 案2: constraint_fill トグル (= False で _apply_next_count_constraint を skip)。
        # デフォルト True = 従来挙動維持 (backwards compat)。
        # --no-constraint-fill 等で False にすると CNN/HSV 高確信セルが誤置換される問題を
        # 完全回避できるが、 count 補正も無効化される点に注意。
        self._enable_constraint_fill: bool = bool(enable_constraint_fill)
        # T2 高確信 yield トグル (= True で T2 prev_stable 上書きを CNN 一致セルで解除)。
        # デフォルト False = 従来挙動維持 (backwards compat)。
        # True にすると「cnn_board が cur_v と同一有色」のセルは T2 の prev_stable
        # 上書きをスキップし、infer_placement 誤推論 + T2 自己強化フリーズを解除する。
        # B1 禁忌 (= 色→空 変化を無差別保護) とは逆方向: prev_stable の古い色による
        # 上書きを「CNN が正色を支持している箇所でのみ」解除する (保護を弱める方向)。
        self._enable_t2_highconf_yield: bool = bool(enable_t2_highconf_yield)
        # 案 B1: infer_placement 空セル hallucination ガード。
        # True にすると、 pattern の非 diff セルが cnn_after で COLOR_EMPTY な候補を
        # スキップし、 CNN が確信して空なセルへの NEXT 色書込を防ぐ。
        # 非 diff セルが COLOR_UNKNOWN なら従来通り補完を許容 (= 物理的に自然)。
        # デフォルト False = 従来挙動維持 (backwards compat)。
        self._enable_infer_empty_guard: bool = bool(enable_infer_empty_guard)
        # game-event ベース連鎖終了 (C-1/C-2 plan, 2026-06-01)。
        # True で次ツモ変化 / お邪魔出現をトリガーとして CHAIN 終了する。
        # False = 従来 timing hold のみ (backwards compat)。
        self._enable_game_event_chain_exit: bool = bool(enable_game_event_chain_exit)
        # 機能B: score 急増 CHAIN 早期発火フラグ。
        self._enable_chain_score_early_fire: bool = bool(enable_chain_score_early_fire)
        # 機能C: CHAIN → STABLE warmup フラグ + 遷移時刻 (1P/2P 別)。
        # float: CHAIN→STABLE 遷移 time_sec。凍結中は time_sec < _chain_exit_until_* 。
        self._enable_chain_exit_warmup: bool = bool(enable_chain_exit_warmup)
        self._chain_exit_until_1p: float = 0.0  # CHAIN→STABLE 後の凍結終了 time_sec
        self._chain_exit_until_2p: float = 0.0
        # 機能D: 掛け算式検知フラグ + 連続フレームカウンタ (1P/2P 別 state-holding wrapper)。
        # カウンタは update 呼出毎に条件成立なら +1、不成立でリセット。
        # CHAIN_FORMULA_CONSEC_FRAMES 達した時点で _apply_chain_score_early_fire を呼ぶ。
        self._enable_chain_formula_detection: bool = bool(enable_chain_formula_detection)
        self._formula_consec_1p: int = 0  # 掛け算式 連続フレームカウンタ 1P
        self._formula_consec_2p: int = 0  # 掛け算式 連続フレームカウンタ 2P
        # 修正D (2026-07-24): 機能D 疑似発火 起点盤面の ChainSimulator 検証フラグ。
        self._enable_chain_formula_simulate_verify: bool = bool(
            enable_chain_formula_simulate_verify
        )
        # 着地色修正 案1 (2026-06-01): TSUMO_FALL→STABLE 着地時の falling_pair を
        # prev_next_queue[-2] から _landing_pending (消費ツモ色) に切り替える。
        # True で修正ロジック有効。False (default) = 従来挙動完全維持 (backwards compat)。
        self._enable_landing_color_fix: bool = bool(enable_landing_color_fix)
        # X1/X4 短連鎖ちらつき対策 (2026-06-01):
        # True で CHAIN 最小表示時間 (CHAIN_MIN_DISPLAY_SEC) / 短連鎖 game-event exit 抑止を有効化。
        # False = 従来挙動完全維持 (backwards compat)。
        self._enable_chain_min_display: bool = bool(enable_chain_min_display)
        # fix/v70-zeropatch-redyellow (2026-06-01): HSV 分類 fallback トグル。
        # True で _classify_next_pair_by_hsv の 2 択強制確定を回避。
        # 黄→赤誤分類 (~900 件) 発火点対策。default False = 従来挙動完全維持。
        self._enable_hsv_classify_fallback: bool = bool(enable_hsv_classify_fallback)
        # 真因 A 対処 (2026-06-01): 着地セルの CNN==HSV 一致色で falling_pair ズレを補正。
        # True で infer_placement 出力に post-correction を適用。
        # default False = 従来挙動完全維持 (backwards compat)。
        self._enable_landing_observed_color: bool = bool(enable_landing_observed_color)
        # 色フリッカ根因への防御的修正 案(iii) (2026-07-25):
        # True で着地セルの CNN 観測色/baseline 不一致フラグを計算し、
        # P7 (着地投票) に伝播する。default False = 従来挙動完全維持
        # (backwards compat)。
        self._enable_placement_color_cnn_check: bool = bool(
            enable_placement_color_cnn_check,
        )
        # 修正方針 甲: P2 設置推論の防御的 CNN 照合 (2026-07-25)。
        # default False = 従来挙動完全維持 (backwards compat)。
        self._enable_placement_cnn_veto: bool = bool(enable_placement_cnn_veto)
        self._placement_cnn_veto_mode: str = str(placement_cnn_veto_mode)
        # 計装用カウンタ (A/B 計測での書き込み保留セル数の直接観測用)。
        self._placement_cnn_veto_held_count_1p: int = 0
        self._placement_cnn_veto_held_count_2p: int = 0
        # 案 Y-4 HSV-first commit + deferred consensus (2026-06-03)。
        # True で infer_placement が HSV 拮抗と判定した着地 2 候補を保留し、
        # 後続フレームの CNN==HSV consensus 投票で確定させる。
        # default False = 従来挙動完全維持 (backwards compat)。
        self._enable_hsv_deferred_consensus: bool = bool(enable_hsv_deferred_consensus)
        # deferred landing state: 1P/2P 独立。None = pending なし。
        # dict keys:
        #   board_std: Board (HSV 素返し候補 = 安全 fallback, 既に confirmed に書込済)
        #   board_rev: Board (逆順候補)
        #   base_cells: list[(r, c)] (着地 2 cell 座標)
        #   votes_std: int (board_std の累積合意票)
        #   votes_rev: int (board_rev の累積合意票)
        #   frames_left: int (残り保留フレーム数)
        self._deferred_landing_1p: dict | None = None
        self._deferred_landing_2p: dict | None = None
        # _set_deferred_confirmed 直後の T2 スキップ用フラグ (1P/2P 別)。
        # deferred 確定した frame で T2 が旧 prev_stable で上書きするのを防ぐ。
        self._deferred_just_committed_1p: bool = False
        self._deferred_just_committed_2p: bool = False
        # 不具合B 対処: 予告おじゃま発光ガード (2026-06-04)。
        # True で GlowGuardState を 1P/2P 別に保持し、発光中は confirmed 保護。
        # False (default) なら None のままで全処理をスキップ (完全挙動不変)。
        self._enable_ojama_warning_glow_guard: bool = bool(enable_ojama_warning_glow_guard)
        if self._enable_ojama_warning_glow_guard:
            from src.ojama_warning_glow_guard import GlowGuardState as _GGS
            self._glow_guard_1p: "_GGS | None" = _GGS()
            self._glow_guard_2p: "_GGS | None" = _GGS()
        else:
            self._glow_guard_1p = None
            self._glow_guard_2p = None
        # 案P3: active_chain が CHAIN_MAX_HOLD_SEC 超過で強制クリアされた直後の 1 frame フラグ。
        # _step_side に渡す DetectorSignals に chain_max_hold_expired として反映される。
        # active_chain が None にクリアされた frame だけ True、翌 frame は False にリセット。
        self._chain_max_hold_expired_1p: bool = False
        self._chain_max_hold_expired_2p: bool = False
        # 案X*(A)(B)+warmup: NextSlide signal による CHAIN 即終了フラグ (2026-06-05)。
        # True で以下が有効になる:
        #   (A) 機能D 再点火抑制 (CHAIN 中は掛け算式発火でmax_until延長しない)
        #   (B) slide_motion=True → その side の active_chain を即クリア
        #   warmup 連動: enable_chain_exit_warmup を内部で自動 ON
        # default False = 従来挙動完全維持。
        self._enable_chain_exit_next_signal: bool = bool(enable_chain_exit_next_signal)
        # warmup 連動: フラグ ON 時は enable_chain_exit_warmup も強制 ON にする。
        # フラグ OFF の場合は enable_chain_exit_warmup の指定をそのまま使う。
        if self._enable_chain_exit_next_signal:
            self._enable_chain_exit_warmup = True
        # feat/gravity-settle-2026-06-05: gravity_settle=True なら warmup も ON にする。
        # _enable_chain_exit_next_signal は上記 warmup 連動で設定済み。
        if self._enable_gravity_settle_state:
            self._enable_chain_exit_warmup = True
        # X1 用: CHAIN 突入時刻 (time_sec) を記録する (1P/2P 別)。
        # CHAIN 発火時に代入し、_on_match_end でリセット。
        self._chain_entry_t_1p: float = 0.0
        self._chain_entry_t_2p: float = 0.0
        # game-event モード用: CHAIN 発火時刻の上限 (安全弁)。
        # CHAIN 発火 time_sec + CHAIN_MAX_HOLD_SEC を超えたら強制終了。
        # 1P/2P 別に管理 (float: 未発火時は 0.0)。
        self._chain_event_max_until_1p: float = 0.0
        self._chain_event_max_until_2p: float = 0.0
        # game-event モード: CHAIN 発火時の連鎖側 prev_next_pair (next 変化検知用)。
        # 連鎖が始まった瞬間の next_pair をスナップショットし、変化したら終了する。
        # ※②お邪魔信号撤去 (2026-06-01): board snapshot (_chain_start_board_*) は不要。
        self._chain_start_next_1p: tuple[int, int] | None = None
        self._chain_start_next_2p: tuple[int, int] | None = None

    @staticmethod
    def _build_hybrid_reader(
        cnn_model_path: Path,
        vote_mode: bool = False,
        cnn_override_prob: float | None = None,
        mask_ojama_logit: bool = False,
        use_puyo_gate: bool = False,
        patch_ncc_threshold: float | None = None,
        ui_mask_cells: frozenset[tuple[int, int]] | None = None,
    ) -> ImageReader:
        """HybridClassifier (HSV + CNN) で ImageReader を組み立てる.

        2026-05-11 サイクル71: CNN メイン化方針で default を 0.90 → 0.70 に
        下げた (= HybridClassifier の DEFAULT_CNN_OVERRIDE_PROB に従う).
        過去 Phase B-7 の 0.90 は CNN ぶれ伝搬を抑える慎重設定だったが、
        単色 vote 方式の限界 (= 目/縁/ハイライト揺らぎ) を踏まえ画像全体
        パターンを学習した CNN を主軸化する.

        CUDA 利用可能なら GPU で推論する (CPU の 10〜15 倍速)。

        vote_mode=True (cycle 71) で HSV 分類器を per-pixel 投票方式に切替.
        cnn_override_prob: None なら DEFAULT_CNN_OVERRIDE_PROB を使用.
        ui_mask_cells: 案B (2026-07-30)。HybridClassifier.classify_batch の
            UI マスク判定対象セルを限定する。None (既定) では従来通り
            全セルで判定する (backwards compat、bit-identical)。
            速度優先で有効化する場合は src.ui_mask.UI_MASK_TARGET_CELLS を渡す。
        """
        from src.hybrid_classifier import HybridClassifier
        from src.image_reader import ColorClassifier
        from src.patch_classifier import (
            CnnPatchClassifier, CnnPatchClassifierLarge,
        )

        if not cnn_model_path.exists():
            raise FileNotFoundError(f"CNN model not found: {cnn_model_path}")
        import torch
        state = torch.load(
            str(cnn_model_path), map_location="cpu", weights_only=True,
        )
        # cycle 71v (案 D): state_dict shape で small/large を自動検出.
        # small: 1st conv = [16, 6, 3, 3] / large: [32, 6, 3, 3].
        first_key = next(iter(state.keys()))
        first_shape = tuple(state[first_key].shape)
        is_large = len(first_shape) >= 1 and first_shape[0] == 32
        if is_large:
            cnn = CnnPatchClassifierLarge()
        else:
            cnn = CnnPatchClassifier()
        cnn._model.load_state_dict(state)
        # GPU 切替 (CUDA_VISIBLE_DEVICES が空でなく cuda 利用可能ならば)
        try:
            import os as _os
            import torch as _torch
            if (
                _os.environ.get("CUDA_VISIBLE_DEVICES", "all") != ""
                and _torch.cuda.is_available()
            ):
                cnn.to_device("cuda")
        except Exception:
            pass
        hsv = ColorClassifier(vote_mode=vote_mode)
        from src.hybrid_classifier import DEFAULT_CNN_OVERRIDE_PROB
        eff_override = (
            float(cnn_override_prob)
            if cnn_override_prob is not None
            else DEFAULT_CNN_OVERRIDE_PROB
        )
        classifier = HybridClassifier(
            hsv_classifier=hsv,
            cnn_classifier=cnn, cnn_override_prob=eff_override,
            mask_ojama_logit=mask_ojama_logit,
            use_puyo_gate=use_puyo_gate,
            ui_mask_cells=ui_mask_cells,
        )
        # use_telop_mask=True で中央テロップ被覆 cell を COLOR_UNKNOWN に倒す
        # (V3.1 機能、A 統合の一環で 2026-05-09 から有効化)
        return ImageReader(
            classifier=classifier,
            use_match_state=False,
            use_telop_mask=True,
            patch_ncc_threshold=patch_ncc_threshold,
        )

    @staticmethod
    def _smooth_board(history: deque[Board]) -> Board:
        """直近 N frame の cell 単位 majority vote で平滑化盤面を生成.

        N=1 (履歴 1) ならそのまま返す。複数 frame ある場合は各 cell で
        最頻値を採用 (= 1 frame だけの CNN ぶれを除去)。
        """
        if not history:
            return Board()
        if len(history) == 1:
            return history[-1].copy()
        smoothed = Board()
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                votes: Counter[int] = Counter()
                for b in history:
                    votes[int(b.get(r, c))] += 1
                best = votes.most_common(1)[0][0]
                smoothed.set(r, c, best)
        return smoothed

    @staticmethod
    def _build_state_machine(
        stable_n: int,
        *,
        enable_warmup_guard: bool = False,
        enable_stable_recovery_gate: bool = False,
        enable_ojama_visual_detection: bool = False,
        enable_ojama_visual_chain_exit: bool = False,
        enable_ojama_settle_detection: bool = False,
        # 2026-07-24 採用 (default ON): 呼び出し元 __init__ が常に明示値を
        # 渡すため実運用では未使用だが、直接呼び出し時も採用済み挙動を
        # 既定にする (__init__ の既定値と同期)。
        enable_ojama_fall_board_settle: bool = True,
        enable_chain_max_hold_override: bool = False,
        enable_gravity_settle_state: bool = False,
        enable_slide_override_ojama_hold: bool = False,
        enable_gravity_filter_support: bool = True,
        merge_use_majority_value: bool = True,
        enable_column_partial_support: bool = False,
        # 2026-07-25 user レビュー (c34 v6) 承認・既定 ON 化: 呼び出し元 __init__
        # が常に明示値を渡すため実運用では未使用だが、直接呼び出し時も
        # 採用済み挙動を既定にする (__init__ の既定値と同期)。
        enable_match_start_full_clear: bool = True,
        # 復旧カウンタ carryover (2026-07-26, A/B 計測用)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        # user 承認前の savepoint 実装のため default OFF 固定。
        enable_recovery_counter_carryover: bool = False,
        # CNN 乱高下セル HSV フォールバック (#51 後半, 2026-07-26, A/B 計測用)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        # user 承認前の savepoint 実装のため default OFF 固定。
        enable_cnn_flicker_hsv_fallback: bool = False,
        # 色→空凍結の修正3点セット③ (2026-07-27): 初回STABLE確定の多数決ガード。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        enable_initial_confirm_vote: bool = False,
        initial_confirm_min_votes: int = DEFAULT_INITIAL_CONFIRM_MIN_VOTES,
        # 色→空 HSV 照合ガード (2026-07-30)。c34 型の列デッドロックには有効だが
        # 4動画測定で c58/c26 の 2P tail 悪化・c26/c69 の 1P 効果ゼロ、汎化未確認の
        # ため default OFF。True で有効化 (backwards compat)。
        enable_puyo_to_empty_hsv_guard: bool = False,
        # 復旧ゲート方向別しきい値 非対称化 (2026-07-30, A/B 計測用)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        enable_asymmetric_recovery_min_frames: bool = False,
        recovery_add_min_frames: int = STABLE_RECOVERY_ADD_MIN_FRAMES,
    ) -> BoardStateMachine:
        # cycle 49 (2026-05-20): ChainPhaseDetector に ChainSimulator を注入。
        # 前 STABLE 盤面に 4 連結がない場合の chain 偽遷移を拒否する gate を有効化。
        # B1 (M1 warmup guard): enable_warmup_guard=True で STABLE 遷移直後 N frame
        # confirmed 更新を凍結する (A/B 対照実験用 optional 機能)。
        # 設計C 事後復旧ゲート: enable_stable_recovery_gate=True で BoardStateMachine に伝播。
        # フェーズ A 精緻化: enable_ojama_visual_detection=True で OjamaVisualDetector を
        # OjamaPhaseDetector の前 (優先順 3) に挿入する。score 差分ベース fallback は維持。
        # 案B (第2の根本原因対処, 2026-07-24): enable_ojama_fall_board_settle=True で
        #   OjamaVisualDetector の全盤面 settle 判定 + OjamaPhaseDetector の
        #   defer_ojama_fall_exit_to_visual を同時に有効化する (単一フラグに直結)。
        # 案P3: enable_chain_max_hold_override=True で ChainPhaseDetector に伝播。
        # feat/gravity-settle-2026-06-05: enable_gravity_settle_state=True で
        #   ChainPhaseDetector が CHAIN → GRAVITY_SETTLE を返し、
        #   GravitySettleDetector が GRAVITY_SETTLE → STABLE を担当する。
        # 案γ: enable_slide_override_ojama_hold=True で ChainPhaseDetector に伝播。
        #   CHAIN 中 slide_motion=True が来た場合 ojama-hold 保留を上書きして終了する。
        # #45 おじゃま merge 統合修正 案(a)(b) (2026-07-24):
        #   enable_gravity_filter_support / merge_use_majority_value を
        #   BoardStateMachine にそのまま伝播する (独立 flag)。
        from src.chain import ChainSimulator
        from src.board_state_machine import STABLE_WARMUP_FRAMES
        # ChainPhaseDetector に chain_ojama_exit + 案P3 + GRAVITY_SETTLE + 案γ フラグを伝播する
        chain_det = ChainPhaseDetector(
            chain_sim=ChainSimulator(),
            enable_chain_ojama_exit=enable_ojama_visual_chain_exit,
            enable_chain_max_hold_override=enable_chain_max_hold_override,
            enable_gravity_settle_state=enable_gravity_settle_state,
            enable_slide_override_ojama_hold=enable_slide_override_ojama_hold,
        )
        detectors: list = [
            chain_det,
            EffectPhaseDetector(),
        ]
        if enable_ojama_visual_detection:
            from src.ojama_visual_detector import OjamaVisualDetector
            ovd = OjamaVisualDetector(
                enable_ojama_visual_chain_exit=enable_ojama_visual_chain_exit,
                enable_ojama_settle_detection=enable_ojama_settle_detection,
                enable_ojama_fall_board_settle=enable_ojama_fall_board_settle,
            )
            detectors.append(ovd)
        detectors.append(
            OjamaPhaseDetector(
                defer_ojama_fall_exit_to_visual=enable_ojama_fall_board_settle,
            )
        )
        detectors.append(TsumoPhaseDetector())
        # feat/gravity-settle-2026-06-05: GravitySettleDetector を最低優先 (末尾) で登録。
        # CHAIN より低優先 → settle 中に次連鎖 drop 検知で CHAIN detector が優先発火し
        # 多段連鎖に対応する。
        if enable_gravity_settle_state:
            detectors.append(GravitySettleDetector())
        return BoardStateMachine(
            detectors=detectors,
            stable_frame_count=stable_n,
            enable_warmup_guard=enable_warmup_guard,
            warmup_frames=STABLE_WARMUP_FRAMES,
            enable_stable_recovery_gate=enable_stable_recovery_gate,
            enable_gravity_filter_support=enable_gravity_filter_support,
            merge_use_majority_value=merge_use_majority_value,
            enable_column_partial_support=enable_column_partial_support,
            enable_match_start_full_clear=enable_match_start_full_clear,
            enable_recovery_counter_carryover=enable_recovery_counter_carryover,
            enable_cnn_flicker_hsv_fallback=enable_cnn_flicker_hsv_fallback,
            enable_initial_confirm_vote=enable_initial_confirm_vote,
            initial_confirm_min_votes=initial_confirm_min_votes,
            enable_puyo_to_empty_hsv_guard=enable_puyo_to_empty_hsv_guard,
            enable_asymmetric_recovery_min_frames=(
                enable_asymmetric_recovery_min_frames
            ),
            recovery_add_min_frames=recovery_add_min_frames,
        )

    # cycle 71v (2026-05-14): val 98.87% を達成した Large CNN を system default に昇格.
    # `load_default(cnn_model_path=None)` 時に自動採用される. ファイル不在なら
    # HSV-only fallback (= 旧挙動) になるため、 古い環境でも例外にはならない.
    DEFAULT_CNN_MODEL_PATH: Path = Path("models/cnn_phase_b_large_v2.pt")

    @classmethod
    def load_default(
        cls,
        stable_frame_count: int = 6,
        load_score_ocr: bool = True,
        enable_chain_tracker: bool = True,
        cnn_model_path: Path | None = None,
        temporal_smoothing: int = 1,
        load_next_detector: bool = True,
        force_in_match: bool = False,
        enable_pseudo_label: bool = False,
        pseudo_label_store: object | None = None,
        enable_next_slide_detector: bool = True,
        vote_mode: bool = False,
        cnn_override_prob: float | None = None,
        mask_ojama_logit: bool = False,
        use_puyo_gate: bool = False,
        enable_warmup_guard: bool = False,
        bg_fp_force_max_puyo: int | None = None,
        patch_ncc_threshold: float | None = None,
        # ⚠️ B1 撤回済 (2026-05-28): enable_piece_persistence は常に False のまま。
        # 連鎖中エフェクト誤認 / ojama 消失 / 序盤誤認が確認された。
        enable_piece_persistence: bool = False,
        enable_tier1_warmup: bool = False,
        enable_ojama_tier1_warmup: bool = True,
        enable_constraint_fill: bool = False,
        enable_t2_highconf_yield: bool = True,
        enable_infer_empty_guard: bool = True,
        enable_game_event_chain_exit: bool = True,
        enable_landing_color_fix: bool = False,
        enable_chain_min_display: bool = False,
        enable_hsv_classify_fallback: bool = False,
        # 2026-07-25 user レビュー (c34 v6) 承認で既定 ON 化。
        # False を明示指定すれば旧挙動 (bit-identical) に戻せる (backwards compat)。
        enable_landing_observed_color: bool = True,
        # 色フリッカ根因への防御的修正 案(iii) (2026-07-25)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        enable_placement_color_cnn_check: bool = False,
        # 修正方針 甲: P2 設置推論の防御的 CNN 照合 (2026-07-25)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        enable_placement_cnn_veto: bool = False,
        placement_cnn_veto_mode: str = "hold",
        enable_red_hue_wrap_fix: bool = True,
        # 案D (fix/v70-zeropatch-redyellow): 光沢ハイライト除外彩度計算。
        # 2026-06-02: user viz 採用承認により default True に変更。
        enable_specular_robust_saturation: bool = True,
        # 設計C 事後復旧ゲート (2026-06-02): user viz 採用承認により default True に変更。
        enable_stable_recovery_gate: bool = True,
        # フェーズ A 精緻化 (2026-06-02): おじゃま視覚検知フラグ群。
        # user viz 採用承認により全フラグ default True に変更。
        enable_ojama_visual_detection: bool = True,
        enable_ojama_visual_chain_exit: bool = True,
        enable_ojama_infer_guard: bool = True,
        enable_ojama_settle_detection: bool = True,
        # 案B (第2の根本原因対処, 2026-07-24): OJAMA_FALL 退出を全盤面 settle
        # 判定に一本化する (OjamaVisualDetector + OjamaPhaseDetector 連動)。
        # 2026-07-24 採用 (default ON): A/B 検証 (次ツモ遅延 2.80s→0.65s・
        # 浮き誤消去 -28%・採用 +38、user viz 全画像レビュー承認) で採用。
        # False を明示指定すれば旧挙動 (bit-identical) に戻せる (backwards compat)。
        enable_ojama_fall_board_settle: bool = True,
        # 機能B: score 急増 CHAIN 早期発火 (2026-06-02)。
        # デフォルト False = 従来挙動完全維持 (backwards compat)。
        enable_chain_score_early_fire: bool = False,
        # 機能C: CHAIN → STABLE warmup (2026-06-02)。
        # デフォルト False = 従来挙動完全維持 (backwards compat)。
        enable_chain_exit_warmup: bool = False,
        # 機能D: 連鎖開始 掛け算式 検知 (2026-06-02 実装, 2026-06-03 採用 default ON)。
        # 全軸改善 + user viz 承認。False に戻すには明示指定する。
        enable_chain_formula_detection: bool = True,
        # 修正D (2026-07-24): 機能D 疑似発火 起点盤面の ChainSimulator 検証。
        # 2026-07-24 採用 (default ON): 偽イベント率 27.5%→0% (user viz 承認)。
        # False で旧挙動 (bit-identical) に戻せる (backwards compat)。
        enable_chain_formula_simulate_verify: bool = True,
        # 案 Y-4 HSV-first commit + deferred consensus (2026-06-03)。
        # default False = 従来挙動完全維持 (backwards compat)。
        enable_hsv_deferred_consensus: bool = False,
        # 不具合B 対処: 予告おじゃま発光ガード (2026-06-04)。
        # default False = 従来挙動完全維持 (backwards compat)。
        enable_ojama_warning_glow_guard: bool = False,
        # 案P3: CHAIN_MAX_HOLD_SEC 超過後の ojama 保留を無効化 (2026-06-05)。
        # default False = 従来挙動完全維持 (backwards compat)。
        enable_chain_max_hold_override: bool = False,
        # 案X*(A)(B)+warmup: NextSlide signal による CHAIN 即終了 (2026-06-05)。
        # default False = 従来挙動完全維持 (backwards compat)。
        enable_chain_exit_next_signal: bool = False,
        # feat/gravity-settle-2026-06-05: 連鎖終了直後 GRAVITY_SETTLE 状態を有効化。
        # 2026-06-06 採用: 退行ゼロ + 連鎖境界正確化。False で無効化可。
        enable_gravity_settle_state: bool = True,
        # 案γ (2026-06-06 採用): CHAIN 中 slide_motion=True が ojama-hold を上書き。
        # user 目視 OK + 退行なし (corr +0.004% 誤差) で採用確定 → default True。
        # False を渡すと無効化できる (backwards compat のため optional 引数として維持)。
        enable_slide_override_ojama_hold: bool = True,
        # 案1 (2026-07-23): estimated_board の stale_hold フォールバック。
        # user viz 承認前の savepoint 実装のため default True だが、
        # False で従来挙動 (常に None) に戻せる (backwards compat)。
        enable_chain_estimate_stale_hold: bool = True,
        # A0 (2026-07-24): CHAIN 保持時間モデルの較正値注入用。
        # 従来 __init__ には chain_hold_per_step_sec が存在したが load_default
        # には露出していなかった (評価スクリプト経由で注入不可という抜け漏れ)。
        # 今回まとめて露出する。全て None (既定) で従来値 (0.0 / 0.3 / 5.0) と
        # bit-identical (backwards compat)。
        chain_hold_base_sec: float | None = None,
        chain_hold_per_step_sec: float | None = None,
        chain_max_hold_sec: float | None = None,
        # 修正C (2026-07-24): VideoChainTracker の偽イベント抑制 debounce。
        # 既定 1 = 従来通り即時確定 (bit-identical, backwards compat)。
        # 2 以上で debounce_confirm_frames 回連続の drop 観測を要求する。
        chain_debounce_confirm_frames: int = DEBOUNCE_CONFIRM_FRAMES,
        # #45 おじゃま merge 統合修正 案(a)(b) (2026-07-24): 案B
        # (enable_ojama_fall_board_settle) 適用後に判明した _merge_diff_only
        # の 2 副作用を個別 flag で修正する (A/B 切り分け用、独立 flag)。
        # 2026-07-24 採用 (default ON): 案B (enable_ojama_fall_board_settle)
        # と併せた A/B 検証 (user viz 全画像レビュー承認) で採用。それぞれ
        # False を明示指定すれば旧挙動 (bit-identical) に戻せる
        # (backwards compat)。
        enable_gravity_filter_support: bool = True,
        merge_use_majority_value: bool = True,
        # DriftDetector 再同期ループ暴走ガード (2026-07-25, c34 実測)。
        # 2026-07-25 user レビュー (c34 v6) 承認で既定 ON 化。
        # False を明示指定すれば旧挙動 (bit-identical) に戻せる (backwards
        # compat)。
        enable_drift_resync_match_start_guard: bool = True,
        enable_drift_resync_hsv_gate: bool = True,
        # cycle 31 baseline_broken 自己リセット 制御フラグ (2026-07-25,
        # A/B 計測用)。default True/False = 従来挙動完全維持 (backwards compat)。
        enable_baseline_broken_reset: bool = True,
        enable_baseline_broken_grace: bool = False,
        # 列ゲート緩和 (2026-07-25, A/B 計測用)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        enable_column_partial_support: bool = False,
        # 前試合盤面残骸リーク修正 (2026-07-23, A/B 計測用)。
        # 2026-07-25 user レビュー (c34 v6) 承認で既定 ON 化。
        # False を明示指定すれば旧挙動 (bit-identical) に戻せる (backwards compat)。
        enable_match_start_full_clear: bool = True,
        # score-reset 境界誤発火修正 (2026-07-26, A/B 計測用)。誤発火は確定
        # バグの修正であるため既定 True。False で旧 (片側 OR・デバウンス無し)
        # 挙動に戻せる (backwards compat)。
        enable_score_reset_strict: bool = True,
        # 復旧カウンタ carryover (#51, 2026-07-26, A/B 計測用)。
        # 2026-07-27 user レビュー (video_84, #51系3修正全6観点OK) 承認で
        # 既定 ON 化。False を明示指定すれば旧挙動 (bit-identical) に戻せる
        # (backwards compat)。
        enable_recovery_counter_carryover: bool = True,
        # CNN 乱高下セル HSV フォールバック (#51 後半, 2026-07-26, A/B 計測用)。
        # 2026-07-27 user レビュー (video_84, #51系3修正全6観点OK) 承認で
        # 既定 ON 化。False を明示指定すれば旧挙動 (bit-identical) に戻せる
        # (backwards compat)。
        enable_cnn_flicker_hsv_fallback: bool = True,
        # 色→空凍結の修正3点セット③ (2026-07-27): 初回STABLE確定の多数決ガード。
        # 2026-07-27 user レビュー (video_84, #51系3修正全6観点OK) 承認で
        # 既定 ON 化。False を明示指定すれば旧挙動 (bit-identical) に戻せる
        # (backwards compat)。
        enable_initial_confirm_vote: bool = True,
        initial_confirm_min_votes: int = DEFAULT_INITIAL_CONFIRM_MIN_VOTES,
        # 大 ROI 走査 (match_end / telop) の間引き (2026-07-30、2026-07-31 既定ON)。
        # 走査を飛ばすので原理的に bit-identical にならないが、試合終了時刻を
        # またぐ窓での実測 (3動画 x 2イベント = 1800フレーム) で
        # **試合終了検出のずれ 0フレーム・盤面差分 0/1800** を確認した。
        # 遅延が伝播しないのは hard_match_off が score_zero_both との OR で、
        # score_zero / MatchStateDetector は間引き対象外のため独立経路が
        # 同一フレームで発火するから (設計時の有界性の主張が実証された)。
        # 速度 +19.5〜53.4%。False で従来の毎フレーム走査に戻る。
        enable_large_roi_throttle: bool = True,
        large_roi_throttle_frames: int = LARGE_ROI_THROTTLE_FRAMES,
        # 色→空 HSV 照合ガード (2026-07-30): c34 型の列デッドロックには有効だが、
        # 4動画測定 (c34/c58/c26/c69) で c58/c26 の 2P tail 悪化、c26/c69 の 1P
        # 効果ゼロ、8フレーム達成率は OFF/ON 不変と判明。汎化未確認のため
        # default OFF。True で有効化 (backwards compat)。
        enable_puyo_to_empty_hsv_guard: bool = False,
        # 復旧ゲート方向別しきい値 非対称化 (2026-07-30, A/B 計測用)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        # user 承認前の savepoint 実装のため default OFF 固定。
        enable_asymmetric_recovery_min_frames: bool = False,
        recovery_add_min_frames: int = STABLE_RECOVERY_ADD_MIN_FRAMES,
        # 案B (2026-07-30): UI マスク判定 (is_ui 呼出) をセル限定する高速化フラグ。
        # None (既定) = 従来通り全セルで判定 (backwards compat、bit-identical)。
        # 既定 ON 化 (2026-07-30)。それまで既定 None のため **本番の収集・レンダで
        # 一切効いていなかった** (渡していたのは診断スクリプト1本だけ)。
        # 引き継ぎの「4.4→8.07fps 出荷済み」はその診断スクリプト内の値で、
        # 本番はずっと絞り込み無しで動いていた
        # (memory project_ui_mask_cells_never_wired_2026-07-30)。
        #
        # 実測 (video_c56/c60/c65 × 300フレーム = 900フレーム):
        #   確定盤面の差分 0/900 フレーム (0.00%) = 挙動は完全に不変
        #   速度 +19.1% 〜 +25.6% (c60 t=1451 の別測定では 226.3→122.1ms)
        # 従来動作に戻す場合は明示的に None を渡す (backwards compat)。
        # スコアOCRの NCC を行列積1回に束ねる高速経路 (2026-07-30)。
        # 実測: 認識全体の19.5%を占める score_ocr.read_side が対象、
        # 1セル分1777us→12.1us (146倍速)、1フレーム換算28.43ms→0.19ms。
        # cv2(float32) と numpy(float64) の差でスコアに最大5.5e-07の
        # 乖離が出るため bit-identical ではないが、実動画3本 x 300フレームの
        # A/B で **OCRスコア差分 0/900・確定盤面差分 0/900** を実測したため
        # 既定 ON (速度 +23.5〜26.8%)。従来経路に戻すには False を渡す。
        enable_score_ocr_matmul: bool = True,
        ui_mask_cells: "frozenset[tuple[int, int]] | None" = UI_MASK_TARGET_CELLS,
    ) -> "RecognitionPipeline":
        """デフォルト構成でロードする。

        Args:
            cnn_model_path: HybridClassifier (HSV + CNN) で ImageReader を
                構築する CNN モデルのパス。
                - None (default): `DEFAULT_CNN_MODEL_PATH` (= cnn_phase_b_large_v2.pt)
                  を試し、 不在なら HSV-only fallback。
                - Path 指定: そのモデルを明示的にロード。
                - HSV-only を強制したい場合は `cnn_model_path=Path("__hsv_only__")`
                  のような存在しないパスを渡すと FileNotFoundError になるため、
                  HSV-only を狙うコードは別途用意のこと (現状そのような caller は無し)。
            vote_mode: True で HSV 分類器を per-pixel 投票方式に切替 (cycle 71).
                False (default) は HSV 中央値 + cycle 69-B サブ region vote.
            cnn_override_prob: HybridClassifier の CNN 採用閾値. None なら
                DEFAULT_CNN_OVERRIDE_PROB=0.70 (= cycle 71 CNN メイン化).
                0.5 で CNN 強信頼、 0.9 で CNN 慎重 (旧挙動互換).
            enable_specular_robust_saturation: True にすると光沢ハイライト除外彩度計算を有効化。
                白ハイライト画素を彩度 median 計算から除外して EMPTY 誤判定を防ぐ (案D)。
                backwards compat: デフォルト False = 従来挙動。
        """
        from src.image_reader import ColorClassifier
        # cycle 71v: None なら DEFAULT_CNN_MODEL_PATH に解決 (存在する場合のみ).
        effective_cnn_path: Path | None = (
            Path(cnn_model_path) if cnn_model_path is not None
            else (cls.DEFAULT_CNN_MODEL_PATH
                  if cls.DEFAULT_CNN_MODEL_PATH.exists() else None)
        )
        if effective_cnn_path is not None:
            reader = cls._build_hybrid_reader(
                effective_cnn_path,
                vote_mode=vote_mode,
                cnn_override_prob=cnn_override_prob,
                mask_ojama_logit=mask_ojama_logit,
                use_puyo_gate=use_puyo_gate,
                patch_ncc_threshold=patch_ncc_threshold,
                ui_mask_cells=ui_mask_cells,
            )
        else:
            reader = ImageReader(
                use_match_state=False,
                classifier=ColorClassifier(vote_mode=vote_mode),
                patch_ncc_threshold=patch_ncc_threshold,
            )
        match_detector = MatchStateDetector.load_default()
        score: ScoreOcr | None = None
        if load_score_ocr:
            try:
                score = ScoreOcr.load_default(
                    enable_matmul_ncc=enable_score_ocr_matmul,
                )
            except FileNotFoundError:
                score = None
        ctracker_1p = (
            VideoChainTracker(debounce_confirm_frames=chain_debounce_confirm_frames)
            if enable_chain_tracker else None
        )
        ctracker_2p = (
            VideoChainTracker(debounce_confirm_frames=chain_debounce_confirm_frames)
            if enable_chain_tracker else None
        )
        next_det: NextDetector | None = None
        if load_next_detector:
            try:
                from src.patch_classifier import CnnPatchClassifier
                from pathlib import Path as _Path
                cnn_for_next = CnnPatchClassifier()
                gbest = _Path("models/cnn_global_best.pt")
                if gbest.exists():
                    import torch
                    state = torch.load(
                        str(gbest), map_location="cpu", weights_only=True,
                    )
                    cnn_for_next._model.load_state_dict(state)
                # GPU 切替
                try:
                    import os as _os
                    import torch as _torch
                    if (
                        _os.environ.get("CUDA_VISIBLE_DEVICES", "all") != ""
                        and _torch.cuda.is_available()
                    ):
                        cnn_for_next.to_device("cuda")
                except Exception:
                    pass
                next_det = NextDetector(classifier=cnn_for_next)
            except Exception as e:
                print(f"[pipeline] next_detector load failed: {e}")
                next_det = None

        # A: 既存 detector 統合 (テンプレ不在環境では None フォールバック)
        score_zero_det: ScoreZeroDetector | None = None
        match_end_det: MatchEndDetector | None = None
        telop_det: TelopDetector | None = None
        try:
            score_zero_det = ScoreZeroDetector.load_default()
        except Exception as e:
            print(f"[pipeline] score_zero load skipped: {e}")
        try:
            match_end_det = MatchEndDetector.load_default()
        except Exception as e:
            print(f"[pipeline] match_end load skipped: {e}")
        try:
            telop_det = TelopDetector.load_default()
        except Exception as e:
            print(f"[pipeline] telop load skipped: {e}")
        # Phase I.c: OnlineHsvCalibrator (動画別 HSV 自動学習).
        # 短時間動画 (90 秒程度) でも inject されるよう HIGH_CONF=0.85 /
        # MIN_SAMPLES=50 に緩和 (default 0.99/200 は試合長時の安定性重視).
        # cycle 71f (提案 C): CNN model 不在の HSV-only 構成でも起動するように
        # 条件を撤廃. require_cnn_proba=False で HSV 一致のみを信頼条件として
        # 動作可能 (= 動画固有の色相変動に追従、 ユーザー指摘の色誤認対策).
        online_hsv: OnlineHsvCalibrator = OnlineHsvCalibrator(
            high_conf=0.85, min_samples=50,
            require_cnn_proba=False,
        )

        return cls(
            image_reader=reader,
            match_state_detector=match_detector,
            score_ocr=score,
            chain_tracker_1p=ctracker_1p,
            chain_tracker_2p=ctracker_2p,
            stable_frame_count=stable_frame_count,
            temporal_smoothing=temporal_smoothing,
            next_detector=next_det,
            force_in_match=force_in_match,
            enable_pseudo_label=enable_pseudo_label,
            pseudo_label_store=pseudo_label_store,
            enable_next_slide_detector=enable_next_slide_detector,
            score_zero_detector=score_zero_det,
            match_end_detector=match_end_det,
            telop_detector=telop_det,
            online_hsv=online_hsv,
            enable_warmup_guard=enable_warmup_guard,
            bg_fp_force_max_puyo=bg_fp_force_max_puyo,
            enable_piece_persistence=enable_piece_persistence,
            enable_tier1_warmup=enable_tier1_warmup,
            enable_ojama_tier1_warmup=enable_ojama_tier1_warmup,
            enable_constraint_fill=enable_constraint_fill,
            enable_t2_highconf_yield=enable_t2_highconf_yield,
            enable_infer_empty_guard=enable_infer_empty_guard,
            enable_game_event_chain_exit=enable_game_event_chain_exit,
            enable_landing_color_fix=enable_landing_color_fix,
            enable_chain_min_display=enable_chain_min_display,
            enable_hsv_classify_fallback=enable_hsv_classify_fallback,
            enable_landing_observed_color=enable_landing_observed_color,
            enable_placement_color_cnn_check=enable_placement_color_cnn_check,
            enable_placement_cnn_veto=enable_placement_cnn_veto,
            placement_cnn_veto_mode=placement_cnn_veto_mode,
            enable_red_hue_wrap_fix=enable_red_hue_wrap_fix,
            enable_specular_robust_saturation=enable_specular_robust_saturation,
            enable_stable_recovery_gate=enable_stable_recovery_gate,
            enable_ojama_visual_detection=enable_ojama_visual_detection,
            enable_ojama_visual_chain_exit=enable_ojama_visual_chain_exit,
            enable_ojama_infer_guard=enable_ojama_infer_guard,
            enable_ojama_settle_detection=enable_ojama_settle_detection,
            enable_ojama_fall_board_settle=enable_ojama_fall_board_settle,
            enable_chain_score_early_fire=enable_chain_score_early_fire,
            enable_chain_exit_warmup=enable_chain_exit_warmup,
            enable_chain_formula_detection=enable_chain_formula_detection,
            enable_chain_formula_simulate_verify=enable_chain_formula_simulate_verify,
            enable_hsv_deferred_consensus=enable_hsv_deferred_consensus,
            enable_ojama_warning_glow_guard=enable_ojama_warning_glow_guard,
            enable_chain_max_hold_override=enable_chain_max_hold_override,
            enable_chain_exit_next_signal=enable_chain_exit_next_signal,
            enable_gravity_settle_state=enable_gravity_settle_state,
            enable_slide_override_ojama_hold=enable_slide_override_ojama_hold,
            enable_chain_estimate_stale_hold=enable_chain_estimate_stale_hold,
            chain_hold_base_sec=chain_hold_base_sec,
            chain_hold_per_step_sec=chain_hold_per_step_sec,
            chain_max_hold_sec=chain_max_hold_sec,
            enable_gravity_filter_support=enable_gravity_filter_support,
            merge_use_majority_value=merge_use_majority_value,
            enable_drift_resync_match_start_guard=(
                enable_drift_resync_match_start_guard
            ),
            enable_drift_resync_hsv_gate=enable_drift_resync_hsv_gate,
            enable_baseline_broken_reset=enable_baseline_broken_reset,
            enable_baseline_broken_grace=enable_baseline_broken_grace,
            enable_column_partial_support=enable_column_partial_support,
            enable_match_start_full_clear=enable_match_start_full_clear,
            enable_score_reset_strict=enable_score_reset_strict,
            enable_recovery_counter_carryover=enable_recovery_counter_carryover,
            enable_cnn_flicker_hsv_fallback=enable_cnn_flicker_hsv_fallback,
            enable_initial_confirm_vote=enable_initial_confirm_vote,
            initial_confirm_min_votes=initial_confirm_min_votes,
            enable_large_roi_throttle=enable_large_roi_throttle,
            large_roi_throttle_frames=large_roi_throttle_frames,
            enable_puyo_to_empty_hsv_guard=enable_puyo_to_empty_hsv_guard,
            enable_asymmetric_recovery_min_frames=(
                enable_asymmetric_recovery_min_frames
            ),
            recovery_add_min_frames=recovery_add_min_frames,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def set_video_id(self, video_id: str | None) -> None:
        """案 R3 改: 動画 ID を設定する。

        per-video ぷよ色プロファイルのロードに使用する。
        bg_fp 採取完了直後に profile が自動ロードされる。

        Args:
            video_id: 動画 ID (例: "v29")。None でリセット。
        """
        self._video_id = video_id

    def _should_run_large_roi_scan(self, frame_idx: int) -> bool:
        """大 ROI 走査 (match_end / telop) をこのフレームで実行すべきか。

        間引きが無効 (既定) なら常に True = 従来通り毎フレーム実行 (bit-identical)。
        有効時は LARGE_ROI_THROTTLE_FRAMES に 1 回だけ True を返す。

        Args:
            frame_idx: 動画全体での絶対フレーム番号。
        """
        if not self._enable_large_roi_throttle:
            return True
        interval = max(1, self._large_roi_throttle_frames)
        return (frame_idx % interval) == 0

    def tsumo_count(self, side: str) -> int:
        """試合開始からの確定ツモ設置数 (手数, I-1 指標用 getter)。

        内部の `_tsumo_count_Xp` Counter は TSUMO_FALL→STABLE 着地ごとに
        ペアの 2 色を各 +1 する (= 1 ツモで合計 +2)。本 getter は
        Counter の総和を 2 で割り「設置したツモ数 (手数)」を返す。
        試合境界 (score 大幅減少 / MENU) で Counter は自動 clear されるため、
        この値は現在の試合開始からの相対手数となる。

        Args:
            side: "1P" または "2P"。

        Returns:
            int: 現在の試合での確定ツモ設置数 (手数)。
        """
        counter = self._tsumo_count_1p if side == "1P" else self._tsumo_count_2p
        return int(sum(counter.values()) // 2)

    def reset(self) -> None:
        """全 state を初期化 (試合切替時など)。"""
        self._sm_1p.reset()
        self._sm_2p.reset()
        self._gen_1p.reset()
        self._gen_2p.reset()
        self._drift_1p.reset()
        self._drift_2p.reset()
        self._active_chain_1p = None
        self._chain_until_1p = 0.0
        self._active_chain_2p = None
        self._chain_until_2p = 0.0
        # 根治 (2026-07-23): 退避 ChainEvent も reset 時にクリア。
        self._last_chain_event_for_settle_1p = None
        self._last_chain_event_for_settle_2p = None
        # 反復4 (2026-07-23): board_none_reason 診断計装用 state も
        # 試合切替 (reset) 時にクリア (= 新しい試合では cold_start から)。
        self._ever_had_confirmed_1p = False
        self._ever_had_confirmed_2p = False
        self._pending_menu_reset_1p = False
        self._pending_menu_reset_2p = False
        # 反復5 (2026-07-23): 物理推論スルー state も試合切替時にクリア。
        self._chain_estimate_result_1p = None
        self._chain_estimate_result_2p = None
        self._chain_estimate_trigger_1p = 0.0
        self._chain_estimate_trigger_2p = 0.0
        self._chain_estimate_end_1p = 0.0
        self._chain_estimate_end_2p = 0.0
        self._chain_estimate_low_confidence_1p = False
        self._chain_estimate_low_confidence_2p = False
        # 案1 (2026-07-23): stale_hold state も試合切替時にクリア。
        self._chain_estimate_last_board_1p = None
        self._chain_estimate_last_board_2p = None
        self._chain_estimate_stale_since_1p = None
        self._chain_estimate_stale_since_2p = None
        self._chain_verify_pending_1p = None
        self._chain_verify_pending_2p = None
        self._cnn_history_1p.clear()
        self._cnn_history_2p.clear()
        self._last_active_frame_idx = -1
        self._match_active_started_frame = -1
        self._last_active_frame_time = -1.0
        self._match_active_started_time = -1.0
        self._bg_fp_captured = False
        # ImageReader の bg_fp も解除 + I1 対応 A: pre_capture_mode も reset
        if hasattr(self._reader, "set_background_fingerprints"):
            self._reader.set_background_fingerprints(None, None)
        if hasattr(self._reader, "set_pre_capture_mode"):
            self._reader.set_pre_capture_mode(True)
        if self._score_tracker_1p is not None:
            self._score_tracker_1p.reset()
        if self._score_tracker_2p is not None:
            self._score_tracker_2p.reset()
        # chain tracker 内部 state は再構築 (リセット API なし)。
        # 修正C: debounce 設定は _chain_debounce_confirm_frames から引き継ぐ。
        if self._chain_tracker_1p is not None:
            self._chain_tracker_1p = VideoChainTracker(
                debounce_confirm_frames=self._chain_debounce_confirm_frames,
            )
        if self._chain_tracker_2p is not None:
            self._chain_tracker_2p = VideoChainTracker(
                debounce_confirm_frames=self._chain_debounce_confirm_frames,
            )
        # cycle 71d (案 D8): VideoChainTracker 入力 cache もリセット.
        self._prev_confirmed_1p = None
        self._prev_confirmed_2p = None
        # cycle 71f (提案 A): score 履歴もリセット.
        self._recent_scores_1p = []
        self._recent_scores_2p = []
        self._recent_score_times_1p = []
        self._recent_score_times_2p = []
        # cycle 71h: 着地後 vote 蓄積もリセット.
        self._pending_landing_vote_1p = []
        self._pending_landing_vote_2p = []
        # cycle 71j (案 1a): 直前 STABLE confirmed もリセット.
        self._prev_stable_confirmed_1p = None
        self._prev_stable_confirmed_2p = None
        # cycle 71n (案 θ): STABLE CNN 履歴もリセット.
        self._stable_cnn_history_1p.clear()
        self._stable_cnn_history_2p.clear()
        # NEXT slide detector + prev_frame もリセット
        if self._slide_detector_1p is not None:
            self._slide_detector_1p.reset()
        if self._slide_detector_2p is not None:
            self._slide_detector_2p.reset()
        self._prev_frame = None
        # T4: PuyoErasureMonitor + StaticBoardMask もリセット
        self._erasure_monitor_1p.reset()
        self._erasure_monitor_2p.reset()
        self._static_mask_captured = False
        if hasattr(self._reader, "set_static_mask"):
            self._reader.set_static_mask(None, None)
        # C2 StableTransitionMonitor もリセット
        self._transition_monitor_1p.reset()
        self._transition_monitor_2p.reset()
        # B1 PiecePersistenceGuard: 試合切替時に保護リセット
        if self._piece_persistence_1p is not None:
            self._piece_persistence_1p.reset()
        if self._piece_persistence_2p is not None:
            self._piece_persistence_2p.reset()
        # tier1 warmup guard リセット
        self._tier1_warmup_remaining_1p = 0
        self._tier1_warmup_remaining_2p = 0
        # 経路 A': OJAMA 専用 tier1 warmup guard リセット
        self._ojama_tier1_warmup_remaining_1p = 0
        self._ojama_tier1_warmup_remaining_2p = 0
        # game-event ベース連鎖終了 (C-1/C-2) リセット
        self._chain_event_max_until_1p = 0.0
        self._chain_event_max_until_2p = 0.0
        self._chain_start_next_1p = None
        self._chain_start_next_2p = None
        # ※②お邪魔信号撤去 (2026-06-01): _chain_start_board_* は削除済。
        # X1: CHAIN 突入時刻リセット
        self._chain_entry_t_1p = 0.0
        self._chain_entry_t_2p = 0.0
        # 機能C: CHAIN→STABLE warmup 凍結終了時刻リセット
        self._chain_exit_until_1p = 0.0
        self._chain_exit_until_2p = 0.0
        # 案 Y-4 deferred landing state リセット
        self._deferred_landing_1p = None
        self._deferred_landing_2p = None
        self._deferred_just_committed_1p = False
        self._deferred_just_committed_2p = False
        # 不具合B 対処: 予告おじゃま発光ガード state リセット (試合切替時)
        if self._glow_guard_1p is not None:
            from src.ojama_warning_glow_guard import GlowGuardState as _GGS
            self._glow_guard_1p = _GGS()
        if self._glow_guard_2p is not None:
            from src.ojama_warning_glow_guard import GlowGuardState as _GGS
            self._glow_guard_2p = _GGS()
        # 案P3: MAX_HOLD 超過 expired フラグをリセット (試合切替時)
        self._chain_max_hold_expired_1p = False
        self._chain_max_hold_expired_2p = False
        # feat/gravity-settle-2026-06-05: GravitySettleDetector 内部 state をリセット。
        # BoardStateMachine.detectors から GravitySettleDetector を探してリセットする。
        if self._enable_gravity_settle_state:
            self._reset_gravity_settle_detectors()
        # 追修 (2026-07-25): score リセット境界検知用キャッシュもリセット。
        self._prev_score_for_reset_1p = None
        self._prev_score_for_reset_2p = None
        self._match_start_boundary_latched = False
        # score-reset 境界誤発火修正 (2026-07-26): デバウンスカウンタもリセット。
        self._score_reset_boundary_streak = 0

    def update(
        self, frame_idx: int, time_sec: float, frame: np.ndarray,
    ) -> PipelineResult:
        """1 frame 投入、結果を返す."""
        # 0. 解像度依存 S_min 調整は呼び出し側 (viz/diag script) が
        # set_resolution_aware_s_min で明示設定する. pipeline.update に渡る
        # frame は image_reader で 1920x1080 にリサイズ済のため、 ここでの
        # 自動検出は不可能.
        # 1. 試合状態判定 + hysteresis
        match_res = self._match_detector.detect(frame)
        raw_active = (
            match_res.state == MatchState.IN_MATCH
            or self._force_in_match
        )

        # 1a. ScoreZero / MatchEnd / Telop による境界補強 (A 統合):
        # - score_zero (両側 00000000) → 試合外確定 = 強制 inactive
        # - match_end (やった/ばたんきゅー) → 試合終了演出中、lockdown_sec 内は
        #   recognition 凍結
        # - telop (中央テロップ) → 視覚的占有を後段で活用 (将来 cell mask)
        score_zero_both = False
        # 大 ROI 走査の間引き判定に使う (下の match_end / telop で参照)
        if self._score_zero_detector is not None:
            try:
                sz = self._score_zero_detector.detect(frame)
                score_zero_both = bool(sz.both_zero)
            except Exception:
                pass
        match_end_locked = False
        if self._match_end_detector is not None:
            # 大 ROI 走査 (800x600) の間引き: 有効時は LARGE_ROI_THROTTLE_FRAMES に
            # 1 回だけ実行し、間のフレームは前回結果を流用する。
            # 既定 OFF (フラグ無効時は従来通り毎フレーム実行 = bit-identical)。
            if self._should_run_large_roi_scan(frame_idx):
                try:
                    match_end_locked = bool(
                        self._match_end_detector.update(frame, time_sec),
                    )
                    self._last_match_end_locked = match_end_locked
                except Exception:
                    pass
            else:
                match_end_locked = self._last_match_end_locked
        # cycle 71f (提案 A): score 動き情報を追跡 (= 試合 2 開始直後の演出で
        # MatchStateDetector / MatchEndDetector が「試合外」 と判定しても、
        # score が継続的に動いていれば「試合中」 と判定する確実な信号).
        cur_score_1p = (
            self._score_tracker_1p.last_score
            if self._score_tracker_1p is not None else None
        )
        cur_score_2p = (
            self._score_tracker_2p.last_score
            if self._score_tracker_2p is not None else None
        )
        self._recent_scores_1p.append(cur_score_1p)
        self._recent_scores_2p.append(cur_score_2p)
        self._recent_score_times_1p.append(time_sec)
        self._recent_score_times_2p.append(time_sec)
        # フレーム定数→時間定数化 Stage1 (2026-07-25): 旧「直近 N 件」保持を
        # 「直近 SCORE_MOVE_WINDOW_SEC 秒」保持に置換 (_trim_score_window 参照)。
        # 60fps 動画では 1 要素 ≒ 1/60 秒のため件数ベースの窓と一致し bit-identical。
        self._recent_scores_1p, self._recent_score_times_1p = (
            self._trim_score_window(
                self._recent_scores_1p, self._recent_score_times_1p, time_sec,
            )
        )
        self._recent_scores_2p, self._recent_score_times_2p = (
            self._trim_score_window(
                self._recent_scores_2p, self._recent_score_times_2p, time_sec,
            )
        )
        score_actively_moving = self._is_score_actively_moving(
            self._recent_scores_1p
        ) or self._is_score_actively_moving(self._recent_scores_2p)
        # 強い「試合外」シグナル (hysteresis を上書き)
        # 2026-05-10 FIX-C: score=0 でも cnn_board に puyo があれば試合中継続
        # (試合開始直後の最初の数手が menu 誤判定される問題を回避)。
        hard_match_off = score_zero_both or match_end_locked
        if hard_match_off:
            # match_end は確定信号、 score_zero は puyo 確認で否定
            puyo_observed = False
            try:
                from src.board import COLOR_EMPTY
                p1_count = sum(
                    1 for r in range(13) for c in range(6)
                    if int(cnn_1p_raw.get(r, c)) != COLOR_EMPTY
                ) if 'cnn_1p_raw' in dir() else 0
            except Exception:
                p1_count = 0
            # cnn_1p_raw / cnn_2p_raw はまだ未計算なので、 sm.context.confirmed_board
            # から puyo 数を推定。
            if score_zero_both and not match_end_locked:
                p1_b = self._sm_1p.context.confirmed_board
                p2_b = self._sm_2p.context.confirmed_board
                if p1_b is not None and p1_b.count_puyos() >= 2:
                    score_zero_both = False  # 既に puyo 観測 = 試合中
                if p2_b is not None and p2_b.count_puyos() >= 2:
                    score_zero_both = False
                hard_match_off = score_zero_both or match_end_locked
            if hard_match_off:
                raw_active = False
        # Telop visible 状態を保存 (EffectPhaseDetector で利用)
        # 修正2 (2026-07-30): is_visible(frame) は内部で detect(frame) を呼ぶだけの
        # 薄いラッパーなので、detect() を直接呼んで結果全体 (bbox 込み) を保持する。
        # これにより後段の self._reader.read_both_boards() へ telop_result として
        # 引き渡し、ImageReader 側の 2 回目の detect() 実行 (二重走査) を省略できる。
        # is_visible() だけ呼ぶ場合と挙動は bit-identical (同じ detect() 呼び出し)。
        if self._telop_detector is not None:
            # 大 ROI 走査 (720x400) の間引き: match_end と同じ方針。
            # 既定 OFF では毎フレーム実行され従来と bit-identical。
            if self._should_run_large_roi_scan(frame_idx):
                try:
                    self._last_telop_result = self._telop_detector.detect(frame)
                    self._last_telop_visible = bool(
                        self._last_telop_result.is_visible,
                    )
                except Exception:
                    self._last_telop_visible = False
                    self._last_telop_result = None
            # else: 前回の _last_telop_result / _last_telop_visible をそのまま流用
        else:
            self._last_telop_visible = False
            self._last_telop_result = None
        # score-based 補強: ScoreOcr が 1 度でも score>0 を読めれば
        # 試合中確定 (試合外/メニュー画面では 8 桁数字は読めないか 0 のまま)。
        # MatchStateDetector の試合中誤判定 (= NOT_IN_MATCH 返却) を補正。
        if not raw_active and (
            (self._score_tracker_1p is not None
             and (self._score_tracker_1p.last_score or 0) > 0)
            or (self._score_tracker_2p is not None
                and (self._score_tracker_2p.last_score or 0) > 0)
        ):
            raw_active = True
        # 直前 N 秒以内に active 観測歴があれば強制 True (1 frame ぶれ吸収)。
        # フレーム定数→時間定数化 Stage1 (2026-07-25): 旧 `frame_idx -
        # self._last_active_frame_idx` (frame 差分) を time_sec 差分に置換。
        # 60fps 動画では (frame_idx 差分)/60 == time_sec 差分 が恒等式のため
        # bit-identical、30fps 動画では実秒基準になる。
        recent_active = (
            self._last_active_frame_time >= 0
            and (time_sec - self._last_active_frame_time)
            <= self.MATCH_ACTIVE_HOLD_SEC
        )
        # 1P/2P state machine が現在 NON-STABLE state にある場合も active 強制
        # (= state machine 内部で active 認識中 → MENU に倒さない)
        sm_active = (
            self._sm_1p.context.state in (
                BoardState.STABLE, BoardState.TSUMO_FALL,
                BoardState.CHAIN, BoardState.OJAMA_FALL, BoardState.EFFECT,
                BoardState.GRAVITY_SETTLE,  # feat/gravity-settle-2026-06-05
            )
            or self._sm_2p.context.state in (
                BoardState.STABLE, BoardState.TSUMO_FALL,
                BoardState.CHAIN, BoardState.OJAMA_FALL, BoardState.EFFECT,
                BoardState.GRAVITY_SETTLE,  # feat/gravity-settle-2026-06-05
            )
        )
        # 反復3 (2026-07-23): 連鎖/重力沈下中は score 急変+フラッシュ演出で
        # ScoreZeroDetector/MatchEndDetector が瞬間誤爆しやすい
        # (物理harness実測: 連鎖中の誤 hard_match_off 率 0.95)。
        # CHAIN/GRAVITY_SETTLE 中は sm_active による保護が effective_hard_off に
        # 上書きされないよう、この 2 state 限定で hard_match_off を無効化する。
        # 正当な試合終了 (致死連鎖でゲームセット等) は連鎖アニメ完了後の
        # STABLE/MENU 遷移時に演出が視認可能になる想定のため、CHAIN/
        # GRAVITY_SETTLE 中限定の抑制では検出を妨げない (連鎖終了後は
        # chain_in_progress=False に戻り通常判定に復帰する)。
        chain_in_progress = (
            self._sm_1p.context.state in (
                BoardState.CHAIN, BoardState.GRAVITY_SETTLE,
            )
            or self._sm_2p.context.state in (
                BoardState.CHAIN, BoardState.GRAVITY_SETTLE,
            )
        )
        # hard_match_off は hysteresis (recent/sm) を上書きする確定シグナル.
        # cycle 71f (提案 A): score が直近 window 内で SCORE_MOVE_MIN_DELTA 以上
        # 動いていれば、 hard_match_off を打ち消して試合中継続を保証する.
        # 「演出/READY/GO! で MatchEnd が誤発火するが score は動いている」
        # シナリオ (= v50 51-63s) を解消.
        effective_hard_off = (
            hard_match_off and not score_actively_moving and not chain_in_progress
        )
        is_active = (
            (raw_active or recent_active or sm_active or score_actively_moving)
            and not effective_hard_off
        )

        # 試合 active 開始 frame の記録 (chain ban の起点)
        if is_active:
            if self._match_active_started_frame < 0:
                self._match_active_started_frame = frame_idx
                self._match_active_started_time = time_sec
            self._last_active_frame_idx = frame_idx
            self._last_active_frame_time = time_sec
        else:
            # 試合 active が完全に切れたら start もリセット
            self._match_active_started_frame = -1
            self._match_active_started_time = -1.0
            self._bg_fp_captured = False
            self._bg_frame_buffer.clear()
            if hasattr(self._reader, "set_background_fingerprints"):
                self._reader.set_background_fingerprints(None, None)
            # I1 対応 A: 試合終了時も pre_capture_mode を on に戻す
            if hasattr(self._reader, "set_pre_capture_mode"):
                self._reader.set_pre_capture_mode(True)
            # サイクル66: NEXT 累積制約も試合切り替えでリセット
            self._tsumo_count_1p.clear()
            self._tsumo_count_2p.clear()
            self._pending_tsumo_1p.clear()
            self._pending_tsumo_2p.clear()
            self._last_seen_next_1p = None
            self._last_seen_next_2p = None
            self._constraint_valid_1p = True
            self._constraint_valid_2p = True
            # cycle 29: landing_pending も試合切替でリセット
            self._landing_pending_1p = None
            self._landing_pending_2p = None
            # 着地色修正 案1 修正版 (2026-06-01): last_consumed も試合切替でリセット
            self._last_consumed_color_1p = None
            self._last_consumed_color_2p = None
            # cycle 71v-B: ever_seen も試合切り替えでリセット
            self._ever_seen_colors_1p.clear()
            self._ever_seen_colors_2p.clear()
            # 機能D: 試合切替で掛け算式 連続フレームカウンタをリセット
            self._formula_consec_1p = 0
            self._formula_consec_2p = 0

        # 2. CNN raw board 取得 (BG FP 採取より先に必要)
        # I1 対応 A: bg_fp 採取前は pre_capture_mode を on にして tier 1 を無効化
        # (= HSV-only 経路に強制)。 bg_fp 採取後は通常モードに戻す。
        if hasattr(self._reader, "set_pre_capture_mode"):
            self._reader.set_pre_capture_mode(not self._bg_fp_captured)
        # tier1 warmup guard: NON-STABLE → STABLE 遷移直後 TIER1_WARMUP_FRAMES は
        # tier1 をスキップして着地直後の cell が誤 EMPTY 化されるのを防ぐ。
        # カウンタは _step_side の state 遷移後に更新するため、 ここでは現在値を読む。
        # 経路 A': OJAMA 専用 warmup (ojama_remaining > 0) が汎用 warmup と OR で発火。
        _skip_t1_1p = (
            (self._enable_tier1_warmup and self._tier1_warmup_remaining_1p > 0)
            or (self._enable_ojama_tier1_warmup and self._ojama_tier1_warmup_remaining_1p > 0)
        )
        _skip_t1_2p = (
            (self._enable_tier1_warmup and self._tier1_warmup_remaining_2p > 0)
            or (self._enable_ojama_tier1_warmup and self._ojama_tier1_warmup_remaining_2p > 0)
        )
        cnn_1p_raw, cnn_2p_raw = self._reader.read_both_boards(
            frame,
            skip_tier1_1p=_skip_t1_1p,
            skip_tier1_2p=_skip_t1_2p,
            # 修正2 (2026-07-30): 上で計算済の telop 検出結果を使い回す
            # (ImageReader 側の detect() 二重走査を防ぐ)。
            telop_result=self._last_telop_result,
        )

        # 背景 FP 自動採取 (Phase C-5: robust 化):
        # 試合 active 開始から 5 frame 経過後、CNN 盤面が puyo 0 個 (= 真の空盤面)
        # の frame のみバッファに蓄積。5 frame 蓄積で capture_robust_fingerprint
        # にて中央値 FP を生成 (キャラ顔のチラつき・UI 装飾の瞬間値を平均化)。
        if (
            is_active
            and not self._bg_fp_captured
            and self._match_active_started_frame >= 0
            and (frame_idx - self._match_active_started_frame) >= 5
        ):
            # 真の空盤面チェック (= cnn 盤面が両 side とも 0 puyo)
            # cycle 71h (v50 試合 2 開始時 bg_fp 採取の鶏卵問題対策):
            # 試合切替直後 (= match_age < BG_FP_FORCE_WINDOW_FRAMES) で CNN が
            # 背景を puyo として誤認すると puyo_count_total > 0 になり bg_fp 採取が
            # 始まらない. 試合切替直後の short window では puyo_count_total <=
            # BG_FP_FORCE_MAX_PUYO まで許容して bg_fp 採取を強制起動する.
            puyo_count_total = (
                cnn_1p_raw.count_puyos() + cnn_2p_raw.count_puyos()
            )
            # フレーム定数→時間定数化 Stage1 (2026-07-25): 旧 `frame_idx -
            # self._match_active_started_frame` (frame 差分) を time_sec 差分
            # に置換。60fps では bit-identical、30fps では実秒基準になる。
            match_age_sec = time_sec - self._match_active_started_time
            bg_fp_relaxed = (
                match_age_sec <= self.BG_FP_FORCE_WINDOW_SEC
                and puyo_count_total <= self._bg_fp_force_max_puyo
            )
            if puyo_count_total == 0 or bg_fp_relaxed:
                self._bg_frame_buffer.append(frame.copy())
            else:
                self._bg_frame_buffer.clear()  # 途中で puyo 出現したらリセット

            if len(self._bg_frame_buffer) >= 5:
                try:
                    p1r = DEFAULT_P1_REGION
                    p2r = DEFAULT_P2_REGION
                    frames_list = list(self._bg_frame_buffer)
                    # 案 d: PatchBackgroundFingerprint で 1P/2P を一括採取
                    bg1, bg2 = capture_patch_pair_robust(
                        frames_list,
                        (p1r.x, p1r.y, p1r.width, p1r.height),
                        (p2r.x, p2r.y, p2r.width, p2r.height),
                    )
                    if hasattr(self._reader, "set_background_fingerprints"):
                        self._reader.set_background_fingerprints(bg1, bg2)
                    # T4: StaticBoardMask を同フレームバッファから採取して inject
                    # 2026-05-28 user 目視で v40m7 お邪魔ぷよ認識消失で撤回。
                    # PuyoErasureMonitor は「色→空」 のみで「初回から空」 系
                    # fail-silent を検知できず。 将来 ojama 救済機構 (案 b/c 等)
                    # 追加で再評価可能性のため inject コードは残置。
                    # 詳細: memory feedback_fail_silent_initial_zero.md
                    if False and not self._static_mask_captured:  # noqa: SIM210
                        try:
                            from src.background_fingerprint import (
                                capture_static_mask_pair,
                            )
                            smask1, smask2 = capture_static_mask_pair(
                                frames_list,
                                (p1r.x, p1r.y, p1r.width, p1r.height),
                                (p2r.x, p2r.y, p2r.width, p2r.height),
                            )
                            if hasattr(self._reader, "set_static_mask"):
                                self._reader.set_static_mask(smask1, smask2)
                            self._static_mask_captured = True
                        except Exception:
                            pass  # 採取失敗は無視 (保守的動作維持)
                    self._bg_fp_captured = True
                    # I1 対応 A: bg_fp 採取完了 → pre_capture_mode を解除
                    if hasattr(self._reader, "set_pre_capture_mode"):
                        self._reader.set_pre_capture_mode(False)
                    # 案 R3 改: bg_fp 採取完了直後に per-video プロファイル自動ロード
                    # 2026-05-28 fail-silent 確認 (v40m7 + v95m15 でぷよが消える) により
                    # デフォルト OFF 化で撤回。コードは将来 threshold 緩和 sweep で
                    # 再評価可能性のため残置。set_puyo_profile_db / ImageReader 側も残置。
                    if False and hasattr(self._reader, "set_puyo_profile_db"):  # noqa: SIM210
                        try:
                            from src.puyo_color_profile import PuyoColorProfileDB
                            profile_db = PuyoColorProfileDB.load_for_video(
                                self._video_id,
                            )
                            self._reader.set_puyo_profile_db(profile_db)
                        except Exception as _e:
                            pass  # プロファイルロード失敗は無視 (保守的動作維持)
                    self._bg_frame_buffer.clear()
                except Exception:
                    pass

        # 2'. (旧) CNN raw board 取得 ← 上に移動済
        # 時系列平均 (δ): 直近 N frame の majority vote。N=1 で no-op。
        self._cnn_history_1p.append(cnn_1p_raw)
        self._cnn_history_2p.append(cnn_2p_raw)
        if self._smoothing_n > 1:
            cnn_1p = self._smooth_board(self._cnn_history_1p)
            cnn_2p = self._smooth_board(self._cnn_history_2p)
        else:
            cnn_1p = cnn_1p_raw
            cnn_2p = cnn_2p_raw

        # 3. 連鎖検出 (各 side 独立)。
        # VideoChainTracker は drop 観測 frame で 1 度だけ ChainEvent を返す。
        # state machine が CHAIN にロックされ続けるよう、event 受信後
        # chain_hold_per_step_sec × chain_count 秒間 signals に保持する。
        # 試合開始から CHAIN_BAN_SEC_AFTER_MATCH_START 秒以内の event は破棄
        # (1 手目から連鎖はあり得ない、誤検出 ban)。
        # フレーム定数→時間定数化 Stage1 (2026-07-25): 旧 `frame_idx -
        # self._match_active_started_frame` (frame 差分) を time_sec 差分に
        # 置換。60fps では bit-identical、30fps では実秒基準になる。
        chain_banned = (
            self._match_active_started_time >= 0
            and (time_sec - self._match_active_started_time)
            < self.CHAIN_BAN_SEC_AFTER_MATCH_START
        )
        # cycle 71d (案 D8): VideoChainTracker への入力は前 frame の confirmed_board.
        # raw CNN 振動 (= cnn 32↔27 1 frame スパイク) は confirmed が 1 frame では動かないため
        # 吸収される. 初回 frame は confirmed が無いため cnn_1p / cnn_2p にフォールバック.
        board_for_tracker_1p = (
            self._prev_confirmed_1p if self._prev_confirmed_1p is not None
            else cnn_1p
        )
        board_for_tracker_2p = (
            self._prev_confirmed_2p if self._prev_confirmed_2p is not None
            else cnn_2p
        )
        if is_active and self._chain_tracker_1p is not None:
            ev = self._chain_tracker_1p.update(time_sec, board_for_tracker_1p)
            if ev is not None and not chain_banned:
                self._active_chain_1p = ev
                # 反復5 Step2: 物理推論スルー開始 (起点盤面から連鎖を前進)
                self._start_chain_estimate("1P", ev)
                # 全消し連鎖は overlay 表示時間ぶん CHAIN を延長して、 CHAIN→STABLE
                # 遷移時の _merge_diff_only が overlay corrupted cnn_board を
                # 使わないようにする (v50 全消し overlay 誤認の構造的解消)。
                extra_all_clear = (
                    self.ALL_CLEAR_OVERLAY_HOLD_SEC if ev.is_all_clear else 0.0
                )
                self._chain_until_1p = (
                    time_sec
                    + self._chain_hold_base_sec
                    + self._chain_hold_per_step_sec * ev.chain_count
                    + extra_all_clear
                )
                # game-event モード: 安全弁上限 + 連鎖開始時 next_pair snapshot を記録。
                # enable_game_event_chain_exit=False 時も安全弁は無害 (使われない)。
                # ※②お邪魔信号撤去 (2026-06-01) により board snapshot は不要になった。
                if self._enable_game_event_chain_exit:
                    self._chain_event_max_until_1p = (
                        time_sec + self._chain_max_hold_sec
                    )
                    self._chain_start_next_1p = self._last_seen_next_1p
                # X1: CHAIN 突入時刻を記録 (enable_chain_min_display 用)。
                # enable_game_event_chain_exit 非依存で常に更新。
                self._chain_entry_t_1p = time_sec
        if is_active and self._chain_tracker_2p is not None:
            ev = self._chain_tracker_2p.update(time_sec, board_for_tracker_2p)
            if ev is not None and not chain_banned:
                self._active_chain_2p = ev
                # 反復5 Step2: 物理推論スルー開始 (起点盤面から連鎖を前進)
                self._start_chain_estimate("2P", ev)
                extra_all_clear = (
                    self.ALL_CLEAR_OVERLAY_HOLD_SEC if ev.is_all_clear else 0.0
                )
                self._chain_until_2p = (
                    time_sec
                    + self._chain_hold_base_sec
                    + self._chain_hold_per_step_sec * ev.chain_count
                    + extra_all_clear
                )
                # game-event モード: 安全弁上限 + 連鎖開始時 next_pair snapshot を記録。
                # ※②お邪魔信号撤去 (2026-06-01) により board snapshot は不要になった。
                if self._enable_game_event_chain_exit:
                    self._chain_event_max_until_2p = (
                        time_sec + self._chain_max_hold_sec
                    )
                    self._chain_start_next_2p = self._last_seen_next_2p
                # X1: CHAIN 突入時刻を記録 (enable_chain_min_display 用)。
                # enable_game_event_chain_exit 非依存で常に更新。
                self._chain_entry_t_2p = time_sec

        # 有効期限内の chain_event を signals に乗せる
        # game-event モード (enable_game_event_chain_exit=True) の場合、
        # timing hold だけでは終了させず「最大 CHAIN_MAX_HOLD_SEC」まで維持する。
        # 実際の終了は 4b. next_pair 計算後に game-event チェックで行う。
        # 案P3: 直前 frame の expired フラグをリセットしてから再評価する。
        self._chain_max_hold_expired_1p = False
        self._chain_max_hold_expired_2p = False
        chain_ev_1p: ChainEvent | None = None
        chain_ev_2p: ChainEvent | None = None
        if self._active_chain_1p is not None:
            # game-event モード: timing holdを超えても max_until まで維持
            eff_until_1p = (
                self._chain_event_max_until_1p
                if (
                    self._enable_game_event_chain_exit
                    and self._chain_event_max_until_1p > self._chain_until_1p
                )
                else self._chain_until_1p
            )
            if time_sec < eff_until_1p:
                chain_ev_1p = self._active_chain_1p
            else:
                # 案P3: MAX_HOLD 超過による強制クリア → expired フラグを立てる
                if self._enable_chain_max_hold_override:
                    self._chain_max_hold_expired_1p = True
                # 根治: GRAVITY_SETTLE 経由の final_board 反映用に退避してからクリア
                self._stash_and_clear_active_chain("1P")
        if self._active_chain_2p is not None:
            eff_until_2p = (
                self._chain_event_max_until_2p
                if (
                    self._enable_game_event_chain_exit
                    and self._chain_event_max_until_2p > self._chain_until_2p
                )
                else self._chain_until_2p
            )
            if time_sec < eff_until_2p:
                chain_ev_2p = self._active_chain_2p
            else:
                # 案P3: MAX_HOLD 超過による強制クリア → expired フラグを立てる
                if self._enable_chain_max_hold_override:
                    self._chain_max_hold_expired_2p = True
                # 根治: GRAVITY_SETTLE 経由の final_board 反映用に退避してからクリア
                self._stash_and_clear_active_chain("2P")

        # 4. score 差分
        # 修正1 (2026-07-30): 生 score 値 (_score_ocr_val_Xp) も同時に受け取り、
        # 機能D (_check_formula_detected) が同一 frame・同一 side のスコアを
        # 再度フルで読み直す (score_ocr.read_side 完全重複呼び出し) のを防ぐ。
        score_d_1p, _score_ocr_val_1p = self._update_score_tracker(
            self._score_tracker_1p, frame,
        )
        score_d_2p, _score_ocr_val_2p = self._update_score_tracker(
            self._score_tracker_2p, frame,
        )

        # 追修 (2026-07-25): force_in_match=True 構成では is_match_active=False
        # 分岐 (MENU 強制、confirmed_board 等クリアの発火点) が一度も走らない
        # ため、score リセット境界 (新ゲーム開始/全消し) をここで直接検知して
        # pipeline 全体を明示的にリセットする。sm_1p/sm_2p の confirmed_board
        # だけでなく、_active_chain_Xp / _chain_estimate_last_board_Xp
        # (CHAIN 中の estimated_board 表示に使われる stale_hold キャッシュ) 等
        # 試合単位の全キャッシュに前試合の値が残るため (2026-07-25 実測:
        # sm 側 5 field のみのクリアでは 2P 側の推定盤面表示に幽霊セルが残存)、
        # self.reset() (既存の包括的試合切替 API) をそのまま流用する。
        # edge-trigger ラッチ: 「両者スコアほぼ0」は真の試合開始直後の数秒間
        # 継続して真になりうるため、境界条件が一旦 False に戻るまでは
        # 毎フレーム re-fire しないようにする (連続 reset() で序盤の tsumo
        # 認識進行を妨げないため)。
        # enable_match_start_full_clear=False (default) では無効 (backwards compat)。
        if self._enable_match_start_full_clear:
            cur_score_1p = (
                self._score_tracker_1p.last_score
                if self._score_tracker_1p is not None else None
            )
            cur_score_2p = (
                self._score_tracker_2p.last_score
                if self._score_tracker_2p is not None else None
            )
            # 誤発火修正 (2026-07-26): boundary_candidate は 1 フレーム単位の
            # 生の境界候補 (strict=False なら従来通り即 fire 相当)。
            # strict=True の場合は SCORE_RESET_BOUNDARY_DEBOUNCE_FRAMES 回
            # 連続成立して初めて実際の発火 (boundary_now) とみなす。
            # ラッチの解除判定は生の boundary_candidate で行う (デバウンス中の
            # 一時的な boundary_now=False で誤ってラッチ解除しないため。
            # 誤解除すると継続 near-zero 期間中に再発火してしまう)。
            boundary_candidate = _is_score_reset_boundary(
                cur_score_1p, cur_score_2p,
                self._prev_score_for_reset_1p, self._prev_score_for_reset_2p,
                strict=self._enable_score_reset_strict,
            )
            if boundary_candidate:
                self._score_reset_boundary_streak += 1
            else:
                self._score_reset_boundary_streak = 0
            boundary_now = (
                self._score_reset_boundary_streak
                >= SCORE_RESET_BOUNDARY_DEBOUNCE_FRAMES
                if self._enable_score_reset_strict
                else boundary_candidate
            )
            if boundary_now and not self._match_start_boundary_latched:
                if os.environ.get(_DEBUG_RESET_PROBE_ENV):
                    tsumo_1p_before = self.tsumo_count("1P")
                    tsumo_2p_before = self.tsumo_count("2P")
                    print(
                        f"[reset_probe] t_sec={time_sec:.2f} "
                        f"cur_score_1p={cur_score_1p} cur_score_2p={cur_score_2p} "
                        f"prev_score_1p={self._prev_score_for_reset_1p} "
                        f"prev_score_2p={self._prev_score_for_reset_2p} "
                        f"tsumo_1p_before_reset={tsumo_1p_before} "
                        f"tsumo_2p_before_reset={tsumo_2p_before}",
                        flush=True,
                    )
                self.reset()
                self._match_start_boundary_latched = True
            elif not boundary_candidate:
                self._match_start_boundary_latched = False
            self._prev_score_for_reset_1p = cur_score_1p
            self._prev_score_for_reset_2p = cur_score_2p

        # 4a. 機能B: score 急増 CHAIN 早期発火 (enable_chain_score_early_fire=True 時のみ)。
        # VideoChainTracker の puyo 減少検知を待たずに、自 side score_delta が
        # CHAIN_SCORE_EARLY_FIRE_DELTA 以上急増した frame で即 CHAIN state に突入させる。
        # 既存 VideoChainTracker 経路との OR 追加 (フォールバック完全維持)。
        # score 取得失敗 (score_d == 0) 時は発火しない → OCR 失敗時は従来経路が継続。
        # chain_banned / STABLE 状態以外 (既に CHAIN 中) は発火をスキップ。
        if self._enable_chain_score_early_fire and is_active and not chain_banned:
            self._apply_chain_score_early_fire(
                side="1P", score_delta=score_d_1p, time_sec=time_sec,
                prev_confirmed=self._prev_confirmed_1p,
            )
            self._apply_chain_score_early_fire(
                side="2P", score_delta=score_d_2p, time_sec=time_sec,
                prev_confirmed=self._prev_confirmed_2p,
            )
            # 早期発火後の chain_ev を再評価 (timing hold / game-event hold に従う)
            if self._active_chain_1p is not None and chain_ev_1p is None:
                eff_until_1p = (
                    self._chain_event_max_until_1p
                    if (
                        self._enable_game_event_chain_exit
                        and self._chain_event_max_until_1p > self._chain_until_1p
                    )
                    else self._chain_until_1p
                )
                if time_sec < eff_until_1p:
                    chain_ev_1p = self._active_chain_1p
            if self._active_chain_2p is not None and chain_ev_2p is None:
                eff_until_2p = (
                    self._chain_event_max_until_2p
                    if (
                        self._enable_game_event_chain_exit
                        and self._chain_event_max_until_2p > self._chain_until_2p
                    )
                    else self._chain_until_2p
                )
                if time_sec < eff_until_2p:
                    chain_ev_2p = self._active_chain_2p

        # 4c. 機能D: 掛け算式検知 CHAIN 早期発火 (enable_chain_formula_detection=True 時のみ)。
        # score ROI の OCR が None (掛け算式表示) かつ ink_ratio > MIN かつ last_score > 0 が
        # CHAIN_FORMULA_CONSEC_FRAMES 連続で成立した frame で即 CHAIN state に突入させる。
        # 連続カウンタは pipeline 側で管理 (state-holding wrapper)、検出本体は stateless。
        # chain_banned / 既に CHAIN 中の場合は発火をスキップ。
        # 案X*(A): enable_chain_exit_next_signal=True の場合、既に CHAIN 中
        #   (active_chain 有効) なら機能D の発火 (_apply_chain_formula_early_fire 呼び出し)
        #   をスキップして _chain_event_max_until の延長を止める。
        #   連鎖開始検知 (active=None 時の初回発火) は引き続き有効。
        if self._enable_chain_formula_detection and is_active and not chain_banned:
            _last_1p = (
                self._score_tracker_1p.last_score
                if self._score_tracker_1p is not None else None
            )
            _last_2p = (
                self._score_tracker_2p.last_score
                if self._score_tracker_2p is not None else None
            )
            # 修正1 (2026-07-30): 上の 4. で同一 frame・同一 side を既に
            # score_ocr.read_side() 済のため、その結果 (_score_ocr_val_Xp)
            # を渡して完全重複読み (経路①②) を排除する。
            _formula_1p = self._check_formula_detected(
                frame, self._score_ocr, "1P", _last_1p,
                cached_score_val=_score_ocr_val_1p,
            )
            _formula_2p = self._check_formula_detected(
                frame, self._score_ocr, "2P", _last_2p,
                cached_score_val=_score_ocr_val_2p,
            )
            # 連続カウンタ更新
            self._formula_consec_1p = (
                self._formula_consec_1p + 1 if _formula_1p else 0
            )
            self._formula_consec_2p = (
                self._formula_consec_2p + 1 if _formula_2p else 0
            )
            # 連続フレーム数を満たしたら発火
            # 案X*(A): フラグ ON かつ既に CHAIN 中 (active_chain 有効) なら発火をスキップ
            _formula_skip_1p = (
                self._enable_chain_exit_next_signal
                and self._active_chain_1p is not None
            )
            if (
                self._formula_consec_1p >= CHAIN_FORMULA_CONSEC_FRAMES
                and not _formula_skip_1p
            ):
                self._apply_chain_formula_early_fire(
                    side="1P", time_sec=time_sec,
                    prev_confirmed=self._prev_confirmed_1p,
                )
                self._formula_consec_1p = 0  # 発火後リセット
                if self._active_chain_1p is not None and chain_ev_1p is None:
                    eff_until_1p = (
                        self._chain_event_max_until_1p
                        if (
                            self._enable_game_event_chain_exit
                            and self._chain_event_max_until_1p > self._chain_until_1p
                        )
                        else self._chain_until_1p
                    )
                    if time_sec < eff_until_1p:
                        chain_ev_1p = self._active_chain_1p
            _formula_skip_2p = (
                self._enable_chain_exit_next_signal
                and self._active_chain_2p is not None
            )
            if (
                self._formula_consec_2p >= CHAIN_FORMULA_CONSEC_FRAMES
                and not _formula_skip_2p
            ):
                self._apply_chain_formula_early_fire(
                    side="2P", time_sec=time_sec,
                    prev_confirmed=self._prev_confirmed_2p,
                )
                self._formula_consec_2p = 0  # 発火後リセット
                if self._active_chain_2p is not None and chain_ev_2p is None:
                    eff_until_2p = (
                        self._chain_event_max_until_2p
                        if (
                            self._enable_game_event_chain_exit
                            and self._chain_event_max_until_2p > self._chain_until_2p
                        )
                        else self._chain_until_2p
                    )
                    if time_sec < eff_until_2p:
                        chain_ev_2p = self._active_chain_2p

        # 4b. next_pair 検出 (1P/2P 両方、共通色なので detect_both で OK)
        # 2026-05-10: slide motion 中は ネクスト puyo が画面で動いており認識結果が
        # 不安定になるため、 検出結果を None に倒して state machine に渡さない。
        next_pair_1p: tuple[int, int] | None = None
        next_pair_2p: tuple[int, int] | None = None
        dnext_pair_1p: tuple[int, int] | None = None
        dnext_pair_2p: tuple[int, int] | None = None
        if is_active and self._next_detector is not None:
            try:
                both = self._next_detector.detect_both(frame)
                p1_pair = both.p1.next_pair
                p2_pair = both.p2.next_pair
                p1_dpair = both.p1.dnext_pair
                p2_dpair = both.p2.dnext_pair
                # slide_1p / slide_2p は後段の slide_motion 検出ロジックの結果を
                # 参照する必要があるが、 ここではまだ計算されていない。
                # 代わりに事前計算 (= 簡易版): _slide_detector.update で取得
                slide_check_1p = False
                slide_check_2p = False
                if (
                    self._slide_detector_1p is not None
                    and self._prev_frame is not None
                ):
                    try:
                        slide_check_1p = bool(
                            self._slide_detector_1p.update(
                                self._prev_frame, frame,
                            ).slide_motion,
                        )
                    except Exception:
                        slide_check_1p = False
                if (
                    self._slide_detector_2p is not None
                    and self._prev_frame is not None
                ):
                    try:
                        slide_check_2p = bool(
                            self._slide_detector_2p.update(
                                self._prev_frame, frame,
                            ).slide_motion,
                        )
                    except Exception:
                        slide_check_2p = False
                # 2026-05-11: 0 (= empty / 分類失敗) は除外. 有効色は 1-5.
                if not slide_check_1p and all(int(c) > 0 for c in p1_pair):
                    next_pair_1p = p1_pair
                if not slide_check_2p and all(int(c) > 0 for c in p2_pair):
                    next_pair_2p = p2_pair
                if not slide_check_1p and all(int(c) > 0 for c in p1_dpair):
                    dnext_pair_1p = p1_dpair
                if not slide_check_2p and all(int(c) > 0 for c in p2_dpair):
                    dnext_pair_2p = p2_dpair
            except Exception:
                pass

        # 4c. NEXT ROI スライド motion: 4b で既に update 済 (slide_check_*)
        # ここでは結果を再利用 (= 二重 update 禁止)
        slide_1p = locals().get("slide_check_1p", False)
        slide_2p = locals().get("slide_check_2p", False)

        # 4d. サイクル66/67: NEXT 履歴ベース累積色 count 更新
        # next_pair が前回と「異なる」 = 1 ツモ消費 = pending_tsumo に追加 (= 落下中)
        # 後段の TSUMO_FALL→STABLE で tsumo_count にコミット (= 着地確定).
        # 注: NEXT が稀に OJAMA(9) を返すケースを除外、 有効 puyo 色 (1-5) のみカウント.
        VALID_PUYO_COLORS = {1, 2, 3, 4, 5}
        if is_active and next_pair_1p is not None:
            top_v, bot_v = int(next_pair_1p[0]), int(next_pair_1p[1])
            if top_v in VALID_PUYO_COLORS and bot_v in VALID_PUYO_COLORS:
                if (self._last_seen_next_1p is not None
                        and self._last_seen_next_1p != (top_v, bot_v)):
                    consumed = self._last_seen_next_1p
                    if (consumed[0] in VALID_PUYO_COLORS
                            and consumed[1] in VALID_PUYO_COLORS):
                        # in-flight queue に追加 (TSUMO_FALL→STABLE で commit)
                        self._pending_tsumo_1p.append(consumed)
                        # cycle 29 (2026-05-18): NEXT 変化 = 着地確定 signal
                        # _step_side で grace + landing_vote を起動
                        self._landing_pending_1p = (frame_idx, consumed)
                        # 着地色修正 案1 修正版 (2026-06-01):
                        # _last_consumed_color はgraceクリアと独立して保持し
                        # 次の TSUMO_FALL→STABLE 着地フレームで falling_pair に使用する
                        self._last_consumed_color_1p = consumed
                self._last_seen_next_1p = (top_v, bot_v)
        if is_active and next_pair_2p is not None:
            top_v, bot_v = int(next_pair_2p[0]), int(next_pair_2p[1])
            if top_v in VALID_PUYO_COLORS and bot_v in VALID_PUYO_COLORS:
                if (self._last_seen_next_2p is not None
                        and self._last_seen_next_2p != (top_v, bot_v)):
                    consumed = self._last_seen_next_2p
                    if (consumed[0] in VALID_PUYO_COLORS
                            and consumed[1] in VALID_PUYO_COLORS):
                        self._pending_tsumo_2p.append(consumed)
                        # cycle 29: NEXT 変化 = 着地確定 signal
                        self._landing_pending_2p = (frame_idx, consumed)
                        # 着地色修正 案1 修正版 (2026-06-01): 同上
                        self._last_consumed_color_2p = consumed
                self._last_seen_next_2p = (top_v, bot_v)
        # game-event ベース連鎖終了チェック (C-1/C-2 plan, 2026-06-01)。
        # enable_game_event_chain_exit=True の場合のみ実行。
        # next_pair が確定した直後に「終了 game-event」を確認する。
        # 終了 game-event:
        #   ① 次ツモ変化: 連鎖した side の next_pair が chain 開始時から変化
        #   ② 連鎖側お邪魔降下: 連鎖した side の盤面に新規 COLOR_OJAMA 出現
        # どちらかを検知したら chain_ev を None に倒し _active_chain を解放する。
        if self._enable_game_event_chain_exit and chain_ev_1p is not None:
            # X1/X4 ガード: enable_chain_min_display=True 時は抑止条件を先に確認。
            _suppress_1p = (
                self._enable_chain_min_display
                and _should_suppress_game_event_exit(
                    time_sec=time_sec,
                    chain_entry_t=self._chain_entry_t_1p,
                    chain_count=chain_ev_1p.chain_count,
                    chain_min_display_sec=self.CHAIN_MIN_DISPLAY_SEC,
                    chain_game_event_min_count=self.CHAIN_GAME_EVENT_MIN_COUNT,
                )
            )
            if not _suppress_1p and _is_game_event_chain_exit(
                current_next=self._last_seen_next_1p,
                start_next=self._chain_start_next_1p,
            ):
                chain_ev_1p = None
                # 根治: GRAVITY_SETTLE 経由の final_board 反映用に退避してからクリア
                self._stash_and_clear_active_chain("1P")
        if self._enable_game_event_chain_exit and chain_ev_2p is not None:
            # X1/X4 ガード: enable_chain_min_display=True 時は抑止条件を先に確認。
            _suppress_2p = (
                self._enable_chain_min_display
                and _should_suppress_game_event_exit(
                    time_sec=time_sec,
                    chain_entry_t=self._chain_entry_t_2p,
                    chain_count=chain_ev_2p.chain_count,
                    chain_min_display_sec=self.CHAIN_MIN_DISPLAY_SEC,
                    chain_game_event_min_count=self.CHAIN_GAME_EVENT_MIN_COUNT,
                )
            )
            if not _suppress_2p and _is_game_event_chain_exit(
                current_next=self._last_seen_next_2p,
                start_next=self._chain_start_next_2p,
            ):
                chain_ev_2p = None
                # 根治: GRAVITY_SETTLE 経由の final_board 反映用に退避してからクリア
                self._stash_and_clear_active_chain("2P")

        # 案X*(B): NextSlide signal で CHAIN 即終了 (enable_chain_exit_next_signal=True 時)。
        # slide_motion=True が確認された side の active_chain を即クリアする。
        # 次ツモがスライドした = 連鎖は確実に終わった、という物理的証拠を活用する。
        # enable_game_event_chain_exit の状態・max_until の延長に関係なく即終了する。
        # warmup 連動: _enable_chain_exit_warmup は __init__ で True に設定済。
        if self._enable_chain_exit_next_signal and is_active:
            # slide_1p / slide_2p は 4c (l.2035-2036) で定義済みのローカル変数
            _slide_1p_now: bool = bool(slide_1p)
            _slide_2p_now: bool = bool(slide_2p)
            if _slide_1p_now and self._active_chain_1p is not None:
                # 1P 側: slide 検知 → CHAIN 即終了
                # 根治: GRAVITY_SETTLE 経由の final_board 反映用に退避してからクリア
                self._stash_and_clear_active_chain("1P")
                chain_ev_1p = None
            if _slide_2p_now and self._active_chain_2p is not None:
                # 2P 側: slide 検知 → CHAIN 即終了
                # 根治: GRAVITY_SETTLE 経由の final_board 反映用に退避してからクリア
                self._stash_and_clear_active_chain("2P")
                chain_ev_2p = None

        # 連鎖発火で constraint invalidate (= 連鎖中は puyo 消える + ojama 落下)
        # 注: score_d > 0 だけでは invalidate しない (通常 placement で score 増加するため).
        # chain_event があれば連鎖確定 → invalidate 両 side.
        if chain_ev_1p is not None:
            self._constraint_valid_1p = False
            # 1P 連鎖発火 → 2P 側に ojama 落下する可能性 → 2P も invalidate
            self._constraint_valid_2p = False
        if chain_ev_2p is not None:
            self._constraint_valid_2p = False
            self._constraint_valid_1p = False

        # 5. side ごとに state machine + 推論 + drift
        # tier1 warmup guard: _step_side 呼び出し前の state を保存
        # (= NON-STABLE → STABLE 遷移の検知に必要)。
        _pre_state_1p = self._sm_1p.context.state
        _pre_state_2p = self._sm_2p.context.state
        p1 = self._step_side(
            "1P", frame_idx, time_sec, is_active, cnn_1p,
            chain_ev_1p, score_d_2p_for_ojama=score_d_2p,
            sm=self._sm_1p, gen=self._gen_1p, drift=self._drift_1p,
            score_tracker=self._score_tracker_1p,
            next_pair=next_pair_1p,
            dnext_pair=dnext_pair_1p,
            slide_motion=slide_1p,
            frame_bgr=frame,  # cycle 71l β2' = HSV 距離による NEXT 色順序確定
            score_d_for_self=score_d_1p,  # cycle 71n 案 ε
            chain_max_hold_expired=self._chain_max_hold_expired_1p,  # 案P3
        )
        p2 = self._step_side(
            "2P", frame_idx, time_sec, is_active, cnn_2p,
            chain_ev_2p, score_d_2p_for_ojama=score_d_1p,
            sm=self._sm_2p, gen=self._gen_2p, drift=self._drift_2p,
            score_tracker=self._score_tracker_2p,
            next_pair=next_pair_2p,
            dnext_pair=dnext_pair_2p,
            slide_motion=slide_2p,
            frame_bgr=frame,  # cycle 71l β2'
            score_d_for_self=score_d_2p,  # cycle 71n 案 ε
            chain_max_hold_expired=self._chain_max_hold_expired_2p,  # 案P3
        )
        # tier1 warmup guard: _step_side 後にカウンタを更新。
        # _pre_state_* = _step_side 呼び出し前 (= 前フレームの state)。
        # p1.state / p2.state = _step_side が返した現フレームの state。
        if self._enable_tier1_warmup:
            self._tier1_warmup_remaining_1p = _update_tier1_warmup_counter(
                prev_state=_pre_state_1p,
                p_state=p1.state,
                remaining=self._tier1_warmup_remaining_1p,
            )
            self._tier1_warmup_remaining_2p = _update_tier1_warmup_counter(
                prev_state=_pre_state_2p,
                p_state=p2.state,
                remaining=self._tier1_warmup_remaining_2p,
            )
        # 経路 A': OJAMA 専用 tier1 warmup guard カウンタ更新。
        # OJAMA_FALL → STABLE 遷移時のみ OJAMA_TIER1_WARMUP_FRAMES をセット。
        if self._enable_ojama_tier1_warmup:
            self._ojama_tier1_warmup_remaining_1p = _update_ojama_tier1_warmup_counter(
                prev_state=_pre_state_1p,
                p_state=p1.state,
                remaining=self._ojama_tier1_warmup_remaining_1p,
            )
            self._ojama_tier1_warmup_remaining_2p = _update_ojama_tier1_warmup_counter(
                prev_state=_pre_state_2p,
                p_state=p2.state,
                remaining=self._ojama_tier1_warmup_remaining_2p,
            )

        # cycle 71d (案 D8): VideoChainTracker 次 frame 入力用に confirmed_board を保存.
        # None (= STABLE 以外) なら前回値を維持し、 直近の安定 board を提供し続ける.
        if p1.confirmed_board is not None:
            self._prev_confirmed_1p = p1.confirmed_board.copy()
        if p2.confirmed_board is not None:
            self._prev_confirmed_2p = p2.confirmed_board.copy()

        # Phase I.c: OnlineHsvCalibrator update (動画別 HSV 自動学習)
        # 1P/2P STABLE 中の信頼サンプルを蓄積、ready 後に ColorClassifier ranges
        # を 1 度だけ上書きする。連鎖中はスキップ (puyo HSV 変動)。
        # suppress guard: 外部 (visualize_recognition.py 等) が per-video JSON を
        # pre-inject 済の場合は OnlineHsv 段階的 inject ループを完全に skip する。
        # _online_hsv_injected = True は __init__ で False 初期化済のため、
        # 既存スクリプトのデフォルト挙動 (suppress なし) は変わらない。
        if self._online_hsv is not None and is_active and not self._online_hsv_injected:
            try:
                from src.hybrid_classifier import HybridClassifier
                hc = self._reader._classifier
                if isinstance(hc, HybridClassifier):
                    # 反復9 #40 triage (2026-07-23): is_chain は常に False の
                    # 死にフラグ (構造的に到達不能)。外側で既に
                    # `state == BoardState.STABLE` を要求しているため、この
                    # 内側の `state == BoardState.CHAIN` は state の相互排他性
                    # 上、絶対に True にならない。
                    # GS_EXEMPT: これは GRAVITY_SETTLE 導入で dead code 化した
                    # (根治で直した) バグとは別種で、GRAVITY_SETTLE を考慮に
                    # 加えても解決しない (STABLE と CHAIN/GRAVITY_SETTLE は
                    # どのみち同時に真になり得ない)。正しい修正には
                    # OnlineHsvCalibrator へ供給する対象 state 自体の再設計
                    # (例: CHAIN/GRAVITY_SETTLE 中の estimated_board も候補に
                    # 含め、is_chain で正しく除外する等) が必要だが、これは
                    # HSV 較正という認識コアの挙動を変える変更であり、
                    # Phase I (認識精度 99.99% 目標) の他フェーズ凍結方針・
                    # 専用の診断/viz 評価を要する対象。再発防止ガード
                    # (低リスク) の対象外として、意図的に現状維持する。
                    sides_to_update = []
                    if (self._sm_1p.context.state == BoardState.STABLE
                            and self._sm_1p.context.confirmed_board is not None):
                        sides_to_update.append((
                            DEFAULT_P1_REGION,
                            self._sm_1p.context.confirmed_board,
                            False,  # is_chain: 上記 GS_EXEMPT 参照、常に False
                        ))
                    if (self._sm_2p.context.state == BoardState.STABLE
                            and self._sm_2p.context.confirmed_board is not None):
                        sides_to_update.append((
                            DEFAULT_P2_REGION,
                            self._sm_2p.context.confirmed_board,
                            False,  # is_chain: 上記 GS_EXEMPT 参照、常に False
                        ))
                    if sides_to_update:
                        for region, board, is_chain in sides_to_update:
                            proba_g, hsv_g = hc.predict_proba_and_hsv_grid(
                                frame, region,
                            )
                            self._online_hsv.update(
                                frame, region, board, proba_g, hsv_g,
                                is_chain=is_chain,
                            )
                    # cycle 71v (2026-05-14): 段階的 inject に変更.
                    # 旧実装は全 6 色 ≥200 samples を待ってから 1 度だけ inject。
                    # v97 で 54s (= frame 3276) 経過まで kick in せず、 序盤の
                    # B/P 誤認の主因。 部分注入 + 色追加時の再注入で序盤から
                    # 動画別 ranges を適用する.
                    ranges = self._online_hsv.get_per_video_ranges()
                    if ranges and hasattr(
                        hc._hsv, "set_color_ranges_from_simple",
                    ):
                        injected_colors = self._online_hsv_injected_colors
                        new_colors = set(ranges.keys()) - injected_colors
                        if new_colors:
                            hc._hsv.set_color_ranges_from_simple(ranges)
                            self._online_hsv_injected_colors |= set(ranges.keys())
                            self._online_hsv_injected = True
                            print(
                                f"[pipeline] online_hsv injected: "
                                f"{len(ranges)} colors (new={sorted(new_colors)}) "
                                f"at frame={frame_idx}",
                            )
            except Exception:
                pass  # silent skip (本流に影響させない)

        # frame buffer 更新 (次回呼び出し時の prev_frame 用)。
        # 試合外なら None に戻して slide detector が走らないようにする。
        if is_active:
            self._prev_frame = frame
        else:
            self._prev_frame = None

        result = PipelineResult(
            frame_idx=frame_idx,
            time_sec=time_sec,
            is_match_active=is_active,
            p1=p1, p2=p2,
        )

        # Phase I: 擬似ラベル抽出 hook (error は silent skip)
        if self._enable_pseudo_label and self._pseudo_validators:
            for validator in self._pseudo_validators:
                try:
                    validator.update(frame_idx, time_sec, result, frame)
                except Exception:
                    # 擬似ラベル抽出失敗は本流に影響させない
                    pass

        return result

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _apply_next_count_constraint(
        self, board: "Board", tsumo_count: "Counter",
        side: str, frame_idx: int,
        *,
        protect_board: "Board | None" = None,
    ) -> "Board":
        """NEXT 累積色制約による confirmed_board 補正.

        試合開始〜最初の連鎖発火まで:
        field の puyo 数 = 累積 NEXT 数 (色別も一致)

        不一致時の対応:
        - 過剰色 (= field の方が多い): 該当色の中で最 row 大 (= 最近置かれた)
          cell を 不足色に置換
        - 不足色 (= field の方が少ない): スキップ (= 認識漏れ、 後で補完)

        Args:
            board: 補正対象 confirmed_board
            tsumo_count: 累積 NEXT 色 count (Counter)
            side: '1P' or '2P' (ログ用)
            frame_idx: 現 frame index (ログ用)
            protect_board: None でない場合、 board と protect_board のセル色が
                一致するセルを excess 置換候補から除外する (= CNN 高確信セル保護)。
                cnn_board を渡すと「CNN が認識した色と confirmed が一致するセル」
                = 高確信正解セルを保護できる。None 時は従来挙動 (全セル対象)。
                backwards compat: デフォルト None で旧挙動と完全同一。

        Returns:
            補正後 board (差分があれば新 instance、 なければ元 board)
        """
        from collections import Counter
        from src.board import (
            COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN, HIDDEN_ROWS,
        )

        # field の puyo 色 count (ojama / unknown / empty 除外)
        field_count: Counter = Counter()
        cell_by_color: dict[int, list[tuple[int, int]]] = {}
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                v = int(board.get(r, c))
                if v in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN):
                    continue
                field_count[v] += 1
                cell_by_color.setdefault(v, []).append((r, c))

        excess: dict[int, int] = {}
        deficit: dict[int, int] = {}
        for color, expected in tsumo_count.items():
            actual = field_count.get(color, 0)
            if actual > expected:
                excess[color] = actual - expected
            elif actual < expected:
                deficit[color] = expected - actual
        # field にある余分な色 (= tsumo_count に未出現の色) も excess 扱い
        for color, actual in field_count.items():
            if color not in tsumo_count and actual > 0:
                excess[color] = actual

        if not excess and not deficit:
            return board  # 整合済
        if frame_idx % 60 == 0:  # debug: 1 秒毎
            print(
                f"[constraint-mismatch] {side} frame={frame_idx} "
                f"field={dict(field_count)} tsumo={dict(tsumo_count)} "
                f"excess={dict(excess)} deficit={dict(deficit)}",
            )

        # 補正: excess color の最 row 小 (= 最上段、 後置きほど誤認しがち) cell を
        # deficit color に置換. 単純割当 (greedy).
        new_board = board.copy()
        deficit_list: list[int] = []
        for c, n in deficit.items():
            deficit_list.extend([c] * n)
        # 過剰色 cell を「上から (row 小)」 候補リスト化
        # 案1: protect_board が指定された場合、 board と protect_board のセル色が
        # 一致するセルは保護 = 置換候補から除外する (CNN 高確信正解セル保護)。
        # protect_board=None (default) の場合は全セル対象 (従来挙動)。
        # 注意: n_extra 個の excess があるとき、 保護されたものをスキップしつつ
        # 非保護セルから n_extra 個収集する (= 保護分を後続 cell で補う)。
        excess_cells: list[tuple[int, int, int]] = []
        for color, n_extra in excess.items():
            cells = sorted(
                cell_by_color.get(color, []),
                key=lambda rc: rc[0],  # row 昇順
            )
            # 保護されていない候補を n_extra 個収集 (全 cell を走査して保護除外)
            collected = 0
            for rc in cells:
                if collected >= n_extra:
                    break
                r_c, c_c = rc[0], rc[1]
                # 保護チェック: protect_board が指定されていて、 かつ
                # protect_board のそのセルが board の色 (= excess color) と一致するなら
                # 「CNN も同色と判断した = 高確信正解」 → 置換候補から除外
                if protect_board is not None:
                    protect_color = int(protect_board.get(r_c, c_c))
                    if protect_color == color:
                        continue  # 高確信正解セルは保護 → 次の候補へ
                excess_cells.append((r_c, c_c, color))
                collected += 1
        # ペア化して置換 (最小 len(excess_cells) と len(deficit_list))
        n_replace = min(len(excess_cells), len(deficit_list))
        for i in range(n_replace):
            r, c, _ = excess_cells[i]
            new_color = deficit_list[i]
            new_board.set(r, c, new_color)
        if n_replace > 0:
            # ログ (rare event なので OK)
            print(
                f"[constraint] {side} frame={frame_idx} "
                f"replaced {n_replace} cells "
                f"(excess={dict(excess)} deficit={dict(deficit)})",
            )

        # サイクル67/68: deficit-only force fill (= 認識漏れ補完)
        # excess なしで deficit のみ残った場合、 各列の「最高 filled cell の上」 から
        # 順に EMPTY cell を deficit 色で埋める. 列ごとに複数 cell 埋め可.
        remaining_deficit_list = deficit_list[n_replace:]
        if remaining_deficit_list:
            # 各列の「最高 filled cell の row」を求める
            top_filled: dict[int, int] = {}
            for c in range(BOARD_COLS):
                for r in range(BOARD_ROWS):
                    v = int(new_board.get(r, c))
                    if v not in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN):
                        top_filled[c] = r
                        break
            # 候補 cell: 最高 filled の上から順に EMPTY な cell を全部列挙
            # (= 認識漏れた縦連の puyo を一気に救済)
            # 隠し段 (row < HIDDEN_ROWS) は対象外
            candidates: list[tuple[int, int]] = []
            for c, top_r in top_filled.items():
                for cand_r in range(top_r - 1, HIDDEN_ROWS - 1, -1):
                    if int(new_board.get(cand_r, c)) == COLOR_EMPTY:
                        candidates.append((cand_r, c))
                    else:
                        break  # 既に何かある = それより上は調べない
            # row 大 (= 下) から埋める (= 物理的に下から積み上がる)
            candidates.sort(key=lambda rc: -rc[0])
            n_fill = min(len(candidates), len(remaining_deficit_list))
            for i in range(n_fill):
                r, c = candidates[i]
                new_board.set(r, c, remaining_deficit_list[i])
            if n_fill > 0:
                print(
                    f"[constraint-fill] {side} frame={frame_idx} "
                    f"filled {n_fill} EMPTY cells with deficit colors",
                )
        return new_board

    # 2026-05-11 サイクル71 Phase 1a: 旧 _compute_landing_inferred と
    # _inject_pseudo_chain_event を削除. placement_inferrer.infer_placement
    # + resolve_after_placement に置換済 (= 物理推論主軸化).

    def _start_landing_vote(
        self, side: str, frame_idx: int,
        prev_confirmed: Board, final_board: Board,
        next_colors: tuple[int, int] | None = None,  # cycle 71m β2''
        distrust_cells: set[tuple[int, int]] | None = None,
        time_sec: float | None = None,  # フレーム定数→時間定数化 Stage1 (2026-07-25)
    ) -> None:
        """cycle 71h: 着地時に vote 蓄積エントリを追加.

        prev_confirmed と final_board の差分 cells (= 着地で追加された cells) を
        抽出し、 期待色つきで vote_buffer を初期化する.

        cycle 71m (β2''): next_colors を保存し、 vote 期間中に HSV 距離で
        NEXT 色 2 種類のどちらかに分類する追加 vote も蓄積する.

        色フリッカ根因への防御的修正 案(iii) (2026-07-25): distrust_cells に
        座標が含まれるセルは、_update_landing_votes 側で NEXT 色 2 択バイアスを
        迂回し、生 CNN 多数決フォールバックに必ず落ちる (backwards compat:
        None なら空集合扱いで従来挙動と完全に同一)。

        Args:
            time_sec: フレーム定数→時間定数化 Stage1 (2026-07-25)。呼び出し元
                (_step_side) の time_sec。省略時 (backwards compat, 白箱テスト用)
                は `frame_idx / 60` で代替し、旧 frame ベース挙動を保つ。
        """
        cells_with_expected: list[tuple[int, int, int]] = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                pv = int(prev_confirmed.get(r, c))
                fv = int(final_board.get(r, c))
                if pv != fv and fv not in (COLOR_EMPTY, COLOR_UNKNOWN):
                    cells_with_expected.append((r, c, fv))
        if not cells_with_expected:
            return
        start_time = (
            time_sec if time_sec is not None else frame_idx / 60
        )
        entry: dict = {
            "start": frame_idx,
            "start_time": start_time,
            "cells": cells_with_expected,
            "votes": {
                (r, c): [] for (r, c, _) in cells_with_expected
            },
            # cycle 71m β2'': NEXT 色 2 種類への HSV 距離分類 vote.
            "next_color_votes": {
                (r, c): [] for (r, c, _) in cells_with_expected
            },
            "next_colors": next_colors,
            "side": side,
            # 案(iii): 疑わしいセル座標集合 (デフォルト空集合 = 従来挙動不変)。
            "distrust_cells": distrust_cells or set(),
        }
        if side == "1P":
            self._pending_landing_vote_1p.append(entry)
        else:
            self._pending_landing_vote_2p.append(entry)

    def _update_landing_votes(
        self, side: str, frame_idx: int,
        cnn_board: Board, confirmed_board: Board | None,
        frame_bgr: np.ndarray | None = None,  # cycle 71m β2''
        time_sec: float | None = None,  # フレーム定数→時間定数化 Stage1 (2026-07-25)
    ) -> Board | None:
        """cycle 71h: 着地後 vote 累積 + 完了時の confirmed 更新.

        各 pending entry について:
        - LANDING_VOTE_SEC 秒経過前: cnn_board の対象 cells を vote_buffer に追加
        - LANDING_VOTE_SEC 秒経過: 最頻値で confirmed_board の cell 色を更新

        cycle 71m (β2''): frame_bgr が渡されれば、 各 cell の HSV を NEXT 色 2 種類
        への距離で分類する vote も並行で蓄積. 蓄積終了時、 NEXT 色 votes の多数決を
        優先採用 (= CNN 完全誤認時の救済).

        Args:
            time_sec: フレーム定数→時間定数化 Stage1 (2026-07-25)。呼び出し元
                (_step_side) の time_sec。省略時 (backwards compat, 白箱テスト用)
                は `frame_idx / 60` で代替し、旧 frame ベース挙動を保つ。
                entry 側に "start_time" が無い場合も同様に entry["start"]/60 を
                fallback として使う。

        Returns:
            更新後の confirmed_board. None なら更新なし.
        """
        from collections import Counter
        from src.placement_inferrer import (
            COLOR_HSV_CENTERS, _hsv_distance,
            _extract_cell_patch_from_frame,
        )
        pending = (
            self._pending_landing_vote_1p if side == "1P"
            else self._pending_landing_vote_2p
        )
        if not pending:
            return None
        region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
        updated_board = confirmed_board.copy() if confirmed_board else None
        next_pending: list[dict] = []
        # フレーム定数→時間定数化 Stage1 (2026-07-25): cur_time / start_time は
        # ともに time_sec 基準。 呼び出し元省略時 (白箱テスト互換) は
        # frame_idx/60 を代替値として使い、旧 frame ベース挙動を維持する。
        cur_time = time_sec if time_sec is not None else frame_idx / 60
        for entry in pending:
            start_time = entry.get("start_time", entry["start"] / 60)
            elapsed_sec = cur_time - start_time
            if elapsed_sec < self.LANDING_VOTE_SEC:
                # cycle 26 (A2): 着地直後 CNN ぶれ大 → 蓄積 skip
                in_init_skip = elapsed_sec < self.LANDING_VOTE_INIT_SKIP_SEC
                # 蓄積期間中: cnn の対象 cells 色を追加 (init_skip 除く)
                if not in_init_skip:
                    for (r, c, _) in entry["cells"]:
                        v = int(cnn_board.get(r, c))
                        if v not in (COLOR_EMPTY, COLOR_UNKNOWN):
                            entry["votes"][(r, c)].append(v)
                # cycle 71m β2'': frame_bgr あれば HSV → NEXT 色分類 vote
                # cycle 26 (A2): NEXT 色 vote も init_skip 期間は除外
                if (
                    not in_init_skip
                    and frame_bgr is not None
                    and entry.get("next_colors") is not None
                ):
                    nc0, nc1 = entry["next_colors"]
                    if (
                        nc0 in COLOR_HSV_CENTERS
                        and nc1 in COLOR_HSV_CENTERS
                        and nc0 != nc1
                    ):
                        confirmed_set = entry.setdefault(
                            "confirmed_cells", set(),
                        )
                        for (r, c, _) in entry["cells"]:
                            # cycle 26 (A4): 既に早期確定済 cell は skip
                            if (r, c) in confirmed_set:
                                continue
                            patch = _extract_cell_patch_from_frame(
                                frame_bgr, region, r, c,
                            )
                            if patch is None or patch.size == 0:
                                continue
                            import cv2 as _cv2
                            hsv = _cv2.cvtColor(patch, _cv2.COLOR_BGR2HSV)
                            h = int(np.median(hsv[:, :, 0]))
                            s = int(np.median(hsv[:, :, 1]))
                            v_ = int(np.median(hsv[:, :, 2]))
                            d0 = _hsv_distance(
                                h, s, v_, COLOR_HSV_CENTERS[nc0],
                            )
                            d1 = _hsv_distance(
                                h, s, v_, COLOR_HSV_CENTERS[nc1],
                            )
                            entry["next_color_votes"][(r, c)].append(
                                nc0 if d0 < d1 else nc1,
                            )
                            # cycle 26 (A4): 早期確定経路。
                            # len>=5, ratio>=0.8 で即 updated_board に反映、
                            # confirmed_set に登録して以降の発火を抑止。
                            # 案(iii) (2026-07-25): distrust セルは NEXT 色バイアス
                            # による早期確定を迂回し、後段の生 CNN 多数決に委ねる。
                            nc_obs_now = entry["next_color_votes"][(r, c)]
                            cell_distrusted = (r, c) in entry.get(
                                "distrust_cells", set(),
                            )
                            if (
                                updated_board is not None
                                and not cell_distrusted
                                and len(nc_obs_now)
                                    >= self.LANDING_VOTE_NEXT_EARLY_COUNT
                            ):
                                nc_counter = Counter(nc_obs_now)
                                nc_winner, nc_count = (
                                    nc_counter.most_common(1)[0]
                                )
                                ratio_now = nc_count / len(nc_obs_now)
                                if (
                                    ratio_now
                                    >= self.LANDING_VOTE_NEXT_EARLY_RATIO
                                ):
                                    updated_board.set(r, c, nc_winner)
                                    confirmed_set.add((r, c))
                next_pending.append(entry)
            else:
                # 蓄積期間終了: NEXT 色 votes 優先、 fallback で従来 votes 最頻値
                if updated_board is None:
                    continue
                confirmed_set = entry.get("confirmed_cells", set())
                distrust_cells = entry.get("distrust_cells", set())
                for (r, c, expected) in entry["cells"]:
                    # cycle 26 (A4): 早期確定済 cell は再適用 skip (上書き禁止)
                    if (r, c) in confirmed_set:
                        continue
                    # 案(iii) (2026-07-25): distrust セルは NEXT 色 votes 優先ロジック
                    # を完全に迂回し、下記の生 CNN 多数決フォールバックに必ず落ちる。
                    # distrust_cells が空集合 (デフォルト/フラグ OFF) の場合は
                    # 従来ロジックと bit-identical。
                    if (r, c) not in distrust_cells:
                        # cycle 71m β2'': NEXT 色 votes が十分なら採用 (= 多数決優先)
                        # cycle 26 (A4): len>=3 のみ → ratio>=0.7 も必須化で誤分類抑制
                        nc_obs = entry.get(
                            "next_color_votes", {},
                        ).get((r, c), [])
                        if len(nc_obs) >= self.LANDING_VOTE_NEXT_MIN_COUNT:
                            nc_counter = Counter(nc_obs)
                            nc_winner, nc_count = nc_counter.most_common(1)[0]
                            nc_ratio = nc_count / len(nc_obs)
                            if nc_ratio >= self.LANDING_VOTE_NEXT_MIN_RATIO:
                                updated_board.set(r, c, nc_winner)
                                continue
                    # fallback: 既存 CNN 観測色の最頻値
                    obs = entry["votes"][(r, c)]
                    if not obs:
                        continue
                    counter = Counter(obs)
                    most_common, mc_count = counter.most_common(1)[0]
                    ratio = mc_count / len(obs)
                    # cycle 26 (A2): NEXT 色一致なら緩い ratio (0.3)、
                    # 不一致なら厳しい ratio (0.5) 必須で誤色採用を抑制。
                    nc_pair = entry.get("next_colors")
                    is_next_color = (
                        nc_pair is not None
                        and most_common in nc_pair
                    )
                    min_ratio = (
                        self.LANDING_VOTE_MIN_RATIO if is_next_color
                        else self.LANDING_VOTE_MISMATCH_MIN_RATIO
                    )
                    if ratio >= min_ratio:
                        updated_board.set(r, c, most_common)
                # entry は完了で破棄
        if side == "1P":
            self._pending_landing_vote_1p = next_pending
        else:
            self._pending_landing_vote_2p = next_pending
        return updated_board

    def _reset_gravity_settle_detectors(self) -> None:
        """1P/2P の BoardStateMachine から GravitySettleDetector を探しリセットする.

        feat/gravity-settle-2026-06-05: reset() が呼ばれた際に
        GravitySettleDetector の内部 settle state をクリアする。
        BoardStateMachine が detector list を外部公開しないため、
        _detectors 属性を直接参照する。
        """
        for sm in (self._sm_1p, self._sm_2p):
            for det in getattr(sm, "_detectors", []):
                if isinstance(det, GravitySettleDetector):
                    det.reset()

    def _trim_score_window(
        self,
        scores: list[int | None],
        times: list[float],
        cur_time: float,
    ) -> tuple[list[int | None], list[float]]:
        """`SCORE_MOVE_WINDOW_SEC` 秒より古い要素を先頭から削除する.

        フレーム定数→時間定数化 Stage1 (2026-07-25): 旧「直近 N 件保持」を
        「直近 SCORE_MOVE_WINDOW_SEC 秒保持」に変換するヘルパー。
        60fps 動画では 1 要素 ≒ 1/60 秒のため件数ベースの窓と一致し
        bit-identical。30fps 動画では実秒基準の窓になる。

        Args:
            scores: score 履歴 (時系列順、times と同じ長さ)。
            times: 各 score 観測時の time_sec (時系列順)。
            cur_time: 現フレームの time_sec。

        Returns:
            トリム後の (scores, times) タプル。
        """
        cutoff = cur_time - self.SCORE_MOVE_WINDOW_SEC
        drop = 0
        for t in times:
            if t > cutoff:
                break
            drop += 1
        if drop == 0:
            return scores, times
        return scores[drop:], times[drop:]

    def _is_score_actively_moving(
        cls, recent_scores: list[int | None],
    ) -> bool:
        """直近 N frame の score 履歴が SCORE_MOVE_MIN_DELTA 以上動いているか.

        cycle 71f (提案 A): 試合 2 開始直後の演出で MatchEnd lockdown が誤発火
        する場合でも、 score が継続的に増加していれば試合中と判定する.

        Args:
            recent_scores: 直近 N frame の score (None は OCR 失敗).

        Returns:
            None でない最大値 - 最小値 >= SCORE_MOVE_MIN_DELTA なら True.
        """
        valid = [s for s in recent_scores if s is not None]
        if len(valid) < 2:
            return False
        return (max(valid) - min(valid)) >= cls.SCORE_MOVE_MIN_DELTA

    @staticmethod
    def _update_score_tracker(
        tracker: ScoreTracker | None, frame: np.ndarray,
    ) -> tuple[int, int | None]:
        """tracker があれば update。

        戻り値は (delta (>=0 のみ), 今回 frame の生 score OCR 値)。
        修正1 (2026-07-30): 生 score 値は機能D (_check_formula_detected) が
        同一 frame・同一 side のスコア再読み取り (score_ocr.read_side) を
        避けるためのキャッシュとして呼出元から渡される。
        戻り値の tuple 化は private staticmethod (呼出元はこのファイル内の
        2 箇所のみ) のため backwards compat 上の懸念はない。
        """
        if tracker is None:
            return 0, None
        d = tracker.update(frame)
        delta = max(0, d.delta) if d.is_valid else 0
        return delta, d.cur_score

    def _apply_chain_score_early_fire(
        self,
        side: str,
        score_delta: int,
        time_sec: float,
        prev_confirmed: "Board | None",
    ) -> None:
        """機能B: score 急増を検知して即 CHAIN 突入シグナルを設定する。

        VideoChainTracker の puyo 減少検知より先に CHAIN state に入るための
        早期発火経路。 score_delta が CHAIN_SCORE_EARLY_FIRE_DELTA 未満または
        既に _active_chain_* が有効な場合はスキップ (既存経路優先)。

        Args:
            side: "1P" or "2P"
            score_delta: 自 side の今フレーム score 増分。
            time_sec: 現フレームの時刻。
            prev_confirmed: 連鎖前確定盤面 (before_board 用)。
        """
        if score_delta < CHAIN_SCORE_EARLY_FIRE_DELTA:
            return
        # 既に active_chain が有効 → 従来経路で管理中、早期発火は不要
        if side == "1P":
            if self._active_chain_1p is not None:
                return
        else:
            if self._active_chain_2p is not None:
                return
        # prev_confirmed が空の場合は before_board を空 Board() で代替
        before = prev_confirmed.copy() if prev_confirmed is not None else Board()
        # 疑似 ChainEvent を生成 (chain_count=1 の最小ガード)
        pseudo = ChainEvent(
            trigger_sec=time_sec,
            end_sec=(
                time_sec + self._chain_hold_base_sec + self._chain_hold_per_step_sec
            ),
            before_board=before,
            chain_count=1,
            total_erased=0, total_score=score_delta, base_score=score_delta,
            all_clear_bonus_applied=0,
            ojama_sent=0, leftover_score=0,
            is_all_clear=False,
        )
        chain_until = (
            time_sec + self._chain_hold_base_sec + self._chain_hold_per_step_sec
        )
        # 反復5 修正2 (2026-07-23): 疑似連鎖経路 (機能B 早期発火) も物理推論
        # スルーの対象にする (estimated_board 補完率向上)。起点盤面
        # (before) は prev_confirmed 由来で cold_start 等では信頼度が
        # 低い場合があるが、それは Step3(a)(b)(c) の答え合わせで拾う。
        self._start_chain_estimate(side, pseudo)
        if side == "1P":
            self._active_chain_1p = pseudo
            self._chain_until_1p = chain_until
            if self._enable_game_event_chain_exit:
                self._chain_event_max_until_1p = (
                    time_sec + self._chain_max_hold_sec
                )
                self._chain_start_next_1p = self._last_seen_next_1p
            self._chain_entry_t_1p = time_sec
        else:
            self._active_chain_2p = pseudo
            self._chain_until_2p = chain_until
            if self._enable_game_event_chain_exit:
                self._chain_event_max_until_2p = (
                    time_sec + self._chain_max_hold_sec
                )
                self._chain_start_next_2p = self._last_seen_next_2p
            self._chain_entry_t_2p = time_sec

    @staticmethod
    def _check_formula_detected(
        frame: "np.ndarray",
        score_ocr: "ScoreOcr | None",
        side: str,
        last_score: "int | None",
        cached_score_val: "int | None | _ScoreValNotCached" = (
            _SCORE_VAL_NOT_CACHED
        ),
    ) -> bool:
        """機能D: 掛け算式表示を stateless に判定する。

        AND 条件:
          1. score_ocr が None でなく OCR 結果が None (NCC conf 低下で読めない)
          2. score ROI の ink_ratio > CHAIN_FORMULA_INK_RATIO_MIN (黒 ROI 除外)
          3. last_score > 0 (試合進行中の担保)

        Args:
            frame: 1920x1080 BGR フレーム。
            score_ocr: ScoreOcr インスタンス。None なら常に False。
            side: "1P" or "2P"
            last_score: 直前に読めた score 値。None または 0 なら試合外とみなす。
            cached_score_val: 修正1 (2026-07-30)。呼出元 (ScoreTracker.update
                経由の _update_score_tracker) が同一 frame・同一 side で既に
                score_ocr.read_side() を実行済の場合、その戻り値 (int | None)
                をここに渡すと本メソッド内部での再読み取りを省略する。
                既定値 _SCORE_VAL_NOT_CACHED (未指定) では従来通り
                score_ocr.read_side() をここで実行する
                (backwards compat、bit-identical)。

        Returns:
            True = 掛け算式検知条件成立。
        """
        from src.score_ocr import (
            _crop_score_roi, _ensure_1080p, compute_score_roi_ink_ratio,
        )
        if score_ocr is None:
            return False
        if last_score is None or last_score <= 0:
            return False
        f = _ensure_1080p(frame)
        if f is None:
            return False
        if cached_score_val is not _SCORE_VAL_NOT_CACHED:
            score_val = cached_score_val
        else:
            score_val, _conf = score_ocr.read_side(f, side)  # type: ignore[arg-type]
        if score_val is not None:
            return False  # OCR 成功 = 通常スコア表示
        roi = _crop_score_roi(f, side)  # type: ignore[arg-type]
        ir = compute_score_roi_ink_ratio(roi)
        return ir > CHAIN_FORMULA_INK_RATIO_MIN

    def _simulate_before_board(
        self, before_board: "Board",
    ) -> "ChainResult | None":
        """起点盤面を ChainSimulator で検証する (修正D, 2026-07-24 追加)。

        _start_chain_estimate と機能D 早期発火ゲート (_resolve_formula_chain_count)
        で共用する simulate 呼び出しの共通ヘルパー。stateless
        (self._chain_sim は同一盤面の simulate を高速化するための lazy
        キャッシュ属性のみ保持、ChainSimulator 自体は副作用なし)。

        Args:
            before_board: 検証対象の起点盤面。

        Returns:
            ChainSimulator.simulate の結果。simulate 失敗時は None。
        """
        from src.chain import ChainSimulator
        if not hasattr(self, "_chain_sim"):
            self._chain_sim = ChainSimulator()  # type: ignore[attr-defined]
        try:
            return self._chain_sim.simulate(before_board)  # type: ignore[attr-defined]
        except Exception:
            return None

    def _resolve_formula_chain_count(
        self, before_board: "Board",
    ) -> "tuple[int | None, ChainResult | None]":
        """機能D 早期発火の chain_count を検証つきで解決する (修正D, 2026-07-24)。

        真因診断 (_diag_false_event_source_2026-07-24.py) で機能D 早期発火
        77件中35件=45.5%が「連鎖ゼロの起点盤面」からの疑似発火(偽イベント)
        と確定した対策。2026-07-24 user viz 承認により
        enable_chain_formula_simulate_verify=True が既定 (偽イベント率
        27.5%→0%)。True の場合は before_board を simulate し、
        chain_count==0 (連鎖が実在しない) なら (None, None) を返し、
        呼び出し元に疑似発火を抑制させる。chain_count>0 ならその実測値と
        ChainResult を返す (二重 simulate 回避のため _start_chain_estimate
        に precomputed_result として渡す)。False を明示指定した場合のみ
        検証せず chain_count=1 固定を返す (旧挙動, bit-identical)。

        Args:
            before_board: 早期発火の起点とする確定盤面。

        Returns:
            (chain_count, verified_result) のタプル。chain_count が None
            の場合は疑似発火を抑制すべきことを示す。
        """
        if not self._enable_chain_formula_simulate_verify:
            return 1, None
        verified = self._simulate_before_board(before_board)
        if verified is None or verified.chain_count <= 0:
            return None, None
        return verified.chain_count, verified

    def _apply_chain_formula_early_fire(
        self,
        side: str,
        time_sec: float,
        prev_confirmed: "Board | None",
    ) -> None:
        """機能D: 掛け算式検知で即 CHAIN 突入シグナルを設定する。

        _apply_chain_score_early_fire と同パターン。
        既に _active_chain_* が有効な場合はスキップ (既存経路優先)。

        修正D (2026-07-24): enable_chain_formula_simulate_verify=True (既定,
        2026-07-24 user viz 承認) の場合、起点盤面 (before_board) を
        ChainSimulator で事前検証し、連鎖が実在しない起点盤面での疑似発火を
        抑制する (偽イベント対策)。False を明示指定した場合のみ従来通り
        検証なしで chain_count=1 固定発火 (旧挙動, bit-identical)。

        Args:
            side: "1P" or "2P"
            time_sec: 現フレームの時刻。
            prev_confirmed: 連鎖前確定盤面 (before_board 用)。
        """
        if side == "1P":
            if self._active_chain_1p is not None:
                return
        else:
            if self._active_chain_2p is not None:
                return
        before = prev_confirmed.copy() if prev_confirmed is not None else Board()
        chain_count, verified = self._resolve_formula_chain_count(before)
        if chain_count is None:
            return  # 起点盤面に連鎖が実在しない (検証ON) → 疑似発火を抑制
        # 疑似 ChainEvent を生成 (score は不明なため 0)
        pseudo = ChainEvent(
            trigger_sec=time_sec,
            end_sec=(
                time_sec + self._chain_hold_base_sec
                + self._chain_hold_per_step_sec * chain_count
            ),
            before_board=before,
            chain_count=chain_count,
            total_erased=0, total_score=0, base_score=0,
            all_clear_bonus_applied=0,
            ojama_sent=0, leftover_score=0,
            is_all_clear=False,
        )
        chain_until = (
            time_sec + self._chain_hold_base_sec
            + self._chain_hold_per_step_sec * chain_count
        )
        # 反復5 修正2 (2026-07-23): 疑似連鎖経路 (機能D 掛け算式早期発火) も
        # 物理推論スルーの対象にする。修正D: 検証済みなら precomputed_result
        # を渡して _start_chain_estimate 内での二重 simulate を避ける。
        self._start_chain_estimate(side, pseudo, precomputed_result=verified)
        if side == "1P":
            self._active_chain_1p = pseudo
            self._chain_until_1p = chain_until
            if self._enable_game_event_chain_exit:
                self._chain_event_max_until_1p = time_sec + self._chain_max_hold_sec
                self._chain_start_next_1p = self._last_seen_next_1p
            self._chain_entry_t_1p = time_sec
        else:
            self._active_chain_2p = pseudo
            self._chain_until_2p = chain_until
            if self._enable_game_event_chain_exit:
                self._chain_event_max_until_2p = time_sec + self._chain_max_hold_sec
                self._chain_start_next_2p = self._last_seen_next_2p
            self._chain_entry_t_2p = time_sec

    def _start_chain_estimate(
        self,
        side: str,
        ev: ChainEvent,
        precomputed_result: "ChainResult | None" = None,
    ) -> None:
        """Step2 (2026-07-23): 連鎖検出時に物理推論スルーを開始する。

        ev.before_board (= 起点盤面、Step1 診断で 85.7% 有効と確認済み) から
        ChainSimulator で連鎖を 1 度だけシミュレートし、再生用の ChainResult
        を保持する。Step3(a) 答え合わせ: score 由来 chain_count (ev から算出)
        と物理予測 chain_count (before_board を simulate した実測値) が
        一致しなければ低信頼度フラグを立てる (= 起点盤面自体が誤認の疑い)。

        Args:
            side: "1P" または "2P"。
            ev: 新規検出された ChainEvent。
            precomputed_result: 呼び出し元で既に ev.before_board を simulate
                済みの結果があれば渡す (修正D, 2026-07-24 追加、二重 simulate
                回避)。None (既定) の場合は従来通りここで simulate する
                (backwards compat, bit-identical)。
        """
        if precomputed_result is not None:
            cr = precomputed_result
        else:
            cr = self._simulate_before_board(ev.before_board)
        result = cr if (cr is not None and cr.chain_count > 0) else None
        low_confidence = (
            result is not None and result.chain_count != ev.chain_count
        )
        if side == "1P":
            self._chain_estimate_result_1p = result
            self._chain_estimate_trigger_1p = ev.trigger_sec
            self._chain_estimate_end_1p = ev.end_sec
            self._chain_estimate_low_confidence_1p = low_confidence
            # 案1: cold start (= この CHAIN 継続区間でまだ一度も推定盤面を
            # 保持していない) の場合のみ起点盤面で seed する。既に途中まで
            # 進行した推定盤面 (last_board) があるなら、それより情報の少ない
            # before_board で上書きしない (= より進んだ推定を優先温存)。
            if self._chain_estimate_last_board_1p is None:
                self._chain_estimate_last_board_1p = ev.before_board.copy()
        else:
            self._chain_estimate_result_2p = result
            self._chain_estimate_trigger_2p = ev.trigger_sec
            self._chain_estimate_end_2p = ev.end_sec
            self._chain_estimate_low_confidence_2p = low_confidence
            if self._chain_estimate_last_board_2p is None:
                self._chain_estimate_last_board_2p = ev.before_board.copy()

    def _stash_and_clear_active_chain(self, side: str) -> None:
        """active_chain_* を None にする前に退避してからクリアする。

        根治 (2026-07-23): GRAVITY_SETTLE 経由の STABLE 復帰でも連鎖後
        final_board 反映 (Phase C-6 の C) を機能させるための共通ヘルパー。
        active_chain が None クリアされる全箇所 (timing hold 超過 /
        game-event 次ツモ変化 exit / NextSlide 即終了) から呼び出す。

        Args:
            side: "1P" または "2P"。
        """
        if side == "1P":
            if self._active_chain_1p is not None:
                self._last_chain_event_for_settle_1p = self._active_chain_1p
            self._active_chain_1p = None
        else:
            if self._active_chain_2p is not None:
                self._last_chain_event_for_settle_2p = self._active_chain_2p
            self._active_chain_2p = None

    def _classify_board_none_reason(
        self, side: str, is_active: bool,
        published_confirmed: "Board | None", state: BoardState,
    ) -> str | None:
        """confirmed_board=None の理由を分類する (反復4 診断計装、挙動非変更)。

        SideResult.board_none_reason に載せる分類ロジック本体。CHAIN 中の
        confirmed_board=None が「is_match_active→MENU 経路」由来か
        「別経路」由来かを切り分けるための計装 (真因調査用、修正ではない)。

        Args:
            side: "1P" または "2P"。
            is_active: このフレームの is_match_active
                (board_state_machine.py:480 の MENU 強制条件そのもの)。
            published_confirmed: このフレームで SideResult に載る確定盤面。
            state: このフレームの BoardState (ctx.state)。

        Returns:
            None (confirmed_board が非 None) / "cold_start" / "menu_reset" /
            "chain_hold_none" / "other"。
        """
        ever_had = (
            self._ever_had_confirmed_1p if side == "1P"
            else self._ever_had_confirmed_2p
        )
        if not is_active:
            # board_state_machine.py:480-488 の MENU 強制が今フレーム発生。
            if side == "1P":
                self._pending_menu_reset_1p = True
            else:
                self._pending_menu_reset_2p = True
        pending_menu_reset = (
            self._pending_menu_reset_1p if side == "1P"
            else self._pending_menu_reset_2p
        )
        if published_confirmed is not None:
            if side == "1P":
                self._ever_had_confirmed_1p = True
                self._pending_menu_reset_1p = False
            else:
                self._ever_had_confirmed_2p = True
                self._pending_menu_reset_2p = False
            return None
        if not ever_had:
            return "cold_start"
        if pending_menu_reset:
            return "menu_reset"
        if state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE):
            return "chain_hold_none"
        return "other"

    def _compute_chain_estimate(
        self, side: str, state: BoardState, time_sec: float,
    ) -> "tuple[Board | None, str]":
        """Step2 (2026-07-23): CHAIN/GRAVITY_SETTLE 中の物理推定盤面を返す。

        confirmed_board 自体は変更しない (標準 eval 経路への影響ゼロ)。
        estimated_board / board_provenance の算出専用ヘルパー。
        案1 (2026-07-23): フレッシュな推定が計算できないフレーム
        (起点盤面 simulate が chain_count=0 等) は _stale_hold_fallback に
        委譲する (enable_chain_estimate_stale_hold=True の場合のみ)。

        Args:
            side: "1P" または "2P"。
            state: このフレームの BoardState (ctx.state)。
            time_sec: 現フレームの時刻。

        Returns:
            (estimated_board, board_provenance)。 非該当時は (None, "observed")。
        """
        if state not in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE):
            # CHAIN/GRAVITY_SETTLE を抜けたら次回開始まで state を全てクリア
            # (案1: stale_hold 用 state も含む。次回 CHAIN 突入は cold start)。
            if side == "1P":
                self._chain_estimate_result_1p = None
                self._chain_estimate_last_board_1p = None
                self._chain_estimate_stale_since_1p = None
            else:
                self._chain_estimate_result_2p = None
                self._chain_estimate_last_board_2p = None
                self._chain_estimate_stale_since_2p = None
            return None, "observed"
        result = (
            self._chain_estimate_result_1p if side == "1P"
            else self._chain_estimate_result_2p
        )
        board: "Board | None" = None
        low_confidence = False
        if result is not None:
            trigger = (
                self._chain_estimate_trigger_1p if side == "1P"
                else self._chain_estimate_trigger_2p
            )
            end = (
                self._chain_estimate_end_1p if side == "1P"
                else self._chain_estimate_end_2p
            )
            low_confidence = (
                self._chain_estimate_low_confidence_1p if side == "1P"
                else self._chain_estimate_low_confidence_2p
            )
            board = _progressed_chain_board(result, trigger, end, time_sec)
        if board is not None:
            # フレッシュな推定成功: last_board 更新 + stale streak 解除。
            if side == "1P":
                self._chain_estimate_last_board_1p = board.copy()
                self._chain_estimate_stale_since_1p = None
            else:
                self._chain_estimate_last_board_2p = board.copy()
                self._chain_estimate_stale_since_2p = None
            provenance = (
                "chain_estimate_low_confidence" if low_confidence
                else "chain_estimate"
            )
            return board, provenance
        if not self._enable_chain_estimate_stale_hold:
            return None, "observed"
        return self._stale_hold_fallback(side, time_sec)

    def _stale_hold_fallback(
        self, side: str, time_sec: float,
    ) -> "tuple[Board | None, str]":
        """案1 (2026-07-23): estimated_board の stale_hold フォールバック本体。

        フレッシュな推定 (_compute_chain_estimate) が計算できないフレームで
        直前に成功した推定盤面 (last_board、無ければ CHAIN 突入時の起点盤面)
        を保持して返す。「推定が信頼できない保持中」であることが下流で
        区別できるよう board_provenance を明示する。
        安全弁: 連続 stale_hold が CHAIN_ESTIMATE_STALE_HOLD_MAX_SEC を
        超えたら None に戻す (古い盤面を無期限に貼り続ける事故を防ぐ)。

        Args:
            side: "1P" または "2P"。
            time_sec: 現フレームの時刻。

        Returns:
            (estimated_board, board_provenance)。 last_board が無い、または
            安全弁超過なら (None, "observed")。
        """
        last_board = (
            self._chain_estimate_last_board_1p if side == "1P"
            else self._chain_estimate_last_board_2p
        )
        if last_board is None:
            return None, "observed"
        stale_since = (
            self._chain_estimate_stale_since_1p if side == "1P"
            else self._chain_estimate_stale_since_2p
        )
        if stale_since is None:
            stale_since = time_sec
            if side == "1P":
                self._chain_estimate_stale_since_1p = stale_since
            else:
                self._chain_estimate_stale_since_2p = stale_since
        if time_sec - stale_since > self.CHAIN_ESTIMATE_STALE_HOLD_MAX_SEC:
            return None, "observed"
        return last_board.copy(), "chain_estimate_stale_hold"

    def _update_chain_estimate_verification(
        self, side: str, state: BoardState, cnn_board: Board,
    ) -> "tuple[str | None, Board | None]":
        """反復5 修正 Step3(b)(c): 連鎖後 final_board 適用の事後答え合わせ。

        単一フレームの生 CNN でなく、直近 CHAIN_VERIFY_FRAMES 分の STABLE
        cnn_board の多数決盤面と物理予測 (final_board) を照合する
        (GRAVITY_SETTLE 直後の残光ノイズに強くするため)。適用は既に完了
        済 (Phase C-6 の C で無条件適用) のため、ここでは止めずに
        不一致時のみ多数決盤面で confirmed_board を補正する
        (呼出元が返り値の補正盤面を ctx.confirmed_board に反映する)。

        Args:
            side: "1P" または "2P"。
            state: このフレームの BoardState (ctx.state)。
            cnn_board: この frame の生 CNN 観測。

        Returns:
            (answer_check_result, correction_board) のタプル。
            answer_check_result: None (検証対象外/進行中) /
                "verified_match" / "verified_mismatch_corrected"。
            correction_board: "verified_mismatch_corrected" のときのみ
                非 None (呼出元が ctx.confirmed_board に適用する多数決盤面)。
        """
        pending = (
            self._chain_verify_pending_1p if side == "1P"
            else self._chain_verify_pending_2p
        )
        if pending is None or state != BoardState.STABLE:
            return None, None
        pending["cnn_history"].append(cnn_board.copy())
        if len(pending["cnn_history"]) < self.CHAIN_VERIFY_FRAMES:
            return None, None
        from src.board_state_machine import _vote_majority_board
        history = pending["cnn_history"]
        min_votes = max(1, len(history) // 2 + 1)  # 単純多数決
        majority = _vote_majority_board(history, min_votes=min_votes)
        diff = DriftDetector._count_mismatch(pending["expected"], majority)
        if side == "1P":
            self._chain_verify_pending_1p = None
        else:
            self._chain_verify_pending_2p = None
        if diff <= self.CHAIN_VERIFY_MISMATCH_CELLS:
            return "verified_match", None
        # 不一致: 起点誤認の疑いが事後に判明 → 多数決盤面で補正する。
        from src.board_state_machine import _apply_gravity_filter
        _apply_gravity_filter(majority)
        return "verified_mismatch_corrected", majority

    def _check_baseline_broken_reset(
        self,
        side: str,
        frame_idx: int,
        time_sec: float,
        is_active: bool,
        ctx: "StateContext",
        cnn_board: Board,
        prev_state: BoardState,
        sm: BoardStateMachine,
        drift: DriftDetector,
        gen: InferenceBoardGenerator,
    ) -> None:
        """cycle 31 baseline 整合性 check + 自己修復 (baseline_broken reset)。

        制御フラグ化 (2026-07-25, A/B 計測用):
        enable_baseline_broken_reset=False で機能全体をスキップする
        (default True = 従来挙動完全維持、backwards compat)。
        enable_baseline_broken_grace=True の場合、STABLE 突入から
        BASELINE_BROKEN_STABLE_GRACE_SEC 秒間はカウンタ加算 (自己修復 reset
        の判定材料) を抑制する (default False = 従来挙動、猶予なし)。
        """
        if not self._enable_baseline_broken_reset:
            return
        if not (
            ctx.state == BoardState.STABLE
            and ctx.confirmed_board is not None
            and is_active
        ):
            return
        stable_since_attr = (
            "_stable_entered_time_1p" if side == "1P"
            else "_stable_entered_time_2p"
        )
        if prev_state != BoardState.STABLE:
            setattr(self, stable_since_attr, time_sec)
        if self._enable_baseline_broken_grace:
            stable_since = getattr(self, stable_since_attr)
            if (
                stable_since >= 0
                and (time_sec - stable_since) < self.BASELINE_BROKEN_STABLE_GRACE_SEC
            ):
                grace_attr = (
                    "_baseline_broken_grace_suppressed_1p" if side == "1P"
                    else "_baseline_broken_grace_suppressed_2p"
                )
                setattr(self, grace_attr, getattr(self, grace_attr) + 1)
                return
        self._apply_baseline_broken_counter(
            side, frame_idx, ctx.confirmed_board, cnn_board, sm, drift, gen,
        )

    def _apply_baseline_broken_counter(
        self,
        side: str,
        frame_idx: int,
        confirmed_board: Board,
        cnn_board: Board,
        sm: BoardStateMachine,
        drift: DriftDetector,
        gen: InferenceBoardGenerator,
    ) -> None:
        """baseline/CNN puyo数差の連続フレーム数を数え、閾値到達で自己修復 reset する。

        v97 53 秒 TSUMO_FALL 詰まり問題への救済策 (cycle 31, 2026-05-18)。
        """
        baseline_count = confirmed_board.count_puyos()
        cur_count = cnn_board.count_puyos()
        diff = cur_count - baseline_count
        BASELINE_BROKEN_DIFF_THRESHOLD = 8
        BASELINE_BROKEN_CONSEC_FRAMES = 60  # 1 秒
        consec_attr = (
            "_baseline_broken_consec_1p" if side == "1P"
            else "_baseline_broken_consec_2p"
        )
        if abs(diff) <= BASELINE_BROKEN_DIFF_THRESHOLD:
            setattr(self, consec_attr, 0)
            return
        setattr(self, consec_attr, getattr(self, consec_attr) + 1)
        if getattr(self, consec_attr) < BASELINE_BROKEN_CONSEC_FRAMES:
            return
        print(
            f"[baseline-reset] {side} frame={frame_idx} "
            f"baseline_count={baseline_count} "
            f"cnn_count={cur_count} diff={diff} "
            f"reset_after={getattr(self, consec_attr)} frames",
        )
        # 実発火回数カウンタ (A/B 効果測定用、2026-07-25)。
        count_attr = (
            "_baseline_broken_reset_count_1p" if side == "1P"
            else "_baseline_broken_reset_count_2p"
        )
        setattr(self, count_attr, getattr(self, count_attr) + 1)
        sm.reset(keep_match_state=False)
        drift.reset()
        gen.reset()
        setattr(self, consec_attr, 0)
        self._reacquire_background_fingerprints(side)

    def _reacquire_background_fingerprints(self, side: str) -> None:
        """baseline_broken reset 後の bg_fp 再採取トリガー (image_reader 側)。

        試合 active を再起動し (_bg_fp_captured を False に戻す)、
        次フレーム以降で背景指紋 (bg_fp) を再採取させる。
        """
        if hasattr(self._reader, "set_background_fingerprints"):
            if side == "1P":
                self._reader.set_background_fingerprints(
                    None, getattr(self._reader, "_bg_fp_p2", None),
                )
            else:
                self._reader.set_background_fingerprints(
                    getattr(self._reader, "_bg_fp_p1", None), None,
                )
        # 試合 active 再起動: _bg_fp_captured フラグも reset
        if hasattr(self, "_bg_fp_captured"):
            self._bg_fp_captured = False
        # I1 対応 A: bg_fp 再採取中は pre_capture_mode を on に戻す
        if hasattr(self._reader, "set_pre_capture_mode"):
            self._reader.set_pre_capture_mode(True)

    def _step_side(
        self,
        side: str,
        frame_idx: int,
        time_sec: float,
        is_active: bool,
        cnn_board: Board,
        chain_event: ChainEvent | None,
        *,
        score_d_2p_for_ojama: int,
        sm: BoardStateMachine,
        gen: InferenceBoardGenerator,
        drift: DriftDetector,
        score_tracker: ScoreTracker | None,
        next_pair: tuple[int, int] | None = None,
        dnext_pair: tuple[int, int] | None = None,
        slide_motion: bool = False,
        frame_bgr: np.ndarray | None = None,  # cycle 71l β2'
        score_d_for_self: int = 0,  # cycle 71n 案 ε
        chain_max_hold_expired: bool = False,  # 案P3: MAX_HOLD 超過フラグ
    ) -> SideResult:
        """1 side 分の pipeline 処理."""
        # 着地色診断フィールド: 非着地フレームは None のまま戻り値に載る。
        # TSUMO_FALL→STABLE 遷移時のみ上書きされる。
        _landing_diag: dict | None = None
        # Phase I R-1: 自己整合性チェック (TSUMO_FALL 中のみ意味あり)。
        # baseline (= 直前 STABLE 確定盤面) と current cnn_board の
        # 色 count delta が、落下中ツモ (next_queue[-2]) と整合しているか。
        # 整合 = 期待通り puyo が増えている → 着地早期復帰の hint。
        placement_validated = False
        if (
            sm.context.state == BoardState.TSUMO_FALL
            and sm.context.confirmed_board is not None
            and len(sm.context.next_queue) >= 2
        ):
            try:
                falling = sm.context.next_queue[-2]
                pv = validate_tsumo_placement(
                    sm.context.confirmed_board, cnn_board, falling,
                )
                placement_validated = bool(pv.consistent)
            except Exception:
                placement_validated = False

        # OJAMA は「相手の score 増分」で発火するので、自 side の signals に
        # 相手 score_delta を渡す。
        # 演出中は CNN 信用せず STABLE hold (= EFFECT state)。
        # 各 detector の lockdown / visible を OR で集約 (2026-05-10)。
        # 2026-05-11: telop は image_reader でセル単位 UNKNOWN マスクするため
        # state=EFFECT は不要 (= 静的バナーで盤面全体を凍結すると STABLE 中の
        # 非被覆セル更新も止まり、 試合開始から数十秒間「ずっと EFFECT」 になる
        # 問題が発生)。 match_end のみ effect_vis に残す.
        effect_vis = bool(
            self._match_end_detector is not None
            and self._match_end_detector.is_locked(time_sec)
        )
        # cycle 71g (提案 A 補強): score が継続的に動いていれば effect_visible を
        # 解除. 試合 1 終了演出の lockdown が試合 2 期間中まで残るのを防ぐ
        # (= v50 48-75s ですべて effect 状態になる問題の対策).
        if effect_vis and (
            self._is_score_actively_moving(self._recent_scores_1p)
            or self._is_score_actively_moving(self._recent_scores_2p)
        ):
            effect_vis = False
        # cycle 71v 汎用化: 試合開始直後 window 判定 (初回 STABLE 確定で
        # 空フィールド強制するため state_machine に伝搬)。
        # フレーム定数→時間定数化 Stage1 (2026-07-25): 旧 `frame_idx -
        # self._match_active_started_frame` (frame 差分) を time_sec 差分に
        # 置換。60fps では bit-identical、30fps では実秒基準になる。
        match_just_started = (
            is_active
            and self._match_active_started_time >= 0
            and (time_sec - self._match_active_started_time)
            < self.MATCH_JUST_STARTED_WINDOW_SEC
        )
        # 設計C 事後復旧ゲート用 HSV-only 盤面を取得する。
        # フラグ OFF または frame_bgr なし の場合は None (安全弁A により発火しない)。
        # 色→空 HSV 照合ガード (2026-07-30): 復旧ゲートは STABLE 定常でしか
        # HSV を要求しないが、案A のガードは NON-STABLE→STABLE 遷移フレーム
        # (signals 構築時点の state はまだ NON-STABLE) の merge で HSV を必要と
        # する。そのため enable_puyo_to_empty_hsv_guard 有効時は state を問わず
        # HSV を供給する (実測: 遷移フレームで hsv_board=None のためガード不発
        # だった)。フラグ OFF (既定) では OR 右項が False となり従来の
        # 「復旧ゲート ON かつ STABLE」条件と bit-identical。
        _hsv_board_for_signals: "Board | None" = None
        _need_hsv = frame_bgr is not None and (
            (
                self._enable_stable_recovery_gate
                and sm.context.state == BoardState.STABLE
            )
            or self._enable_puyo_to_empty_hsv_guard
        )
        if _need_hsv:
            region_for_hsv = (
                DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
            )
            try:
                _hsv_board_for_signals = self._reader.read_board_hsv_only(
                    frame_bgr, region_for_hsv,
                )
            except Exception:
                _hsv_board_for_signals = None
        # フェーズ A 精緻化 (2026-06-02): OjamaVisualDetector の内部 state 更新前に
        # 一次判定 (ROI お邪魔 count) を計算し、 signals に ojama_top_positive をセット。
        # OjamaVisualDetector はこれを受けて内部カウンタを進める。
        _ojama_top_positive: bool = False
        if self._enable_ojama_visual_detection:
            from src.ojama_visual_detector import _count_top_ojama as _cnt_oj
            _ojama_top_positive = _cnt_oj(cnn_board, _hsv_board_for_signals) > 0
        signals = DetectorSignals(
            time_sec=time_sec,
            cnn_board=cnn_board,
            is_match_active=is_active,
            chain_event=chain_event,
            score_delta=score_d_2p_for_ojama,
            next_pair=next_pair,
            slide_motion=slide_motion,
            placement_validated=placement_validated,
            effect_visible=effect_vis,
            match_just_started=match_just_started,
            hsv_board=_hsv_board_for_signals,
            ojama_top_positive=_ojama_top_positive,
            chain_max_hold_expired=chain_max_hold_expired,  # 案P3
        )
        # 着地推論用: sm.update 前のスナップショット
        # TSUMO_FALL 中は confirmed_board が更新されないため、
        # prev_confirmed = TSUMO_FALL 開始前の盤面 = 真の baseline
        prev_state = sm.context.state
        prev_confirmed = (
            sm.context.confirmed_board.copy()
            if sm.context.confirmed_board is not None else None
        )
        prev_next_queue = list(sm.context.next_queue)

        ctx: StateContext = sm.update(frame_idx, signals)

        # 根治 (2026-07-23): GRAVITY_SETTLE 経由の STABLE 復帰では chain_event
        # 引数が既に None 化されているため、退避しておいた ChainEvent を
        # fallback として使う「実効 chain_event」をこの frame の先頭で 1 回だけ
        # 確定する。Phase C-6 の C (final_board 反映) と T2 (STABLE→STABLE
        # 誤色棄却の連鎖直後 skip 判定) の両方がこの同一値を参照することで、
        # T2 が Phase C-6 の C の直後に fresh な final_board を古い
        # prev_stable で即座に上書きしてしまう相互作用を防ぐ
        # (cycle 71 系との相性確認: architect 指摘事項)。
        # backward compat 注記: prev_state==CHAIN (= enable_gravity_settle_state
        # =False 時の従来経路) では _effective_chain_event は raw chain_event と
        # 完全に同値にする (= 退避 stash を一切参照しない)。これにより
        # GRAVITY_SETTLE 未経由の既存経路の挙動を 1 bit も変えない
        # (退避 stash はフラグに関係なく常時更新されるが、CHAIN 直行経路では
        # 一切参照されないため無害)。
        if prev_state == BoardState.GRAVITY_SETTLE:
            _effective_chain_event = (
                self._last_chain_event_for_settle_1p if side == "1P"
                else self._last_chain_event_for_settle_2p
            )
        else:
            _effective_chain_event = chain_event

        # W-α (Phase G C-1): TSUMO_FALL → STABLE 復帰時に隠し段推論結果の
        # ProbabilisticBoard を保持して下流に publish する。
        # NON-STABLE 中・初回 STABLE 等は None のままにし、下流で fallback。
        side_prob_board: ProbabilisticBoard | None = None

        # Option C-2/E-2: TSUMO_FALL → STABLE 復帰時に物理推論で confirmed 上書き
        # サイクル67: TSUMO_FALL→STABLE 着地確定で pending_tsumo を tsumo_count にコミット
        if (
            prev_state == BoardState.TSUMO_FALL
            and ctx.state == BoardState.STABLE
        ):
            pending = (
                self._pending_tsumo_1p if side == "1P"
                else self._pending_tsumo_2p
            )
            tsumo_count_target = (
                self._tsumo_count_1p if side == "1P"
                else self._tsumo_count_2p
            )
            if pending:
                committed = pending.popleft()
                tsumo_count_target[committed[0]] += 1
                tsumo_count_target[committed[1]] += 1
        # 2026-05-11 サイクル71 Phase 1a: 物理推論主軸化.
        # 旧 _compute_landing_inferred (= CNN 差分位置採用) を廃止し、
        # placement_inferrer.infer_placement (= 物理パターン全列挙 + NEXT 色固定
        # + CNN 一致度で候補絞り込み) に置換. 連鎖即時判定も resolve_after_placement
        # で実行し、 仮説 B (= 連鎖検出 1 秒遅延) を解消する.
        # フェーズ A 精緻化: ojama_infer_guard かつ ojama_tier1_warmup 中なら
        # infer_placement をスキップする。OJAMA_FALL 直後の warmup 期間に
        # ツモが存在しないのに幽霊配置が走るのを防ぐ。
        _ojama_warmup_remaining = (
            self._ojama_tier1_warmup_remaining_1p if side == "1P"
            else self._ojama_tier1_warmup_remaining_2p
        )
        _skip_infer_by_ojama_guard = (
            self._enable_ojama_infer_guard
            and _ojama_warmup_remaining > 0
        )
        if (
            prev_state == BoardState.TSUMO_FALL
            and ctx.state == BoardState.STABLE
            and prev_confirmed is not None
            and not _skip_infer_by_ojama_guard
        ):
            # 落下中ツモ色 = TSUMO 開始時の next (= 直前の next、現 next の 1 つ前)
            # 診断: 従来ロジック (prev_next_queue[-2]) と
            #        修正ロジック (_last_consumed_color) の両者を計算し比較する。
            # enable_landing_color_fix=True の場合は修正ロジックを優先する。
            #
            # 【案1 修正版 (2026-06-01) の根拠】
            # _landing_pending は NEXT変化フレームの grace処理 (landing_pending[0] == frame_idx)
            # でクリアされるため、実際の着地フレーム (TSUMO_FALL→STABLE) では常にNone。
            # _last_consumed_color はグレース処理から独立して保持し、
            # 着地フレームでも正しく参照できる変数として新設した。
            falling_pair: tuple[int, int] | None = None
            falling_pair_old: tuple[int, int] | None = None
            falling_pair_new: tuple[int, int] | None = None
            _diag_source: str = "none"
            # 従来ロジック: prev_next_queue[-2]
            if len(prev_next_queue) >= 2:
                falling_pair_old = prev_next_queue[-2]
            elif prev_next_queue:
                falling_pair_old = prev_next_queue[-1]
            # 修正ロジック: _last_consumed_color から消費済みツモ色を取得
            # (_landing_pending と異なりgraceクリアされないため着地フレームで有効)
            falling_pair_new = (
                self._last_consumed_color_1p if side == "1P"
                else self._last_consumed_color_2p
            )
            # フラグに応じて falling_pair を決定
            if self._enable_landing_color_fix and falling_pair_new is not None:
                falling_pair = falling_pair_new
                _diag_source = "last_consumed_color"
            elif falling_pair_old is not None:
                falling_pair = falling_pair_old
                _diag_source = (
                    "next_queue_2" if len(prev_next_queue) >= 2 else "next_queue_1"
                )
            # 診断フィールド: TSUMO_FALL→STABLE 着地フレームで両者の差を記録。
            # 冒頭で None 初期化済のため再アノテーションなし。
            _landing_diag = {
                "falling_pair_old": list(falling_pair_old) if falling_pair_old else None,
                "falling_pair_new": list(falling_pair_new) if falling_pair_new else None,
                "source": _diag_source,
            }
            # Phase 1a: 物理推論で着地後盤面を確定.
            # cycle 71b: chain_sim を渡して連鎖整合性で候補絞り込み (案 A).
            # 着地 frame では自 side の score_delta はまだ確定していない
            # (= 連鎖アニメは数 frame 後) ため 0 を渡し、 候補絞り込みは
            # 「連鎖なし候補優先」 になる. score_delta による事後修正は
            # 後段の修正トリガー (= 案 C) で実装予定.
            # cycle 71l (β2'): frame_bgr + region 渡しで HSV 距離による NEXT 色
            # 順序確定 (= 回転落下対応).
            region_for_side = (
                DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
            )
            # cycle 71v (2026-05-15): bg_fp を渡して背景一致 cells を物理推論段階で reject.
            bg_fp_for_side = getattr(
                self._reader,
                "_bg_fp_p1" if side == "1P" else "_bg_fp_p2",
                None,
            )
            # 案 Y-4: deferred_out リストを用意し、拮抗時に候補を受け取る。
            _deferred_buf: list = []
            inferred_landing = infer_placement(
                prev_confirmed, cnn_board, falling_pair,
                chain_sim=self._chain_sim,
                score_delta_observed=0,
                frame_bgr=frame_bgr,
                region=region_for_side,
                bg_fp=bg_fp_for_side,
                guard_empty_hallucination=self._enable_infer_empty_guard,
                enable_hsv_classify_fallback=self._enable_hsv_classify_fallback,
                enable_hsv_deferred_consensus=self._enable_hsv_deferred_consensus,
                deferred_out=_deferred_buf if self._enable_hsv_deferred_consensus else None,
            )
            if inferred_landing is not None:
                # 修正方針 甲: P2 設置推論の防御的 CNN 照合 (2026-07-25)。
                # 着地セルへ色を書く前に現フレーム CNN 観測と照合し、不一致なら
                # 書き込みを保留する (門番、案(iii) より先に適用)。フラグ OFF
                # (default) 時は inferred_landing を素通しし bit-identical。
                if (
                    self._enable_placement_cnn_veto
                    and prev_confirmed is not None
                ):
                    _before_veto = inferred_landing
                    inferred_landing = _apply_placement_cnn_veto(
                        inferred_landing, prev_confirmed, cnn_board,
                        mode=self._placement_cnn_veto_mode,
                    )
                    _held_n = sum(
                        1 for _r in range(BOARD_ROWS) for _c in range(BOARD_COLS)
                        if int(_before_veto.get(_r, _c))
                        != int(inferred_landing.get(_r, _c))
                    )
                    if side == "1P":
                        self._placement_cnn_veto_held_count_1p += _held_n
                    else:
                        self._placement_cnn_veto_held_count_2p += _held_n
                # 色フリッカ根因への防御的修正 案(iii) (2026-07-25):
                # 着地セル (= P2 設置推論の出力) のうち CNN 観測色が baseline と
                # 食い違う「疑わしいセル」をフラグし、P7 (_start_landing_vote /
                # _update_landing_votes) に伝播する。baseline (inferred_landing)
                # 自体は一切書き換えない (地雷再発防止)。
                # フラグ OFF (default) 時は常に空集合 = 下流は完全に bit-identical。
                distrust_cells_for_side: set[tuple[int, int]] = (
                    _flag_landing_distrust_cells(
                        inferred_landing, prev_confirmed, cnn_board,
                    )
                    if self._enable_placement_color_cnn_check
                    and prev_confirmed is not None
                    else set()
                )
                if side == "1P":
                    self._landing_distrust_1p = distrust_cells_for_side
                else:
                    self._landing_distrust_2p = distrust_cells_for_side
                # 真因 A 対処 (2026-06-01): 着地セル CNN==HSV 一致色で補正。
                # falling_pair タイミングずれで infer_placement が誤色を書いても
                # 2 つの独立認識器 (CNN/HSV) が一致した色があれば優先採用する。
                # フラグ OFF 時は完全にスキップし、挙動不変を保証する。
                if (
                    self._enable_landing_observed_color
                    and frame_bgr is not None
                    and falling_pair is not None
                    and prev_confirmed is not None
                ):
                    # 直前の惑星パターンを特定するため、着地 2 cell の差分から
                    # 着地 pattern を再現する (pattern は infer_placement 内で選択済み)。
                    # 最小コストで pattern を得るには diff_cells から再構築するより、
                    # prev_confirmed と inferred_landing の差分 2 cell を使う。
                    inferred_landing = _apply_landing_observed_color_correction(
                        inferred_landing,
                        prev_confirmed,
                        cnn_board,
                        self._reader,
                        frame_bgr,
                        region_for_side,
                    )
                # T5: NextDetector 統合 — 着地直後 confirmed の色が NEXT にない場合 alert。
                # next_pair (= 今消費された NEXT) が明示されていれば整合性チェック。
                # alert のみ (= 棄却はしない。 現時点は fail-silent 検知用)。
                if (
                    falling_pair is not None
                    and next_pair is not None
                ):
                    valid_colors = {
                        c for c in falling_pair
                        if c not in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA)
                    }
                    for r in range(BOARD_ROWS):
                        for c in range(BOARD_COLS):
                            before_v = int(prev_confirmed.get(r, c))
                            after_v = int(inferred_landing.get(r, c))
                            if (
                                before_v == COLOR_EMPTY
                                and after_v not in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA)
                                and valid_colors
                                and after_v not in valid_colors
                            ):
                                # NEXT 色以外が着地 → 認識誤り可能性 (alert のみ)
                                pass  # 将来: alert 記録 or 棄却
                # W-α: 隠し段 (row 0) の量子推論 (= 既存ロジック維持)
                if (
                    falling_pair is not None
                    and falling_pair[0] not in (
                        COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA,
                    )
                    and falling_pair[1] not in (
                        COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA,
                    )
                ):
                    try:
                        pboard, _ = infer_hidden_row(
                            prev_confirmed, inferred_landing, falling_pair,
                        )
                        for col in range(BOARD_COLS):
                            cell = pboard.cell(0, col)
                            color, p = cell.most_likely()
                            if p >= 0.95:
                                inferred_landing.set(0, col, color)
                            else:
                                inferred_landing.set(
                                    0, col, COLOR_UNKNOWN,
                                )
                        side_prob_board = pboard
                    except Exception:
                        side_prob_board = None
                # Phase 1a: 連鎖即時判定. 連鎖発生時は final_board に切替し、
                # 疑似 ChainEvent を生成して state machine を CHAIN に固定する
                # (= 連鎖アニメ中 CNN 誤更新からの保護).
                # cycle 71c: prev_confirmed を渡し、 認識欠落補填による大量 add で
                # 偽陽性連鎖が発火する A=hit (α) ケースを抑止.
                # cycle 71n (案 ε): score_delta_observed を渡し、 score 動いていない
                # 場合は連鎖発生偽陽性として final_board 採用 skip.
                final_board, chain_count = resolve_after_placement(
                    inferred_landing, self._chain_sim,
                    prev_confirmed=prev_confirmed,
                    score_delta_observed=score_d_for_self,
                )
                ctx.confirmed_board = final_board
                ctx.pending_board = final_board.copy()
                # cycle 48 (2026-05-20): 偽 chain ガード
                # placement_inferrer hallucination で「prev_confirmed → inferred_landing」 で
                # 異常に大量の cell が追加された場合 (= 通常 1 ツモ = 2 cell)、
                # chain 発火を ban。 6 cell 超 (= 3 ツモ分相当) は明らかに不審。
                # 強化アナリストの retrospective_chain_missing / chain_no_puyo_loss と
                # 同方針の事前ガード。
                if chain_count >= 1 and prev_confirmed is not None:
                    from src.board import BOARD_ROWS as _BR, BOARD_COLS as _BC
                    diff_cells = 0
                    for _r in range(_BR):
                        for _c in range(_BC):
                            pv = int(prev_confirmed.get(_r, _c))
                            iv = int(inferred_landing.get(_r, _c))
                            if (pv == COLOR_EMPTY or pv == COLOR_UNKNOWN) \
                               and iv not in (COLOR_EMPTY, COLOR_UNKNOWN):
                                diff_cells += 1
                    if diff_cells > 6:
                        # 大量 hallucination = 偽 chain として ban
                        chain_count = 0
                        ctx.confirmed_board = prev_confirmed.copy() \
                            if prev_confirmed is not None else final_board
                if chain_count >= 1:
                    # A0 (2026-07-24) バグ修正: 従来ここだけハードコード 0.3 で
                    # self._chain_hold_per_step_sec (config 可能な設定値) を
                    # 無視していた。既定値 0.3 なら数値上は不変だが、較正値を
                    # 渡した場合にこの経路 (cycle48 大量 hallucination ガード
                    # 通過済の着地直後即時連鎖) だけ較正が効かない不整合が
                    # あったため、他経路と同じ式に統一する。
                    pseudo = ChainEvent(
                        trigger_sec=time_sec,
                        end_sec=(
                            time_sec
                            + self._chain_hold_base_sec
                            + self._chain_hold_per_step_sec * chain_count
                        ),
                        before_board=inferred_landing,
                        chain_count=chain_count,
                        total_erased=0, total_score=0, base_score=0,
                        all_clear_bonus_applied=0,
                        ojama_sent=0, leftover_score=0,
                        is_all_clear=False,
                    )
                    chain_until = (
                        time_sec
                        + self._chain_hold_base_sec
                        + self._chain_hold_per_step_sec * chain_count
                    )
                    # 反復5 修正2 (2026-07-23): 疑似連鎖経路 (着地直後の即時連鎖
                    # 判定、cycle48 大量 hallucination ガード通過済) も
                    # 物理推論スルーの対象にする。
                    self._start_chain_estimate(side, pseudo)
                    if side == "1P":
                        self._active_chain_1p = pseudo
                        self._chain_until_1p = chain_until
                    else:
                        self._active_chain_2p = pseudo
                        self._chain_until_2p = chain_until
                # cycle 29 (2026-05-18): grace + landing_vote の起動は NEXT
                # 移動検知ベース (= _step_side 末尾の landing_pending 経路) に
                # 統一。 ここでは final_board の確定だけ行う (= ctx.confirmed_board
                # は既に上書き済)。 NEXT 経路は次の frame で発火する。

            # 案 Y-4: deferred_buf に候補が入っていれば deferred state を開始する。
            # inferred_landing は board_std (安全 fallback) として既に ctx に書込済。
            if (
                self._enable_hsv_deferred_consensus
                and _deferred_buf
                and inferred_landing is not None
            ):
                board_std_d, board_rev_d, base_cells_d = _deferred_buf[0]
                _deferred_state: dict = {
                    "board_std": board_std_d,
                    "board_rev": board_rev_d,
                    "base_cells": base_cells_d,
                    "votes_std": 0,
                    "votes_rev": 0,
                    "frames_left": DEFERRED_MAX_FRAMES,
                }
                if side == "1P":
                    self._deferred_landing_1p = _deferred_state
                else:
                    self._deferred_landing_2p = _deferred_state
            # 着地色修正 案1 修正版 (2026-06-01):
            # infer_placement 完了後 (success/failどちらでも) _last_consumed_color をクリア。
            # 1ツモ分のみ使用し次ツモと混同しないようにする。
            if side == "1P":
                self._last_consumed_color_1p = None
            else:
                self._last_consumed_color_2p = None

        # 2026-05-11 サイクル65: NEXT 履歴ベース placement 検証 (補強)
        # 「NEXT のぷよは必ずフィールドに置かれる」 前提を継続的に活用する.
        # STABLE 内で confirmed_board に 2 cell 新規追加 (= placement) が発生し、
        # かつ TSUMO_FALL 経由していない場合でも next_pair で色補正.
        # 適用条件:
        #   1) prev_state in {OJAMA_FALL, MENU} → STABLE (= TSUMO_FALL skip)
        #   2) STABLE → STABLE で confirmed_board が 2 cell 増えた
        #   3) 増えた 2 cell が隣接 (= 縦置き or 横置き)
        # 既存 TSUMO_FALL ハンドラと重複しないよう除外.
        if (
            prev_state != BoardState.TSUMO_FALL
            and prev_state != BoardState.CHAIN
            # フェーズ A 精緻化: OJAMA_FALL → STABLE 復帰時は infer 発火禁止。
            # お邪魔降下中の幽霊配置を根絶するための常時ガード。フラグ非依存。
            and prev_state != BoardState.OJAMA_FALL
            and ctx.state == BoardState.STABLE
            and prev_confirmed is not None
            and ctx.confirmed_board is not None
        ):
            # diff 2 cell が adjacent placement か確認
            diffs: list[tuple[int, int, int]] = []
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    base_v = int(prev_confirmed.get(r, c))
                    cur_v = int(ctx.confirmed_board.get(r, c))
                    if base_v == 0 and cur_v != 0 and cur_v != 10:
                        diffs.append((r, c, cur_v))
            if len(diffs) == 2:
                (r1, c1, _), (r2, c2, _) = diffs[0], diffs[1]
                is_vertical = c1 == c2 and abs(r1 - r2) == 1
                is_horizontal = r1 == r2 and abs(c1 - c2) == 1
                if is_vertical or is_horizontal:
                    falling_pair_b: tuple[int, int] | None = None
                    if len(prev_next_queue) >= 2:
                        falling_pair_b = prev_next_queue[-2]
                    elif prev_next_queue:
                        falling_pair_b = prev_next_queue[-1]
                    if (
                        falling_pair_b is not None
                        and falling_pair_b[0] not in (
                            COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA,
                        )
                        and falling_pair_b[1] not in (
                            COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA,
                        )
                    ):
                        # Phase 1a: placement_inferrer 経由で物理整合置き判定.
                        # cycle 71b: 案 A (連鎖整合性) + 案 B (縦/横幾何) を活用.
                        # cycle 71l (β2'): frame_bgr + region で HSV 距離分類.
                        region_for_side_b = (
                            DEFAULT_P1_REGION if side == "1P"
                            else DEFAULT_P2_REGION
                        )
                        bg_fp_for_side_b = getattr(
                            self._reader,
                            "_bg_fp_p1" if side == "1P" else "_bg_fp_p2",
                            None,
                        )
                        inferred_b = infer_placement(
                            prev_confirmed, ctx.confirmed_board, falling_pair_b,
                            chain_sim=self._chain_sim,
                            score_delta_observed=0,
                            frame_bgr=frame_bgr,
                            region=region_for_side_b,
                            bg_fp=bg_fp_for_side_b,
                            guard_empty_hallucination=self._enable_infer_empty_guard,
                            enable_hsv_classify_fallback=self._enable_hsv_classify_fallback,
                        )
                        if inferred_b is not None:
                            # 即時連鎖判定 (= 短い TSUMO_FALL 取りこぼし時も適用)
                            # cycle 71c: prev_confirmed を渡して大量 add ガード適用.
                            # cycle 71n (案 ε): score_delta ガードも適用.
                            final_b, _ = resolve_after_placement(
                                inferred_b, self._chain_sim,
                                prev_confirmed=prev_confirmed,
                                score_delta_observed=score_d_for_self,
                            )
                            ctx.confirmed_board = final_b
                            ctx.pending_board = final_b.copy()

        # Phase C-6 の C: CHAIN_FALL → STABLE 復帰時に ChainSimulator
        # final_board (= 物理推論で確定した連鎖後盤面) で上書き。
        # 旧実装は CNN 盤面を採用していたが、連鎖アニメ残光・エフェクトで
        # CNN は信頼できないため、物理推論結果を真値として採用する。
        # 根治 (2026-07-23): enable_gravity_settle_state=True (default) では
        # CHAIN は必ず GRAVITY_SETTLE を経由してから STABLE に遷移するため、
        # この遷移フレームでは prev_state==GRAVITY_SETTLE かつ chain_event 引数は
        # 既に None (active_chain が上流で None 化済) になっており、旧条件
        # (prev_state==CHAIN) は dead code だった。GRAVITY_SETTLE も許容し、
        # _effective_chain_event (frame 先頭で確定済、 chain_event None 時は
        # 退避 ChainEvent を fallback) を使う。
        if (
            prev_state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)
            and ctx.state == BoardState.STABLE
            and _effective_chain_event is not None
        ):
            # 退避 event は one-shot 消費 (次回以降の誤爆防止のため即クリア)
            if side == "1P":
                self._last_chain_event_for_settle_1p = None
            else:
                self._last_chain_event_for_settle_2p = None
            try:
                from src.board_state_machine import _apply_gravity_filter
                from src.chain import ChainSimulator
                if not hasattr(self, "_chain_sim"):
                    self._chain_sim = ChainSimulator()  # type: ignore[attr-defined]
                cr = self._chain_sim.simulate(  # type: ignore[attr-defined]
                    _effective_chain_event.before_board,
                )
                if cr.chain_count > 0 and cr.final_board is not None:
                    final = cr.final_board.copy()
                    _apply_gravity_filter(final)
                    # 反復5 修正 (2026-07-23, user承認): Step3(b)(c) の答え合わせを
                    # 「事前ゲート」から「事後検証」に作り直す。
                    # 旧実装は GRAVITY_SETTLE 直後の残光で汚れた単一フレーム CNN と
                    # 比較して正しい final_board 注入 (反復1の残像修正) まで過剰
                    # 棄却し、残像/連鎖後不一致率を悪化させる回帰を起こしていた
                    # (物理レビュー実測: 0.09→0.28)。まず素直に適用し
                    # (= 反復1の修正を邪魔しない)、直近数 STABLE frame の
                    # 多数決盤面が揃ってから答え合わせする
                    # (_update_chain_estimate_verification)。
                    ctx.confirmed_board = final
                    ctx.pending_board = final.copy()
                    verify_state = {"expected": final.copy(), "cnn_history": []}
                    if side == "1P":
                        self._chain_verify_pending_1p = verify_state
                    else:
                        self._chain_verify_pending_2p = verify_state
                    # cycle 28a (H3, 2026-05-18): ChainSimulator chain_result
                    # から消去 puyo 色を集計、 自 side tsumo_count から減算。
                    # 連鎖前 board 認識誤りで「消去数 > 累積数」 になるケース
                    # は warning + 0 にクランプ (cycle 28b H4 自己評価 signal)。
                    erased_color_count: dict[int, int] = {}
                    for step in cr.steps:
                        for group in step.erased_groups:
                            erased_color_count[group.color] = (
                                erased_color_count.get(group.color, 0)
                                + group.size
                            )
                    target_tsumo = (
                        self._tsumo_count_1p if side == "1P"
                        else self._tsumo_count_2p
                    )
                    for color, n in erased_color_count.items():
                        accumulated = target_tsumo.get(color, 0)
                        if accumulated >= n:
                            target_tsumo[color] = accumulated - n
                        else:
                            if frame_idx % 60 == 0:
                                print(
                                    f"[chain-tsumo-undershoot] {side} "
                                    f"frame={frame_idx} color={color} "
                                    f"erased={n} accumulated={accumulated} "
                                    f"→ clamp 0",
                                )
                            target_tsumo[color] = 0
                    # cycle 28a (H2): 連鎖完了で自 side constraint_valid を
                    # 再有効化。 tsumo_count 減算 (= H3) が成立したので
                    # 累積 - 消去 = field count の恒等式が再び成立する。
                    # 相手 side は OJAMA_FALL 完了待ち (= cycle 29 H6 で対応)。
                    if side == "1P":
                        self._constraint_valid_1p = True
                    else:
                        self._constraint_valid_2p = True
            except Exception:
                pass

        # 機能C: CHAIN → STABLE 遷移直後の confirmed 凍結。
        # enable_chain_exit_warmup=True の場合のみ有効。
        # エフェクト残光色が _merge_diff_only 経由で confirmed に混入するのを防ぐ。
        # 凍結終了時刻 (_chain_exit_until_*) を更新し、下流の confirmed 更新をスキップさせる。
        # 案X 連動: enable_chain_exit_next_signal=True 時は CHAIN_EXIT_NEXT_WARMUP_SEC(0.5s) を使用。
        # 案X が連鎖を早く終わらせると置き直後・エフェクト残光が STABLE 露出するため
        # 通常の CHAIN_EXIT_WARMUP_SEC(0.1s) より長い凍結時間が必要。
        # 根治 (2026-07-23): GRAVITY_SETTLE 経由の STABLE 復帰も同様に凍結対象とする
        # (enable_gravity_settle_state=True では CHAIN が直接 STABLE に遷移しないため)。
        if (
            self._enable_chain_exit_warmup
            and prev_state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)
            and ctx.state == BoardState.STABLE
        ):
            # 案X 時は専用の長い凍結時間を使用、それ以外は機能C の短い凍結時間を使用
            _warmup_sec = (
                CHAIN_EXIT_NEXT_WARMUP_SEC
                if self._enable_chain_exit_next_signal
                else CHAIN_EXIT_WARMUP_SEC
            )
            warmup_until = time_sec + _warmup_sec
            if side == "1P":
                self._chain_exit_until_1p = warmup_until
            else:
                self._chain_exit_until_2p = warmup_until

        # cycle 29 (2026-05-18): NEXT 移動検知ベースで grace + landing_vote 起動。
        # 既存の TSUMO_FALL → STABLE 経路は placement_inferrer で confirmed を
        # 物理推論済 (= ctx.confirmed_board 更新済)。 NEXT 検知 frame で
        # その盤面を grace に hold + landing_vote 開始。
        # state machine が詰まる動画 (= v97 53 秒問題) でも NEXT は動くので
        # grace / landing_vote が確実に起動する救済になる。
        landing_pending = (
            self._landing_pending_1p if side == "1P"
            else self._landing_pending_2p
        )
        if (
            landing_pending is not None
            and landing_pending[0] == frame_idx
            and ctx.confirmed_board is not None
        ):
            _, falling_pair_for_grace = landing_pending
            # 案(iii) (2026-07-25): 着地 infer 完了時に計算済の疑わしいセル集合を
            # P7 (_start_landing_vote) に伝播する。
            distrust_cells_for_vote = (
                self._landing_distrust_1p if side == "1P"
                else self._landing_distrust_2p
            )
            if prev_confirmed is not None:
                self._start_landing_vote(
                    side, frame_idx, prev_confirmed, ctx.confirmed_board,
                    next_colors=falling_pair_for_grace,
                    distrust_cells=distrust_cells_for_vote,
                    time_sec=time_sec,
                )
            grace_until = frame_idx + self.LANDING_GRACE_FRAMES
            # フレーム定数→時間定数化 Stage1 (2026-07-25): time_sec 基準の
            # 満了時刻を併せて記録する (実ロジックはこちらを正として使う)。
            grace_until_time = time_sec + self.LANDING_GRACE_SEC
            if side == "1P":
                self._landing_grace_1p = (
                    grace_until, ctx.confirmed_board.copy(), grace_until_time,
                )
                self._landing_pending_1p = None
                # 伝播済のため 1 ツモ分だけ生存させてクリア (次着地まで持ち越さない)。
                self._landing_distrust_1p = set()
            else:
                self._landing_grace_2p = (
                    grace_until, ctx.confirmed_board.copy(), grace_until_time,
                )
                self._landing_pending_2p = None
                self._landing_distrust_2p = set()

        # cycle 31 (B 軸, 2026-05-18): baseline 整合性 check + 自己修復。
        # 制御フラグ化 (2026-07-25, A/B 計測用): 実装は _check_baseline_broken_reset
        # に分離済み。enable_baseline_broken_reset=False で機能全体をスキップ、
        # enable_baseline_broken_grace=True で STABLE 突入直後の猶予を追加できる。
        self._check_baseline_broken_reset(
            side, frame_idx, time_sec, is_active, ctx, cnn_board, prev_state,
            sm, drift, gen,
        )

        inferred = gen.generate(
            ctx, chain_event=chain_event, time_sec=time_sec,
        )
        drift_res = drift.update(inferred, cnn_board)
        if drift_res.needs_resync:
            # DriftDetector 再同期ループ暴走ガード (2026-07-25, c34 実測)。
            # 両ガードとも独立 flag のため、それぞれ単独評価 (どちらか一方でも
            # 抑制条件を満たせば reset をスキップする)。counter は各ガードが
            # 単独で適用された場合の抑制回数を記録する (効果測定用)。
            _drift_resync_suppress = False
            if self._enable_drift_resync_match_start_guard:
                _since_match_start = (
                    time_sec - self._match_active_started_time
                )
                if (
                    self._match_active_started_time >= 0
                    and _since_match_start
                    < self.DRIFT_RESYNC_MATCH_START_GUARD_SEC
                ):
                    _drift_resync_suppress = True
                    if side == "1P":
                        self._drift_resync_start_guard_suppressed_1p += 1
                    else:
                        self._drift_resync_start_guard_suppressed_2p += 1
            if self._enable_drift_resync_hsv_gate:
                _calibrated_colors = len(self._online_hsv_injected_colors)
                if _calibrated_colors < self.DRIFT_RESYNC_MIN_CALIBRATED_COLORS:
                    _drift_resync_suppress = True
                    if side == "1P":
                        self._drift_resync_hsv_gate_suppressed_1p += 1
                    else:
                        self._drift_resync_hsv_gate_suppressed_2p += 1
            if not _drift_resync_suppress:
                sm.reset(keep_match_state=True)
                drift.reset()
                gen.reset()

        cur_score = (
            score_tracker.last_score if score_tracker is not None else None
        )

        # cycle 26 (2026-05-18, A1): grace 期間判定を先に行い、grace 中は
        # NEXT 累積制約・landing_vote 反映・long-term vote override を skip。
        # grace 中は物理推論 final_board (= NEXT pair 由来) を 100% 信頼する。
        grace_state_pre = (
            self._landing_grace_1p if side == "1P" else self._landing_grace_2p
        )
        # フレーム定数→時間定数化 Stage1 (2026-07-25): 旧 `frame_idx <
        # grace_state_pre[0]` (frame 差分) を time_sec 基準 (grace_state_pre[2])
        # に置換。60fps では bit-identical、30fps では実秒基準になる。
        in_grace = (
            grace_state_pre is not None
            and time_sec < grace_state_pre[2]
            and ctx.state == BoardState.STABLE
        )

        # サイクル66 (2026-05-11): NEXT 累積制約による色 count 補正
        # 試合開始〜最初の連鎖発火まで: tsumo_count[c] == field の c puyo 数 (厳密)
        # 不一致なら mismatch をログ + 不足色を「疑わしい cell (= 過剰色)」 に置換
        # cycle 26 (A1): grace 中は skip (final_board hold を尊重)
        # cycle 26b (案 X, 2026-05-18): tsumo_total == 0 (= 試合開始直後 or
        # 試合切替直後で最初のツモが未着地) の STABLE は field 必ず empty。
        # 背景誤認が confirmed に乗るのを物理推論で排除。
        if (
            not in_grace
            and ctx.state == BoardState.STABLE
            and ctx.confirmed_board is not None
        ):
            tsumo_count = (
                self._tsumo_count_1p if side == "1P"
                else self._tsumo_count_2p
            )
            constraint_valid = (
                self._constraint_valid_1p if side == "1P"
                else self._constraint_valid_2p
            )
            # 案2: enable_constraint_fill=False のとき constraint_fill を完全 skip
            if (
                self._enable_constraint_fill
                and constraint_valid
                and sum(tsumo_count.values()) > 0
            ):
                # 案1: cnn_board を protect_board として渡す。
                # CNN が認識した色と confirmed が一致するセルは excess でも保護される。
                # hsv_board は image_reader 改修なしでは取得不可のため cnn のみ渡す。
                ctx.confirmed_board = self._apply_next_count_constraint(
                    ctx.confirmed_board, tsumo_count, side, frame_idx,
                    protect_board=cnn_board,
                )
                ctx.pending_board = ctx.confirmed_board.copy()

        # W-α: STABLE 中は prob_board を publish。TSUMO_FALL → STABLE 経路で
        # infer_hidden_row 由来の側面情報が乗っていればそれを優先採用。
        # それ以外の STABLE 経路 (初回 STABLE / CHAIN→STABLE / OJAMA→STABLE)
        # は from_board(confirmed_board) でフォールバック。
        publish_prob_board: ProbabilisticBoard | None = None
        if (
            PROB_BOARD_PUBLISH_ON_STABLE
            and ctx.state == BoardState.STABLE
            and ctx.confirmed_board is not None
        ):
            if side_prob_board is not None:
                publish_prob_board = side_prob_board
            else:
                publish_prob_board = ProbabilisticBoard.from_board(
                    ctx.confirmed_board,
                )
        # 着地直後 grace period: 置いた直後 N frame 以内なら inferred_landing で
        # confirmed_board を hold する (CNN/HSV 不安定対策、 ユーザー指摘 2026-05-10)
        grace_state = self._landing_grace_1p if side == "1P" else self._landing_grace_2p
        if in_grace and grace_state is not None:
            ctx.confirmed_board = grace_state[1].copy()
            ctx.pending_board = grace_state[1].copy()
        elif grace_state is not None and time_sec >= grace_state[2]:
            # grace 終了: クリア (time_sec 基準、 Stage1 2026-07-25)
            if side == "1P":
                self._landing_grace_1p = None
            else:
                self._landing_grace_2p = None

        # cycle 71h: 着地後 vote refinement.
        # TSUMO_FALL→STABLE 着地時に登録された pending エントリの cnn 観測色を蓄積、
        # LANDING_VOTE_FRAMES 経過時に最頻値で confirmed_board の cell 色を更新.
        # 1 秒経過後の正しい色判別 (= ユーザー要件) を実現.
        # cycle 71m (β2''): frame_bgr を渡して HSV 距離分類 vote も並行蓄積.
        # 機能C: CHAIN → STABLE warmup 中は confirmed 更新をスキップ。
        # CHAIN_EXIT_WARMUP_SEC 秒経過後に通常更新を再開する。
        # in_grace と独立して管理する (着地 grace とは別のシグナル)。
        _chain_exit_until = (
            self._chain_exit_until_1p if side == "1P"
            else self._chain_exit_until_2p
        )
        in_chain_exit_warmup = (
            self._enable_chain_exit_warmup
            and ctx.state == BoardState.STABLE
            and time_sec < _chain_exit_until
        )

        # cycle 26 (A1): grace 中は updated 反映を skip (蓄積は継続)。
        # grace 終了後の vote 完了で正しく反映される。
        # 機能C: chain exit warmup 中も skip (残光色混入防止)。
        if ctx.confirmed_board is not None:
            vote_updated = self._update_landing_votes(
                side, frame_idx, cnn_board, ctx.confirmed_board,
                frame_bgr=frame_bgr, time_sec=time_sec,
            )
            if vote_updated is not None and not in_grace and not in_chain_exit_warmup:
                ctx.confirmed_board = vote_updated

        # cycle 71n (案 θ): STABLE 中の長期不一致 vote.
        # 各 cell の CNN 観測色を直近 N frame の最頻値で評価、 confirmed と
        # 異なる色が 7 割以上の vote 持ったら CNN 側で上書き (= 自動修正).
        if (
            ctx.state == BoardState.STABLE
            and ctx.confirmed_board is not None
        ):
            history = (
                self._stable_cnn_history_1p if side == "1P"
                else self._stable_cnn_history_2p
            )
            from collections import Counter as _CounterClass
            override_fired = False
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    cnn_v = int(cnn_board.get(r, c))
                    h_list = history.setdefault((r, c), [])
                    h_list.append(cnn_v)
                    if len(h_list) > self.STABLE_CNN_HISTORY_FRAMES:
                        h_list.pop(0)
                    if len(h_list) >= self.STABLE_CNN_HISTORY_FRAMES:
                        counter = _CounterClass(h_list)
                        most_common, mc_count = counter.most_common(1)[0]
                        ratio = mc_count / len(h_list)
                        conf_v = int(ctx.confirmed_board.get(r, c))
                        if (
                            most_common != conf_v
                            and most_common not in (
                                COLOR_EMPTY, COLOR_UNKNOWN,
                            )
                            and ratio >= self.STABLE_OVERRIDE_MIN_RATIO
                        ):
                            # cycle 71v 浮きぷよ防止 (2026-05-14):
                            # EMPTY→puyo の override は cell が物理的に支持
                            # されている場合のみ許可 (= 下段が non-EMPTY or row=12)
                            if conf_v == COLOR_EMPTY:
                                if r < BOARD_ROWS - 1:
                                    below = int(
                                        ctx.confirmed_board.get(r + 1, c)
                                    )
                                    if below == COLOR_EMPTY or below == COLOR_UNKNOWN:
                                        # 下が空 → 浮き → 上書き禁止 (history は維持)
                                        continue
                            # cycle 26 (A1): grace 中は override skip
                            # (history append は続行、grace 終了後に発火)
                            # 機能C: chain exit warmup 中も override skip
                            if in_grace or in_chain_exit_warmup:
                                continue
                            ctx.confirmed_board.set(r, c, most_common)
                            h_list.clear()
                            override_fired = True
            # cycle 71v: override が走った frame は gravity filter で
            # 浮きぷよ残留を最終 sweep. v51/v70 の背景誤認対策。
            if override_fired:
                from src.board_state_machine import _apply_gravity_filter
                _apply_gravity_filter(ctx.confirmed_board)
        else:
            # STABLE 以外 → 履歴クリア (= state 切替で reset)
            if side == "1P":
                self._stable_cnn_history_1p.clear()
            else:
                self._stable_cnn_history_2p.clear()

        # cycle 71j (案 1a): STABLE 中の UNKNOWN cell を直前 STABLE 値で埋める.
        # cycle 71t (2026-05-13 副作用対策): 案 B (= puyo→puyo / puyo→EMPTY 復元)
        # を撤回. 連鎖後に「消えた cell」 が prev で復元されて「ずっと残る」 問題.
        # 案 1a の UNKNOWN 補完のみ維持.
        if (
            ctx.state == BoardState.STABLE
            and ctx.confirmed_board is not None
        ):
            prev_stable = (
                self._prev_stable_confirmed_1p if side == "1P"
                else self._prev_stable_confirmed_2p
            )
            if prev_stable is not None:
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        cur_v = int(ctx.confirmed_board.get(r, c))
                        if cur_v == COLOR_UNKNOWN:
                            prev_v = int(prev_stable.get(r, c))
                            if prev_v != COLOR_UNKNOWN:
                                ctx.confirmed_board.set(r, c, prev_v)
            # cycle 71v-A (2026-05-15): UNKNOWN を CNN 現観測でも埋める拡張.
            # 旧 cycle 71j は prev_stable (= 1 frame 前 STABLE) のみで埋めるため、
            # 直前 frame でも UNKNOWN だった cells は永久に UNKNOWN のまま残る。
            # 現 frame の cnn_board が非 UNKNOWN ならそちらで埋める (= 物理整合性 +
            # gravity filter を経た値を信頼)。 ユーザー要件「? は誤認と同レベル」 対策.
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    cur_v = int(ctx.confirmed_board.get(r, c))
                    if cur_v == COLOR_UNKNOWN:
                        cnn_v = int(cnn_board.get(r, c))
                        if cnn_v not in (COLOR_UNKNOWN,):
                            ctx.confirmed_board.set(r, c, cnn_v)
            # 案 Y-4: deferred landing の consensus 投票を進める (T2 の前に実行)。
            # 拮抗ケースで着地直後から DEFERRED_MAX_FRAMES 内に投票が集まれば確定する。
            if self._enable_hsv_deferred_consensus:
                self._update_deferred_landing(
                    side, frame_bgr, cnn_board, ctx, ctx.state,
                )
            # T2: STABLE → STABLE 遷移時の色変化検証。
            # 前 STABLE と現 STABLE で「色A → 色B」(異色間変化) かつ
            # 間に CHAIN signal がなければ認識誤りと判断し前値で上書き。
            # grace 中・chain_event あり・prev_stable なし は skip。
            # 根治 (2026-07-23): GRAVITY_SETTLE 経由の STABLE 復帰では、この
            # frame の chain_event 引数は既に None だが実質は連鎖直後
            # (Phase C-6 の C が final_board を注入した frame) であるため、
            # 生の chain_event でなく _effective_chain_event (frame 先頭で
            # 確定済) で判定する。 でなければ T2 が Phase C-6 の C の直後に
            # fresh な final_board を古い prev_stable で即座に上書きしてしまう。
            prev_stable_t2 = (
                self._prev_stable_confirmed_1p if side == "1P"
                else self._prev_stable_confirmed_2p
            )
            # 案 Y-4: deferred 確定直後の frame は T2 をスキップして旧色フリーズを防ぐ。
            _deferred_committed_this_frame = (
                self._deferred_just_committed_1p if side == "1P"
                else self._deferred_just_committed_2p
            )
            # deferred commit フラグをリセット (1 frame 限定)
            if _deferred_committed_this_frame:
                if side == "1P":
                    self._deferred_just_committed_1p = False
                else:
                    self._deferred_just_committed_2p = False
            if (
                prev_stable is not None
                and _effective_chain_event is None
                and not in_grace
                and not _deferred_committed_this_frame  # 案 Y-4: deferred 確定 frame は T2 スキップ
            ):
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        cur_v = int(ctx.confirmed_board.get(r, c))
                        pv = int(prev_stable.get(r, c))
                        # 「色A → 色B」 (いずれも非 EMPTY / 非 UNKNOWN / 異色)
                        both_colored = (
                            pv not in (COLOR_EMPTY, COLOR_UNKNOWN)
                            and cur_v not in (COLOR_EMPTY, COLOR_UNKNOWN)
                        )
                        if both_colored and pv != cur_v:
                            # T2 高確信 yield: enable_t2_highconf_yield=True かつ
                            # CNN が現在の confirmed 色 (cur_v) を支持している場合は
                            # prev_stable での上書きをスキップ。
                            # 理由: infer_placement が誤色を confirmed に書き込み、
                            # それを T2 が毎フレーム prev_stable で維持することで
                            # 誤色フリーズが数百フレーム継続する問題を解除する。
                            # B1 禁忌との違い: B1 は「色→空」保護 (連鎖エフェクト誤固定)。
                            # 本修正は逆に「prev_stable の古い色による上書き」を
                            # CNN 支持時に解除する (保護を弱める方向)。
                            # CHAIN 非 STABLE 中は T2 自体が実行されないため
                            # 連鎖エフェクト誤固定リスクはゼロ。
                            if self._enable_t2_highconf_yield:
                                cnn_v = int(cnn_board.get(r, c))
                                if (
                                    cnn_v == cur_v
                                    and cnn_v not in (COLOR_EMPTY, COLOR_UNKNOWN)
                                    # prev_stable が空のセルは yield しない。
                                    # 背景 FP (pv=空 → cur_v=色) は T2 で空に戻す。
                                    # yield は「色 → 別色フリーズ (pv=色付き)」のみ対象。
                                    and pv not in (COLOR_EMPTY, COLOR_UNKNOWN)
                                ):
                                    # CNN が cur_v を支持 かつ pv も色付き → T2 上書きをスキップ
                                    continue
                            # 前 STABLE 値で上書き (= 認識誤り棄却)
                            ctx.confirmed_board.set(r, c, pv)
            # 現 STABLE を「直前 STABLE」 として記憶 (= 次 frame で参照)
            if side == "1P":
                self._prev_stable_confirmed_1p = ctx.confirmed_board.copy()
            else:
                self._prev_stable_confirmed_2p = ctx.confirmed_board.copy()

        # ユーザー提案: ネクスト履歴整合性 sanity check
        # 「ネクストとして来たぷよが盤面に出現するはず」 = 履歴に無い色は誤認識
        # → STABLE 中の confirmed_board で履歴外の色を COLOR_UNKNOWN に倒す
        # cycle 71v-B (2026-05-15): 永続的 ever_seen 集合を維持して、 NEXT 履歴が
        # 8 ペア cap でスクロールアウトしても色は「観測済」 扱いを継続。 旧実装は
        # 試合途中で古い NEXT 履歴が消えると、 該当色のセルが定期的に UNKNOWN
        # 化する問題があった (= ユーザー報告「1色が定期的に?になる」)。
        ever_seen = (
            self._ever_seen_colors_1p if side == "1P"
            else self._ever_seen_colors_2p
        )
        for pair in ctx.next_queue:
            for c in pair:
                if c not in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN):
                    ever_seen.add(int(c))
        published_confirmed = ctx.confirmed_board
        if (
            ctx.state == BoardState.STABLE
            and ctx.confirmed_board is not None
            and len(ctx.next_queue) >= 3  # 履歴 3 ペア = 6 色観測でようやく信頼
        ):
            region_for_validate = (
                DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
            )
            published_confirmed = self._validate_next_history(
                ctx.confirmed_board, ctx.next_queue,
                ever_seen=ever_seen,
                frame_bgr=frame_bgr,
                region=region_for_validate,
            )
        # T4 PuyoErasureMonitor: STABLE 中「色→EMPTY」遷移を記録。
        # prev_stable と current confirmed_board を比較する。
        erasure_mon = (
            self._erasure_monitor_1p if side == "1P"
            else self._erasure_monitor_2p
        )
        prev_stable_for_monitor = (
            self._prev_stable_confirmed_1p if side == "1P"
            else self._prev_stable_confirmed_2p
        )
        erasure_mon.update(
            frame_idx, ctx.state,
            prev_stable_for_monitor,
            published_confirmed,
        )
        # alert リストを SideResult に格納 (frame 座標 = (row, col) のみ)
        recent_alerts = [
            (r, c) for (fi, r, c) in erasure_mon.alerts
            if fi == frame_idx
        ]
        # C2 StableTransitionMonitor: STABLE 遷移イベントを通知。
        # STABLE→NON-STABLE 遷移: on_stable_end で board をスナップショット。
        # NON-STABLE 中の連鎖 / ojama イベント: on_non_stable_event で記録。
        # NON-STABLE→STABLE 遷移: on_stable_start でぷよ数比較。
        transition_mon = (
            self._transition_monitor_1p if side == "1P"
            else self._transition_monitor_2p
        )
        transition_drop_alerts = _update_transition_monitor(
            monitor=transition_mon,
            prev_state=prev_state,
            curr_state=ctx.state,
            frame_idx=frame_idx,
            time_sec=time_sec,
            chain_event=chain_event,
            confirmed_board=published_confirmed,
        )
        # B1 PiecePersistenceGuard: STABLE 中 cell 色の物理保護。
        # NON-STABLE → STABLE 遷移: on_non_stable_enter は NON-STABLE 遷移時に呼ぶ。
        # STABLE 中 confirmed_board 確定後: on_stable_confirmed で保護登録 + guard 適用。
        published_confirmed = _apply_piece_persistence_guard(
            guard=(
                self._piece_persistence_1p if side == "1P"
                else self._piece_persistence_2p
            ),
            prev_state=prev_state,
            curr_state=ctx.state,
            published_confirmed=published_confirmed,
        )
        # 不具合B 対処: 予告おじゃま発光ガード (2026-06-04)。
        # STABLE 中のみ適用。CHAIN 中は既存凍結機構で保護済みのためスキップ。
        # 発光 OFF 中は frozen_board を現 confirmed で更新し次の発光に備える。
        # 発光 ON 中は frozen_board の色で confirmed を保護する。
        if self._enable_ojama_warning_glow_guard and frame_bgr is not None:
            glow_state = (
                self._glow_guard_1p if side == "1P" else self._glow_guard_2p
            )
            if glow_state is not None:
                from src.ojama_warning_glow_guard import (
                    compute_glow_score,
                    update_glow_state,
                    apply_glow_guard,
                )
                region_for_glow = (
                    DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
                )
                glow_score = compute_glow_score(frame_bgr, region_for_glow)
                is_glow = update_glow_state(glow_state, glow_score, frame_idx)
                if ctx.state == BoardState.STABLE and published_confirmed is not None:
                    if is_glow:
                        # 発光中: frozen で confirmed を保護する
                        published_confirmed = apply_glow_guard(
                            published_confirmed, glow_state, is_glow,
                        )
                    else:
                        # 発光 OFF 中 (STABLE 確定時のみ): frozen を更新する
                        glow_state.frozen_board = published_confirmed.copy()
        # 反復5 修正 Step3(b)(c) (2026-07-23): 事後答え合わせ。連鎖後
        # final_board 適用から CHAIN_VERIFY_FRAMES 分の STABLE cnn_board が
        # 集まったフレームでのみ発火。不一致なら多数決盤面で confirmed_board
        # を補正する (適用そのものは止めない = 反復1の残像修正を維持)。
        answer_check_result, correction_board = (
            self._update_chain_estimate_verification(side, ctx.state, cnn_board)
        )
        if correction_board is not None:
            ctx.confirmed_board = correction_board
            ctx.pending_board = correction_board.copy()
            published_confirmed = correction_board
        # 反復4 (2026-07-23): confirmed_board=None の理由分類 (診断計装のみ、
        # 挙動には一切影響しない optional フィールド)。
        board_none_reason = self._classify_board_none_reason(
            side, is_active, published_confirmed, ctx.state,
        )
        # 反復5 Step2 (2026-07-23): 物理推論スルー。confirmed_board 自体は
        # 一切変更しない (標準 eval 経路は従来通り None を見るのみ)。
        estimated_board, board_provenance = self._compute_chain_estimate(
            side, ctx.state, time_sec,
        )
        return SideResult(
            side=side,
            state=ctx.state,
            cnn_board=cnn_board,
            inferred_board=inferred,
            confirmed_board=published_confirmed,
            drift=drift_res,
            score=cur_score,
            score_delta=score_d_2p_for_ojama,
            chain_event=chain_event,
            prob_board=publish_prob_board,
            next_pair=next_pair,
            dnext_pair=dnext_pair,
            erasure_alerts=recent_alerts if recent_alerts else None,
            transition_drop_alerts=transition_drop_alerts if transition_drop_alerts else None,
            landing_diag=_landing_diag,
            board_none_reason=board_none_reason,
            estimated_board=estimated_board,
            board_provenance=board_provenance,
            answer_check_result=answer_check_result,
        )


    def _set_deferred_confirmed(
        self,
        side: str,
        winner_board: "Board",
        ctx: "StateContext",
    ) -> None:
        """deferred 確定時に confirmed_board と prev_stable を winner で上書きする.

        案 Y-4 (2026-06-03): consensus 投票で勝者が確定したとき呼ばれる。
        T2 が直後に旧 prev_stable で干渉しないよう _deferred_just_committed フラグを立てる。

        Args:
            side: "1P" or "2P"。
            winner_board: 着地後の確定盤面 (多数票候補)。
            ctx: BoardStateMachine が管理する StateContext (confirmed_board を上書き)。
        """
        ctx.confirmed_board = winner_board.copy()
        if ctx.pending_board is not None:
            ctx.pending_board = winner_board.copy()
        # prev_stable も勝者で更新し、 T2 が旧色を再フリーズするのを防ぐ。
        if side == "1P":
            self._prev_stable_confirmed_1p = winner_board.copy()
            self._deferred_just_committed_1p = True
            self._deferred_landing_1p = None
        else:
            self._prev_stable_confirmed_2p = winner_board.copy()
            self._deferred_just_committed_2p = True
            self._deferred_landing_2p = None

    def _update_deferred_landing(
        self,
        side: str,
        frame_bgr: "np.ndarray",
        cnn_board: "Board",
        ctx: "StateContext",
        state: "BoardState",
    ) -> None:
        """deferred 保留中に consensus 投票を進め、確定条件を満たせば確定させる.

        案 Y-4 (2026-06-03): STABLE 毎 frame、T2 の前に呼ぶ。
        連鎖遷移 (STABLE 外) / 新 TSUMO_FALL 発生で deferred state をクリアする。

        Args:
            side: "1P" or "2P"。
            frame_bgr: 現 frame の BGR 画像。
            cnn_board: 現 frame の CNN+HSV 融合観測盤面。
            ctx: BoardStateMachine StateContext (confirmed_board 上書き用)。
            state: 現在の BoardState。
        """
        from src.board_state_machine import BoardState as _BS
        from src.placement_inferrer import (
            _score_consensus_for_candidate,
            DEFERRED_CONSENSUS_THRESHOLD, DEFERRED_MAX_FRAMES,
        )
        deferred = (
            self._deferred_landing_1p if side == "1P"
            else self._deferred_landing_2p
        )
        if deferred is None:
            return
        # STABLE 外 (連鎖/TSUMO_FALL 等) に遷移したら即クリア (物理シミュ優先)
        if state != _BS.STABLE:
            if side == "1P":
                self._deferred_landing_1p = None
            else:
                self._deferred_landing_2p = None
            return
        # HSV-only 分類器を取得
        classifier = getattr(self._reader, "_classifier", None)
        hsv_clf = getattr(classifier, "_hsv", classifier)
        region = (
            DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
        )
        base_cells: list[tuple[int, int]] = deferred["base_cells"]
        # 各候補に CNN/HSV 票を加算
        votes_std = deferred["votes_std"] + _score_consensus_for_candidate(
            deferred["board_std"], cnn_board, frame_bgr, region, base_cells,
            hsv_classifier=hsv_clf,
        )
        votes_rev = deferred["votes_rev"] + _score_consensus_for_candidate(
            deferred["board_rev"], cnn_board, frame_bgr, region, base_cells,
            hsv_classifier=hsv_clf,
        )
        frames_left = deferred["frames_left"] - 1
        deferred["votes_std"] = votes_std
        deferred["votes_rev"] = votes_rev
        deferred["frames_left"] = frames_left
        # 確定条件: 閾値到達 or frames_left 消化
        should_commit = (
            votes_std >= DEFERRED_CONSENSUS_THRESHOLD
            or votes_rev >= DEFERRED_CONSENSUS_THRESHOLD
            or frames_left <= 0
        )
        if not should_commit:
            return
        # 勝者は票数多い方。同票 or frames_left 超過は board_std (安全 fallback) を採用。
        winner = (
            deferred["board_rev"]
            if votes_rev > votes_std
            else deferred["board_std"]
        )
        self._set_deferred_confirmed(side, winner, ctx)

    @staticmethod
    def _validate_next_history(
        board: Board, next_queue: list[tuple[int, int]],
        ever_seen: set[int] | None = None,
        frame_bgr: "np.ndarray | None" = None,
        region: "object | None" = None,
    ) -> Board:
        """ネクスト履歴 整合性 + 浮きぷよ除去 (2026-05-10 FIX-B 拡張).

        1. ネクスト履歴に無い色は HSV 距離最小の seen 色で置換 (= cycle 8 Innovation D)
           (旧挙動は UNKNOWN マークだったが、 "?" 視認問題のため fill 化)
        2. **浮きぷよ除去**: 列の下に empty があり上に puyo があれば、 上の puyo は
           物理的にあり得ない → empty に倒す

        cycle 71v-B (2026-05-15): ever_seen 集合で NEXT cap スクロールアウト対策.
        cycle 8 (2026-05-15, Innovation D): UNKNOWN→HSV 距離最小 seen 色 replace.
            frame_bgr + region 渡しなら HSV ベース、 無ければ UNKNOWN フォールバック.
        """
        from src.board_state_machine import _apply_gravity_filter
        if not next_queue:
            out = board.copy()
            _apply_gravity_filter(out)
            return out
        seen: set[int] = {COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN}
        for pair in next_queue:
            for c in pair:
                seen.add(int(c))
        if ever_seen is not None:
            seen.update(ever_seen)
        out = board.copy()
        # 1. ネクスト履歴外色を 処理 (= 6 色全色出尽くし時はスキップ)
        if len(seen) < 9:
            # ever_seen から puyo 色 (= EMPTY/OJAMA/UNKNOWN 除く) を抽出
            seen_puyo_colors = {
                col for col in seen
                if col not in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN)
            }
            use_hsv = (
                frame_bgr is not None
                and region is not None
                and len(seen_puyo_colors) > 0
            )
            if use_hsv:
                try:
                    import cv2 as _cv2
                    from src.placement_inferrer import (
                        COLOR_HSV_CENTERS, _hsv_distance,
                        _extract_cell_patch_from_frame,
                    )
                except Exception:
                    use_hsv = False
            for r in range(BOARD_ROWS):
                for c in range(BOARD_COLS):
                    v = int(out.get(r, c))
                    if v in seen:
                        continue
                    # 履歴外色 → HSV 距離最小 seen 色で置換 or UNKNOWN
                    if use_hsv:
                        try:
                            patch = _extract_cell_patch_from_frame(
                                frame_bgr, region, r, c,
                            )
                            if patch is None or patch.size == 0:
                                out.set(r, c, COLOR_UNKNOWN)
                                continue
                            hsv_patch = _cv2.cvtColor(
                                patch, _cv2.COLOR_BGR2HSV,
                            )
                            h_med = int(np.median(hsv_patch[:, :, 0]))
                            s_med = int(np.median(hsv_patch[:, :, 1]))
                            v_med = int(np.median(hsv_patch[:, :, 2]))
                            best_c = None
                            best_d = float("inf")
                            for col_code in seen_puyo_colors:
                                if col_code not in COLOR_HSV_CENTERS:
                                    continue
                                d = _hsv_distance(
                                    h_med, s_med, v_med,
                                    COLOR_HSV_CENTERS[col_code],
                                )
                                if d < best_d:
                                    best_d = d
                                    best_c = col_code
                            out.set(
                                r, c,
                                best_c if best_c is not None else COLOR_UNKNOWN,
                            )
                        except Exception:
                            out.set(r, c, COLOR_UNKNOWN)
                    else:
                        out.set(r, c, COLOR_UNKNOWN)
        # 2. 浮きぷよ除去 (= 物理推論)
        _apply_gravity_filter(out)
        return out

    # ------------------------------------------------------------------
    # Phase I: 擬似ラベル抽出 (semi-supervised)
    # ------------------------------------------------------------------

    def _init_pseudo_validators(self) -> None:
        """擬似ラベル validator を初期化 (silent skip on import error).

        Phase I 改良: HiddenRowValidator を追加 (lenient reveal でも emit)。
        各 validator の import が個別に失敗しても他は登録継続する。
        """
        self._pseudo_validators = []
        try:
            from src.self_supervised.score_validator import ScoreValidator
            self._pseudo_validators.append(ScoreValidator())
        except Exception:
            pass
        try:
            from src.self_supervised.next_validator import NextValidator
            self._pseudo_validators.append(NextValidator())
        except Exception:
            pass
        try:
            from src.self_supervised.chain_validator import ChainValidator
            self._pseudo_validators.append(ChainValidator())
        except Exception:
            pass
        try:
            from src.self_supervised.hidden_row_validator import (
                HiddenRowValidator,
            )
            self._pseudo_validators.append(HiddenRowValidator())
        except Exception:
            pass
        # Phase I.b: cell 色 CNN 自己学習用 validator
        try:
            from src.self_supervised.cell_color_validator import (
                CellColorValidator,
            )
            self._pseudo_validators.append(CellColorValidator())
        except Exception:
            pass

    def flush_pseudo_labels(self) -> int:
        """蓄積された擬似ラベルを LabelStore に書き出し, 件数を返す.

        store が無ければ単に集計件数を返す (debug 用)。
        """
        if not self._enable_pseudo_label or not self._pseudo_validators:
            return 0
        total = 0
        all_samples: list = []
        for validator in self._pseudo_validators:
            try:
                samples = validator.collect()
                all_samples.extend(samples)
                total += len(samples)
            except Exception:
                continue
        if self._pseudo_label_store is not None and all_samples:
            try:
                self._pseudo_label_store.append(all_samples)
            except Exception:
                pass
        return total

    def collect_pseudo_labels(self) -> list:
        """擬似ラベルを取得 (store 書き出しなし、テスト用)."""
        if not self._enable_pseudo_label or not self._pseudo_validators:
            return []
        out: list = []
        for validator in self._pseudo_validators:
            try:
                out.extend(validator.collect())
            except Exception:
                continue
        return out


__all__ = [
    "PipelineResult",
    "RecognitionPipeline",
    "SideResult",
]


# ============================
# C2: StableTransitionMonitor 呼出ヘルパー (= クラス外 stateless 関数)
# ============================

def _update_transition_monitor(
    monitor: "object",  # StableTransitionMonitor (型循環回避のため文字列)
    prev_state: "BoardState",
    curr_state: "BoardState",
    frame_idx: int,
    time_sec: float,
    chain_event: "ChainEvent | None",
    confirmed_board: "Board | None",
) -> list[tuple]:
    """STABLE 遷移を StableTransitionMonitor に通知し、 alert リストを返す。

    遷移パターン:
        STABLE → NON-STABLE: on_stable_end (= board をスナップショット)
        NON-STABLE 中の chain_event: on_non_stable_event("chain_start")
        NON-STABLE → STABLE: on_stable_start (= ぷよ数比較 → alert)

    Args:
        monitor: StableTransitionMonitor インスタンス。
        prev_state: 前 frame の BoardState。
        curr_state: 現 frame の BoardState (= state machine 更新後)。
        frame_idx: 現 frame の index。
        time_sec: 現 frame の時刻 (秒)。
        chain_event: 現 frame で検出された ChainEvent (= None なら無し)。
        confirmed_board: 現 frame の confirmed_board (STABLE 時のみ非 None)。

    Returns:
        alert の tuple リスト。 通常は空。
    """
    alerts: list[tuple] = []
    # STABLE → NON-STABLE 遷移
    if prev_state == BoardState.STABLE and curr_state != BoardState.STABLE:
        if confirmed_board is not None:
            monitor.on_stable_end(frame_idx, confirmed_board)
        elif hasattr(monitor, "_last_stable_board") and monitor._last_stable_board is not None:
            # confirmed_board が None でも既存スナップショットを維持 (= 上書きしない)
            pass
    # NON-STABLE 中の chain イベントを記録
    if curr_state != BoardState.STABLE and chain_event is not None:
        monitor.on_non_stable_event("chain_start", frame_idx, time_sec)
    # NON-STABLE → STABLE 遷移
    if prev_state != BoardState.STABLE and curr_state == BoardState.STABLE:
        if confirmed_board is not None:
            drop_alerts = monitor.on_stable_start(frame_idx, time_sec, confirmed_board)
            for a in drop_alerts:
                alerts.append(a.to_tuple())
    return alerts


def _update_tier1_warmup_counter(
    prev_state: "BoardState",
    p_state: "BoardState",
    remaining: int,
) -> int:
    """tier1 warmup カウンタを更新して新しい残余 frame 数を返す。

    NON-STABLE → STABLE 遷移を検知したら TIER1_WARMUP_FRAMES をセット。
    STABLE 継続中はデクリメント (0 未満にはしない)。
    STABLE 以外への遷移は 0 にリセット (= 次の NON-STABLE→STABLE 待ち)。

    Args:
        prev_state: 現 frame の _step_side 後の state machine state。
            ※ update() では _step_side 後に呼ぶため「現フレームの state」 を渡す。
        p_state: SideResult.state (= _step_side が返した現フレームの state)。
        remaining: 現在の warmup 残余 frame 数。

    Returns:
        更新後の warmup 残余 frame 数 (0 以上)。
    """
    # NON-STABLE → STABLE 遷移: warmup 開始
    if (
        prev_state in NON_STABLE_STATES
        and p_state == BoardState.STABLE
    ):
        return TIER1_WARMUP_FRAMES
    # STABLE 継続: デクリメント
    if p_state == BoardState.STABLE and remaining > 0:
        return remaining - 1
    # STABLE 以外: リセット
    if p_state != BoardState.STABLE:
        return 0
    return remaining


def _update_ojama_tier1_warmup_counter(
    prev_state: "BoardState",
    p_state: "BoardState",
    remaining: int,
) -> int:
    """経路 A': OJAMA_FALL → STABLE 遷移専用の tier1 warmup カウンタ更新。

    OJAMA_FALL → STABLE 遷移のみ OJAMA_TIER1_WARMUP_FRAMES をセット。
    汎用 tier1 warmup (全 NON_STABLE_STATES) と異なり、OJAMA_FALL 起因の
    セル背景化による列崩壊だけを対処する (v51m2 退行を回避するため分離)。

    Args:
        prev_state: _step_side 呼び出し前の state (= 前フレームの state)。
        p_state: _step_side が返した現フレームの state。
        remaining: 現在の warmup 残余 frame 数。

    Returns:
        更新後の warmup 残余 frame 数 (0 以上)。
    """
    # OJAMA_FALL → STABLE 遷移: ojama 専用 warmup 開始
    if (
        prev_state == BoardState.OJAMA_FALL
        and p_state == BoardState.STABLE
    ):
        return OJAMA_TIER1_WARMUP_FRAMES
    # STABLE 継続: デクリメント
    if p_state == BoardState.STABLE and remaining > 0:
        return remaining - 1
    # STABLE 以外: リセット
    if p_state != BoardState.STABLE:
        return 0
    return remaining


def _apply_piece_persistence_guard(
    guard: "object | None",  # PiecePersistenceGuard | None (型循環回避)
    prev_state: "BoardState",
    curr_state: "BoardState",
    published_confirmed: "Board | None",
) -> "Board | None":
    """B1 PiecePersistenceGuard: 状態遷移に応じてフックを呼び保護後盤面を返す。

    遷移パターン:
        STABLE → NON-STABLE: on_non_stable_enter (= 保護リセット)
        STABLE 中 (confirmed_board あり): on_stable_confirmed + guard 適用

    Args:
        guard: PiecePersistenceGuard インスタンス (None なら無効)。
        prev_state: 前 frame の BoardState。
        curr_state: 現 frame の BoardState。
        published_confirmed: 現 frame の確定盤面 (None なら STABLE 以外)。

    Returns:
        guard 適用後の confirmed_board。guard=None or confirmed=None ならそのまま。
    """
    if guard is None:
        return published_confirmed
    # STABLE → NON-STABLE: 保護リセット
    if (
        prev_state == BoardState.STABLE
        and curr_state in NON_STABLE_STATES
    ):
        guard.on_non_stable_enter()
    # STABLE 中で confirmed あり: 保護登録 + guard 適用
    if curr_state == BoardState.STABLE and published_confirmed is not None:
        guard.on_stable_confirmed(published_confirmed)
        return guard.guard(published_confirmed)
    return published_confirmed


def _progressed_chain_board(
    chain_result: "ChainResult",
    trigger_sec: float,
    end_sec: float,
    time_sec: float,
) -> "Board | None":
    """反復5 Step2: 経過時刻に応じた連鎖進行中の推定盤面を返す (stateless)。

    src/inference_board.py の InferenceBoardGenerator._chain_board_at と
    同じ時刻→段数マッピングを用いる (別ロジック化による退行防止のため
    同一の計算式を採用、inference_board.py 自体は変更しない)。

    Args:
        chain_result: ChainSimulator.simulate(起点盤面) の結果。
        trigger_sec: 連鎖開始時刻。
        end_sec: 連鎖終了予定時刻 (表示ホールド込み)。
        time_sec: 現フレームの時刻。

    Returns:
        経過段数に対応する盤面。chain_count==0 なら None。
    """
    n_steps = chain_result.chain_count
    if n_steps == 0:
        return None
    duration = max(0.001, end_sec - trigger_sec)
    progress = max(0.0, min(1.0, (time_sec - trigger_sec) / duration))
    idx = int(progress * n_steps)
    if idx >= n_steps:
        return chain_result.final_board
    return chain_result.steps[idx].board_after


def _is_game_event_chain_exit(
    current_next: "tuple[int, int] | None",
    start_next: "tuple[int, int] | None",
) -> bool:
    """game-event ベース連鎖終了条件を判定する (stateless)。

    連鎖終了を示す game-event を検知したら True を返す:
      ① 次ツモ変化: current_next != start_next (どちらも有効色の場合のみ)

    ※②お邪魔信号は撤去済 (2026-06-01):
      confirmed 凍結が連鎖終了後に「既存お邪魔に追いつく」だけで新規落下と
      誤認し、短連鎖を 0.1 秒で早期終了させていた問題を解消。
      NextDetector は精度 100% が確認済であり、①のみで十分。

    Args:
        current_next: 現在の next_pair (None = 未検知)
        start_next: CHAIN 開始時の next_pair スナップショット (None = 未記録)

    Returns:
        True = 連鎖終了 game-event 検知、 False = 維持
    """
    # ① 次ツモ変化検知: start_next が記録済 かつ current_next が変化した場合のみ。
    # slide_motion や None の場合は誤検知防止のためスキップ。
    if (
        start_next is not None
        and current_next is not None
        and current_next != start_next
    ):
        return True

    return False


def _should_suppress_game_event_exit(
    time_sec: float,
    chain_entry_t: float,
    chain_count: int,
    chain_min_display_sec: float,
    chain_game_event_min_count: int,
) -> bool:
    """X1/X4 ガード: game-event exit を抑止すべきか判定する (stateless)。

    X1: CHAIN 突入から chain_min_display_sec 以内は exit 抑止。
    X4: chain_count < chain_game_event_min_count の短連鎖は exit 抑止 (timing hold のみ)。

    Args:
        time_sec: 現在の動画内時刻 (秒)。
        chain_entry_t: CHAIN 突入時の time_sec (= ChainEvent 受信時刻)。
        chain_count: 連鎖数。
        chain_min_display_sec: X1 の最小表示時間 (秒)。
        chain_game_event_min_count: X4 の最小連鎖数 (この数未満は exit 抑止)。

    Returns:
        True = exit 抑止 (= CHAIN 状態維持)、 False = exit 許可。
    """
    # X1: 最小表示時間以内は exit を抑止する
    if time_sec - chain_entry_t < chain_min_display_sec:
        return True
    # X4: 短連鎖 (chain_count < min_count) は game-event exit を発動しない
    if chain_count < chain_game_event_min_count:
        return True
    return False
