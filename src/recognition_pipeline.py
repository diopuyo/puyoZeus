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
    DetectorSignals,
    NON_STABLE_STATES,
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
from pathlib import Path

from src.background_fingerprint import (
    BackgroundFingerprint, capture_robust_fingerprint,
    capture_patch_pair_robust,
)
from src.chain_detector import ChainEvent, VideoChainTracker
from src.drift_detector import DriftDetector, DriftResult
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION, ImageReader
from src.inference_board import InferenceBoardGenerator
from src.match_end_detector import MatchEndDetector
from src.match_state import MatchState, MatchStateDetector
from src.next_detector import NextDetector
from src.online_hsv_calibrator import OnlineHsvCalibrator
from src.score_zero import ScoreZeroDetector
from src.telop_detector import TelopDetector
from src.next_slide_detector import (
    NextSlideDetector,
    SlideMotionResult,
    validate_tsumo_placement,
)
from src.placement_inferrer import (
    infer_placement, resolve_after_placement,
    correct_landing_cells_by_observed_color,
)
from src.score_ocr import ScoreOcr, ScoreTracker
from src.state_detectors import (
    ChainPhaseDetector,
    EffectPhaseDetector,
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

    # game-event ベース連鎖終了: CHAIN 状態を timing hold だけでなく
    # 「次ツモ出現 (next_pair 変化)」または「連鎖した側の盤面にお邪魔新規出現」
    # を検知するまで維持する。安全弁として以下の秒数を上限とする。
    # 60fps × 5.0s = 300 frame。通常の長い連鎖 (= 11 連鎖 × 0.3s = 3.3s) を
    # 十分にカバーし、かつ異常時 (event 永続不達) での CHAIN 永続化を防ぐ。
    CHAIN_MAX_HOLD_SEC: float = 5.0

    # X1: CHAIN 最小表示時間 (enable_chain_min_display=True 時に有効)。
    # 一度 CHAIN に入ったら最低この秒数は game-event 終了を抑止する。
    # 短連鎖 (1-2 連鎖 × 0.3s = 0.3-0.6s) がお邪魔信号で 0.1-0.2s で即終了し
    # 「一瞬表示/ちらつき」になる問題を防ぐ。
    CHAIN_MIN_DISPLAY_SEC: float = 0.8

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

    # 試合状態の hysteresis: 直前 N frame 内で 1P/2P がアクション中
    # (STABLE/TSUMO/CHAIN/OJAMA) なら、MatchStateDetector が NOT_IN_MATCH
    # を返しても is_match_active=True を強制。1 frame の単発誤判定を吸収し、
    # 「試合中に一瞬メニューに落ちる」現象を消す。物理的に試合中はメニュー
    # 状態にならない (ネットワーク切断など特殊状況除く) というユーザー仕様
    # に基づく。
    MATCH_ACTIVE_HOLD_FRAMES: int = 10

    # 試合開始から N frame 内は CHAIN 禁止: 試合 active 開始直後は puyo が
    # 増える時期で連鎖発生はあり得ない。VideoChainTracker が「メニュー画面
    # 0 個 → 試合開始直後 puyo 出現」を「連鎖発火 = 急減」と誤検出する
    # 現象を ban する。最初の 1 手目から CHAIN state に遷移するのを防ぐ。
    CHAIN_BAN_FRAMES_AFTER_MATCH_START: int = 30

    # cycle 71f (提案 A): score 動きで in_match 強制復帰判定の window と閾値.
    # SCORE_MOVE_WINDOW_FRAMES 内に SCORE_MOVE_MIN_DELTA 以上動いていれば
    # hard_match_off (= MatchEnd lockdown 等) を打ち消して試合中継続.
    # 60 frame = 1 秒. 連鎖発火 (= 100+ 点増加) でなくとも、 ツモ落下中の
    # +1 点増加が継続的にあれば検出可能.
    SCORE_MOVE_WINDOW_FRAMES: int = 60
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
    # cycle 4 (2026-05-15, F7): 0.4 → 0.3 に下げて landing_vote 補正速度↑.
    LANDING_VOTE_MIN_RATIO: float = 0.3  # 3 割で確定 (NEXT 色一致時)
    # cycle 26 (2026-05-18, A2): 着地直後 5 frame の CNN ぶれを除外。
    # この期間は raw CNN が着地フラッシュ・揺らぎで不安定なため vote 蓄積から除外。
    LANDING_VOTE_INIT_SKIP_FRAMES: int = 5
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

    def __init__(
        self,
        image_reader: ImageReader,
        match_state_detector: MatchStateDetector,
        score_ocr: ScoreOcr | None = None,
        chain_tracker_1p: VideoChainTracker | None = None,
        chain_tracker_2p: VideoChainTracker | None = None,
        stable_frame_count: int = 6,
        chain_hold_per_step_sec: float | None = None,
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
        # デフォルト False = 従来挙動完全維持 (backwards compat)。
        enable_landing_observed_color: bool = False,
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
        self._landing_grace_1p: tuple[int, Board] | None = None
        self._landing_grace_2p: tuple[int, Board] | None = None
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
        # match active hysteresis 用
        self._last_active_frame_idx: int = -1
        self._match_active_started_frame: int = -1
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
        # cycle 31 (2026-05-18, B 軸): baseline 整合性 check + 自己修復。
        # STABLE 中に baseline と CNN 出力の diff が連続異常なら baseline 壊れ
        # 判定 → state reset (= 試合 active 再起動 + bg_fp 再採取)。
        # v97 53 秒 TSUMO_FALL 詰まり問題への救済策。
        self._baseline_broken_consec_1p: int = 0
        self._baseline_broken_consec_2p: int = 0
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
        # 1P/2P state machine (独立)
        # B1 (M1 warmup guard): enable_warmup_guard=True で STABLE 遷移直後 N frame confirmed 凍結。
        # フェーズ A 精緻化: OjamaVisualDetector 登録フラグを伝播。
        self._sm_1p = self._build_state_machine(
            stable_frame_count, enable_warmup_guard=enable_warmup_guard,
            enable_stable_recovery_gate=enable_stable_recovery_gate,
            enable_ojama_visual_detection=self._enable_ojama_visual_detection,
            enable_ojama_visual_chain_exit=self._enable_ojama_visual_chain_exit,
            enable_ojama_settle_detection=self._enable_ojama_settle_detection,
        )
        self._sm_2p = self._build_state_machine(
            stable_frame_count, enable_warmup_guard=enable_warmup_guard,
            enable_stable_recovery_gate=enable_stable_recovery_gate,
            enable_ojama_visual_detection=self._enable_ojama_visual_detection,
            enable_ojama_visual_chain_exit=self._enable_ojama_visual_chain_exit,
            enable_ojama_settle_detection=self._enable_ojama_settle_detection,
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
    ) -> BoardStateMachine:
        # cycle 49 (2026-05-20): ChainPhaseDetector に ChainSimulator を注入。
        # 前 STABLE 盤面に 4 連結がない場合の chain 偽遷移を拒否する gate を有効化。
        # B1 (M1 warmup guard): enable_warmup_guard=True で STABLE 遷移直後 N frame
        # confirmed 更新を凍結する (A/B 対照実験用 optional 機能)。
        # 設計C 事後復旧ゲート: enable_stable_recovery_gate=True で BoardStateMachine に伝播。
        # フェーズ A 精緻化: enable_ojama_visual_detection=True で OjamaVisualDetector を
        # OjamaPhaseDetector の前 (優先順 3) に挿入する。score 差分ベース fallback は維持。
        from src.chain import ChainSimulator
        from src.board_state_machine import STABLE_WARMUP_FRAMES
        # ChainPhaseDetector に chain_ojama_exit フラグを伝播する
        chain_det = ChainPhaseDetector(
            chain_sim=ChainSimulator(),
            enable_chain_ojama_exit=enable_ojama_visual_chain_exit,
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
            )
            detectors.append(ovd)
        detectors.append(OjamaPhaseDetector())
        detectors.append(TsumoPhaseDetector())
        return BoardStateMachine(
            detectors=detectors,
            stable_frame_count=stable_n,
            enable_warmup_guard=enable_warmup_guard,
            warmup_frames=STABLE_WARMUP_FRAMES,
            enable_stable_recovery_gate=enable_stable_recovery_gate,
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
        enable_landing_observed_color: bool = False,
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
                score = ScoreOcr.load_default()
            except FileNotFoundError:
                score = None
        ctracker_1p = VideoChainTracker() if enable_chain_tracker else None
        ctracker_2p = VideoChainTracker() if enable_chain_tracker else None
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
            enable_red_hue_wrap_fix=enable_red_hue_wrap_fix,
            enable_specular_robust_saturation=enable_specular_robust_saturation,
            enable_stable_recovery_gate=enable_stable_recovery_gate,
            enable_ojama_visual_detection=enable_ojama_visual_detection,
            enable_ojama_visual_chain_exit=enable_ojama_visual_chain_exit,
            enable_ojama_infer_guard=enable_ojama_infer_guard,
            enable_ojama_settle_detection=enable_ojama_settle_detection,
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
        self._cnn_history_1p.clear()
        self._cnn_history_2p.clear()
        self._last_active_frame_idx = -1
        self._match_active_started_frame = -1
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
        # chain tracker 内部 state は再構築 (リセット API なし)
        if self._chain_tracker_1p is not None:
            self._chain_tracker_1p = VideoChainTracker()
        if self._chain_tracker_2p is not None:
            self._chain_tracker_2p = VideoChainTracker()
        # cycle 71d (案 D8): VideoChainTracker 入力 cache もリセット.
        self._prev_confirmed_1p = None
        self._prev_confirmed_2p = None
        # cycle 71f (提案 A): score 履歴もリセット.
        self._recent_scores_1p = []
        self._recent_scores_2p = []
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
        if self._score_zero_detector is not None:
            try:
                sz = self._score_zero_detector.detect(frame)
                score_zero_both = bool(sz.both_zero)
            except Exception:
                pass
        match_end_locked = False
        if self._match_end_detector is not None:
            try:
                match_end_locked = bool(
                    self._match_end_detector.update(frame, time_sec),
                )
            except Exception:
                pass
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
        if len(self._recent_scores_1p) > self.SCORE_MOVE_WINDOW_FRAMES:
            self._recent_scores_1p = self._recent_scores_1p[
                -self.SCORE_MOVE_WINDOW_FRAMES:
            ]
        if len(self._recent_scores_2p) > self.SCORE_MOVE_WINDOW_FRAMES:
            self._recent_scores_2p = self._recent_scores_2p[
                -self.SCORE_MOVE_WINDOW_FRAMES:
            ]
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
        self._last_telop_visible = False
        if self._telop_detector is not None:
            try:
                self._last_telop_visible = bool(
                    self._telop_detector.is_visible(frame),
                )
            except Exception:
                self._last_telop_visible = False
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
        # 直前 N frame 以内に active 観測歴があれば強制 True (1 frame ぶれ吸収)
        recent_active = (
            self._last_active_frame_idx >= 0
            and (frame_idx - self._last_active_frame_idx)
            <= self.MATCH_ACTIVE_HOLD_FRAMES
        )
        # 1P/2P state machine が現在 NON-STABLE state にある場合も active 強制
        # (= state machine 内部で active 認識中 → MENU に倒さない)
        sm_active = (
            self._sm_1p.context.state in (
                BoardState.STABLE, BoardState.TSUMO_FALL,
                BoardState.CHAIN, BoardState.OJAMA_FALL, BoardState.EFFECT,
            )
            or self._sm_2p.context.state in (
                BoardState.STABLE, BoardState.TSUMO_FALL,
                BoardState.CHAIN, BoardState.OJAMA_FALL, BoardState.EFFECT,
            )
        )
        # hard_match_off は hysteresis (recent/sm) を上書きする確定シグナル.
        # cycle 71f (提案 A): score が直近 window 内で SCORE_MOVE_MIN_DELTA 以上
        # 動いていれば、 hard_match_off を打ち消して試合中継続を保証する.
        # 「演出/READY/GO! で MatchEnd が誤発火するが score は動いている」
        # シナリオ (= v50 51-63s) を解消.
        effective_hard_off = hard_match_off and not score_actively_moving
        is_active = (
            (raw_active or recent_active or sm_active or score_actively_moving)
            and not effective_hard_off
        )

        # 試合 active 開始 frame の記録 (chain ban の起点)
        if is_active:
            if self._match_active_started_frame < 0:
                self._match_active_started_frame = frame_idx
            self._last_active_frame_idx = frame_idx
        else:
            # 試合 active が完全に切れたら start もリセット
            self._match_active_started_frame = -1
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
            match_age = frame_idx - self._match_active_started_frame
            bg_fp_relaxed = (
                match_age <= self.BG_FP_FORCE_WINDOW_FRAMES
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
        # 試合開始から CHAIN_BAN_FRAMES_AFTER_MATCH_START 以内の event は破棄
        # (1 手目から連鎖はあり得ない、誤検出 ban)。
        chain_banned = (
            self._match_active_started_frame >= 0
            and (frame_idx - self._match_active_started_frame)
            < self.CHAIN_BAN_FRAMES_AFTER_MATCH_START
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
                # 全消し連鎖は overlay 表示時間ぶん CHAIN を延長して、 CHAIN→STABLE
                # 遷移時の _merge_diff_only が overlay corrupted cnn_board を
                # 使わないようにする (v50 全消し overlay 誤認の構造的解消)。
                extra_all_clear = (
                    self.ALL_CLEAR_OVERLAY_HOLD_SEC if ev.is_all_clear else 0.0
                )
                self._chain_until_1p = (
                    time_sec
                    + self._chain_hold_per_step_sec * ev.chain_count
                    + extra_all_clear
                )
                # game-event モード: 安全弁上限 + 連鎖開始時 next_pair snapshot を記録。
                # enable_game_event_chain_exit=False 時も安全弁は無害 (使われない)。
                # ※②お邪魔信号撤去 (2026-06-01) により board snapshot は不要になった。
                if self._enable_game_event_chain_exit:
                    self._chain_event_max_until_1p = (
                        time_sec + self.CHAIN_MAX_HOLD_SEC
                    )
                    self._chain_start_next_1p = self._last_seen_next_1p
                # X1: CHAIN 突入時刻を記録 (enable_chain_min_display 用)。
                # enable_game_event_chain_exit 非依存で常に更新。
                self._chain_entry_t_1p = time_sec
        if is_active and self._chain_tracker_2p is not None:
            ev = self._chain_tracker_2p.update(time_sec, board_for_tracker_2p)
            if ev is not None and not chain_banned:
                self._active_chain_2p = ev
                extra_all_clear = (
                    self.ALL_CLEAR_OVERLAY_HOLD_SEC if ev.is_all_clear else 0.0
                )
                self._chain_until_2p = (
                    time_sec
                    + self._chain_hold_per_step_sec * ev.chain_count
                    + extra_all_clear
                )
                # game-event モード: 安全弁上限 + 連鎖開始時 next_pair snapshot を記録。
                # ※②お邪魔信号撤去 (2026-06-01) により board snapshot は不要になった。
                if self._enable_game_event_chain_exit:
                    self._chain_event_max_until_2p = (
                        time_sec + self.CHAIN_MAX_HOLD_SEC
                    )
                    self._chain_start_next_2p = self._last_seen_next_2p
                # X1: CHAIN 突入時刻を記録 (enable_chain_min_display 用)。
                # enable_game_event_chain_exit 非依存で常に更新。
                self._chain_entry_t_2p = time_sec

        # 有効期限内の chain_event を signals に乗せる
        # game-event モード (enable_game_event_chain_exit=True) の場合、
        # timing hold だけでは終了させず「最大 CHAIN_MAX_HOLD_SEC」まで維持する。
        # 実際の終了は 4b. next_pair 計算後に game-event チェックで行う。
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
                self._active_chain_1p = None
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
                self._active_chain_2p = None

        # 4. score 差分
        score_d_1p = self._update_score_tracker(self._score_tracker_1p, frame)
        score_d_2p = self._update_score_tracker(self._score_tracker_2p, frame)

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
                self._active_chain_1p = None
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
                self._active_chain_2p = None

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
                    sides_to_update = []
                    if (self._sm_1p.context.state == BoardState.STABLE
                            and self._sm_1p.context.confirmed_board is not None):
                        sides_to_update.append((
                            DEFAULT_P1_REGION,
                            self._sm_1p.context.confirmed_board,
                            self._sm_1p.context.state == BoardState.CHAIN,
                        ))
                    if (self._sm_2p.context.state == BoardState.STABLE
                            and self._sm_2p.context.confirmed_board is not None):
                        sides_to_update.append((
                            DEFAULT_P2_REGION,
                            self._sm_2p.context.confirmed_board,
                            self._sm_2p.context.state == BoardState.CHAIN,
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
    ) -> None:
        """cycle 71h: 着地時に vote 蓄積エントリを追加.

        prev_confirmed と final_board の差分 cells (= 着地で追加された cells) を
        抽出し、 期待色つきで vote_buffer を初期化する.

        cycle 71m (β2''): next_colors を保存し、 vote 期間中に HSV 距離で
        NEXT 色 2 種類のどちらかに分類する追加 vote も蓄積する.
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
        entry: dict = {
            "start": frame_idx,
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
        }
        if side == "1P":
            self._pending_landing_vote_1p.append(entry)
        else:
            self._pending_landing_vote_2p.append(entry)

    def _update_landing_votes(
        self, side: str, frame_idx: int,
        cnn_board: Board, confirmed_board: Board | None,
        frame_bgr: np.ndarray | None = None,  # cycle 71m β2''
    ) -> Board | None:
        """cycle 71h: 着地後 vote 累積 + 完了時の confirmed 更新.

        各 pending entry について:
        - LANDING_VOTE_FRAMES 経過前: cnn_board の対象 cells を vote_buffer に追加
        - LANDING_VOTE_FRAMES 経過: 最頻値で confirmed_board の cell 色を更新

        cycle 71m (β2''): frame_bgr が渡されれば、 各 cell の HSV を NEXT 色 2 種類
        への距離で分類する vote も並行で蓄積. 蓄積終了時、 NEXT 色 votes の多数決を
        優先採用 (= CNN 完全誤認時の救済).

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
        for entry in pending:
            elapsed = frame_idx - entry["start"]
            if elapsed < self.LANDING_VOTE_FRAMES:
                # cycle 26 (A2): 着地直後 5 frame は CNN ぶれ大 → 蓄積 skip
                in_init_skip = elapsed < self.LANDING_VOTE_INIT_SKIP_FRAMES
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
                            nc_obs_now = entry["next_color_votes"][(r, c)]
                            if (
                                updated_board is not None
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
                for (r, c, expected) in entry["cells"]:
                    # cycle 26 (A4): 早期確定済 cell は再適用 skip (上書き禁止)
                    if (r, c) in confirmed_set:
                        continue
                    # cycle 71m β2'': NEXT 色 votes が十分なら採用 (= 多数決優先)
                    # cycle 26 (A4): len>=3 のみ → ratio>=0.7 も必須化で誤分類抑制
                    nc_obs = entry.get("next_color_votes", {}).get((r, c), [])
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

    @classmethod
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
    ) -> int:
        """tracker があれば update、戻り値 delta (>=0 のみ)。"""
        if tracker is None:
            return 0
        d = tracker.update(frame)
        return max(0, d.delta) if d.is_valid else 0

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
        # 空フィールド強制するため state_machine に伝搬)
        match_just_started = (
            is_active
            and self._match_active_started_frame >= 0
            and (frame_idx - self._match_active_started_frame)
            < self.MATCH_JUST_STARTED_WINDOW_FRAMES
        )
        # 設計C 事後復旧ゲート用 HSV-only 盤面を取得する。
        # フラグ OFF または frame_bgr なし の場合は None (安全弁A により発火しない)。
        _hsv_board_for_signals: "Board | None" = None
        if (
            self._enable_stable_recovery_gate
            and frame_bgr is not None
            and sm.context.state == BoardState.STABLE
        ):
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
            inferred_landing = infer_placement(
                prev_confirmed, cnn_board, falling_pair,
                chain_sim=self._chain_sim,
                score_delta_observed=0,
                frame_bgr=frame_bgr,
                region=region_for_side,
                bg_fp=bg_fp_for_side,
                guard_empty_hallucination=self._enable_infer_empty_guard,
                enable_hsv_classify_fallback=self._enable_hsv_classify_fallback,
            )
            if inferred_landing is not None:
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
                    pseudo = ChainEvent(
                        trigger_sec=time_sec,
                        end_sec=time_sec + 0.3 * chain_count,
                        before_board=inferred_landing,
                        chain_count=chain_count,
                        total_erased=0, total_score=0, base_score=0,
                        all_clear_bonus_applied=0,
                        ojama_sent=0, leftover_score=0,
                        is_all_clear=False,
                    )
                    chain_until = (
                        time_sec
                        + self._chain_hold_per_step_sec * chain_count
                    )
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
        if (
            prev_state == BoardState.CHAIN
            and ctx.state == BoardState.STABLE
            and chain_event is not None
        ):
            try:
                from src.board_state_machine import _apply_gravity_filter
                from src.chain import ChainSimulator
                if not hasattr(self, "_chain_sim"):
                    self._chain_sim = ChainSimulator()  # type: ignore[attr-defined]
                cr = self._chain_sim.simulate(  # type: ignore[attr-defined]
                    chain_event.before_board,
                )
                if cr.chain_count > 0 and cr.final_board is not None:
                    final = cr.final_board.copy()
                    _apply_gravity_filter(final)
                    ctx.confirmed_board = final
                    ctx.pending_board = final.copy()
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
            if prev_confirmed is not None:
                self._start_landing_vote(
                    side, frame_idx, prev_confirmed, ctx.confirmed_board,
                    next_colors=falling_pair_for_grace,
                )
            grace_until = frame_idx + self.LANDING_GRACE_FRAMES
            if side == "1P":
                self._landing_grace_1p = (
                    grace_until, ctx.confirmed_board.copy(),
                )
                self._landing_pending_1p = None
            else:
                self._landing_grace_2p = (
                    grace_until, ctx.confirmed_board.copy(),
                )
                self._landing_pending_2p = None

        # cycle 31 (B 軸, 2026-05-18): baseline 整合性 check + 自己修復。
        # STABLE 中なのに baseline と CNN puyo 数 diff が連続異常な場合、
        # baseline 自体が壊れている (= 背景誤認込み等) と判定して reset。
        # state を MENU に戻して試合 active を再起動 → bg_fp 再採取。
        # v97 53 秒 TSUMO_FALL 詰まり問題への救済。
        if (
            ctx.state == BoardState.STABLE
            and ctx.confirmed_board is not None
            and is_active
        ):
            baseline_count = ctx.confirmed_board.count_puyos()
            cur_count = cnn_board.count_puyos()
            diff = cur_count - baseline_count
            BASELINE_BROKEN_DIFF_THRESHOLD = 8
            BASELINE_BROKEN_CONSEC_FRAMES = 60  # 1 秒
            consec_attr = (
                "_baseline_broken_consec_1p" if side == "1P"
                else "_baseline_broken_consec_2p"
            )
            if abs(diff) > BASELINE_BROKEN_DIFF_THRESHOLD:
                setattr(self, consec_attr, getattr(self, consec_attr) + 1)
                if getattr(self, consec_attr) >= BASELINE_BROKEN_CONSEC_FRAMES:
                    print(
                        f"[baseline-reset] {side} frame={frame_idx} "
                        f"baseline_count={baseline_count} "
                        f"cnn_count={cur_count} diff={diff} "
                        f"reset_after={getattr(self, consec_attr)} frames",
                    )
                    sm.reset(keep_match_state=False)
                    drift.reset()
                    gen.reset()
                    setattr(self, consec_attr, 0)
                    # bg_fp 再採取トリガー: image_reader の bg_fp を None に
                    if hasattr(self._reader, "set_background_fingerprints"):
                        if side == "1P":
                            self._reader.set_background_fingerprints(
                                None,
                                getattr(self._reader, "_bg_fp_p2", None),
                            )
                        else:
                            self._reader.set_background_fingerprints(
                                getattr(self._reader, "_bg_fp_p1", None),
                                None,
                            )
                    # 試合 active 再起動: _bg_fp_captured フラグも reset
                    if hasattr(self, "_bg_fp_captured"):
                        self._bg_fp_captured = False
                    # I1 対応 A: bg_fp 再採取中は pre_capture_mode を on に戻す
                    if hasattr(self._reader, "set_pre_capture_mode"):
                        self._reader.set_pre_capture_mode(True)
            else:
                setattr(self, consec_attr, 0)

        inferred = gen.generate(
            ctx, chain_event=chain_event, time_sec=time_sec,
        )
        drift_res = drift.update(inferred, cnn_board)
        if drift_res.needs_resync:
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
        in_grace = (
            grace_state_pre is not None
            and frame_idx < grace_state_pre[0]
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
        elif grace_state is not None and frame_idx >= grace_state[0]:
            # grace 終了: クリア
            if side == "1P":
                self._landing_grace_1p = None
            else:
                self._landing_grace_2p = None

        # cycle 71h: 着地後 vote refinement.
        # TSUMO_FALL→STABLE 着地時に登録された pending エントリの cnn 観測色を蓄積、
        # LANDING_VOTE_FRAMES 経過時に最頻値で confirmed_board の cell 色を更新.
        # 1 秒経過後の正しい色判別 (= ユーザー要件) を実現.
        # cycle 71m (β2''): frame_bgr を渡して HSV 距離分類 vote も並行蓄積.
        # cycle 26 (A1): grace 中は updated 反映を skip (蓄積は継続)。
        # grace 終了後の vote 完了で正しく反映される。
        if ctx.confirmed_board is not None:
            vote_updated = self._update_landing_votes(
                side, frame_idx, cnn_board, ctx.confirmed_board,
                frame_bgr=frame_bgr,
            )
            if vote_updated is not None and not in_grace:
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
                            if in_grace:
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
            # T2: STABLE → STABLE 遷移時の色変化検証。
            # 前 STABLE と現 STABLE で「色A → 色B」(異色間変化) かつ
            # 間に CHAIN signal がなければ認識誤りと判断し前値で上書き。
            # grace 中・chain_event あり・prev_stable なし は skip。
            prev_stable_t2 = (
                self._prev_stable_confirmed_1p if side == "1P"
                else self._prev_stable_confirmed_2p
            )
            if (
                prev_stable is not None
                and chain_event is None
                and not in_grace
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
        )


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
