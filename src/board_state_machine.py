"""BoardStateMachine — 盤面状態遷移管理 (Phase B-1).

ぷよぷよの 1 プレイヤー側盤面の状態 (STABLE / TSUMO_FALL / CHAIN /
OJAMA_FALL / EFFECT / MENU) を保持し、各状態で「認識を信用するか / 物理推論
を使うか」を切り替えるためのフレームワーク。

新方針 (project_recognition_strategy_pivot):
    - STABLE 状態のみ CNN 認識結果を盤面確定に使う。
    - それ以外の state ではアクション中なので CNN 出力は drift detector
      参照のみ。盤面確定は ChainSimulator / ネクスト履歴 / score OCR で行う。

このモジュールは骨組み (state enum + StateContext + 遷移ルール) のみ提供する。
各 detector の実装は Phase B-2 で `StateTransitionDetector` Protocol を満たす
形で個別に追加する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, Board

# ============================
# 定数
# ============================

# 平常時 = 連続 N frame で同一盤面が観測されたら STABLE 確定
DEFAULT_STABLE_FRAME_COUNT: int = 6

# F = STABLE 復帰ゲート (2026-05-21 cycle 56):
# NON-STABLE 中の cnn_board を最大 N frame 保持し、 STABLE 復帰時に
# 多数決を取って「瞬間的な背景誤認」 を抑制する。
# ユーザー指摘「置く直前の推論で背景がぷよ誤認、 2 秒残る」 への対策。
# 2026-05-23: G 拡張 (= N=8/votes=5) は v89m7 で逆効果 (= flicker +8%)、
# 8 動画でも c56_v3b 単独不採用と合わせ「baseline 維持」 判定 → revert。
DEFAULT_NON_STABLE_HISTORY_SIZE: int = 5
# EMPTY → 色 遷移 cell を採用するための NON-STABLE 中の最低観測 frame 数。
# N=5 中 3 frame 以上で同色観測 = 信頼。 それ以下は瞬間的誤認とみなし baseline 維持。
DEFAULT_EMPTY_TO_COLOR_MIN_VOTES: int = 3

# ネクスト裏付け確定 (2026-07-31)。直近に消費された next/dnext ペアの色に
# 一致する cell だけ、空→色の必要票数をこの値に緩める。
# 裏付けの無い色には DEFAULT_EMPTY_TO_COLOR_MIN_VOTES を維持する。
NEXT_CORROBORATED_MIN_VOTES: int = 1
# 裏付けに使う next_queue の末尾件数 (直近に消費されたペアのみを見る)。
# 大きくすると盤面上のあらゆる色が裏付け扱いになり緩和が無意味になる。
NEXT_CORROBORATION_QUEUE_TAIL: int = 2

# 初回STABLE確定の多数決ガード (enable_initial_confirm_vote, 2026-07-27):
# baseline is None (初回確定) 時、直前 NON-STABLE 滞在中に蓄積した
# non_stable_cnn_history の何 frame 以上一致すれば採用するか。
# EMPTY_TO_COLOR と同じ N=5/votes=3 の実績値を踏襲する。
DEFAULT_INITIAL_CONFIRM_MIN_VOTES: int = 3

# B1 (M1 warmup guard): STABLE 判定直後に confirmed 更新を N frame スキップ。
# NON-STABLE → STABLE 遷移直後は CNN 出力がまだ不安定 (= エフェクト残光、
# 背景誤認の溶け残り) であり、この期間に _update_within_current_state 経由で
# confirmed が更新されると誤色が焼き付く。
# WARMUP 中は STABLE state を維持したまま confirmed を凍結する。
# 0.4 秒 @ 30fps。
STABLE_WARMUP_FRAMES: int = 12

# 推論優先 (NON-STABLE) state の集合
NON_STABLE_STATES: frozenset["BoardState"] = frozenset()  # 後で初期化

# GRAVITY_SETTLE 状態定数 (feat/gravity-settle-2026-06-05)
# -------------------------------------------------------
# 連鎖終了直後の重力settle/着地中 window 用定数。
# GravitySettleDetector はこれらを参照して STABLE 復帰タイミングを制御する。
# -------------------------------------------------------

# ぷよ数変化が「ほぼ静止」と判定するための最小連続フレーム数。
# @30fps: 8f ≈ 0.27s。連鎖終了直後の board_log 分析で physics_fix が
# 遷移+0fに10-13件集中するため、それが収まるまで待つ設計。
GRAVITY_SETTLE_MIN_FRAMES: int = 8

# GRAVITY_SETTLE の最大保持時間 (秒)。タイムアウトで強制 STABLE 復帰。
# 最長連鎖でも settle は 0.5-1.0s 程度。1.5s は十分な安全マージン。
GRAVITY_SETTLE_MAX_SEC: float = 1.5

# 最低待機フレーム数 (physics_fix 解消確認用)。
# GRAVITY_SETTLE_MIN_FRAMES より小さい値を設定できる下限ガード。
# 連鎖エフェクト残光が消えるまでの最小保護時間。
GRAVITY_SETTLE_PHYSICS_CLEAR_MIN: int = 3

# フレーム定数→時間定数化 Stage1 (2026-07-25):
# GRAVITY_SETTLE_PHYSICS_CLEAR_MIN の秒定数版。GravitySettleDetector の
# 実ロジックはこちらを正として使う (frame 定数は互換のため残置)。
# 60fps 動画では (frame_idx 差分)/60 == time_sec 差分 が恒等式のため判定は
# bit-identical。30fps 動画では実秒基準になり、体感の遅延 (フレーム差分基準
# だと実質 2 倍の待ち時間になっていた) を解消する。
GRAVITY_SETTLE_PHYSICS_CLEAR_MIN_SEC: float = GRAVITY_SETTLE_PHYSICS_CLEAR_MIN / 60

# settle 中に「ぷよ数変化が安定」と見なすフレーム間差分の上限。
# raw CNN ぷよ数が連続フレームで ±N 以内なら静止とみなす。
GRAVITY_SETTLE_PUYO_DIFF_THRESHOLD: int = 2

# 設計C 事後復旧ゲート (fix/v70-zeropatch-redyellow, 2026-06-02):
# STABLE 中に confirmed != (CNN==HSV の合意値) が N フレーム継続したセルを
# confirmed に追従させる双方向ゲート。
#   方向1 (空→色): confirmed=EMPTY, CNN==HSV=有効色 → 色に復活 (幽霊ぷよ追加)
#   方向2 (色→空): confirmed=色,   CNN==HSV=EMPTY   → 空に修正 (幽霊ぷよ除去)
#   方向3 (色→色): confirmed=色A,  CNN==HSV=色B     → 色Bに訂正 (誤色修正)
# N フレーム = 8 frame (約 0.27s @ 30fps) で実質的な持続合意と判断。
STABLE_RECOVERY_MIN_FRAMES: int = 8

# 復旧ゲート方向別しきい値 (enable_asymmetric_recovery_min_frames, 2026-07-30):
# 方向1 (空→色: ぷよを追加する復旧) のみを短縮する非対称化。誤って追加しても
# 実ぷよが無ければ次の観測で消えるため相対的に安全。方向2/3 (色→空/色→色) は
# 誤消去・誤訂正が gravity filter で上のぷよまで連鎖消去し増幅する
# (2026-07-30 列デッドロック実測) ため STABLE_RECOVERY_MIN_FRAMES (8) を維持する。
# 「消す方が危険」という非対称を一貫させる (色→空 HSV 照合ガードと同じ考え方)。
# default OFF (enable_asymmetric_recovery_min_frames) では未使用 (backwards compat)。
STABLE_RECOVERY_ADD_MIN_FRAMES: int = 4

# 復旧ゲートの「合意値」として無効な色 = UNKNOWN(10) のみ。
# EMPTY(0) / OJAMA(9) は方向2の除去ターゲットなので含めない。
RECOVERY_EXCLUDED_COLORS: frozenset[int] = frozenset({10})

# 列ゲート緩和 (enable_column_partial_support, 2026-07-25):
# _check_recovery_column の浮き判定で「下のセルが空」でも stable_recovery_counters
# が一定フレーム以上進行中 (= 復旧合意が積み上がりつつある) なら浮き扱いしない
# ための下限フレーム数。STABLE_RECOVERY_MIN_FRAMES (8) より緩い値にすることで
# 「本線側が先に 8f 到達し、支持側がまだ 8f 未満」の組合せを救済する。
# default False (enable_column_partial_support) では未使用 (backwards compat)。
RECOVERY_COLUMN_SUPPORT_MIN_FRAMES: int = 2

# 復旧カウンタ carryover (enable_recovery_counter_carryover, 2026-07-26):
# STABLE → NON-STABLE 遷移時に stable_recovery_counters/recovery_cells を
# 即クリアせず、非 STABLE 滞在がこの秒数以内なら STABLE 復帰後もカウンタを
# 引き継ぐ。連鎖等で長時間 NON-STABLE のままだと盤面が実際に変化して
# いるため、この秒数を超えたら安全側にクリアする
# (実例: 8f 到達直前の counter=4-5 が NON-STABLE 遷移で毎回消え未反映化する
# 問題への対処、 diag `recovery_cell_timeseries_2026-07-25`)。
# default False (enable_recovery_counter_carryover) では未使用 (backwards compat)。
RECOVERY_COUNTER_CARRYOVER_MAX_SEC: float = 2.0

# CNN 乱高下セル HSV フォールバック (enable_cnn_flicker_hsv_fallback, #51 後半,
# 2026-07-26): 深部セルで CNN が光沢ハイライトにより判定境界に張り付き、
# フレーム毎に出力が反転 (9↔1↔0↔4) する事象への対策。復旧ゲートは
# 「CNN==HSV の合意」が min_frames 連続することを要求するが、CNN 自体が
# 毎フレーム反転していると合意が一度も成立せずカウンタが毎回リセットされ、
# セルが長時間 (実測 14 秒) 未反映のまま残る。
# 直近 CNN_FLICKER_WINDOW_FRAMES フレームの CNN 出力の変化回数が
# CNN_FLICKER_MIN_CHANGES 以上のセルは「乱高下中」とみなし、その間は
# HSV 出力を合意値とみなして扱う (= HSV を信頼する fallback)。
# default False (enable_cnn_flicker_hsv_fallback) では未使用 (backwards compat)。
CNN_FLICKER_WINDOW_FRAMES: int = 8
CNN_FLICKER_MIN_CHANGES: int = 3

# エフェクト時間ゲート (enable_effect_gate, 2026-08-03):
# 満杯盤面 47 セル誤りの真因確定 (memory
# `project_full_board_error_taxonomy_2026-08-02`)。相手の連鎖 1 リンクごとに
# 約 0.2 秒発生する「予告おじゃま送付エフェクト」+ お邪魔落下時の煙が、自盤面
# 上段 row1-3 の色→空/空→色/色→色ちらつきとして _apply_stable_recovery_gate
# 経由で confirmed_board に混入する。既存の復旧ゲート (STABLE_RECOVERY_MIN_FRAMES
# =8 frame ≈0.27s@30fps) はフレーム数ベースのため 60fps 動画では 0.2 秒のエフェクト
# (=12 frame) が閾値を超えて通過してしまう (フレーム定数の fps 依存問題、
# `GRAVITY_SETTLE_PHYSICS_CLEAR_MIN_SEC` と同種)。
# 本ゲートは「領域限定 (上段 row1-3) + 持続確認 (実秒 EFFECT_PERSIST_SEC)」を
# time_sec ベースで行う (fps 非依存)。相手が連鎖中 / 自身がお邪魔着弾直後の
# window 中のみ発動し、それ以外の cell・時間帯は従来の frame ベース復旧ゲートを
# 一切変更しない。エフェクト中もツモを振る可能性があるため全面凍結はせず、
# 領域 (上段のみ) と窓 (エフェクト発生し得る期間のみ) を限定する
# (user 承認済み設計方針)。
# default False (enable_effect_gate) = 従来挙動完全維持 (backwards compat)。
EFFECT_GATE_TOP_ROWS: "frozenset[int]" = frozenset({1, 2, 3})
EFFECT_PERSIST_SEC: float = 0.4

# 盤面確定窓 3中2多数決 (stable_majority_window, 2026-08-13 user承認):
# 現行の初回STABLE確定窓 (`_update_within_current_state` の pending_count/
# pending_board、stable_frame_count=3 連続で厳密一致) は 1 フレームのノイズ
# 混入で連続カウンタが 1 に戻り、2値交互ノイズには原理的に無限に弱い
# (実測: scripts/_measure_stable_window_restart_2026-08-13.py、
# logs/stable_window_restart_measure_2026-08-13.json。確定の 8.3% で延び
# 発生・最悪 9.05 秒、3中2多数決の反実仮想で超過時間 -99.6%)。
# 本フラグ ON 時は直近 STABLE_MAJORITY_WINDOW_FRAMES 観測 (raw cnn_board) の
# うち STABLE_MAJORITY_MIN_VOTES 以上一致した盤面を、その場で確定候補として
# 採用する (`_majority_window_vote` 参照)。認識精度の物差し (99.5%基準)
# 回帰検証を条件に採用確定 (2026-08-13)。
# default False (stable_majority_window) = 従来の厳密連続一致を完全維持
# (backwards compat、148動画収集走行中のため既定OFF必須)。
STABLE_MAJORITY_WINDOW_FRAMES: int = 3
STABLE_MAJORITY_MIN_VOTES: int = 2


class BoardState(Enum):
    """1 プレイヤー側盤面の状態。"""

    MENU = "menu"            # 試合外 (タイトル / リザルト / リトライ)
    STABLE = "stable"        # 平常時、CNN 出力を盤面確定に使う
    TSUMO_FALL = "tsumo_fall"  # ツモ落下中
    CHAIN = "chain"          # 連鎖中 (消去 + 重力)
    OJAMA_FALL = "ojama_fall"  # おじゃま落下中
    EFFECT = "effect"        # 全消し演出 / 連鎖カットイン等
    # feat/gravity-settle-2026-06-05: 連鎖終了直後の重力 settle/着地中。
    # CHAIN 終了後から board が物理的に静止するまでの window。
    # NON_STABLE_STATES に含まれるため採点外・confirmed 凍結対象。
    # backwards compat: 既存コードは GRAVITY_SETTLE を認識しないが、
    # NON_STABLE_STATES に加えることで「採点外」として既存の is_action() が True を返し、
    # 挙動上は他の NON-STABLE state と同等に扱われる。
    GRAVITY_SETTLE = "gravity_settle"


# 推論優先 state (= 認識結果を盤面確定に使わない state)
NON_STABLE_STATES = frozenset({
    BoardState.TSUMO_FALL,
    BoardState.CHAIN,
    BoardState.OJAMA_FALL,
    BoardState.EFFECT,
    # feat/gravity-settle-2026-06-05: GRAVITY_SETTLE は採点外 (confirmed 凍結)。
    # backwards compat: default OFF (enable_gravity_settle_state=False) なら
    # この state に遷移しないため挙動は完全不変。
    BoardState.GRAVITY_SETTLE,
})


# ============================
# データクラス
# ============================


@dataclass
class StateContext:
    """1 プレイヤー側 state machine の状態スナップショット。

    Attributes:
        state: 現在の BoardState。
        frame_idx: 入力 frame 番号 (連番)。
        time_sec: 現在 frame の時刻。
        confirmed_board: 直近 STABLE 確定盤面 (None なら未確定)。
        last_stable_idx: confirmed_board の frame_idx。
        pending_board: STABLE 候補だがまだ多数決を満たしていない盤面。
        pending_count: pending_board の連続観測 frame 数。
        next_queue: 直近のネクスト履歴 (ペア色)。新しいほど末尾。
        chain_count: CHAIN state 時の連鎖カウンタ (0 なら CHAIN 外)。
        ojama_pending: OJAMA_FALL state 時の落下予定個数。
    """

    state: BoardState = BoardState.MENU
    frame_idx: int = 0
    time_sec: float = 0.0
    confirmed_board: Board | None = None
    last_stable_idx: int = -1
    pending_board: Board | None = None
    pending_count: int = 0
    next_queue: list[tuple[int, int]] = field(default_factory=list)
    chain_count: int = 0
    ojama_pending: int = 0
    # F (cycle 56): NON-STABLE 中の cnn_board 履歴。 STABLE 復帰時に多数決を取り、
    # 瞬間的な背景誤認が confirmed_board に焼き付くのを防ぐ。 最大 N frame 保持。
    non_stable_cnn_history: list[Board] = field(default_factory=list)
    # B1 (M1 warmup guard): NON-STABLE → STABLE 遷移直後の confirmed 凍結残りフレーム数。
    # 0 = warmup なし (通常 STABLE)。0 より大きければ confirmed 更新をスキップ。
    # backwards compat: default=0 で既存動作と同一。
    stable_warmup_remaining: int = 0
    # 設計C 事後復旧ゲート (2026-06-02):
    # STABLE 中に confirmed==EMPTY かつ CNN==HSV が同一有効色で連続している
    # フレーム数を (row, col) → int で管理するカウンタ。
    # STABLE→NON-STABLE 遷移時にクリアする。
    # backwards compat: default=empty dict で既存動作と完全同一。
    stable_recovery_counters: "dict[tuple[int,int],int]" = field(
        default_factory=dict,
    )
    # 設計C: 復旧済みセルの集合。STABLE→NON-STABLE 遷移時にクリア。
    # 復旧したセルは non_stable_cnn_history へ復旧色を強制投入し
    # 次の NON-STABLE→STABLE 多数決で再崩壊しにくくする。
    recovery_cells: "set[tuple[int,int]]" = field(
        default_factory=set,
    )
    # 復旧カウンタ carryover (2026-07-26): 直近の STABLE→NON-STABLE 遷移の
    # time_sec。NON-STABLE 滞在中のみ値を持ち、STABLE 復帰時や carryover
    # 機能無効時は None。 backwards compat: default None で既存動作と完全同一。
    non_stable_entry_time_sec: float | None = None
    # CNN 乱高下セル HSV フォールバック (#51 後半, 2026-07-26):
    # (row, col) → 直近 CNN 出力の履歴 (最新が末尾)。
    # enable_cnn_flicker_hsv_fallback=True の場合のみ更新される。
    # stable_recovery_counters と同じタイミングでクリアする。
    # backwards compat: default={} で既存動作と完全同一。
    cnn_flicker_history: "dict[tuple[int, int], list[int]]" = field(
        default_factory=dict,
    )
    # エフェクト時間ゲート (2026-08-03): 領域限定セルの実秒ベース持続観測。
    # (row, col) -> (候補色, 最初に観測した time_sec)。候補色が変わったら
    # リセットする (= 別の色に切り替わったら再度 EFFECT_PERSIST_SEC 分の
    # 持続を要求)。stable_recovery_counters (フレーム数ベース) とは独立に
    # 管理し、単位混在を避ける。
    # backwards compat: default={} で既存動作と完全同一 (enable_effect_gate=False
    # の間は一切書き込まれない)。
    effect_gate_hold: "dict[tuple[int, int], tuple[int, float]]" = field(
        default_factory=dict,
    )
    # 盤面確定窓 3中2多数決 (stable_majority_window, 2026-08-13): 直近
    # STABLE_MAJORITY_WINDOW_FRAMES 分の raw cnn_board 履歴 (最新が末尾)。
    # `_majority_window_vote` の入力。state 遷移 (`_apply_transition`) の
    # pending リセットに合わせてクリアする。
    # backwards compat: stable_majority_window=False (default) の間は
    # 一切書き込まれない (常に空リストのまま)。
    confirm_window_history: "list[Board]" = field(default_factory=list)

    def is_stable(self) -> bool:
        """STABLE 確定中か (= 認識結果を盤面確定に使う)。"""
        return self.state == BoardState.STABLE

    def is_action(self) -> bool:
        """アクション中か (= NON-STABLE)。"""
        return self.state in NON_STABLE_STATES


# ============================
# Detector Protocol (B-2 で具体実装を差し込む)
# ============================


@dataclass(frozen=True)
class DetectorSignals:
    """1 frame 分の入力シグナル。各 detector はこの dataclass を読み取る。

    Attributes:
        time_sec: 現 frame の時刻。
        cnn_board: CNN 認識結果の生盤面 (drift detector 入力にも使う)。
        is_match_active: match_state.py の試合中判定。
        chain_event: VideoChainTracker から渡される連鎖イベント (確定時のみ)。
        chain_counter_visible: 画面上の ★n連鎖★ が検出されたか (B-3 で実装)。
        score_delta: 直前 STABLE からの score 差分 (B-4 で実装)。
        next_pair: 直近で確定した next ペア (B-2 で next_detector hookup)。
        slide_motion: NEXT ROI のスライド motion (= ツモが画面から消えて
            次のツモに繰り上がった signal)。Phase I R-7。default False で
            backward compat。
        placement_validated: 着地後の cnn_board と baseline の色 count delta が
            落下ペアと整合 (= 自己整合性チェック合格) しているか。
            Phase I R-1。default False で backward compat。
    """

    time_sec: float
    cnn_board: Board
    is_match_active: bool
    chain_event: object | None = None
    chain_counter_visible: bool = False
    score_delta: int = 0
    next_pair: tuple[int, int] | None = None
    slide_motion: bool = False
    placement_validated: bool = False
    # 設計C 事後復旧ゲート用 HSV-only 盤面 (2026-06-02)。
    # ImageReader の HSV 単独分類結果を渡し、CNN との独立二重合意チェックに使う。
    # None なら復旧ゲートの「CNN==HSV 一致」要件が満たせないため発火しない。
    # backwards compat: default None で既存挙動と完全同一。
    hsv_board: "Board | None" = None
    # 演出中フラグ (telop/match_end/all_clear/win_panel/chain_animation の OR)。
    # True なら EffectPhaseDetector が EFFECT state に遷移し、 CNN 出力を
    # 信用せず直前 STABLE 盤面を hold する (memory `feedback_chain_phase_physics_only`)。
    effect_visible: bool = False
    # 試合開始直後 window (cycle 71v, 2026-05-14). True の間は初回 STABLE 確定で
    # confirmed_board に空 Board() を強制し、 CNN の背景誤認による
    # 開始フィールド corruption を防ぐ。 ぷよぷよ eスポーツの物理ルール:
    # 「試合開始時のフィールドは必ず空」 を構造的に取り込んだ防御層。
    match_just_started: bool = False
    # フェーズ A 精緻化: 可視最上段付近の ROI にお邪魔が存在するか (一次判定)。
    # OjamaVisualDetector の内部カウンタ更新前に RecognitionPipeline 側で
    # _count_top_ojama() を呼び出してセットする。default False で既存挙動不変。
    # backwards compat: default False で既存動作と完全同一。
    ojama_top_positive: bool = False
    # 案P3: CHAIN_MAX_HOLD_SEC を超過して active_chain が強制クリアされたか。
    # True のとき ChainPhaseDetector は ojama_top_positive による STABLE 復帰保留を
    # 無視して強制 STABLE に遷移させる。これにより安全弁が本来機能する。
    # pipeline 側で active_chain が None にクリアされた瞬間の 1 frame だけ True になり、
    # 翌 frame 以降は False に戻る (新連鎖発火時 / STABLE 復帰確定時もリセット)。
    # backwards compat: default False で既存動作と完全同一。
    chain_max_hold_expired: bool = False
    # エフェクト時間ゲート (enable_effect_gate, 2026-08-03): 今フレームが
    # 「相手連鎖中」または「自身お邪魔着弾直後 window」であるか。
    # RecognitionPipeline 側で相手 side の ChainEvent 有無 / 自 side の
    # OJAMA_FALL→STABLE 遷移からの経過秒数を見て計算する
    # (state machine 自身は相手 side の情報を持たないため外部から注入する)。
    # True の間、 EFFECT_GATE_TOP_ROWS の cell は通常の frame ベース復旧ゲート
    # でなく実秒ベースの持続確認 (EFFECT_PERSIST_SEC) を経由する。
    # backwards compat: default False で既存動作と完全同一。
    effect_gate_window_active: bool = False
    # マージンタイム逓減用 (2026-08-09): **最初の1手 (最初のツモ設置)** からの
    # 経過秒。 おじゃまレートは一定時間後に 16 秒ごと ×0.75 で下がるため、
    # OjamaPhaseDetector の閾値をこの経過時間から算出する。
    # 起点を試合開始でなく最初の1手にするのは user 伝授 (試合開始時刻は演出が
    # あり実装で正確に取れないが、 最初のツモ設置は認識で確実に取れる)。
    # None = 未取得 (この場合は従来の固定レートへフォールバックし、
    # 推測で減衰させない)。 backwards compat: default None で既存動作と同一。
    elapsed_since_first_move_sec: float | None = None
    # 案2 (enable_ojama_fall_placement_override, 2026-08-13、OJAMA_FALL
    # 誤分類根因調査): 自 side (score_delta フィールドは相手側のため区別する)
    # の今フレーム score 増分 (常に 0 以上、RecognitionPipeline._step_side の
    # score_d_for_self をそのまま渡す)。 OJAMA_FALL 滞在中に自 score が動く
    # ことは通常無い (おじゃま受け側は連鎖しないため) ため、 増分があれば
    # 実設置イベント (落下ボーナス等) の証拠として使う。
    # backwards compat: default 0 で既存動作と完全同一。
    own_score_delta: int = 0
    # 案4-lite 拡張 (coordinator追加指示, 2026-08-13、OJAMA_FALL誤分類根因
    # 調査 場面2): 自 side の直近 chain hold 終了予定時刻
    # (RecognitionPipeline._chain_until_1p/_chain_until_2p をそのまま渡す。
    # 新規の状態追跡を増やさず既存の追跡値を再利用する)。 chain_event が
    # 現在アクティブなら time_sec 以降の未来時刻、 終了済みなら最後に更新
    # された時刻のまま残る (試合開始時は 0.0)。
    # OjamaVisualDetector が `signals.time_sec - own_chain_hold_until_sec` で
    # 「直近まで自 chain がアクティブだったか」を state 非依存に判定する
    # (ctx.state==CHAIN の瞬間条件だけでは chain_event 検出の瞬間空白で
    # state が既に CHAIN を離れた後の割り込みを捉えられないため)。
    # backwards compat: default 0.0 で既存動作と完全同一
    # (0.0 は「経過大 = 直近アクティブでない」と等価に働く)。
    own_chain_hold_until_sec: float = 0.0
    # 案1 (enable_ojama_fall_scoped_exit, 2026-08-13、OJAMA_FALL出口の根治):
    # 自 side の未着弾おじゃま予告量 (お邪魔会計トラッカー由来)。
    # RecognitionPipeline が会計トラッカーを有効化している構成でのみ算出して
    # 渡す (`_all_clear_pending` と同じ「外部で計算し signals に注入する」
    # パターン、 detector 側は getattr で安全参照する)。
    # None = 未計測/トラッカー無効構成を意味し、 OjamaVisualDetector は
    # 滞在短縮ロジックを一切適用しない (= 従来相当のフレーム数で判定)。
    # backwards compat: default None で既存動作と完全同一。
    own_pending_ojama_forecast: int | None = None


class StateTransitionDetector(Protocol):
    """state 遷移検出器の Protocol.

    各 detector は現 context + 入力 signals を見て、遷移先 state を返す。
    遷移しないなら None を返す。
    """

    def detect(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        ...


# ============================
# State Machine 本体
# ============================


def _boards_equal(a: Board | None, b: Board | None) -> bool:
    """2 つの Board が完全一致か (None 同士は False)。"""
    if a is None or b is None:
        return False
    return a == b


def _vote_majority_board(
    history: list[Board], min_votes: int,
    *, fallback: Board | None = None,
    corroborated_colors: "frozenset[int] | None" = None,
    corroborated_min_votes: int | None = None,
) -> Board:
    """history の各 cell について多数決を取り、 min_votes 以上同色なら採用、
    そうでなければ EMPTY (または fallback 値) を入れた Board を返す.

    F (cycle 56): STABLE 復帰時の「EMPTY → 色 遷移ガード」 用。
    NON-STABLE 中の cnn_board 履歴から「安定して観測されている色」 のみを抽出。

    Args:
        history: NON-STABLE 中の cnn_board 履歴 (= 最大 N frame)
        min_votes: cell を採用するための最低観測 frame 数
        fallback: 観測不足 cell (min_votes 未達) に採用する代替 board。
            None (default) なら従来通り EMPTY のまま (backwards compat、
            既存呼び出し元である F ガードは bit-identical)。
            初回STABLE確定の多数決ガード (enable_initial_confirm_vote,
            2026-07-27) では fallback=new_cnn を渡し、観測不足セルを
            問答無用で EMPTY 化する新規バグ (色→空誤凍結の増設) を防ぐ。
        corroborated_colors: **ネクスト裏付け色** (2026-07-31)。
            直近で消費された next/dnext ペアに含まれる色の集合。
            多数決の勝者色がこの集合に含まれる cell だけ、必要票数を
            corroborated_min_votes に緩める。
            None (default) なら一切緩めない = 従来と bit-identical。
        corroborated_min_votes: 裏付けありセルの必要票数 (通常 min_votes 未満)。

    ## なぜ「閾値を一律に下げる」のと違うか
    閾値の一律緩和は**情報量を減らして速くする**ので精度を売る。実測でも
    recovery_min_frames 8→4 で初期色不一致が 2→3 に増えた。
    こちらは「ピクセル以外の独立した情報 (ネクスト表示)」を足して
    **裏付けのある色にだけ**緩めるので、速度と精度が同時に上がりうる。
    裏付けの無い色 (= 誤読の疑いが濃い) には従来どおりフル票数を要求する。

    ## 弱い使い方にしている理由
    「どのネクストがどの設置に対応するか」の追跡は過去に信頼できなかった
    (キューに正解無し57%、memory project_color_flicker_p2_root_cause_2026-07-25)。
    そこで対応付けを真値として使わず、**「直近に消費されたペアの色に
    含まれるか」という集合メンバシップだけ**を裏付けに使う。
    対応付けが多少ずれても安全側に倒れる。

    Returns:
        多数決 board (= 観測不足 cell は EMPTY または fallback 値)
    """
    from src.board import COLOR_EMPTY

    result = Board()
    if not history:
        return fallback.copy() if fallback is not None else result
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            counter: dict[int, int] = {}
            for b in history:
                v = b.get(r, c)
                counter[v] = counter.get(v, 0) + 1
            if not counter:
                continue
            max_v, max_n = max(counter.items(), key=lambda x: x[1])
            # ネクスト裏付けのある色だけ必要票数を緩める (2026-07-31)。
            # 裏付けが無ければ従来どおり min_votes を要求する。
            need = min_votes
            if (
                corroborated_colors is not None
                and corroborated_min_votes is not None
                and max_v in corroborated_colors
            ):
                need = corroborated_min_votes
            if max_n >= need and max_v != COLOR_EMPTY:
                result.set(r, c, max_v)
            elif fallback is not None:
                result.set(r, c, fallback.get(r, c))
            # else: EMPTY (= Board default、 観測不足、fallback 未指定)
    return result


def _majority_window_vote(
    history: "list[Board]", min_votes: int,
) -> "Board | None":
    """直近 history (最大 window frame) 中で min_votes 以上一致する盤面を返す.

    盤面確定窓 3中2多数決 (stable_majority_window, 2026-08-13) のコア判定。
    厳密な N 連続一致 (pending_count 方式) は 1 フレームのノイズ混入で連続
    カウンタが 1 に戻ってしまい、2値交互ノイズには原理的に無限に弱い
    (実測: scripts/_measure_stable_window_restart_2026-08-13.py)。
    本関数は window 内の raw cnn_board を厳密一致 (`Board.grid_bytes()` を
    キーにした投票) で集計し、最多得票の盤面が min_votes 以上ならそれを
    返す (= 過半数でなくても「window 中で最も支持された盤面」を採用する)。

    Args:
        history: 直近 window frame 分の raw cnn_board (append 順、最新が末尾)。
        min_votes: 採用に必要な最低一致票数。

    Returns:
        多数決で選ばれた盤面 (票数最多)。history が min_votes 未満、または
        全 window 内で min_votes に達する盤面が無い場合は None
        (= 未確定、呼び出し側は従来通り継続観測する)。
    """
    if len(history) < min_votes:
        return None
    counts: "dict[bytes, tuple[Board, int]]" = {}
    for b in history:
        key = b.grid_bytes()
        prev = counts.get(key)
        if prev is None:
            counts[key] = (b, 1)
        else:
            counts[key] = (prev[0], prev[1] + 1)
    best_board, best_n = max(counts.values(), key=lambda item: item[1])
    if best_n >= min_votes:
        return best_board
    return None


def _build_initial_confirmed_board(
    new_cnn: Board,
    initial_confirm_history: list[Board] | None,
    initial_confirm_min_votes: int,
) -> Board:
    """初回STABLE確定 (baseline is None) の盤面を組み立てる (2026-07-27).

    _merge_diff_only の 50 行規約超過を避けるため分離した専用ヘルパー。
    history が十分にあれば多数決 (fallback=new_cnn で観測不足セルは
    単発 CNN 値を維持、EMPTY 新規誤化を防ぐ)、 history が無ければ
    従来通り new_cnn そのまま (backwards compat)。

    Args:
        new_cnn: 新たに観測された CNN 盤面 (単発フレーム)。
        initial_confirm_history: 直前 NON-STABLE 滞在中に蓄積した
            cnn_board 履歴。 None または空リストなら従来挙動。
        initial_confirm_min_votes: 多数決の最低観測 frame 数。

    Returns:
        初回 confirmed_board (浮きぷよ ban 適用済み)。
    """
    if initial_confirm_history:
        new = _vote_majority_board(
            initial_confirm_history, initial_confirm_min_votes,
            fallback=new_cnn,
        )
    else:
        new = new_cnn.copy()
    _apply_gravity_filter(new)
    return new


def _should_keep_puyo_over_empty(
    base_v: int, cnn_v: int, hsv_v: int | None,
    allow_puyo_to_empty: bool, enable_hsv_guard: bool,
) -> bool:
    """色→空 遷移で baseline の puyo を維持すべきか判定する (色→空 保護).

    従来の休眠ガード (allow_puyo_to_empty=False の blanket ban) に加え、
    enable_hsv_guard=True 時は HSV が同 cell に色を保持している場合のみ
    単一フレーム CNN の空誤読 (光沢→空) を退ける。CNN と HSV が共に空を
    読む本物の連鎖消去は退けない (両者一致時は False を返す)。

    Args:
        base_v: baseline (直前確定盤面) の色。
        cnn_v: 現フレーム CNN の色。
        hsv_v: 現フレーム HSV の色。None なら HSV 無し (照合ガード不発)。
        allow_puyo_to_empty: False で休眠 blanket ban を発動 (従来 dead code)。
        enable_hsv_guard: True で HSV 照合ガードを発動する。

    Returns:
        True なら消さず baseline の puyo を維持する。
    """
    from src.board import COLOR_UNKNOWN
    if base_v == COLOR_EMPTY or cnn_v != COLOR_EMPTY:
        return False  # そもそも色→空 遷移でない
    if not allow_puyo_to_empty:
        return True  # 休眠ガード (blanket ban、従来挙動)
    if (
        enable_hsv_guard
        and hsv_v is not None
        and hsv_v != COLOR_EMPTY
        and hsv_v != COLOR_UNKNOWN
    ):
        return True  # HSV が色を裏付ける → 単一フレーム CNN の空誤読を退ける
    return False


def _merge_diff_only(
    baseline: Board | None, new_cnn: Board,
    *, allow_puyo_to_empty: bool = True,
    empty_to_color_guard: Board | None = None,
    enable_gravity_filter_support: bool = False,
    merge_use_majority_value: bool = False,
    initial_confirm_history: list[Board] | None = None,
    initial_confirm_min_votes: int = DEFAULT_INITIAL_CONFIRM_MIN_VOTES,
    hsv_board: Board | None = None,
    enable_puyo_to_empty_hsv_guard: bool = False,
    history_board: Board | None = None,
) -> Board:
    """baseline をベースに、CNN との差分 cell のみ new_cnn 値で上書き.

    Phase C-5 の「差分のみ反映」コア。baseline cell の情報を絶対に
    壊さないことで、CNN ぶれ (= 背景誤認、エフェクト残光等) が
    confirmed_board に乗るのを構造的に防ぐ。

    Args:
        baseline: 直前の確定盤面 (None なら初回 = CNN 全採用)
        new_cnn: 新たに観測された CNN 盤面
        allow_puyo_to_empty: True なら puyo→空 遷移を許容、False なら
            「置いた puyo は CNN の puyo→空 転換を ban」(= 連鎖発火以外で
            消えない、Phase C-6 の B 強化)
        empty_to_color_guard: F (cycle 56) STABLE 復帰ゲート用の多数決 board.
            None でない場合、 EMPTY → 色 遷移 cell について
            「guard 内で同色が観測されている」 場合のみ採用する。
            これにより、 NON-STABLE 中の瞬間的な背景誤認 (= 1-2 frame 限定)
            が confirmed_board に焼き付くのを防ぐ。 default None で従来挙動。
        enable_gravity_filter_support: 案(a) 重力フィルタ支持緩和
            (#45 おじゃま merge 統合修正, 2026-07-24)。 True にすると
            `_apply_gravity_filter` に `empty_to_color_guard` を
            support_board として渡す。 F ガード (empty_to_color_guard) 起因で
            EMPTY のまま残った cell を「浮き判定の gap」 として扱わなくなり、
            積もり中のおじゃまが誤って浮きぷよ扱いで消去されるのを防ぐ。
            default False = 従来挙動完全維持 (backwards compat)。
        merge_use_majority_value: 案(b) 退出 merge 書込値の多数決化
            (#45 おじゃま merge 統合修正, 2026-07-24)。 True にすると
            EMPTY → 色 遷移ガード分岐で、 単一フレーム cnn_v でなく
            多数決値 empty_to_color_guard (guard_v) を書き込む。
            退出時の単一フレーム CNN ちらつきによる却下を解消する。
            default False = 従来挙動完全維持 (backwards compat)。
        initial_confirm_history: 初回STABLE確定の多数決ガード
            (enable_initial_confirm_vote, 2026-07-27)。 baseline is None
            (初回確定) 時、直前 NON-STABLE 滞在中の cnn_board 履歴。
            None または空リストなら従来挙動 (単発 new_cnn 採用、
            backwards compat)。
        initial_confirm_min_votes: 上記多数決の最低観測 frame 数。
        hsv_board: 現フレームの HSV 認識盤面。 enable_puyo_to_empty_hsv_guard
            用の照合対象。 None なら照合ガード不発 (従来挙動)。
        enable_puyo_to_empty_hsv_guard: 色→空 HSV 照合ガード (2026-07-30)。
            True にすると、 baseline が色・cnn が空 の cell について HSV が
            色を保持している場合は消さず baseline を維持する。 単一フレーム
            CNN の光沢→空 誤読が STABLE 復帰 merge で無投票消去され、直後の
            gravity filter で上のぷよまで連鎖消去される列デッドロック
            (c34 1P col=1, frame 14332 実測) を根で止める。
            default False = 従来挙動完全維持・bit-identical (backwards compat)。
        history_board: R2 浮きぷよ是正機構 (2026-08-17)。 `_apply_gravity_filter`
            にそのまま渡す (直前 STABLE 盤面を想定)。呼び出し側が
            from_state スコープ判定 (色→空 が物理的に説明可能な遷移か) を
            済ませた上で渡す前提。 default None = 従来挙動完全維持
            (backwards compat)。

    Returns:
        merged: baseline + 物理整合性 filter 後の差分のみ更新された盤面
    """
    from src.board import COLOR_EMPTY

    if baseline is None:
        return _build_initial_confirmed_board(
            new_cnn, initial_confirm_history, initial_confirm_min_votes,
        )

    from src.board import COLOR_UNKNOWN

    merged = baseline.copy()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            base_v = baseline.get(r, c)
            cnn_v = new_cnn.get(r, c)
            if base_v == cnn_v:
                continue
            # D (2026-05-11): cnn が UNKNOWN を返した cell は baseline 維持.
            # telop 被覆や image_reader 不確実時に旧色を保持.
            if cnn_v == COLOR_UNKNOWN:
                continue
            # B: 色→空 遷移の保護 (= 確定 cell 保護)。休眠 blanket ban
            # (allow_puyo_to_empty=False) に加え、HSV 照合ガード
            # (enable_puyo_to_empty_hsv_guard) で HSV が色を裏付ける単一
            # フレーム CNN の空誤読 (光沢→空) を退ける。
            hsv_v = hsv_board.get(r, c) if hsv_board is not None else None
            if _should_keep_puyo_over_empty(
                base_v, cnn_v, hsv_v, allow_puyo_to_empty,
                enable_puyo_to_empty_hsv_guard,
            ):
                continue  # baseline の puyo を維持
            # F (cycle 56): EMPTY → 色 遷移ガード。 NON-STABLE 中の多数決で
            # 同色が確認できない cell は背景誤認の可能性高 → baseline 維持。
            if (
                base_v == COLOR_EMPTY
                and cnn_v != COLOR_EMPTY
                and empty_to_color_guard is not None
            ):
                guard_v = empty_to_color_guard.get(r, c)
                if guard_v == COLOR_EMPTY:
                    continue  # 多数決も空 = 誤認可能性高 → baseline 維持
                if merge_use_majority_value:
                    # 案(b): 単一フレーム cnn_v でなく多数決値 guard_v を書込む
                    # (退出時の単一フレームちらつきによる却下を解消)。
                    merged.set(r, c, guard_v)
                    continue
                if guard_v != cnn_v:
                    continue  # 従来挙動: 多数決と不一致な単一フレームは却下
            merged.set(r, c, cnn_v)
    # A: 浮きぷよ ban (空 cell の上に puyo は物理的にあり得ない、空に戻す)
    # 案(a): flag ON かつ guard あれば support_board として渡す。
    support_board = (
        empty_to_color_guard if enable_gravity_filter_support else None
    )
    _apply_gravity_filter(
        merged, support_board=support_board, history_board=history_board,
    )
    return merged


# バーストガード Stage1.5 (2026-08-05 アーキ追補、
# docs/BURST_GUARD_DESIGN_2026-08-05.md §10)。from_state ごとに
# 「この遷移で物理的に説明可能な新規値クラス」を静的に定義する。
# TSUMO_FALL: ツモ設置は空セルへの色puyo(1-5)出現のみ説明可能。
# OJAMA_FALL: おじゃまは空セルにのみ落下する (reference_ojama_landing_pattern)。
# 対象外 from_state (CHAIN/GRAVITY_SETTLE/EFFECT等) は §10.2 により
# 明示的にスコープ外 (無条件no-op)。
_TRANSITION_MERGE_GUARD_SCOPE: "dict[BoardState, frozenset[int]]" = {
    BoardState.TSUMO_FALL: frozenset({1, 2, 3, 4, 5}),
    BoardState.OJAMA_FALL: frozenset({9}),  # COLOR_OJAMA
}


def _filter_transition_new_cnn_for_burst_guard(
    baseline: "Board | None", new_cnn: "Board", from_state: "BoardState",
) -> "Board":
    """遷移merge直前のバースト対策フィルタ (stateless純関数、Stage1.5 §10.4)。

    from_state ごとに `_TRANSITION_MERGE_GUARD_SCOPE` で定義した「物理的に
    説明可能な diff」以外は COLOR_UNKNOWN に差し替える。`_merge_diff_only`
    の既存 D guard (605-608行、cnn_v==COLOR_UNKNOWN は baseline維持) が
    そのまま棄却処理を担うため、`_merge_diff_only` 自体は一切変更しない。

    対象外 from_state (`_TRANSITION_MERGE_GUARD_SCOPE` に無い state) と
    baseline is None (初回STABLE確定、§10.3) は new_cnn をそのまま返す
    (恒等、コピーもしない = 呼び出し側で `is` 比較できる)。

    Args:
        baseline: 直前の確定盤面 (= confirmed_board、遷移前の from_state 時点)。
        new_cnn: 遷移フレームの CNN 認識盤面。
        from_state: 遷移前の BoardState (呼び出し側は state 再代入前に読むこと)。

    Returns:
        フィルタ後の new_cnn (説明不可能な diff は COLOR_UNKNOWN に差し替え済み)。
    """
    allowed_colors = _TRANSITION_MERGE_GUARD_SCOPE.get(from_state)
    if allowed_colors is None or baseline is None:
        return new_cnn
    from src.board import COLOR_UNKNOWN

    filtered = new_cnn.copy()
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            base_v = baseline.get(r, c)
            cnn_v = new_cnn.get(r, c)
            if base_v == cnn_v:
                continue
            if base_v == COLOR_EMPTY and cnn_v in allowed_colors:
                continue  # 物理的に説明可能な diff (設置/おじゃま着弾)
            filtered.set(r, c, COLOR_UNKNOWN)  # 説明不可能 → baseline維持 (D guard)
    return filtered


def _apply_gravity_filter(
    board: Board, *, support_board: Board | None = None,
    history_board: Board | None = None,
) -> None:
    """浮きぷよ ban (Phase C-6 の A): 空 cell の上に puyo がある場合、
    その上の puyo を空に戻す。連鎖中の重力再配置中に CNN が誤検出した
    「浮きぷよ」を物理的にあり得ないと却下する。

    実装: 各列を下から走査、最下段 (row=12) から上に進み、空セルが
    現れた次の non-EMPTY cell があれば消去 (= 浮き判定)。

    UNKNOWN cell は「判定保留」 なので gap_found に立てない (= 上のぷよを
    erase しない)。 cycle 71v で一時的に UNKNOWN を gap 扱いにしたが、
    2P telop/UI で UNKNOWN になる cell 上のぷよが大量 erase される副作用が
    確認されたため撤回 (2026-05-14)。

    Args:
        support_board: 案(a) 重力フィルタ支持緩和 (#45 おじゃま merge
            統合修正, 2026-07-24)。 None でない場合、 board 側が EMPTY でも
            support_board の同 cell が非 EMPTY/非 UNKNOWN なら
            「gap 扱いしない」 (= その cell を浮き判定の穴として使わない)。
            F ガード (empty_to_color_guard) 起因で EMPTY のまま残った cell が
            積もり中のおじゃまを浮きぷよ誤消去するのを防ぐ目的で渡す想定。
            default None = 従来挙動完全維持 (backwards compat)。
        history_board: R2 浮きぷよ是正機構 (2026-08-17)。 None でない場合、
            gap 判定される EMPTY cell について history_board の同 cell が
            非空色ならその色で復元し gap 扱いにしない (= 「上の puyo が
            浮いている」でなく「下が誤 EMPTY」という解釈を優先する)。
            復元できなければ (history_board が None、または同 cell が
            EMPTY/UNKNOWN) 従来通り gap 扱い (= 上の puyo を消す) に
            フォールバックする。呼び出し側 (BoardStateMachine) が直前
            STABLE 盤面を保持・供給する外部 wrapper であり、本関数自身は
            history を保持しない (stateless)。
            default None = 従来挙動完全維持・bit-identical (backwards compat)。
    """
    from src.board import COLOR_EMPTY, COLOR_UNKNOWN

    for c in range(BOARD_COLS):
        gap_found = False
        for r in range(BOARD_ROWS - 1, -1, -1):
            v = board.get(r, c)
            if v == COLOR_EMPTY:
                # 案(a): support_board が同 cell を「空でない」と裏付ける
                # 場合、この EMPTY を gap として扱わない (浮き判定の穴にしない)。
                if support_board is not None:
                    sup_v = support_board.get(r, c)
                    if sup_v != COLOR_EMPTY and sup_v != COLOR_UNKNOWN:
                        continue
                # R2: history_board が同 cell に非空色の記録を持つなら
                # 「誤 EMPTY 疑い」として復元する (= 消す方向でなく疑う方向)。
                if history_board is not None:
                    hist_v = history_board.get(r, c)
                    if hist_v != COLOR_EMPTY and hist_v != COLOR_UNKNOWN:
                        board.set(r, c, hist_v)
                        continue  # 復元成功、 gap 扱いにしない
                gap_found = True
            elif gap_found:
                # 空 cell の上に puyo → 浮き → 空に戻す (復元不能時のフォールバック)
                board.set(r, c, COLOR_EMPTY)


class BoardStateMachine:
    """1 プレイヤー側の盤面状態を時系列で追跡する.

    Usage:
        sm = BoardStateMachine(detectors=[...])
        for frame_idx, signals in enumerate(stream):
            ctx = sm.update(frame_idx, signals)
            if ctx.is_stable():
                # CNN 確定盤面が ctx.confirmed_board に入っている
                ...
            else:
                # 推論モード、別途 ChainSimulator 等で盤面を生成
                ...
    """

    def __init__(
        self,
        detectors: list[StateTransitionDetector] | None = None,
        stable_frame_count: int = DEFAULT_STABLE_FRAME_COUNT,
        *,
        non_stable_history_size: int = DEFAULT_NON_STABLE_HISTORY_SIZE,
        empty_to_color_min_votes: int = DEFAULT_EMPTY_TO_COLOR_MIN_VOTES,
        enable_next_corroborated_confirm: bool = False,
        next_corroborated_min_votes: int = NEXT_CORROBORATED_MIN_VOTES,
        enable_stable_resume_gate: bool = True,
        enable_warmup_guard: bool = False,
        warmup_frames: int = STABLE_WARMUP_FRAMES,
        enable_stable_recovery_gate: bool = False,
        recovery_min_frames: int = STABLE_RECOVERY_MIN_FRAMES,
        enable_gravity_filter_support: bool = False,
        merge_use_majority_value: bool = False,
        enable_column_partial_support: bool = False,
        enable_match_start_full_clear: bool = False,
        enable_recovery_counter_carryover: bool = False,
        recovery_counter_carryover_max_sec: float = (
            RECOVERY_COUNTER_CARRYOVER_MAX_SEC
        ),
        enable_cnn_flicker_hsv_fallback: bool = False,
        cnn_flicker_window_frames: int = CNN_FLICKER_WINDOW_FRAMES,
        cnn_flicker_min_changes: int = CNN_FLICKER_MIN_CHANGES,
        enable_initial_confirm_vote: bool = False,
        initial_confirm_min_votes: int = DEFAULT_INITIAL_CONFIRM_MIN_VOTES,
        # 色→空 HSV 照合ガード (2026-07-30): c34 型の列デッドロックには有効だが、
        # 4動画測定 (c34/c58/c26/c69) で c58/c26 の 2P は tail 悪化、c26/c69 の 1P は
        # 効果ゼロと判明 (data/verify/placement_confirm_frames_generalization_2026-07-30)。
        # 8フレーム達成率は OFF/ON 不変で、改善は不合格イベントの重症度低下のみ。
        # 汎化未確認のため default OFF を維持する。False で bit-identical (backwards compat)。
        enable_puyo_to_empty_hsv_guard: bool = False,
        # 復旧ゲート方向別しきい値 非対称化 (2026-07-30, A/B 計測用): True で
        # 方向1 (空→色) のみ recovery_add_min_frames (既定 4) で発火させ、
        # 方向2/3 (色→空/色→色) は recovery_min_frames (既定 8) を維持する。
        # 「誤認が治るまでのラグ」短縮用。誤って追加しても実ぷよが無ければ次の
        # 観測で消えるため空→色のみ短縮する (消す方向は増幅リスクで据え置き)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        # user 承認前のため default OFF 固定。
        enable_asymmetric_recovery_min_frames: bool = False,
        recovery_add_min_frames: int = STABLE_RECOVERY_ADD_MIN_FRAMES,
        # エフェクト時間ゲート (enable_effect_gate, 2026-08-03、A/B 計測用):
        # True で `_apply_stable_recovery_gate` が signals.effect_gate_window_active
        # 中のみ EFFECT_GATE_TOP_ROWS を実秒ベース持続確認 (effect_gate_persist_sec)
        # に切り替える。enable_stable_recovery_gate=False の場合は本フラグに
        # 関係なく STABLE 中の per-frame confirmed 更新経路自体が無い (無害)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        # user 承認前の savepoint 実装のため default OFF 固定。
        enable_effect_gate: bool = False,
        effect_gate_persist_sec: float = EFFECT_PERSIST_SEC,
        # バーストガード再設計 (2026-08-05、Stage1、A/B 計測用):
        # docs/BURST_GUARD_DESIGN_2026-08-05.md §3。True で
        # `_apply_stable_recovery_gate` のゲート対象 cell が持続確認を行わず
        # Window ON 中は無条件で発火しないハード凍結に切り替わる
        # (`_update_effect_gate_hold` の persist逆転を構造的に排除する)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        effect_gate_hard_freeze: bool = False,
        # バーストガード Stage1.5 (2026-08-05 アーキ追補、A/B 計測用):
        # docs/BURST_GUARD_DESIGN_2026-08-05.md §10。True で NON-STABLE→STABLE
        # 遷移時の `_merge_diff_only` 呼び出しに渡す new_cnn を、
        # signals.effect_gate_window_active 中のみ物理的期待値フィルタ
        # (`_filter_transition_new_cnn_for_burst_guard`) 経由に切り替える。
        # `_merge_diff_only` 自体・既存引数は一切変更しない。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        enable_transition_merge_guard: bool = False,
        # 盤面確定窓 3中2多数決 (stable_majority_window, 2026-08-13 user承認、
        # 認識99.5%物差し条件付き採用)。True にすると `_update_within_current_state`
        # の初回STABLE確定窓が「stable_frame_count 連続厳密一致」から
        # 「直近 stable_majority_window_frames 観測中 stable_majority_min_votes
        # 以上一致」に切り替わる (`_majority_window_vote` 参照)。
        # default False = 従来の厳密連続一致を完全維持・bit-identical
        # (backwards compat、148動画収集走行中のため既定OFF必須)。
        stable_majority_window: bool = False,
        stable_majority_window_frames: int = STABLE_MAJORITY_WINDOW_FRAMES,
        stable_majority_min_votes: int = STABLE_MAJORITY_MIN_VOTES,
        # R2 浮きぷよ是正機構 (enable_floating_gap_restore, 2026-08-17):
        # True にすると、 TSUMO_FALL/OJAMA_FALL → STABLE 遷移の
        # `_merge_diff_only` で「下が空・上に puyo」の物理矛盾を検出した際、
        # 上の puyo を消すのでなく直前 confirmed_board (= 直近 STABLE 盤面)
        # の同 cell から色ぷよ復元を試みる (`_apply_gravity_filter` 参照)。
        # CHAIN/GRAVITY_SETTLE 遷移では色→空 が物理的に正当な消去でありうる
        # ため対象外 (`_TRANSITION_MERGE_GUARD_SCOPE` と同じスコープを再利用)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        enable_floating_gap_restore: bool = False,
        # 持続誤認26件系統2 (enable_ojama_column_stack_fix, 2026-08-17、
        # docs/KNOWN_WEAKNESSES.md W10 参照、c109 実測): OJAMA_FALL → STABLE
        # 遷移 merge で、既存の色ぷよ (1-5) が同フレームの CNN 誤読で
        # おじゃま(9) に上書きされる物理違反 (= 同一列内の二重着地衝突、
        # 煙/バーストの半透明重畳で既存puyoがおじゃま色に誤分類される事故)
        # を防ぐ。True の場合、`_filter_transition_new_cnn_for_burst_guard`
        # (既存関数、無改修で再利用) を OJAMA_FALL からの遷移merge時に
        # signals.effect_gate_window_active に関係なく必ず適用し、
        # 「base が EMPTY でない cell が おじゃま(9) に化ける」diff を
        # 却下する (base=EMPTY→9 の正当な着地のみ許可)。
        # `enable_transition_merge_guard` (Stage1.5) と独立フラグ
        # (effect_gate_window_active 依存を外した狭いスコープの方が
        # 効果を確認しやすいため)。default False = 従来挙動完全維持・
        # bit-identical (backwards compat、user承認前の savepoint 実装)。
        enable_ojama_column_stack_fix: bool = False,
    ) -> None:
        self._non_stable_history_size = int(non_stable_history_size)
        self._empty_to_color_min_votes = int(empty_to_color_min_votes)
        # ネクスト裏付け確定 (2026-07-31)。default OFF = bit-identical。
        self._enable_next_corroborated_confirm = bool(
            enable_next_corroborated_confirm
        )
        self._next_corroborated_min_votes = max(
            1, int(next_corroborated_min_votes)
        )
        self._enable_stable_resume_gate = bool(enable_stable_resume_gate)
        # B1 (M1 warmup guard): STABLE 遷移直後の confirmed 凍結を有効化するか。
        # backwards compat: default False で既存動作と同一。
        self._enable_warmup_guard = bool(enable_warmup_guard)
        self._warmup_frames = max(0, int(warmup_frames))
        # 設計C 事後復旧ゲート (2026-06-02):
        # True で STABLE 中の confirmed==EMPTY かつ CNN==HSV 継続セルを復旧する。
        # default False = B1 禁忌隣接のため OFF。viz 検証後に判断する。
        self._enable_stable_recovery_gate = bool(enable_stable_recovery_gate)
        self._recovery_min_frames = max(1, int(recovery_min_frames))
        # #45 おじゃま merge 統合修正 案(a)(b) (2026-07-24):
        # 案B (OJAMA_FALL 退出遅延, savepoint/ojama-fall-board-settle) 適用後に
        # 判明した _merge_diff_only の 2 副作用を個別 flag で修正する。
        # (a) enable_gravity_filter_support: F ガード起因の EMPTY を浮き判定の
        #     穴にしない (積もり中おじゃまの誤消去防止)。
        # (b) merge_use_majority_value: 退出 merge 書込値を多数決化し
        #     単一フレーム CNN ちらつきによる却下を解消する。
        # 両者は独立 flag (A/B 切り分け用)。 default False = 従来挙動完全維持
        # (backwards compat)。
        self._enable_gravity_filter_support = bool(enable_gravity_filter_support)
        self._merge_use_majority_value = bool(merge_use_majority_value)
        # R2 浮きぷよ是正機構 (2026-08-17)。 default False = bit-identical。
        self._enable_floating_gap_restore = bool(enable_floating_gap_restore)
        # 持続誤認26件系統2 (2026-08-17)。 default False = bit-identical。
        self._enable_ojama_column_stack_fix = bool(
            enable_ojama_column_stack_fix
        )
        # 列ゲート緩和 (enable_column_partial_support, 2026-07-25):
        # True で _apply_stable_recovery_gate の安全弁C浮き判定/最終重力
        # フィルタに stable_recovery_counters 由来の support を渡す。
        # default False = 従来挙動完全維持 (backwards compat)。
        self._enable_column_partial_support = bool(enable_column_partial_support)
        # 前試合盤面残骸リーク修正 (feat/recognition-postchain-fix-2026-07-23):
        # is_match_active=False (MENU 強制) 時、従来は confirmed_board /
        # pending_board / pending_count / last_stable_idx / chain_count /
        # ojama_pending の 6 field しかクリアしておらず、
        # non_stable_cnn_history / stable_recovery_counters / recovery_cells /
        # stable_warmup_remaining / next_queue が前試合の値のまま残留していた。
        # 残留した non_stable_cnn_history は次試合の NON-STABLE→STABLE 復帰時
        # (empty_guard 多数決) に、stable_recovery_counters/recovery_cells は
        # 設計C 事後復旧ゲートに、それぞれ前試合の色を「証拠」として渡してしまい、
        # 前試合終盤に実在したぷよが次試合序盤へ幽霊セルとして書き戻る経路になる。
        # True でこれら 5 field も試合境界で完全クリアする。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        self._enable_match_start_full_clear = bool(enable_match_start_full_clear)
        # 復旧カウンタ carryover (enable_recovery_counter_carryover, 2026-07-26):
        # True で STABLE→NON-STABLE 遷移時の stable_recovery_counters /
        # recovery_cells 即クリアを保留し、非 STABLE 滞在が
        # recovery_counter_carryover_max_sec 以内なら STABLE 復帰後も引き継ぐ。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        self._enable_recovery_counter_carryover = bool(
            enable_recovery_counter_carryover
        )
        self._recovery_counter_carryover_max_sec = max(
            0.0, float(recovery_counter_carryover_max_sec)
        )
        # CNN 乱高下セル HSV フォールバック (#51 後半, 2026-07-26):
        # True で復旧ゲートの合意判定において、直近 N フレームの CNN 出力が
        # 乱高下しているセルを HSV 優先に切り替える (詳細は定数定義部を参照)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        self._enable_cnn_flicker_hsv_fallback = bool(
            enable_cnn_flicker_hsv_fallback
        )
        self._cnn_flicker_window_frames = max(1, int(cnn_flicker_window_frames))
        self._cnn_flicker_min_changes = max(1, int(cnn_flicker_min_changes))
        # 初回STABLE確定の多数決ガード (enable_initial_confirm_vote, 2026-07-27):
        # 色→空凍結の修正3点セット③。 単一フレームCNNでなく、直前
        # NON-STABLE滞在中に蓄積した non_stable_cnn_history の多数決で
        # 初回confirmedを構成する (fallback=new_cnn で観測不足セルはEMPTY化しない)。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        self._enable_initial_confirm_vote = bool(enable_initial_confirm_vote)
        self._initial_confirm_min_votes = max(1, int(initial_confirm_min_votes))
        # 色→空 HSV 照合ガード (2026-07-30): True で NON-STABLE→STABLE 復帰
        # merge の色→空 遷移について HSV が色を保持する cell を消さない。
        # 光沢→空 の単一フレーム CNN 誤読が無投票消去され gravity filter で
        # 上のぷよまで連鎖消去される列デッドロックを根で止める。
        # default False = 従来挙動完全維持・bit-identical (backwards compat)。
        self._enable_puyo_to_empty_hsv_guard = bool(
            enable_puyo_to_empty_hsv_guard
        )
        # 復旧ゲート方向別しきい値 非対称化 (2026-07-30): 方向1(空→色)のみ
        # 短縮する。default False = 従来挙動完全維持 (backwards compat)。
        self._enable_asymmetric_recovery_min_frames = bool(
            enable_asymmetric_recovery_min_frames
        )
        self._recovery_add_min_frames = max(1, int(recovery_add_min_frames))
        # エフェクト時間ゲート (2026-08-03): default False = 従来挙動完全維持
        # (backwards compat)。
        self._enable_effect_gate = bool(enable_effect_gate)
        self._effect_gate_persist_sec = max(0.0, float(effect_gate_persist_sec))
        # バーストガード再設計 (2026-08-05): default False = 従来挙動完全維持。
        self._effect_gate_hard_freeze = bool(effect_gate_hard_freeze)
        # バーストガード Stage1.5 (2026-08-05): default False = 従来挙動完全維持。
        self._enable_transition_merge_guard = bool(enable_transition_merge_guard)
        # 盤面確定窓 3中2多数決 (2026-08-13): default False = 従来挙動完全維持。
        self._enable_stable_majority_window = bool(stable_majority_window)
        self._stable_majority_window_frames = max(
            2, int(stable_majority_window_frames),
        )
        self._stable_majority_min_votes = max(1, int(stable_majority_min_votes))
        self._detectors: list[StateTransitionDetector] = (
            list(detectors) if detectors else []
        )
        self._stable_n = max(1, stable_frame_count)
        self._ctx = StateContext()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    @property
    def context(self) -> StateContext:
        return self._ctx

    def add_detector(self, detector: StateTransitionDetector) -> None:
        """検出器を追加。優先順は登録順。"""
        self._detectors.append(detector)

    def reset(self, *, keep_match_state: bool = False) -> None:
        """state machine を初期化。drift 検出時の re-sync で使う想定。"""
        if keep_match_state:
            kept_state = self._ctx.state if self._ctx.state == BoardState.MENU \
                else BoardState.STABLE
            self._ctx = StateContext(state=kept_state)
        else:
            self._ctx = StateContext()

    def clear_match_start_residue(self) -> None:
        """試合境界で残留しうる非表示 side-channel field のみをクリアする。

        前試合盤面残骸リーク修正 (2026-07-23): confirmed_board/state 等の
        主 state には触れず、non_stable_cnn_history / stable_recovery_counters
        / recovery_cells / stable_warmup_remaining / next_queue の 5 field
        のみを初期化する (`reset()` ほど広範囲ではない、狙い撃ちの部分クリア)。

        force_in_match=True (raw_active 常時 True) 構成では update() の
        is_match_active=False 分岐 (MENU 強制) が一度も発火しないため、
        score リセット検知など別経路の試合境界シグナルから明示的に
        このメソッドを呼び出す必要がある
        (`recognition_pipeline.RecognitionPipeline.update` の
        `enable_match_start_full_clear` 参照)。
        """
        self._ctx.non_stable_cnn_history = []
        self._ctx.stable_recovery_counters = {}
        self._ctx.recovery_cells = set()
        self._ctx.stable_warmup_remaining = 0
        self._ctx.next_queue = []
        self._ctx.cnn_flicker_history = {}

    def force_match_boundary_reset(self) -> None:
        """試合境界を外部から明示的に注入する (update() の
        is_match_active=False 分岐と同一内容)。

        追修 (2026-07-25): force_in_match=True (raw_active 常時 True) 構成
        では update() に is_match_active=False が一度も渡らず、上記分岐
        (state=MENU 強制 + confirmed_board 等 6 field クリア) が発火しない。
        この構成では試合境界を score リセット (新ゲーム開始で score が
        大幅減少/両者ほぼ0) からしか検知できないため、その検知経路
        (`recognition_pipeline.RecognitionPipeline.update`) から本メソッドを
        明示的に呼び出し、is_match_active=False 分岐と bit-identical な
        クリアを行う (confirmed_board=None を含む主 state + 残骸 5 field)。
        """
        self._ctx.state = BoardState.MENU
        self._ctx.pending_board = None
        self._ctx.pending_count = 0
        self._ctx.confirmed_board = None
        self._ctx.last_stable_idx = -1
        self._ctx.chain_count = 0
        self._ctx.ojama_pending = 0
        if self._enable_match_start_full_clear:
            self.clear_match_start_residue()

    def _next_corroborated_colors(self) -> "frozenset[int] | None":
        """直近に消費された next/dnext ペアの色集合を返す (裏付け用)。

        ネクスト裏付け確定 (2026-07-31) の情報源。フラグ OFF なら None を返し、
        `_vote_majority_board` は従来どおり一律 min_votes で判定する
        (= bit-identical)。

        **キューの対応付けは真値として使わない。**「どのネクストがどの設置に
        対応するか」の追跡は過去に信頼できなかった (キューに正解無し57%)。
        ここでは末尾 NEXT_CORROBORATION_QUEUE_TAIL 件のペアに含まれる色の
        **集合メンバシップだけ**を使うので、対応付けがずれても安全側に倒れる。

        Returns:
            裏付け色の集合。フラグ OFF / キューが空なら None。
        """
        if not self._enable_next_corroborated_confirm:
            return None
        queue = getattr(self._ctx, "next_queue", None)
        if not queue:
            return None
        colors: set[int] = set()
        for pair in list(queue)[-NEXT_CORROBORATION_QUEUE_TAIL:]:
            if not pair:
                continue
            for v in pair:
                iv = int(v)
                # 有効なぷよ色 (1-5) のみ。おじゃま(9)/UNKNOWN(10)/空(0) は除く
                if 1 <= iv <= 5:
                    colors.add(iv)
        return frozenset(colors) if colors else None

    def update(
        self, frame_idx: int, signals: DetectorSignals,
    ) -> StateContext:
        """新 frame を投入し、更新後の StateContext を返す."""
        self._ctx.frame_idx = frame_idx
        self._ctx.time_sec = signals.time_sec

        # 試合外なら全部 MENU に倒す (= 認識結果を保持しない)
        if not signals.is_match_active:
            self.force_match_boundary_reset()
            return self._ctx

        # 検出器を順に適用、最初に発火したものを採用
        new_state: BoardState | None = None
        for det in self._detectors:
            res = det.detect(self._ctx, signals)
            if res is not None:
                new_state = res
                break

        if new_state is not None:
            self._apply_transition(new_state, signals)
        else:
            self._update_within_current_state(signals)

        # next_queue 更新 (next_pair が新規確定なら追加)
        if signals.next_pair is not None:
            if (
                not self._ctx.next_queue
                or self._ctx.next_queue[-1] != signals.next_pair
            ):
                self._ctx.next_queue.append(signals.next_pair)
                # 履歴は最新 8 件まで保持 (ダブルネクスト + α 余裕)
                if len(self._ctx.next_queue) > 8:
                    self._ctx.next_queue.pop(0)

        return self._ctx

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _apply_transition(
        self, new_state: BoardState, signals: DetectorSignals,
    ) -> None:
        """detector が指定した state へ強制遷移."""
        if new_state == self._ctx.state:
            self._update_within_current_state(signals)
            return
        # state 切り替えで pending をリセット
        self._ctx.pending_board = None
        self._ctx.pending_count = 0
        # 盤面確定窓 3中2多数決 (2026-08-13): pending リセットと同じタイミングで
        # window 履歴もクリアする (前 state の観測を次 window に持ち込まない)。
        if self._enable_stable_majority_window:
            self._ctx.confirm_window_history = []
        # NON-STABLE → STABLE 復帰時: 差分のみ反映で baseline 維持
        # (Phase C-5: 全 cell コピーは CNN ぶれを取り込むため廃止)
        if (
            self._ctx.state in NON_STABLE_STATES
            and new_state == BoardState.STABLE
        ):
            # F (cycle 56): NON-STABLE 中の cnn_board 履歴から多数決 board を生成。
            # EMPTY → 色 遷移ガードに渡し、 瞬間的な背景誤認の焼き付きを防ぐ。
            empty_guard: Board | None = None
            if (
                self._enable_stable_resume_gate
                and self._ctx.non_stable_cnn_history
            ):
                empty_guard = _vote_majority_board(
                    self._ctx.non_stable_cnn_history,
                    min_votes=self._empty_to_color_min_votes,
                    corroborated_colors=self._next_corroborated_colors(),
                    corroborated_min_votes=(
                        self._next_corroborated_min_votes
                        if self._enable_next_corroborated_confirm else None
                    ),
                )
            # 初回STABLE確定の多数決ガード (enable_initial_confirm_vote,
            # 2026-07-27): baseline is None (初回確定) 時のみ使われる
            # (baseline あり時は _merge_diff_only 側で無視される)。
            initial_history = (
                self._ctx.non_stable_cnn_history
                if self._enable_initial_confirm_vote
                else None
            )
            # バーストガード Stage1.5 (2026-08-05 §10.4): Window ON 中のみ
            # 遷移merge直前の new_cnn を物理的期待値フィルタに通す。
            # self._ctx.state はこの時点でまだ from_state (再代入は後段)。
            # `_merge_diff_only` 自体・既存引数 (signals.cnn_board) は不変、
            # ローカル変数 new_cnn_for_merge を新たに merge の入力に使うのみ。
            new_cnn_for_merge = signals.cnn_board
            # 持続誤認26件系統2 (2026-08-17): OJAMA_FALL からの遷移では
            # effect_gate_window_active の狭い窓に関係なく物理フィルタを
            # 常時適用する (c109 実測=衝突フレームは窓終了後だった)。
            _apply_ojama_stack_fix = (
                self._enable_ojama_column_stack_fix
                and self._ctx.state == BoardState.OJAMA_FALL
            )
            if (
                (
                    self._enable_transition_merge_guard
                    and signals.effect_gate_window_active
                )
                or _apply_ojama_stack_fix
            ):
                new_cnn_for_merge = _filter_transition_new_cnn_for_burst_guard(
                    self._ctx.confirmed_board, signals.cnn_board, self._ctx.state,
                )
            # R2 浮きぷよ是正機構 (2026-08-17): 色→空 が物理的に正当化できない
            # from_state (TSUMO_FALL/OJAMA_FALL、`_TRANSITION_MERGE_GUARD_SCOPE`
            # と同じスコープ) のときのみ history_board (= 遷移前 confirmed_board)
            # を渡す。CHAIN/EFFECT/GRAVITY_SETTLE 等では色ぷよの消滅が正当な
            # 物理事象でありうるため None のまま (= 復元を試みない)。
            _floating_restore_history: "Board | None" = None
            if (
                self._enable_floating_gap_restore
                and self._ctx.state in _TRANSITION_MERGE_GUARD_SCOPE
            ):
                _floating_restore_history = self._ctx.confirmed_board
            self._ctx.confirmed_board = _merge_diff_only(
                self._ctx.confirmed_board, new_cnn_for_merge,
                empty_to_color_guard=empty_guard,
                enable_gravity_filter_support=self._enable_gravity_filter_support,
                merge_use_majority_value=self._merge_use_majority_value,
                initial_confirm_history=initial_history,
                initial_confirm_min_votes=self._initial_confirm_min_votes,
                hsv_board=signals.hsv_board,
                enable_puyo_to_empty_hsv_guard=(
                    self._enable_puyo_to_empty_hsv_guard
                ),
                history_board=_floating_restore_history,
            )
            self._ctx.last_stable_idx = self._ctx.frame_idx
            self._ctx.pending_board = self._ctx.confirmed_board.copy()
            self._ctx.pending_count = 1
            # B1 (M1 warmup guard): STABLE 復帰直後は N frame 間 confirmed 凍結。
            # 遷移直後の CNN 出力が不安定 (エフェクト残光・背景誤認) な期間に
            # _update_within_current_state が confirmed を書き換えるのを防ぐ。
            if self._enable_warmup_guard:
                self._ctx.stable_warmup_remaining = self._warmup_frames
            # 復旧カウンタ carryover: STABLE 復帰完了時点で滞在計測をリセット。
            if self._enable_recovery_counter_carryover:
                self._ctx.non_stable_entry_time_sec = None
        old_state = self._ctx.state
        self._ctx.state = new_state
        # F: state 切替で NON-STABLE history をリセット
        self._ctx.non_stable_cnn_history = []
        if new_state != BoardState.CHAIN:
            self._ctx.chain_count = 0
        if new_state != BoardState.OJAMA_FALL:
            self._ctx.ojama_pending = 0
        # 設計C: STABLE → NON-STABLE 遷移時に復旧カウンタ・復旧済みセル集合をクリア。
        # 次の STABLE 期間は新規状態から積み上げ直す (carryover 有効時は例外)。
        if new_state in NON_STABLE_STATES:
            self._handle_recovery_clear_on_non_stable_entry(
                old_state, signals.time_sec,
            )

    def _handle_recovery_clear_on_non_stable_entry(
        self, old_state: BoardState, time_sec: float,
    ) -> None:
        """STABLE(等) → NON-STABLE 遷移時の復旧カウンタ処理.

        enable_recovery_counter_carryover=False (default): 即時クリア
        (従来挙動、 backwards compat)。
        True: NON-STABLE 同士の遷移 (例: TSUMO_FALL→CHAIN) では滞在計測を
        継続し (entry_time を上書きしない)、 STABLE から初めて非 STABLE に
        入った瞬間のみ entry_time を記録する。 いずれの場合も
        recovery_counter_carryover_max_sec 超過分は即クリアする。

        Args:
            old_state: 遷移前の state (呼び出し時点で self._ctx.state は
                既に new_state に更新済みのため、判定用に別途受け取る)。
            time_sec: 現フレームの時刻。
        """
        if not self._enable_recovery_counter_carryover:
            self._ctx.stable_recovery_counters.clear()
            self._ctx.recovery_cells.clear()
            self._ctx.cnn_flicker_history.clear()
            return
        if old_state not in NON_STABLE_STATES:
            self._ctx.non_stable_entry_time_sec = time_sec
        self._maybe_clear_recovery_on_timeout(time_sec)

    def _maybe_clear_recovery_on_timeout(self, time_sec: float) -> None:
        """carryover 有効時、非 STABLE 滞在が上限秒数を超えたらクリアする.

        連鎖等で長時間 NON-STABLE のままだと盤面が実際に変化している
        可能性が高く、持ち越した復旧カウンタの証拠は信頼できないため
        安全側にクリアする (recovery_counter_carryover_max_sec)。
        """
        entry = self._ctx.non_stable_entry_time_sec
        if entry is None:
            return
        if time_sec - entry > self._recovery_counter_carryover_max_sec:
            self._ctx.stable_recovery_counters.clear()
            self._ctx.recovery_cells.clear()
            self._ctx.cnn_flicker_history.clear()
            self._ctx.non_stable_entry_time_sec = None

    def _update_pending_confirmation(self, cnn_board: "Board") -> None:
        """初回STABLE確定窓の pending_board/pending_count を1frame分更新する.

        盤面確定窓 3中2多数決 (stable_majority_window, 2026-08-13 user承認):
        True 時は直近 window 観測 (`confirm_window_history`) の多数決
        (`_majority_window_vote`) で判定し、多数決が成立した瞬間に
        pending_count を self._stable_n まで進めて即時確定させる
        (呼び出し元の `pending_count >= self._stable_n` 判定を変更せずに
        再利用するため)。不成立なら pending を未確定 (None/0) にリセットする。
        False (default) 時は従来の厳密連続一致を維持する
        (backwards compat、bit-identical)。

        Args:
            cnn_board: 現フレームの raw CNN 観測盤面。
        """
        if self._enable_stable_majority_window:
            self._ctx.confirm_window_history.append(cnn_board.copy())
            if (
                len(self._ctx.confirm_window_history)
                > self._stable_majority_window_frames
            ):
                self._ctx.confirm_window_history.pop(0)
            majority_board = _majority_window_vote(
                self._ctx.confirm_window_history, self._stable_majority_min_votes,
            )
            if majority_board is not None:
                self._ctx.pending_board = majority_board
                self._ctx.pending_count = self._stable_n
            else:
                self._ctx.pending_board = None
                self._ctx.pending_count = 0
            return
        if _boards_equal(self._ctx.pending_board, cnn_board):
            self._ctx.pending_count += 1
        else:
            self._ctx.pending_board = cnn_board.copy()
            self._ctx.pending_count = 1

    def _update_within_current_state(
        self, signals: DetectorSignals,
    ) -> None:
        """現 state を維持したまま内部メトリクスのみ更新。"""
        if self._ctx.state in NON_STABLE_STATES:
            # 復旧カウンタ carryover: NON-STABLE 滞在が長引く場合、上限秒数
            # 超過分をこの idle フレームでも検知してクリアする
            # (_apply_transition は state 遷移が起きた frame でしか動かない
            # ため、同一 state に留まり続けるケースはここで拾う必要がある)。
            if self._enable_recovery_counter_carryover:
                self._maybe_clear_recovery_on_timeout(signals.time_sec)
            # feat/gravity-settle-2026-06-05: GRAVITY_SETTLE 中は
            # non_stable_cnn_history に蓄積しない (連鎖後エフェクト残光・
            # 落下中ぷよの混入による F ガード汚染を防ぐ)。
            # GRAVITY_SETTLE → STABLE 遷移時の _merge_diff_only には
            # empty_guard として None が渡されるため F ガードは発火しない
            # (保守的: 連鎖後は全 cell を新規 STABLE で直接評価)。
            if self._ctx.state == BoardState.GRAVITY_SETTLE:
                # アクション中: 認識結果は盤面確定に使わない
                return
            # F (cycle 56): NON-STABLE 中の cnn_board 履歴を保持。
            # STABLE 復帰時の多数決ガードに使う。 最大 N frame 保持。
            self._ctx.non_stable_cnn_history.append(
                signals.cnn_board.copy(),
            )
            if (
                len(self._ctx.non_stable_cnn_history)
                > self._non_stable_history_size
            ):
                self._ctx.non_stable_cnn_history.pop(0)
            # アクション中: 認識結果は盤面確定に使わない
            return

        # B1 (M1 warmup guard): STABLE 遷移直後の warmup 期間中は confirmed 更新をスキップ。
        # エフェクト残光・背景誤認の CNN 出力が confirmed に焼き付くのを防ぐ。
        # last_stable_idx は更新し「STABLE 中」であることだけ伝える。
        if self._ctx.state == BoardState.STABLE and self._ctx.stable_warmup_remaining > 0:
            self._ctx.stable_warmup_remaining -= 1
            self._ctx.last_stable_idx = self._ctx.frame_idx
            return

        # MENU (試合復帰直後) または STABLE: 連続多数決で confirmed を更新。
        # MENU から N 連続一致で STABLE へ自動遷移する。
        cnn_board = signals.cnn_board
        self._update_pending_confirmation(cnn_board)

        if self._ctx.pending_count >= self._stable_n:
            # Phase C-7 (E-1): CNN を盤面更新ソースから完全排除
            # 初回 STABLE 確定 (= 試合開始時) は連続 N frame 観測の
            # pending_board を採用 (= 試合開始時の初期盤面確定)。
            # その後の連続多数決による confirmed 更新は停止し、盤面確定は
            # state 遷移時 (TSUMO/CHAIN/OJAMA → STABLE) の物理推論のみで行う。
            if self._ctx.confirmed_board is None:
                # cycle 71v (2026-05-14): 試合開始直後 window では物理ルール
                # 「試合開始フィールドは空」 を CNN 多数決より優先。 v51/v70 の
                # 2 試合目開始時の背景誤認が confirmed_board に乗るのを防ぐ。
                if signals.match_just_started:
                    self._ctx.confirmed_board = Board()  # all empty
                elif self._ctx.pending_board is not None:
                    # cycle 71v: 初回 STABLE 確定パスでも gravity filter を適用.
                    # 旧実装は pending_board.copy() で gravity filter が走らず、
                    # CNN の背景誤認が浮きぷよとして confirmed に残留した。
                    b = self._ctx.pending_board.copy()
                    _apply_gravity_filter(b)
                    self._ctx.confirmed_board = b
            # 既に confirmed あれば維持 (= CNN 多数決経路で更新しない)
            self._ctx.last_stable_idx = self._ctx.frame_idx
            self._ctx.state = BoardState.STABLE

        # 設計C 事後復旧ゲート: STABLE 確定後に実行。
        # warmup 期間中は発火させない (背景誤認残光期間を避けるため)。
        if (
            self._enable_stable_recovery_gate
            and self._ctx.state == BoardState.STABLE
            and self._ctx.confirmed_board is not None
            and self._ctx.stable_warmup_remaining == 0
        ):
            # 非対称化 有効時のみ方向1 (空→色) に短縮値を渡す。None なら従来通り
            # 全方向 recovery_min_frames (bit-identical、backwards compat)。
            _add_min = (
                self._recovery_add_min_frames
                if self._enable_asymmetric_recovery_min_frames
                else None
            )
            _apply_stable_recovery_gate(
                self._ctx, signals, self._recovery_min_frames,
                add_min_frames=_add_min,
                enable_column_partial_support=self._enable_column_partial_support,
                enable_cnn_flicker_hsv_fallback=self._enable_cnn_flicker_hsv_fallback,
                cnn_flicker_window_frames=self._cnn_flicker_window_frames,
                cnn_flicker_min_changes=self._cnn_flicker_min_changes,
                enable_effect_gate=self._enable_effect_gate,
                effect_gate_persist_sec=self._effect_gate_persist_sec,
                effect_gate_hard_freeze=self._effect_gate_hard_freeze,
            )


# ============================
# 設計C 事後復旧ゲート ヘルパー
# ============================


def _check_recovery_column(
    confirmed: "Board", col: int, candidates: list[tuple[int, int, int]],
    *,
    recovery_counters: "dict[tuple[int, int], int] | None" = None,
    support_min_frames: int = RECOVERY_COLUMN_SUPPORT_MIN_FRAMES,
) -> list[tuple[int, int, int]]:
    """列の重力整合チェック: 下から連続するブロックのみ復旧候補として残す.

    浮きぷよ防止 (安全弁C): 復旧候補セルの下に空 confirmed があれば浮きぷよに
    なるため除外する。列を下段から走査し、confirmed が空でない連続区間のみ許可。

    列ゲート緩和 (enable_column_partial_support, 2026-07-25): recovery_counters を
    渡すと、下のセルが confirmed==EMPTY でも stable_recovery_counters が
    support_min_frames 以上進行中 (= 復旧合意が積み上がりつつある) なら
    「支持セル」とみなし浮き扱いしない。 None (default) なら従来挙動と
    bit-identical (backwards compat)。

    Args:
        confirmed: 現在の confirmed_board。
        col: 対象列番号 (0-5)。
        candidates: [(row, col, recovery_color), ...] — 復旧候補リスト。
        recovery_counters: (row, col) → 連続合意フレーム数。None で無効化。
        support_min_frames: 支持セルとみなす最低カウンタ値。

    Returns:
        重力整合チェック通過後の復旧候補リスト。
    """
    # 対象列の既存 confirmed 色マップ (row → color)
    col_confirmed: dict[int, int] = {}
    for r in range(BOARD_ROWS):
        col_confirmed[r] = int(confirmed.get(r, col))

    # 候補を row の降順 (最下段優先) にソート
    col_candidates = sorted(
        [(r, c_col, color) for (r, c_col, color) in candidates if c_col == col],
        key=lambda x: -x[0],
    )
    if not col_candidates:
        return []

    # 最下段から連続して confirmed が空か候補セル (= 仮に復旧した場合) か確認
    # 合格した候補 (下から連続ブロック) のみを返す
    result: list[tuple[int, int, int]] = []
    # まず候補 row の集合を得る
    cand_rows = {r for (r, _, _) in col_candidates}

    for r_desc, c_col, color in col_candidates:
        # このセルの下 (r+1..12) に、確定 EMPTY かつ候補でもないセルがあれば浮き
        floating = False
        for below in range(r_desc + 1, BOARD_ROWS):
            below_v = col_confirmed[below]
            if below_v == COLOR_EMPTY and below not in cand_rows:
                # 列ゲート緩和: カウンタ進行中の支持セルなら浮き扱いしない
                if (
                    recovery_counters is not None
                    and recovery_counters.get((below, col), 0) >= support_min_frames
                ):
                    continue
                floating = True
                break
        if not floating:
            result.append((r_desc, c_col, color))
    return result


def _update_cnn_flicker_history_and_check(
    history: "dict[tuple[int, int], list[int]]",
    r: int,
    c: int,
    cnn_v: int,
    window_frames: int,
    min_changes: int,
) -> bool:
    """CNN 乱高下セル HSV フォールバック (#51 後半): セル (r, c) の CNN 出力
    履歴を更新し、直近 window_frames フレーム内の変化回数が min_changes 以上
    なら「乱高下中」と判定する。

    Args:
        history: ctx.cnn_flicker_history (in-place 更新)。
        r, c: 対象セル。
        cnn_v: 現フレームの CNN 出力色。
        window_frames: 履歴保持フレーム数。
        min_changes: 乱高下とみなす最小変化回数 (連続フレーム間の値変化回数)。

    Returns:
        True なら乱高下中 (= HSV を合意値とみなすべき)。
    """
    hist = history.setdefault((r, c), [])
    hist.append(cnn_v)
    if len(hist) > window_frames:
        del hist[: len(hist) - window_frames]
    changes = sum(1 for i in range(1, len(hist)) if hist[i] != hist[i - 1])
    return changes >= min_changes


def _update_effect_gate_hold(
    hold: "dict[tuple[int, int], tuple[int, float]]",
    cell: "tuple[int, int]",
    candidate_color: int,
    time_sec: float,
    persist_sec: float,
) -> bool:
    """エフェクト時間ゲート: 領域限定セルの持続時間を実秒ベースで追跡する.

    候補色が変わったら計測をリセットする (= 別の色に切り替わったら再度
    persist_sec 分の持続観測を要求する)。フレーム数でなく time_sec の差分で
    判定するため、30fps/60fps 動画が混在しても同一の実時間基準で公平に動作する
    (フレーム定数の fps 依存問題を避ける、EFFECT_PERSIST_SEC コメント参照)。

    Args:
        hold: ctx.effect_gate_hold (in-place 更新)。
        cell: (row, col)。
        candidate_color: 現フレームの合意値 (CNN==HSV 一致色)。
        time_sec: 現フレームの時刻。
        persist_sec: 確定に必要な持続秒数。

    Returns:
        True なら persist_sec 以上継続観測された (= 確定してよい)。
    """
    prev = hold.get(cell)
    if prev is None or prev[0] != candidate_color:
        hold[cell] = (candidate_color, time_sec)
        return False
    return (time_sec - prev[1]) >= persist_sec


def _update_burst_visual_gate(
    is_open: bool,
    opened_at: "float | None",
    quiet_since: "float | None",
    score: float,
    time_sec: float,
    *,
    open_threshold: float,
    close_threshold: float,
    min_window_sec: float,
    max_window_sec: float,
    quiescence_min_sec: float,
    force_close: bool = False,
) -> "tuple[bool, float | None, float | None]":
    """バースト視覚検出の Schmitt trigger 1frame更新 (stateless純関数)。

    2026-08-05 バーストガード再設計 (docs/BURST_GUARD_DESIGN_2026-08-05.md §2.2)。
    Window ON 中は無条件凍結 (呼び出し側が is_open を effect_gate_active な
    行の凍結条件として使う)。旧 `_update_effect_gate_hold` の persist逆転
    (issue b: バーストが0.4秒超続くと誤値が「安定」として採用されてしまう)
    を構造的に排除するため、「確定に必要な持続」ではなく「解除に必要な静穏」
    を計測する設計にする (§1.3)。

    Args:
        is_open: 直前frameのWindow状態。
        opened_at: Window が開いた time_sec (open中のみ値を持つ)。
        quiet_since: score が close_threshold 未満に落ちた最初の time_sec
            (close中に再びopen_threshold以上に戻ったらNoneにリセット)。
        score: 今frameの視覚スコア (compute_effect_glow_score の戻り値)。
        time_sec: 今frameの時刻。
        open_threshold: Window を開く閾値 (score >= で即時open)。
        close_threshold: 静穏判定の閾値 (score < が続くことを要求、
            open_threshold 以下の値を推奨 = ヒステリシス帯を作る)。
        min_window_sec: 一度開いたら最低この秒数は維持する
            (1リンクの演出持続時間 ≒0.2秒に対応、単発frameでの開閉振動防止)。
        max_window_sec: 安全弁。この秒数を超えたら score に関係なく強制close
            (視覚検出が誤って張り付いた場合の永久凍結防止)。
        quiescence_min_sec: close確定に必要な連続静穏秒数
            (リンク間flicker gap ≒0.1秒 の1回だけでは閉じないマージンを持たせる)。
        force_close: True の場合、他条件を無視して即時close
            (own_chain_active / all_clear_pending 等の外部安全条件)。

    Returns:
        (new_is_open, new_opened_at, new_quiet_since)
    """
    if force_close:
        return False, None, None

    if not is_open:
        if score >= open_threshold:
            return True, time_sec, None
        return False, None, None

    # is_open == True
    if opened_at is not None and (time_sec - opened_at) >= max_window_sec:
        return False, None, None  # 安全弁: 強制close

    if score < close_threshold:
        _quiet_since = quiet_since if quiet_since is not None else time_sec
        elapsed_open = time_sec - opened_at if opened_at is not None else 0.0
        quiescent = time_sec - _quiet_since
        if elapsed_open >= min_window_sec and quiescent >= quiescence_min_sec:
            return False, None, None
        return True, opened_at, _quiet_since
    # score >= close_threshold: まだバースト中、静穏タイマーをリセット
    return True, opened_at, None


def _recovery_or_effect_gate_pass(
    ctx: "StateContext",
    cell: "tuple[int, int]",
    confirmed_v: int,
    agreed_v: int,
    recovery_counters: "dict[tuple[int, int], int]",
    min_frames: int,
    add_min_frames: "int | None",
    effect_gate_active_rows: "frozenset[int] | None",
    effect_gate_persist_sec: float,
    effect_gate_hard_freeze: bool = False,
) -> bool:
    """1 cell 分の発火判定 (通常フレームカウント or エフェクト時間ゲート).

    `_collect_recovery_candidates` の 50 行規約超過を避けるため分離した
    ヘルパー。 effect_gate_active_rows が None、または cell の行がゲート対象
    外なら従来のフレームカウント判定 (bit-identical、backwards compat)。
    ゲート対象なら stable_recovery_counters でなく ctx.effect_gate_hold の
    実秒ベース持続確認に切り替える (2 つのカウンタは排他的に使う、単位混在防止)。

    effect_gate_hard_freeze=True (2026-08-05 バーストガード再設計 §3):
    ゲート対象 cell は持続確認を一切行わず、Window ON の間は無条件で発火
    させない (persist逆転の構造的排除)。既定 False では従来動作と
    bit-identical。
    """
    r, c = cell
    is_gated = (
        effect_gate_active_rows is not None and r in effect_gate_active_rows
    )
    if is_gated:
        recovery_counters.pop(cell, None)
        if effect_gate_hard_freeze:
            ctx.effect_gate_hold.pop(cell, None)
            return False
        return _update_effect_gate_hold(
            ctx.effect_gate_hold, cell, agreed_v,
            ctx.time_sec, effect_gate_persist_sec,
        )
    if effect_gate_active_rows is not None:
        ctx.effect_gate_hold.pop(cell, None)
    new_count = recovery_counters.get(cell, 0) + 1
    recovery_counters[cell] = new_count
    if confirmed_v == COLOR_EMPTY:
        effective_min = (
            add_min_frames if add_min_frames is not None else min_frames
        )
        return new_count >= effective_min
    return new_count >= min_frames


def _collect_recovery_candidates(
    ctx: "StateContext",
    cnn_board: "Board",
    hsv_board: "Board",
    min_frames: int,
    *,
    add_min_frames: "int | None" = None,
    enable_cnn_flicker_hsv_fallback: bool = False,
    cnn_flicker_window_frames: int = CNN_FLICKER_WINDOW_FRAMES,
    cnn_flicker_min_changes: int = CNN_FLICKER_MIN_CHANGES,
    effect_gate_active_rows: "frozenset[int] | None" = None,
    effect_gate_persist_sec: float = EFFECT_PERSIST_SEC,
    effect_gate_hard_freeze: bool = False,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """各セルの合意値チェックとカウンタ更新を行い、候補を方向別に返す.

    双方向の発火候補を収集する:
        - add_candidates: 方向1 (空→色) — 重力整合チェックが必要
        - fix_candidates: 方向2/3 (色→空/色→別色) — 重力整合チェック不要

    安全弁: UNKNOWN(10) は合意値として無効 (RECOVERY_EXCLUDED_COLORS)。

    CNN 乱高下セル HSV フォールバック (enable_cnn_flicker_hsv_fallback,
    #51 後半, 2026-07-26): True の場合、直近 N フレームの CNN 出力の変化回数
    が閾値以上のセルは「乱高下中」とみなし、CNN==HSV の一致要件を HSV 値との
    自明一致 (agreed_v = hsv_v) に置き換える。これにより光沢ハイライトで
    判定境界に張り付き反転を続ける CNN の代わりに、一貫して正しい HSV の
    合意形成を許可する。default False = 従来挙動完全維持 (backwards compat)。

    Args:
        ctx: StateContext (stable_recovery_counters / cnn_flicker_history を
            in-place 更新)。
        cnn_board: CNN 認識盤面。
        hsv_board: HSV 認識盤面。
        min_frames: 発火に必要な連続フレーム数 (方向2/3 = 色→空/色→色)。
        add_min_frames: 方向1 (空→色) の発火に必要な連続フレーム数。None なら
            min_frames を使う (従来と bit-identical、backwards compat)。非対称化
            (enable_asymmetric_recovery_min_frames) 有効時のみ短縮値が渡る。
        enable_cnn_flicker_hsv_fallback: True で乱高下セルの HSV フォール
            バックを有効化する。
        cnn_flicker_window_frames: 乱高下判定用の履歴保持フレーム数。
        cnn_flicker_min_changes: 乱高下とみなす最小変化回数。
        effect_gate_active_rows: エフェクト時間ゲート (2026-08-03) 対象行。
            None (default) ならゲート無効 (bit-identical、backwards compat)。
            非 None なら、その行に属する cell は frame カウントでなく
            ctx.effect_gate_hold の実秒ベース持続確認で判定する
            (`_recovery_or_effect_gate_pass` 参照)。
        effect_gate_persist_sec: エフェクト時間ゲートの確定に必要な持続秒数。
        effect_gate_hard_freeze: 2026-08-05 バーストガード再設計 §3。True で
            ゲート対象 cell の持続確認を無効化し、Window ON 中は無条件で
            発火させない。既定 False = 従来動作と bit-identical。

    Returns:
        (add_candidates, fix_candidates) のタプル。
        各要素は [(row, col, target_color), ...] 形式。
    """
    recovery_counters = ctx.stable_recovery_counters
    confirmed = ctx.confirmed_board
    assert confirmed is not None

    def _reset_counters(r: int, c: int) -> None:
        """通常カウンタ + エフェクト時間ゲートの持続計測を両方リセットする.

        effect_gate_hold を消し忘れると、フリッカ解消後に同じ色が
        (無関係な理由で) 再出現した際、古い time_sec が残ったまま
        `_update_effect_gate_hold` が「既に持続済み」と誤判定して即確定して
        しまう (= ゲートが機能しない致命的バグ)。両方を必ず揃えて消す。
        """
        recovery_counters.pop((r, c), None)
        ctx.effect_gate_hold.pop((r, c), None)

    add_candidates: list[tuple[int, int, int]] = []
    fix_candidates: list[tuple[int, int, int]] = []

    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            confirmed_v = int(confirmed.get(r, c))
            cnn_v = int(cnn_board.get(r, c))
            hsv_v = int(hsv_board.get(r, c))

            # UNKNOWN は合意値として無効 → カウンタリセット (raw 値で判定)
            if cnn_v in RECOVERY_EXCLUDED_COLORS or hsv_v in RECOVERY_EXCLUDED_COLORS:
                _reset_counters(r, c)
                continue

            # CNN 乱高下セル HSV フォールバック: 乱高下中なら HSV を合意値と
            # みなす (agreed_v = hsv_v で以降の一致判定を自明成立させる)。
            agreed_v = cnn_v
            if enable_cnn_flicker_hsv_fallback:
                is_flickering = _update_cnn_flicker_history_and_check(
                    ctx.cnn_flicker_history, r, c, cnn_v,
                    cnn_flicker_window_frames, cnn_flicker_min_changes,
                )
                if is_flickering:
                    agreed_v = hsv_v

            # CNN≠HSV (乱高下フォールバック未発動時) → 独立二重合意なし → カウンタリセット
            if agreed_v != hsv_v:
                _reset_counters(r, c)
                continue
            # confirmed == 合意値 → 差分なし → カウンタリセット
            if confirmed_v == agreed_v:
                _reset_counters(r, c)
                continue

            # CNN==HSV (=合意値) かつ confirmed != 合意値 → 発火判定
            # (通常は frame カウント、エフェクトゲート対象 cell は実秒ベース)。
            passed = _recovery_or_effect_gate_pass(
                ctx, (r, c), confirmed_v, agreed_v, recovery_counters,
                min_frames, add_min_frames,
                effect_gate_active_rows, effect_gate_persist_sec,
                effect_gate_hard_freeze,
            )
            if not passed:
                continue
            if confirmed_v == COLOR_EMPTY:
                add_candidates.append((r, c, agreed_v))  # 方向1 (重力整合チェック必要)
            else:
                fix_candidates.append((r, c, agreed_v))  # 方向2/3

    return add_candidates, fix_candidates


def _build_recovery_support_board(
    confirmed: "Board",
    recovery_counters: "dict[tuple[int, int], int]",
    cnn_board: "Board",
    min_frames: int = RECOVERY_COLUMN_SUPPORT_MIN_FRAMES,
) -> "Board":
    """列ゲート緩和 (enable_column_partial_support) 用の support_board を構築.

    counter >= min_frames のセルに CNN 観測色をプレースホルダとして書き込み、
    最終 _apply_gravity_filter がそのセルを「浮き判定の穴」として扱わない
    ようにする。EMPTY/UNKNOWN は安全弁として書き込まない。

    Args:
        confirmed: 現在の confirmed_board (この copy をベースにする)。
        recovery_counters: (row, col) → 連続合意フレーム数。
        cnn_board: 現フレームの CNN 認識盤面 (プレースホルダ色の取得元)。
        min_frames: プレースホルダを書き込む最低カウンタ値。

    Returns:
        support_board (confirmed のコピー + プレースホルダ)。
    """
    from src.board import COLOR_EMPTY

    support = confirmed.copy()
    for (r, c), count in recovery_counters.items():
        if count < min_frames:
            continue
        placeholder = int(cnn_board.get(r, c))
        if placeholder == COLOR_EMPTY or placeholder in RECOVERY_EXCLUDED_COLORS:
            continue
        support.set(r, c, placeholder)
    return support


def _apply_stable_recovery_gate(
    ctx: "StateContext",
    signals: "DetectorSignals",
    min_frames: int,
    *,
    add_min_frames: "int | None" = None,
    enable_column_partial_support: bool = False,
    enable_cnn_flicker_hsv_fallback: bool = False,
    cnn_flicker_window_frames: int = CNN_FLICKER_WINDOW_FRAMES,
    cnn_flicker_min_changes: int = CNN_FLICKER_MIN_CHANGES,
    enable_effect_gate: bool = False,
    effect_gate_persist_sec: float = EFFECT_PERSIST_SEC,
    effect_gate_hard_freeze: bool = False,
) -> None:
    """設計C 事後復旧ゲート本体 (in-place で confirmed_board を更新).

    双方向発火条件 (STABLE state、hsv_board!=None、warmup 外):
        方向1 (空→色): confirmed=EMPTY, CNN==HSV=有効色 が add_min_frames 連続
                       (None なら min_frames)
        方向2 (色→空): confirmed=色,   CNN==HSV=EMPTY  が min_frames 連続
        方向3 (色→色): confirmed=色A,  CNN==HSV=色B    が min_frames 連続

    安全弁A: hsv_board が None (= CNN 単独) のとき発火しない。
    安全弁B: UNKNOWN(10) は合意値として無効 (RECOVERY_EXCLUDED_COLORS)。
    安全弁C: 方向1のみ列単位の重力整合チェック (_check_recovery_column)。
             方向2/3 は除去・上書き方向なので適用しない。

    振動抑制: 復旧済みセルを recovery_cells に記録し、
    non_stable_cnn_history に復旧後盤面を強制投入する。

    Args:
        ctx: StateContext (in-place 更新)。
        signals: 現フレームのシグナル (cnn_board, hsv_board を参照)。
        min_frames: 発火に必要な連続フレーム数。
        enable_column_partial_support: 列ゲート緩和 (2026-07-25)。True にすると
            安全弁C の浮き判定と最終重力フィルタに stable_recovery_counters
            由来の support_board を渡し、「支持セルが復旧カウント進行中」の
            ケースを浮きぷよ扱いしないようにする。
            default False = 従来挙動完全維持 (backwards compat)。
        enable_cnn_flicker_hsv_fallback: CNN 乱高下セル HSV フォールバック
            (#51 後半, 2026-07-26)。True にすると光沢ハイライトで判定境界に
            張り付き反転を続ける CNN 出力セルを検出し、その間 HSV を合意値
            とみなす。default False = 従来挙動完全維持 (backwards compat)。
        cnn_flicker_window_frames: 乱高下判定用の履歴保持フレーム数。
        cnn_flicker_min_changes: 乱高下とみなす最小変化回数。
        enable_effect_gate: エフェクト時間ゲート (2026-08-03)。True かつ
            signals.effect_gate_window_active=True の間、EFFECT_GATE_TOP_ROWS
            の cell は frame カウントでなく実秒ベース持続確認に切り替わる。
            default False = 従来挙動完全維持 (backwards compat)。
        effect_gate_persist_sec: エフェクト時間ゲートの確定に必要な持続秒数。
        effect_gate_hard_freeze: 2026-08-05 バーストガード再設計 §3。True で
            `_collect_recovery_candidates` の持続確認を無効化しハード凍結する。
            既定 False = 従来動作と bit-identical。
    """
    if ctx.confirmed_board is None:
        return
    # 安全弁A: hsv_board が None → 発火しない
    hsv_board = signals.hsv_board
    if hsv_board is None:
        return

    # エフェクト時間ゲート: フラグ ON かつ今フレームが window 中 (相手連鎖中/
    # 自お邪魔着弾直後) のときだけ EFFECT_GATE_TOP_ROWS を対象行にする。
    # フラグ OFF なら None (= 全 cell 従来ロジック、bit-identical)。
    _effect_gate_rows: "frozenset[int] | None" = None
    if enable_effect_gate:
        _effect_gate_rows = (
            EFFECT_GATE_TOP_ROWS if signals.effect_gate_window_active
            else frozenset()
        )

    # パス1: 候補収集 + カウンタ更新
    add_candidates, fix_candidates = _collect_recovery_candidates(
        ctx, signals.cnn_board, hsv_board, min_frames,
        add_min_frames=add_min_frames,
        enable_cnn_flicker_hsv_fallback=enable_cnn_flicker_hsv_fallback,
        cnn_flicker_window_frames=cnn_flicker_window_frames,
        cnn_flicker_min_changes=cnn_flicker_min_changes,
        effect_gate_active_rows=_effect_gate_rows,
        effect_gate_persist_sec=effect_gate_persist_sec,
        effect_gate_hard_freeze=effect_gate_hard_freeze,
    )

    if not add_candidates and not fix_candidates:
        return

    # パス2 (方向1のみ): 列ごとに重力整合チェック (安全弁C)
    passed_add: list[tuple[int, int, int]] = []
    cols_with_add = {c for (_, c, _) in add_candidates}
    recovery_counters_for_check = (
        ctx.stable_recovery_counters if enable_column_partial_support else None
    )
    for col in cols_with_add:
        passed_add.extend(
            _check_recovery_column(
                ctx.confirmed_board, col, add_candidates,
                recovery_counters=recovery_counters_for_check,
            ),
        )

    # パス3: confirmed 更新 (方向1/2/3 合算)
    all_passed = passed_add + fix_candidates
    if not all_passed:
        return

    for r, c, target in all_passed:
        ctx.confirmed_board.set(r, c, target)
        ctx.recovery_cells.add((r, c))
        # カウンタリセット (発火済み)
        ctx.stable_recovery_counters.pop((r, c), None)

    # 方向1に対しのみ重力整合最終確認 (方向2/3は浮きぷよを生じさせない)
    if passed_add:
        support_board: "Board | None" = None
        if enable_column_partial_support:
            support_board = _build_recovery_support_board(
                ctx.confirmed_board, ctx.stable_recovery_counters, signals.cnn_board,
            )
        _apply_gravity_filter(ctx.confirmed_board, support_board=support_board)

    # 振動抑制: non_stable_cnn_history に復旧後盤面を強制投入。
    # 次の NON-STABLE→STABLE 遷移の F ガード多数決で復旧色が票を持ち再崩壊防止。
    reinforce = ctx.confirmed_board.copy()
    ctx.non_stable_cnn_history.insert(0, reinforce)
    if len(ctx.non_stable_cnn_history) > DEFAULT_NON_STABLE_HISTORY_SIZE:
        ctx.non_stable_cnn_history.pop(-1)


# ============================
# 補助 detector (テスト + デバッグ用)
# ============================


class NullDetector:
    """常に None を返す detector (= state machine 単体テスト用)."""

    def detect(
        self, ctx: StateContext, signals: DetectorSignals,
    ) -> BoardState | None:
        return None


__all__ = [
    "BOARD_COLS",
    "BOARD_ROWS",
    "BoardState",
    "BoardStateMachine",
    "DEFAULT_STABLE_FRAME_COUNT",
    "DetectorSignals",
    "NON_STABLE_STATES",
    "NullDetector",
    "_apply_gravity_filter",
    "_apply_stable_recovery_gate",
    "_build_recovery_support_board",
    "_check_recovery_column",
    "_collect_recovery_candidates",
    "_merge_diff_only",
    "_should_keep_puyo_over_empty",
    "_vote_majority_board",
    "DEFAULT_NON_STABLE_HISTORY_SIZE",
    "DEFAULT_EMPTY_TO_COLOR_MIN_VOTES",
    "DEFAULT_INITIAL_CONFIRM_MIN_VOTES",
    "_build_initial_confirmed_board",
    "STABLE_WARMUP_FRAMES",
    "STABLE_RECOVERY_MIN_FRAMES",
    "STABLE_RECOVERY_ADD_MIN_FRAMES",
    "RECOVERY_COLUMN_SUPPORT_MIN_FRAMES",
    "RECOVERY_EXCLUDED_COLORS",
    "CNN_FLICKER_WINDOW_FRAMES",
    "CNN_FLICKER_MIN_CHANGES",
    "_update_cnn_flicker_history_and_check",
    "StateContext",
    "StateTransitionDetector",
    # feat/gravity-settle-2026-06-05: GRAVITY_SETTLE 関連定数
    "GRAVITY_SETTLE_MIN_FRAMES",
    "GRAVITY_SETTLE_MAX_SEC",
    "GRAVITY_SETTLE_PHYSICS_CLEAR_MIN",
    "GRAVITY_SETTLE_PUYO_DIFF_THRESHOLD",
    # エフェクト時間ゲート (2026-08-03)
    "EFFECT_GATE_TOP_ROWS",
    "EFFECT_PERSIST_SEC",
    "_update_effect_gate_hold",
    "_recovery_or_effect_gate_pass",
    # バーストガード再設計 (2026-08-05)
    "_update_burst_visual_gate",
    # バーストガード Stage1.5 (2026-08-05 アーキ追補)
    "_TRANSITION_MERGE_GUARD_SCOPE",
    "_filter_transition_new_cnn_for_burst_guard",
]
