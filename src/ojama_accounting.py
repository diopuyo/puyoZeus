"""お邪魔ぷよ会計モジュール (Ojama Accounting) — アーキ案A 全面差し替え版。

2026-06-10 マージンタイム試合相対修正:
    - _elapsed() が _match_start_sec=None のとき t_sec をクリップ先頭からの経過秒として
      返していた。viz は reset(match_start_sec=...) を呼ばないため常に None のまま →
      クリップ全体経過秒(最大168秒)がマージンタイム計算に渡り過剰計上の原因になっていた。
    - 修正: 試合境界(score大幅減少 or MENU)検知時に _match_start_sec = t_sec を設定。
      最初の試合は「最初に score を観測した時刻」を match_start とみなす。
    - _elapsed() は _match_start_sec=None を「初期化前」として扱い、
      初期化済みの場合のみ t_sec - _match_start_sec を返す。
      未初期化の場合は 0.0 を返す(マージンタイム非適用 = 安全側)。

2026-07-09 物理イベント基準 finalize:
    - 連鎖終了検知を「物理イベント基準」に強化。
    - 新トリガー: chain_active 中に TSUMO_FALL/OJAMA_FALL への遷移を検知したとき
      score 上昇停止ゲート付きで settle 待ちを開始 → K_SETTLE_FRAMES 後に finalize。
    - 補完トリガー: (CHAIN|GRAVITY_SETTLE)→STABLE は取りこぼし補完として併存。
    - 二重 finalize 防止: coalesce window + _already_started 判定で既存通り保護。
    - 狙い: 撃ち合い時の 2P 副連鎖(TSUMO_FALL で区切り)を正確に捕捉。

2026-06-09 設計方針:
    - 生成は連鎖終了時に一括: chain_total_score // rate, 繰越 leftover を各 side 独立で管理。
    - 相殺の正しい向き: 自分が連鎖を撃ったとき、自分に向かってくる予告(incoming)を打ち消す。
      旧実装バグ(小連鎖が大予告を消し有利不利反転)を根本修正。
    - 全消し特別処理は廃止: ALL_CLEAR_BONUS を score に乗せる処理は得点計算側の責務。
    - 連鎖終了検知: on_state_transition で (CHAIN or GRAVITY_SETTLE) → STABLE 遷移を捕捉。
    - score None 時: chain_end_pending=True にして後続フレームで遅延確定。
      タイムアウト(30フレーム≒1秒)で破棄+warning。
    - 試合境界: score 減少(≥SCORE_RESET_THRESHOLD)または MENU 遷移で reset。

後方互換:
    - OjamaAccountSnapshot の既存フィールドは全て維持(削除・順序変更禁止)。
    - pending_p1/p2 には forecast_incoming を格納(= 正しい予告)。
    - all_clear_pending_p1/p2 は False 固定(廃止だが削除不可)。
    - 末尾追加フィールド: forecast_p1/p2, chain_total_score_p1/p2,
      chain_end_triggered_p1/p2, score_at_chain_start_p1/p2。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.scoring import (
    OJAMA_MAX_DROP_PER_TURN,
    OJAMA_RATE_STANDARD,
    score_to_ojama,
)

if TYPE_CHECKING:
    pass  # 型ヒント専用 import はここに追加

logger = logging.getLogger(__name__)

# ============================
# 定数
# ============================

# 後方互換のため旧定数名も維持(外部参照あり)
CHAIN_FIRE_MIN_SCORE: int = 40  # 後方互換 export 用(新実装では未使用)
VISIBLE_OJAMA_MISMATCH_THRESHOLD: int = 6  # 後方互換 export 用
CONFIDENCE_SCORE_OCR_ONLY: float = 0.85
CONFIDENCE_VISUAL_AGREE: float = 0.95
CONFIDENCE_VISUAL_MISMATCH_PENALTY: float = 0.30
DROP_SANITY_CLAMP: int = OJAMA_MAX_DROP_PER_TURN  # 後方互換 export 用

# 試合境界検知: score がこの値以上減少したら試合切り替えと判定
SCORE_RESET_THRESHOLD: int = 500

# 理論落下: TSUMO 着地 1 ターン当たりの最大 drain 量
THEORY_DROP_PER_TURN: int = OJAMA_MAX_DROP_PER_TURN

# オンフィールド容量: 可視フィールド(12行×6列)全セル数
ON_FIELD_CAP: int = 72

# forecast のオンフィールド有界上限 (指標用)
PENDING_HARD_CAP: int = ON_FIELD_CAP

# forecast の絶対サニティ上限(約3画面分)
PENDING_ABS_CAP: int = ON_FIELD_CAP * 3

# chain_total の下限ガード: score OCR 端数誤読による幻の連鎖を弾く
# 根拠: 1連鎖最小スコア ≈ 4ぷよ×10×(連鎖ボーナス0) = 40 点
# 40点未満の chain_total は正当な連鎖でなく OCR 誤読の可能性が極めて高い。
# お邪魔1個=70点未満のノイズのみ対象。正当な小連鎖(1-2連鎖=数十~数百点)は
# chain_total=40点以上になるため弾かれない。
CHAIN_TOTAL_MIN_SCORE: int = CHAIN_FIRE_MIN_SCORE  # = 40

# chain_total_score のサニティ上限: 1試合の最大スコアを超えたら OCR 異常
# 実測: ぷよぷよeスポーツ上級者試合で最大でも約 200,000 点以内
CHAIN_TOTAL_SANITY_MAX: int = 200_000

# cap 切り捨てなし並行帳簿 (pending_p1/p2_uncapped) のサニティ上限 (2026-08-24)。
# PENDING_ABS_CAP(216) は表示/指標用の絶対上限として妥当だが、致死判定
# (kill_override) の「相殺の引き算」に使うと、実送付517個が216に丸められた後で
# 相手の720個との差を取る形になり、架空の攻撃 (720-216=504、真値は 720-517=203)
# が生まれる (memory project_pm100_display_flip_2026-08-24 根因②)。
# 並行帳簿は cap で切らず、既存サニティ定数 CHAIN_TOTAL_SANITY_MAX
# (1連鎖の得点上限200,000点) をおじゃま換算した値だけを防波堤にする
# (新しいマジックナンバーを導入しない。200,000点 ÷ 70点/個 = 2,857個)。
PENDING_UNCAPPED_SANITY_MAX: int = CHAIN_TOTAL_SANITY_MAX // OJAMA_RATE_STANDARD

# 連鎖終了後 score None が続いた場合のタイムアウトフレーム数(≒ 1秒 @ 30fps)
CHAIN_END_PENDING_TIMEOUT_FRAMES: int = 30

# ===== score settle 待ち =====
# 連鎖の真の終了(スコアが上昇を止めて落ち着いた)を検知するための連続不変フレーム数閾値。
#
# 設計根拠:
#   - 実機観測: 連鎖内のスコア表示ステップ間の「止まっているように見えるポーズ」は
#     最大 ~0.4秒 ≒ 12フレーム (@30fps)。大連鎖(4連鎖以上)でも各ステップは
#     ~0.1〜0.3秒で進行し、最終ステップ後のポーズが最も長い。
#   - 安全側として 12フレームの 1.67倍 = 20フレーム (≒0.67秒) を採用。
#     → 連鎖内ポーズ(最大12f)を誤 settle 判定しない十分なマージン。
#   - 次連鎖との混合は起こらない:
#     次の連鎖を撃つには STABLE で TSUMO 着地(≥2〜3秒)が必要なため、
#     settle 待ち20フレーム中に次連鎖のスコア上昇が始まることはない。
K_SETTLE_FRAMES: int = 20

# MENU 状態の連続フレーム数でリセット
MENU_RESET_CONSEC_FRAMES: int = 3

# 連鎖合体ウィンドウ (秒): finalize 完了後この時間以内に再び CHAIN 遷移が来ても
# 同一連鎖の state 明滅とみなし score_at_chain_start を上書きしない(< で比較)。
# 実機観測:
#   - 連鎖中 CHAIN↔STABLE↔GRAVITY_SETTLE の state 明滅は 0.1〜2.0 秒以内に収まる
#   - 実際に次の連鎖を撃つには STABLE で TSUMO 着地 (≥1 ターン) が必要
#   - TSUMO_FALL 時間 + 次 CHAIN 開始まで最低 2-3 秒かかる (@ 30fps, 実機タイミング)
# 2.5 秒: state 明滅 (最大 2.0 秒) を捕捉しつつ、次の本物の連鎖 (≥ 3 秒後) を区別。
CHAIN_COALESCE_WINDOW_SEC: float = 2.5


# ============================
# 相殺の純関数 (stateless、2026-08-03 抽出)
# ============================
#
# _finalize_chain_end 内でインライン実装されていた「相殺の正しい向き」
# (1. 生成 gen で自分の incoming を相殺 2. 余剰を相手の incoming に加算)
# を、状態を持たない純関数として抽出したもの。scripts側 (打ち合い計測器の
# 「空中おじゃまの相殺会計」欠陥G対処) が同じ計算をオフライン評価で再利用
# できるようにする (コピペ再実装しない、_finalize_chain_end もこの関数を
# 呼ぶよう更新済み)。

def cancel_own_pending_then_send_surplus(
    gen: int, own_pending: int, other_pending: int,
) -> tuple[int, int]:
    """自分の生成量 gen で自分の pending を相殺し、余剰を相手の pending に送る。

    Args:
        gen: 今回の連鎖で生成したお邪魔数 (0以上)。
        own_pending: 自分に対して既に確定している pending (=forecast_incoming)。
        other_pending: 相手に対して既に確定している pending。

    Returns:
        (相殺後の own_pending, 余剰加算後の other_pending) のタプル。
    """
    canceled = min(gen, own_pending)
    new_own_pending = own_pending - canceled
    surplus = gen - canceled
    new_other_pending = other_pending + surplus
    return new_own_pending, new_other_pending


# ============================
# スナップショット (stateless)
# ============================

@dataclass(frozen=True)
class OjamaAccountSnapshot:
    """1 時刻の両者お邪魔ぷよ会計スナップショット。

    後方互換:
        既存フィールドは全て維持(削除・順序変更禁止)。
        pending_p1/p2 = forecast_incoming(自分に向かう予告個数)。
        all_clear_pending_p1/p2 = False 固定(廃止)。

    追加フィールド(末尾):
        forecast_p1/p2: alias of pending_p1/p2(予告個数、画面表示値と一致目標)。
        chain_total_score_p1/p2: 最後の連鎖合計得点(検証用)。
        chain_end_triggered_p1/p2: この時刻に連鎖終了イベントが立ったか(検証用)。
        score_at_chain_start_p1/p2: 連鎖開始直前 score(検証用)。
    """
    t_sec: float
    # --- 既存フィールド (変更禁止) ---
    pending_p1: int              # = forecast_incoming to p1 (自分に向かう予告)
    pending_p2: int              # = forecast_incoming to p2
    total_generated_by_p1: int
    total_generated_by_p2: int
    total_offset_by_p1: int
    total_offset_by_p2: int
    total_dropped_to_p1: int     # tsumo_settled drain の累積
    total_dropped_to_p2: int
    net_ojama_balance: int       # pending_p2 - pending_p1 (正=1P有利)
    overflow_risk_p1: bool
    overflow_risk_p2: bool
    confidence: float            # 固定値 CONFIDENCE_SCORE_OCR_ONLY
    leftover_p1: int
    leftover_p2: int
    all_clear_pending_p1: bool   # 廃止=常 False
    all_clear_pending_p2: bool   # 廃止=常 False
    # 修正C 追加フィールド (backwards compat: 末尾追加)
    pending_p1_capped: int = 0
    pending_p2_capped: int = 0
    net_balance_capped: int = 0
    offboard_p1: int = 0
    offboard_p2: int = 0
    # --- 新フィールド (末尾追加、後方互換) ---
    forecast_p1: int = 0                  # pending_p1 のエイリアス(予告個数)
    forecast_p2: int = 0                  # pending_p2 のエイリアス(予告個数)
    chain_total_score_p1: int = 0         # 最後の連鎖合計得点 (1P)
    chain_total_score_p2: int = 0         # 最後の連鎖合計得点 (2P)
    chain_end_triggered_p1: bool = False  # この時刻に連鎖終了イベントが立ったか (1P)
    chain_end_triggered_p2: bool = False  # この時刻に連鎖終了イベントが立ったか (2P)
    score_at_chain_start_p1: int | None = None   # 連鎖開始直前 score (1P)
    score_at_chain_start_p2: int | None = None   # 連鎖開始直前 score (2P)
    # --- cap切り捨てなし並行帳簿 (2026-08-24、kill判定入力専用。末尾追加) ---
    # PENDING_ABS_CAP(216) の切り捨てを行わずに同じ相殺/落下規則で回した値。
    # 表示/指標には使わない (従来の pending_p1/p2 を維持)。cap 適用後に相殺が
    # 走ると capped 側は「架空の余剰」を相手へ送るため、本フィールドは
    # pending_p1/p2 より小さくなることがある (どちらが大きいかは履歴依存)。
    # 既定 0 = 本 dataclass を直接構築する旧来コード (テスト等) では未使用のまま。
    pending_p1_uncapped: int = 0
    pending_p2_uncapped: int = 0


# ============================
# 内部状態 (1 サイド分)
# ============================

@dataclass
class _SideState:
    """片側 (1P or 2P) のお邪魔会計内部状態。"""
    leftover: int = 0                    # score 換算端数繰越
    total_generated: int = 0             # 累積生成量 (相殺前)
    total_offset: int = 0               # 累積相殺量
    total_dropped: int = 0              # 累積落下 drain 量
    forecast_incoming: int = 0          # 自分に向かう予告お邪魔個数
    # cap切り捨てなし並行帳簿 (2026-08-24)。forecast_incoming と同じ相殺/落下
    # 規則で更新するが PENDING_ABS_CAP で切らない (サニティ上限は
    # PENDING_UNCAPPED_SANITY_MAX のみ)。kill_override の相殺の引き算専用。
    forecast_incoming_uncapped: int = 0
    # --- 連鎖終了検知 ---
    chain_active: bool = False           # 連鎖中フラグ
    score_at_chain_start: int | None = None  # 連鎖開始直前の score (last_valid_score から設定)
    chain_end_pending: bool = False      # score None 時の遅延確定待ち
    pending_since_frame: int = 0        # chain_end_pending 開始フレーム番号
    # --- 最後に読めた有効 score ---
    last_valid_score: int | None = None  # 毎フレーム非 None score を更新(連鎖開始前に使用)
    # --- score settle 待ち (連鎖の真の終了検知) ---
    # 既存 chain_end_pending (score None 遅延確定) を内包した上位状態。
    # (CHAIN or GRAVITY_SETTLE)→STABLE の遷移後、スコアが K_SETTLE_FRAMES フレーム
    # 連続で変化しなくなった時点を「連鎖の真の終了」として finalize する。
    # score None フレームはカウントせず「まだ上昇中の可能性」として待機継続。
    score_settle_pending: bool = False    # settle 待ち中フラグ
    score_settle_candidate: int | None = None  # settle 待ち中の score 候補(現在の最大値)
    score_settle_consec: int = 0          # score が変化しなかった連続フレーム数
    score_settle_since_frame: int = 0     # settle 待ち開始フレーム番号
    # --- 連鎖間基準スコア (新方針: 連鎖間 stable スコア差分で chain_total を算出) ---
    # 連鎖中でない STABLE フレームで score を読めたときに更新する。
    # 「前の連鎖終了直後の落ち着いた score」を基準(0)とし、
    # 次の finalize 時に score_after との差分を chain_total とする。
    # これにより連鎖開始検知(score_at_chain_start)への依存を最小化する。
    last_stable_score: int | None = None  # 連鎖間の落ち着いた STABLE score 基準値
    # --- snapshot 検証用 ---
    last_chain_total_score: int = 0     # 最後の連鎖合計得点
    chain_end_triggered: bool = False   # 今フレームで連鎖終了イベントが立ったか
    # --- MENU state 管理 ---
    menu_consec_frames: int = 0         # MENU 連続フレーム数
    # --- state 明滅デバウンス ---
    # finalize 完了時刻 (秒)。CHAIN_COALESCE_WINDOW_SEC 以内の再 chain_start は
    # score_at_chain_start を上書きせず 1 連鎖 = 1 finalize を保証する。
    chain_finalized_at_sec: float | None = None  # 最後に finalize した時刻
    # 最後に finalize した時点の score_after (本物の2本目連鎖判定用)。
    # coalesce window 内で last_valid_score > last_finalized_score であれば
    # スコアが前回 finalize から増加しており、本物の新規連鎖と判断できる。
    # last_stable_score は finalize 後も STABLE フレームで更新されてしまうため
    # 使えない(直後の STABLE スコアで上書きされると差分がゼロになる)。
    last_finalized_score: int | None = None  # 最後に finalize した時の score_after


# ============================
# Tracker (stateful wrapper)
# ============================

class OjamaAccountingTracker:
    """試合 1 本分のお邪魔ぷよ会計を管理する状態保持 wrapper (アーキ案A)。

    CLAUDE.md 規約「state は外部 wrapper」準拠。

    使い方(新 API):
        tracker = OjamaAccountingTracker()
        tracker.reset()
        # 各フレームで state 遷移を通知
        tracker.on_state_transition("p1", prev_state, curr_state, score, t_sec)
        # tsumo 着地時に drain
        tracker.on_tsumo_settled("p1", t_sec)
        snap = tracker.get_snapshot(t_sec)

    後方互換 API(旧 visualize_recognition.py 呼出):
        update_from_score / update_from_boards / update_accounting_with_chain
        これらは新実装に内部的にルーティングされる。
    """

    def __init__(
        self,
        ojama_rate_base: int = OJAMA_RATE_STANDARD,
        overflow_threshold: int = OJAMA_MAX_DROP_PER_TURN,
    ) -> None:
        self._rate_base = int(ojama_rate_base)
        self._overflow_threshold = int(overflow_threshold)
        self._p1 = _SideState()
        self._p2 = _SideState()
        self._frame_idx: int = 0
        self._match_start_sec: float | None = None
        # score 前回値 (試合境界 reset 用)
        self._prev_score_p1: int | None = None
        self._prev_score_p2: int | None = None

    def reset(self, match_start_sec: float | None = None) -> None:
        """試合開始時に全帳簿をクリア。"""
        self._p1 = _SideState()
        self._p2 = _SideState()
        self._frame_idx = 0
        self._match_start_sec = match_start_sec
        self._prev_score_p1 = None
        self._prev_score_p2 = None
        # 後方互換: reset 後も get_snapshot が呼べるよう visible_ojama を初期化
        self._visible_ojama_p1: int | None = None
        self._visible_ojama_p2: int | None = None

    # ============================
    # 新 API: 状態遷移通知
    # ============================

    def on_state_transition(
        self,
        side: str,
        prev_state: object,  # BoardState
        curr_state: object,  # BoardState
        score: int | None,
        t_sec: float,
    ) -> None:
        """BoardState の遷移を通知して連鎖終了イベントを駆動する。

        Args:
            side: "p1" または "p2"。
            prev_state: 前フレームの BoardState。
            curr_state: 今フレームの BoardState。
            score: 今フレームの score (OCR 失敗時 None)。
            t_sec: 現在時刻 (秒)。
        """
        from src.board_state_machine import BoardState
        s = self._side(side)
        other = self._other(side)
        self._frame_idx += 1
        # -- 試合境界: MENU 遷移でリセット (エッジトリガ) --
        # MENU 継続中(prev も MENU)は毎フレーム呼ばれるため、
        # prev_state != MENU のときだけ(=MENU 入場の最初の 1 回だけ)リセットする。
        # これにより「1試合境界 = 1回リセット」を保証し、
        # MENU 継続中の多重発火(video_124 で 22 回 → 6 回相当)を防止する。
        if curr_state == BoardState.MENU:
            if prev_state != BoardState.MENU:
                # MENU 入場エッジ: 1 回だけリセット
                self._reset_side_boundary(s, side, score, t_sec)
                self._match_start_sec = None  # MENU後は次score受信まで待つ
            return
        # -- 試合境界: score 大幅減少 --
        if score is not None and self._prev_score(side) is not None:
            prev_score = self._prev_score(side)
            assert prev_score is not None
            if prev_score - score >= SCORE_RESET_THRESHOLD:
                self._reset_side_boundary(s, side, score, t_sec)
                self._set_prev_score(side, score)
                return
        self._set_prev_score(side, score)
        # -- 最後に読めた有効 score を更新(連鎖開始前の STABLE score をキャッシュ) --
        if score is not None:
            s.last_valid_score = score
            # 試合開始時刻が未設定の場合、最初の score 受信時刻を試合開始とみなす
            self._initialize_match_start(t_sec)
        # -- 連鎖間基準スコア (last_stable_score) を更新 --
        # 条件: STABLE 状態 かつ score が読める かつ 連鎖中でない かつ settle 待ちでない
        # CHAIN/GRAVITY_SETTLE 中・掛け算式 None 中は更新しない (途中値で汚さない)。
        _is_stable_state = (curr_state == BoardState.STABLE)
        if (_is_stable_state and score is not None
                and not s.chain_active and not s.score_settle_pending):
            s.last_stable_score = score
        # -- score settle 待ち処理 --
        # settle 待ち中は score 観測ごとに候補更新 or 連続不変カウントを進める。
        # スコアが K_SETTLE_FRAMES 連続不変 → 連鎖の真の終了 → finalize。
        # score=None は「掛け算式(連鎖継続の可能性)」とみなしカウントしない。
        if s.score_settle_pending:
            self._process_score_settle(s, other, score, side, t_sec, curr_state)
        # -- 連鎖終了待ちの遅延確定 (score None 継続時タイムアウト) --
        # score_settle_pending 中に settle 開始フレームからの経過を監視し
        # CHAIN_END_PENDING_TIMEOUT_FRAMES を超えたら timeout 破棄する。
        # これにより score None が long 続いた場合の無限待ちを防ぐ。
        if s.chain_end_pending:
            frames_waited = self._frame_idx - s.pending_since_frame
            if frames_waited >= CHAIN_END_PENDING_TIMEOUT_FRAMES:
                logger.warning(
                    "chain_end_pending timeout[%s]: %d frames waited, discarding",
                    side, frames_waited,
                )
                self._abort_settle(s, side, t_sec)
        # -- 連鎖開始: 非CHAIN → CHAIN --
        _chain_states = {BoardState.CHAIN}
        _is_chain_start = (
            prev_state not in _chain_states
            and curr_state == BoardState.CHAIN
        )
        if _is_chain_start:
            s.chain_active = True
            s.chain_end_triggered = False
            # --- デバウンス判定 ---
            # settle 待ち中 or finalize 完了直後 (CHAIN_COALESCE_WINDOW_SEC 以内) の
            # 非CHAIN→CHAIN は state 明滅 (同一連鎖の継続) とみなし
            # score_at_chain_start を上書きしない。
            # これにより「1 連鎖 = 1 finalize」を保証し、
            # state 明滅 (CHAIN→STABLE→CHAIN→GRAVITY_SETTLE→STABLE) での
            # 複数 finalize による過剰計上を根絶する。
            _in_coalesce_window = (
                s.chain_finalized_at_sec is not None
                and (t_sec - s.chain_finalized_at_sec) < CHAIN_COALESCE_WINDOW_SEC
            )
            # 連鎖途中フリッカー判定:
            # score_at_chain_start が既に設定済み(≠None)の場合は「連鎖継続中の
            # state 明滅(GRAVITY_SETTLE→CHAIN, OJAMA_FALL→CHAIN 等)」とみなし
            # score_at_chain_start を上書きしない。
            # 根拠: finalize/_reset_side_boundary/timeout の各パスで必ず
            # score_at_chain_start=None に戻すため、「None でないなら連鎖中」が保証される。
            _already_started = s.score_at_chain_start is not None
            if s.score_settle_pending or _already_started:
                logger.info(
                    "chain_start[%s]: skip (settle_pending=%s, "
                    "already_started=%s score_at_start=%s) t=%.2f",
                    side, s.score_settle_pending,
                    _already_started, s.score_at_chain_start, t_sec,
                )
            elif _in_coalesce_window:
                # coalesce window 内: state 明滅(flicker)か本物の2本目連鎖かを区別する。
                # 判定基準: last_valid_score が前回 finalize 時の score_after(last_finalized_score)
                # より増加しているか。
                #   - 明滅 = スコアが増えていない → score_at_chain_start を設定しない(破棄)。
                #   - 本物の2本目連鎖 = スコアが増加している → score_at_chain_start を設定する。
                # 注: last_stable_score ではなく last_finalized_score を使う理由:
                #   last_stable_score は finalize 後の STABLE フレームで更新されてしまい、
                #   2本目連鎖の直前 STABLE score と同じ値になって差分がゼロになる場合がある。
                #   last_finalized_score は finalize 時点でのみ更新するため安定した基準になる。
                _score_rose = (
                    s.last_finalized_score is not None
                    and s.last_valid_score is not None
                    and s.last_valid_score > s.last_finalized_score
                )
                if _score_rose:
                    # 本物の2本目連鎖: coalesce window 内でもスコア増加があるので計上する
                    s.score_at_chain_start = s.last_valid_score
                    logger.info(
                        "chain_start[%s]: in_coalesce but score_rose "
                        "(last_valid=%s > last_finalized=%s) → treating as new chain t=%.2f",
                        side, s.last_valid_score, s.last_finalized_score, t_sec,
                    )
                else:
                    # 明滅: スコアが増えていない → score_at_chain_start を設定しない
                    logger.info(
                        "chain_start[%s]: skip (in_coalesce=True, score_not_rose "
                        "last_valid=%s last_finalized=%s) t=%.2f",
                        side, s.last_valid_score, s.last_finalized_score, t_sec,
                    )
            else:
                # 新規連鎖: score_at_chain_start を確定
                # 連鎖開始直前 score スナップ:
                # 遷移フレームの score は掛け算式表示で None になりがちなため、
                # 最後に読めた有効 score (last_valid_score) を優先使用する。
                # last_valid_score も None の場合は score_at_chain_start=None のまま
                # → _finalize_chain_end で警告+破棄(過剰計上防止)。
                s.score_at_chain_start = s.last_valid_score
                logger.debug(
                    "chain_start[%s]: score_at_start=%s (last_valid=%s, frame_score=%s) t=%.2f",
                    side, s.score_at_chain_start, s.last_valid_score, score, t_sec,
                )
        # -- 連鎖終了候補 [新基準]: TSUMO_FALL / OJAMA_FALL への遷移 + score 上昇停止 --
        # 物理的根拠: 次ツモ出現またはお邪魔落下が来た = 「盤面が次の局面に移った」
        # = 連鎖の得点計算が完了している可能性が高い。
        # ただし score がまだ上昇中(=連鎖得点計算中)の場合は finalize しない。
        # score 上昇停止ゲートは既存 settle 機構(_process_score_settle)に委ねる:
        #   - _begin_score_settle を呼んで settle 待ちを開始する。
        #   - score が K_SETTLE_FRAMES 連続不変になった時点で finalize が発火する。
        # これにより「1P の大連鎖中に OJAMA_FALL が来ても score 上昇中は finalize しない」
        # (1P 誤分割防止) と「2P の小連鎖直後の TSUMO_FALL で正しく分離」が両立する。
        _phys_end_states = {BoardState.TSUMO_FALL, BoardState.OJAMA_FALL}
        _is_phys_chain_end = (
            s.chain_active
            and not s.score_settle_pending  # 既に settle 待ち中は重複開始しない
            and curr_state in _phys_end_states
        )
        if _is_phys_chain_end:
            s.chain_end_triggered = True
            logger.info(
                "chain_end_phys[%s]: %s→%s score=%s → settle待ち開始 t=%.2f",
                side, prev_state, curr_state, score, t_sec,
            )
            self._begin_score_settle(s, score, side, t_sec)
        # -- 連鎖終了候補 [補完]: (CHAIN or GRAVITY_SETTLE) → STABLE --
        # 新基準 (TSUMO_FALL/OJAMA_FALL) で取りこぼした場合の補完トリガー。
        # 例: OJAMA_FALL の後そのまま STABLE に遷移するシナリオ等。
        # 二重 finalize は coalesce window (_already_started 判定) で防止済み。
        # 即 finalize せず score_settle_pending に入り、スコアが settle するまで待機する。
        # settle 判定は _process_score_settle() が担う。
        _chain_or_settle = {BoardState.CHAIN, BoardState.GRAVITY_SETTLE}
        _is_chain_end = (
            s.chain_active
            and prev_state in _chain_or_settle
            and curr_state == BoardState.STABLE
        )
        if _is_chain_end:
            s.chain_end_triggered = True
            self._begin_score_settle(s, score, side, t_sec)

    def on_tsumo_settled(self, side: str, t_sec: float) -> None:
        """TSUMO_FALL → STABLE 遷移時に相手から受けた予告を 1 ターン分 drain する。

        実際のぷよぷよでは、tsumo が着地するたびに予告お邪魔が
        最大 OJAMA_MAX_DROP_PER_TURN 個まで盤面に降ってくる。

        Args:
            side: "p1" または "p2"。
            t_sec: 現在時刻 (秒)。
        """
        s = self._side(side)
        # (2026-08-24 並行帳簿) cap切り捨てなし側も同じ落下規則で drain する。
        # capped 側の早期 return (forecast<=0) より前に行う: cap 由来の乖離で
        # capped=0 / uncapped>0 になっても実際の落下は起きるため。drain 量は
        # 各帳簿の残量から独立に計算する (capped が 216 で頭打ちの間、実残量
        # 517 側は 30個/手 で減り続けるのが物理的に正しい)。
        if s.forecast_incoming_uncapped > 0:
            s.forecast_incoming_uncapped -= min(
                THEORY_DROP_PER_TURN, s.forecast_incoming_uncapped)
        if s.forecast_incoming <= 0:
            return
        drain = min(THEORY_DROP_PER_TURN, s.forecast_incoming)
        s.forecast_incoming -= drain
        s.total_dropped += drain
        logger.debug(
            "tsumo_settled[%s]: drain=%d -> forecast=%d t=%.2f",
            side, drain, s.forecast_incoming, t_sec,
        )

    def get_snapshot(self, t_sec: float) -> OjamaAccountSnapshot:
        """現在状態のスナップショットを返す(イベントなしでも呼べる)。"""
        return self._build_snapshot(t_sec)

    # ============================
    # 後方互換 API
    # ============================

    def update_from_score(
        self,
        score_p1: int | None,
        score_p2: int | None,
        t_sec: float,
        chain_p1: bool = False,
        chain_p2: bool = False,
        visible_ojama_p1: int | None = None,
        visible_ojama_p2: int | None = None,
        tsumo_settled_p1: bool = False,
        tsumo_settled_p2: bool = False,
    ) -> OjamaAccountSnapshot:
        """後方互換 API。旧 visualize_recognition.py 呼出継続のためルーティング。

        新実装では on_state_transition/on_tsumo_settled が本体。
        この API は state 遷移を on_state_transition で再現する。
        """
        from src.board_state_machine import BoardState
        # tsumo_settled = True → on_tsumo_settled 呼び出し
        if tsumo_settled_p1:
            self.on_tsumo_settled("p1", t_sec)
        if tsumo_settled_p2:
            self.on_tsumo_settled("p2", t_sec)
        # chain_p1/p2 は連鎖中フラグ: chain_active を直接更新
        # (旧 API では毎フレーム chain フラグを渡すだけで遷移は外側)
        self._update_chain_flag_legacy("p1", chain_p1, score_p1, t_sec)
        self._update_chain_flag_legacy("p2", chain_p2, score_p2, t_sec)
        # visible_ojama を保存 (後方互換 confidence 計算用)
        if visible_ojama_p1 is not None:
            self._visible_ojama_p1 = visible_ojama_p1
        if visible_ojama_p2 is not None:
            self._visible_ojama_p2 = visible_ojama_p2
        return self._build_snapshot(t_sec)

    def update_from_boards(
        self,
        board_p1: object,
        board_p2: object,
        score_p1: int | None = None,
        score_p2: int | None = None,
        is_chain_p1: bool = False,
        is_chain_p2: bool = False,
    ) -> None:
        """後方互換 API(旧盤面増分落下・全消し検出)。新実装ではno-op。

        新実装では on_tsumo_settled / on_state_transition が落下・生成を担う。
        旧 API は削除せず no-op として維持(呼出元が壊れないよう)。
        """
        # 後方互換: visible_ojama を保存して confidence 計算に使う
        try:
            self._visible_ojama_p1 = self._count_visible_ojama(board_p1)
            self._visible_ojama_p2 = self._count_visible_ojama(board_p2)
        except Exception:
            pass

    def update_accounting_with_chain(
        self,
        t_sec: float,
        chain_p1: bool,
        chain_p2: bool,
    ) -> OjamaAccountSnapshot:
        """後方互換 API(旧相殺処理)。新実装では単にスナップショットを返す。"""
        return self._build_snapshot(t_sec)

    # ============================
    # 内部: 連鎖終了 一括換算 + 相殺
    # ============================

    def _finalize_chain_end(
        self,
        s: _SideState,
        other: _SideState,
        score_after: int,
        side: str,
        t_sec: float,
    ) -> None:
        """連鎖合計得点を計算し、お邪魔生成 + 相殺を一括で処理する。

        相殺の正しい向き(設計書3):
            1. 生成 G で自分の forecast_incoming を相殺(自分への予告を消す)。
            2. 余剰 surplus を相手の forecast_incoming に加算。
        """
        s.chain_end_pending = False
        # --- 連鎖合計スコア計算 ---
        # 新方針: last_stable_score (連鎖間の落ち着いた STABLE score) を基準に優先使用する。
        # score_at_chain_start は coalesce/dedup の有効性チェックにのみ使用する:
        #   - score_at_chain_start=None = coalesce skip で連鎖開始が認識されていない
        #     → 偽連鎖(state 明滅)として破棄する(過剰計上防止)
        #   - score_at_chain_start=設定済み = 正規の連鎖開始あり
        #     → last_stable_score を優先して差分計算する
        # これにより、score_at_chain_start の取り違えによる計算誤差を排除しつつ、
        # coalesce window 保護も維持する。
        if s.score_at_chain_start is None:
            # 連鎖開始が認識されていない = coalesce skip で保護された偽連鎖
            # → 過剰計上防止のため破棄する
            logger.warning("chain_end[%s]: score_at_chain_start=None, discarding t=%.2f", side, t_sec)
            s.chain_active = False
            s.chain_finalized_at_sec = t_sec  # 破棄でも coalesce window を開く
            return
        # score_at_chain_start が設定されている = 正規の連鎖。
        # last_stable_score を優先して差分計算する(フォールバック: score_at_chain_start)。
        if s.last_stable_score is not None:
            score_start = s.last_stable_score
            logger.debug(
                "chain_end[%s]: using last_stable_score=%d "
                "(score_at_chain_start=%d) t=%.2f",
                side, score_start, s.score_at_chain_start, t_sec,
            )
        else:
            # last_stable_score が未設定(試合最初の連鎖など): score_at_chain_start でフォールバック
            score_start = s.score_at_chain_start
            logger.info(
                "chain_end[%s]: last_stable_score=None, fallback to score_at_chain_start=%d t=%.2f",
                side, score_start, t_sec,
            )
        chain_total = score_after - score_start
        # --- sanity check ---
        if chain_total <= 0:
            # score が増えていない = 試合境界 or OCR 異常
            logger.info(
                "chain_end[%s]: chain_total=%d <= 0 (trial reset or OCR error?), "
                "score_after=%d start=%d t=%.2f",
                side, chain_total, score_after, score_start, t_sec,
            )
            s.chain_active = False
            s.score_at_chain_start = None
            s.chain_finalized_at_sec = t_sec
            return
        if chain_total < CHAIN_TOTAL_MIN_SCORE:
            # 極小 chain_total = score OCR 端数誤読の疑い
            # leftover への誤累積を防ぐため破棄する。
            # 正当な小連鎖(1-2連鎖)は chain_total >= 40 になるため弾かれない。
            logger.info(
                "chain_end[%s]: chain_total=%d < min=%d (OCR端数誤読の疑い), "
                "discarding t=%.2f",
                side, chain_total, CHAIN_TOTAL_MIN_SCORE, t_sec,
            )
            s.chain_active = False
            s.score_at_chain_start = None
            s.chain_finalized_at_sec = t_sec
            return
        if chain_total > CHAIN_TOTAL_SANITY_MAX:
            logger.warning(
                "chain_end[%s]: chain_total=%d > sanity_max=%d, discarding t=%.2f",
                side, chain_total, CHAIN_TOTAL_SANITY_MAX, t_sec,
            )
            s.chain_active = False
            s.score_at_chain_start = None
            s.chain_finalized_at_sec = t_sec
            return
        # --- お邪魔生成量計算 ---
        elapsed = self._elapsed(t_sec)
        leftover_before = s.leftover
        forecast_before = s.forecast_incoming
        other_forecast_before = other.forecast_incoming
        result = score_to_ojama(
            score=chain_total,
            prev_leftover=leftover_before,
            elapsed_sec=elapsed,
            rate_base=self._rate_base,
        )
        s.leftover = result.leftover_score
        gen = result.ojama_count
        s.total_generated += gen
        s.last_chain_total_score = chain_total
        # --- 詳細デバッグログ (過剰計上診断用) ---
        logger.info(
            "finalize[%s]: score_start=%d score_after=%d chain_total=%d "
            "leftover_before=%d -> gen=%d leftover_after=%d "
            "self.forecast_before=%d other.forecast_before=%d t=%.2f",
            side, score_start, score_after, chain_total,
            leftover_before, gen, s.leftover,
            forecast_before, other_forecast_before, t_sec,
        )
        # --- 相殺(正しい向き、cancel_own_pending_then_send_surplus に集約) ---
        canceled = min(gen, s.forecast_incoming)  # total_offset 集計用に維持
        s.forecast_incoming, other.forecast_incoming = cancel_own_pending_then_send_surplus(
            gen, s.forecast_incoming, other.forecast_incoming,
        )
        # (2026-08-24 並行帳簿) cap切り捨てなし側にも同一 gen で同じ相殺を適用。
        # capped 側が 216 に丸められた後だと余剰 (gen - 216) が架空の攻撃として
        # 相手へ送られるが、こちらは実額 (例: 517) で相殺するため余剰が正しく
        # 小さくなる (kill_override の入力専用、根因② project_pm100_display_
        # flip_2026-08-24)。サニティ上限のみ適用 (下の cap 節参照)。
        (s.forecast_incoming_uncapped,
         other.forecast_incoming_uncapped) = cancel_own_pending_then_send_surplus(
            gen, s.forecast_incoming_uncapped, other.forecast_incoming_uncapped,
        )
        s.forecast_incoming_uncapped = min(
            s.forecast_incoming_uncapped, PENDING_UNCAPPED_SANITY_MAX)
        other.forecast_incoming_uncapped = min(
            other.forecast_incoming_uncapped, PENDING_UNCAPPED_SANITY_MAX)
        s.total_offset += canceled
        surplus = gen - canceled
        logger.info(
            "offset[%s]: gen=%d canceled=%d surplus=%d "
            "self.forecast=%d other.forecast=%d t=%.2f",
            side, gen, canceled, surplus,
            s.forecast_incoming, other.forecast_incoming, t_sec,
        )
        # --- pending cap ---
        if s.forecast_incoming > PENDING_ABS_CAP:
            logger.warning(
                "forecast cap[%s]: %d > abs_cap=%d, clamping",
                side, s.forecast_incoming, PENDING_ABS_CAP,
            )
            s.forecast_incoming = PENDING_ABS_CAP
        if other.forecast_incoming > PENDING_ABS_CAP:
            other_label = "p2" if side == "p1" else "p1"
            logger.warning(
                "forecast cap[%s]: %d > abs_cap=%d, clamping",
                other_label, other.forecast_incoming, PENDING_ABS_CAP,
            )
            other.forecast_incoming = PENDING_ABS_CAP
        # --- 後処理 ---
        s.chain_active = False
        s.score_at_chain_start = None
        # finalize 後は score_after を新しい last_stable_score として設定する。
        # これにより次の連鎖の chain_total が正確に「この連鎖終了直後から」計算される。
        s.last_stable_score = score_after
        # last_finalized_score: この finalize の score_after を記録する。
        # coalesce window 内の本物の2本目連鎖判定 (last_valid_score > last_finalized_score)
        # に使用する。last_stable_score は finalize 後も STABLE フレームで更新されるため
        # 代わりに last_finalized_score を使う(更新タイミングが finalize のみ)。
        s.last_finalized_score = score_after
        # coalesce window を開く: この時刻から CHAIN_COALESCE_WINDOW_SEC 以内の
        # 再 chain_start は同一連鎖の state 明滅とみなし score_at_chain_start を守る。
        s.chain_finalized_at_sec = t_sec

    # ============================
    # 内部: score settle 待ち
    # ============================

    def _begin_score_settle(
        self,
        s: _SideState,
        score: int | None,
        side: str,
        t_sec: float,
    ) -> None:
        """連鎖終了候補を検知したとき settle 待ちを開始する。

        score が None の場合は候補を None のまま保持し、
        既存 chain_end_pending タイムアウトを起動する。

        すでに settle 待ち中(score_settle_pending=True)の場合は、
        state 明滅 (STABLE→CHAIN→STABLE) による再 begin を防ぐため
        候補スコアの上限更新のみ行う(カウンタリセットせず継続)。
        これにより連鎖途中の明滅で settle が中断されない。
        """
        if s.score_settle_pending:
            # すでに settle 待ち中: 候補を最大値に更新するだけ(カウンタ維持)
            if score is not None:
                if s.score_settle_candidate is None or score > s.score_settle_candidate:
                    logger.debug(
                        "score_settle_begin[%s]: already pending, update candidate %s→%d t=%.2f",
                        side, s.score_settle_candidate, score, t_sec,
                    )
                    s.score_settle_candidate = score
                    s.score_settle_consec = 0  # 上昇があったのでリセット
            return
        s.score_settle_pending = True
        s.score_settle_candidate = score  # None の場合もそのまま保持
        s.score_settle_consec = 0
        s.score_settle_since_frame = self._frame_idx
        # score None の場合は既存タイムアウト機構(chain_end_pending)も起動する
        if score is None:
            s.chain_end_pending = True
            s.pending_since_frame = self._frame_idx
            logger.debug(
                "score_settle_begin[%s]: score=None, pending started t=%.2f", side, t_sec,
            )
        else:
            s.chain_end_pending = False
            logger.debug(
                "score_settle_begin[%s]: candidate=%d t=%.2f", side, score, t_sec,
            )

    def _process_score_settle(
        self,
        s: _SideState,
        other: _SideState,
        score: int | None,
        side: str,
        t_sec: float,
        curr_state: object = None,
    ) -> None:
        """settle 待ち中のフレームごと処理。

        score=None は「まだ掛け算式表示中 = 連鎖継続の可能性」とみなしカウントしない。
        curr_state が CHAIN/GRAVITY_SETTLE の場合は「連鎖継続中」とみなしカウントをリセット。
        score が前フレームから上昇した場合は candidate を更新してカウントをリセット。
        score が K_SETTLE_FRAMES フレーム連続不変なら finalize を発火する。
        """
        if score is None:
            # None は連鎖継続の可能性としてスキップ(カウント進めない)
            return
        # CHAIN / GRAVITY_SETTLE 状態は「まだ連鎖得点計算中」とみなす。
        # score が一時停止していても state が CHAIN 系なら settle させない。
        # 物理的根拠: 連鎖中の score 一時停止はステップ間ポーズであり、
        # 連鎖が完全に終了した「真の停止」ではない。
        # 例: 1P 大連鎖で score=825 → 20 フレーム不変(連鎖途中ポーズ)→ → 1465 と続く。
        # この場合 CHAIN state が継続しているため settle させない。
        if curr_state is not None:
            from src.board_state_machine import BoardState
            _still_chaining = curr_state in {BoardState.CHAIN, BoardState.GRAVITY_SETTLE}
            if _still_chaining:
                # 連鎖継続中: score 不変カウントをリセット(但し candidate は更新しない)
                if s.score_settle_consec > 0:
                    logger.debug(
                        "score_settle[%s]: state=%s (chaining), reset consec=%d→0 t=%.2f",
                        side, curr_state, s.score_settle_consec, t_sec,
                    )
                    s.score_settle_consec = 0
                return
        # score が読めた: chain_end_pending は解除(None 待ち解消)
        s.chain_end_pending = False
        if s.score_settle_candidate is None:
            # 初回 score 受信
            s.score_settle_candidate = score
            s.score_settle_consec = 1
            logger.debug(
                "score_settle[%s]: first score=%d t=%.2f", side, score, t_sec,
            )
            return
        if score > s.score_settle_candidate:
            # スコア上昇 = 連鎖継続中 → candidate 更新・カウントリセット
            logger.debug(
                "score_settle[%s]: score rose %d→%d, reset counter t=%.2f",
                side, s.score_settle_candidate, score, t_sec,
            )
            s.score_settle_candidate = score
            s.score_settle_consec = 1
        elif score == s.score_settle_candidate:
            # スコア不変 → カウントアップ
            s.score_settle_consec += 1
            logger.debug(
                "score_settle[%s]: consec=%d/%d score=%d t=%.2f",
                side, s.score_settle_consec, K_SETTLE_FRAMES, score, t_sec,
            )
            if s.score_settle_consec >= K_SETTLE_FRAMES:
                # settle 確定 → finalize 発火
                logger.info(
                    "score_settle[%s]: settled at score=%d (consec=%d) t=%.2f",
                    side, score, s.score_settle_consec, t_sec,
                )
                s.score_settle_pending = False
                self._finalize_chain_end(s, other, score, side, t_sec)
        else:
            # score が減少 = 試合境界 or OCR 誤読 → settle 破棄
            # (試合境界の場合は on_state_transition の score 大幅減少検知が先に走る想定。
            #  ここでは score_settle_pending=False にしてお邪魔を出さないよう破棄する)
            logger.info(
                "score_settle[%s]: score dropped %d→%d, aborting t=%.2f",
                side, s.score_settle_candidate, score, t_sec,
            )
            self._abort_settle(s, side, t_sec)

    def _abort_settle(self, s: _SideState, side: str, t_sec: float) -> None:
        """settle 待ちを破棄して連鎖状態をリセットする(タイムアウト・境界・スコア減少時)。"""
        logger.info(
            "score_settle_abort[%s]: settle破棄 candidate=%s consec=%d t=%.2f",
            side, s.score_settle_candidate, s.score_settle_consec, t_sec,
        )
        s.score_settle_pending = False
        s.score_settle_candidate = None
        s.score_settle_consec = 0
        s.chain_end_pending = False
        s.chain_active = False
        s.score_at_chain_start = None
        s.chain_finalized_at_sec = t_sec  # coalesce window を開いて二重 finalize を防ぐ

    # ============================
    # 後方互換 API の内部処理
    # ============================

    def _update_chain_flag_legacy(
        self,
        side: str,
        chain: bool,
        score: int | None,
        t_sec: float,
    ) -> None:
        """旧 update_from_score から chain フラグを処理する(後方互換パス)。

        chain False→True: 連鎖終了 (旧 API では False→True が「完了エッジ」)。
        この API では score_at_chain_start が設定できないため、
        chain_total は score の前回値から推定する。
        """
        from src.board_state_machine import BoardState
        s = self._side(side)
        other = self._other(side)
        prev_score = self._prev_score(side)
        # 試合境界: score 大幅減少
        if score is not None and prev_score is not None:
            if prev_score - score >= SCORE_RESET_THRESHOLD:
                self._reset_side_boundary(s, side, score, t_sec)
                self._set_prev_score(side, score)
                return
        self._set_prev_score(side, score)
        # chain False→True: 連鎖完了エッジ扱い (旧互換)
        was_chain = s.chain_active
        if chain and not was_chain:
            # 連鎖完了エッジ: chain_total = score - score_at_chain_start
            # 旧 API では score_at_start が不明なのでここで差分から推定
            # (スコアが増加しているなら発火とみなす)
            if score is not None and prev_score is not None:
                delta = score - prev_score
                if delta >= CHAIN_FIRE_MIN_SCORE:
                    s.score_at_chain_start = prev_score
                    s.chain_active = True
                    s.chain_end_triggered = True
                    self._finalize_chain_end(s, other, score, side, t_sec)
            s.chain_active = True  # chain 中フラグ ON
        elif not chain and was_chain:
            s.chain_active = False  # chain 終了
        elif chain:
            s.chain_active = True

    # ============================
    # 内部: 試合境界リセット
    # ============================

    def _reset_side_boundary(
        self, s: _SideState, label: str, new_score: int | None, t_sec: float,
    ) -> None:
        """試合境界(score大幅減少 or MENU)でそのサイドの帳簿をリセット。

        試合境界時に _match_start_sec を t_sec に更新し、次試合のマージンタイムを
        試合相対経過秒で計算できるようにする。
        同一試合境界で p1/p2 両側から呼ばれた場合、ほぼ同時刻で二重設定になるが無害。

        MENU 遷移の場合は t_sec を None にリセットし、次の score 観測時に再設定する。
        (MENU 後の実際のゲーム開始タイミングは MENU 遷移時刻より後のため。)
        """
        logger.info(
            "match_boundary[%s]: forecast=%d leftover=%d -> reset t=%.2f",
            label, s.forecast_incoming, s.leftover, t_sec,
        )
        s.forecast_incoming = 0
        s.forecast_incoming_uncapped = 0  # 並行帳簿も試合境界で必ずゼロ (2026-08-24)
        s.leftover = 0
        s.chain_active = False
        s.score_at_chain_start = None
        s.chain_end_pending = False
        # 試合境界では chain_end_triggered もクリア。
        # クリアしないと前試合の triggered=True が次試合に持ち越されて
        # overlay の誤表示 (最大30秒以上) を引き起こす。
        s.chain_end_triggered = False
        # 試合境界では last_valid_score もクリア(前試合の値が次試合冒頭に使われないよう)
        s.last_valid_score = None
        # 試合境界では last_stable_score もクリア(前試合のスコアが次試合の基準に使われないよう)
        s.last_stable_score = None
        # coalesce window もクリア(前試合の window が次試合に引き継がれないよう)
        s.chain_finalized_at_sec = None
        # last_finalized_score もクリア(前試合の finalize score が次試合に引き継がれないよう)
        s.last_finalized_score = None
        # settle 待ちもクリア(試合境界をまたいだ settle は破棄)
        s.score_settle_pending = False
        s.score_settle_candidate = None
        s.score_settle_consec = 0
        s.score_settle_since_frame = 0
        # --- マージンタイム基準を試合相対に更新 ---
        # score 大幅減少 = 次試合開始とみなし、この時刻を新しい試合開始とする。
        # MENU 遷移 (label で "menu" を区別するのではなく呼出元が判断して渡す) の場合は
        # None をセットし、最初の score 受信時に再設定する。
        self._match_start_sec = t_sec

    # ============================
    # 内部: ヘルパー
    # ============================

    def _side(self, side: str) -> _SideState:
        return self._p1 if side == "p1" else self._p2

    def _other(self, side: str) -> _SideState:
        return self._p2 if side == "p1" else self._p1

    def _prev_score(self, side: str) -> int | None:
        return self._prev_score_p1 if side == "p1" else self._prev_score_p2

    def _set_prev_score(self, side: str, score: int | None) -> None:
        if side == "p1":
            self._prev_score_p1 = score
        else:
            self._prev_score_p2 = score

    def _initialize_match_start(self, t_sec: float) -> None:
        """試合開始時刻が未設定の場合のみ t_sec を試合開始とみなして設定する。

        - reset() 直後で _match_start_sec=None の場合: 最初の score 受信時刻を設定。
        - MENU 遷移後で _match_start_sec=None の場合: MENU 後の最初の score 受信時刻を設定。
        - _match_start_sec が既に設定済みの場合: 何もしない(上書き禁止)。
        """
        if self._match_start_sec is None:
            self._match_start_sec = float(t_sec)
            logger.info(
                "match_start initialized: t=%.2f (first score observed)", t_sec,
            )

    def _elapsed(self, t_sec: float) -> float:
        """試合開始からの経過秒を返す(マージンタイム計算用)。

        _match_start_sec が None の場合は「試合開始前 or 初期化前」とみなし
        0.0 を返す(マージンタイム非適用 = 安全側)。
        クリップ先頭からの経過秒をそのまま返す旧実装はマージンタイム過剰適用の原因
        であったため廃止(2026-06-10 修正)。
        """
        if self._match_start_sec is None:
            return 0.0
        return max(0.0, float(t_sec) - self._match_start_sec)

    @staticmethod
    def _count_visible_ojama(board: object) -> int:
        """後方互換: 盤面の可視領域のおじゃまぷよ数をカウント。"""
        try:
            from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, HIDDEN_ROWS
            count = 0
            for row in range(HIDDEN_ROWS, BOARD_ROWS):
                for col in range(BOARD_COLS):
                    if int(board.get(row, col)) == COLOR_OJAMA:  # type: ignore[union-attr]
                        count += 1
            return count
        except Exception:
            return 0

    # ============================
    # 内部: スナップショット構築
    # ============================

    def _build_snapshot(self, t_sec: float) -> OjamaAccountSnapshot:
        """現在の内部状態から OjamaAccountSnapshot を構築する。"""
        f1 = self._p1.forecast_incoming
        f2 = self._p2.forecast_incoming
        net_balance = f2 - f1
        p1_capped = min(f1, ON_FIELD_CAP)
        p2_capped = min(f2, ON_FIELD_CAP)
        p1_offboard = max(0, f1 - ON_FIELD_CAP)
        p2_offboard = max(0, f2 - ON_FIELD_CAP)
        return OjamaAccountSnapshot(
            t_sec=t_sec,
            # 既存フィールド
            pending_p1=f1,
            pending_p2=f2,
            total_generated_by_p1=self._p1.total_generated,
            total_generated_by_p2=self._p2.total_generated,
            total_offset_by_p1=self._p1.total_offset,
            total_offset_by_p2=self._p2.total_offset,
            total_dropped_to_p1=self._p1.total_dropped,
            total_dropped_to_p2=self._p2.total_dropped,
            net_ojama_balance=net_balance,
            overflow_risk_p1=f1 >= self._overflow_threshold,
            overflow_risk_p2=f2 >= self._overflow_threshold,
            confidence=float(CONFIDENCE_SCORE_OCR_ONLY),
            leftover_p1=self._p1.leftover,
            leftover_p2=self._p2.leftover,
            all_clear_pending_p1=False,  # 廃止=常 False
            all_clear_pending_p2=False,  # 廃止=常 False
            # 修正C フィールド
            pending_p1_capped=p1_capped,
            pending_p2_capped=p2_capped,
            net_balance_capped=p2_capped - p1_capped,
            offboard_p1=p1_offboard,
            offboard_p2=p2_offboard,
            # 新フィールド (末尾)
            forecast_p1=f1,
            forecast_p2=f2,
            chain_total_score_p1=self._p1.last_chain_total_score,
            chain_total_score_p2=self._p2.last_chain_total_score,
            chain_end_triggered_p1=self._p1.chain_end_triggered,
            chain_end_triggered_p2=self._p2.chain_end_triggered,
            score_at_chain_start_p1=self._p1.score_at_chain_start,
            score_at_chain_start_p2=self._p2.score_at_chain_start,
            # cap切り捨てなし並行帳簿 (2026-08-24、kill判定入力専用)
            pending_p1_uncapped=self._p1.forecast_incoming_uncapped,
            pending_p2_uncapped=self._p2.forecast_incoming_uncapped,
        )


__all__ = [
    "CHAIN_FIRE_MIN_SCORE",
    "CHAIN_TOTAL_MIN_SCORE",
    "CONFIDENCE_SCORE_OCR_ONLY",
    "CONFIDENCE_VISUAL_AGREE",
    "CONFIDENCE_VISUAL_MISMATCH_PENALTY",
    "VISIBLE_OJAMA_MISMATCH_THRESHOLD",
    "DROP_SANITY_CLAMP",
    "SCORE_RESET_THRESHOLD",
    "THEORY_DROP_PER_TURN",
    "ON_FIELD_CAP",
    "PENDING_HARD_CAP",
    "PENDING_ABS_CAP",
    "PENDING_UNCAPPED_SANITY_MAX",
    "CHAIN_COALESCE_WINDOW_SEC",
    "K_SETTLE_FRAMES",
    "OjamaAccountSnapshot",
    "OjamaAccountingTracker",
]
