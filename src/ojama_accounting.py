"""お邪魔ぷよ会計モジュール (Ojama Accounting)。

5 帳簿分離で素朴実装の「相殺ゼロ」失敗を構造的に排除する:

  生成 → 相殺 → confirmed_pending → 落下 → 画面内増加 / 画面外あふれ

設計方針:
    - OjamaAccountSnapshot: 静的スナップショット (frozen dataclass, stateless)
    - OjamaAccountingTracker: 状態保持 wrapper (CLAUDE.md「state は外部 wrapper」準拠)
    - あふれ閾値は学習任せ (設計思想2): overflow_risk は補助 bool、raw pending を必ず生値で保持
    - score OCR 差分を唯一の生成源とする (視覚予告クロスチェックは後段)
    - 相殺ロジック: 自分が連鎖完了した時、自分に向かう相手 pending をまず相殺 → 余剰を相手 pending に加算
    - 落下方式: 理論落下 (stable 復帰ターンごとに OJAMA_MAX_DROP_PER_TURN drain) が主系統。
      盤面増分観測は補助 (実データで落下後すぐ連鎖消去でSTABLE時増分≒0のため非依存)
    - 全消し自動検出: confirmed_board が全 EMPTY (色ぷよ 0) になったら all_clear_pending を自動セット
    - 試合境界: score 減少 (>= SCORE_RESET_THRESHOLD) で検知し pending/leftover/all_clear を reset

Step1.5 解消点:
    ① total_dropped: update_from_boards() でおじゃま増分を落下量として計上
    ② 全消し自動検出: update_from_boards() で all_clear_detector.is_all_clear() 呼び出し
    ③ 相殺1フレーム遅延: state_pipeline.extract() 末尾で chain 確定後に
       update_accounting_with_chain() を呼ぶ構造分離
    ④ 落下クロスチェック: update_from_boards() の visible_ojama を update_from_score() に渡し
       VISIBLE_OJAMA_MISMATCH_THRESHOLD 機構で乖離を confidence に反映

2026-06-09 アーキ再設計 (欠陥A/B対処):
    修正A: 試合境界 reset = score 減少 (>= SCORE_RESET_THRESHOLD) で検知
           (増分閾値でのreset誤判定: 正当大連鎖 +14850 等があるため増分側は使わない)
    修正B: 理論落下 = stable 復帰ターンごとに min(30, pending) を drain (観測非依存)
    修正C: pending hard cap = PENDING_HARD_CAP=72 (盤面セル数)

流用資産:
    - src/scoring.py: score_to_ojama(), ALL_CLEAR_BONUS, OJAMA_MAX_DROP_PER_TURN
    - src/all_clear_detector.py: is_all_clear() でフィールド全 EMPTY 判定
    - src/ojama_score_inferrer.py: leftover / 全消し持越し管理パターンを参考に内包
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.scoring import (
    ALL_CLEAR_BONUS,
    OJAMA_MAX_DROP_PER_TURN,
    OJAMA_RATE_STANDARD,
    score_to_ojama,
)

if TYPE_CHECKING:
    from src.board import Board

logger = logging.getLogger(__name__)

# ============================
# 定数
# ============================

# score 差分がこの値以上で連鎖発火イベントとみなす (OCR ノイズ吸収)
CHAIN_FIRE_MIN_SCORE: int = 40

# visible_ojama との差がこの個数を超えたら confidence を下げる
VISIBLE_OJAMA_MISMATCH_THRESHOLD: int = 6

# 確度が十分な時の confidence 値
CONFIDENCE_SCORE_OCR_ONLY: float = 0.85
# visible_ojama 一致時の confidence
CONFIDENCE_VISUAL_AGREE: float = 0.95
# visible_ojama 不一致時の confidence ペナルティ (引き算)
CONFIDENCE_VISUAL_MISMATCH_PENALTY: float = 0.30

# 1ターンあたりの落下量上限 (sanity bound, src/scoring.py の OJAMA_MAX_DROP_PER_TURN と同値)
# confirmed_board での増分がこの値を超えたら異常として clamp する
DROP_SANITY_CLAMP: int = OJAMA_MAX_DROP_PER_TURN

# 修正A: 試合境界検知用 score 減少閾値 (OCR ノイズによる小ゆれと区別)
# 正当な score 減少はほぼ存在しない (次試合リセット: 23194→0 等)
SCORE_RESET_THRESHOLD: int = 500

# 修正B: 理論落下 = stable 復帰 1 ターンで drain する ojama 個数上限
# OJAMA_MAX_DROP_PER_TURN(=30) と同値
THEORY_DROP_PER_TURN: int = OJAMA_MAX_DROP_PER_TURN

# 修正C: pending hard cap (盤面全セル数 6×12=72)
# この値を超える pending は物理的にありえず、score OCR 誤積みを防ぐ
PENDING_HARD_CAP: int = 72


# ============================
# スナップショット (stateless)
# ============================

@dataclass(frozen=True)
class OjamaAccountSnapshot:
    """1 時刻の両者お邪魔ぷよ会計スナップショット。

    Attributes:
        t_sec: 観測時刻 (秒)。
        pending_p1: 1P に向かう確定 pending ojama 数 (相殺後の純 pending)。
        pending_p2: 2P に向かう確定 pending ojama 数。
        total_generated_by_p1: 1P がこれまでに生成した累積 ojama 数 (相殺前)。
        total_generated_by_p2: 2P がこれまでに生成した累積 ojama 数 (相殺前)。
        total_offset_by_p1: 1P がこれまでに相殺した累積 ojama 数。
        total_offset_by_p2: 2P がこれまでに相殺した累積 ojama 数。
        total_dropped_to_p1: 1P フィールドに落下した累積 ojama 数 (理論値)。
        total_dropped_to_p2: 2P フィールドに落下した累積 ojama 数 (理論値)。
        net_ojama_balance: pending_p2 - pending_p1。正なら 1P 有利 (自分への ojama が少ない)。
        overflow_risk_p1: 1P に向かう pending が大きく落下が迫っている可能性 (補助 bool)。
        overflow_risk_p2: 2P に向かう pending が大きく落下が迫っている可能性。
        confidence: 0.0-1.0。score OCR 有効なら高。visual 乖離で低下。
        leftover_p1: 1P 側の score 換算端数繰越 (次連鎖への繰越点)。
        leftover_p2: 2P 側の score 換算端数繰越。
        all_clear_pending_p1: 1P 側に全消し持越しボーナスが残っているか。
        all_clear_pending_p2: 2P 側に全消し持越しボーナスが残っているか。
        pending_p1_capped: min(pending_p1, PENDING_HARD_CAP)。指標用有界値。
        pending_p2_capped: min(pending_p2, PENDING_HARD_CAP)。指標用有界値。
        net_balance_capped: pending_p2_capped - pending_p1_capped。有界 -72..+72。
            (net_balance_capped + 72) / 144 で 0-1 正規化可能。
    """
    t_sec: float
    pending_p1: int
    pending_p2: int
    total_generated_by_p1: int
    total_generated_by_p2: int
    total_offset_by_p1: int
    total_offset_by_p2: int
    total_dropped_to_p1: int
    total_dropped_to_p2: int
    net_ojama_balance: int
    overflow_risk_p1: bool
    overflow_risk_p2: bool
    confidence: float
    leftover_p1: int
    leftover_p2: int
    all_clear_pending_p1: bool
    all_clear_pending_p2: bool
    # --- 修正C 追加フィールド (backwards compat: 末尾追加) ---
    pending_p1_capped: int = 0
    pending_p2_capped: int = 0
    net_balance_capped: int = 0


# ============================
# 内部状態 (1 サイド分)
# ============================

@dataclass
class _SideState:
    """片側 (1P or 2P) のお邪魔会計内部状態。"""
    leftover: int = 0                  # score 換算端数繰越
    all_clear_pending: bool = False    # 全消し持越しフラグ
    total_generated: int = 0           # 累積生成量
    total_offset: int = 0             # 累積相殺量
    total_dropped: int = 0            # 累積落下量
    pending: int = 0                  # 現在の確定 pending (相殺後)
    prev_visible_ojama: int = 0       # 前フレームの盤面内おじゃま数 (落下増分検出用)


# ============================
# Tracker (stateful wrapper)
# ============================

class OjamaAccountingTracker:
    """試合 1 本分のお邪魔ぷよ会計を管理する状態保持 wrapper。

    CLAUDE.md 規約「state は外部 wrapper」準拠。
    OjamaAccountSnapshot は stateless なスナップショットとして返す。

    使い方:
        tracker = OjamaAccountingTracker()
        tracker.reset()
        snapshot = tracker.update_from_score(
            score_p1=500, score_p2=300, t_sec=15.0,
            chain_p1=True, chain_p2=False,
        )
        print(snapshot.pending_p1, snapshot.pending_p2)
    """

    def __init__(
        self,
        ojama_rate_base: int = OJAMA_RATE_STANDARD,
        overflow_threshold: int = OJAMA_MAX_DROP_PER_TURN,
    ) -> None:
        """
        Args:
            ojama_rate_base: おじゃまレート基本値 (通常 70)。
            overflow_threshold: この値以上の pending を overflow_risk=True とする補助閾値。
                学習任せの raw pending も必ず返すので、指標層はこの bool を使わなくてもよい。
        """
        self._rate_base = int(ojama_rate_base)
        self._overflow_threshold = int(overflow_threshold)
        # 内部状態
        self._p1 = _SideState()
        self._p2 = _SideState()
        # スコア前回値 (差分検出用・試合境界検知用)
        self._prev_score_p1: int | None = None
        self._prev_score_p2: int | None = None
        self._match_start_sec: float | None = None
        # 連鎖完了フラグの前フレーム値 (立ち上がりエッジ検出用)
        self._prev_chain_p1: bool = False
        self._prev_chain_p2: bool = False

    def reset(self, match_start_sec: float | None = None) -> None:
        """試合開始時に全帳簿をクリア。"""
        self._p1 = _SideState()
        self._p2 = _SideState()
        self._prev_score_p1 = None
        self._prev_score_p2 = None
        self._match_start_sec = match_start_sec
        self._prev_chain_p1 = False
        self._prev_chain_p2 = False
        # visible_ojama の最新値 (クロスチェック用、update_from_boards() が更新)
        self._visible_ojama_p1: int | None = None
        self._visible_ojama_p2: int | None = None

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
        """毎フレーム呼べる主 API。

        処理順序:
            1. score 差分 → 試合境界判定 (修正A) + 生成量計算
            2. 理論落下 drain (修正B): stable 復帰ターンで pending を drain
            3. 連鎖完了エッジ (False→True 立ち上がり) で相殺処理
            4. pending hard cap 適用 (修正C)
            5. visible_ojama が渡された場合は pending との乖離を confidence に反映
            6. スナップショット構築・返却

        Args:
            score_p1: 1P の現在スコア (OCR 失敗時 None)。
            score_p2: 2P の現在スコア (OCR 失敗時 None)。
            t_sec: 現在時刻 (秒)。
            chain_p1: 1P が連鎖中 (True) または完了 (False→True の立ち上がりが相殺トリガー)。
            chain_p2: 2P が連鎖中。
            visible_ojama_p1: 盤面から観測した 1P の ojama cell 数 (クロスチェック用)。
            visible_ojama_p2: 同 2P。
            tsumo_settled_p1: 1P が今フレームで STABLE に復帰したか (stable 復帰エッジ)。
                True の場合、理論落下 drain を実行する (修正B)。
            tsumo_settled_p2: 2P 側同上。

        Returns:
            OjamaAccountSnapshot: 現時点のお邪魔ぷよ会計スナップショット。
        """
        elapsed = self._elapsed(t_sec)

        # step1: score 差分から境界検知 + 生成量を計算 (修正A)
        self._process_score_delta(score_p1, score_p2, elapsed)

        # step2: 理論落下 drain (修正B)
        if tsumo_settled_p1:
            self._theory_drop(self._p1, "1P")
        if tsumo_settled_p2:
            self._theory_drop(self._p2, "2P")

        # step3: 連鎖完了 (False→True 立ち上がり) で相殺を実行
        self._process_chain_offset(chain_p1, chain_p2)

        # step4: 前フレームの chain フラグを保存
        self._prev_chain_p1 = chain_p1
        self._prev_chain_p2 = chain_p2

        # step5: pending hard cap 適用 (修正C)
        self._apply_pending_cap(self._p1, "1P")
        self._apply_pending_cap(self._p2, "2P")

        # step6: visible_ojama クロスチェック → confidence 計算
        # 引数 visible_ojama が None の場合は _visible_ojama_* (update_from_boards 経由) を利用
        vis_p1 = visible_ojama_p1 if visible_ojama_p1 is not None else self._visible_ojama_p1
        vis_p2 = visible_ojama_p2 if visible_ojama_p2 is not None else self._visible_ojama_p2
        confidence = self._compute_confidence(vis_p1, vis_p2)

        return self._build_snapshot(t_sec, confidence)

    def get_snapshot(self, t_sec: float) -> OjamaAccountSnapshot:
        """イベントなしで現在状態のスナップショットを返す。"""
        confidence = CONFIDENCE_SCORE_OCR_ONLY
        return self._build_snapshot(t_sec, confidence)

    def update_from_boards(
        self,
        board_p1: "Board",
        board_p2: "Board",
        score_p1: int | None = None,
        score_p2: int | None = None,
        is_chain_p1: bool = False,
        is_chain_p2: bool = False,
    ) -> None:
        """confirmed_board からお邪魔落下・全消しを検出して帳簿に反映する。

        ① 落下検出 (補助系統): 画面内おじゃま増分 = 前フレームより増えた分 → pending 減算 + total_dropped 加算
           **STABLE ガード**: is_chain_p* = True の間は落下計上をスキップし、
           prev_visible_ojama ベースラインを更新しない。
           連鎖中は盤面が不安定でおじゃま増分が誤検知されるため。
           STABLE 復帰時に連鎖前ベースラインとの正味増分を 1 回だけ計上する。
           ※主系統は理論落下 (update_from_score の tsumo_settled 経由)。
             本メソッドの落下計上は補助的なクロスチェックとして残す。
        ② 全消し自動検出: confirmed_board が全 EMPTY (色ぷよ 0) かつ score > 0 → all_clear_pending セット
        ④ クロスチェック用 visible_ojama を内部に保持 (update_from_score で参照)

        Args:
            board_p1: 1P の confirmed_board。
            board_p2: 2P の confirmed_board。
            score_p1: 全消し判定用スコア (None なら全消し判定スキップ)。
            score_p2: 同 2P。
            is_chain_p1: 1P が連鎖中なら True。True の間は 1P 側の落下計上をスキップ。
                state_pipeline 層では ChainPhaseDetector.is_chain_p1 を渡すこと。
                TSUMO_FALL / OJAMA_FALL / GRAVITY_SETTLE は state_pipeline 層で
                判別不能なため is_chain ガードのみ適用し、残 limitation は申し送り参照。
            is_chain_p2: 2P が連鎖中なら True。同上。
        """
        vis_p1 = self._count_visible_ojama(board_p1)
        vis_p2 = self._count_visible_ojama(board_p2)

        # ① 落下検出 (STABLE ガード付き、補助系統)
        # 連鎖中フレームは prev_visible_ojama を固定して誤検知を防ぐ
        self._process_drop_from_boards(self._p1, vis_p1, skip_if_chain=is_chain_p1)
        self._process_drop_from_boards(self._p2, vis_p2, skip_if_chain=is_chain_p2)

        # ② 全消し自動検出 (score > 0 かつ色ぷよ 0)
        if score_p1 is not None:
            self._detect_all_clear(self._p1, board_p1, score_p1, "1P")
        if score_p2 is not None:
            self._detect_all_clear(self._p2, board_p2, score_p2, "2P")

        # ④ クロスチェック用 visible_ojama を保存
        self._visible_ojama_p1 = vis_p1
        self._visible_ojama_p2 = vis_p2

    def update_accounting_with_chain(
        self,
        t_sec: float,
        chain_p1: bool,
        chain_p2: bool,
    ) -> OjamaAccountSnapshot:
        """chain 確定後に相殺のみを実行してスナップショットを返す (③遅延修正用)。

        state_pipeline.extract() 末尾で chain_phase 確定後に呼ぶことで
        相殺の 1 フレーム遅延を解消する。
        update_from_score() とは独立して呼べる (score 差分処理は行わない)。

        Args:
            t_sec: 現在時刻 (秒)。
            chain_p1: 1P の連鎖完了フラグ (chain_phase 確定値)。
            chain_p2: 2P の連鎖完了フラグ。

        Returns:
            OjamaAccountSnapshot: 相殺反映後のスナップショット。
        """
        # 相殺処理 (False→True エッジで実行)
        self._process_chain_offset(chain_p1, chain_p2)
        # chain フラグ更新
        self._prev_chain_p1 = chain_p1
        self._prev_chain_p2 = chain_p2
        # pending hard cap 適用 (修正C)
        self._apply_pending_cap(self._p1, "1P")
        self._apply_pending_cap(self._p2, "2P")
        # visible_ojama クロスチェック込みで confidence 計算
        confidence = self._compute_confidence(
            self._visible_ojama_p1, self._visible_ojama_p2,
        )
        return self._build_snapshot(t_sec, confidence)

    # ============================
    # 内部: 理論落下 drain (修正B)
    # ============================

    def _theory_drop(self, side: _SideState, label: str) -> None:
        """stable 復帰ターンに pending から理論落下分を drain する。

        1 ターンで落下する ojama = min(THEORY_DROP_PER_TURN, pending)。
        pending=0 なら過剰 drain しない (下限 0)。

        Args:
            side: 片側お邪魔会計内部状態。
            label: ログ用 "1P"/"2P"。
        """
        if side.pending <= 0:
            return
        drop = min(THEORY_DROP_PER_TURN, side.pending)
        side.pending -= drop
        side.total_dropped += drop
        logger.debug(
            "theory_drop[%s]: drop=%d -> pending=%d total_dropped=%d",
            label, drop, side.pending, side.total_dropped,
        )

    # ============================
    # 内部: pending hard cap (修正C)
    # ============================

    def _apply_pending_cap(self, side: _SideState, label: str) -> None:
        """pending が PENDING_HARD_CAP を超えた場合に clamp する。

        超過分は total_generated に計上済のまま (pending 発散のみ防ぐ)。

        Args:
            side: 片側お邪魔会計内部状態。
            label: ログ用 "1P"/"2P"。
        """
        if side.pending > PENDING_HARD_CAP:
            logger.warning(
                "pending_cap[%s]: pending=%d > cap=%d, clamping",
                label, side.pending, PENDING_HARD_CAP,
            )
            side.pending = PENDING_HARD_CAP

    # ============================
    # 内部: confirmed_board からの落下・全消し検出
    # ============================

    @staticmethod
    def _count_visible_ojama(board: "Board") -> int:
        """盤面の可視領域 (行 1〜12) のおじゃまぷよ数をカウントする。

        隠し段 (行 0) は認識ノイズが多いため除外する。
        """
        from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, HIDDEN_ROWS
        count = 0
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                if int(board.get(row, col)) == COLOR_OJAMA:
                    count += 1
        return count

    def _process_drop_from_boards(
        self,
        side: _SideState,
        visible_ojama: int,
        skip_if_chain: bool = False,
    ) -> None:
        """盤面内おじゃま増分を落下量として帳簿に計上する (補助系統)。

        増分 = visible_ojama - prev_visible_ojama が正の場合:
            - pending から増分を減算 (落下した分は pending から消える)
            - total_dropped に加算
            - 増分が DROP_SANITY_CLAMP を超える場合は clamp して警告

        おじゃまが減った場合 (連鎖で消えた) は prev を更新するのみ。

        STABLE ガード:
            skip_if_chain=True の場合は連鎖中として落下計上とベースライン更新を
            両方スキップする。連鎖中→STABLE 復帰時に prev_visible_ojama が
            連鎖前の値のまま残るため、STABLE 復帰初回フレームで正味増分のみを
            1 回計上できる (連鎖中の中間変動を飲み込まない設計)。

        Args:
            side: 片側お邪魔会計内部状態。
            visible_ojama: 今フレームの盤面内おじゃまぷよ数。
            skip_if_chain: True なら落下計上・ベースライン更新を両方スキップ。
        """
        if skip_if_chain:
            # 連鎖中: 盤面不安定のため落下計上も prev 更新も行わない
            # prev_visible_ojama を固定することで、STABLE 復帰時に
            # 正味増分 (連鎖前 → 落下後) を 1 回だけ正しく計上できる
            logger.debug(
                "ojama_drop skipped (chain): visible=%d prev=%d",
                visible_ojama, side.prev_visible_ojama,
            )
            return
        delta = visible_ojama - side.prev_visible_ojama
        if delta > 0:
            # sanity clamp: 1ターン上限を超える増分は異常値
            if delta > DROP_SANITY_CLAMP:
                logger.warning(
                    "ojama_drop sanity clamp: delta=%d > limit=%d, "
                    "clamping to %d",
                    delta, DROP_SANITY_CLAMP, DROP_SANITY_CLAMP,
                )
                delta = DROP_SANITY_CLAMP
            # pending から落下分を減算 (下限 0)
            actual_drop = min(delta, max(0, side.pending))
            side.pending = max(0, side.pending - delta)
            side.total_dropped += actual_drop
            logger.debug(
                "ojama_drop: delta=%d actual=%d -> pending=%d total_dropped=%d",
                delta, actual_drop, side.pending, side.total_dropped,
            )
        # 前フレーム値を更新 (増減どちらでも)
        side.prev_visible_ojama = visible_ojama

    def _detect_all_clear(
        self,
        side: _SideState,
        board: "Board",
        score: int,
        side_label: str,
    ) -> None:
        """全消し状態を検出して all_clear_pending をセットする。

        検出条件: score > 0 かつ色ぷよ 0 個 (all_clear_detector.is_all_clear() 準拠)。
        既に all_clear_pending=True の場合は再セットしない (2回目の誤検知防止)。
        """
        if side.all_clear_pending:
            # 既に全消し持越し中なら二重セット不要
            return
        from src.all_clear_detector import is_all_clear
        result = is_all_clear(board, score)
        if result.is_all_clear:
            side.all_clear_pending = True
            logger.info(
                "all_clear_detected: side=%s score=%d -> all_clear_pending=True",
                side_label, score,
            )

    # ============================
    # 内部: score 差分から生成量を計算 (修正A: 境界reset含む)
    # ============================

    def _process_score_delta(
        self,
        score_p1: int | None,
        score_p2: int | None,
        elapsed: float,
    ) -> None:
        """score 差分が CHAIN_FIRE_MIN_SCORE 以上なら生成量を計算して相手 pending に加算。

        修正A: score 減少 (prev - cur >= SCORE_RESET_THRESHOLD) を試合境界とみなし、
        そのサイドの pending / leftover / all_clear_pending を 0 にリセットして return。
        リセット後は prev_score を新スコアで更新 (戻りジャンプを巨大増分計上しない)。
        試合横断の total_generated/total_offset/total_dropped はリセットしない
        (per-match 集計が必要な場合は呼出元でreset()を使うこと)。
        """
        if score_p1 is not None:
            if self._prev_score_p1 is not None:
                delta = score_p1 - self._prev_score_p1
                if score_p1 < self._prev_score_p1 - SCORE_RESET_THRESHOLD:
                    # 修正A: 試合境界検知 (score 大幅減少)
                    self._reset_match_side(self._p1, "1P", score_p1)
                    self._prev_score_p1 = score_p1
                elif CHAIN_FIRE_MIN_SCORE <= delta:
                    self._fire_generation(
                        sender=self._p1,
                        receiver=self._p2,
                        score_delta=delta,
                        elapsed=elapsed,
                    )
            self._prev_score_p1 = score_p1

        if score_p2 is not None:
            if self._prev_score_p2 is not None:
                delta = score_p2 - self._prev_score_p2
                if score_p2 < self._prev_score_p2 - SCORE_RESET_THRESHOLD:
                    # 修正A: 試合境界検知 (score 大幅減少)
                    self._reset_match_side(self._p2, "2P", score_p2)
                    self._prev_score_p2 = score_p2
                elif CHAIN_FIRE_MIN_SCORE <= delta:
                    self._fire_generation(
                        sender=self._p2,
                        receiver=self._p1,
                        score_delta=delta,
                        elapsed=elapsed,
                    )
            self._prev_score_p2 = score_p2

    def _reset_match_side(
        self,
        side: _SideState,
        label: str,
        new_score: int,
    ) -> None:
        """試合境界検知時にそのサイドの帳簿 (pending/leftover/all_clear) をリセットする。

        total_generated/total_offset/total_dropped は試合横断累積として保持する。
        per-match 集計が必要なら呼出元で reset() を使うこと。

        Args:
            side: 片側お邪魔会計内部状態。
            label: ログ用 "1P"/"2P"。
            new_score: 新しいスコア (ログ用)。
        """
        logger.info(
            "match_boundary[%s]: score reset detected (new_score=%d), "
            "clearing pending=%d leftover=%d all_clear=%s",
            label, new_score, side.pending, side.leftover, side.all_clear_pending,
        )
        side.pending = 0
        side.leftover = 0
        side.all_clear_pending = False
        side.prev_visible_ojama = 0

    def _fire_generation(
        self,
        sender: _SideState,
        receiver: _SideState,
        score_delta: int,
        elapsed: float,
    ) -> None:
        """発火側 score_delta から ojama 生成量を計算し受信側 pending に加算。"""
        # 全消し持越しボーナスを今回の score に上乗せ
        ac_bonus = ALL_CLEAR_BONUS if sender.all_clear_pending else 0
        effective_score = score_delta + ac_bonus
        sender.all_clear_pending = False  # 消費

        result = score_to_ojama(
            score=effective_score,
            prev_leftover=sender.leftover,
            elapsed_sec=elapsed,
            rate_base=self._rate_base,
        )
        sender.leftover = result.leftover_score
        generated = result.ojama_count
        sender.total_generated += generated
        receiver.pending += generated

        if generated > 0:
            logger.debug(
                "ojama_generation: score_delta=%d ac_bonus=%d -> %d ojama "
                "(leftover=%d rate=%d)",
                score_delta, ac_bonus, generated,
                result.leftover_score, result.effective_rate,
            )

    # ============================
    # 内部: 連鎖完了エッジで相殺
    # ============================

    def _process_chain_offset(self, chain_p1: bool, chain_p2: bool) -> None:
        """連鎖完了 (False→True の立ち上がり) タイミングで相殺を実行。

        相殺ロジック:
            1P が連鎖完了 →「1P が今回送った ojama」で「1P に向かう相手 pending (p1.pending)」を相殺
            余剰は 2P.pending に加算 (= 相手に上乗せ)
        """
        p1_edge = chain_p1 and not self._prev_chain_p1
        p2_edge = chain_p2 and not self._prev_chain_p2
        if p1_edge:
            # 1P 発火完了: 1P に向かう pending を 1P の今回生成で相殺
            self._offset(sender_state=self._p1, receiver_state=self._p2)
        if p2_edge:
            self._offset(sender_state=self._p2, receiver_state=self._p1)

    def _offset(
        self, sender_state: _SideState, receiver_state: _SideState,
    ) -> None:
        """sender の pending を receiver の pending で相殺する。

        相殺: receiver (自分に向かう pending) から sender の今回生成量を差し引き、
        余剰は送り返し (receiver.pending += surplus)。
        """
        # 今回の生成量 = total_generated の増分はここでは参照しにくいので
        # MVP では「自分に向かう pending が減る」方向で実装:
        # receiver.pending を sender.pending で相殺し、sender.pending をゼロにする
        # (設計思想: 1P が連鎖発火 → 自分に向かう2Pの pending を自分の連鎖で消す)
        sender_ojama = sender_state.pending
        target_pending = receiver_state.pending  # 相手が自分に向けた分
        if sender_ojama <= 0 or target_pending <= 0:
            return
        if sender_ojama <= target_pending:
            # 相手の pending を減らす (全部相殺できない)
            receiver_state.pending -= sender_ojama
            sender_state.total_offset += sender_ojama
            sender_state.pending = 0
            logger.debug(
                "ojama_offset: partial sender=%d target_before=%d -> target=%d",
                sender_ojama, target_pending, receiver_state.pending,
            )
        else:
            # 相手の pending を全消し、余剰は相手に加算
            surplus = sender_ojama - target_pending
            receiver_state.pending = 0
            sender_state.total_offset += target_pending
            sender_state.pending = 0
            receiver_state.pending += surplus
            logger.debug(
                "ojama_offset: full + surplus=%d -> receiver.pending=%d",
                surplus, receiver_state.pending,
            )

    # ============================
    # 内部: visible_ojama クロスチェック
    # ============================

    def _compute_confidence(
        self,
        visible_ojama_p1: int | None,
        visible_ojama_p2: int | None,
    ) -> float:
        """visible_ojama との乖離を見て confidence を返す。"""
        confidence = CONFIDENCE_SCORE_OCR_ONLY
        mismatch = False
        if visible_ojama_p1 is not None:
            diff = abs(self._p1.pending - visible_ojama_p1)
            if diff > VISIBLE_OJAMA_MISMATCH_THRESHOLD:
                mismatch = True
                logger.warning(
                    "visible_ojama_p1 mismatch: pending=%d visible=%d diff=%d",
                    self._p1.pending, visible_ojama_p1, diff,
                )
        if visible_ojama_p2 is not None:
            diff = abs(self._p2.pending - visible_ojama_p2)
            if diff > VISIBLE_OJAMA_MISMATCH_THRESHOLD:
                mismatch = True
                logger.warning(
                    "visible_ojama_p2 mismatch: pending=%d visible=%d diff=%d",
                    self._p2.pending, visible_ojama_p2, diff,
                )
        if not mismatch and (
            visible_ojama_p1 is not None or visible_ojama_p2 is not None
        ):
            confidence = CONFIDENCE_VISUAL_AGREE
        elif mismatch:
            confidence = max(
                0.0,
                CONFIDENCE_SCORE_OCR_ONLY - CONFIDENCE_VISUAL_MISMATCH_PENALTY,
            )
        return confidence

    # ============================
    # 内部: スナップショット構築
    # ============================

    def _build_snapshot(self, t_sec: float, confidence: float) -> OjamaAccountSnapshot:
        """現在の内部状態から OjamaAccountSnapshot を生成する。"""
        net_balance = self._p2.pending - self._p1.pending
        # 修正C: capped フィールド (有界値、指標用)
        p1_capped = min(self._p1.pending, PENDING_HARD_CAP)
        p2_capped = min(self._p2.pending, PENDING_HARD_CAP)
        return OjamaAccountSnapshot(
            t_sec=t_sec,
            pending_p1=self._p1.pending,
            pending_p2=self._p2.pending,
            total_generated_by_p1=self._p1.total_generated,
            total_generated_by_p2=self._p2.total_generated,
            total_offset_by_p1=self._p1.total_offset,
            total_offset_by_p2=self._p2.total_offset,
            total_dropped_to_p1=self._p1.total_dropped,
            total_dropped_to_p2=self._p2.total_dropped,
            net_ojama_balance=net_balance,
            overflow_risk_p1=self._p1.pending >= self._overflow_threshold,
            overflow_risk_p2=self._p2.pending >= self._overflow_threshold,
            confidence=float(confidence),
            leftover_p1=self._p1.leftover,
            leftover_p2=self._p2.leftover,
            all_clear_pending_p1=self._p1.all_clear_pending,
            all_clear_pending_p2=self._p2.all_clear_pending,
            # 修正C 追加フィールド (末尾)
            pending_p1_capped=p1_capped,
            pending_p2_capped=p2_capped,
            net_balance_capped=p2_capped - p1_capped,
        )

    def _elapsed(self, t_sec: float) -> float:
        """試合開始からの経過秒を返す (マージンタイム計算用)。"""
        if self._match_start_sec is None:
            return max(0.0, float(t_sec))
        return max(0.0, float(t_sec) - self._match_start_sec)


__all__ = [
    "CHAIN_FIRE_MIN_SCORE",
    "CONFIDENCE_SCORE_OCR_ONLY",
    "CONFIDENCE_VISUAL_AGREE",
    "CONFIDENCE_VISUAL_MISMATCH_PENALTY",
    "VISIBLE_OJAMA_MISMATCH_THRESHOLD",
    "DROP_SANITY_CLAMP",
    "SCORE_RESET_THRESHOLD",
    "THEORY_DROP_PER_TURN",
    "PENDING_HARD_CAP",
    "OjamaAccountSnapshot",
    "OjamaAccountingTracker",
]
