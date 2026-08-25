"""連鎖イベントに安定した chain_id を割り当てる純粋モジュール (2026-08-24)。

仕様: `docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md` §5 (chain_id の定義 — 断片の統合)。
`src/exchange_ledger.py` (Gate 2 会計コア) が要求する
「外側 (chain_id 付与器)」(同ファイル §5.2.3) を実装する。

## 責務の分離 (最重要)

この解決器は「このイベントは既存のどの連鎖のものか (同一性)」だけを答える。
「その値を信じるか」は答えない。それは `ExchangeLedger.finalize` の
許容帯ゲート (`FINALIZE_DOWNWARD_TOLERANCE`) の仕事である。
この分離をしないと、条件を外れたイベントが新しい chain_id を取って二重計上する。

## 既定 OFF・未配線

`src/recognition_pipeline.py` / `src/chain_detector.py` / `src/score_ocr.py` /
`src/ojama_accounting.py` / `scripts/visualize_advantage_overlay.py` /
`src/production_config.py` のいずれからも呼ばれない。呼び出し配線は別タスク。

## 純粋性

I/O なし、グローバル状態なし、乱数なし、時計を読まない (時刻は必ず引数で受ける)。
`resolve_chain_ids` は同じ入力なら必ず同じ出力を返し、入力を書き換えない。
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto

# ============================
# 定数 (物理量から導出。シーンからの逆算はしない)
# ============================

# 連鎖アニメーション時間の実測式 (23動画418イベント):
#   連鎖アニメ秒数 ≈ 2.61 + 1.17 × 連鎖数
# 出典: docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md:367-368
#   (「実測の連鎖アニメ時間は 2.61 + 1.17×N (23動画418イベント)。
#     15連鎖でも 20.2秒であり、30秒は十分な余裕がある。」)
# CHAIN_ID_MAX_SEC 自体の既定値 30.0 も同ドキュメント §5.4 (:364) が定める。
# 信号 (CHAIN_END_SIGNAL / MATCH_BOUNDARY) を取り逃した場合の保険であり、
# 通常経路では発火しない (発火したらカウンタで必ず可視化する)。
CHAIN_ID_MAX_SEC: float = 30.0

# SCORE_FINALIZE による確定クローズの後、同じ物理連鎖の残響 (echo) として
# CHAIN_SETTLED を吸収できる時間 (P1-2、2026-08-25)。
# 物理的根拠 (シーンからの逆算ではない): 同一 side の連鎖は直列であり、
# 確定クローズの後に始まった**次の**連鎖の終わり (CHAIN_SETTLED) は、
# 連鎖アニメ実測式 2.61 + 1.17×N (23動画418イベント、上記出典と同一) より
# 最短でも 2.61 + 1.17×1 = 3.78 秒後にしか観測できない。それより早く届いた
# CHAIN_SETTLED は、確定クローズ済みの同一連鎖に対する遅れた合図しかあり得ない。
SETTLED_ECHO_MAX_SEC: float = 2.61 + 1.17

# 確定値 (`ResolvedChain.finalized_source`) の出所を表す文字列定数
# (2026-08-25 追加)。`src/exchange_ledger.py` の FINALIZE 供給源ゲート
# (I16、`FINALIZE_SOURCE_SCORE_OCR_DIFF`) と文字列として一致していなければ
# ならない (`test_finalize_source_constant_matches_chain_id_resolver` で固定)。
# ここで import せず定数で重複させるのは、純粋コアどうしの依存を
# 増やさない方針 (`OJAMA_MAX_DROP_PER_TURN` と同じ理由)。
FINALIZED_SOURCE_SCORE_OCR_DIFF: str = "score_ocr_diff"
FINALIZED_SOURCE_SIMULATE_FALLBACK: str = "simulate_fallback"


class ObservationKind(Enum):
    """chain_id 解決器が受け付ける観測イベントの種別。

    **【2026-08-25 是正】** 旧 `BASELINE` は「score_ocr 由来の確定値」と
    誤って docstring に書かれていたが、実体は `ChainSimulator` による
    盤面からの**推定**だった (`src/chain_detector.py:277-317`
    `_try_emit_event` が `self._simulator.simulate(...)` の結果をそのまま
    `total_score` に入れている)。推定値の信頼性は既に両方向で否定されて
    いる (memory `project_chain_count_both_untrustworthy_2026-07-30`:
    過小方向の実例。2026-08-25 実測: 過大方向にも 38→745 個という壊滅的な
    外れを確認)。**「連鎖が終わった合図」と「値の権威」を分離**するため、
    `BASELINE` を `CHAIN_SETTLED` (合図のみ) と `SCORE_FINALIZE`
    (値の権威) に分割した (fable アーキ裁定)。
    """

    FORMULA_STEP = auto()      # 掛け算式 1 段の確定 (成長フェーズ)
    CHAIN_SETTLED = auto()     # 推定 (ChainSimulator) 由来の「連鎖が終わった」合図。
                               # **値は使わない。** growth_observed=True なら
                               # AWAITING_FINALIZE へ進めるだけ (CHAIN_END_SIGNAL と同じ経路)。
                               # growth_observed=False かつ SCORE_FINALIZE も来ない
                               # ときだけ、値そのものを低信頼フォールバックとして使う。
    SCORE_FINALIZE = auto()    # score OCR 確定差分 (`OjamaAccountingTracker` 由来)。
                               # **値の権威を持つのはこれだけ。** GROWING /
                               # AWAITING_FINALIZE のどちらからでも FINALIZE する。
    CHAIN_END_SIGNAL = auto()  # 連鎖の終わりの絶対律 (ネクスト始動 or おじゃま着弾)
    MATCH_BOUNDARY = auto()    # 試合の切り替わり


class CloseReason(Enum):
    """連鎖が CLOSED になった経路 (診断用。なぜ閉じたかを必ず追えるようにする)。"""

    FINALIZED = auto()      # SCORE_FINALIZE、または CHAIN_SETTLED の低信頼フォールバックによる確定
    SUPERSEDED = auto()     # AWAITING_FINALIZE 中に次の成長が来て確定なしで閉じた
    STEP_DECREASE = auto()  # GROWING 中に段数が (running max より) 減少した
    FORCE_CUT = auto()      # CHAIN_ID_MAX_SEC 超過による強制打ち切り
    MATCH_BOUNDARY = auto()  # 試合境界による強制終了
    STREAM_END = auto()     # 入力ストリーム終端 (flush)。5 種のいずれにも該当しない残骸
    #
    # 【報告】STREAM_END はコーディネーター指定の 5 種 (FINALIZED / SUPERSEDED /
    # STEP_DECREASE / FORCE_CUT / MATCH_BOUNDARY) に含まれない。`flush()` API
    # (入力ストリームが尽きたときに残っている in-flight を閉じる) を維持するために
    # 追加した第 6 の値。5 種のどれにも本質的に一致しないため、既存の値へ
    # 無理に押し込まず新設した。承認が必要なら差し戻してほしい。


@dataclass(frozen=True)
class ChainObservation:
    """連鎖に関する観測事実 1 件 (不変値)。

    **`trigger_sec` を持たない。** `src/chain_detector.py` の baseline イベントは
    `trigger_sec=self._last_stable_t` (連鎖開始時刻ではない) を使うため、
    `trigger_sec` の一致で連鎖の同一性を判定してはならない
    (docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md 起因の実測不整合)。
    """

    side: str                       # "1P" / "2P"
    t_sec: float                    # 観測時刻
    kind: ObservationKind
    chain_count: int = 0            # 掛け算式の段の通番 (CHAIN_SETTLED/SCORE_FINALIZE は確定連鎖数)
    total_score: int = 0            # 累積スコア相当値 (kind により意味が異なる)
    mechanism: str | None = None    # 診断用。判定には使わない


@dataclass(frozen=True)
class ResolvedChain:
    """1 本の連鎖として解決された結果 (不変値)。"""

    chain_id: int
    side: str
    opened_at_sec: float
    closed_at_sec: float
    step_count: int
    provisional_score: int
    finalized_score: int | None
    was_finalized: bool
    force_cut: bool
    close_reason: CloseReason
    # 成長フェーズ (FORMULA_STEP) を 1 件でも観測できたか。
    # False は「掛け算式を読めず CHAIN_SETTLED/SCORE_FINALIZE だけで発行即クローズ
    # した連鎖」を指す。
    # provisional_score にはこの場合 CHAIN_SETTLED/SCORE_FINALIZE の値を入れるが、
    # それは実測ではなく唯一手に入った値の転記であることをこのフラグで正直に示す
    # (実測していない値に 0 を入れない、という直近の方針に合わせる)。
    growth_observed: bool
    # 確定値 (finalized_score) の出所 (2026-08-25 追加)。
    # "score_ocr_diff" = SCORE_FINALIZE (score OCR 確定差分、値の権威)。
    # "simulate_fallback" = CHAIN_SETTLED の値を低信頼フォールバックとして使った
    # (growth_observed=False かつ SCORE_FINALIZE が一度も来なかった場合のみ)。
    # None = was_finalized=False (値が確定していない)。
    finalized_source: str | None = None


@dataclass(frozen=True)
class ResolverStats:
    """診断用カウンタ。黙って打ち切らない・黙って捨てないための可視化。"""

    opened_count: int
    finalized_count: int
    unfinalized_close_count: int
    force_cut_count: int
    count_decrease_split_count: int
    # 対応する in-flight が無いまま届いた CHAIN_END_SIGNAL の件数。
    # 絶対律検出器は既知の事故源 (project_slide_false_positive_root_cause_2026-08-22) なので、
    # ここが多ければ検出器が壊れている証拠として扱う。
    orphan_end_signal_count: int
    # ---- P1-2 (2026-08-25) の可視化カウンタ。0 は必ず母数と並べて読む ----
    # 受け取った CHAIN_SETTLED の総数 (settled_echo_absorbed_count の母数)。
    chain_settled_received_count: int = 0
    # SCORE_FINALIZE 確定クローズ直後 (SETTLED_ECHO_MAX_SEC 以内) の
    # CHAIN_SETTLED を同一連鎖の残響として吸収した件数。
    settled_echo_absorbed_count: int = 0
    # 途中確定クローズの後に段継続 (cc が前 chain の running max 超) で
    # 新 chain を開き、前 chain の確定クローズ済み累積点を控除した件数
    # (母数 = opened_count)。
    continuation_reopen_count: int = 0
    # 継続控除で累積点が控除基準を下回った (OCR 異常の証拠) 件数
    # (母数 = continuation_reopen_count)。黙って 0 に丸めず必ず数える。
    continuation_base_underflow_count: int = 0


class _SideState(Enum):
    """side ごとの内部状態。"""

    GROWING = auto()
    AWAITING_FINALIZE = auto()


@dataclass
class _InFlight:
    """side ごとに最大 1 本だけ存在する進行中の連鎖 (内部可変状態)。"""

    chain_id: int
    opened_at_sec: float
    last_t_sec: float
    state: _SideState
    step_count: int
    provisional_score: int
    # FORMULA_STEP を 1 件でも観測したか (P1-2、2026-08-25)。
    # False = CHAIN_SETTLED だけで開いた「確定待ち保留」(settled-pending)。
    growth_observed: bool = True
    # settled-pending が保持する低信頼推定値 (CHAIN_SETTLED の total_score)。
    # SCORE_FINALIZE が来ないまま閉じるときだけフォールバックとして使う。
    settled_fallback_score: int | None = None
    # 途中確定クローズからの段継続 (P1-2 経路B) で開いた場合の控除基準
    # (前 chain の確定クローズ済み累積点)。FORMULA_STEP の累積値から
    # これを引いた値を provisional_score に入れる。通常は 0。
    score_base: int = 0


@dataclass(frozen=True)
class _FinalizedTail:
    """side ごとの「直近に SCORE_FINALIZE で確定クローズした連鎖」の記憶
    (P1-2、2026-08-25)。段継続の控除 (経路B) と settled 残響の吸収 (経路A の
    順序違い) の判定にだけ使う。MATCH_BOUNDARY で必ず消す。"""

    step_count: int          # 確定クローズ時の running max (段番号の継続判定に使う)
    cumulative_score: int    # 掛け算式の累積点 (score_base 込みの生値。控除基準)
    opened_at_sec: float     # 物理連鎖の最大長 (CHAIN_ID_MAX_SEC) の判定基準
    closed_at_sec: float     # 残響時間 (SETTLED_ECHO_MAX_SEC) の判定基準
    growth_observed: bool    # False なら cumulative_score は控除に使えない (単位が違う)


class ChainIdResolver:
    """side ごとに独立した状態機械で chain_id を割り当てる。

    仕様の状態機械 (docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md §5.2〜§5.4):
    なし → GROWING → AWAITING_FINALIZE → CLOSED。
    「連鎖が終わった」ことと「会計が閉じた」ことは別として扱う。
    """

    def __init__(self) -> None:
        self._in_flight: dict[str, _InFlight] = {}
        self._resolved: list[ResolvedChain] = []
        self._next_chain_id: int = 1
        self._opened_count: int = 0
        self._finalized_count: int = 0
        self._unfinalized_close_count: int = 0
        self._force_cut_count: int = 0
        self._count_decrease_split_count: int = 0
        self._orphan_end_signal_count: int = 0
        # P1-2 (2026-08-25): side ごとの直近確定クローズの記憶と可視化カウンタ。
        self._finalized_tail: dict[str, _FinalizedTail] = {}
        self._chain_settled_received_count: int = 0
        self._settled_echo_absorbed_count: int = 0
        self._continuation_reopen_count: int = 0
        self._continuation_base_underflow_count: int = 0

    # ------------------------------
    # 受付
    # ------------------------------

    def push(self, obs: ChainObservation) -> None:
        """観測を 1 件受け付ける。t_sec の順序は問わない (呼び出し側は昇順推奨)。"""
        if obs.kind is ObservationKind.MATCH_BOUNDARY:
            self._handle_match_boundary(obs)
            return
        # 強制打ち切りは **両 side** に対して行う。時間はどちらの side でも同じだけ
        # 進むため、観測が届いた side だけを見ると、相手が窒息する等で観測が途絶えた
        # side の連鎖が永久に開いたままになり、台帳の episode が閉じられなくなる
        # (相手を見失うこと自体は「応手不能」という正常な情報であり、
        #  memory reference_opponent_blindness_is_information_2026-07-29 のとおり
        #  欠測として扱ってはならない)。
        for _side in list(self._in_flight):
            self._maybe_force_cut(_side, obs.t_sec)
        state = self._in_flight.get(obs.side)
        if obs.kind is ObservationKind.FORMULA_STEP:
            self._handle_formula_step(obs, state)
        elif obs.kind is ObservationKind.CHAIN_SETTLED:
            self._handle_chain_settled(obs, state)
        elif obs.kind is ObservationKind.SCORE_FINALIZE:
            self._handle_score_finalize(obs, state)
        elif obs.kind is ObservationKind.CHAIN_END_SIGNAL:
            self._handle_chain_end_signal(obs, state)

    def flush(self) -> None:
        """残っている in-flight を未確定のまま閉じる (入力ストリーム終端)。"""
        for side in list(self._in_flight):
            self._close_side(side, finalized_score=None, reason=CloseReason.STREAM_END)

    # ------------------------------
    # 種別ごとの遷移
    # ------------------------------

    def _handle_formula_step(self, obs: ChainObservation, state: _InFlight | None) -> None:
        """FORMULA_STEP の遷移 (段を積む / 段減少で分割 / 新規開始)。"""
        if state is None:
            self._open(obs)
            return
        if state.state is _SideState.AWAITING_FINALIZE:
            self._handle_formula_step_while_awaiting_finalize(obs, state)
            return
        # 段数比較は「これまでの最大値 (running max)」に対して行う。**直前の生値とは
        # 比較しない。** 掛け算式の段カウントは実測で 38/38 単調
        # (data/verify/gate3_chainid_2026-08-24/summary.json)。読み取りの一時的な
        # 揺れ (OCR jitter) 1 回で連鎖を誤って分割しないための安全側の比較。
        if obs.chain_count < state.step_count:
            # running max を下回る = 別の物理連鎖の開始とみなす
            self._close_side(
                obs.side, finalized_score=None, reason=CloseReason.STEP_DECREASE,
            )
            self._open(obs)
            return
        state.last_t_sec = obs.t_sec
        state.step_count = max(state.step_count, obs.chain_count)
        state.provisional_score = self._subtract_base(obs.total_score, state.score_base)

    def _handle_formula_step_while_awaiting_finalize(
        self, obs: ChainObservation, state: _InFlight,
    ) -> None:
        """AWAITING_FINALIZE 中に FORMULA_STEP が来たときの遷移 (Fix【3】、2026-08-25)。

        **物理的根拠 (シーン逆算ではなくゲームの規則からの導出)**: すべての
        連鎖は 1 段目から始まる。新しい連鎖の最初の段は必ず `chain_count=1`。
        掛け算式の段カウントは実測で 38/38 単調
        (`data/verify/gate3_chainid_2026-08-24/summary.json`)。したがって
        「現在の running max より大きい段数」が観測できるのは、
        **その連鎖が (終了信号を早合点しただけで) まだ続いている場合しか
        あり得ない**。新規連鎖なら必ず 1 から数え直すため、running max を
        上回ることは物理的に起こらない。

        `obs.chain_count > state.step_count`: 同一連鎖の継続とみなし、
        chain_id を変えずに GROWING へ戻す (段・スコアを更新するだけ)。
        `obs.chain_count <= state.step_count`: 新しい連鎖の開始 (1 段目
        から数え直された、または running max 以下の値) とみなし、
        従来どおり SUPERSEDED で閉じてから新規発行する。

        **実データでの検証結果 (v51、4/4 正解、報告参照)**: 1→2 (継続) /
        1→2 (継続) / 2→6 (継続) / 10→1・10 秒後 (新規) の 4 例すべてで
        この判定規則が正解と一致した。

        **settled-pending (growth_observed=False) の場合 (P1-2、2026-08-25)**:
        保留中の推定値と掛け算式の累積値は別の物理連鎖に属するとみなし、
        保留を閉じて (SCORE_FINALIZE が来なかったので低信頼フォールバック
        確定になる、`_close_side` 参照) 新規に開く。単位も観測経路も異なる
        値を同一 chain に混ぜない。
        """
        if not state.growth_observed:
            self._close_side(obs.side, finalized_score=None, reason=CloseReason.SUPERSEDED)
            self._open(obs)
            return
        if obs.chain_count > state.step_count:
            state.state = _SideState.GROWING
            state.last_t_sec = obs.t_sec
            state.step_count = obs.chain_count
            state.provisional_score = self._subtract_base(obs.total_score, state.score_base)
            return
        self._close_side(obs.side, finalized_score=None, reason=CloseReason.SUPERSEDED)
        self._open(obs)

    def _handle_chain_settled(self, obs: ChainObservation, state: _InFlight | None) -> None:
        """CHAIN_SETTLED (推定由来の「連鎖が終わった」合図)。**値は使わない。**

        growth_observed=True (in-flight が既にある) なら、絶対律
        (`CHAIN_END_SIGNAL`) と全く同じ経路で AWAITING_FINALIZE へ
        進めるだけ (値には一切触れない)。

        **【P1-2 是正、2026-08-25 Codex レビュー】in-flight が無い場合は
        即クローズしない。** 旧実装はここで発行即クローズしていたため、
        直後に届く SCORE_FINALIZE が別の chain_id を取り、1 物理連鎖から
        2 ID が生まれていた (Codex 最小再現: settled t=10.0 → finalize
        t=10.1 で opened_count=2)。是正後は:
        - 直近の確定クローズの残響時間内なら吸収する (`_is_settled_echo`)。
        - それ以外は settled-pending (AWAITING_FINALIZE、growth なし) として
          保留し、後続の SCORE_FINALIZE と統合する。SCORE_FINALIZE が
          来ないまま閉じるときだけ、従来どおり低信頼フォールバックとして
          推定値を確定値に使う (`_close_side` の変換、fable アーキ裁定の維持)。
        """
        self._chain_settled_received_count += 1
        if state is None:
            if self._is_settled_echo(obs):
                self._settled_echo_absorbed_count += 1
                return
            self._open_settled_pending(obs)
            return
        if not state.growth_observed:
            # 保留中にもう 1 つ settled が来た = 別の物理連鎖の終わり。
            # 先の保留を閉じ (低信頼フォールバック確定になる)、新たに保留する。
            self._close_side(obs.side, finalized_score=None, reason=CloseReason.SUPERSEDED)
            self._open_settled_pending(obs)
            return
        self._handle_chain_end_signal(obs, state)

    def _is_settled_echo(self, obs: ChainObservation) -> bool:
        """直近の SCORE_FINALIZE 確定クローズの残響 (同一物理連鎖の遅れた合図) か。

        同一 side の連鎖は直列なので、確定クローズの後に始まった次の連鎖の
        終わりは最短でも SETTLED_ECHO_MAX_SEC (= 2.61+1.17 秒、連鎖アニメ
        実測式の 1 連鎖ぶん) 後にしか来ない (定数定義の物理的根拠を参照)。
        """
        tail = self._finalized_tail.get(obs.side)
        if tail is None:
            return False
        return 0.0 <= obs.t_sec - tail.closed_at_sec <= SETTLED_ECHO_MAX_SEC

    def _open_settled_pending(self, obs: ChainObservation) -> None:
        """CHAIN_SETTLED (in-flight 無し) を確定待ちの保留として開く (P1-2)。

        「score 確定が来ない」と判断できる時点 (別イベントによるクローズ)
        までフォールバック確定を遅延する。値は保留に保持するだけで、
        この時点では会計上の意味を持たない。
        """
        chain_id = self._issue_chain_id()
        self._in_flight[obs.side] = _InFlight(
            chain_id=chain_id, opened_at_sec=obs.t_sec, last_t_sec=obs.t_sec,
            state=_SideState.AWAITING_FINALIZE, step_count=obs.chain_count,
            provisional_score=obs.total_score, growth_observed=False,
            settled_fallback_score=obs.total_score,
        )

    def _handle_score_finalize(self, obs: ChainObservation, state: _InFlight | None) -> None:
        """SCORE_FINALIZE (score OCR 確定差分)。**値の権威を持つのはこれだけ。**

        GROWING/AWAITING_FINALIZE のどちらからでも、この値で FINALIZE する。
        settled-pending (growth なしの保留) があれば同じ chain_id に統合する
        (P1-2、1 物理連鎖 1 ID)。
        """
        if state is None:
            # 掛け算式を一度も観測できなかった連鎖。発行して即クローズする。
            # provisional_score は「実測していない値」なので 0 埋めせず、
            # 唯一手に入った SCORE_FINALIZE の値をそのまま入れる
            # (growth_observed=False でそれが成長フェーズの実測ではないことを明示)。
            self._close_immediately_without_growth(
                obs, finalized_source=FINALIZED_SOURCE_SCORE_OCR_DIFF,
            )
            return
        if not state.growth_observed:
            self._merge_settled_pending_with_finalize(obs, state)
            return
        self._close_side(
            obs.side, finalized_score=obs.total_score, reason=CloseReason.FINALIZED,
            event_t_sec=obs.t_sec, finalized_source=FINALIZED_SOURCE_SCORE_OCR_DIFF,
        )

    def _merge_settled_pending_with_finalize(
        self, obs: ChainObservation, state: _InFlight,
    ) -> None:
        """settled-pending へ届いた SCORE_FINALIZE を同一 chain_id に統合する (P1-2)。

        provisional_score にも権威値 (SCORE_FINALIZE) を転記する。保留していた
        推定値 (settled_fallback_score) は**捨てる** — 単位 (スコア点) も
        観測経路 (ChainSimulator) も権威値 (おじゃま個数、score OCR 確定差分)
        と異なり、混ぜると下流の単位換算 (`_provisional_ojama_for` の
        growth_observed=False 例外) を壊すため。
        """
        self._in_flight.pop(obs.side, None)
        closed_at = obs.t_sec
        self._resolved.append(ResolvedChain(
            chain_id=state.chain_id, side=obs.side,
            opened_at_sec=state.opened_at_sec, closed_at_sec=closed_at,
            step_count=max(state.step_count, obs.chain_count),
            provisional_score=obs.total_score,
            finalized_score=obs.total_score, was_finalized=True, force_cut=False,
            close_reason=CloseReason.FINALIZED, growth_observed=False,
            finalized_source=FINALIZED_SOURCE_SCORE_OCR_DIFF,
        ))
        self._finalized_count += 1
        self._record_finalized_tail(
            obs.side, step_count=max(state.step_count, obs.chain_count),
            cumulative_score=0, opened_at_sec=state.opened_at_sec,
            closed_at_sec=closed_at, growth_observed=False,
        )

    def _close_immediately_without_growth(
        self, obs: ChainObservation, finalized_source: str,
    ) -> None:
        """成長フェーズを一度も観測できなかった連鎖を、発行して即クローズする。

        (P1-2 是正後は `SCORE_FINALIZE` の「in-flight が無い」分岐だけが使う。
        `CHAIN_SETTLED` 側は保留 `_open_settled_pending` に置き換えた。)
        `finalized_source` で値の出所を必ず区別して記録する
        (黙って同じ扱いにしない)。
        """
        chain_id = self._issue_chain_id()
        self._resolved.append(ResolvedChain(
            chain_id=chain_id, side=obs.side,
            opened_at_sec=obs.t_sec, closed_at_sec=obs.t_sec,
            step_count=obs.chain_count, provisional_score=obs.total_score,
            finalized_score=obs.total_score, was_finalized=True, force_cut=False,
            close_reason=CloseReason.FINALIZED, growth_observed=False,
            finalized_source=finalized_source,
        ))
        self._finalized_count += 1
        if finalized_source == FINALIZED_SOURCE_SCORE_OCR_DIFF:
            # 残響吸収 (`_is_settled_echo`) のために確定クローズを記憶する。
            # cumulative_score=0 / growth_observed=False: この値は
            # おじゃま個数であり、段継続の控除 (スコア点) には使えない。
            self._record_finalized_tail(
                obs.side, step_count=obs.chain_count, cumulative_score=0,
                opened_at_sec=obs.t_sec, closed_at_sec=obs.t_sec,
                growth_observed=False,
            )

    def _handle_chain_end_signal(self, obs: ChainObservation, state: _InFlight | None) -> None:
        """終わりの絶対律。GROWING を AWAITING_FINALIZE へ進める。"""
        if state is None:
            # 対応する in-flight が無い迷子信号。絶対律検出器は既知の事故源
            # (project_slide_false_positive_root_cause_2026-08-22) なので、
            # 無視はするが黙って捨てず必ずカウンタに残す。
            self._orphan_end_signal_count += 1
            return
        state.state = _SideState.AWAITING_FINALIZE
        state.last_t_sec = obs.t_sec

    def _handle_match_boundary(self, obs: ChainObservation) -> None:
        """試合境界。両サイドの in-flight を確定値なしで閉じ、状態を全消去する。

        確定クローズの記憶 (`_finalized_tail`) も必ず消す (P1-2)。前試合の
        確定分を次試合の連鎖から控除したり、前試合の残響として次試合の
        settled を吸収したりしてはならない。
        """
        for side in list(self._in_flight):
            self._close_side(side, finalized_score=None, reason=CloseReason.MATCH_BOUNDARY)
        self._finalized_tail.clear()

    # ------------------------------
    # 開始・終了・強制打ち切り
    # ------------------------------

    def _maybe_force_cut(self, side: str, t_sec: float) -> None:
        """CHAIN_ID_MAX_SEC を超えて開いたままの連鎖を強制的に打ち切る (信号の取り逃し対策)。"""
        state = self._in_flight.get(side)
        if state is None:
            return
        if t_sec - state.opened_at_sec > CHAIN_ID_MAX_SEC:
            self._close_side(side, finalized_score=None, reason=CloseReason.FORCE_CUT)

    def _open(self, obs: ChainObservation) -> None:
        """新しい chain_id を発行して GROWING で開始する (FORMULA_STEP 専用)。

        **【P1-2 経路B 是正、2026-08-25】** 直前に同 side が SCORE_FINALIZE で
        確定クローズした後、段番号が継続 (running max 超) して開く場合は
        同一物理連鎖の続きなので、前 chain の確定クローズ済み累積点を
        暫定から控除して初期化する (v51 実測: chain5 の暫定 54,230 が
        chain4 確定済み 21,570 を丸ごと包含 → D7 で +308 個の二重計上)。
        """
        base = self._continuation_base(obs)
        chain_id = self._issue_chain_id()
        self._in_flight[obs.side] = _InFlight(
            chain_id=chain_id, opened_at_sec=obs.t_sec, last_t_sec=obs.t_sec,
            state=_SideState.GROWING, step_count=obs.chain_count,
            provisional_score=self._subtract_base(obs.total_score, base),
            growth_observed=True, score_base=base,
        )

    def _continuation_base(self, obs: ChainObservation) -> int:
        """段継続で開く場合の控除基準 (前 chain の確定クローズ済み累積点) を返す。

        継続の判定は物理量からの導出のみ (シーンからの逆算はしない):
        - 段番号が前 chain の running max を上回る (すべての連鎖は 1 段目
          から始まるため、新規連鎖では物理的に起こらない。実測 38/38 単調)。
        - 前 chain の開始から CHAIN_ID_MAX_SEC 以内 (1 本の物理連鎖の最大長)。
        - 前 chain が成長観測済み (cumulative_score がスコア点であること)。
        継続でなければ記憶を破棄して 0 を返す (次の物理連鎖が始まった証拠)。
        """
        tail = self._finalized_tail.get(obs.side)
        if tail is None:
            return 0
        is_continuation = (
            tail.growth_observed
            and obs.chain_count > tail.step_count
            and 0.0 <= obs.t_sec - tail.opened_at_sec <= CHAIN_ID_MAX_SEC
        )
        if not is_continuation:
            self._finalized_tail.pop(obs.side, None)
            return 0
        self._continuation_reopen_count += 1
        return tail.cumulative_score

    def _subtract_base(self, total_score: int, base: int) -> int:
        """累積点から控除基準を引く。下回ったら 0 に丸め、必ず数える。

        base=0 (通常の連鎖) では常に無変換で、丸めもカウントも起こらない
        (total_score は OCR 由来の非負値)。
        """
        adjusted = total_score - base
        if adjusted < 0:
            self._continuation_base_underflow_count += 1
            return 0
        return adjusted

    def _record_finalized_tail(
        self, side: str, *, step_count: int, cumulative_score: int,
        opened_at_sec: float, closed_at_sec: float, growth_observed: bool,
    ) -> None:
        """SCORE_FINALIZE による確定クローズを side ごとに記憶する (P1-2)。"""
        self._finalized_tail[side] = _FinalizedTail(
            step_count=step_count, cumulative_score=cumulative_score,
            opened_at_sec=opened_at_sec, closed_at_sec=closed_at_sec,
            growth_observed=growth_observed,
        )

    def _issue_chain_id(self) -> int:
        chain_id = self._next_chain_id
        self._next_chain_id += 1
        self._opened_count += 1
        return chain_id

    def _close_side(
        self, side: str,
        finalized_score: int | None, reason: CloseReason,
        event_t_sec: float | None = None,
        finalized_source: str | None = None,
    ) -> None:
        """side の in-flight を確定し、結果を記録してカウンタを更新する。

        `closed_at_sec` は経路によって基準が異なる (コーディネーター指定):
        - FINALIZED: `event_t_sec` (= SCORE_FINALIZE の t_sec。確定した瞬間が分かっている)
        - それ以外すべて: `state.last_t_sec` (その連鎖の存在を最後に観測した時刻。
          境界イベント自体の時刻等、根拠のない時刻を記録しないため)

        `finalized_source` は FINALIZED のときだけ意味を持つ。
        `was_finalized=False` の経路 (SUPERSEDED/STEP_DECREASE/FORCE_CUT/
        MATCH_BOUNDARY/STREAM_END) では、呼び出し側が誤って渡しても
        黙って `None` に落とす (値が確定していないのに出所だけ残る
        矛盾を防ぐ)。

        **settled-pending の変換 (P1-2、2026-08-25)**: growth なしの保留
        (`settled_fallback_score` 保持) が SCORE_FINALIZE 以外の経路で
        閉じるときは、「score 確定は来なかった」と確定した時点なので、
        従来の即クローズと同じ形 (FINALIZED / simulate_fallback) で閉じる
        (旧挙動と出力を揃える。fable アーキ裁定のフォールバック仕様の維持)。
        """
        state = self._in_flight.pop(side, None)
        if state is None:
            return
        if reason is not CloseReason.FINALIZED and self._close_as_settled_fallback(side, state):
            return
        was_finalized = reason is CloseReason.FINALIZED
        closed_at_sec = event_t_sec if was_finalized else state.last_t_sec
        self._resolved.append(ResolvedChain(
            chain_id=state.chain_id, side=side,
            opened_at_sec=state.opened_at_sec, closed_at_sec=closed_at_sec,
            step_count=state.step_count, provisional_score=state.provisional_score,
            finalized_score=finalized_score, was_finalized=was_finalized,
            force_cut=(reason is CloseReason.FORCE_CUT),
            close_reason=reason, growth_observed=state.growth_observed,
            finalized_source=finalized_source if was_finalized else None,
        ))
        if was_finalized:
            self._finalized_count += 1
        else:
            self._unfinalized_close_count += 1
        if reason is CloseReason.FORCE_CUT:
            self._force_cut_count += 1
        if reason is CloseReason.STEP_DECREASE:
            self._count_decrease_split_count += 1
        if was_finalized and finalized_source == FINALIZED_SOURCE_SCORE_OCR_DIFF:
            # 段継続の控除 (経路B) と残響吸収 (経路A の順序違い) のための記憶。
            # cumulative_score は score_base 込みの生の累積点 (次の継続 chain の
            # FORMULA_STEP はさらに大きい生の累積値を運んでくるため)。
            self._record_finalized_tail(
                side, step_count=state.step_count,
                cumulative_score=state.provisional_score + state.score_base,
                opened_at_sec=state.opened_at_sec,
                closed_at_sec=closed_at_sec if closed_at_sec is not None else state.last_t_sec,
                growth_observed=state.growth_observed,
            )

    def _close_as_settled_fallback(self, side: str, state: _InFlight) -> bool:
        """settled-pending を低信頼フォールバックとして確定クローズする (P1-2)。

        対象は growth なしで `settled_fallback_score` を保持する保留のみ。
        SCORE_FINALIZE が最後まで来なかったので、保留していた推定値を
        そのまま確定値に使う (旧「即クローズ」と同じ出力: close_reason=
        FINALIZED / finalized_source=simulate_fallback / force_cut=False)。
        呼び出し元 (`_close_side`) は state を pop 済みであること。
        """
        if state.growth_observed or state.settled_fallback_score is None:
            return False
        self._resolved.append(ResolvedChain(
            chain_id=state.chain_id, side=side,
            opened_at_sec=state.opened_at_sec, closed_at_sec=state.last_t_sec,
            step_count=state.step_count, provisional_score=state.provisional_score,
            finalized_score=state.settled_fallback_score, was_finalized=True,
            force_cut=False, close_reason=CloseReason.FINALIZED,
            growth_observed=False,
            finalized_source=FINALIZED_SOURCE_SIMULATE_FALLBACK,
        ))
        self._finalized_count += 1
        return True

    # ------------------------------
    # 取得
    # ------------------------------

    def resolved(self) -> list[ResolvedChain]:
        """これまでに解決 (CLOSED) した連鎖の一覧。"""
        return list(self._resolved)

    def stats(self) -> ResolverStats:
        """診断カウンタのスナップショット。"""
        return ResolverStats(
            opened_count=self._opened_count,
            finalized_count=self._finalized_count,
            unfinalized_close_count=self._unfinalized_close_count,
            force_cut_count=self._force_cut_count,
            count_decrease_split_count=self._count_decrease_split_count,
            orphan_end_signal_count=self._orphan_end_signal_count,
            chain_settled_received_count=self._chain_settled_received_count,
            settled_echo_absorbed_count=self._settled_echo_absorbed_count,
            continuation_reopen_count=self._continuation_reopen_count,
            continuation_base_underflow_count=self._continuation_base_underflow_count,
        )


def resolve_chain_ids(observations: Iterable[ChainObservation]) -> list[ResolvedChain]:
    """観測列から chain_id を解決する純関数のショートカット。

    入力順にソートを仮定せず、t_sec 昇順の安定ソートをしてから処理する。
    入力を書き換えない。ストリーム終端まで処理し、残った in-flight は
    未確定のまま閉じる (`flush`)。
    """
    resolver = ChainIdResolver()
    for obs in sorted(observations, key=lambda o: o.t_sec):
        resolver.push(obs)
    resolver.flush()
    return resolver.resolved()


__all__ = [
    "CHAIN_ID_MAX_SEC",
    "FINALIZED_SOURCE_SCORE_OCR_DIFF",
    "FINALIZED_SOURCE_SIMULATE_FALLBACK",
    "SETTLED_ECHO_MAX_SEC",
    "ChainIdResolver",
    "ChainObservation",
    "CloseReason",
    "ObservationKind",
    "ResolvedChain",
    "ResolverStats",
    "resolve_chain_ids",
]
