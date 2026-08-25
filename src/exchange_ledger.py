"""交換エピソード会計の純粋コア (2026-08-24、Gate 2)。

仕様: `docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md`

## なぜ作るのか

user 指摘「99% の勝率から 1% に急降下する」の根因は、
**一つの撃ち合いを一つの出来事として数えていない**ことにある。

seg01 game2 の実測 (`logs/_diag_zenchi_seg01_pm100_trace_2026-08-24.log`):

    t=176.30  1P が 525 個ぶん生成
    t=186.67  2P が 720 個で撃ち返す。同時に 1P の 525 が計算から消える
              → kpend1 = 720 + 79 - 30 = 769 (1P が送った 525 がどこにもない)
    t=197.53  会計がようやく追いつく (完走から 11.5 秒遅れ)。しかも cap 216 で丸め
    t=211.43  断片化で生成量が 442 → 101。画面では何も起きていないのに符号反転

**往復は一つの出来事**であり、往復が終わるまで残量は確定しない。
本モジュールはその「往復一つ」を `ExchangeEpisode` として定義し、
生成・相殺・送付・着弾が足し引きして必ず合う会計を与える。

## 責務の境界

- **純粋処理**。I/O なし、グローバル状態なし、乱数なし、時計を読まない
  (時刻は必ず引数で受ける)。同じイベント列なら必ず同じ結果になる。
- 認識しない。掛け算式の読み取りは呼び出し側の責務。
- 表示しない。色や文言は持たない。
- 勝率を出さない。符号付きの純残量を出すだけ。
- 既存クラスを書き換えない。`ResolvedExchangeTracker` /
  `OjamaAccountingTracker` へは Gate 3 で wrapper 経由で接続する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

# ============================
# 定数 (すべて物理量から導出。シーンからの逆算はしない)
# ============================

# episode を強制終了するまでの上限秒数。
# 根拠: 実測で最も長い撃ち合い (seg01 game2) が t=176.30〜222.00 の約 46 秒。
# それを包含する丸めた値。ここに達したら会計の取りこぼしを疑うべきなので、
# 発火したら必ずカウンタに出す (黙って切らない)。
EPISODE_MAX_SEC: float = 60.0

# 送りおじゃまがこれ以下の発火は「整地」とみなし episode を開かない。
# user 伝授 (reference_saisoku_exchange_model_2026-07-22) の 4 個。
# 「おじゃま 3 個は無害」(reference_ojama_damage_nonlinear_2026-07-29) とも整合。
# 実装側 (scripts/measure_exchange_dynamics.py:89 SEICHI_OJAMA_MAX_COUNT) と同値。
SEICHI_OJAMA_MAX_COUNT: float = 4.0

# 1 ターンに降るおじゃまの上限。src/scoring.py の OJAMA_MAX_DROP_PER_TURN と同値。
# ここで再定義せず import したいが、純粋コアの依存を増やさないため定数で持ち、
# テストで src.scoring と一致することを固定する。
OJAMA_MAX_DROP_PER_TURN: int = 30

# I16: FINALIZE の値供給源として既定で唯一許可する文字列 (2026-08-25 追加)。
# `src/chain_id_resolver.py` の `FINALIZED_SOURCE_SCORE_OCR_DIFF` と
# 文字列として一致していなければならない (`test_finalize_source_constant_matches_chain_id_resolver`
# で固定)。ここで import せず定数で重複させるのは、純粋コアの依存を
# 増やさない方針 (`OJAMA_MAX_DROP_PER_TURN` と同じ理由)。
FINALIZE_SOURCE_SCORE_OCR_DIFF: str = "score_ocr_diff"


class Side(Enum):
    """どちらのプレイヤーか。"""

    P1 = auto()
    P2 = auto()

    @property
    def other(self) -> "Side":
        return Side.P2 if self is Side.P1 else Side.P1

    @property
    def sign(self) -> int:
        """1P 視点の符号 (§3)。P1 の生成は +、P2 の生成は −。"""
        return 1 if self is Side.P1 else -1


class EventKind(Enum):
    """観測事実の種別 (§7.1)。"""

    FIRE = auto()          # 連鎖の開始 (chain_id を開く)
    STEP = auto()          # 掛け算式 1 段の確定 (同一 chain_id へ量を積む)
    FINALIZE = auto()      # 確定スコアによる置換 (§4.1。加算ではない)
    CANCEL = auto()        # 相殺で消滅
    LAND = auto()          # 盤面へ着弾
    TSUMO_PLACED = auto()  # 受け側がツモを 1 手置いた (chain_id を持たない)


class ChainState(Enum):
    """chain_id 単位の状態 (§4)。イベントではなく chain_id が状態を持つ。"""

    PROVISIONAL = auto()  # 発火は観測したが確定スコアがまだ来ていない
    FINALIZED = auto()    # 確定スコア差分で生成量が確定した
    RECONCILED = auto()   # 相手の生成量と突き合わせて相殺処理を終えた
    LANDED = auto()       # 実際に盤面へ降った
    CANCELED = auto()     # 相殺で全量消滅し、降らなかった


class EpisodeStatus(Enum):
    """episode の状態 (§2.4 / §2.5)。"""

    OPEN = auto()
    CLOSED = auto()         # 保存則を満たして正常に閉じた
    CLOSED_FORCED = auto()  # 安全弁で強制的に閉じた (unreconciled が残りうる)


class EpisodeStage(Enum):
    """episode の進行段階 (§13.5.2、Codex 指標監査の要求)。

    **その時点で確定している観測だけで決める。** 連鎖終了後にしか分からない
    消費比率のような量は使わない (§9.4.1 の lazy open と同じ因果的な基準)。
    """

    HARASS_RESPONSE = auto()      # 催促に対応している最中
    OPPONENT_MAIN_FIRED = auto()  # 相手の本線が発火した
    OWN_MAIN_HELD = auto()        # 自分の本線を溜めている
    OWN_MAIN_FIRED = auto()       # 自分の本線を撃った
    SETTLING = auto()             # 相殺・着弾の処理中


@dataclass(frozen=True)
class ExchangeEvent:
    """交換における観測事実 1 件 (§7.1)。

    **事実の記録のみ。計算も判断もしない。**
    frozen なので可変の状態は持たない (状態は台帳が chain_id 単位で持つ)。
    """

    kind: EventKind
    side: Side
    t_sec: float
    amount: float = 0.0
    chain_id: int | None = None
    chain_count: int = 0
    score_delta: int = 0
    source: str = ""
    # 同一時刻・同一 chain の正当な複数イベント (1 ターン上限による着弾分割等) を
    # 区別するためだけの連番 (2026-08-24 追加)。**時刻を偽装して重複排除 (I4) を
    # 回避しないための機構。** 時刻はあくまで観測値のまま変えず、正当な複数件を
    # 区別する専用のキーをここに用意する。既定 0 は既存呼び出し (未指定) の
    # 挙動を変えないため (後方互換)。
    seq: int = 0

    @property
    def dedup_key(self) -> tuple:
        """重複排除の主キー (I4)。"""
        return (self.kind, self.side, self.chain_id, self.t_sec, self.seq)


@dataclass(frozen=True)
class PhysicalContext:
    """episode の参加・終了・早期解除の判定に使う物理状態 (§2.3 / §7.4)。

    台帳はここに書かれた値だけを見る。盤面そのものは持たない
    (量しか扱わない設計。盤面の受け渡しは台帳の外)。
    """

    # どちらかが連鎖中か (CHAIN / GRAVITY_SETTLE)
    p1_chaining: bool = False
    p2_chaining: bool = False
    # 受け側の予告表示 (cap 前)。残量の裏付け確認に使う (§2.3.1)
    p1_pending_uncapped: float = 0.0
    p2_pending_uncapped: float = 0.0
    # 盤面の空き (セル数)。早期解除の判定に使う (§7.4)
    p1_room: int = 78
    p2_room: int = 78
    # 実死亡が STABLE 確定盤面で成立したか (§2.5)
    p1_dead: bool = False
    p2_dead: bool = False
    # 試合の識別子。跨いだら episode を強制終了する (I11)
    game_idx: int = 0


@dataclass
class ChainRecord:
    """chain_id 1 本ぶんの会計状態 (§4)。

    台帳の内部表現。生成量は `STEP` の合計、または `FINALIZE` による置換で決まる。
    **確定時に加算しない。置換する** (§4.1)。
    """

    chain_id: int
    side: Side
    opened_at_sec: float
    state: ChainState = ChainState.PROVISIONAL
    # 暫定生成量 (STEP の合計)。掛け算式の段の積み上げ。
    provisional_amount: float = 0.0
    # 確定生成量 (FINALIZE で置換)。None なら未確定。
    finalized_amount: float | None = None
    chain_count: int = 0
    canceled: float = 0.0
    landed: float = 0.0
    # 下げ置換の検算ゲート (§4.1.1) で保留した差分
    held_divergence: float = 0.0

    @property
    def amount(self) -> float:
        """現在の生成量。確定していればそれ、していなければ暫定値。"""
        return (
            self.finalized_amount if self.finalized_amount is not None
            else self.provisional_amount
        )

    @property
    def outstanding(self) -> float:
        """まだ相殺も着弾もしていない量。

        **下限 0 でクリップする** (下流の計算が負にならない保証は必要)。
        **超過分はここに現れない。** 相殺・着弾が生成量を超えて供給された分は
        `oversettled` を見なければ気づけない (2026-08-24 追加、fable アーキ裁定)。
        """
        return max(0.0, self.amount - self.canceled - self.landed)

    @property
    def oversettled(self) -> float:
        """相殺・着弾が生成量を超えて供給された量 (2026-08-24 追加)。

        **0 でなければ供給側のバグ。** `outstanding` の下限クリップ
        (`max(0.0, ...)`) で黙って消える超過分をここで可視化する。
        典型例: `src/ojama_accounting.py` から相殺を配線する際、断片化した
        イベントで同じ相殺を二度供給すると、`outstanding` は 0 のまま
        E2 (`_all_settled()`) を満たし、保存則の検査を素通りして
        「きれいに帳簿が合った」ように見えてしまう
        (`feedback_wiring_gap_vs_wiring_error_2026-08-22` の「間違い」型配線事故)。
        """
        return max(0.0, self.canceled + self.landed - self.amount)


@dataclass
class LedgerSnapshot:
    """timeline dump / 学習列へそのまま落とせる粒度の観測値一式 (§13.5.6)。

    **表示用ではなく「学習列に落とせる粒度」で定義している。**
    Gate 3 以降で B (局面文脈付きモデル) と D (ExchangeEpisode 専門モデル) の
    入力になることを前提にしている。
    """

    episode_id: int | None
    stage: EpisodeStage | None
    status: EpisodeStatus | None
    # 1P 視点の符号付き純残量 (cap 前)。§3 の規約。
    net_raw: float
    # 表示用に cap を適用した純残量
    net_display: float
    # 生成・相殺・着弾・未照合の累計 (保存則の検査に使う。D1)
    # **注意 (2026-08-25 追記): `total_generated`/`total_canceled`/
    # `total_landed` は台帳生存時間全体の累計だが、試合境界
    # (`_force_close("match_boundary")`) のたびに対象の `ChainRecord` が
    # `self._chains` から退役 (クリア) されるため、試合境界を跨ぐと
    # 0 相当にリセットされる。**「動画全体の合計」としては読めない。**
    # 動画全体で見たい場合は `closed_episodes()` (`ClosedEpisodeSummary`
    # の一覧) を合算すること。
    total_generated: float
    total_canceled: float
    total_landed: float
    unreconciled: float
    # 未確定のまま残っている暫定量
    provisional_residual: float
    # 撃ち合いが未解決か
    is_unresolved: bool
    # ±100 の完全上書きを許してよいか (§7.4 の決定不変性)
    allows_hard_override: bool
    # 強制終了カウンタ 3 種 (D3。黙って切らないための実体)
    forced_close_count: int = 0
    chain_id_force_cut_count: int = 0
    unbacked_residual_count: int = 0
    # finalize 乖離 (確定 − 暫定) と下げ置換ゲートの発動有無 (D2)
    finalize_divergence: float = 0.0
    finalize_gate_held: bool = False
    # 相殺・着弾が生成量を超えて供給された量の合計 (2026-08-24 追加)。
    # 0 でなければ供給側のバグ (`ChainRecord.oversettled` の docstring参照)。
    oversettled_total: float = 0.0
    # 試合境界で退役させた chain の件数と、退役前に残っていた未決着量の合計
    # (2026-08-25 追加。`_retire_all_chains_at_match_boundary` 参照)。
    # **`unreconciled` とは重複しない (足して良い)。** 直前に force_close
    # した episode 自身の chain は `unreconciled` へ既に計上済みなので
    # ここには含めない。episode に属さない chain (整地等) の残量だけが
    # 対象 (2026-08-25 是正)。
    retired_chain_count: int = 0
    retired_unreconciled: float = 0.0
    # 大域で一度でも episode の要約に計上した chain_id を二度と計上しない
    # ための安全網が実際に働いた件数と量 (2026-08-25 追加)。正常系では
    # 0 が期待値。0 でなければ「試合境界の退役」以外の経路 (例: `max_sec`
    # 強制終了後に残った chain が別 episode の lazy open に紛れ込む) で
    # 二重計上が起きていたことを意味する。
    # **生成・相殺・着弾をまとめて (chain 丸ごと) 抑制する** — 生成だけ
    # 抑制すると「生成 < 相殺+着弾」という偽の保存則違反/超過決済を
    # 生むため (`_split_chains_by_global_dedup` 参照、2026-08-25 是正)。
    duplicate_generated_suppressed_count: int = 0
    duplicate_generated_suppressed_amount: float = 0.0
    # I16: FINALIZE の値供給源が限定に反したため拒否した件数・量
    # (2026-08-25 追加)。既定 (`allow_simulate_fallback=False`) では
    # `source != FINALIZE_SOURCE_SCORE_OCR_DIFF` の FINALIZE を黙って
    # 受け取らず、ここに記録して捨てる (例外は投げない)。
    finalize_rejected_count: int = 0
    finalize_rejected_amount: float = 0.0
    # 【実装2、2026-08-25 追加】退役 (`retire_side_chains`/
    # `_retire_all_chains_at_match_boundary`) で `self._chains` から
    # 削除される chain の相殺・着弾・生成量の退避合計。
    # `_chains_for_episode` は `if cid in self._chains` で絞り込むため、
    # chain が削除された瞬間にその chain の相殺・着弾・生成量ごと
    # episode の集計から弾かれる (2026-08-25 実測: v51 chain6 の相殺
    # 18 個がこの経路で `total_canceled` から消えていた、327 -> 309)。
    # **黙って落とさない。** 大域重複排除 (`_counted_chain_ids`) 済みの
    # chain は除外する (二重計上防止、`_record_chain_retirement_totals`
    # 参照)。
    retired_canceled: float = 0.0
    retired_landed: float = 0.0
    retired_generated: float = 0.0
    # 【コーディネーター指摘、2026-08-25 追加】close 済み (`_counted_chain_ids`
    # に大域計上済み) chain へ、close 後に届いた相殺・着弾の件数・量。
    # **根治は未着手。この値が大きければ close の判定が早すぎる証拠。**
    # CLOSED_FORCED の chain は outstanding>0 のまま `self._chains` に残る
    # ため、close 後も FIFO 帰属で相殺・着弾を受け取り続けられるが、その
    # episode の要約 (`ClosedEpisodeSummary`) は close 時点のスナップ
    # ショットなので、以後増えた分はどの合計にも現れない
    # (`_maybe_count_post_close_settlement` 参照)。
    post_close_settlement_dropped_count: int = 0
    post_close_settlement_dropped_amount: float = 0.0


# ============================
# 台帳本体
# ============================

# 下げ置換を許容する差の上限 (§4.1.1)。
# 根拠: 確定スコア差分は落下ボーナス (≤ 約 250 点) と全消しボーナス
# (2100 点 = おじゃま 30 個相当) を含むが、掛け算式の素点和は純連鎖得点のみ。
# よって「確定 < 暫定」は本来起きにくく、起きても落下ボーナス相当の小差のはず。
# その幅をおじゃま個数へ換算して 250/70 ≈ 3.6 個、丸めて 4 個。
# シーンからの逆算ではなく、得点式に含まれる差分から導出している。
FINALIZE_DOWNWARD_TOLERANCE: float = 4.0

# 表示用 pending の上限。src/ojama_accounting.py の PENDING_ABS_CAP と同値。
# **判定 (net_raw) には使わない。** cap 後の値どうしを引き算すると
# 架空の攻撃が生まれる (project_pm100_display_flip_2026-08-24 根因②)。
PENDING_ABS_CAP: float = 216.0


@dataclass
class Episode:
    """一つの撃ち合い。参加したイベントと進行段階を持つ。"""

    episode_id: int
    opened_at_sec: float
    closed_at_sec: float | None = None
    status: EpisodeStatus = EpisodeStatus.OPEN
    stage: EpisodeStage = EpisodeStage.HARASS_RESPONSE
    close_reason: str = ""
    events: list[ExchangeEvent] = field(default_factory=list)

    def touch(self, ev: ExchangeEvent) -> None:
        """イベントを受け入れ、進行段階を更新する (§13.5.2)。

        **その時点で確定している観測だけで決める。** 連鎖終了後にしか
        分からない消費比率のような量は使わない。
        """
        self.events.append(ev)
        if ev.kind is not EventKind.FIRE:
            return
        sides = {e.side for e in self.events if e.kind is EventKind.FIRE}
        if len(sides) >= 2:
            self.stage = EpisodeStage.OWN_MAIN_FIRED
        elif ev.side is Side.P2:
            self.stage = EpisodeStage.OPPONENT_MAIN_FIRED


@dataclass(frozen=True)
class ClosedEpisodeSummary:
    """CLOSED (または CLOSED_FORCED) した episode 1 件の要約 (2026-08-24 Gate 3-2a 追加)。

    `ExchangeLedger` は episode を閉じた瞬間に内部の `Episode` を破棄する
    (`self._episode = None`) ため、外部から episode 単位の値 (D1: 保存則の
    episode 単位機械検査に必須) を取得する手段が無かった
    (`src/exchange_episode_tracker.py` の Gate 3-2a 報告で判明)。

    **既存 API は削除・変更していない (追加のみ)。**
    `unreconciled` は正常 CLOSED (E2、2026-08-24 に E1 は削除済み) では
    常に 0 (§8 I7)。CLOSED_FORCED のときだけ、そのepisode に属する chain
    の未解決残量 (outstanding) の合計を積む。もし正常 CLOSED なのに
    outstanding が残っていれば、それは `total_generated` と
    `total_canceled + total_landed + unreconciled(=0)` の不一致として
    保存則違反が可視化される (§8 I1)。

    `has_settlement_input` (2026-08-24 追加): この episode に `CANCEL` /
    `LAND` イベントが 1 件でも供給されたか。相殺・着弾の供給経路
    (`src/ojama_accounting.py`) が Gate 3-2b で配線されるまで、
    `FIRE`/`FINALIZE` しか無い episode は構造的に保存則を満たせない
    (生成だけが積まれ、相殺/着弾が 0 のまま残るため)。呼び出し側が
    「まだ配線していないだけ」と「本当に会計が壊れている」を区別する
    ために必須のフラグ。

    `oversettled` (2026-08-24 追加): この episode に属する chain の
    超過決済 (`ChainRecord.oversettled`) の合計。0 でなければ供給側の
    バグ (相殺・着弾の二重供給等)。`outstanding` のクリップにより
    E2 (`_all_settled()`) は正常に成立してしまうため、保存則の検査
    (`total_generated` と `total_canceled + total_landed + unreconciled`
    の一致) だけでは検出できない。

    `oversettled_chain_count` (2026-08-24 追加): 上記の超過決済が
    発生した chain_id の件数 (合計額だけでなく、何本の chain で
    起きたかを黙って捨てないため)。
    """

    episode_id: int
    status: EpisodeStatus
    close_reason: str
    opened_at_sec: float
    closed_at_sec: float
    total_generated: float
    total_canceled: float
    total_landed: float
    unreconciled: float
    has_settlement_input: bool
    oversettled: float
    oversettled_chain_count: int


class ExchangeLedger:
    """イベント列を受けて episode を編成し、符号付き純残量を返す純粋な会計コア。

    **I/O なし、グローバル状態なし、乱数なし、時計を読まない。**
    同じイベント列を与えれば必ず同じ結果になる。
    """

    def __init__(self, *, allow_simulate_fallback: bool = False) -> None:
        """
        Args:
            allow_simulate_fallback: I16 (FINALIZE の値供給源の限定) を
                緩めるかどうか。既定 `False` では
                `source != FINALIZE_SOURCE_SCORE_OCR_DIFF` の FINALIZE を
                拒否する。低信頼フォールバック経路を明示的に許容したい
                呼び出し側だけ `True` を渡す (2026-08-25 追加、既存呼び出し
                互換のため keyword-only + 既定値あり)。
        """
        self._chains: dict[int, ChainRecord] = {}
        self._episode: Episode | None = None
        self._next_episode_id: int = 1
        self._seen: set[tuple] = set()
        self._last_t: float | None = None
        self._game_idx: int | None = None
        self._forced_close_count: int = 0
        self._chain_id_force_cut_count: int = 0
        self._unbacked_residual_count: int = 0
        self._finalize_divergence: float = 0.0
        self._finalize_gate_held: bool = False
        self._unreconciled: float = 0.0
        # CLOSED / CLOSED_FORCED した episode の要約履歴 (追加のみ、2026-08-24)。
        self._closed_episodes: list[ClosedEpisodeSummary] = []
        # 試合境界で退役させた chain の件数・残量 (2026-08-25 追加)。
        self._retired_chain_count: int = 0
        self._retired_unreconciled: float = 0.0
        # 大域で一度 total_generated に計上した chain_id (2026-08-25 追加)。
        # 試合境界をまたいでも保持する (同じ chain_id が二度と生成量として
        # 数えられないことを恒久的に保証するため)。
        self._counted_chain_ids: set[int] = set()
        self._duplicate_generated_suppressed_count: int = 0
        self._duplicate_generated_suppressed_amount: float = 0.0
        # I16: FINALIZE の値供給源ゲート (2026-08-25 追加)。
        self._allow_simulate_fallback = allow_simulate_fallback
        self._finalize_rejected_count: int = 0
        self._finalize_rejected_amount: float = 0.0
        # 【実装2、2026-08-25 追加】退役で削除される chain の相殺・着弾・
        # 生成量の退避合計 (`LedgerSnapshot` docstring参照)。
        self._retired_canceled: float = 0.0
        self._retired_landed: float = 0.0
        self._retired_generated: float = 0.0
        # 【コーディネーター指摘、2026-08-25 追加】close 済み chain へ
        # 届いた相殺・着弾 (黙って落とさず数えるだけ、根治は未着手)。
        self._post_close_settlement_dropped_count: int = 0
        self._post_close_settlement_dropped_amount: float = 0.0

    # ------------------------------
    # 受付
    # ------------------------------

    def push(self, ev: ExchangeEvent, ctx: PhysicalContext) -> None:
        """イベントを 1 件受け付ける。t_sec の非減少順で呼ぶこと。"""
        self._validate(ev)
        if ev.dedup_key in self._seen:
            return                      # I4: 重複は 1 回分
        self._seen.add(ev.dedup_key)
        self._last_t = ev.t_sec
        if self._reject_if_unauthoritative_finalize(ev):
            return                      # I16: 受け取らなかったことにする
        self._check_boundaries(ev, ctx)
        self._apply(ev)
        self._maybe_open_or_close(ev, ctx)

    def _reject_if_unauthoritative_finalize(self, ev: ExchangeEvent) -> bool:
        """I16: FINALIZE の値供給源を `FINALIZE_SOURCE_SCORE_OCR_DIFF` に限定する。

        配線の「間違い」型 (別の値が届く) は --help 突合でも採用登録の確認でも
        検出できない (`feedback_wiring_gap_vs_wiring_error_2026-08-22`)。
        黙って受け取らず、拒否して件数・量をカウンタに残す。
        **例外は投げない** (`allow_simulate_fallback=True` の低信頼
        フォールバック経路が正当に存在するため)。
        """
        if ev.kind is not EventKind.FINALIZE:
            return False
        if ev.source == FINALIZE_SOURCE_SCORE_OCR_DIFF or self._allow_simulate_fallback:
            return False
        self._finalize_rejected_count += 1
        self._finalize_rejected_amount += ev.amount
        return True

    def _validate(self, ev: ExchangeEvent) -> None:
        """量と時刻の健全性を確かめる (I10 / I12)。"""
        if self._last_t is not None and ev.t_sec < self._last_t:
            raise ValueError(
                f"t_sec が減少した: {ev.t_sec} < {self._last_t}。"
                "黙って並べ替えない (同一時刻は許可)",
            )
        if ev.amount < 0.0:
            raise ValueError(f"負の量は受け取らない: {ev.amount}")
        if ev.kind is EventKind.LAND and ev.amount > OJAMA_MAX_DROP_PER_TURN:
            raise ValueError(
                f"1 回の着弾が 1 ターン上限を超えた: {ev.amount} > "
                f"{OJAMA_MAX_DROP_PER_TURN}。着弾の一括計上バグの疑い",
            )

    def _check_boundaries(self, ev: ExchangeEvent, ctx: PhysicalContext) -> None:
        """試合境界と上限秒数で強制終了する (I11 / §2.5)。"""
        if self._game_idx is None:
            self._game_idx = ctx.game_idx
        elif ctx.game_idx != self._game_idx:
            self._force_close("match_boundary")
            self._game_idx = ctx.game_idx
        ep = self._episode
        if ep is not None and ev.t_sec - ep.opened_at_sec > EPISODE_MAX_SEC:
            self._force_close("max_sec")

    # ------------------------------
    # 反映
    # ------------------------------

    def _apply(self, ev: ExchangeEvent) -> None:
        """イベント種別ごとに chain の会計を更新する。"""
        if ev.kind is EventKind.TSUMO_PLACED or ev.chain_id is None:
            return
        rec = self._chains.get(ev.chain_id)
        if rec is None:
            rec = ChainRecord(
                chain_id=ev.chain_id, side=ev.side, opened_at_sec=ev.t_sec,
            )
            self._chains[ev.chain_id] = rec
        if ev.kind in (EventKind.FIRE, EventKind.STEP):
            rec.provisional_amount += ev.amount
            rec.chain_count = max(rec.chain_count, ev.chain_count)
        elif ev.kind is EventKind.FINALIZE:
            self._finalize(rec, ev.amount)
        elif ev.kind is EventKind.CANCEL:
            self._maybe_count_post_close_settlement(ev)
            rec.canceled += ev.amount
            rec.state = ChainState.RECONCILED
        elif ev.kind is EventKind.LAND:
            # LAND の side は「受けた側」。生成した chain は相手側にある。
            self._maybe_count_post_close_settlement(ev)
            rec.landed += ev.amount
            rec.state = ChainState.LANDED
        if self._episode is not None:
            self._episode.touch(ev)

    def _maybe_count_post_close_settlement(self, ev: ExchangeEvent) -> None:
        """close 済み chain へ届いた相殺・着弾を黙って落とさず数える
        (コーディネーター指摘、2026-08-25 追加。**根治は未着手**)。

        `_counted_chain_ids` は大域で一度でも `ClosedEpisodeSummary` に
        計上した chain_id の永続集合 (`_split_chains_by_global_dedup` 参照)。
        CLOSED_FORCED の chain は outstanding>0 のまま `self._chains` に
        残り続けるため、close 後も FIFO 帰属 (`_attribute`) で
        CANCEL/LAND を受け取り続けられる。しかしその episode の要約
        (`ClosedEpisodeSummary.total_canceled`/`total_landed`) は close
        した瞬間のスナップショットであり、以後増えた分はどの合計にも
        現れない (retired_* も、後で退役するまでは捕捉しない)。

        **この値が大きければ、close (`_should_close`/`_force_close`) の
        判定が早すぎる証拠である。** 根治 (close 済み chain には最初から
        イベントを帰属させない等) は別タスクとし、ここでは黙って落とさず
        件数・量を可視化するだけに留める。
        """
        if ev.chain_id in self._counted_chain_ids:
            self._post_close_settlement_dropped_count += 1
            self._post_close_settlement_dropped_amount += ev.amount

    def _finalize(self, rec: ChainRecord, confirmed: float) -> None:
        """確定スコアで**置換**する (§4.1)。加算しない。冪等。

        下げ置換だけは検算ゲートを通す (§4.1.1)。
        `max(予測, 確定)` の恒久維持は保存則 I1 と両立しないので採らない
        (生成量が実際より大きい値でラチェットされると、相殺 + 着弾の実測合計と
        永久に一致せず、全 episode で unreconciled が偽陽性に残る)。

        **【P2-1 是正、2026-08-25 Codex レビュー】保留差分も冪等にする。**
        旧実装は下げ保留のたびに `self._unreconciled += abs(divergence)` を
        無条件加算していたため、同じ下方確定が異なる時刻で再入力される
        (dedup_key が異なり I4 では防げない) と保留差が二重計上された
        (Codex 最小再現: provisional=500 へ confirmed=42 を 2 回 →
        正しい 458 でなく 916)。この chain が過去に保留した差分
        (`rec.held_divergence`) を先に取り消してから再計算することで、
        再入力・値更新・後続正常確定による解消のすべてを冪等に扱う。
        """
        divergence = confirmed - rec.provisional_amount
        self._finalize_divergence = divergence
        # この chain の過去の保留分を一度取り消す (無ければ 0 で何もしない)。
        self._unreconciled -= rec.held_divergence
        rec.held_divergence = 0.0
        if divergence < -FINALIZE_DOWNWARD_TOLERANCE:
            # 大きい下げ。暫定側を保持し、差を未照合として明示的に記録する。
            self._finalize_gate_held = True
            rec.held_divergence = abs(divergence)
            self._unreconciled += rec.held_divergence
            rec.state = ChainState.FINALIZED
            return
        rec.finalized_amount = confirmed
        rec.state = ChainState.FINALIZED

    # ------------------------------
    # episode の開閉
    # ------------------------------

    def _maybe_open_or_close(
        self, ev: ExchangeEvent, ctx: PhysicalContext,
    ) -> None:
        """lazy open (§9.4.1) と終了条件 (§2.4) を判定する。"""
        if self._episode is None:
            if self._largest_open_amount() > SEICHI_OJAMA_MAX_COUNT:
                self._episode = Episode(
                    episode_id=self._next_episode_id, opened_at_sec=ev.t_sec,
                )
                self._next_episode_id += 1
                for past in self._fire_events_of_open_chains():
                    self._episode.touch(past)
            return
        if self._should_close(ctx):
            self._close_current_episode(ev.t_sec, "normal_close")

    def _close_current_episode(self, t_sec: float, reason: str) -> None:
        """OPEN な episode を正常 CLOSED として閉じる共通処理 (2026-08-25 追加)。

        従来 `_maybe_open_or_close` にインライン化されていた処理をそのまま
        切り出しただけ (挙動は変えていない、`close_reason` を引数化しただけ)。
        `retire_side_chains` (Fix【2】) がワイプ検知直後に episode を閉じる
        ためにも同じ処理を再利用する。
        """
        ep = self._episode
        if ep is None:
            return
        ep.status = EpisodeStatus.CLOSED
        ep.closed_at_sec = t_sec
        self._closed_episodes.append(self._summarize_episode(ep, reason))
        self._episode = None

    def _fire_events_of_open_chains(self) -> list[ExchangeEvent]:
        """lazy open 時に、既に開いている chain の発火を episode へ引き継ぐ。

        **報告 (2026-08-25、Fix【4】実装時の発見)**: ここが outstanding>0 の
        **全 chain** (side/整地量を問わず) を無条件で引き継ぐため、
        `_should_close` の episode 限定化 (`_all_settled`/
        `_provisional_residual`) の実効果を弱める可能性がある。
        `Episode.touch` も OPEN な episode がある間に来た全イベントを
        無条件で touch するため、outstanding>0 の chain は通常の push
        経路では必ずどこかの episode の events に巻き込まれる。この
        巻き込みの是非は本メソッドの変更範囲外として報告に委ねた。
        """
        return [
            ExchangeEvent(
                kind=EventKind.FIRE, side=r.side, t_sec=r.opened_at_sec,
                chain_id=r.chain_id, source="lazy_open",
            )
            for r in self._chains.values() if r.outstanding > 0.0
        ]

    def _largest_open_amount(self) -> float:
        """未決着の chain のうち最大の生成量 (lazy open の判定に使う)。

        **合計ではなく最大**を見る。整地が何本あっても撃ち合いにはならない。
        """
        vals = [r.amount for r in self._chains.values() if r.outstanding > 0.0]
        return max(vals) if vals else 0.0

    def _should_close(self, ctx: PhysicalContext) -> bool:
        """§2.4 の E2 で閉じる。

        **【訂正 — E1 の削除 (2026-08-24)】** 初版は「`net_raw() == 0`」を
        終了条件 E1 として `_all_settled()` (E2) と or で結んでいたが削除した。
        `net_raw() == 0` は相殺が起きる**運命**にあることを示すだけで、
        相殺という取引の**記録ではない**。相殺は
        `cancel_own_pending_then_send_surplus` (`src/ojama_accounting.py:742`)
        が攻撃確定の瞬間に計算する実イベント (`CANCEL` イベント) であり、
        正しく配線された系では CANCEL 供給後に E2 が必ず成立するため、
        E1 が E2 と異なる結論を出すのは「相殺が観測されていない」異常時のみ。
        その場合に閉じるのは「撃ち合いが会計に載らないまま決着扱いになる」
        ことであり、これは ±100 張り付き・47 試合の符号反転の根因
        (`project_pm100_display_flip_2026-08-24`) そのものだったので許さない。
        相殺・着弾の供給が来ないまま `EPISODE_MAX_SEC` に達した episode は
        `CLOSED_FORCED` の安全弁で処理する (残量は unreconciled として明示記録)。
        **「net が 0 なら閉じてよいのでは」と戻さないこと。**
        """
        if ctx.p1_chaining or ctx.p2_chaining:
            return False
        ep = self._episode
        if ep is None:
            return False  # 呼び出し元は self._episode is not None のときだけ呼ぶ
        if self._provisional_residual(ep) > 1e-9:
            return False
        return self._all_settled(ep)

    def _all_settled(self, ep: Episode) -> bool:
        """episode に実際に参加した chain だけを見る (2026-08-25 是正)。

        **修正前は台帳全体 `self._chains` を無条件スキャンしていた。**
        `_summarize_episode`/`_force_close` は既に `_chains_for_episode(ep)`
        に限定済みで、この 2 つ (`_all_settled`/`_provisional_residual`) だけ
        流儀が違っていた (fable アーキ裁定「1 件の未確定が全体を道連れに
        する」設計は 8/24 確定の凍結盤面デッドロックと構造的に同型、
        `project_stable_freeze_deadlock_2026-08-24`)。episode が閉じるのは
        「その episode の撃ち合いが決着したとき」であり、台帳のどこかに
        無関係な未決着連鎖 (整地・前 episode の残骸等) が残っていることとは
        無関係。
        """
        return all(r.outstanding <= 1e-9 for r in self._chains_for_episode(ep))

    def _force_close(self, reason: str) -> None:
        """安全弁で強制終了する。残量は unreconciled として明示的に記録する。

        **黙って捨てない。** 件数もカウンタに出す。

        unreconciled の集計は `_summarize_episode` と同じく、その episode に
        実際に参加した chain (`_chains_for_episode`) に限定する。**以前は
        台帳全体 `self._chains` をそのまま合計しており**、他の episode に
        属する chain の残量まで混ざっていた (2026-08-25 是正)。

        `reason == "match_boundary"` のときは、これに加えて台帳全体の
        chain 記録を退役させる (`_retire_all_chains_at_match_boundary`)。
        試合が終われば、その試合の未決着量は次の試合には持ち越されない
        (I11「episode は試合を跨がない」を連鎖記録にも適用する)。これを
        しないと、前の試合の chain が次の試合の lazy open
        (`_fire_events_of_open_chains`) に紛れ込み、生成量が試合を跨いで
        再計上され続ける (2026-08-25 実測: 全体の 74.5% が二重計上)。

        **`already_recorded_chain_ids`**: この呼び出しで `self._unreconciled`
        へ既に計上した chain_id の集合を退役処理へ渡す。`unreconciled` と
        `retired_unreconciled` が同じ残量を二重に表現しないため
        (2026-08-25 是正、コーディネーター指摘)。
        """
        already_recorded_chain_ids: set[int] = set()
        if self._episode is not None:
            chains = self._chains_for_episode(self._episode)
            already_recorded_chain_ids = {r.chain_id for r in chains}
            self._unreconciled += sum(r.outstanding for r in chains)
            self._episode.status = EpisodeStatus.CLOSED_FORCED
            self._episode.close_reason = reason
            self._closed_episodes.append(
                self._summarize_episode(self._episode, reason),
            )
            self._episode = None
            self._forced_close_count += 1
        if reason == "match_boundary":
            self._retire_all_chains_at_match_boundary(already_recorded_chain_ids)

    def _record_chain_retirement_totals(self, rec: ChainRecord) -> None:
        """退役させる chain の相殺・着弾・生成量を退避カウンタへ転記する
        (実装2、2026-08-25 追加)。`self._chains` からの削除前に必ず呼ぶこと。

        `_chains_for_episode` の `if cid in self._chains` ガードは、chain が
        削除された瞬間にその chain の相殺・着弾・生成量ごと episode の
        集計から弾いてしまう (`LedgerSnapshot.retired_canceled` docstring
        参照)。呼び出し側は「大域で一度も集計されていない chain だけを
        渡す」ことを保証すること (`_counted_chain_ids` によるガード。
        既に集計済みの chain をここに渡すと二重計上になる)。
        """
        self._retired_canceled += rec.canceled
        self._retired_landed += rec.landed
        self._retired_generated += rec.amount

    def _retire_all_chains_at_match_boundary(
        self, already_recorded_chain_ids: set[int],
    ) -> None:
        """試合境界で、台帳に残る全 chain 記録を退役させる (2026-08-25 追加)。

        試合が終われば、その試合の未決着量は次の試合には持ち越されない
        (相殺も着弾も、ゲームのルール上ありえない)。退役前の残量は
        黙って捨てず `retired_unreconciled` に積み、件数を
        `retired_chain_count` に積んでから `self._chains` を空にする。

        **`already_recorded_chain_ids` は除外する。** `_force_close` が
        同じ呼び出しの中で、直前に閉じた episode 自身の chain として
        既に `self._unreconciled` へ計上済みだからである。ここでも数えると
        `unreconciled` と `retired_unreconciled` が同じ残量を二重に表現して
        しまい、2つを合算すると水増しになる (2026-08-25 是正)。
        `retired_unreconciled` の対象は、episode に属さない chain
        (整地等で episode を一度も開かなかったもの) の残量だけになる。
        `self._chains` は episode をまたいで永続する台帳全体であり、
        試合が変われば全件が対象になる。

        **【実装2、2026-08-25 追加】** 相殺・着弾・生成量の退避
        (`retired_canceled`/`retired_landed`/`retired_generated`) は
        `outstanding>0.0` の制限を掛けない (整地チェインは outstanding=0 の
        まま一度も episode に集計されずに消えるため、outstanding だけを
        条件にすると見落とす)。代わりに `_counted_chain_ids` (大域で一度
        でも episode の要約に計上した chain_id の永続集合) で絞り込む。
        `already_recorded_chain_ids` はこの呼び出し内で `_summarize_episode`
        が今しがた `_counted_chain_ids` へ追加した部分集合なので、
        `_counted_chain_ids` によるガードは `already_recorded_chain_ids` を
        自動的に含む (二重計上の心配は無い)。
        """
        residual = [
            r for r in self._chains.values()
            if r.outstanding > 0.0 and r.chain_id not in already_recorded_chain_ids
        ]
        self._retired_unreconciled += sum(r.outstanding for r in residual)
        self._retired_chain_count += len(residual)
        for r in self._chains.values():
            if r.chain_id not in self._counted_chain_ids:
                self._record_chain_retirement_totals(r)
        self._chains.clear()

    def retire_side_chains(
        self, side: Side, t_sec: float, ctx: PhysicalContext, reason: str = "side_wipe",
    ) -> None:
        """ワイプ (ラウンド境界での片側の予告消滅) を検知したら呼ぶ (Fix【2】、追加のみ)。

        `src/ojama_accounting.py:_reset_side_boundary` は負けた側の
        `forecast_incoming`/`forecast_incoming_uncapped` を無音でゼロクリアする。
        物理的には正しい (負けた側は受け取らずに終わる、
        `reference_ojama_landing_gated_by_placement_2026-07-29`) が、観測経路
        (台帳) はこれを見る手段を持っていなかった。

        **なぜ side 単位か**: ワイプは片側 (負けた側) で起きる。試合境界
        (両側、`_force_close("match_boundary")`) とは別の事象であり、
        両側を退役させる既存経路を流用すると、片側だけ死んだ場面で
        **相手の正当な未決着まで消してしまう**。

        `side` は「予告を受け取れなくなった側」(負けた側、
        `forecast_incoming_uncapped` が 0 になった側自身)。退役対象は
        「その side に向いていた」= **相手側 (`side.other`) の chain が
        生成した、まだ outstanding が残っている量** (`ChainRecord.side`
        は生成した側を指すため)。

        退役後、この呼び出しの中で `_should_close(ctx)` を試す
        (`_close_current_episode` 経由)。試合境界の遅い統合 `game_idx` を
        待たずに、ワイプそのものを episode 決着のトリガーにできるようにする
        (episode の試合跨ぎ禁止 I11 自体は統合 `game_idx` のまま変えない。
        ここで閉じるのは「決着した」という判定だけで、`game_idx` によるチェック
        `_check_boundaries` を置き換えるものではない)。
        """
        residual = [
            r for r in self._chains.values()
            if r.side is side.other and r.outstanding > 0.0
        ]
        if residual:
            self._retired_unreconciled += sum(r.outstanding for r in residual)
            self._retired_chain_count += len(residual)
            # 【実装2】削除は summarize より**前**に起こるため、削除前に
            # 必ず退避する (`_record_chain_retirement_totals` docstring参照)。
            for r in residual:
                if r.chain_id not in self._counted_chain_ids:
                    self._record_chain_retirement_totals(r)
                del self._chains[r.chain_id]
        if self._episode is not None and self._should_close(ctx):
            self._close_current_episode(t_sec, reason)

    def _chains_for_episode(self, ep: Episode) -> list[ChainRecord]:
        """episode に実際に参加した chain_id 集合に限定して ChainRecord を返す。

        台帳全体の `self._chains` をそのまま合計すると、他の episode に
        属する chain まで混ざってしまうため (`_force_close` と
        `_summarize_episode` で共通に使う)。
        """
        chain_ids = {e.chain_id for e in ep.events if e.chain_id is not None}
        return [self._chains[cid] for cid in chain_ids if cid in self._chains]

    def _split_chains_by_global_dedup(
        self, chains: list[ChainRecord],
    ) -> tuple[list[ChainRecord], list[ChainRecord]]:
        """大域で一度でも episode の要約に計上した chain_id を二度と数えない。

        **安全網。** 本来は `_retire_all_chains_at_match_boundary` により
        試合境界での二重計上は構造的に起きないはずだが、試合境界以外
        (`max_sec` 強制終了で chain が台帳に残ったまま、同じ試合内で別の
        lazy open に紛れ込む経路) でも同じ形の二重計上が起こり得るため、
        大域集合で構造的に不可能にする。

        **chain は丸ごと計上しないか、丸ごと計上するかの二択にする**
        (2026-08-25 是正、コーディネーター指摘)。生成だけを除外し
        相殺・着弾は除外しないと、その episode で「生成 < 相殺+着弾」に
        なり、偽の保存則違反・偽の超過決済を生む。`_summarize_episode` は
        ここで返す `fresh`/`duplicate` を生成・相殺・着弾のすべての
        集計に一貫して使う。
        """
        fresh = [c for c in chains if c.chain_id not in self._counted_chain_ids]
        duplicate = [c for c in chains if c.chain_id in self._counted_chain_ids]
        self._counted_chain_ids.update(c.chain_id for c in fresh)
        return fresh, duplicate

    def _summarize_episode(self, ep: Episode, reason: str) -> ClosedEpisodeSummary:
        """episode 1 件ぶんの要約を作る (D1 の episode 単位機械検査用、追加のみ)。

        大域重複排除された `fresh` (`_split_chains_by_global_dedup`) を
        生成・相殺・着弾・unreconciled・超過決済の**すべて**の集計に使う
        (chain 丸ごと計上する/しないの二択、2026-08-25 是正)。
        """
        chains = self._chains_for_episode(ep)
        fresh, duplicate = self._split_chains_by_global_dedup(chains)
        self._duplicate_generated_suppressed_count += len(duplicate)
        self._duplicate_generated_suppressed_amount += sum(c.amount for c in duplicate)
        unreconciled = (
            sum(c.outstanding for c in fresh)
            if ep.status is EpisodeStatus.CLOSED_FORCED else 0.0
        )
        has_settlement_input = any(
            e.kind in (EventKind.CANCEL, EventKind.LAND) for e in ep.events
        )
        return ClosedEpisodeSummary(
            episode_id=ep.episode_id, status=ep.status, close_reason=reason,
            opened_at_sec=ep.opened_at_sec, closed_at_sec=ep.closed_at_sec or 0.0,
            total_generated=sum(c.amount for c in fresh),
            total_canceled=sum(c.canceled for c in fresh),
            total_landed=sum(c.landed for c in fresh),
            unreconciled=unreconciled,
            has_settlement_input=has_settlement_input,
            oversettled=sum(c.oversettled for c in fresh),
            oversettled_chain_count=sum(1 for c in fresh if c.oversettled > 0.0),
        )

    def closed_episodes(self) -> list[ClosedEpisodeSummary]:
        """CLOSED / CLOSED_FORCED した episode の要約一覧 (D1 用、追加のみ)。"""
        return list(self._closed_episodes)

    def open_chain_ids(self, side: Side) -> list[int]:
        """side の未決着 (outstanding>0) chain の chain_id を、開いた順に返す
        (2026-08-24 追加、追加のみ)。

        おじゃまは先に送られたものから順に処理されるため、相殺・着弾の
        帰属は古い chain から消化する (FIFO)。chain_id は発行順に単調
        増加するため、昇順ソートがそのまま FIFO になる。
        """
        return sorted(
            cid for cid, rec in self._chains.items()
            if rec.side is side and rec.outstanding > 0.0
        )

    def open_episode_outstanding(self) -> float:
        """いま OPEN な episode に属する chain の outstanding 合計
        (実装1、2026-08-25 追加)。

        D1 の `total_unreconciled` は `closed_episodes()` だけを合算するため、
        窓の終わりにまだ OPEN な episode の残量が構造的に見えない
        (2026-08-25 実測: v51 で窓を t1=533 に切ると `total_unreconciled` が
        全項目 0 になるが、`self._chains` の生値は 1,163 のまま変化しない。
        窓を早く切れば「未照合 0」という数字が出るが、それは「決着した」
        ではなく「まだ数えていない」)。OPEN な episode が無ければ 0.0。
        """
        ep = self._episode
        if ep is None:
            return 0.0
        return sum(r.outstanding for r in self._chains_for_episode(ep))

    def total_outstanding_all_chains(self) -> float:
        """台帳に残っている全 chain (episode に属さないものも含む) の
        outstanding 合計 (実装1、2026-08-25 追加)。台帳の生値そのもの。

        `open_episode_outstanding` は OPEN な episode に属する chain だけに
        限定するが、こちらは episode の内外を問わず `self._chains` 全体を
        見る (整地チェイン等、episode を一度も開いていない残骸も含む)。
        """
        return sum(r.outstanding for r in self._chains.values())

    def outstanding_of(self, chain_id: int) -> float:
        """chain_id の現在の未決着残量 (2026-08-24 追加、追加のみ)。

        存在しない chain_id には `0.0` を返す (例外にしない)。呼び出し側
        (`ExchangeEpisodeTracker._attribute`) が帰属先の消失を検知する
        経路 (`unattributed_settlement_total`) を既に持っているため、
        ここで例外にして二重に扱う必要が無い。
        """
        rec = self._chains.get(chain_id)
        return rec.outstanding if rec is not None else 0.0

    # ------------------------------
    # 集計
    # ------------------------------

    def _net_raw(self) -> float:
        """1P 視点の符号付き純残量 (cap 前)。毎回イベント列から再計算する。"""
        return sum(r.side.sign * r.outstanding for r in self._chains.values())

    def _provisional_residual(self, ep: Episode) -> float:
        """まだ確定していない暫定量の残り (episode 単位、2026-08-25 是正)。

        `_all_settled` と同じ理由で `_chains_for_episode(ep)` に限定する。
        E2 判定 (`_should_close`) 専用。台帳全体の値は
        `_provisional_residual_all` を使う (`snapshot()` 参照)。
        """
        return sum(
            r.outstanding for r in self._chains_for_episode(ep)
            if r.finalized_amount is None and r.held_divergence == 0.0
        )

    def _provisional_residual_all(self) -> float:
        """台帳全体で見た、まだ確定していない暫定量の残り。

        `LedgerSnapshot.provisional_residual` (timeline dump / 学習列用) は
        episode の境界を持たない「動画全体でいま未確定な量」を報告する
        フィールドであり、E2 判定 (`_should_close`) の episode 限定版とは
        意味が異なるため分離した (2026-08-25、Fix【4】適用時に分離)。
        """
        return sum(
            r.outstanding for r in self._chains.values()
            if r.finalized_amount is None and r.held_divergence == 0.0
        )

    def current_episode(self) -> Episode | None:
        """いま OPEN な episode。無ければ None。"""
        return self._episode

    def open_episode_count(self) -> int:
        """OPEN な episode の数。§2.4.1 により高々 1。"""
        return 1 if self._episode is not None else 0

    def net_raw(self) -> float:
        return self._net_raw()

    def net_display(self, cap: float = PENDING_ABS_CAP) -> float:
        """表示用に cap を適用した純残量。**判定には使わない。**"""
        return max(-cap, min(cap, self._net_raw()))

    def is_unresolved(self) -> bool:
        """撃ち合いが未解決か。"""
        return self._episode is not None

    def allows_hard_override(self, ctx: PhysicalContext) -> bool:
        """±100 の完全上書きを許してよいか (§7.4 の決定不変性)。

        **未確定量を受け側にもっとも有利な側へ倒して解決してもなお死ぬ**場合だけ
        許す。判定材料は決定的な物理量のみで、確率推定 (W15: 応手確率は実際に
        成功した応手を 25〜40% としか見積もれない) は使わない。
        これにより「未解決中は断定しない」と「真の致死を弱めない」が
        構造的に両立する。
        """
        if ctx.p1_dead or ctx.p2_dead:
            return True          # 実死亡は物理的に確定している
        if not self.is_unresolved():
            return True          # 撃ち合いが無ければ従来どおり
        return self._dies_even_in_best_case(ctx)

    def _dies_even_in_best_case(self, ctx: PhysicalContext) -> bool:
        """未確定量を受け側に最も有利に倒しても受け側が死ぬか。

        受け側に有利 = 未確定の生成はすべて無かったことにし、
        受け側が撃ち返せる可能性 (進行中の連鎖) は最大限効いたとみなす。
        """
        net = self._net_certain()
        if abs(net) < 1e-9:
            return False
        victim_room = ctx.p2_room if net > 0 else ctx.p1_room
        victim_chaining = ctx.p2_chaining if net > 0 else ctx.p1_chaining
        if victim_chaining:
            return False         # 撃ち返しの結果が未確定
        return abs(net) > victim_room

    def _net_certain(self) -> float:
        """確定済みの生成だけで見た純残量 (受け側に最も有利な仮定)。"""
        return sum(
            r.side.sign
            * max(0.0, (r.finalized_amount or 0.0) - r.canceled - r.landed)
            for r in self._chains.values()
        )

    def snapshot(self, ctx: PhysicalContext | None = None) -> LedgerSnapshot:
        """timeline dump / 学習列へそのまま落とせる観測値一式 (§13.5.6)。"""
        ctx = ctx or PhysicalContext()
        ep = self._episode
        return LedgerSnapshot(
            episode_id=ep.episode_id if ep else None,
            stage=ep.stage if ep else None,
            status=ep.status if ep else None,
            net_raw=self._net_raw(),
            net_display=self.net_display(),
            total_generated=sum(r.amount for r in self._chains.values()),
            total_canceled=sum(r.canceled for r in self._chains.values()),
            total_landed=sum(r.landed for r in self._chains.values()),
            unreconciled=self._unreconciled,
            provisional_residual=self._provisional_residual_all(),
            is_unresolved=self.is_unresolved(),
            allows_hard_override=self.allows_hard_override(ctx),
            forced_close_count=self._forced_close_count,
            chain_id_force_cut_count=self._chain_id_force_cut_count,
            unbacked_residual_count=self._unbacked_residual_count,
            finalize_divergence=self._finalize_divergence,
            finalize_gate_held=self._finalize_gate_held,
            oversettled_total=sum(r.oversettled for r in self._chains.values()),
            retired_chain_count=self._retired_chain_count,
            retired_unreconciled=self._retired_unreconciled,
            duplicate_generated_suppressed_count=self._duplicate_generated_suppressed_count,
            duplicate_generated_suppressed_amount=self._duplicate_generated_suppressed_amount,
            finalize_rejected_count=self._finalize_rejected_count,
            finalize_rejected_amount=self._finalize_rejected_amount,
            retired_canceled=self._retired_canceled,
            retired_landed=self._retired_landed,
            retired_generated=self._retired_generated,
            post_close_settlement_dropped_count=self._post_close_settlement_dropped_count,
            post_close_settlement_dropped_amount=self._post_close_settlement_dropped_amount,
        )


__all__ = [
    "EPISODE_MAX_SEC",
    "FINALIZE_DOWNWARD_TOLERANCE",
    "FINALIZE_SOURCE_SCORE_OCR_DIFF",
    "PENDING_ABS_CAP",
    "ClosedEpisodeSummary",
    "Episode",
    "ExchangeLedger",
    "OJAMA_MAX_DROP_PER_TURN",
    "SEICHI_OJAMA_MAX_COUNT",
    "ChainRecord",
    "ChainState",
    "EpisodeStage",
    "EpisodeStatus",
    "EventKind",
    "ExchangeEvent",
    "LedgerSnapshot",
    "PhysicalContext",
    "Side",
]
