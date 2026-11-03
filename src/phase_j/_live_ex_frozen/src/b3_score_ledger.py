"""B3 台帳 v2: 得点ベースの予告おじゃま会計 (オンライン・因果的)。

旧台帳 (`src/ojama_accounting.py` の `_finalize_chain_end`) は last_stable_score との差で
複数連鎖を 1 塊に記帳し、予告が連鎖終了後の短時間しか立たず、上限 216 で張り付いた。
本モジュールはこれを作り直し、**スコア OCR の差分** を一次の根拠にする
(memory project_ojama_inference_design: 得点差分には高速落下 1 点/マスと全消しボーナスが
すでに含まれるので加算しない)。

## 会計規則 (user 決定・伝授。推測はしない)

* 換算: 得点の増分を `acc` (点) に貯め、**連鎖の得点が入った時点**で
  `n = acc // rate`、`acc %= rate` (端数繰越)。落下 (非連鎖) の点も `acc` に入る
  (実測: 1P 累計 510 点で 2P の予告が 7 個 = 510 // 70。落下だけの 230 点を含む)。
* レート: 最初の 1 手から 95.5 秒まで 70、以降 16 秒ごと ×0.75、下限 1、最大 14 回
  (`src.scoring.compute_effective_rate(from_first_move=True)` を再利用)。
  **起点は両者のうち早い方の最初の 1 手** (side 別に持たない、reference_drop_bonus_and_margin_origin)。
* 相殺: 連鎖の各段で生じた n 個は **先に自分の受け予告を相殺**し、余りを相手の予告へ積む。
* 着弾: 相手の連鎖が確定 (連鎖終了信号) し、受け側の手が置かれ、受け側が連鎖しない場合に降る。
  受け側が連鎖した場合はその連鎖の終了後に降る。相殺で消えれば降らない。
  1 回の着弾は最大 30 個 (5 段)。**予告の個数は上限で切らない** (72 個以上は死亡確定の根拠)。
  実測で追加した規則 (video_29/30/31 の画面照合から。user 伝授の範囲内の具体化):
  (a) 受け側が連鎖を始めた後に確定した予告は、その連鎖の終了では降りない。
  (b) 置いた時点で相手が連鎖中 (式が出てから終了前) なら降らない。
  (c) 置き終わりの証拠は 置き終わり信号 / NEXT 移動 / おじゃま落下状態への遷移 のいずれか。
  (d) 証拠の時点で相手の連鎖が「終わっていそう」(最後の式から 1.0 秒以上、全段反映済み) なら
      **仮に降らせ**、待ち時間内に受け側が連鎖を始める / 相手の連鎖に次の段が来たら取り消す
      (`landing_revoked`)。仮の着弾も snapshot の pending/dropped に即反映される (画面は即時に減る)。
* 連鎖の終わり: 連鎖している側の NEXT 移動 または 連鎖している側へのおじゃま落下。
* 連鎖の分割: 掛け算式 (formula_step) の間隔が `CHAIN_STEP_GAP_MAX_SEC` を超えるか、
  最後の式から `CHAIN_END_SIGNAL_MIN_SEC` 以上たってから終了信号が出たら別の連鎖。
  連鎖ごとに段と得点増分を対応づけ、1 塊にしない。段の間 (式から 1.5 秒未満) の信号は
  ちらつき (NEXT 移動・静止・おじゃま落下) として無視する。

## 因果性

時刻 t の出力は t 以前の観測だけで決まる (prefix 不変、テストで検証)。
得点 OCR の頑健化は「直近 2 フレームの最小値が採用値を超えたら採用」
(1 フレームの読み違い・数値アニメを弾く。採用は 1 フレーム遅れ) と単調性 (試合内で
得点は下がらない。下がる読みは無視) による。採用は遅れるので、増分の発生時刻は
`t_event` (値が最初に現れたフレーム)、使える時刻は `available_at_sec` に分けて持つ。
"""
from __future__ import annotations
from src.phase_j._live_ex_frozen.src.b3_profile import timed

import gzip
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from src.phase_j._live_ex_frozen.src.scoring import (
    compute_effective_rate,
    ALL_CLEAR_BONUS,
    OJAMA_MAX_DROP_PER_TURN,
    MARGIN_TIME_DECAY_FACTOR,
    MARGIN_TIME_DECAY_INTERVAL_SEC,
    MARGIN_TIME_MAX_DECAYS,
    MARGIN_TIME_START_FROM_FIRST_MOVE_SEC,
    OJAMA_RATE_MIN,
    OJAMA_RATE_STANDARD,
)

# ============================ 定数 (出典つき) ============================

SIDE_NAMES: Tuple[str, str] = ("1P", "2P")
SIDE_COUNT: int = 2

# 落下 (高速落下 1 点/マス) 1 手の得点上限。user 伝授 2026-09-17: 1 手上限は約 26 点。
DROP_ONLY_MAX_DELTA: int = 26
# 連鎖の得点に同時に乗りうる落下点の許容 (同じ採用フレームに混ざる分)。
FORMULA_DROP_SLACK: int = DROP_ONLY_MAX_DELTA
# 試合開始直後の最初の読みがこれ以下なら「0 点から見ていた」とみなす。
# 試合再開の検出が最大数秒遅れるため、数手 (落下のみ) 分を許す。
# 実測 video_29: 開始直後の最初の読みは 27-38 点 (1 手 26 点を超える)。
FIRST_READ_PIECES_MAX: int = 4
FIRST_READ_COMPLETE_MAX: int = DROP_ONLY_MAX_DELTA * FIRST_READ_PIECES_MAX
# 得点 OCR の頑健化: 直近この秒数の窓 (2 サンプル以上) の最小値を採用 (1 フレームの読み違いを弾く)。
# 時間で持つのは入力が 30Hz でも 60Hz でも意味を変えないため (2026-10-05 第3版、フレーム数定数の秒化)。
SCORE_PERSIST_SEC: float = 0.04
# 同じ低い読みが続いたときに診断フラグを付ける秒数（採用値は下げない）。
REGRESS_RESYNC_SEC: float = 1.5
# 掛け算式の読み取りから得点反映までの実測遅れ (video_29 n=1081: 0.87-0.97 秒、max 4.97)。
# 超えても反映されない式段は「未対応」として捨てる。
FORMULA_EXPIRE_SEC: float = 3.0
# 同じ連鎖の段間隔の上限 (video_29: 段間隔は 1.0-1.5 秒に 719 件、1.5 秒以上は急減)。
CHAIN_STEP_GAP_MAX_SEC: float = 2.0
# 連鎖終了の信号を信じてよい、最後の式からの最短時間。式段の間隔は 1.0 秒未満が 0 件、
# 大半が 1.0-1.5 秒 (video_29: 719 件) で、1.5 秒未満に出る信号 (NEXT 移動・静止・おじゃま落下)
# は連鎖の段間のちらつき (信号で割ると連鎖が段の間隔 1.4 秒で 87 件に割れた)。
CHAIN_END_SIGNAL_MIN_SEC: float = 1.5
# 割り込み (相殺中に相手の新しい連鎖) の後、この秒数以内の着弾に品質フラグを立てる。
INTERRUPT_FLAG_WINDOW_SEC: float = 10.0
# 連鎖終了の安全弁 (CLAUDE.md の連鎖保持 5.0 秒と同じ)。信号が取れない場合のみ。
CHAIN_END_TIMEOUT_SEC: float = 5.0
# 連鎖終了信号: 連鎖している側が STABLE/TSUMO_FALL に戻ってこの秒数続いた (画面が静止に戻った)。
CHAIN_END_QUIET_SEC: float = 0.08
STATE_CHAIN_END_QUIET: Tuple[str, ...] = ("STABLE", "TSUMO_FALL")
STATE_OJAMA_FALL: str = "OJAMA_FALL"
# 置き終わり信号の重複を 1 回にまとめる不応期。video_29 の置き終わり間隔は 0.1-0.2 秒 (ノイズの連発)
# と 0.6-0.9 秒 (実際の手) に分かれ、谷が 0.4-0.5 秒。NEXT 移動は置き終わりの 0.2-0.4 秒後に出る。
PLACEMENT_REFRACTORY_SEC: float = 0.45
# 置き終わりから「連鎖が始まらなかった」と判断するまでの待ち
# (video_29 n=307: 置き終わり→最初の式段の中央値 0.27 秒、75% 点 0.43 秒)。
LANDING_DECISION_DELAY_SEC: float = 0.6
# 着弾判定で「相手が連鎖中だったか」を調べるため遡る、完了済み連鎖の件数。
LANDING_CHAIN_LOOKBACK: int = 8
# おじゃま落下状態への遷移を着弾の証拠にする時の不応期 (状態の揺れで二重に降らせない)。
OJAMA_EDGE_REFRACTORY_SEC: float = 1.5
# 連鎖を始めた手の置き終わりから最初の式が読めるまで (video_29 n=307: 中央値 0.27 秒)。
LOCK_TO_FIRST_FORMULA_SEC: float = 0.3
# 1 回の着弾の上限 (5 段 = 30 個、src.scoring.OJAMA_MAX_DROP_PER_TURN)。
LANDING_MAX_PER_EVENT: int = OJAMA_MAX_DROP_PER_TURN
# 両者の得点が 0 に戻ってこの秒数続いたら試合境界 (game_idx が取れない場合の保険)。
RESET_ZERO_SEC: float = 0.05
# --- 第4版 (2026-10-05): 着弾は予測せず観測する ---
MODE_OBSERVED: str = "observed"     # 発生・相殺は得点から、着弾は受け側盤面のおじゃま増加から (第4版)
MODE_TIMING: str = "timing"         # 着弾も時刻規則で予測する (第3版、比較用)
# 受け側の確定盤面 (STABLE) のおじゃま数が増えて、この秒数続いたら着弾として差し引く (認識の1フレームのぶれ除け)。
LAND_OBS_HOLD_SEC: float = 0.1
# おじゃま数が減る側 (連鎖による消去など。着弾ではない) は、この秒数続いたら基準を下げる。
LAND_OBS_DROP_HOLD_SEC: float = 0.5
# 盤面のセル数 (6 列 x 13 行) と、受け側がほぼ窒息して着弾が空きで頭打ちになるとみなす空きの上限
# (1 回の着弾の最大 = LANDING_MAX_PER_EVENT 個)。空きがこれ以下で予告が残るなら board_capacity_limited の印。
BOARD_CELLS: int = 78
BOARD_COLS: int = 6
FLAG_CAPACITY_LIMITED: str = "board_capacity_limited"
# --- 第3版 (2026-10-05): 置き終わりの裏取り・曖昧な場面・決着 ---
# 置き終わりの証拠 (信号 / tsumo_count 増分) を盤面の確定変化か NEXT 移動で裏取りする猶予 (NEXT 移動は
# 置き終わりの 0.2-0.4 秒後、video_29)。
PLACEMENT_CORROBORATE_SEC: float = 1.0
# 置き終わりと攻撃側の連鎖終了がこの秒数以内なら着弾の有無を時刻だけでは決めない (user: 画面のぷよが
# 全部接地した時が終わりで少しブレがあり、厳密計算は難しい)。
AMBIGUOUS_WINDOW_SEC: float = 0.3
# 曖昧な着弾を受け側盤面のおじゃま増加で確定するまで待つ上限。
AMBIGUOUS_EVIDENCE_SEC: float = 3.0
# 攻撃側の連鎖終了が未確定 (終了信号の後に次の段が来るかもしれない) の間、着弾判断を待つ上限。
LANDING_WAIT_MAX_SEC: float = 3.0
# 盤面のおじゃまぷよの色コード (CLAUDE.md)、窒息位置 = 3 列目 (index 2) の可視最上段 (row 1)。
OJAMA_CELL: int = 9
DEATH_ROW: int = 1
DEATH_COL: int = 2
# 窒息位置にぷよがある状態がこの秒数続いたら決着 (台帳を打ち切る。user 回答 2026-10-05)。
# 実測 (video_c122/37/c131、試合前デモ除く): 偽の連続は最大 2.0 秒 (置き中の一時的な占有)、
# 本当の決着は 7〜12 秒続いて試合終了。1.0 秒では偽の決着で試合の 4〜87 秒前に台帳が止まった。
DEAD_HOLD_SEC: float = 3.0
# 置き終わり証拠の合流 (同じ手の信号・NEXT 移動をまとめる) 不応期。新しい tsumo_count は別の手。
EVIDENCE_MERGE_SEC: float = PLACEMENT_REFRACTORY_SEC

# 品質フラグ
FLAG_HISTORY_INCOMPLETE: str = "history_incomplete"
FLAG_ORIGIN_UNKNOWN: str = "margin_origin_unknown"
FLAG_UNMATCHED_JUMP: str = "unmatched_score_jump"
FLAG_UNMATCHED_FORMULA: str = "unmatched_formula_step"
FLAG_SCORE_RESYNC: str = "score_resync_down"  # 互換名を維持。現在は減少観測の診断のみ。
FLAG_NO_FORMULA_CHAIN: str = "chain_without_formula"
FLAG_END_TIMEOUT: str = "chain_end_by_timeout"
FLAG_TIMING_AMBIGUOUS: str = "timing_ambiguous"
FLAG_MATCH_OVER: str = "match_over"
FLAG_LANDING_AFTER_INTERRUPT: str = "landing_after_interrupt"
FLAG_RESET_BY_SCORE: str = "reset_by_score_zero"
REASON_INACTIVE_GAP: str = "inactive_gap"
# 連鎖の終了理由
REASON_SIGNAL_SLIDE: str = "signal_next_slide"
REASON_SIGNAL_OJAMA: str = "signal_ojama_fall"
REASON_SIGNAL_QUIET: str = "signal_quiet"
REASON_FORMULA_GAP: str = "formula_gap"
REASON_STEP_GAP: str = "step_gap"
REASON_PLACEMENT_BETWEEN: str = "placement_between_steps"
REASON_MATCH_OVER: str = "match_over"
REASON_TIMEOUT: str = "timeout"
REASON_MATCH_RESET: str = "match_reset"

# 増分の帰属
KIND_DROP: str = "drop"
KIND_CHAIN: str = "chain"
KIND_ALL_CLEAR: str = "all_clear"
KIND_UNMATCHED: str = "unmatched_jump"


def margin_rate_table(rate_base: int = OJAMA_RATE_STANDARD) -> Tuple[int, ...]:
    """マージンタイムのレート表 (user 回答 2026-10-05: 毎回切り捨て、切り捨て後の値に ×0.75)。

    70,52,39,29,21,15,11,8,6,4,3,2,1 ... 先頭が減衰 0 回。共通計算から作る。
    """
    out = [rate_base]
    for _ in range(MARGIN_TIME_MAX_DECAYS):
        elapsed = MARGIN_TIME_START_FROM_FIRST_MOVE_SEC + len(out) * MARGIN_TIME_DECAY_INTERVAL_SEC
        out.append(compute_effective_rate(elapsed - MARGIN_TIME_DECAY_INTERVAL_SEC / 2, rate_base, True))
    return tuple(out)


def margin_rate(elapsed_sec: float, rate_base: int = OJAMA_RATE_STANDARD) -> int:
    """最初の 1 手からの経過秒に対するレート (95.5 秒まで基本、以降 16 秒ごとに表を 1 つ進める)。"""
    return compute_effective_rate(elapsed_sec, rate_base, from_first_move=True)


# ============================ 入力 ============================


@dataclass(frozen=True)
class FormulaStepObs:
    """掛け算式 (消去数×10 と倍率) を読んだ 1 段。式は 1 段につき最初のフレームだけ立つ。"""

    t_sec: float
    product: int
    erased: Optional[int] = None
    multiplier: Optional[int] = None


@dataclass(frozen=True)
class SideObs:
    """片側の 1 フレーム分の観測。"""

    score: Optional[int]
    state: Optional[str] = None
    formula_step: Optional[FormulaStepObs] = None
    tsumo_count: Optional[int] = None
    next_slide: bool = False
    placed: bool = False
    board_sig: Optional[int] = None      # 確定盤面の署名 (置き終わりの着地の裏取り用)。無ければ裏取りなしで通す
    ojama_cells: Optional[int] = None    # 確定盤面のおじゃまセル数 (曖昧な着弾の証拠)
    dead: bool = False                   # 窒息位置にぷよがある (決着の判定)
    icon_total: Optional[int] = None     # 受け予告アイコン合計 (参考列。台帳の値・印には一切使わない: user 2026-10-05)
    icon_count: Optional[int] = None     # アイコンの個数 (参考列)
    filled_cells: Optional[int] = None   # 確定盤面の埋まりセル数 (着弾が空きで頭打ちになる印の判定)
    ojama_cols: Optional[Tuple[int, ...]] = None   # 列ごとのおじゃま数 (着弾パターンの診断)
    chain_info: Optional[dict] = None  # 到着済み発火前盤と段得点。全消し検証フラグ時だけ使用。


@dataclass(frozen=True)
class FrameObs:
    """両側の 1 フレーム分の観測。"""

    t_sec: float
    frame: int
    game_idx: int
    match_active: bool
    sides: Tuple[SideObs, SideObs]


# ============================ 出力 ============================


@dataclass(frozen=True)
class LedgerEvent:
    """台帳が発したイベント。t_event=現象の時刻、available_at_sec=台帳が確定できた時刻。"""

    kind: str
    side: int
    t_event: float
    available_at_sec: float
    detail: Tuple[Tuple[str, object], ...] = ()

    def get(self, key: str, default: object = None) -> object:
        """detail から値を引く。"""
        for k, v in self.detail:
            if k == key:
                return v
        return default


@dataclass(frozen=True)
class SideLedger:
    """片側の台帳の状態。pending は「この側が受ける予告」。"""

    score: Optional[int]
    cum_drop: int
    cum_chain: int
    cum_all_clear: int
    cum_unmatched: int
    carry: int
    generated_total: int
    sent_total: int
    offset_total: int
    pending: int
    pending_confirmed: int
    dropped_total: int
    chain_open: bool
    flags: frozenset
    observed_landing: int = 0
    board_increase_total: int = 0
    board_excess_total: int = 0


@dataclass(frozen=True)
class LedgerSnapshot:
    """1 フレーム時点の台帳。値はすべて available_at_sec 以前の観測だけで決まる。"""

    t_sec: float
    frame: int
    available_at_sec: float
    game_idx: int
    match_reset: bool
    rate: int
    margin_origin_sec: Optional[float]
    sides: Tuple[SideLedger, SideLedger]
    ambiguous: Tuple[int, int]
    match_over: bool
    events: Tuple[LedgerEvent, ...]


@dataclass
class StepRecord:
    """連鎖の 1 段。score_applied は得点が反映されるまで None。"""

    t_formula: float
    product: int
    t_score: Optional[float] = None
    delta: Optional[int] = None
    generated: int = 0
    offset: int = 0
    sent: int = 0


@dataclass
class ChainRecord:
    """1 回の連鎖 (複数連鎖を 1 塊にしない)。"""

    side: int
    chain_id: int
    t_start: float
    steps: List[StepRecord] = field(default_factory=list)
    t_end: Optional[float] = None
    end_reason: Optional[str] = None
    last_step_t: float = 0.0
    last_applied_t: Optional[float] = None
    flags: Set[str] = field(default_factory=set)

    def generated(self) -> int:
        """連鎖全体の生成個数。"""
        return sum(s.generated for s in self.steps)

    def sent(self) -> int:
        """相殺後に相手へ送った個数。"""
        return sum(s.sent for s in self.steps)

    def offset(self) -> int:
        """自分の受け予告の相殺に使った個数。"""
        return sum(s.offset for s in self.steps)


@dataclass
class _Batch:
    """受け側の予告の 1 まとまり (どの連鎖由来か・確定済みか)。"""

    n: int
    src_side: int
    chain_id: int
    confirmed: bool = False
    confirmed_t: Optional[float] = None
    gen_t: float = 0.0


@dataclass
class _Lock:
    """受け側の「手が置かれた」証拠 1 件と、その着弾判断の状態。"""

    t: float
    state: str = "wait"          # wait=判断待ち / amb=曖昧で盤面の証拠待ち
    baseline: Optional[int] = None
    ambiguous_since: float = 0.0
    fallback_land: bool = False


@dataclass
class _SideState:
    """片側の可変状態 (台帳内部)。"""

    adopted: Optional[int] = None
    regress_val: Optional[int] = None
    regress_since: Optional[float] = None
    raw_win: Deque[Tuple[float, int]] = field(default_factory=deque)
    cand_end_t: Optional[float] = None
    cand_reason: Optional[str] = None
    quiet_since: Optional[float] = None
    zero_since: Optional[float] = None
    dead_since: Optional[float] = None
    last_placed_raw_t: Optional[float] = None
    pending_corr: Optional[Tuple[float, Optional[int]]] = None
    last_ev_t: Optional[float] = None
    last_ev_tsumo: Optional[int] = None
    prev_board_sig: Optional[int] = None
    lock: Optional[_Lock] = None
    ojama_hist: Deque[Tuple[float, int]] = field(default_factory=deque)
    acc: int = 0
    cum_drop: int = 0
    cum_chain: int = 0
    cum_all_clear: int = 0
    cum_unmatched: int = 0
    generated_total: int = 0
    sent_total: int = 0
    offset_total: int = 0
    dropped_total: int = 0
    queue: Deque[StepRecord] = field(default_factory=deque)
    chain: Optional[ChainRecord] = None
    chain_seq: int = 0
    prev_next_slide: bool = False
    slide_edge: bool = False
    prev_state: Optional[str] = None
    ojama_edge: bool = False
    last_decision_t: Optional[float] = None
    last_interrupt_t: Optional[float] = None
    prev_tsumo: Optional[int] = None
    prev_tsumo_ev: Optional[int] = None
    obs_base: Optional[int] = None
    obs_hold_val: Optional[int] = None
    obs_hold_since: Optional[float] = None
    observed_landing_total: int = 0
    board_increase_total: int = 0
    board_excess_total: int = 0
    obs_cols: Optional[Tuple[int, ...]] = None
    last_edge_t: Optional[float] = None
    board_changed: bool = False
    last_formula_t: Optional[float] = None
    flags: Set[str] = field(default_factory=set)


# ============================ 台帳本体 ============================


class ScoreLedgerV2:
    """オンライン (因果的) な得点ベース予告台帳。`update(obs)` を 1 フレームずつ呼ぶ。"""

    def __init__(
        self,
        rate_base: int = OJAMA_RATE_STANDARD,
        placement_refractory_sec: float = PLACEMENT_REFRACTORY_SEC,
        landing_delay_sec: float = LANDING_DECISION_DELAY_SEC,
        mode: str = MODE_OBSERVED,
        newest_first_offset: bool = False,
        defer_all_clear: bool = False,
        all_clear_native: object = None,
    ) -> None:
        self.mode = mode
        self.newest_first_offset = newest_first_offset
        self.defer_all_clear = defer_all_clear
        self.all_clear_native = getattr(all_clear_native, 'simulate_chain_trace_py', all_clear_native)
        self.rate_base = rate_base
        self.placement_refractory_sec = placement_refractory_sec
        self.landing_delay_sec = landing_delay_sec
        self.chains: List[ChainRecord] = []
        self._game_idx: Optional[int] = None
        self._prev_t: Optional[float] = None
        self._events: List[LedgerEvent] = []
        self._reset_state()

    # ---------- 試合境界 ----------

    def _reset_state(self) -> None:
        """試合境界で全状態を捨てる (台帳・起点・予告)。完了済み chains は残す。"""
        self.st: List[_SideState] = [_SideState() for _ in range(SIDE_COUNT)]
        self.batches: List[List[_Batch]] = [[] for _ in range(SIDE_COUNT)]
        self.origin_sec: Optional[float] = None
        self._origin_unknown: bool = False
        self._over: bool = False
        from src.phase_j._live_ex_frozen.src.b3_all_clear_credit import AllClearCredit
        self.all_clear = [AllClearCredit(), AllClearCredit()]

    def _detect_reset(self, obs: FrameObs) -> Optional[str]:
        """試合境界の検出。game_idx 変化を主、試合外の後の再開と得点 0 復帰 (両側) を保険にする。"""
        if self._game_idx is None:
            return "first_frame"
        if obs.game_idx != self._game_idx:
            return "game_idx"
        # 試合内の一時的な MENU (演出・誤判定、数秒続く) は試合境界にしない。境界は game_idx の変化か、
        # 試合中表示 (match_active) での両者の得点 0 リセットで判定する (統括報告: MENU 後の累計リセット 18/281 試合)。
        for i, so in enumerate(obs.sides):
            st = self.st[i]
            zero = obs.match_active and so.score == 0
            st.zero_since = (st.zero_since if st.zero_since is not None else obs.t_sec) if zero else None
        both = all(st.zero_since is not None and obs.t_sec - st.zero_since >= RESET_ZERO_SEC for st in self.st)
        had = any((st.adopted or 0) > 0 for st in self.st)
        return FLAG_RESET_BY_SCORE if both and had else None

    # ---------- 公開 API ----------

    def update(self, obs: FrameObs) -> LedgerSnapshot:
        """1 フレームを取り込み、その時点の台帳を返す (t 以前の観測のみ使用)。"""
        self._events = []
        reason = self._detect_reset(obs)
        if reason is not None:
            self._close_open_chains(obs.t_sec, REASON_MATCH_RESET)
            self._reset_state()
            self._events.append(self._event("match_reset", -1, obs.t_sec, obs.t_sec, (("reason", reason),)))
        self._game_idx = obs.game_idx
        if obs.match_active:
            if not self._over:
                self._step_frame(obs)
        else:
            for st in self.st:
                st.raw_win.clear()
        self._prev_t = obs.t_sec
        return self._snapshot(obs, reason is not None)

    # ---------- 1 フレームの処理 ----------

    def _step_frame(self, obs: FrameObs) -> None:
        """試合中の 1 フレーム。式 → 得点 → 連鎖終了 → 置き終わりの証拠 → 着弾 → 決着の順。"""
        for i, so in enumerate(obs.sides):
            if self.defer_all_clear:
                self._observe_all_clear(i, so, obs.t_sec)
            self._intake_formula(i, so, obs.t_sec)
        for i, so in enumerate(obs.sides):
            self._update_origin(i, so, obs.t_sec)
            for delta, t_event in self._adopt_score(i, so.score, obs.t_sec):
                self._attribute(i, delta, t_event, obs.t_sec)
        for i, so in enumerate(obs.sides):
            self._observe_edges(i, so, obs.t_sec)
            self._check_chain_end(i, so, obs)
        for i, so in enumerate(obs.sides):
            self._resolve_lock(i, so, obs.t_sec)      # 先に前の手の判断を済ませてから、新しい手の証拠を拾う
            self._collect_evidence(i, so, obs.t_sec)
        for i, so in enumerate(obs.sides):
            self._observe_landing(i, so, obs.t_sec)     # 第3版でも診断としては観測する (値を変えるのは第4版だけ)
        self._check_match_over(obs)

    def _update_origin(self, i: int, so: SideObs, t: float) -> None:
        """レート起点 = 両者のうち早い方の最初の 1 手 (置き終わり信号 or NEXT 手数 0→1)。"""
        st = self.st[i]
        rising = st.prev_tsumo == 0 and (so.tsumo_count or 0) >= 1
        if (so.placed or rising) and self.origin_sec is None and not self._origin_unknown:
            self.origin_sec = t
        if so.tsumo_count is not None:
            st.prev_tsumo = so.tsumo_count

    def rate_at(self, t_sec: float) -> int:
        """時刻 t のおじゃまレート (起点が未確定の間は基本レート)。"""
        if self.origin_sec is None or t_sec <= self.origin_sec:
            return self.rate_base
        return margin_rate(t_sec - self.origin_sec, self.rate_base)

    # ---------- 式段の取り込み ----------

    def _intake_formula(self, i: int, so: SideObs, t: float) -> None:
        """式段を待ち行列に積み、連鎖 (ChainRecord) に属させる。

        別連鎖の条件: 式の間隔が CHAIN_STEP_GAP_MAX_SEC 超、または 2 式の間 (前の段の得点反映後) に
        この側の置き終わりがある (user: 置き終わりの合図の 1 つが連鎖開始 = 掛け算式)。
        """
        fs = so.formula_step
        if fs is None:
            return
        st = self.st[i]
        ch = st.chain
        if ch is not None and self._splits_chain(st, ch, fs.t_sec):
            self._end_chain(i, t, REASON_PLACEMENT_BETWEEN if fs.t_sec - ch.last_step_t <= CHAIN_STEP_GAP_MAX_SEC
                            else REASON_STEP_GAP, t_end=st.cand_end_t)
            ch = None
        if ch is None:
            ch = self._open_chain(i, fs.t_sec)
        rec = StepRecord(t_formula=fs.t_sec, product=fs.product)
        ch.steps.append(rec)
        ch.last_step_t = fs.t_sec
        st.cand_end_t, st.last_edge_t = None, None   # 次の段が来た = これまでの終了信号は段間のちらつき
        st.queue.append(rec)
        st.last_formula_t = t

    @staticmethod
    def _splits_chain(st: _SideState, ch: ChainRecord, t_formula: float) -> bool:
        """この式は現在の連鎖の続きか (False) 別の連鎖か (True)。"""
        if t_formula - ch.last_step_t > CHAIN_STEP_GAP_MAX_SEC:
            return True
        base = ch.last_applied_t if ch.last_applied_t is not None else ch.last_step_t
        placed = st.last_placed_raw_t
        return placed is not None and placed > base and placed < t_formula

    def _open_chain(self, i: int, t: float) -> ChainRecord:
        """連鎖を開く。"""
        st = self.st[i]
        st.chain_seq += 1
        ch = ChainRecord(side=i, chain_id=st.chain_seq, t_start=t, last_step_t=t)
        st.chain = ch
        self._note_interrupt(i, t)
        return ch

    def _note_interrupt(self, i: int, t: float) -> None:
        """攻撃側 i の新しい連鎖が、受け側 (相手) の自連鎖による相殺の最中に始まったら記録する。"""
        o = 1 - i
        pending = self._pending(o)
        if self.st[o].chain is None or pending <= 0:
            return
        self.st[o].last_interrupt_t = t
        self._events.append(
            self._event("interrupt", i, t, t, (("receiver", o), ("pending_receiver", pending),
                                                ("pending_attacker", self._pending(i))))
        )

    # ---------- 得点の採用 (頑健化) ----------

    def _adopt_score(self, i: int, raw: Optional[int], t_now: float) -> List[Tuple[int, float]]:
        """生の読みから採用値を更新し、(増分, 発生時刻) を返す。窓 SCORE_PERSIST_SEC の最小値を採用 (遅れる)。"""
        st = self.st[i]
        if raw is None:
            st.raw_win.clear()
            return []
        st.raw_win.append((t_now, raw))
        while st.raw_win and t_now - st.raw_win[0][0] > SCORE_PERSIST_SEC:
            st.raw_win.popleft()
        if len(st.raw_win) < 2:
            return []
        cand = min(v for _, v in st.raw_win)
        t_event = st.raw_win[-2][0]
        if st.adopted is None:
            return self._adopt_first(i, cand, t_event)
        if cand > st.adopted:
            delta = cand - st.adopted
            st.adopted = cand
            st.regress_since = None
            return [(delta, t_event)]
        self._track_regress(i, raw, t_now)
        return []

    def _adopt_first(self, i: int, cand: int, t_event: float) -> List[Tuple[int, float]]:
        """試合の最初の採用値。小さければ 0 点から見ていた、大きければ履歴不完全。"""
        st = self.st[i]
        st.adopted = cand
        if cand <= FIRST_READ_COMPLETE_MAX:
            st.acc = st.cum_drop = cand
            return []
        st.flags.add(FLAG_HISTORY_INCOMPLETE)
        self._origin_unknown = True
        self.origin_sec = None
        st.flags.add(FLAG_ORIGIN_UNKNOWN)
        return []

    def _track_regress(self, i: int, raw: int, t: float) -> None:
        """試合内で下がる読みは継続しても採用せず、増分の基準を維持する。"""
        st = self.st[i]
        if st.adopted is None or raw >= st.adopted:
            return
        if raw != st.regress_val or st.regress_since is None:
            st.regress_val, st.regress_since = raw, t
        if t - st.regress_since >= REGRESS_RESYNC_SEC and raw > 0:
            st.flags.add(FLAG_SCORE_RESYNC)
            st.regress_since = None

    # ---------- 増分の帰属と換算 ----------

    def _expire_formulas(self, i: int, t: float) -> None:
        """得点が反映されないまま古くなった式段を捨てる。"""
        st = self.st[i]
        while st.queue and t - st.queue[0].t_formula > FORMULA_EXPIRE_SEC:
            st.queue.popleft()
            st.flags.add(FLAG_UNMATCHED_FORMULA)
            if st.chain is not None:
                st.chain.flags.add(FLAG_UNMATCHED_FORMULA)

    def _match_steps(self, i: int, delta: int) -> Tuple[int, int]:
        """delta に対応する先頭 k 段を探す。返り値=(k, 余り)。k=0 は不一致。"""
        total = 0
        for k, rec in enumerate(self.st[i].queue, start=1):
            total += rec.product
            resid = delta - total
            if resid < 0:
                break
            if resid <= FORMULA_DROP_SLACK:
                return k, resid
            if ALL_CLEAR_BONUS <= resid <= ALL_CLEAR_BONUS + FORMULA_DROP_SLACK:
                return k, resid
        return 0, delta

    def _observe_all_clear(self, i: int, so: SideObs, t: float) -> None:
        """物理証拠を得点採用より先に登録する。フラグOFF時は呼ばない。"""
        if so.chain_info and self.all_clear_native is None:
            import puyo_core
            self.all_clear_native = puyo_core.simulate_chain_trace_py
        self.all_clear[i].observe(so.chain_info, so.state, self.st[i].adopted,
                                  t, self.all_clear_native)

    def _defer_bonus(self, i: int, delta: int, t_event: float, t_now: float) -> int:
        """表示得点を保持したまま、確認済み全消しの2100点だけを別勘定へ移す。"""
        st, credit = self.st[i], self.all_clear[i]
        ordinary, evidence = credit.split(st.adopted, delta)
        if evidence:
            st.cum_all_clear += ALL_CLEAR_BONUS
            if st.chain is not None:
                credit.recognized_chain_ids.add(st.chain.chain_id)
            self._emit_attr(i, KIND_ALL_CLEAR, ALL_CLEAR_BONUS, t_event, t_now)
            self._events.append(self._event('all_clear_deferred', i, t_event, t_now,
                (('trigger', credit.current['trigger']), ('credit', 30), ('evidence', evidence))))
        return ordinary

    def _attribute(self, i: int, delta: int, t_event: float, t_now: float) -> None:
        """採用した得点増分を 連鎖段 / 落下 / 全消し / 未対応 に帰属し、連鎖なら換算する。"""
        st = self.st[i]
        if self.defer_all_clear:
            delta = self._defer_bonus(i, delta, t_event, t_now)
            if delta == 0:
                return
        self._expire_formulas(i, t_now)
        k, resid = self._match_steps(i, delta)
        if k > 0:
            self._apply_matched(i, k, resid, t_event, t_now)
        elif delta <= DROP_ONLY_MAX_DELTA:
            st.acc += delta
            st.cum_drop += delta
            self._emit_attr(i, KIND_DROP, delta, t_event, t_now)
        else:
            self._apply_unmatched(i, delta, t_event, t_now)

    def _apply_matched(self, i: int, k: int, resid: int, t_event: float, t_now: float) -> None:
        """式に対応する k 段を反映する。余りは全消しボーナス or 落下点。段ごとに換算する。"""
        st = self.st[i]
        recs = [st.queue.popleft() for _ in range(k)]
        st.acc += resid
        if resid >= ALL_CLEAR_BONUS:
            if self.defer_all_clear:
                # 式の残差だけでは認定しない。不明分は攻撃に使わず、死亡証明のガードを残す。
                st.acc -= ALL_CLEAR_BONUS
                self.all_clear[i].uncertain_points += ALL_CLEAR_BONUS
            st.cum_all_clear += ALL_CLEAR_BONUS
            st.cum_drop += resid - ALL_CLEAR_BONUS
            self._emit_attr(i, KIND_ALL_CLEAR, ALL_CLEAR_BONUS, t_event, t_now)
        else:
            st.cum_drop += resid
            if resid > 0:
                self._emit_attr(i, KIND_DROP, resid, t_event, t_now)
        for rec in recs:
            rec.t_score, rec.delta = t_event, rec.product
            st.acc += rec.product
            st.cum_chain += rec.product
            self._emit_attr(i, KIND_CHAIN, rec.product, t_event, t_now)
            self._convert(i, rec, t_event)
        if st.chain is not None:
            st.chain.last_applied_t = t_event

    def _apply_unmatched(self, i: int, delta: int, t_event: float, t_now: float) -> None:
        """式が読めていない大きな増分。得点ベースなので換算はするが品質フラグを立てる。"""
        st = self.st[i]
        st.acc += delta
        st.cum_unmatched += delta
        st.flags.add(FLAG_UNMATCHED_JUMP)
        ch = st.chain or self._open_chain(i, t_event)
        ch.flags.update((FLAG_UNMATCHED_JUMP, FLAG_NO_FORMULA_CHAIN))
        rec = StepRecord(t_formula=t_event, product=delta, t_score=t_event, delta=delta)
        ch.steps.append(rec)
        ch.last_step_t = t_event
        ch.last_applied_t = t_event
        self._emit_attr(i, KIND_UNMATCHED, delta, t_event, t_now)
        self._convert(i, rec, t_event)

    def _convert(self, i: int, rec: StepRecord, t: float) -> None:
        """連鎖の得点が入った時点で acc をおじゃまに換算し、相殺→相手の予告へ。"""
        st = self.st[i]
        rate = self.rate_at(t)
        n = st.acc // rate
        st.acc -= n * rate
        if self.defer_all_clear:
            n += self.all_clear[i].release()
            credit = self.all_clear[i]
            if st.chain is not None and credit.current and credit.current['trigger'] in credit.awarded:
                credit.recognized_chain_ids.add(st.chain.chain_id)
        used = self._offset_own(i, n)
        sent = n - used
        rec.generated, rec.offset, rec.sent = n, used, sent
        st.generated_total += n
        st.offset_total += used
        st.sent_total += sent
        if sent > 0:
            ch = st.chain.chain_id if st.chain is not None else st.chain_seq
            self.batches[1 - i].append(_Batch(n=sent, src_side=i, chain_id=ch, gen_t=t))

    def _offset_own(self, i: int, n: int) -> int:
        """自分の受け予告を n 個まで相殺し、使った個数を返す。

        本番既定は割り込みの消費順をモデル化しない旧動作を維持する (user 指示 2026-10-05)。
        既定は旧順序を保持。v4.7実験の newest_first_offset では後着から相殺し、
        確定済みの古い分を着弾用に残す（2026-10-15 user指定）。
        """
        used = 0
        batches = reversed(self.batches[i]) if self.newest_first_offset else self.batches[i]
        for b in batches:
            take = min(b.n, n - used)
            b.n -= take
            used += take
            if used >= n:
                break
        self.batches[i] = [b for b in self.batches[i] if b.n > 0]
        return used

    def _emit_attr(self, i: int, kind: str, pts: int, t_event: float, t_now: float) -> None:
        """増分の帰属イベントを積む。"""
        self._events.append(
            self._event("score_attr", i, t_event, t_now, (("kind", kind), ("points", pts)))
        )

    # ---------- 連鎖終了 ----------

    def _end_candidate(self, st: _SideState, ch: ChainRecord, t: float) -> Optional[float]:
        """連鎖の終わりの候補時刻。最後の段の得点反映後に「画面が静止に戻った」始点、無ければ NEXT 移動 /
        おじゃま落下の時刻 (user 定義: 画面のぷよが全て接地した時)。次の段が来る可能性があるので候補。"""
        applied = ch.last_applied_t
        if applied is None:
            return None
        q = st.quiet_since
        if q is not None and q > applied and t - q >= CHAIN_END_QUIET_SEC:
            return q
        e = st.last_edge_t
        return e if e is not None and e > applied else None

    def _check_chain_end(self, i: int, so: SideObs, obs: FrameObs) -> None:
        """終了候補を毎フレーム更新し、最後の式から CHAIN_END_SIGNAL_MIN_SEC たって候補があれば確定する。

        確定した連鎖の終了時刻は候補の時刻 (遡及)。検出は遅れるが時刻は信号が立った時を指す。
        """
        st = self.st[i]
        ch = st.chain
        if ch is None:
            return
        t = obs.t_sec
        self._expire_formulas(i, t)
        if st.queue:
            st.cand_end_t = None
            return
        st.cand_end_t = self._end_candidate(st, ch, t)
        since = t - ch.last_step_t
        if st.cand_end_t is not None and since >= CHAIN_END_SIGNAL_MIN_SEC:
            self._end_chain(i, t, st.cand_reason or REASON_SIGNAL_QUIET, t_end=st.cand_end_t)
        elif ch.last_applied_t is not None and since > CHAIN_STEP_GAP_MAX_SEC:
            self._end_chain(i, t, REASON_FORMULA_GAP)
        elif since > CHAIN_END_TIMEOUT_SEC:
            ch.flags.add(FLAG_END_TIMEOUT)
            self._end_chain(i, t, REASON_TIMEOUT)

    def _end_chain(self, i: int, t: float, reason: str, t_end: Optional[float] = None) -> None:
        """連鎖を閉じ、その連鎖が送った予告を確定にし、自分側の着弾を判定する。

        t_end = 終わりの時刻 (遡及。無ければ検出時刻)。t = 検出した時刻 (available_at)。
        """
        st = self.st[i]
        ch = st.chain
        if ch is None:
            return
        end = t if t_end is None else min(t_end, t)
        ch.t_end, ch.end_reason = end, reason
        self.chains.append(ch)
        st.chain = None
        if st.queue:
            ch.flags.add(FLAG_UNMATCHED_FORMULA)
            st.flags.add(FLAG_UNMATCHED_FORMULA)
        st.queue.clear()
        st.cand_end_t = None
        for b in self.batches[1 - i]:
            if b.src_side == i and b.chain_id == ch.chain_id and not b.confirmed:
                b.confirmed, b.confirmed_t = True, end
        self._events.append(
            self._event("chain_end", i, end, t, (("chain_id", ch.chain_id), ("reason", reason),
                                                  ("generated", ch.generated()), ("sent", ch.sent())))
        )
        # 受け側の連鎖終了 = 相殺し切れなかった残りが降る (user 回答)。連鎖を始めた手の時刻を基準にする。
        self._land(i, ch.t_start - LOCK_TO_FIRST_FORMULA_SEC, t, check_busy=True)

    def _close_open_chains(self, t: float, reason: str) -> None:
        """試合境界で開いたままの連鎖を閉じる (記録だけ。着弾は起こさない)。"""
        for st in self.st:
            if st.chain is not None:
                st.chain.t_end, st.chain.end_reason = t, reason
                self.chains.append(st.chain)
                st.chain = None

    # ---------- 着弾 ----------

    def _observe_edges(self, i: int, so: SideObs, t: float) -> None:
        """立ち上がり (NEXT 移動・おじゃま落下)、静止の始点、tsumo_count の増分、盤面の確定変化を更新する。"""
        st = self.st[i]
        st.slide_edge = so.next_slide and not st.prev_next_slide
        st.ojama_edge = so.state == STATE_OJAMA_FALL and st.prev_state != STATE_OJAMA_FALL
        st.prev_state, st.prev_next_slide = so.state, so.next_slide
        quiet = so.state in STATE_CHAIN_END_QUIET
        st.quiet_since = (st.quiet_since if st.quiet_since is not None else t) if quiet else None
        if st.slide_edge or st.ojama_edge:
            st.last_edge_t = t
            st.cand_reason = REASON_SIGNAL_SLIDE if st.slide_edge else REASON_SIGNAL_OJAMA
        inc = so.tsumo_count is not None and st.prev_tsumo_ev is not None and so.tsumo_count > st.prev_tsumo_ev
        if so.tsumo_count is not None:
            st.prev_tsumo_ev = so.tsumo_count
        if so.placed or inc:
            st.last_placed_raw_t = t
            if st.pending_corr is None:
                st.pending_corr = (t, so.board_sig)
        st.board_changed = (so.board_sig is not None and st.prev_board_sig is not None
                            and so.board_sig != st.prev_board_sig)
        st.prev_board_sig = so.board_sig
        if so.ojama_cells is not None:
            st.ojama_hist.append((t, so.ojama_cells))
        while st.ojama_hist and t - st.ojama_hist[0][0] > AMBIGUOUS_EVIDENCE_SEC + 2.0:
            st.ojama_hist.popleft()

    def _corroborated_placement(self, st: _SideState, so: SideObs, t: float) -> Optional[float]:
        """置き終わりの証拠 (信号 / tsumo_count 増分) を着地で裏取りできたら、その置き終わりの時刻を返す。

        着地 = NEXT 移動 か 盤面の確定変化 (天井付近で組ぷよが回転中に信号だけ立つ誤検知を除く)。
        盤面の情報が無い入力は裏取りなしで通す。
        """
        pc = st.pending_corr
        if pc is None:
            return None
        if t - pc[0] > PLACEMENT_CORROBORATE_SEC:
            st.pending_corr = None
            return None
        landed = st.slide_edge or st.board_changed or pc[1] is None or so.board_sig is None
        if landed:
            st.pending_corr = None
            return pc[0]
        return None

    def _collect_evidence(self, i: int, so: SideObs, t: float) -> None:
        """手が置かれた証拠 (裏取りした置き終わり / 単独の NEXT 移動・おじゃま落下) から着弾判断 (_Lock) を作る。"""
        st = self.st[i]
        t_ev = self._corroborated_placement(st, so, t)
        kind = "placed"
        if t_ev is None and (st.slide_edge or st.ojama_edge) and st.pending_corr is None:
            since = None if st.last_decision_t is None else t - st.last_decision_t
            if st.slide_edge or since is None or since >= OJAMA_EDGE_REFRACTORY_SEC:
                t_ev, kind = t, "edge"
        if t_ev is None:
            return
        gap = None if st.last_ev_t is None else t_ev - st.last_ev_t
        new_piece = kind == "placed" and so.tsumo_count != st.last_ev_tsumo
        if gap is not None and gap < EVIDENCE_MERGE_SEC and not new_piece:
            return
        st.last_ev_t, st.last_ev_tsumo = t_ev, so.tsumo_count
        if st.chain is None and not st.queue and st.lock is None:
            st.lock = _Lock(t=t_ev)

    def _attacker_status(self, o: int, t_lock: float) -> Tuple[str, Optional[float]]:
        """置いた時刻 t_lock の攻撃側の連鎖の状態: busy (連鎖中) / wait (終了候補あり・未確定) / closed (終了時刻) / none。"""
        st = self.st[o]
        ch = st.chain
        if ch is not None and ch.t_start <= t_lock:
            cand = st.cand_end_t
            if cand is None or cand > t_lock + AMBIGUOUS_WINDOW_SEC:
                return "busy", None
            return "wait", cand
        for rec in reversed(self.chains[-LANDING_CHAIN_LOOKBACK:]):
            if rec.side == o and rec.t_start <= t_lock:
                return "closed", rec.t_end
        return "none", None

    def _resolve_lock(self, i: int, so: SideObs, t: float, force: bool = False) -> None:
        """着弾判断。攻撃側の連鎖が置く前に終わっていれば降らせ、連鎖中なら降らさず、同時刻なら証拠待ち。"""
        st = self.st[i]
        lk = st.lock
        if lk is None:
            return
        if st.chain is not None or st.queue or (st.last_formula_t is not None and st.last_formula_t >= lk.t):
            st.lock = None                  # 置いた手が連鎖を始めた: 降るのはその連鎖の終了後 (_end_chain)
            return
        if lk.state == "amb":
            self._resolve_ambiguous(i, t)
            return
        if not force and t - lk.t < self.landing_delay_sec:
            return
        if not any(b.n > 0 for b in self.batches[i]):
            st.lock = None
            return
        kind, end = self._attacker_status(1 - i, lk.t)
        if kind == "busy":
            st.lock, st.last_decision_t = None, t
        elif kind == "wait" and force:
            st.lock = None                  # 攻撃側の連鎖の終わりが未確定のまま次の手が置かれた: 次の手の判断に任せる
        elif kind == "wait" and t - lk.t < self.landing_delay_sec + LANDING_WAIT_MAX_SEC:
            return
        elif end is not None and abs(end - lk.t) <= AMBIGUOUS_WINDOW_SEC:
            self._start_ambiguous(i, lk, t, end <= lk.t)
        elif end is not None and end > lk.t:
            st.lock, st.last_decision_t = None, t
        else:
            self._land(i, lk.t, t, by="time")
            st.lock, st.last_decision_t = None, t

    def _start_ambiguous(self, i: int, lk: _Lock, t: float, fallback_land: bool) -> None:
        """置き終わりと攻撃側の連鎖終了が約 AMBIGUOUS_WINDOW_SEC 以内: 印を付け、盤面の証拠で後から確定する。"""
        st = self.st[i]
        base = [c for tt, c in st.ojama_hist if lk.t - 0.5 <= tt <= lk.t]
        lk.state, lk.ambiguous_since = "amb", t
        lk.baseline = min(base) if base else None
        lk.fallback_land = fallback_land
        st.flags.add(FLAG_TIMING_AMBIGUOUS)
        self._events.append(self._event("timing_ambiguous", i, lk.t, t, (("baseline", lk.baseline),)))

    def _resolve_ambiguous(self, i: int, t: float) -> None:
        """曖昧な着弾を、受け側盤面のおじゃま増加 (2 サンプルの最小値が基準を超える) で確定する (因果的・遅れて確定)。"""
        st = self.st[i]
        lk = st.lock
        assert lk is not None
        recent = [c for tt, c in st.ojama_hist if tt >= lk.ambiguous_since - 0.1]
        if lk.baseline is None or not recent:
            if lk.fallback_land:
                self._land(i, lk.t, t, tol=AMBIGUOUS_WINDOW_SEC, ambiguous=True, by="fallback")
            st.lock, st.last_decision_t = None, t
        elif len(recent) >= 2 and min(recent[-2:]) > lk.baseline:
            self._land(i, lk.t, t, tol=AMBIGUOUS_WINDOW_SEC, ambiguous=True, by="board")
            st.lock, st.last_decision_t = None, t
        elif t - lk.ambiguous_since >= AMBIGUOUS_EVIDENCE_SEC:
            st.lock, st.last_decision_t = None, t

    def _check_match_over(self, obs: FrameObs) -> None:
        """窒息位置にぷよがある状態が DEAD_HOLD_SEC 続いたら決着とみなし台帳を打ち切る (user 回答)。"""
        for i, so in enumerate(obs.sides):
            st = self.st[i]
            dead = so.dead and so.state in STATE_CHAIN_END_QUIET
            st.dead_since = (st.dead_since if st.dead_since is not None else obs.t_sec) if dead else None
            if st.dead_since is not None and obs.t_sec - st.dead_since >= DEAD_HOLD_SEC:
                self._over = True
                self._close_open_chains(obs.t_sec, REASON_MATCH_OVER)
                for sj in self.st:
                    sj.flags.add(FLAG_MATCH_OVER)
                    sj.lock = None
                self._events.append(self._event("match_over", i, st.dead_since, obs.t_sec, ()))
                return

    def _shadow_land(self, i: int, t_lock: float, t_now: float, tol: float, ambiguous: bool) -> int:
        """第4版では時刻規則の着弾は台帳の値を変えない。診断用に「予測した着弾」だけを記録する。"""
        room, total = LANDING_MAX_PER_EVENT, 0
        for b in self.batches[i]:
            if b.confirmed and b.confirmed_t is not None and b.confirmed_t <= t_lock + tol:
                take = min(b.n, room)
                room -= take
                total += take
        if total > 0:
            self._events.append(self._event("landing_predicted", i, t_lock, t_now,
                                            (("n", total), ("ambiguous", int(ambiguous)))))
        return 0

    def _drain(self, i: int, n: int) -> int:
        """側 i の予告から古い順に n 個まで差し引く (観測した着弾)。差し引いた個数を返す。"""
        left = n
        keep: List[_Batch] = []
        for b in self.batches[i]:
            take = min(b.n, left)
            b.n -= take
            left -= take
            if b.n > 0:
                keep.append(b)
        self.batches[i] = keep
        return n - left

    def _observe_landing(self, i: int, so: SideObs, t: float) -> None:
        """受け側の確定盤面 (STABLE) のおじゃま数の増分を着弾として観測する。減る側は着弾ではない。

        第4版 (observed) だけ、観測した着弾を予告から差し引く (予告を超える分は board_excess に記録)。
        アイコンは使わない (user 2026-10-05)。
        """
        st = self.st[i]
        cur = so.ojama_cells
        if cur is None or so.state not in STATE_CHAIN_END_QUIET:
            st.obs_hold_val = None
            return
        if st.obs_base is None:
            st.obs_base, st.obs_cols = cur, so.ojama_cols
        if cur != st.obs_hold_val:
            st.obs_hold_val, st.obs_hold_since = cur, t
        held = t - (st.obs_hold_since if st.obs_hold_since is not None else t)
        if cur > st.obs_base and held >= LAND_OBS_HOLD_SEC:
            self._on_observed_landing(i, so, cur - st.obs_base, t)
            st.obs_base, st.obs_cols = cur, so.ojama_cols
        elif cur < st.obs_base and held >= LAND_OBS_DROP_HOLD_SEC:
            st.obs_base, st.obs_cols = cur, so.ojama_cols

    def _on_observed_landing(self, i: int, so: SideObs, inc: int, t: float) -> None:
        """観測した着弾 inc 個を記録し、第4版なら予告から差し引く。列ごとの増分と頭打ちの印も付ける。"""
        st = self.st[i]
        st.board_increase_total += inc
        cols = None
        if so.ojama_cols is not None and st.obs_cols is not None:
            cols = tuple(a - b for a, b in zip(so.ojama_cols, st.obs_cols))
        landed = 0
        if self.mode == MODE_OBSERVED:
            landed = self._drain(i, inc)
            st.board_excess_total += inc - landed
            st.dropped_total += landed
            st.observed_landing_total += landed
            if landed > 0:
                self._flag_interrupt_landing(i, t)
        free = None if so.filled_cells is None else BOARD_CELLS - so.filled_cells
        if self.mode == MODE_OBSERVED and free is not None and free <= LANDING_MAX_PER_EVENT and self._pending(i) > 0:
            st.flags.add(FLAG_CAPACITY_LIMITED)
        self._events.append(self._event("observed_landing", i, t, t, (("n", landed), ("board_increase", inc), ("by", "observed"),
                                                             ("cols", cols), ("remaining", self._pending(i)))))

    def _flag_interrupt_landing(self, i: int, t: float) -> None:
        """直前 INTERRUPT_FLAG_WINDOW_SEC 以内に割り込みがあった側の着弾に品質フラグを立てる。"""
        ti = self.st[i].last_interrupt_t
        if ti is not None and 0.0 <= t - ti <= INTERRUPT_FLAG_WINDOW_SEC:
            self.st[i].flags.add(FLAG_LANDING_AFTER_INTERRUPT)
            self._events.append(self._event("landing_after_interrupt", i, t, t, (("interrupt_t", ti),)))

    def _chain_open_at(self, i: int, t: float) -> bool:
        """側 i が時刻 t に連鎖中だったか (開いた連鎖、または t をまたいで閉じた連鎖。終了は検出した時刻)。"""
        ch = self.st[i].chain
        if ch is not None and ch.t_start <= t:
            return True
        for rec in reversed(self.chains[-LANDING_CHAIN_LOOKBACK:]):
            if rec.side == i and rec.t_start <= t and (rec.t_end is None or t < rec.t_end):
                return True
        return False

    def _land(self, i: int, t_lock: float, t_now: float, check_busy: bool = False,
              tol: float = 0.0, ambiguous: bool = False, by: str = "time") -> int:
        """側 i に、t_lock (+tol) より前に確定した予告を先頭から最大 30 個降らせる。降らせた個数を返す。"""
        if check_busy and self._chain_open_at(1 - i, t_lock):
            return 0
        if self.mode == MODE_OBSERVED:
            return self._shadow_land(i, t_lock, t_now, tol, ambiguous)
        room = LANDING_MAX_PER_EVENT
        total = 0
        keep: List[_Batch] = []
        for b in self.batches[i]:
            ok = b.confirmed and b.confirmed_t is not None and b.confirmed_t <= t_lock + tol
            take = min(b.n, room) if ok else 0
            b.n -= take
            room -= take
            total += take
            if b.n > 0:
                keep.append(b)
        self.batches[i] = keep
        if total <= 0:
            return 0
        self.st[i].dropped_total += total
        self._flag_interrupt_landing(i, t_lock)
        self._events.append(
            self._event("landing", i, t_lock, t_now, (("n", total), ("remaining", self._pending(i)),
                                                      ("ambiguous", int(ambiguous)), ("by", by)))
        )
        return total

    # ---------- 出力 ----------

    @staticmethod
    def _event(
        kind: str, side: int, t_event: float, avail: float, detail: Tuple[Tuple[str, object], ...]
    ) -> LedgerEvent:
        return LedgerEvent(kind=kind, side=side, t_event=t_event, available_at_sec=avail, detail=detail)

    def _pending(self, i: int, confirmed_only: bool = False) -> int:
        """側 i が受ける予告の個数 (上限で切らない)。"""
        return sum(b.n for b in self.batches[i] if b.confirmed or not confirmed_only)

    def _side_ledger(self, i: int) -> SideLedger:
        st = self.st[i]
        return SideLedger(
            score=st.adopted, cum_drop=st.cum_drop, cum_chain=st.cum_chain,
            cum_all_clear=st.cum_all_clear, cum_unmatched=st.cum_unmatched, carry=st.acc,
            generated_total=st.generated_total, sent_total=st.sent_total,
            offset_total=st.offset_total, pending=self._pending(i),
            pending_confirmed=self._pending(i, True), dropped_total=st.dropped_total,
            chain_open=st.chain is not None, flags=frozenset(st.flags),
            observed_landing=st.observed_landing_total, board_increase_total=st.board_increase_total,
            board_excess_total=st.board_excess_total,
        )

    def _snapshot(self, obs: FrameObs, reset: bool) -> LedgerSnapshot:
        return LedgerSnapshot(
            t_sec=obs.t_sec, frame=obs.frame, available_at_sec=obs.t_sec,
            game_idx=obs.game_idx, match_reset=reset, rate=self.rate_at(obs.t_sec),
            margin_origin_sec=self.origin_sec,
            sides=(self._side_ledger(0), self._side_ledger(1)),
            ambiguous=tuple(int(st.lock is not None and st.lock.state == "amb") for st in self.st),  # type: ignore[arg-type]
            match_over=self._over,
            events=tuple(self._events),
        )


def replay(frames: Iterable[FrameObs], **kwargs: object) -> Iterator[LedgerSnapshot]:
    """観測列を先頭から順に台帳へ流し、フレームごとの snapshot を返す (因果的)。"""
    ledger = ScoreLedgerV2(**kwargs)  # type: ignore[arg-type]
    for obs in frames:
        yield ledger.update(obs)


# ============================ B2 記録のアダプタ ============================


def _formula_from_record(fs: Optional[dict]) -> Optional[FormulaStepObs]:
    """記録の formula_step を FormulaStepObs へ。"""
    if not fs or fs.get("product") is None:
        return None
    return FormulaStepObs(
        t_sec=float(fs.get("t_sec", 0.0)), product=int(fs["product"]),
        erased=fs.get("erased"), multiplier=fs.get("multiplier"),
    )


def _board_features(board: Optional[list]) -> Tuple[Optional[int], Optional[int], bool, Optional[int], Optional[Tuple[int, ...]]]:
    """確定盤面から (署名, おじゃまセル数, 窒息位置にぷよがあるか, 埋まりセル数, 列ごとのおじゃま数) を作る。"""
    if not board:
        return None, None, False, None, None
    flat = tuple(int(c) for row in board for c in row)
    dead = len(board) > DEATH_ROW and len(board[DEATH_ROW]) > DEATH_COL and int(board[DEATH_ROW][DEATH_COL]) != 0
    cols = tuple(sum(1 for row in board if int(row[c]) == OJAMA_CELL) for c in range(BOARD_COLS))
    return hash(flat), sum(1 for c in flat if c == OJAMA_CELL), dead, sum(1 for c in flat if c != 0), cols


def frame_obs_from_record(rec: dict, placed_sides: Sequence[str] = ()) -> FrameObs:
    """B2 の observations 1 行 (kind=frame) を FrameObs へ。placed_sides は同フレームの置き終わり信号。"""
    sides: List[SideObs] = []
    for s in rec["sides"]:
        sig, cells, dead, filled, cols = _board_features(s.get("confirmed_board"))
        ic = s.get("icon") or {}
        sides.append(
            SideObs(
                score=s.get("score"), state=s.get("state"),
                formula_step=_formula_from_record(s.get("formula_step")),
                tsumo_count=s.get("tsumo_count"),
                next_slide=bool(s.get("next_slide_motion")),
                placed=s.get("side") in placed_sides,
                board_sig=sig, ojama_cells=cells, dead=dead,
                icon_total=ic.get("total_count"),
                icon_count=len(ic.get("icons") or []) if ic else None,
                filled_cells=filled, ojama_cols=cols,
                chain_info=s.get('chain_info') or s.get('chain'),
            )
        )
    return FrameObs(
        t_sec=float(rec["t_sec"]), frame=int(rec["frame"]), game_idx=int(rec.get("game_idx", 0)),
        match_active=bool(rec.get("match_active")), sides=(sides[0], sides[1]),
    )


def load_placed_frames(events_path: Path) -> Dict[int, Tuple[str, ...]]:
    """events.jsonl の placement_signal を frame -> 側名 に集める。"""
    out: Dict[int, List[str]] = {}
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            if '"placement_signal"' not in line:
                continue
            e = json.loads(line)
            out.setdefault(int(e["frame"]), []).append(e["side"])
    return {k: tuple(v) for k, v in out.items()}


def iter_record_frames(record_dir: Path) -> Iterator[FrameObs]:
    """B2 の record ディレクトリから FrameObs を順に返す。"""
    placed = load_placed_frames(record_dir / "events.jsonl")
    with gzip.open(record_dir / "observations.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "frame":
                yield frame_obs_from_record(rec, placed.get(int(rec["frame"]), ()))


# ============================ 学習行生成器との接続口 ============================
# src/b3_score_ledger_port.py が期待するインターフェース (ScoreLedger.update(frame: dict) -> dict | None)。

SL_SIDE_KEYS: Tuple[str, ...] = ("pending", "generated", "offset", "dropped", "carry")


def snapshot_to_dict(snap: LedgerSnapshot) -> Dict[str, object]:
    """LedgerSnapshot を接続口の契約 (キー名 *_p1/*_p2, rate, available_at) へ変換する。

    pending_pN = 側 N が受ける予告 (相殺後・上限なし)、generated_pN = 側 N が生成した累計、
    offset_pN = 側 N が自分の受け予告の相殺に使った累計、dropped_pN = 側 N へ着弾した累計、
    carry_pN = 側 N の換算前の端数得点。追加キーは品質・診断用 (契約外)。
    """
    incomplete = any(FLAG_HISTORY_INCOMPLETE in sl.flags for sl in snap.sides)
    waiting = any(sl.score is None for sl in snap.sides)
    out: Dict[str, object] = {"available_at": snap.available_at_sec, "rate": snap.rate,
                             "history_incomplete": incomplete, "history_waiting": waiting}
    for i, sl in enumerate(snap.sides, start=1):
        out[f"pending_p{i}"] = sl.pending
        out[f"generated_p{i}"] = sl.generated_total
        out[f"offset_p{i}"] = sl.offset_total
        out[f"dropped_p{i}"] = sl.dropped_total
        out[f"carry_p{i}"] = sl.carry
        out[f"pending_confirmed_p{i}"] = sl.pending_confirmed
        out[f"sent_p{i}"] = sl.sent_total
        out[f"score_p{i}"] = sl.score
        out[f"chain_open_p{i}"] = int(sl.chain_open)
        out[f"flags_p{i}"] = tuple(sorted(sl.flags))
        out[f"observed_landing_p{i}"] = sl.observed_landing
        out[f"board_increase_p{i}"] = sl.board_increase_total
        out[f"board_excess_p{i}"] = sl.board_excess_total
    out["ambiguous_p1"], out["ambiguous_p2"] = snap.ambiguous
    out["match_over"] = int(snap.match_over)
    out["margin_origin_sec"] = snap.margin_origin_sec
    out["match_reset"] = int(snap.match_reset)
    out["events"] = snap.events
    # 両側の相殺に依存するため、片側の履歴欠落でも両側の予告・超過を欠測にする。
    # 生台帳の累計は診断用に保存し、試合リセットまで推測で履歴を補完しない。
    if incomplete or waiting:
        for i in (1, 2):
            for key in ("pending", "pending_confirmed", "board_excess"):
                out[f"{key}_p{i}"] = float("nan")
    return out


class ScoreLedger:
    """接続口用の薄いラッパー。観測 1 行 (dict) を与えると契約どおりの dict を返す。

    試合外 (match_active=False) と、両側の得点がまだ読めていないフレームは None を返す
    (呼び出し側が mask にする)。置き終わり信号 (events.jsonl の placement_signal) を
    持っているなら placed_sides に同フレームの側名 ("1P"/"2P") を渡すと着弾判断が良くなる
    (無くても NEXT 移動・おじゃま落下状態で代替する)。
    """

    def __init__(self, **kwargs: object) -> None:
        self.core = ScoreLedgerV2(**kwargs)  # type: ignore[arg-type]

    @timed('ledger')
    def update(self, frame: dict, placed_sides: Sequence[str] = ()) -> Optional[Dict[str, object]]:
        """1 フレームを台帳へ流す。"""
        obs = frame_obs_from_record(frame, placed_sides)
        snap = self.core.update(obs)
        if not obs.match_active or all(s.score is None for s in obs.sides):
            return None
        return snapshot_to_dict(snap)
