"""撃ち合い後のフィールド予測の試作 (影ログ専用・状態を持たない関数)。

設計: <work>/POST_EXCHANGE_FIELD_DESIGN.md (追補・user の回答)。
入力は出来事 (連鎖の発火) 時点の状態、出力は「受け側が着弾までの手数の制約の中で作れる複数の応手
候補」と、それぞれの後で相手が返せるか。勝率には換算しない (数値だけを返す)。

user 伝授・回答 (2026-10-13) の反映
- 受け側は、着弾までの手数の制約の中で、複数パターンから最も有利な手を選ぶ。連鎖数の特定の組合せを
  前提にしない (2 連鎖・4 連鎖は例え)。候補は連鎖数ではなく「手数 h で作れる最大火力」の曲線から作る。
- 着地: 受け側が 1 手置くたびに最大 30 個 (5 段) ずつ降る。降る途中で打ち返すと、まだ降っていない
  分だけが相殺される。K 手目で着地が始まるので、h <= K で撃てば全量を相殺できる。
- 発火前の確定盤面から火力が分かる (連鎖中の盤面が見えなくても上限扱いにしない)。
- 本線が理想 3〜4 手で打てる形か、本線を重ねて相手が返せないか、おじゃまが多いほど不利。
- 手数 K は 2 本持つ: 期待値側 (観測した設置間隔の中央値で数える。迷いを含む) と楽観側 (最速の
  手時間で数える。死亡断定の根拠に使う)。

探索は既存の資産だけを使う (新しい探索器は書かない)
- 手数別の最大火力曲線: src.b3_search.ideal_fire_search (Rust、理想ツモ。h=1..20 を 1 回)。
- 既知ツモでの具体的な手順: src.puyo_core_bridge.beam_search (Rust)。盤面を返せるのはこちら。
- 受けて掘る: scripts.mc_counter_estimator._select_build_placement (Rust 融合)。
- 着地・相殺: src.exchange_virtual_board.land_pending_ojama_onto_board、
  src.exchange_event_landing.minimum_cancel。
- 隠し段の確率展開はしない (固まりの原因、LIVE_EVAL_HANG.md)。UNKNOWN は件数で打ち切る。

上限 (件数で決定的に切る。時間では切らない)
- 1 出来事あたりのネイティブ探索呼出は MAX_NATIVE_CALLS 回まで。超えたら以降の段を飛ばし、
  `capped` に理由を残す。既知ツモの深さは KNOWN_DEPTH_CAP、掘る手数は DIG_STEPS_CAP まで。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Optional

import numpy as np

from src.phase_j._live_ex_frozen.src import puyo_core_bridge as native
from src.phase_j._live_ex_frozen.src.b3_search import HMAX_TOTAL_HANDS, ideal_fire_search, points_to_ojama
from src.phase_j._live_ex_frozen.src.b3_timing import TimingTable, column_heights
from src.phase_j._live_ex_frozen.src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, COLOR_UNKNOWN, Board
from src.phase_j._live_ex_frozen.src.exchange_virtual_board import land_pending_ojama_onto_board
from src.phase_j._live_ex_frozen.src.indicators_v2 import estimate_chain_anim_duration_sec
from src.phase_j._live_ex_frozen.src.ojama_accounting import cancel_own_pending_then_send_surplus
from src.phase_j._live_ex_frozen.src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.phase_j._live_ex_frozen.src.scoring import OJAMA_MAX_DROP_PER_TURN

Pair = tuple[int, int]
CHAIN_ANIM_CALIBRATION = "empirical_table_2026_08_14"
MIN_HANDS = 1                  # 着地を起こす最後の 1 手 (b3_timeaxis.MIN_HANDS と同じ)
MAIN_HANDS_SHORT, MAIN_HANDS = 3, 4   # user: 本線が理想 3〜4 手で打てる形か
HIDDEN_ROW = 0
FIRST_COLOR, LAST_COLOR = 1, 5
MATCH_COLORS = 4              # 試合は 4 色のみ使用 (user 2026-07-22)
UNCANCELLABLE_OJAMA = 72       # 12 段分。これ以上は相殺不能とみなす (user 2026-10-05、特徴のみ)
# 探索幅 (段階表)。ideal_fire_search の幅は 4 (B3 と同じ) / 8 / 12 / 22 を 10/13 夜 night1 で実測した:
#   幅 4: P50 50ms P95 98ms 最大 146ms、誤った「返せない」2 件 / 幅 8: P50 97 P95 200 最大 268ms、1 件
#   幅 12: P50 141 P95 310 最大 418ms、1 件 / 幅 22: P50 270 P95 563 最大 779ms、1 件
# 低スペックは PC1 の 1.7〜3 倍と見て、P95 が 0.5 秒以内・最大が 1 秒以内に収まる幅 8 を既定にする。
# 既知ツモの具体探索は応手探索と同じ全配置 22。
IDEAL_BEAM_WIDTH = 8
KNOWN_BEAM_WIDTH = 2 * BOARD_COLS + 2 * (BOARD_COLS - 1)
STATUS_OK, STATUS_SKIPPED = "ok", "skipped"


@dataclass(frozen=True)
class SearchBudget:
    """件数上限と段階表。時間では切らない (決定的)。"""

    max_native_calls: int = 14     # 1 出来事あたりのネイティブ探索呼出の合計
    known_depth_cap: int = 4       # 既知ツモの具体探索の最大深さ
    dig_steps_cap: int = 4         # 受けて掘る手数の上限
    ideal_beam_width: int = IDEAL_BEAM_WIDTH   # 手数別最大火力曲線の探索幅 (B3 と同じ既定。広げると遅いが上限に近づく)
    stages: tuple[str, ...] = ("attack", "curve", "known", "dig", "opponent", "main")


DEFAULT_BUDGET = SearchBudget()


@dataclass(frozen=True)
class ExchangeInput:
    """出来事時点の状態。盤面は 13x6、おじゃま数は個、時間は秒。"""

    attacker: int                      # 0=1P, 1=2P (撃った側)
    attacker_board: np.ndarray         # 攻撃側の発火前の確定盤面 (火力はこれで分かる)
    receiver_board: np.ndarray         # 受け側の確定盤面
    pending_receiver: int              # 受け側への既存の予告 (この連鎖の分を含まない)
    pending_attacker: int              # 攻撃側への既存の予告 (この連鎖でまず相殺される)
    remaining_sec: float               # 連鎖の残り時間 N
    elapsed_sec: float                 # マージンタイムの起点 (最初の 1 手) からの経過
    known_receiver: tuple[Pair, ...]   # 受け側の既知ツモ (先頭 = NEXT)。ツモテープの結果
    known_attacker: tuple[Pair, ...]   # 攻撃側の既知ツモ (連鎖後に引くもの)
    colors: tuple[int, ...]            # 試合の使用色
    hand_table: Optional[TimingTable] = None
    receiver_busy_sec: float = 0.0     # 受け側が連鎖中で置けない残り時間
    expected_hand_sec: Optional[float] = None   # 較正した期待値側の手時間 (None=表の中央値)
    expected_table: Optional[TimingTable] = None   # 撃ち合いの窓で較正した高さ別の手時間表 (期待値側。None=hand_table)
    attack_counted: bool = False       # True なら pending_receiver に今回の連鎖の分がすでに入っている
    receiver_credit: int = 0           # 受け側が連鎖中なら、その連鎖がこれから生む未反映のおじゃま (自分の予告を先に相殺する)


@dataclass
class Candidate:
    """受け側の応手候補 1 件。mode 別 (期待値側 exp / 楽観側 opt) に作る。"""

    name: str                          # 'dig' | 'fastest' | 'max' | 'max_known'
    mode: str                          # 'exp' | 'opt'
    hands: Optional[int] = None        # 使う手数 (None=窓の中では届かない)
    basis: str = "none"                # 'known_tsumo' (既知ツモで具体的) | 'ideal_tsumo' (理想ツモ上限)
    chain_count: int = 0
    send: int = 0                      # 自連鎖が作るおじゃま (換算)
    cancel: int = 0
    surplus: int = 0                   # 相手へ送る余剰
    pending_after: int = 0             # 相殺後に受け側へ降る予告
    landed_dead: Optional[bool] = None # 降った後に窒息するか (最初の 1 回分。盤面が無い候補は None)
    known_steps: int = 0               # 既知ツモで具体的に置いた手数 (掘る・具体手順の長さ)
    late: bool = False                 # 着地が始まってから撃つ (降った分は相殺できない、盤面は近似しない)
    board_after: Optional[str] = None  # 着地後の盤面 (13 行を '/' 区切り)
    colors_used: Optional[int] = None  # 自連鎖で消した色ぷよ数 (具体手順だけ。理想ツモの上限は盤面が無いので None)
    opp_time_sec: float = 0.0          # 相手に渡す連鎖時間
    opp_hands: int = MIN_HANDS
    opp_max: int = 0                   # 相手がその時間で打てる最大火力 (1 段近似、おじゃま換算)
    opp_returns: Optional[bool] = None
    main_f3: Optional[int] = None      # 後の盤面から理想 3 手で打てる本線 (おじゃま換算)
    main_f4: Optional[int] = None
    main_chain4: Optional[int] = None
    stacked_send: Optional[int] = None         # 本線を重ねた時に相手へ送る合計
    stacked_opp_max: Optional[int] = None
    stacked_opp_returns: Optional[bool] = None
    margin: Optional[float] = None     # 受け側の有利を表す数値 (おじゃま個数。勝率ではない)


@dataclass
class ExchangeProjection:
    """出力。candidates は mode 別。best は margin が最大の候補名。"""

    status: str = STATUS_OK
    reason: str = ""
    attack_chain: int = 0
    attack_score: int = 0
    attack_send: int = 0
    incoming: int = 0                  # 受け側への予告 (既存 + 余剰)
    hands_exp: int = MIN_HANDS
    hands_opt: int = MIN_HANDS
    window_sec: float = 0.0
    hand_sec_exp: float = 0.0
    hand_sec_opt: float = 0.0
    required_cancel: Optional[int] = None
    max_cancel_opt: Optional[int] = None
    unreturnable_opt: Optional[bool] = None    # 楽観側 K・理想ツモ・着地後の遅い応手を含めても必要量に届かない (証明ではない)
    max_cancel_late_opt: Optional[int] = None  # 遅れて撃つ場合を含む、相殺できる最大量 (楽観側)
    max_send_late_opt: Optional[int] = None    # 窓の中 + 着地に耐えられる間の遅れ撃ちを含む、打てる最大火力 (楽観側・理想ツモ。上限の検証用)
    hidden_colors_dropped: int = 0             # 5 色目が隠し段だけにあったため取り除いたぷよ数
    receiver_dead_at_rest: bool = False        # 確定盤面が静止時の窒息条件 (連鎖で解消する可能性は残る)
    known_len: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    best_exp: Optional[str] = None
    best_opt: Optional[str] = None
    calls: dict[str, int] = field(default_factory=dict)
    capped: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


def board_text(grid: np.ndarray) -> str:
    """盤面を 13 行の文字列に ('.'=空、1-5=色、o=おじゃま、?=不明)。"""
    glyph = {0: ".", COLOR_OJAMA: "o", COLOR_UNKNOWN: "?"}
    return "/".join("".join(glyph.get(int(v), str(int(v))) for v in row) for row in grid)


def sanitize(grid: np.ndarray) -> tuple[Optional[np.ndarray], int]:
    """隠し段の UNKNOWN だけ空とみなす (展開しない)。可視段の UNKNOWN が残れば None。"""
    work = np.array(grid, dtype=np.int8).reshape(BOARD_ROWS, BOARD_COLS)
    work[HIDDEN_ROW][work[HIDDEN_ROW] == COLOR_UNKNOWN] = 0
    unknown = int(np.count_nonzero(work == COLOR_UNKNOWN))
    return (None if unknown else work), unknown


def drop_hidden_extra_colors(grids: tuple[np.ndarray, ...], limit: int = MATCH_COLORS
                             ) -> tuple[tuple[np.ndarray, ...], int]:
    """試合は 4 色のみ (user 2026-07-22)。2 盤面の色の和が 4 を超え、超過色が隠し段にしか無い時だけ、
    その隠し段のぷよを取り除く (10/13 夜 g16 は受け側の隠し段の 1 個が 5 色目になり、探索全体が
    色矛盾で欠測になった)。可視段に超過色が残る場合は何もしない (呼出側が色矛盾として扱う)。

    返り値は (盤面, 取り除いた個数)。
    """
    colors = {int(c) for g in grids for c in np.unique(g) if FIRST_COLOR <= int(c) <= LAST_COLOR}
    if len(colors) <= limit:
        return grids, 0
    visible = {int(c) for g in grids for c in np.unique(g[HIDDEN_ROW + 1:]) if FIRST_COLOR <= int(c) <= LAST_COLOR}
    if len(visible) > limit:
        return grids, 0
    cleaned, dropped = [], 0
    for g in grids:
        work = np.array(g)
        mask = (work[HIDDEN_ROW] >= FIRST_COLOR) & (work[HIDDEN_ROW] <= LAST_COLOR) & ~np.isin(work[HIDDEN_ROW], list(visible))
        dropped += int(mask.sum())
        work[HIDDEN_ROW][mask] = 0
        cleaned.append(work)
    return tuple(cleaned), dropped


def fire_with_pair(grid: np.ndarray, pair: Pair) -> Optional[np.ndarray]:
    """発火前の確定盤面に、発火した組を置いて最大得点で連鎖する配置の設置直後盤面を返す。

    確定盤面は最後の設置を含まないことがある (10/13 夜 g16: 追跡側の予測連鎖数が 0 の発火が 3 件)。
    実際に置いた配置は分からないので、その組で作れる最大火力の配置を上限として使う。連鎖しない
    なら None。
    """
    best, best_score = None, 0
    for _col, _rotation, placed in native.enumerate_placements(_board(grid), pair, filter_dead=False):
        result = native.simulate_chain(placed, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
        if result.chain_count and result.exact_score > best_score:
            best, best_score = placed._grid.copy(), result.exact_score
    return best


def _board(grid: np.ndarray) -> Board:
    board = Board()
    board._grid = np.array(grid, dtype=board._grid.dtype)
    return board


def hands_for(window_sec: float, hand_sec: float) -> int:
    """窓の中で置ける手数 + 着地を起こす最後の 1 手。"""
    return math.floor(max(window_sec, 0.0) / hand_sec) + MIN_HANDS


def hand_secs(table: Optional[TimingTable], grid: np.ndarray, expected_override: Optional[float] = None,
              expected_table: Optional[TimingTable] = None) -> tuple[float, float]:
    """(期待値側の手時間, 楽観側 = 最速の手時間)。盤面の最大高さで表を引く。

    期待値側は、較正した手時間 (スカラー) > 較正した表 > 通常の表の中央値、の順に使う。
    """
    table = table if table is not None else TimingTable()
    height = float(column_heights(grid).max())
    median, fast, _key = table.hand_sec(height)
    if expected_table is not None:
        median = expected_table.hand_sec(height)[0]
    return (expected_override if expected_override else median), fast


def cancellable(pending: int, hands: int, landing_hand: int) -> int:
    """h 手目で撃った時に相殺できる量。landing_hand 手目から 1 手ごとに最大 30 個が降る。"""
    landed = min(pending, OJAMA_MAX_DROP_PER_TURN * max(0, hands - landing_hand))
    return pending - landed


class _Counter:
    """ネイティブ探索呼出の件数管理 (件数で決定的に切る)。"""

    def __init__(self, budget: SearchBudget) -> None:
        self.budget, self.calls, self.capped = budget, {}, []

    def total(self) -> int:
        return sum(self.calls.values())

    def take(self, kind: str) -> bool:
        if self.total() >= self.budget.max_native_calls:
            if "native_calls" not in self.capped:
                self.capped.append("native_calls")
            return False
        self.calls[kind] = self.calls.get(kind, 0) + 1
        return True


def _attack(inp: ExchangeInput) -> tuple[native.ChainSimResult, int]:
    """発火前の盤面から連鎖を完走し、(結果, おじゃま換算) を返す。"""
    result = native.simulate_chain(_board(inp.attacker_board),
                                   exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    return result, points_to_ojama(result.exact_score, inp.elapsed_sec)


def _flat(pairs: tuple[Pair, ...], count: int = 2) -> tuple[int, ...]:
    """ideal_fire_search の queue 形式 (先頭 2 組を平らに)。足りない分は 0 (未知)。"""
    flat: list[int] = []
    for pair in pairs[:count]:
        flat.extend(pair)
    return tuple(flat)


def _curve(grid: np.ndarray, pairs: tuple[Pair, ...], colors: tuple[int, ...], hmax: int,
           counter: _Counter, kind: str) -> Optional[tuple[list[int], list[int]]]:
    """理想ツモでの手数別最大火力 (おじゃま換算は呼出側)。(得点列, 連鎖数列)。"""
    if not counter.take(kind):
        return None
    result = ideal_fire_search(grid, _flat(pairs), colors, min(hmax, HMAX_TOTAL_HANDS),
                               beam_width=counter.budget.ideal_beam_width)
    if result.color_conflict:
        return None
    return list(result.scores), list(result.chains)


def _replay(board: Board, pairs: tuple[Pair, ...], path: list[tuple[int, int]], target_score: int
            ) -> Optional[tuple[native.ChainSimResult, Board, int]]:
    """ビームの最良手順を再生し、最良得点に初めて届く設置で止める。(連鎖結果, 連鎖後の盤面, 設置数)。

    ビームは最良得点を出した後も手順を伸ばすので、手順の末尾まで置いてはいけない。
    """
    current = board
    for steps, (pair, (col, rotation)) in enumerate(zip(pairs, path), 1):
        placed = next((b for c, r, b in native.enumerate_placements(current, pair, filter_dead=False)
                       if c == col and r == rotation), None)
        if placed is None:
            return None
        result = native.simulate_chain(placed, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
        if result.chain_count and int(result.exact_score) == int(target_score):
            return result, result.final_board, steps
        current = result.final_board if result.chain_count else placed
    return None


def _known_search(grid: np.ndarray, pairs: tuple[Pair, ...], depth: int, counter: _Counter,
                  kind: str) -> Optional[native.BeamSearchResult]:
    """既知ツモだけでの具体的な最良手順 (ビーム)。深さは件数で打ち切る。"""
    depth = min(depth, len(pairs), counter.budget.known_depth_cap)
    if depth < 1 or not counter.take(kind):
        return None
    return native.beam_search(_board(grid), list(pairs[:depth]), KNOWN_BEAM_WIDTH,
                              exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
                              use_exact_score=True)


def _dig(grid: np.ndarray, pairs: tuple[Pair, ...], steps: int, counter: _Counter) -> list[Board]:
    """受けて掘る: 消さずに積む配置を既知ツモの範囲で繰り返す (既存の組む配置選択)。

    返り値は「0 手後 (初期盤面)・1 手後・...」の盤面列。手数の違う 2 つの K で 1 回の探索を共有する。
    """
    from scripts.mc_counter_estimator import _SHARED_SIMULATOR, _select_build_placement
    boards = [_board(grid)]
    for pair in pairs[:min(steps, counter.budget.dig_steps_cap)]:
        if not counter.take("build"):
            break
        placed = _select_build_placement(boards[-1], pair, _SHARED_SIMULATOR, True)
        if placed is None:
            break
        boards.append(placed)
    return boards


class _Work:
    """1 出来事の作業領域。探索結果を出来事の中でだけ共有する (出来事をまたいで持たない)。"""

    def __init__(self, inp: ExchangeInput, budget: SearchBudget) -> None:
        self.inp, self.counter = inp, _Counter(budget)
        self.curves: dict[tuple, Optional[tuple[list[int], list[int]]]] = {}
        self.known: dict[int, Optional[native.BeamSearchResult]] = {}
        self._survivable: Optional[int] = None
        self.receiver = np.asarray(inp.receiver_board, dtype=np.int8)
        self.attack_result, self.attack_send = _attack(inp)
        self.attacker_after = self.attack_result.final_board
        cancelled, surplus = cancel_own_pending_then_send_surplus(
            self.attack_send, inp.pending_attacker, 0)
        self.attack_surplus = surplus
        base_incoming = inp.pending_receiver + (0 if inp.attack_counted else surplus)
        self.incoming = max(0, base_incoming - inp.receiver_credit)
        self.credit_surplus = max(0, inp.receiver_credit - base_incoming)   # 相殺しきれず相手へ送られる分
        self.hand_exp, self.hand_opt = hand_secs(inp.hand_table, self.receiver, inp.expected_hand_sec,
                                                   inp.expected_table)
        self.window = max(inp.remaining_sec - inp.receiver_busy_sec, 0.0)
        self.k = {"exp": hands_for(self.window, self.hand_exp), "opt": hands_for(self.window, self.hand_opt)}
        self.opp_hand = dict(zip(("exp", "opt"), hand_secs(inp.hand_table, self.attacker_after._grid)))

    def curve(self, grid: np.ndarray, pairs: tuple[Pair, ...], hmax: int, kind: str):
        """同じ盤面・先頭ツモ・手数の曲線は出来事の中で 1 回だけ探索する。"""
        key = (grid.tobytes(), _flat(pairs), hmax)
        if key not in self.curves:
            self.curves[key] = _curve(grid, pairs, self.inp.colors, hmax, self.counter, kind)
        return self.curves[key]

    def ojama(self, score: float) -> int:
        return points_to_ojama(score, self.inp.elapsed_sec)

    def known_run(self, depth: int):
        depth = min(depth, len(self.inp.known_receiver), self.counter.budget.known_depth_cap)
        if depth not in self.known:
            self.known[depth] = _known_search(self.receiver, self.inp.known_receiver, depth,
                                              self.counter, "known")
        return depth, self.known[depth]

    def survivable(self) -> int:
        """K 手目から 1 手ごとに 30 個ずつ降る着弾に耐えられる回数 (出来事の中で 1 回だけ数える)。"""
        if self._survivable is None:
            self._survivable = _survivable_chunks(self, HMAX_TOTAL_HANDS)
        return self._survivable

    def opp_hands(self, mode: str, seconds: float) -> int:
        return hands_for(seconds, self.opp_hand[mode])

    def opp_max(self, mode: str, seconds: float) -> int:
        """相手 (攻撃側) が連鎖後の盤面から、渡された時間で打てる最大火力 (1 段近似)。"""
        curve = self.curve(self.attacker_after._grid, self.inp.known_attacker, HMAX_TOTAL_HANDS, "opp")
        if curve is None:
            return 0
        hands = min(self.opp_hands(mode, seconds), len(curve[0]))
        return self.ojama(curve[0][hands - 1])


def _anim(chain_count: int) -> float:
    return estimate_chain_anim_duration_sec(chain_count, CHAIN_ANIM_CALIBRATION)


def _main_potential(work: _Work, grid: np.ndarray, used: int) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """後の盤面から理想 3 手・4 手で打てる本線 (おじゃま換算 f3, f4 と 4 手の連鎖数)。"""
    pairs = work.inp.known_receiver[used:]
    curve = work.curve(grid, pairs, MAIN_HANDS, "main")
    if curve is None:
        return None, None, None
    scores, chains = curve
    return (work.ojama(scores[MAIN_HANDS_SHORT - 1]), work.ojama(scores[MAIN_HANDS - 1]),
            chains[MAIN_HANDS - 1])


def _landed(work: _Work, grid: np.ndarray, pending: int) -> tuple[Board, bool]:
    """受け側の盤面へ残りの予告の最初の降り (最大 30 個) を載せ、窒息か返す。"""
    board = _board(grid)
    landed, _dropped, _left = land_pending_ojama_onto_board(board, work.attacker_after, pending)
    return landed, bool(landed.is_dead())


def _fill(work: _Work, cand: Candidate, grid: Optional[np.ndarray], used: int) -> Candidate:
    """候補の相殺・着地・相手が渡された時間・本線の重ねを数値にする (共通部)。"""
    mode = cand.mode
    hands = cand.hands if cand.hands is not None else work.k[mode]
    cand.cancel = min(cand.send, cancellable(work.incoming, hands, work.k[mode]))
    cand.surplus = cand.send - cand.cancel + work.credit_surplus
    cand.pending_after = work.incoming - cand.cancel
    cand.opp_time_sec = _anim(cand.chain_count) if cand.send > 0 else 0.0
    cand.opp_hands = work.opp_hands(mode, cand.opp_time_sec)
    cand.opp_max = work.opp_max(mode, cand.opp_time_sec)
    cand.opp_returns = cand.opp_max >= cand.surplus
    margin = (cand.surplus - cand.opp_max) - cand.pending_after
    if grid is not None:
        landed, cand.landed_dead = _landed(work, grid, cand.pending_after)
        cand.board_after = board_text(landed._grid)
        cand.main_f3, cand.main_f4, cand.main_chain4 = _main_potential(work, landed._grid, used)
        if cand.main_f4 is not None:
            cand.stacked_send = cand.surplus + cand.main_f4
            stacked_time = cand.opp_time_sec + _anim(cand.main_chain4 or 0)
            cand.stacked_opp_max = work.opp_max(mode, stacked_time)
            cand.stacked_opp_returns = cand.stacked_opp_max >= cand.stacked_send
            margin = (cand.stacked_send - cand.stacked_opp_max) - cand.pending_after
    cand.margin = float(margin)
    return cand


def _concrete(work: _Work, cand: Candidate, depth: int, run: native.BeamSearchResult) -> Optional[np.ndarray]:
    """既知ツモの具体的な最良手順を候補へ写す。再生した得点がビームの値と違えば取り下げる。

    最良得点が 0 (深さの中で連鎖が起きない) の時は具体手順を作らない (呼出側が理想曲線へ落とす)。
    """
    if run.best_score <= 0:
        return None
    pairs = work.inp.known_receiver[:depth]
    replay = _replay(_board(work.receiver), pairs, list(run.best_path), int(run.best_score))
    if replay is None:
        work.counter.capped.append("replay_mismatch")
        return None
    result, board, steps = replay
    cand.basis, cand.known_steps, cand.hands = "known_tsumo", steps, steps
    cand.send, cand.chain_count = work.ojama(run.best_score), int(result.chain_count)
    cand.colors_used = int(result.total_erased)
    return board._grid


def _ideal(work: _Work, cand: Candidate, hands: int) -> None:
    """理想ツモの上限曲線から数値だけを写す (盤面は作れない)。"""
    curve = work.curve(work.receiver, work.inp.known_receiver, HMAX_TOTAL_HANDS, "curve")
    if curve is None:
        return
    h = min(hands, len(curve[0]))
    cand.basis, cand.send, cand.chain_count = "ideal_tsumo", work.ojama(curve[0][h - 1]), curve[1][h - 1]


def _max_candidates(work: _Work, mode: str) -> list[Candidate]:
    """K 手で作れる最大火力 (R2)。K が既知ツモの範囲なら具体手順、超えるなら理想ツモの上限。

    K が既知ツモを超える場合は、理想ツモの上限 (数値のみ) に加えて、既知ツモの範囲で作れる具体手順
    (max_known、盤面あり) も別候補として返す。
    """
    hands = work.k[mode]
    cand = Candidate("max", mode, hands=hands)
    depth, run = work.known_run(hands)
    out, grid = [cand], None
    if run is not None and depth >= hands:
        grid = _concrete(work, cand, depth, run)
    elif run is not None:
        known = Candidate("max_known", mode, hands=depth)
        known_grid = _concrete(work, known, depth, run)
        if known.basis == "known_tsumo":
            out.append(_fill(work, known, known_grid, known.known_steps))
    if cand.basis == "none":
        _ideal(work, cand, hands)
    _fill(work, cand, grid, cand.known_steps)
    return out


def _fastest_hands(work: _Work, required: int) -> Optional[int]:
    """理想ツモの曲線で、必要量に最初に届く手数 (届かなければ None)。"""
    curve = work.curve(work.receiver, work.inp.known_receiver, HMAX_TOTAL_HANDS, "curve")
    if curve is None or required <= 0:
        return None
    return next((h for h, score in enumerate(curve[0], 1) if work.ojama(score) >= required), None)


def _fastest_candidate(work: _Work, mode: str, hands: Optional[int]) -> Candidate:
    """相殺に必要な量を最短の手数で作る (R1)。届かない・間に合わない場合は late/None で表す。"""
    cand = Candidate("fastest", mode, hands=hands)
    if hands is None:
        return _fill(work, cand, None, 0)
    cand.late = hands > work.k[mode]
    if cand.late and hands > work.k[mode] + work.survivable():
        cand.basis = "dead_before_fire"      # 撃つ前に着弾で窒息する (遅れて撃つ手は成立しない)
        cand.hands = None
        return _fill(work, cand, None, 0)
    depth, run = work.known_run(hands)
    grid = None
    if run is not None and depth >= hands:
        grid = _concrete(work, cand, depth, run)
    if cand.basis == "none":
        _ideal(work, cand, hands)
    return _fill(work, cand, None if cand.late else grid, cand.known_steps)


def _dig_candidate(work: _Work, mode: str, boards: list[Board]) -> Candidate:
    """受けて掘る (R0)。既知ツモの範囲で積み、残りの手は盤面を変えないものとして数える。"""
    steps = min(work.k[mode], len(boards) - 1)
    cand = Candidate("dig", mode, hands=work.k[mode], basis="known_tsumo", known_steps=steps, colors_used=0)
    return _fill(work, cand, boards[steps]._grid, steps)


def _required(work: _Work) -> int:
    """既存の死亡判定と同じ「生存に必要な相殺量」(着地の最初の 1 回で窒息しない最小の相殺)。"""
    from src.phase_j._live_ex_frozen.src.exchange_event_landing import minimum_cancel
    return int(minimum_cancel(_board(work.receiver), work.attacker_after, work.incoming))


def _returnable(work: _Work, scores: list[int], required: int) -> bool:
    """楽観側 K・理想ツモで、h 手目に撃てば必要量を相殺できる h があるか。

    K 手目に着地が始まり、1 手ごとに最大 30 個降るので、遅れて撃つほど必要な相殺量は減る
    (降った分は消せないが、残りの必要量も降った分だけ減る)。遅れて撃てるのは、降る着弾に耐えている間
    (survivable) だけ。置いた手による盤面の変化・連鎖で盤面が空く可能性は見ない (楽観側)。
    証明ではない: 探索幅・理想ツモ・窓の見積りの範囲で「届かない」というだけ。
    """
    k = work.k["opt"]
    limit = k + work.survivable()
    for hands, score in enumerate(scores[:limit], 1):
        need = max(0, required - OJAMA_MAX_DROP_PER_TURN * max(0, hands - k))
        if work.ojama(score) >= need:
            return True
    return False


def _survivable_chunks(work: _Work, cap: int) -> int:
    """K 手目から 1 手ごとに最大 30 個ずつ降るとき、窒息せずに耐えられる回数 (予告が尽きるか cap で打ち切り)。

    窒息した後に撃つことはできないので、その手数以降の「遅れて撃つ」は成立しない。g16 (1P 66/72 マス) の
    ように最初の 1 回の着弾で窒息する盤面は 0 になる。降るパターンは既存の着弾 (決定的シード) を使う。
    """
    board, remaining, chunks = _board(work.receiver), work.incoming, 0
    while remaining > 0 and chunks < cap:
        board, dropped, _left = land_pending_ojama_onto_board(board, work.attacker_after, remaining)
        if dropped <= 0 or board.is_dead():
            return chunks
        remaining -= dropped
        chunks += 1
    return cap


def _choose(cands: list[Candidate]) -> Optional[str]:
    """margin 最大の候補名。同点は降る予告が少ない方、次に名前順 (決定的)。"""
    valid = [c for c in cands if c.margin is not None]
    if not valid:
        return None
    return min(valid, key=lambda c: (-c.margin, c.pending_after, c.name)).name  # type: ignore[operator]


def _build_candidates(work: _Work, out: ExchangeProjection) -> None:
    stages = work.counter.budget.stages
    required = _required(work)
    out.required_cancel = required
    curve = work.curve(work.receiver, work.inp.known_receiver, HMAX_TOTAL_HANDS, "curve") \
        if "curve" in stages else None
    if curve is not None:
        top = max(work.ojama(s) for s in curve[0][:min(work.k["opt"], len(curve[0]))])
        late = max(min(work.ojama(score), cancellable(work.incoming, h, work.k["opt"]))
                   for h, score in enumerate(curve[0], 1))
        out.max_cancel_opt, out.max_cancel_late_opt = top, late
        reach = work.k["opt"] + work.survivable()
        out.max_send_late_opt = max(work.ojama(score) for score in curve[0][:max(1, reach)])
        out.unreturnable_opt = not _returnable(work, curve[0], required)
    fastest = _fastest_hands(work, required) if "curve" in stages else None
    boards = (_dig(work.receiver, work.inp.known_receiver, max(work.k.values()), work.counter)
              if "dig" in stages and not out.receiver_dead_at_rest else None)
    for mode in ("exp", "opt"):
        if "known" in stages or "curve" in stages:
            out.candidates.extend(_max_candidates(work, mode))
            out.candidates.append(_fastest_candidate(work, mode, fastest))
        if boards is not None:
            out.candidates.append(_dig_candidate(work, mode, boards))


def project_exchange(inp: ExchangeInput, budget: SearchBudget = DEFAULT_BUDGET) -> ExchangeProjection:
    """1 出来事 (連鎖の発火) について、受け側の応手候補と相手が返せるかを数値で返す。

    状態を持たない (引数を破壊しない)。失敗は例外にせず status/reason で返し、評価を止めない。
    """
    started = time.perf_counter()
    out = ExchangeProjection(known_len=len(inp.known_receiver))
    receiver, attacker = sanitize(inp.receiver_board)[0], sanitize(inp.attacker_board)[0]
    if receiver is None or attacker is None:
        out.status, out.reason = STATUS_SKIPPED, "unknown_cells"
    else:
        (receiver, attacker), out.hidden_colors_dropped = drop_hidden_extra_colors((receiver, attacker))
        work = _Work(_replace_boards(inp, receiver, attacker), budget)
        _summarize(work, out)
        if work.incoming <= 0:
            out.status, out.reason = STATUS_SKIPPED, "no_incoming"
        else:
            out.receiver_dead_at_rest = bool(_board(receiver).is_dead())
            _build_candidates(work, out)
            for mode in ("exp", "opt"):
                best = _choose([c for c in out.candidates if c.mode == mode])
                setattr(out, "best_" + mode, best)
        out.calls, out.capped = dict(work.counter.calls), list(work.counter.capped)
    out.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return out


def _replace_boards(inp: ExchangeInput, receiver: np.ndarray, attacker: np.ndarray) -> ExchangeInput:
    from dataclasses import replace
    return replace(inp, receiver_board=receiver, attacker_board=attacker)


def _summarize(work: _Work, out: ExchangeProjection) -> None:
    out.attack_chain, out.attack_score = int(work.attack_result.chain_count), int(work.attack_result.exact_score)
    out.attack_send, out.incoming = work.attack_send, work.incoming
    out.hands_exp, out.hands_opt, out.window_sec = work.k["exp"], work.k["opt"], work.window
    out.hand_sec_exp, out.hand_sec_opt = work.hand_exp, work.hand_opt
