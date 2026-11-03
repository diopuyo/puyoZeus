"""撃ち合い後のフィールド予測の影ログ (外部ラッパー。表示・勝率・既存の記録には一切つながない)。

役割
- 毎コマの通知 (認識結果・おじゃま会計・時刻・試合番号) を観測し、連鎖の発火・対応・本線・着弾の
  出来事ごとに `project_exchange` を呼んで、結果を JSONL の行として sink へ渡す。
- 状態はここ (共通ツモテープ・試合内の色・出来事の履歴) に持ち、`project_exchange` は状態を持たない。
- 入力を変更しない (盤面は複製して渡す)。例外は握りつぶさず `error` 行として記録し、評価は止めない。

出来事の種類 (連鎖数に依存しない一般形)
- fire     : 撃ち合いの最初の連鎖。
- counter  : その撃ち合いで、撃たれた側が最初に撃った連鎖 (応手)。
- main     : すでにその撃ち合いで撃った側が、さらに撃った連鎖 (本線など)。
- placement: 窓が開いている間に、受け側が 1 手置くたびの投影し直し (残り手数・盤面・予告が変わる)。
- landing  : 予告おじゃまが実際に降った (おじゃま会計の降下累計が増えた)。実測の記録だけで投影はしない。

連鎖の「1 件」は、側の状態が連鎖状態 (CHAIN / GRAVITY_SETTLE) に入ってから抜けるまで。途中で
chain_event が断片に分かれて再通知されても、1 件として扱う (10/13 夜 g16 の 13 連鎖は 4 つの断片に
分かれて通知されていた)。chain_event の連鎖数・得点は読み取りの途中経過なので火力には使わず、
発火前の確定盤面 (before_board) を完走させて求める (user 伝授 10/13)。

行の形式は `exchange_projection_shadow_v1`。フレーム番号 (OBSERVATION_FPS で秒から換算)・時刻・
攻撃側/受け側の別を全行に持たせ、後からレビュー動画に重ねられるようにする
(`review_columns` が平坦な列へ展開する)。
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src import puyo_core_bridge as native
from src.b3_search import points_to_ojama
from src.b3_timing import TimingTable
from src.board import Board
from src.exchange_event_tracker import EXCHANGE_IDLE_TIMEOUT_SEC, OBSERVATION_FPS, SIDE_LABELS
from src.exchange_projection import (CHAIN_ANIM_CALIBRATION, DEFAULT_BUDGET, ExchangeInput,
                                     SearchBudget, board_text, fire_with_pair, project_exchange)
from src.exchange_tsumo_tape import TsumoTape
from src.indicators_v2 import estimate_chain_anim_duration_sec
from src.production_config import GHOST_CHAIN_RULE_ENABLED

SCHEMA = "exchange_projection_shadow_v1"
SNAPSHOT_RING_SEC = 6.0        # 発火直前の会計値を引くために保持する秒数
MATCH_COLORS = 4               # 試合は 4 色のみ使用 (user 2026-07-22)
FIRST_COLOR, LAST_COLOR = 1, 5
HIDDEN_ROW = 0
PAIR_CELLS = 2                 # 1 手で増える色ぷよの数
STABLE_STATE = "STABLE"
CHAIN_STATES = ("CHAIN", "GRAVITY_SETTLE")
# 連鎖の途中でおじゃまが降ることがある (10/13 夜 g16 の 1P: 連鎖中に OJAMA_FALL を挟んで再開)。
# 操作が戻る (STABLE / TSUMO_FALL) まで連鎖の 1 件として扱う。
IN_CHAIN_STATES = CHAIN_STATES + ("OJAMA_FALL",)
LEDGER_GRACE_SEC = 1.0         # 発火から台帳の予告が反映されるまでの猶予 (この間は窓を閉じない)
Sink = Callable[[dict], None]


def frame_of(t_sec: float) -> int:
    """秒 → フレーム番号 (撃ち合い追跡と同じ 30fps 換算)。"""
    return int(round(t_sec * OBSERVATION_FPS))


def _board(grid: np.ndarray) -> Board:
    board = Board()
    board._grid = np.array(grid, dtype=board._grid.dtype)
    return board


def pending_pair(snapshot: Any) -> tuple[int, int]:
    """(1P への予告, 2P への予告)。保存入力には forecast_p1 と net_balance_capped しかない。"""
    if hasattr(snapshot, "pending_p2") and hasattr(snapshot, "pending_p1"):
        return int(snapshot.pending_p1), int(snapshot.pending_p2)
    first = int(snapshot.forecast_p1)
    cap = int(getattr(snapshot, "net_balance_capped", 0))
    return first, max(0, cap + first)      # net = p2_capped - p1_capped (1P 側が上限 72 内の時の復元)


class _Event:
    """連鎖の発火 1 件 (連鎖の始まりから終わりまで) の追跡。結果の実測と再投影に使う。"""

    def __init__(self, event_id: int, side: int, trigger: float, before: np.ndarray, pred_count: int,
                 dropped: tuple[int, int], pending_before: tuple[int, int]) -> None:
        self.event_id, self.side, self.trigger, self.before = event_id, side, trigger, before
        self.pred_count, self.dropped, self.pending_before = pred_count, dropped, pending_before
        self.total_sec = estimate_chain_anim_duration_sec(pred_count, CHAIN_ANIM_CALIBRATION)
        self.surplus = 0                      # 最初の投影で分かる、受け側へ送られる余剰
        self.pred_index, self.pred_pairs = 0, []
        self.landing_t: Optional[float] = None
        self.response: Optional[dict] = None
        self.attacker_free_t: Optional[float] = None
        self.outcome_done = False
        self.window_open = True
        self.projections = 0
        self.fire_basis = "before_board"
        self.chain_end_t: Optional[float] = None


class ExchangeProjectionShadow:
    """影ログの外部ラッパー。`observe()` を毎コマ呼ぶ。"""

    def __init__(self, sink: Sink, hand_table: Optional[TimingTable] = None,
                 budget: SearchBudget = DEFAULT_BUDGET, expected_hand_sec: Optional[float] = None,
                 project: Callable[[ExchangeInput, SearchBudget], Any] = project_exchange,
                 reproject: bool = True, expected_table: Optional[TimingTable] = None,
                 pending_fn: Optional[Callable[[Any, float, int], tuple[int, int]]] = None) -> None:
        """pending_fn(snapshot, t_sec, game_idx) -> (1P への予告, 2P への予告)。省略時は snapshot から読む。
        本番では B3 の得点台帳の予告を渡す (保存入力のおじゃま会計は台帳の予告と食い違う)。"""
        self._custom_pending = pending_fn is not None
        self.pending_fn = pending_fn or (lambda snapshot, _t, _g: pending_pair(snapshot))
        self.sink, self.table, self.budget = sink, hand_table, budget
        self.expected_hand_sec, self.project, self.reproject = expected_hand_sec, project, reproject
        self.expected_table = expected_table
        self.tape = TsumoTape()
        self.event_count = 0
        self.game: Optional[int] = None
        self._reset_game()

    def _reset_game(self) -> None:
        self.tape.reset()
        self.last_stable: list[Optional[np.ndarray]] = [None, None]
        self.ring: deque = deque()
        self.active = [False, False]          # 連鎖中 (発火から終了まで)
        self.latest_chain: list[Any] = [None, None]
        self.states = ["", ""]
        self.events: list[_Event] = []
        self.color_counts: Counter = Counter()
        self.exchange_id, self.exchange_sides, self.last_activity = 0, set(), -math.inf
        self.first_t: Optional[float] = None
        self.dropped = (0, 0)
        self.shifted = [False, False]
        self.pending = (0, 0)
        self.prev_states = ["", ""]
        self.fire_board: list[Optional[np.ndarray]] = [None, None]
        self.latest_event: list[Optional[_Event]] = [None, None]
        self.scrubbed = 0
        self.fall_start = [-math.inf, -math.inf]   # 側ごとの、OJAMA_FALL に入った時刻
        self.cells_at_shift = [0, 0]           # 側ごとの、NEXT が繰り上がった時点の色ぷよ数 (操作中の組を置き済みか見る)
        self.receiver_basis = "stable_confirmed"

    # ---- 公開 ----
    def observe(self, result: Any, snapshot: Any, t_sec: float, game_idx: int) -> None:
        """1 コマの通知を観測する。失敗は error 行にして継続する。"""
        try:
            if game_idx != self.game:
                self.finish_game(t_sec)
                self.game = game_idx
            self._observe(result, snapshot, t_sec, game_idx)
        except Exception as error:  # noqa: BLE001 影ログは評価を止めない。握りつぶさず行に残す
            self.sink(dict(schema=SCHEMA, kind="error", t_sec=t_sec, frame_idx=frame_of(t_sec),
                           game_idx=game_idx, error=type(error).__name__, message=str(error)[:300]))

    def finish_game(self, t_sec: float) -> None:
        """試合境界: 未確定の出来事の結果とツモテープの的中を確定して行にする。"""
        for event in self.events:
            if not event.outcome_done:
                self._close(event, t_sec, "game_end")
            self._tape_check(event, t_sec)
            self.sink(dict(schema=SCHEMA, kind="event_final", event_id=event.event_id, trigger_sec=event.trigger,
                           chain_end_sec=event.chain_end_t, fire_basis=event.fire_basis,
                           attacker=SIDE_LABELS[event.side], projections=event.projections))
        self._reset_game()

    # ---- 観測 ----
    def _observe(self, result: Any, snapshot: Any, t_sec: float, game_idx: int) -> None:
        sides = (result.p1, result.p2)
        if self.first_t is None:
            self.first_t = t_sec
        self.prev_states = list(self.states)
        for i, side in enumerate(sides):
            self.states[i] = getattr(getattr(side, "state", None), "name", "")
            self.shifted[i] = self.tape.observe(i, t_sec, side.next_pair, side.dnext_pair,
                                                bool(getattr(side, "next_slide_motion", False)))
            self._remember(i, side)
            if self.shifted[i] and self.last_stable[i] is not None:
                self.cells_at_shift[i] = self._color_cells(self.last_stable[i])
        for i in range(2):
            if self.states[i] == "OJAMA_FALL" and self.prev_states[i] != "OJAMA_FALL":
                self.fall_start[i] = t_sec
        self.pending = self.pending_fn(snapshot, t_sec, game_idx)
        self.ring.append((t_sec, self.pending))
        while self.ring and t_sec - self.ring[0][0] > SNAPSHOT_RING_SEC:
            self.ring.popleft()
        dropped = (int(snapshot.total_dropped_to_p1), int(snapshot.total_dropped_to_p2))
        for i, side in enumerate(sides):
            self._episode(i, getattr(side, "chain_event", None), t_sec, game_idx, dropped)
        if any(s in CHAIN_STATES or s == "OJAMA_FALL" for s in self.states):
            self.last_activity = t_sec
        self._on_landing(t_sec, game_idx)
        self._track(t_sec, game_idx)

    def _remember(self, i: int, side: Any) -> None:
        """STABLE の確定盤面だけを保持し、試合の使用色を数える (NEXT も含む)。"""
        board = getattr(side, "confirmed_board", None)
        if board is not None and self.states[i] == STABLE_STATE:
            grid = np.array(board._grid)
            self.last_stable[i] = grid
            for color in range(FIRST_COLOR, LAST_COLOR + 1):
                self.color_counts[color] += int(np.count_nonzero(grid[HIDDEN_ROW + 1:] == color))
        for pair in (side.next_pair, side.dnext_pair):
            if pair is not None:
                self.color_counts.update(int(v) for v in pair if FIRST_COLOR <= int(v) <= LAST_COLOR)

    def _scrub(self, grid: np.ndarray) -> np.ndarray:
        """隠し段にだけある、試合の 4 色の外のぷよを取り除く (隠し段の色は読み取りが不安定。
        連鎖後に落ちてきて 5 色目になり、探索全体が色矛盾で欠測になるため)。"""
        work = np.array(grid)
        colors = self._colors()
        if len(colors) < MATCH_COLORS:
            return work
        row = work[HIDDEN_ROW]
        bad = (row >= FIRST_COLOR) & (row <= LAST_COLOR) & ~np.isin(row, colors)
        self.scrubbed += int(bad.sum())
        row[bad] = 0
        return work

    def _colors(self) -> tuple[int, ...]:
        """試合は 4 色。観測数の多い 4 色 (4 色に満たなければ観測した分だけ)。"""
        return tuple(sorted(c for c, _n in self.color_counts.most_common(MATCH_COLORS)))

    def _pending_before(self, trigger: float) -> tuple[int, int]:
        """発火の直前の会計値 (この連鎖の分を含めない)。"""
        before = [pair for t, pair in self.ring if t < trigger]
        return before[-1] if before else (0, 0)

    # ---- 連鎖の始まりと終わり ----
    def _episode(self, side: int, chain: Any, t_sec: float, game_idx: int, dropped: tuple) -> None:
        """連鎖状態に入った最初の chain_event で 1 件の出来事を作る。断片の再通知では作らない。"""
        keep = self.active[side] and self.states[side] in IN_CHAIN_STATES
        if self.states[side] not in CHAIN_STATES and not keep:
            if self.active[side] and self.latest_event[side] is not None:
                self.latest_event[side].chain_end_t = t_sec
            self.active[side], self.fire_board[side] = False, None
            return
        if chain is not None:
            self.latest_chain[side] = chain
        if self.active[side] or chain is None or getattr(chain, "before_board", None) is None:
            return
        self.active[side] = True
        recv = 1 - side
        base = self._base("fire", t_sec, game_idx, chain.trigger_sec, side)
        base["event"] = self._kind(side, t_sec)
        base["exchange_id"] = self.exchange_id
        fire, basis, count = self._fire_board(side, self._scrub(chain.before_board._grid))
        self.fire_board[side] = fire
        event = _Event(base["event_id"], side, chain.trigger_sec, fire, count, dropped,
                       self._pending_before(chain.trigger_sec))
        event.fire_basis = basis
        self.events.append(event)
        self.latest_event[side] = event
        if self.last_stable[recv] is None:
            self.sink(dict(base, status="skipped", reason="no_receiver_board"))
            return
        self._project(event, base, t_sec)

    def _fire_board(self, side: int, before: np.ndarray) -> tuple[np.ndarray, str, int]:
        """連鎖が起きる盤面を決める。確定盤面が最後の設置を含まず連鎖 0 なら、発火した組を置いた最大火力の
        配置を使う (上限)。(盤面, 由来, 連鎖数)。"""
        result = native.simulate_chain(_board(before), exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
        if result.chain_count:
            return before, "before_board", int(result.chain_count)
        pair = self.tape.active_pair(side)
        placed = fire_with_pair(before, pair) if pair is not None else None
        if placed is None:
            return before, "no_fire_found", 0
        count = native.simulate_chain(_board(placed), exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED).chain_count
        return placed, "with_active_pair_max", int(count)

    def _base(self, kind: str, t_sec: float, game_idx: int, trigger: float, side: int) -> dict:
        self.event_count += 1
        return dict(schema=SCHEMA, kind="event", event=kind, event_id=self.event_count,
                    exchange_id=self.exchange_id, t_sec=t_sec, frame_idx=frame_of(t_sec), game_idx=game_idx,
                    trigger_sec=trigger, trigger_frame=frame_of(trigger),
                    attacker=SIDE_LABELS[side], receiver=SIDE_LABELS[1 - side])

    def _kind(self, side: int, t_sec: float) -> str:
        """fire=撃ち合いの最初 / counter=撃たれた側の最初の連鎖 / main=すでに撃った側のさらなる連鎖。"""
        if t_sec - self.last_activity > EXCHANGE_IDLE_TIMEOUT_SEC or not self.exchange_sides:
            self.exchange_id += 1
            self.exchange_sides = set()
            kind = "fire"
        else:
            kind = "main" if side in self.exchange_sides else "counter"
        self.exchange_sides.add(side)
        return kind

    # ---- 投影 ----
    def _input(self, event: _Event, t_sec: float) -> ExchangeInput:
        recv, side = 1 - event.side, event.side
        remaining = max(0.0, event.total_sec - max(0.0, t_sec - event.trigger))
        elapsed = max(0.0, event.trigger - (self._first_move() or event.trigger))
        first = event.projections == 0
        landed = max(0, self.dropped[recv] - event.dropped[recv])
        expected = max(0, event.pending_before[recv] + event.surplus - landed)
        pending_r = event.pending_before[recv] if first else max(self.pending[recv], expected)
        return ExchangeInput(
            attacker=side, attacker_board=event.before, receiver_board=self._receiver_board(recv),
            pending_receiver=pending_r, pending_attacker=event.pending_before[side] if first else self.pending[side],
            remaining_sec=remaining, elapsed_sec=elapsed, known_receiver=self._receiver_pairs(recv),
            known_attacker=self.tape.known_future(side), colors=self._colors(), hand_table=self.table,
            receiver_busy_sec=self._busy(recv, t_sec), expected_hand_sec=self.expected_hand_sec,
            expected_table=self.expected_table,
            attack_counted=not first, receiver_credit=self._credit(recv, t_sec))

    def _completion_known(self, recv: int) -> bool:
        """受け側が連鎖の途中で、その連鎖の完走盤面を確定できるか。発火前の確定盤面そのもので連鎖が起きる時
        (fire_basis=before_board) だけ。最後の設置が分からない時の上限の連鎖は、受け側の実際の連鎖と別物で、
        実際より大きな連鎖を完走した盤面 (= 空に近い盤面) を作ってしまう (10/13 夜 g20 の 1P で確認)。"""
        event = self.latest_event[recv]
        return (self.active[recv] and self.states[recv] in CHAIN_STATES and event is not None
                and event.fire_basis == "before_board" and self.fire_board[recv] is not None)

    @staticmethod
    def _color_cells(grid: np.ndarray) -> int:
        return int(np.count_nonzero((grid >= FIRST_COLOR) & (grid <= LAST_COLOR)))

    def _placed_since_shift(self, recv: int) -> bool:
        """NEXT が繰り上がってから、操作中の組をすでに置いたか (確定盤面の色ぷよが 2 個以上増えた)。
        置き終わりから次の組が出るまでの間は、次に置くのは NEXT (操作中の組ではない)。"""
        grid = self.last_stable[recv]
        return grid is not None and self._color_cells(grid) >= self.cells_at_shift[recv] + PAIR_CELLS

    def _receiver_pairs(self, recv: int) -> tuple:
        """受け側がこれから置くツモ。連鎖中・置き済みなら NEXT から、そうでなければ操作中の組が最初の 1 手。

        おじゃま落下中 (OJAMA_FALL) は、落下が始まった後に NEXT が繰り上がっていれば新しい組が出ている
        (10/13 夜 night2 g12: 落下の途中で次の組が出て、直後に 1 手で大連鎖を撃った)。
        """
        shifts = self.tape.sides[recv].shifts
        fell_after_shift = self.states[recv] == "OJAMA_FALL" and (not shifts or shifts[-1] < self.fall_start[recv])
        if self.states[recv] in CHAIN_STATES or fell_after_shift or self._placed_since_shift(recv):
            return self.tape.known_future(recv)
        return self.tape.next_to_place(recv)

    def _credit(self, recv: int, t_sec: float) -> int:
        """受け側が連鎖中なら、その連鎖が (まだ会計に載っていない) これから生むおじゃま。"""
        fire, chain = self.fire_board[recv], self.latest_chain[recv]
        if not self._completion_known(recv) or chain is None:
            return 0
        result = native.simulate_chain(_board(fire), exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
        elapsed = max(0.0, chain.trigger_sec - (self._first_move() or chain.trigger_sec))
        total, seen = points_to_ojama(result.exact_score, elapsed), points_to_ojama(chain.total_score, elapsed)
        return max(0, total - seen)

    def _receiver_board(self, recv: int) -> np.ndarray:
        """受け側の盤面。受け側が連鎖中なら、その連鎖が終わった後の盤面 (見えない間の盤面は使わない)。"""
        fire = self.fire_board[recv]
        self.receiver_basis = "stable_confirmed"
        if self.active[recv] and not self._completion_known(recv):
            self.receiver_basis = "own_chain_completion_unknown"   # 連鎖中だが完走盤面を確定できない (確定盤面をそのまま使う)
        if self._completion_known(recv):
            self.receiver_basis = "own_chain_completion"
            return np.array(native.simulate_chain(_board(fire),
                            exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED).final_board._grid)
        return self._scrub(self.last_stable[recv])

    def _project(self, event: _Event, base: dict, t_sec: float) -> None:
        recv = 1 - event.side
        inp = self._input(event, t_sec)
        projection = self.project(inp, self.budget)
        if event.projections == 0:
            event.surplus = max(0, projection.attack_send - event.pending_before[event.side])
            event.pred_index = (self.tape.counter(recv) or 0) + 2
            event.pred_pairs = list(self.tape.known_future(recv)[2:])
        event.projections += 1
        self.sink(self._row(base, inp, projection, event))

    def _reproject(self, t_sec: float, game_idx: int) -> None:
        """受け側が 1 手置くたびに、残りの手数・盤面・予告で投影し直す。窓が開いている最新の出来事だけ。"""
        latest: dict[int, _Event] = {}
        for event in self.events:
            if event.window_open:
                latest[event.side] = event
        for side, event in latest.items():
            recv = 1 - side
            if not self.shifted[recv] or self.last_stable[recv] is None:
                continue
            base = self._base("placement", t_sec, game_idx, event.trigger, side)
            base["parent_event_id"] = event.event_id
            self._project(event, base, t_sec)

    def _first_move(self) -> Optional[float]:
        times = [s.shifts[0] for s in self.tape.sides if s.shifts]
        return min(times) if times else self.first_t

    def _busy(self, recv: int, t_sec: float) -> float:
        """受け側が連鎖中で置けない残り時間。受け側の連鎖の予測連鎖数 (発火前の盤面から) の演出時間から数える
        (chain_event の連鎖数は読み取りの途中経過なので使わない)。"""
        event = self.latest_event[recv]
        if event is None or not self.active[recv]:
            return 0.0
        return max(0.0, event.total_sec - max(0.0, t_sec - event.trigger))

    def _row(self, base: dict, inp: ExchangeInput, projection: Any, event: _Event) -> dict:
        index = (self.tape.counter(1 - inp.attacker) or 0) + 2
        pred = list(self.tape.known_future(1 - inp.attacker)[2:])
        row = dict(base, status=projection.status, reason=projection.reason, remaining_sec=inp.remaining_sec,
                   recv_shifts=self.tape.placements(1 - inp.attacker), attacker_shifts=self.tape.placements(inp.attacker),
                   elapsed_sec=inp.elapsed_sec,
                   pending_p1=self.pending[0], pending_p2=self.pending[1], incoming_used=inp.pending_receiver,
                   tape_aligned=self.tape.aligned, tape_uncertain=self.tape.uncertain,
                   tape_known_len=len(self.tape.known_future(1 - inp.attacker)),
                   tape_lead=self.tape.lead(1 - inp.attacker), pairs_to_place=len(inp.known_receiver),
                   tape_pred_index=index, tape_pred=[list(p) for p in pred],
                   receiver_board=board_text(inp.receiver_board), attacker_board=board_text(inp.attacker_board),
                   colors=list(inp.colors))
        data = asdict(projection)
        data.pop("status"), data.pop("reason")
        row.update(data)
        row["evidence"] = self._evidence(inp, projection, event)
        return row

    def _evidence(self, inp: ExchangeInput, projection: Any, event: _Event) -> dict:
        """信頼度の材料だけを出す (式は作らない。user とデータを見て決める)。

        断定 (死亡など) は完全な証明のときだけ。本試作は断定せず、proof_complete は常に False。
        """
        chain = self.latest_chain[event.side]
        return dict(
            fire_basis=event.fire_basis,                          # before_board=厳密 / with_active_pair_max=上限
            attack_observed_count=None if chain is None else int(chain.chain_count),
            attack_observed_score=None if chain is None else int(chain.total_score),
            attack_predicted_count=event.pred_count,
            receiver_basis=self.receiver_basis,                   # stable_confirmed / own_chain_completion
            hidden_scrubbed=self.scrubbed, hidden_colors_dropped=projection.hidden_colors_dropped,
            tape_known_len=len(self.tape.known_future(event.side ^ 1)), pairs_to_place=len(inp.known_receiver),
            active_pair_included=len(inp.known_receiver) > len(self.tape.known_future(event.side ^ 1)),
            tape_aligned=self.tape.aligned,
            tape_uncertain=self.tape.uncertain,
            hands_spread=projection.hands_opt - projection.hands_exp,
            native_calls=sum(projection.calls.values()), native_calls_cap=self.budget.max_native_calls,
            capped=list(projection.capped), known_depth_cap=self.budget.known_depth_cap,
            dig_steps_cap=self.budget.dig_steps_cap, projection_index=event.projections,
            pending_source="callback" if self._custom_pending else "snapshot", proof_complete=False)

    # ---- 着弾・結果 ----
    def _on_landing(self, t_sec: float, game_idx: int) -> None:
        """受け側の状態が OJAMA_FALL に入った時を着弾の開始として記録する (実測のみ)。"""
        for i in range(2):
            if self.states[i] == "OJAMA_FALL" and self.prev_states[i] != "OJAMA_FALL":
                row = self._base("landing", t_sec, game_idx, t_sec, 1 - i)
                row.update(pending_p1=self.pending[0], pending_p2=self.pending[1],
                           recv_shifts=self.tape.placements(i))
                self.sink(row)
                for event in self.events:
                    ended = event.chain_end_t is not None and t_sec >= event.chain_end_t
                    if event.side == 1 - i and event.landing_t is None and ended:
                        event.landing_t = t_sec

    def _track(self, t_sec: float, game_idx: int) -> None:
        """出来事ごとに、受け側の応手・攻撃側の操作復帰を実測し、窓を閉じる。"""
        if self.reproject:
            self._reproject(t_sec, game_idx)
        for event in self.events:
            recv = 1 - event.side
            if event.attacker_free_t is None:
                free = [s for s in self.tape.sides[event.side].shifts if s > event.trigger]
                event.attacker_free_t = free[0] if free else None
            chain = self.latest_chain[recv]
            if event.response is None and chain is not None and chain.trigger_sec > event.trigger:
                event.response = dict(trigger_sec=chain.trigger_sec, chain_count=int(chain.chain_count),
                                      score=int(chain.total_score))
            if not event.outcome_done and (event.landing_t is not None or event.response is not None):
                self._close(event, t_sec, "landing" if event.landing_t is not None else "response")
            drained = (not self.active[event.side] and self.pending[recv] == 0
                       and t_sec - event.trigger > LEDGER_GRACE_SEC)
            if event.window_open and (drained or t_sec - self.last_activity > EXCHANGE_IDLE_TIMEOUT_SEC):
                event.window_open = False

    def _close(self, event: _Event, t_sec: float, reason: str) -> None:
        recv = 1 - event.side
        end = event.landing_t if event.landing_t is not None else (
            event.response["trigger_sec"] if event.response else t_sec)
        shifts = [s for s in self.tape.sides[recv].shifts if event.trigger < s <= end]
        event.outcome_done = True
        if reason == "game_end":
            event.window_open = False
        free = event.attacker_free_t
        self.sink(dict(schema=SCHEMA, kind="outcome", event_id=event.event_id, t_sec=t_sec,
                       frame_idx=frame_of(t_sec), trigger_sec=event.trigger, close_reason=reason,
                       landing_sec=event.landing_t, response=event.response,
                       receiver_shifts_to_end=len(shifts), hands_actual=len(shifts) + 1,
                       attacker_free_sec=free, fire_basis=event.fire_basis, pred_chain_count=event.pred_count,
                       chain_end_sec=event.chain_end_t,
                       chain_time_actual=None if event.chain_end_t is None else event.chain_end_t - event.trigger,
                       chain_time_predicted=event.total_sec,
                       landing_delay=None if event.landing_t is None else event.landing_t - event.trigger,
                       response_hands=None if not event.response else len(
                           [s for s in self.tape.sides[recv].shifts
                            if event.trigger < s <= event.response["trigger_sec"]]) + 1))

    def _tape_check(self, event: _Event, t_sec: float) -> None:
        """予測した受け側の未来ツモを、受け側が後で自分の NEXT/NEXT2 として見た値と照合する。"""
        recv = 1 - event.side
        hit = total = 0
        for k, pair in enumerate(event.pred_pairs):
            actual = self.tape.own_at(recv, event.pred_index + k)
            if actual is None:
                continue
            total += 1
            hit += int(tuple(actual) == tuple(pair))
        self.sink(dict(schema=SCHEMA, kind="tape_check", event_id=event.event_id, t_sec=t_sec,
                       frame_idx=frame_of(t_sec), predicted=len(event.pred_pairs), compared=total, hits=hit))


def jsonl_sink(path: Path) -> Sink:
    """追記専用の JSONL sink。行ごとに書き出す (メモリを増やさない)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a", encoding="utf-8", buffering=1)

    def write(row: dict) -> None:
        stream.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    write.close = stream.close  # type: ignore[attr-defined]
    return write


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


REVIEW_FIELDS = ("event", "attacker", "receiver", "hands_exp", "hands_opt", "remaining_sec", "known_len",
                 "tape_lead", "best_exp", "best_opt", "required_cancel", "max_cancel_opt", "unreturnable_opt",
                 "incoming")
CANDIDATE_REVIEW_KEYS = ("hands", "basis", "send", "pending_after", "landed_dead", "board_after", "opp_time_sec",
                         "opp_max", "opp_returns", "main_f4", "stacked_opp_returns", "margin")


def review_columns(row: dict) -> dict:
    """event 行をレビュー動画のフレーム別 CSV に載せる平坦な列へ展開する (pe_ 接頭辞)。

    候補ごとの列は pe_<モード>_<候補名>_<項目> (例: pe_exp_max_send、pe_opt_dig_board_after)。
    時刻は t_sec / frame_idx。既存 b3_review_input の行 (t_sec 以前の最後の行を使う) と同じ流儀。
    """
    out: dict = {"t_sec": row.get("t_sec"), "frame_idx": row.get("frame_idx"), "game_idx": row.get("game_idx"),
                 "pe_event_id": row.get("event_id"), "pe_status": row.get("status")}
    out.update({f"pe_{k}": row.get(k) for k in REVIEW_FIELDS})
    for cand in row.get("candidates") or []:
        stem = f"pe_{cand['mode']}_{cand['name']}"
        for key in CANDIDATE_REVIEW_KEYS:
            out[f"{stem}_{key}"] = cand.get(key)
    return out


def review_rows(rows: list[dict]) -> list[dict]:
    """影ログの全行から、レビュー動画の「t_sec 以前の最後の行を使う」方式 (b3_review_video.row_at) に
    そのまま渡せる平坦な行列を作る。投影した行 (発火・応手・本線・設置のたび) と着弾の行を時刻順に並べ、
    着弾の行は pe_event='landing' と受け側だけを持つ。候補ごとの列は review_columns を参照。

    既存のフレーム別 CSV (b3_review_input.CSV_FIELDS は固定列) には pe_ 列を足さず、別の CSV として出す想定。
    動画の下帯に出す行は、b3_review_panel の流儀 (幅 1280 基準で 18px・1P=水色・2P=橙) で
    pe_best_exp / pe_exp_max_send / pe_exp_dig_board_after などを並べればよい。
    """
    flat = []
    for row in rows:
        if row.get("kind") != "event":
            continue
        if row.get("event") == "landing":
            flat.append(dict(t_sec=row["t_sec"], frame_idx=row["frame_idx"], game_idx=row["game_idx"],
                             pe_event="landing", pe_attacker=row["attacker"], pe_receiver=row["receiver"]))
        elif row.get("status") is not None:
            flat.append(review_columns(row))
    return sorted(flat, key=lambda r: (r["t_sec"], r.get("pe_event_id") or 0))
