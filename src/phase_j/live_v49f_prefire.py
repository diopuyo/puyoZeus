"""段3 (撃つ直前): NEXT 移動で撃つ組の全配置を事前計算し、着地で暫定 X を引く。既定OFF。

設計 <work>/live_x_timing_20261022/DESIGN.md §3・§8 (親裁定 2026-10-22):
- NEXT 移動 = 前の組の置き終わり。撃つ組 = 動く前の NEXT1。STABLE 盤 + 組の全配置 (22) を連鎖計算した表を作る。
- 着地 (新しい2個が支持位置に出た) を見たら表を引き、撃てる配置なら暫定 X を出す。
- 暫定 X は記録 (shadow) と別欄 x_provisional だけに出す。v4.9f の p1 へは入れない。
- 取り下げ: NEXT 移動・おじゃま落下・着地から RETRACT_SEC 以内に CHAIN 通知が来ない。

表と照合は状態を持たない関数。側ごとの状態は PrefireTracker (外側の wrapper) だけが持つ。
隠し段 (13段目 = row 0) のぷよは接着せず消去判定にも連結数にも入らない (user 確定ルール)。
連鎖計算は v4.9f と同じ `exchange_count_prediction.simulate` (隠し段を消去・連結から除外) を使う。
隠し段に置く配置は画像で見えないため照合の対象外 (表には残し、件数で数える)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from time import perf_counter
from typing import Any, Callable

ENV_SHADOW = 'PUYO_LIVE_X_PREFIRE_SHADOW'
ENV_DISPLAY = 'PUYO_LIVE_X_PREFIRE'
COLS, CELLS = 6, 78
VISIBLE_START = COLS            # row 0 (隠し段) を除いた最初のセル
PAIR_SIZE = 2
PAIR_COLORS = frozenset(range(1, 6))
BOARD_CODES = frozenset((0, *range(1, 6), 9))
# 着地 → CHAIN 通知は 21c で P95 0.73 秒 (設計 §3.4、n=66)。これを超えて来なければ撃っていない。
RETRACT_SEC = .8
# NEXT 読取りは滑る途中で None・おじゃま色・入れ替わりを返す (21c で値変化 4,368 件中 1,851 件が非スライド)。
# 同じ値が連続 2 通知で読めたときだけ NEXT の値が変わったとみなす。
NEXT_SETTLE_READS = 2
MS = 1000.
STABLE, CHAIN, OJAMA, MENU, SETTLE = 'STABLE', 'CHAIN', 'OJAMA_FALL', 'MENU', 'GRAVITY_SETTLE'
# 画像で着地を探す状態。連鎖・おじゃま・重力の途中では探さない (CHAIN は発火通知と同じ通知のみ)。
DETECT_STATES = frozenset({'TSUMO_FALL', STABLE, CHAIN})
REJECT_REASONS = ('color_mismatch', 'classifier_disagree', 'unsupported_air')
LABEL = '撃つ直前の火力 X (暫定)'
CLASSIFIERS = (0, 1)   # LiveImages の median / vote

# (読むセル, 分類器の番号) → {セル: 色}。読取りの費用は呼出側が測る。
Reader = Callable[[tuple[int, ...], int], dict[int, int]]


def flags() -> tuple[bool, bool]:
    """(記録, 表示)。表示ONは記録も行う。spawn 子 process へは環境変数で届く。"""
    display = os.environ.get(ENV_DISPLAY) == '1'
    return display or os.environ.get(ENV_SHADOW) == '1', display


@dataclass(frozen=True)
class Entry:
    """1 配置。added は (セル, 色) の昇順。hidden は隠し段に置く配置 (画像で見えない)。"""
    col: int
    rotation: int
    added: tuple[tuple[int, int], ...]
    hidden: bool
    score: int
    chain: int
    erased: tuple[int, ...]
    scores: tuple[int, ...]

    def supports(self) -> tuple[int, ...]:
        """追加2個の真下で、追加セル自身でも床でもないセル (着地なら埋まっている)。"""
        cells = {i for i, _ in self.added}
        return tuple(i+COLS for i in sorted(cells) if i+COLS < CELLS and i+COLS not in cells)


@dataclass(frozen=True)
class Table:
    base: tuple[int, ...]
    pair: tuple[int, int]
    entries: tuple[Entry, ...]
    candidates: tuple[int, ...]
    supports: tuple[int, ...]
    build_ms: float
    base_fires: bool = False

    def firing(self) -> int:
        return sum(e.chain > 0 for e in self.entries)

    def cells(self) -> tuple[int, ...]:
        """画像で読むセル。着地候補と、その支え。"""
        return tuple(sorted({*self.candidates, *self.supports}))


def valid_board(board: tuple[int, ...] | None) -> bool:
    return board is not None and len(board) == CELLS and all(c in BOARD_CODES for c in board)


def build_table(board: tuple[int, ...], pair: tuple[int, int], native: Any) -> Table:
    """配置列挙は v3.4 候補と同じ `enumerate_placements_py(board, a, b, False)`。"""
    from src.exchange_count_prediction import simulate
    began = perf_counter()
    entries: dict[tuple, Entry] = {}
    for col, rotation, placed, _dead in native.enumerate_placements_py(list(board), pair[0], pair[1], False):
        added = tuple(sorted((i, int(placed[i])) for i in range(CELLS) if placed[i] != board[i]))
        if len(added) != PAIR_SIZE or any(board[i] for i, _ in added) or added in entries:
            continue
        value = simulate(list(placed), native, compact=True)
        entries[added] = Entry(int(col), int(rotation), added, any(i < VISIBLE_START for i, _ in added),
                               int(value['score']), len(value['erased']), tuple(value['erased']),
                               tuple(int(s) for s in value['scores']))
    values = tuple(entries.values())
    visible = [e for e in values if not e.hidden]
    candidates = sorted({i for e in visible for i, _ in e.added})
    supports = sorted({i for e in visible for i in e.supports()} - set(candidates))
    # STABLE 盤そのものに消える組がある = 盤の読み違い。そこから引いた X は信用しない。
    base_fires = bool(simulate(list(board), native, compact=True)['erased'])
    return Table(tuple(board), tuple(pair), values, tuple(candidates), tuple(supports),
                 (perf_counter()-began)*MS, base_fires)


def match_cells(table: Table, observed: frozenset) -> tuple[Entry | None, str]:
    """観測した追加2個 (セル, 色) が表のどの配置か。位置一致で色違いは color_mismatch で拒否。"""
    if len(observed) != PAIR_SIZE:
        return None, 'none'
    for entry in table.entries:
        if not entry.hidden and frozenset(entry.added) == observed:
            return entry, 'match'
    cells = frozenset(i for i, _ in observed)
    if any(frozenset(i for i, _ in e.added) == cells for e in table.entries if not e.hidden):
        return None, 'color_mismatch'
    return None, 'unsupported'


def match_read(table: Table, read: dict[int, int]) -> tuple[Entry | None, str]:
    """1 分類器の読取り。着地候補セルの追加2個で引き、支えのセルが空なら空中として拒否。"""
    candidates = set(table.candidates)
    entry, reason = match_cells(table, frozenset((i, c) for i, c in read.items() if c and i in candidates))
    if entry is not None and any(not read.get(i, 0) for i in entry.supports()):
        return None, 'unsupported_air'
    return entry, reason


def match_reads(table: Table, reads: list[dict[int, int]]) -> tuple[Entry | None, str]:
    """2 分類器 (median / vote) が同じ配置を指したときだけ一致とする。"""
    if not reads:
        return None, 'none'
    results = [match_read(table, read) for read in reads]
    entries = {r[0] for r in results}
    if len(entries) == 1 and None not in entries:
        return results[0]
    if any(e is not None for e in entries):
        return None, 'classifier_disagree'
    reasons = [r[1] for r in results]
    return None, next((x for x in (*REJECT_REASONS, 'unsupported') if x in reasons), 'none')


def match_board(table: Table, board: tuple[int, ...]) -> tuple[Entry | None, str]:
    """確定盤 (全セル) から着地を引く。可視セルで表の基準盤との差が追加2個だけのときに限る。"""
    changed = [i for i in range(VISIBLE_START, CELLS) if board[i] != table.base[i]]
    if len(changed) != PAIR_SIZE or any(table.base[i] for i in changed):
        return None, 'board_changed'
    return match_cells(table, frozenset((i, board[i]) for i in changed))


def generated(entry: Entry, stamp: float, ledger: dict | None) -> int | None:
    """相殺前の送り量。得点台帳 (繰越・初手時刻) が揃わない間は欠測。段の時刻は既存の段周期で近似。"""
    if not ledger or ledger.get('carry') is None or ledger.get('first_move') is None:
        return None
    from src.exchange_count_ledger import gross, STEP_SEC
    times = [stamp+i*STEP_SEC for i in range(len(entry.scores))]
    return gross(list(entry.scores), times, ledger['carry'], ledger['first_move'])[0]


@dataclass(frozen=True)
class SideView:
    """1 通知の片側。trigger は chain_event の trigger_sec (無ければ None)。"""
    state: str
    board: tuple[int, ...] | None
    next_pair: tuple[int, int] | None
    trigger: float | None
    ledger: dict | None = None
    dnext_pair: tuple[int, int] | None = None


def pair_of(value: Any) -> tuple[int, int] | None:
    return tuple(int(c) for c in value) if value else None


def view_from_raw(raw: Any, ledger: dict | None = None) -> SideView:
    """認識結果の片側から作る。確定盤は STABLE のときだけ使う (設計思想4)。"""
    board = raw.confirmed_board
    grid = tuple(int(c) for c in board._grid.ravel().tolist()) if board is not None else None
    event = raw.chain_event
    return SideView(raw.state.name, grid, pair_of(raw.next_pair),
                    None if event is None else float(event.trigger_sec), ledger, pair_of(raw.dnext_pair))


NextKey = tuple[tuple[int, int], tuple[int, int]]


@dataclass
class SideState:
    settled: NextKey | None = None
    reading: NextKey | None = None
    reads: int = 0
    pair: tuple[int, int] | None = None
    base: tuple[int, ...] | None = None
    move_base: tuple[int, ...] | None = None
    stale: bool = False
    table: Table | None = None
    consumed: bool = True
    landed: bool = False
    trigger: float | None = None
    state: str | None = None
    ledger: dict | None = None
    pending: dict | None = None
    shown: dict | None = None
    last_reject: tuple | None = None


@dataclass
class PrefireTracker:
    """側ごとの状態所有者。observe は記録する事象 (dict) の一覧を返す。"""
    native: Any
    game: int | None = None
    sides: list[SideState] = field(default_factory=lambda: [SideState(), SideState()])

    def reset(self, game: int | None) -> list[dict]:
        events = [e for s, st in enumerate(self.sides) for e in self._retract(s, st, None, 'game')]
        self.game, self.sides = game, [SideState(), SideState()]
        return events

    def observe(self, game: int, side: int, stamp: float, view: SideView,
                read: Reader | None = None) -> list[dict]:
        events = self.reset(game) if game != self.game else []
        s = self.sides[side]
        s.ledger = view.ledger
        if view.state == MENU:
            events += self._retract(side, s, stamp, 'menu')
            s.table, s.consumed, s.shown, s.state = None, True, None, view.state
            return events
        events += self._next(side, s, view, stamp)
        if view.state == OJAMA:
            # おじゃまが落ちた = 組は撃たずに置かれた。落下後の確定盤が出るまで表を作らない。
            events += self._retract(side, s, stamp, 'ojama')
            s.table, s.stale = None, True
        if s.pending is not None and stamp-s.pending['landing_t'] > RETRACT_SEC:
            events += self._retract(side, s, stamp, 'timeout')
        events += self._board(side, s, view, stamp)
        events += self._prepare(side, s, stamp)
        if read is not None and s.table is not None and not s.consumed and view.state in DETECT_STATES:
            events += self._image(side, s, view, stamp, read)
        events += self._chain(side, s, view, stamp)
        s.state = view.state
        return events

    def _next(self, side: int, s: SideState, view: SideView, stamp: float) -> list[dict]:
        """(NEXT1, NEXT2) が連続一致で変わり、新 NEXT1 = 旧 NEXT2 (滑り) なら、旧 NEXT1 を撃つ組にする。

        同じ色の組が続いても滑りを見落とさないよう NEXT2 も読む。滑りでない変化 (読み違い) は
        撃つ組が分からないので、暫定 X を取り下げて次の滑りまで照合しない。
        """
        key = (view.next_pair, view.dnext_pair)
        if any(p is None or any(c not in PAIR_COLORS for c in p) for p in key):
            s.reading, s.reads = None, 0
            return []
        s.reads = s.reads+1 if key == s.reading else 1
        s.reading = key
        if s.reads < NEXT_SETTLE_READS or key == s.settled:
            return []
        previous, s.settled = s.settled, key
        events = self._retract(side, s, stamp, 'next_moved')
        s.shown, s.table, s.consumed, s.landed = None, None, True, False
        if previous is not None and key[0] == previous[1]:
            s.pair, s.consumed = previous[0], False
            # 前回の滑りから確定盤が変わっていない = 前の組の着地がまだ確定盤に出ていない。出るまで待つ。
            s.stale = s.base is not None and s.base == s.move_base
        s.move_base = s.base
        return events

    def _board(self, side: int, s: SideState, view: SideView, stamp: float) -> list[dict]:
        """STABLE 確定盤が変わったとき: 着地なら表を引く。着地でなければ基準盤を更新して表を作り直す。"""
        if view.state != STABLE or not valid_board(view.board) or view.board == s.base:
            return []
        s.base, s.stale = view.board, False
        if s.table is None or s.consumed:
            return []
        entry, _reason = match_board(s.table, view.board)
        if entry is not None:
            return self._landing(side, s, entry, stamp, 'confirmed', 0.)
        if not s.landed:
            s.table = None
        return []

    def _prepare(self, side: int, s: SideState, stamp: float) -> list[dict]:
        if (s.table is not None or s.consumed or s.stale or s.pair is None
                or not valid_board(s.base)):
            return []
        s.table = build_table(s.base, s.pair, self.native)
        return [dict(kind='table', side=side, t=stamp, pair=list(s.pair), entries=len(s.table.entries),
                     firing=s.table.firing(), hidden=sum(e.hidden for e in s.table.entries),
                     cells=len(s.table.cells()), base_fires=s.table.base_fires, build_ms=s.table.build_ms)]

    def _image(self, side: int, s: SideState, view: SideView, stamp: float, read: Reader) -> list[dict]:
        """毎通知は median で着地候補セルだけを読む (2 側で約 2.4ms)。2 個見えたときだけ支えと vote も読む。"""
        began = perf_counter()
        first = read(s.table.candidates, CLASSIFIERS[0])
        if sum(1 for c in first.values() if c) != PAIR_SIZE:
            return []
        entry, reason = match_reads(s.table, [read(s.table.cells(), i) for i in CLASSIFIERS])
        read_ms = (perf_counter()-began)*MS
        if entry is not None:
            return self._landing(side, s, entry, stamp, 'image', read_ms)
        key = (reason, s.table.base, s.table.pair)
        if reason in REJECT_REASONS and key != s.last_reject:
            s.last_reject = key
            return [dict(kind='reject', side=side, t=stamp, reason=reason, state=view.state, read_ms=read_ms)]
        return []

    def _landing(self, side: int, s: SideState, entry: Entry, stamp: float, source: str,
                 read_ms: float) -> list[dict]:
        """撃てる配置なら暫定 X。撃てない配置へ移ったら取り下げ。同じ配置の再観測は何もしない。"""
        s.landed = True
        if s.pending is not None and s.pending['entry'] == entry:
            return []
        events = self._retract(side, s, stamp, 'moved') if s.pending is not None else []
        if entry.chain == 0 or s.table.base_fires:
            return events+[dict(kind='landing', side=side, t=stamp, source=source, chain=entry.chain,
                                base_fires=s.table.base_fires, read_ms=read_ms)]
        value = dict(side=side, landing_t=stamp, source=source, col=entry.col, rotation=entry.rotation,
                     score=entry.score, chain=entry.chain, generated=generated(entry, stamp, s.ledger),
                     erased=list(entry.erased), scores=list(entry.scores), pair=list(s.table.pair))
        s.pending = dict(value, entry=entry)
        s.shown = dict(value, status='pending')
        return events+[dict(kind='provisional', t=stamp, read_ms=read_ms, **value)]

    def _chain(self, side: int, s: SideState, view: SideView, stamp: float) -> list[dict]:
        """新しい連鎖事象 (trigger) か CHAIN 状態への遷移 = CHAIN 通知。組は使い切った。"""
        fresh = view.trigger is not None and view.trigger != s.trigger
        entered = view.state == CHAIN and s.state not in (CHAIN, SETTLE)
        if view.trigger is not None:
            s.trigger = view.trigger
        if not (fresh or entered):
            return []
        s.table, s.consumed = None, True
        if s.pending is None:
            return []
        pending, s.pending = s.pending, None
        s.shown = dict(s.shown or {}, status='confirmed', confirmed_t=stamp)
        return [dict(kind='confirm', side=side, t=stamp, landing_t=pending['landing_t'],
                     delay_sec=stamp-pending['landing_t'], score=pending['score'], chain=pending['chain'],
                     source=pending['source'])]

    def _retract(self, side: int, s: SideState, stamp: float | None, reason: str) -> list[dict]:
        if s.pending is None:
            return []
        pending, s.pending, s.shown = s.pending, None, None
        delay = None if stamp is None else stamp-pending['landing_t']
        return [dict(kind='retract', side=side, t=stamp, landing_t=pending['landing_t'], reason=reason,
                     delay_sec=delay, score=pending['score'], chain=pending['chain'],
                     source=pending['source'])]


def public(shown: dict | None, t_sec: float) -> dict | None:
    """公開欄の片側。p1 とは独立した別欄 (親裁定 (2))。"""
    if shown is None:
        return None
    keys = ('status', 'score', 'chain', 'generated', 'landing_t', 'source', 'col', 'rotation')
    return dict({k: shown.get(k) for k in keys}, age_sec=max(0., t_sec-shown['landing_t']))


class PrefireCapture:
    """認識 process 側。画像が生きている間に、表の着地候補セルとその支えだけを 2 分類器で読む。"""

    def __init__(self, images: Any, native: Any | None = None) -> None:
        if native is None:
            import puyo_core as native
        self.images, self.tracker = images, PrefireTracker(native)

    def observe(self, game: int, stamp: float, raw_sides: tuple, image: Any,
                ledgers: list[dict | None]) -> dict:
        began = perf_counter()
        events: list[dict] = []
        reads: list[float] = []
        for side, raw in enumerate(raw_sides):
            reader = self._reader(image, side, reads)
            events += self.tracker.observe(game, side, stamp, view_from_raw(raw, ledgers[side]), reader)
        shown = [None if s.shown is None else dict(s.shown) for s in self.tracker.sides]
        return dict(t=stamp, events=events, shown=shown, cost_ms=(perf_counter()-began)*MS,
                    read_ms=sum(reads), reads=len(reads))

    def _reader(self, image: Any, side: int, costs: list[float]) -> Reader:
        def read(cells: tuple[int, ...], which: int) -> dict[int, int]:
            began = perf_counter()
            result = self.images.read_cells(image, side, cells, (which,))[0]
            costs.append((perf_counter()-began)*MS)
            return result
        return read


class PrefireView:
    """評価 process 側。記録 (shadow) と、表示ONのときだけ別欄 x_provisional を行へ足す。"""

    def __init__(self, output: Any, display: bool) -> None:
        from .live_obs_audit import open_audit
        self.output, self.display = output, display
        self.audit = open_audit(output/'live_x_prefire.jsonl')
        self.shown: list[dict | None] = [None, None]
        self.costs: dict[str, list[float]] = dict(cost_ms=[], read_ms=[], build_ms=[])
        self.counts: dict[str, int] = {}

    def take(self, packet: dict, game: int) -> dict:
        """'prefire' を取り出して記録する。v4.9f の作業側へは OFF と同じ packet を渡す。"""
        data = packet.get('prefire')
        if data is None:
            return packet
        self.shown = list(data['shown'])
        self.costs['cost_ms'].append(data['cost_ms'])
        if data['reads']:
            self.costs['read_ms'].append(data['read_ms'])
        for event in data['events']:
            key = event['kind'] if event['kind'] != 'retract' else 'retract:'+event['reason']
            self.counts[key] = self.counts.get(key, 0)+1
            if event['kind'] == 'table':
                self.costs['build_ms'].append(event['build_ms'])
        if data['events']:
            import json
            self.audit.write(json.dumps(dict(t=data['t'], game=game, cost_ms=data['cost_ms'],
                read_ms=data['read_ms'], reads=data['reads'], events=data['events']), ensure_ascii=False)+'\n')
        return {k: v for k, v in packet.items() if k != 'prefire'}

    def decorate(self, row: dict) -> None:
        if not self.display:
            return
        sides = self.shown if row.get('match_active', True) else [None, None]
        values = [public(s, row['t_sec']) for s in sides]
        row['x_provisional'] = dict(label=LABEL, sides=values)
        if 'judgment_id' in row:
            # 完了駆動の重複排除で暫定 X の出入りが捨てられないよう、表示ONのときだけ鍵へ足す。
            marks = tuple(None if v is None else (v['status'], v['landing_t'], v['score']) for v in values)
            row['judgment_id'] = (*row['judgment_id'], marks)

    def close(self) -> None:
        import json
        self.audit.close()
        summary = dict(counts=self.counts, **{k: quantiles(v) for k, v in self.costs.items()})
        (self.output/'live_x_prefire_summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


SUMMARY_PERCENTILES = (50, 95, 99)


def quantiles(values: list[float]) -> dict:
    """母数つきの分位 (受領票に母数を書く)。"""
    ordered = sorted(values)
    if not ordered:
        return dict(count=0)
    picks = {f'P{p}': ordered[min(len(ordered)-1, int(p/100*len(ordered)))] for p in SUMMARY_PERCENTILES}
    return dict(count=len(ordered), **picks, max=ordered[-1])
