"""共通ツモテープ: 両側の NEXT/NEXT2 観測から「絶対手番号 → ツモ」の表を作る (影ログ用の外部ラッパー)。

user 回答 (2026-10-13): ツモの順番は 1P と 2P で共通。手数が進んでいる側がすでに引いたツモは、
遅れている側の未来ツモになる。本クラスは状態を持つので外部ラッパーに置き、判定側の関数には
`known_future()` の結果 (純粋なタプル) だけを渡す。

構造
- 各側が自分の観測列 `own` (NEXT が繰り上がるたびに 1 つ追加される確認済みツモ列) を持つ。
- 絶対手番号 = base(側) + own の添字。base は両側の観測列の一致から決める (同じ手番で
  同じツモを見ているはずなので、直近 ALIGN_WINDOW 個が矛盾なく重なる base を採用する)。
- 受け側の未来ツモ = 受け側の NEXT/NEXT2 + 相手側が先に引いて観測済みの分 (一致が確認できた時だけ)。

誤り方の方針 (fail-silent 警戒、黙って治さない)
- 10/13 夜の記録 (g10) では、試合開始直後に側 2 の最初の確認 NEXT が試合前の古い組で、
  カウンタを 0 始まりとして固定すると全手番が 1 ずれ、以後ずっと食い違った。
  そこで「0 始まり」を仮定せず、観測列の一致から base を決め直す。
- 説明できない NEXT 変化 (繰り上がりでも既知の飛びでもない) は、NEXT_RESYNC_SEC 続くまで
  ノイズとして無視する。続いたら観測列を作り直し (restart)、カウンタ不確実の印を立てる。
- 両側の列が矛盾する間 (aligned=False) は、相手が先に引いた分を受け側の未来ツモに含めない。
  含められない = 「未知に落とす」。矛盾が解消して重なりが確認できれば再び含める。
- NEXT 色が前回と完全に同じ連続ツモは繰り上がりを検出できない (既知の限界)。検出できなかった
  繰り上がりは、観測列の長さの不足として現れ、アライメントの再推定で拾う。
- 認識ノイズ・滑り込み演出中の途中色 (全マス同色など) 対策として、同じ NEXT 組が
  NEXT_CONFIRM_SEC 秒続き、かつ滑り込み演出中 (next_slide_motion) でないコマだけを確定の根拠にする。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Pair = tuple[int, int]

FIRST_COLOR, LAST_COLOR = 1, 5
# 同じ NEXT 組がこの秒数続いたら確定する。NEXT の滑り込み演出中・直後に途中の色が 1〜3 コマ
# (約 0.04〜0.13 秒) だけ読まれる実測 (10/13 夜 記録 g10) があるため、コマ数でなく時間で確認する。
NEXT_CONFIRM_SEC = 0.12
# 繰り上がりで説明できない NEXT 変化を、観測列の作り直しの根拠にするまでの持続秒数。
NEXT_RESYNC_SEC = 0.5
SIDE_COUNT = 2
ALIGN_MAX_SHIFT = 3        # 側どうしの手番ずれとして探す最大値 (±)
ALIGN_WINDOW = 8           # 一致の確認に使う直近の観測数
OVERLAP_MIN = 3            # 「揃っている」と言うのに必要な、矛盾なく重なった観測数


def valid_pair(pair: object) -> bool:
    """色コード 1..5 の 2 要素か。"""
    try:
        return (len(pair) == 2 and all(FIRST_COLOR <= int(v) <= LAST_COLOR for v in pair))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


@dataclass
class _Side:
    """片側の観測状態。"""

    own: list[Pair] = field(default_factory=list)       # 確認済みの NEXT 列 (末尾の 2 つが現在の NEXT/NEXT2)
    base: int = 0                                       # own[0] の絶対手番号
    queue: Optional[tuple[Pair, Pair]] = None           # 直近に確認した (NEXT, NEXT2)
    candidate: Optional[tuple[Pair, Pair]] = None
    since: float = 0.0
    shifts: list[float] = field(default_factory=list)   # 繰り上がりを確認した時刻
    restarts: int = 0

    def abs_next(self) -> int:
        """現在の NEXT の絶対手番号。"""
        return self.base + len(self.own) - 2

    def at(self, index: int) -> Optional[Pair]:
        k = index - self.base
        return self.own[k] if 0 <= k < len(self.own) else None


@dataclass(frozen=True)
class AlignEvent:
    """アライメントの変更・矛盾の記録 (診断用)。"""

    t_sec: float
    side: int
    kind: str          # 'initial' | 'realign' | 'mismatch' | 'restart'
    detail: int        # initial/realign=採用した base (補正)、mismatch=矛盾数、restart=作り直し前の長さ


class TsumoTape:
    """1 試合分の共通ツモテープ。試合が替わったら `reset()` する。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """試合境界。観測列・アライメント・不確実の印をすべて消す。"""
        self.sides = [_Side() for _ in range(SIDE_COUNT)]
        self.events: list[AlignEvent] = []
        self.uncertain = False
        self.aligned = False

    # ---- 観測 ----
    def observe(self, side: int, t_sec: float, next_pair: object, dnext_pair: object,
                motion: bool = False) -> bool:
        """1 コマの観測を渡す。NEXT の繰り上がりを確認したら True。"""
        state = self.sides[side]
        if motion or not (valid_pair(next_pair) and valid_pair(dnext_pair)):
            state.candidate = None
            return False
        queue = (self._pair(next_pair), self._pair(dnext_pair))
        if queue != state.candidate:
            state.candidate, state.since = queue, t_sec
        held = t_sec - state.since
        if held < NEXT_CONFIRM_SEC or queue == state.queue:
            return False
        return self._confirm(side, t_sec, queue, held)

    @staticmethod
    def _pair(pair: object) -> Pair:
        return (int(pair[0]), int(pair[1]))  # type: ignore[index]

    def _confirm(self, side: int, t_sec: float, queue: tuple[Pair, Pair], held: float) -> bool:
        state = self.sides[side]
        if state.queue is None:
            state.own, state.queue = [queue[0], queue[1]], queue
            state.base = self._initial_base(side, queue)
            self.events.append(AlignEvent(t_sec, side, 'initial', state.base))
            self._verify(side, t_sec)
            return False
        if queue[0] == state.queue[1]:
            state.own.append(queue[1])
            state.queue = queue
            state.shifts.append(t_sec)
            self._verify(side, t_sec)
            return True
        if held < NEXT_RESYNC_SEC:
            return False                      # 説明できない変化はまだノイズとして扱う
        self._restart(side, t_sec, queue)
        return False

    def _restart(self, side: int, t_sec: float, queue: tuple[Pair, Pair]) -> None:
        """説明できない変化が続いた: 観測列を作り直し、不確実の印を立てる。"""
        state = self.sides[side]
        self.events.append(AlignEvent(t_sec, side, 'restart', len(state.own)))
        resume = state.abs_next() + 1             # 取りこぼし数は不明。最小 1 手進んだと仮定して仮置き
        state.own, state.queue, state.base = [queue[0], queue[1]], queue, resume
        state.restarts += 1
        self.uncertain, self.aligned = True, False
        self._verify(side, t_sec)

    def _initial_base(self, side: int, queue: tuple[Pair, Pair]) -> int:
        """途中から観測を始めた側は、相手の観測列と矛盾しない手番へ合わせる (最寄りを優先)。"""
        other = self.sides[1 - side]
        if not other.own:
            return 0
        center = other.abs_next()
        for offset in sorted(range(-ALIGN_MAX_SHIFT, ALIGN_MAX_SHIFT + 1), key=abs):
            slot = center + offset
            if other.at(slot) == queue[0] and other.at(slot + 1) in (None, queue[1]):
                return slot
        return center

    # ---- 整合の検査 ----
    def _score(self, side: int, shift: int) -> tuple[int, int]:
        """side の base を shift だけ動かした時の、直近 ALIGN_WINDOW 個の (一致数, 矛盾数)。"""
        state, other = self.sides[side], self.sides[1 - side]
        hit = bad = 0
        for k in range(max(0, len(state.own) - ALIGN_WINDOW), len(state.own)):
            seen = other.at(state.base + shift + k)
            if seen is None:
                continue
            if seen == state.own[k]:
                hit += 1
            else:
                bad += 1
        return hit, bad

    def _verify(self, side: int, t_sec: float) -> None:
        """追加観測のあと、重なり部分が矛盾しないか確かめ、必要なら base を補正する。"""
        if not self.sides[1 - side].own:
            return
        hit, bad = self._score(side, 0)
        if bad == 0:
            self.aligned = hit >= OVERLAP_MIN
            return
        self.events.append(AlignEvent(t_sec, side, 'mismatch', bad))
        self.uncertain, self.aligned = True, False
        best = self._best_shift(side)
        if best is not None:
            self.sides[side].base += best
            self.events.append(AlignEvent(t_sec, side, 'realign', best))
            self.aligned = self._score(side, 0)[0] >= OVERLAP_MIN

    def _best_shift(self, side: int) -> Optional[int]:
        """矛盾なく OVERLAP_MIN 個以上重なる補正のうち、一意に最良のもの。"""
        scored = []
        for shift in range(-ALIGN_MAX_SHIFT, ALIGN_MAX_SHIFT + 1):
            hit, bad = self._score(side, shift)
            if shift != 0 and bad == 0 and hit >= OVERLAP_MIN:
                scored.append((hit, -abs(shift), shift))
        if not scored:
            return None
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
            return None                       # 同点なら決めない (一意でない)
        return scored[0][2]

    # ---- 参照 ----
    def counter(self, side: int) -> Optional[int]:
        """側の NEXT の絶対手番号 (未観測は None)。"""
        state = self.sides[side]
        return state.abs_next() if state.queue is not None else None

    def placements(self, side: int) -> int:
        """確認した NEXT 繰り上がりの回数 (= 観測できた設置数。同じ組の連続は数えられない)。"""
        return len(self.sides[side].shifts)

    def known_future(self, receiver: int) -> tuple[Pair, ...]:
        """受け側の NEXT 以降の既知ツモ (先頭 = 受け側の NEXT)。矛盾中は受け側の NEXT/NEXT2 だけ。"""
        state, other = self.sides[receiver], self.sides[1 - receiver]
        if state.queue is None:
            return ()
        out = list(state.queue)
        if not self.aligned:
            return tuple(out)
        slot = state.abs_next() + 2
        while (pair := other.at(slot)) is not None:
            out.append(pair)
            slot += 1
        return tuple(out)

    def lead(self, receiver: int) -> int:
        """受け側が NEXT/NEXT2 を超えて知っているツモ数。"""
        known = self.known_future(receiver)
        return max(0, len(known) - 2) if known else 0

    def active_pair(self, side: int) -> Optional[Pair]:
        """その側が今操作している (直前に置いた) ツモ = 現在の NEXT の 1 つ前。分からなければ None。

        連鎖の開始は設置の直後で、NEXT が繰り上がるのは連鎖の後なので、発火した組はこれになる。
        """
        state = self.sides[side]
        return state.own[-3] if len(state.own) >= 3 else None

    def next_to_place(self, side: int) -> tuple[Pair, ...]:
        """その側がこれから置くツモ列 (操作中の組 + NEXT 以降の既知ツモ)。操作中の組が分からなければ NEXT から。

        NEXT が繰り上がった直後の操作中の組は、1 つ前の NEXT。まだ置いていなければ最初の 1 手はこれを使う。
        (連鎖中の側は操作中の組をすでに置いているので、呼出側は known_future を使うこと。)
        """
        active = self.active_pair(side)
        future = self.known_future(side)
        return ((active,) + future) if active is not None and future else future

    def own_at(self, side: int, index: int) -> Optional[Pair]:
        """その側が自分で観測した絶対手番 index のツモ (的中率の照合用。未観測は None)。"""
        return self.sides[side].at(index)

    def summary(self) -> dict:
        """診断用の要約 (母数つき)。"""
        kinds: dict[str, int] = {}
        for event in self.events:
            kinds[event.kind] = kinds.get(event.kind, 0) + 1
        return dict(uncertain=self.uncertain, aligned=self.aligned, events=kinds,
                    own_len=[len(s.own) for s in self.sides],
                    placements=[len(s.shifts) for s in self.sides])
