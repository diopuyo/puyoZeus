"""コマ落ちに強い数え方の共通部品 (仮想 30Hz の刻み)。

設計: <work>/frame_drop_robust_20261023/DESIGN.md §2・§4.1。

- **刻み**は名目の処理間隔 (動画は stride、OBS 実機は 1/30 秒)。番号は整数 k。
- 「処理した回数」で数える判定に、並べて「進んだ刻みの数」を持たせる。
  1 回数えるごとに、**前回のパイプライン呼び出しでも同じカウンタが当たっていたら**
  Δk (= 今の刻み − 前回の呼び出しの刻み、最低 1) を、そうでなければ 1 を足す。
- 成立は「回数 ≥ N」または「回数 ≥ m かつ 刻み ≥ N」。m = max(2, ⌊N/4⌋) (実観測の下限)。
- 全コマ処理では毎回 Δk = 1 なので刻み = 回数となり、判定は今と同じになる (要件 E1)。

状態を持つのは小さな影 (TickShadow) だけで、判定そのものは状態を持たない関数にしている。
既存の回数カウンタは書き換えず、その「新しい値」を見て刻みを並べて数える (影の方式)。
既存のリセット箇所を 1 つ見落としても、回数が連続していなければ刻み = 回数へ戻るので、
判定は回数どおり (今のまま) に倒れる。
"""
from __future__ import annotations

from dataclasses import dataclass

# 実観測の下限 m = max(MIN_OBS_FLOOR, N // MIN_OBS_DIVISOR) (設計 §2.3)。
# 2 = 1 回の誤読を打ち消すのに要る最小の観測数。4 = 「処理率 1/4 までは時間どおりに成立」の設計点
# (21c 実機の処理率は約 35%、5 刻み以上の間隔が 9%)。値は先に決め、測定に合わせて逆算しない。
MIN_OBS_FLOOR: int = 2
MIN_OBS_DIVISOR: int = 4
# 1 回の呼び出しで進む刻みの最小値。同じ刻みが 2 回来ても (OBS の丸め) 1 として数える。
MIN_TICK_DELTA: int = 1


def min_obs(n: int) -> int:
    """実観測の下限 m = max(2, ⌊N/4⌋)。N=2〜8 なら 2、N=18 なら 4、N=60 なら 15。"""
    return max(MIN_OBS_FLOOR, int(n) // MIN_OBS_DIVISOR)


def tick_of(frame_idx: int, origin_idx: int, step: int) -> int:
    """コマ番号から整数の刻み番号へ。動画では index − origin が stride の倍数なので割り切れる。"""
    return (int(frame_idx) - int(origin_idx)) // int(step)


def tick_reached(count: int, ticks: int, n: int, m: int | None = None) -> bool:
    """型 A の成立条件: 回数 ≥ N、または 回数 ≥ m かつ 刻み ≥ N。m は省略時 min_obs(N)。"""
    floor = min_obs(n) if m is None else int(m)
    return count >= n or (count >= floor and ticks >= n)


def votes_reached(observations: int, weight: int, need: int) -> bool:
    """型 C の票の条件: 別々の観測 ≥ need、または 観測 ≥ m かつ 重みの合計 ≥ need。

    全コマでは各観測の重みが 1 なので weight = observations となり、従来の「票数 ≥ need」と同じ。
    need が m 未満 (例: ネクスト裏付けの 1 票) のときも第 1 項がそのまま今の判定になる。
    """
    return observations >= need or (observations >= min_obs(need) and weight >= need)


@dataclass(frozen=True)
class TickStamp:
    """1 回のパイプライン呼び出しの刻み。now = 今の刻み、prev_call = 前回の呼び出しの刻み (初回 None)。"""

    now: int
    prev_call: int | None = None

    def delta(self) -> int:
        """前回の呼び出しから進んだ刻み (最低 1)。初回は 1。"""
        if self.prev_call is None:
            return MIN_TICK_DELTA
        return max(MIN_TICK_DELTA, self.now - self.prev_call)

    def weight_after(self, last_hit: int | None) -> int:
        """前回の呼び出しでも当たっていた (last_hit == prev_call) なら Δk、そうでなければ 1。"""
        if last_hit is not None and last_hit == self.prev_call:
            return self.delta()
        return MIN_TICK_DELTA


class TickClock:
    """パイプラインが 1 つだけ持つ時計。呼び出しごとに TickStamp を返す。"""

    def __init__(self, step: int) -> None:
        if type(step) is not int or step < 1:
            raise ValueError(f'media_tick_step は 1 以上の整数: {step!r}')
        self.step = step
        self.origin: int | None = None
        self.last: int | None = None

    def advance(self, frame_idx: int) -> TickStamp:
        """今の呼び出しの刻みを返し、前回の呼び出しの刻みを更新する。最初のコマを原点にする。"""
        if self.origin is None:
            self.origin = int(frame_idx)
        stamp = TickStamp(tick_of(frame_idx, self.origin, self.step), self.last)
        self.last = stamp.now
        return stamp


@dataclass
class TickShadow:
    """既存の回数カウンタ 1 つに並べる刻みの影 (回数・刻み・最後に当たった刻み)。"""

    count: int = 0
    ticks: int = 0
    last_tick: int | None = None

    def observe(self, count_now: int, stamp: TickStamp) -> int:
        """既存カウンタの新しい値を見て刻みを更新し、刻みの合計を返す。

        回数が 1 だけ増えた (同じ並びが続いた) ときだけ刻みを足す。それ以外 (リセット・飛び) は
        刻み = 回数へ戻す (回数どおりの判定に倒れる安全側)。
        """
        count_now = int(count_now)
        if count_now <= 0:
            self.count, self.ticks, self.last_tick = 0, 0, None
            return 0
        if count_now == self.count + 1 and self.last_tick is not None:
            self.ticks += stamp.weight_after(self.last_tick)
        else:
            self.ticks = count_now
        self.count, self.last_tick = count_now, stamp.now
        return self.ticks


# 診断の数え (判定には一切使わない)。刻みモードの経路だけが書く (OFF では空のまま)。
# 鍵 '<箇所>.evaluated' = 刻みの判定をした回数 (母数)、'<箇所>.early' = 刻みで成立したが
# 回数では未成立だった回数。全コマ処理では early が必ず 0 になる (要件 E1 の直接の計器)。
AUDIT: dict[str, int] = {}


def audit_note(site: str, early: bool) -> None:
    """診断の数えに 1 件足す (母数と、刻みで早まった件数)。"""
    key = site+'.evaluated'
    AUDIT[key] = AUDIT.get(key, 0)+1
    if early:
        key = site+'.early'
        AUDIT[key] = AUDIT.get(key, 0)+1


def audit_snapshot() -> dict[str, int]:
    """診断の数えの写し (鍵の順に並べる)。"""
    return dict(sorted(AUDIT.items()))


def audit_reset() -> None:
    """診断の数えを空にする (試験用)。"""
    AUDIT.clear()


def site_enabled(site: str) -> bool:
    """R06・R10・R15 のうち刻みで数える箇所か (production_config.LIVE_FRAME_DROP_ROBUST_SITES が単一情報源)。"""
    from src.production_config import LIVE_FRAME_DROP_ROBUST_SITES
    return site in LIVE_FRAME_DROP_ROBUST_SITES


def append_weighted(history: list, weights: list[int], item: object, weight: int, size: int) -> None:
    """型 C の窓: (値, 重み) を足し、重みの合計が size を超えた分を古い側から削る。

    古い観測の重みを部分的に削る (窓 = 直近 size 刻み)。重みが 0 になった観測は捨てる。
    全コマでは重みがすべて 1 なので「件数が size を超えたら最古を 1 件捨てる」と同じ。
    """
    history.append(item)
    weights.append(int(weight))
    excess = sum(weights) - int(size)
    while excess > 0 and weights:
        if weights[0] <= excess:
            excess -= weights.pop(0)
            history.pop(0)
        else:
            weights[0] -= excess
            excess = 0


__all__ = ['AUDIT', 'MIN_OBS_DIVISOR', 'MIN_OBS_FLOOR', 'MIN_TICK_DELTA', 'TickClock', 'TickShadow',
           'TickStamp', 'append_weighted', 'audit_note', 'audit_reset', 'audit_snapshot', 'min_obs',
           'site_enabled', 'tick_of', 'tick_reached', 'votes_reached']
