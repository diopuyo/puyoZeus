"""単一 worker の完了時計 (dl1k 研究専用・既定 OFF)。

v4.15 の OBS 再生 (`LatestQueue` と `Replay.accept`) と同じ規則: 実行中 1 件は止めず、
未実行の要求は最新 1 件へまとめる (置換)。撃ち合い外の入力が来たら待ちを捨てる。
計算は要求が実際に始まった時点で 1 回だけ呼び、その要求の入力位置の状態で計算させる。
利用は「同じ撃ち合いの鍵・完了済み (available_at <= 利用時刻)・入力から 500ms 以内」だけ。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

MS = 1000.
FRESH_SEC = .5
REASON_OK = 'ok'
REASON_NOT_YET = 'not_yet'
REASON_STALE = 'stale'


@dataclass(frozen=True)
class Request:
    """計算要求。t は到着 (学習=論理入力時刻、OBS=保存受信時刻)、input_t は鮮度の起点。"""
    t: float
    key: tuple
    index: int
    input_t: float


@dataclass(frozen=True)
class Publication:
    """完了した計算。available_at より前には誰も読めない。"""
    request: Request
    started_at: float
    available_at: float
    actual_ms: float
    value: Any


Compute = Callable[[Request, float], tuple[Any, float]]


@dataclass
class WorkerClock:
    """計算は compute(request, started_at) -> (値, 実計算ms)。待ち時間は started_at に含める。"""
    compute: Compute
    busy_until: float = float('-inf')
    waiting: Request | None = None
    published: dict = field(default_factory=dict)
    counts: dict = field(default_factory=lambda: dict(requests=0, executed=0, replaced=0, cancelled=0))
    last_t: float = float('-inf')

    def _now(self, t: float) -> None:
        if t < self.last_t:
            raise ValueError('時計の巻戻し')
        self.last_t = t

    def _start(self, request: Request, start: float) -> Publication:
        value, actual_ms = self.compute(request, start)
        if not actual_ms >= 0:
            raise ValueError('負または非数の計算時間')
        pub = Publication(request, start, start+actual_ms/MS, float(actual_ms), value)
        self.published.setdefault(request.key, []).append(pub)
        self.busy_until = pub.available_at
        self.counts['executed'] += 1
        return pub

    def flush(self, t: float) -> None:
        """worker が t までに空いたら、待ちの要求を空いた時刻 (到着より前ではない) に始める。"""
        self._now(t)
        if self.waiting is not None and self.busy_until <= t:
            request, self.waiting = self.waiting, None
            self._start(request, max(self.busy_until, request.t))

    def submit(self, request: Request) -> None:
        """撃ち合い中の要求。待ちは最新 1 件へ置換し、空いていればすぐ始める。"""
        self.flush(request.t)
        self.counts['requests'] += 1
        self.counts['replaced'] += self.waiting is not None
        self.waiting = request
        self.flush(request.t)

    def cancel_waiting(self, t: float) -> None:
        """撃ち合い外の入力。v4.15 と同じく、空いていれば先に待ちを流し、残った待ちだけ捨てる。"""
        self.flush(t)
        self.counts['cancelled'] += self.waiting is not None
        self.waiting = None

    def idle(self, t: float) -> bool:
        return self.waiting is None and self.busy_until <= t

    def at(self, t: float, key: tuple) -> tuple[Publication | None, str]:
        """利用時刻 t に読める、同じ鍵で最新の完了済み計算。鮮度超過は理由つきで返さない。"""
        self.flush(t)
        for pub in reversed(self.published.get(key, [])):
            if pub.available_at > t:
                continue
            fresh = 0. <= t-pub.request.input_t <= FRESH_SEC
            return (pub, REASON_OK) if fresh else (None, REASON_STALE)
        return None, REASON_NOT_YET

    def summary(self) -> dict:
        """要求・実行・置換・取消と、終了時に未実行で残った待ち。"""
        times = [p.actual_ms for pubs in self.published.values() for p in pubs]
        return dict(self.counts, unexecuted_at_end=int(self.waiting is not None),
                    publications=len(times), keys=len(self.published))
