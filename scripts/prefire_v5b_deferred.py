"""固定Lのオフライン再生用。完了前に置換される計算だけを省略する。

未来フレームは読まない。入力が変わった時点で前の未完了要求を取消す。
完了時刻まで同じ入力なら、開始時に凍結した引数で本体を実計算する。
全要求を先に実計算する再生と表示は同一。実機の処理時間短縮の主張ではない。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.prefire_best_play_v5 import BestPlayV5Layer
from src.prefire_v5_search import OK


class DeferredComputations:
    """仮想完了時刻の前には結果を参照しない、取消し可能な再生計算。"""

    def __init__(self, layer_type: type = BestPlayV5Layer) -> None:
        self.layer_type = layer_type
        self.schedule = layer_type._schedule_v5
        self.show = layer_type._show_v5
        self.pending: dict[int, tuple] = {}
        self.submitted = self.cancelled = self.completed = 0

    def install(self) -> None:
        """再生プロセス内だけで遅延計算へ置換する。"""
        adapter = self
        def schedule(layer: Any, overlay: Any, key: tuple, states: tuple, statuses: tuple,
                     elapsed: float, t_sec: float) -> None:
            adapter.enqueue(layer, overlay, key, states, statuses, elapsed, t_sec)
        def show(layer: Any, tracker: Any, key: tuple, statuses: tuple,
                 t_sec: float, game_idx: int) -> None:
            adapter.complete(layer, key, t_sec)
            adapter.show(layer, tracker, key, statuses, t_sec, game_idx)
        self.layer_type._schedule_v5, self.layer_type._show_v5 = schedule, show

    def enqueue(self, layer: Any, overlay: Any, key: tuple, states: tuple, statuses: tuple,
                elapsed: float, t_sec: float) -> None:
        """新入力が到着した時点で旧要求を取消す。将来の継続時間は参照しない。"""
        previous = self.pending.pop(id(layer), None)
        if previous is not None:
            old_key = previous[1]
            layer._cache.pop(old_key, None)
            layer._ready.pop(old_key, None)
            self.cancelled += 1
        if key in layer._cache:
            return
        if any(code != OK for code in statuses):
            self.schedule(layer, overlay, key, states, statuses, elapsed, t_sec)
            return
        snapshot = SimpleNamespace(**vars(overlay._snapshots[-1][1]))
        frozen = SimpleNamespace(_m0=overlay._m0, _build_static=overlay._build_static,
            _snapshots=[(t_sec, snapshot)], tracker=SimpleNamespace(models=overlay.tracker.models))
        self.pending[id(layer)] = (frozen, key, states, statuses, elapsed, t_sec)
        self.submitted += 1
        layer._cache[key], layer._ready[key] = (None, None), t_sec + layer.latency_sec

    def complete(self, layer: Any, key: tuple, t_sec: float) -> None:
        """同一入力の完了時刻が来た場合だけ、開始時の引数を実評価する。"""
        request = self.pending.get(id(layer))
        if request is None or request[1] != key or t_sec < layer._ready[key]:
            return
        self.pending.pop(id(layer))
        layer._cache.pop(key)
        self.schedule(layer, *request)
        self.completed += 1

    def counts(self) -> dict:
        """取消しは予測成功に数えない。未完了件数も明示する。"""
        return dict(submitted=self.submitted, cancelled=self.cancelled,
                    completed=self.completed, pending=len(self.pending))

    def restore(self) -> None:
        """テストなど同じプロセスで使う場合は元の実装へ戻す。"""
        self.layer_type._schedule_v5, self.layer_type._show_v5 = self.schedule, self.show
