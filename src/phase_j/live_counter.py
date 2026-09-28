"""ライブ専用MC応手探索。世代付き結果を次の更新で回収する。"""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing as mp
import os
from typing import Any

EMPTY_RESULT = (0.0, float('nan'), float('nan'))
JOIN_SEC = 2.0
REFRESH_SEC = 0.5
LIVE_ROLLOUTS = 30
OFFLINE_ROLLOUTS = 60


def warmup() -> None:
    """最初の探索要求までに旧評価モジュールを別processで読み込む。"""
    from .live_cpu import apply_runtime
    import json
    print('MC_CPU '+json.dumps(apply_runtime('mc')), flush=True)
    import scripts.visualize_advantage_overlay  # noqa: F401


def search(arguments: dict[str, Any]) -> tuple[int, tuple, float]:
    """独立プロセス内で既存探索をそのまま実行する。キャッシュは持ち越さない。"""
    from unittest.mock import patch
    import scripts.visualize_advantage_overlay as legacy
    arguments = dict(arguments)
    generation = arguments.pop('generation')
    rollouts = arguments.pop('rollouts', OFFLINE_ROLLOUTS)
    tracker = legacy.CounterReachTracker()
    with patch.object(legacy, 'COUNTER_N_ROLLOUTS', rollouts):
        result = tracker.update(**arguments)
    return generation, result, tracker.last_hands


class AsyncCounter:
    """実行中は一件だけ。古い世代の結果は直前値にも採用しない。"""

    def __init__(self, executor: Any = None, rollouts: int = OFFLINE_ROLLOUTS) -> None:
        self.rollouts = rollouts
        self.owns_executor = executor is None
        from .live_lifetime import protect_parent
        self.executor = executor or ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'),
            initializer=protect_parent, initargs=(os.getpid(),))
        if self.owns_executor:
            self.executor.submit(warmup)
        self.future: Future | None = None
        self.generation = 0
        self.scope: tuple | None = None
        self.job_generation: int | None = None
        self.result_generation: int | None = None
        self.request_time: float | None = None
        self.result_time: float | None = None
        self.expires_at: float | None = None
        self._last_result: tuple | None = None
        self.last_budget_sec = self.last_hands = 0.0
        self.pending = False
        self.submitted = self.accepted = self.discarded = 0
        self.error: str | None = None
        self.context: tuple | None = None

    def invalidate(self, context: tuple) -> None:
        """更新呼出のない通知でも盤面・試合境界を監視し、ABAも区別する。"""
        if context != self.context:
            self.context = context
            self.generation += 1
            self.scope = None
            self.pending = self.last_budget_sec > 0
            if self.future is not None:
                self.future.cancel()

    def _collect(self, now: float | None) -> bool:
        if self.future is None or not self.future.done():
            return False
        future, self.future = self.future, None
        expired = now is not None and self.expires_at is not None and now >= self.expires_at
        if self.job_generation != self.generation or expired:
            self.discarded += 1
            return False
        try:
            generation, result, hands = future.result()
        except Exception as error:
            self.error = str(error)
            return False
        if generation != self.generation:
            self.discarded += 1
            return False
        self._last_result, self.last_hands, self.error = result, hands, None
        self.result_generation = self.generation
        self.result_time = self.request_time
        self.accepted += 1
        self.pending = False
        return True

    def update(self, b1: Any, b2: Any, budget_sec: float = 0.0,
               next1: tuple | None = None, next2: tuple | None = None,
               t_sec: float | None = None, defender_side: str | None = None,
               threshold_ojama: float | None = None, reuse_if_board_unchanged: bool = False,
               quantize_budget_sec: bool = False) -> tuple:
        """前回採用値はpendingとして保持し、新規完了値だけ世代照合して採用する。"""
        scope = (b1.grid_bytes(), b2.grid_bytes(), next1, next2, defender_side,
                 threshold_ojama, budget_sec > 0)
        if scope != self.scope:
            self.scope = scope
            self.generation += 1
        self.last_budget_sec = budget_sec
        collected = self._collect(t_sec)
        if budget_sec <= 0:
            self._last_result, self.pending = EMPTY_RESULT, False
            self.result_generation = None
            return EMPTY_RESULT
        refresh = (not collected and t_sec is not None and self.result_time is not None
                   and t_sec - self.result_time >= REFRESH_SEC)
        self.pending = self.result_generation != self.generation or refresh
        if self.future is None and self.pending:
            arguments = dict(generation=self.generation, rollouts=self.rollouts,
                             b1=b1.copy(), b2=b2.copy(), budget_sec=budget_sec,
                             next1=next1, next2=next2, t_sec=t_sec,
                             defender_side=defender_side, threshold_ojama=threshold_ojama,
                             reuse_if_board_unchanged=reuse_if_board_unchanged,
                             quantize_budget_sec=quantize_budget_sec)
            self.future = self.executor.submit(search, arguments)
            self.job_generation, self.request_time = self.generation, t_sec
            self.expires_at = None if t_sec is None else t_sec + budget_sec
            self.submitted += 1
        return self._last_result or EMPTY_RESULT

    def status(self) -> dict[str, Any]:
        """直前値の基準世代と現在世代を別々に公開する。"""
        return dict(pending=self.pending, generation=self.generation,
                    result_generation=self.result_generation, request_time=self.request_time,
                    result_time=self.result_time,
                    submitted=self.submitted, accepted=self.accepted, discarded=self.discarded,
                    error=self.error)

    def close(self) -> None:
        """終了時も長い探索の完走を評価側で待たない。"""
        if not self.owns_executor:
            return
        processes = list(getattr(self.executor, '_processes', {}).values())
        self.executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(JOIN_SEC)


def factory(bridge: Any) -> type:
    """生成されたtrackerをbridgeに登録し、通知ごとの世代監視と終了処理へつなぐ。"""
    from .live_side_counter import SideCounter
    class LiveCounter(SideCounter):
        def __init__(self) -> None:
            # 試合リセットや補助trackerの生成でも探索processは一つに限定する。
            executor = bridge.counters[0].executor if bridge.counters else None
            super().__init__(executor, getattr(bridge, 'mc_rollouts', LIVE_ROLLOUTS))
            bridge.counters.append(self)
    return LiveCounter
