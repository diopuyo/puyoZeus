"""画像検証だけを単一workerで実行する。入力状態とepochは所有者だけが更新する。"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from .live_verify_split import VerifyFailure
from .live_screen_errors import check_screen
from . import live_verify_entry as entry


@dataclass(frozen=True)
class ScreenResult:
    generation: int
    captured_at: float
    passed: bool
    feedback: bool
    duration_sec: float
    diagnostics: dict | None = None
    # 不合格の段 (live_verify_soft_hold が ON のときだけ verifier が残す。OFF・合格時は常に None)。
    failure: VerifyFailure | None = None


class ScreenWorker:
    """待ち行列は最大1件。世代を跨ぐ結果・例外は採用しない。"""

    def __init__(self, verifier: Callable[[Any], bool]) -> None:
        self.verifier = verifier
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='screen-verify')
        self.future: Future | None = None
        self.generation = 0
        self.pending_generation = 0

    def invalidate(self) -> None:
        self.generation += 1

    def submit(self, image: Any, captured_at: float, size: Any = None) -> bool:
        if self.future is not None:
            return False
        # captureが内部バッファを使い回しても検証対象が途中で変化しない。
        snapshot = image.copy()
        self.pending_generation = self.generation
        self.future = self.pool.submit(self._verify, snapshot, captured_at, self.generation, size)
        return True

    def _verify(self, image: Any, captured_at: float, generation: int, size: Any = None) -> ScreenResult:
        from .live_border_grace import verify_screen
        started = perf_counter()
        passed = check_screen('screen_verify_worker', verify_screen,
                              self.verifier, image, captured_at, size, generation)
        failure = getattr(self.verifier, 'failure', None)
        return ScreenResult(generation, captured_at, passed,
                            bool(getattr(self.verifier, 'feedback', False)), perf_counter()-started,
                            getattr(self.verifier, 'diagnostics', None),
                            failure if not passed and isinstance(failure, VerifyFailure) else None)

    def poll(self) -> ScreenResult | None:
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        if self.pending_generation != self.generation:
            return None
        return future.result()

    def close(self) -> None:
        self.invalidate()
        self.pool.shutdown(wait=True, cancel_futures=True)


class AsyncScreenState:
    """既存のjudgeを認識スレッドから呼び、初回確認と連続失敗規則を維持する。"""

    def __init__(self, verifier: Callable, judge: Callable, period: float,
                 wait_initial: bool = True) -> None:
        self.worker = ScreenWorker(verifier)
        self.judge, self.period = judge, period
        self.entry_fast_retry = entry.enabled()
        self.verified, self.fail_streak, self.feedback = False, 0, False
        self.failure: VerifyFailure | None = None   # 最新の不合格の段 (保留の種別判定に使う)
        self.last_verify = float('-inf')
        self.size: tuple[int, int] | None = None
        self.latest_result: ScreenResult | None = None
        self.wait_initial = wait_initial
        self.application_delay_sec = 0.0

    def update(self, image: Any, size: Any, captured: float, transport_ready: bool) -> tuple[bool, int]:
        changed = self.size is not None and self.size != size
        if not transport_ready or changed:
            self.worker.invalidate()
            self.verified, self.fail_streak, self.feedback = False, 0, False
            self.failure = None
            self.last_verify = float('-inf')
        self.size = size
        self.latest_result = self.worker.poll()
        if self.latest_result is not None and transport_ready:
            result = self.latest_result
            self.verified, self.fail_streak = self.judge(result.passed, self.verified, self.fail_streak)
            self.feedback = result.feedback
            self.failure = result.failure
            self.application_delay_sec = max(0.0, captured-result.captured_at)
        period = entry.verification_period(self.verified, self.period, self.entry_fast_retry)
        if transport_ready and not changed and captured-self.last_verify >= period:
            if self.worker.submit(image, captured, size):
                self.last_verify = captured
                if not self.verified and self.wait_initial:
                    # 初回・保留中は認識未開始。同じ画像で確認を完了し、入口の欠落を防ぐ。
                    self.worker.future.result()
                    self.latest_result = self.worker.poll()
                    self.verified, self.fail_streak = self.judge(
                        self.latest_result.passed, self.verified, self.fail_streak)
                    self.feedback = self.latest_result.feedback
                    self.failure = self.latest_result.failure
        return self.verified, self.fail_streak

    def close(self) -> None:
        self.worker.close()
