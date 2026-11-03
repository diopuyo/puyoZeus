"""全観測FIFOの状態processと、最新要求だけを計算するprocess。"""
from __future__ import annotations

from collections import deque
from copy import copy
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from queue import Empty
import signal
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from src.b3_contents import contents_from_row
from src.b3_profile import MS, profile, span
from src.b3_realtime import RealtimeRows
from src.b3_timing import TimingTable
from src.production_config import B3_WORKER_GRACEFUL_STOP_DEFAULT

POLL_SEC = 0.01
START_TIMEOUT_SEC = 120.0
CLOSE_TIMEOUT_SEC = 10.0
AUDIT_CLOSE_TIMEOUT_SEC = 120.0
POLL_LIMIT = 64
# 作業 process が Ctrl+C を無視する設定 (既定 OFF)。spawn 子へは launcher の設定から環境変数で届く。
GRACEFUL_STOP_ENV = 'PUYO_B3_WORKER_GRACEFUL_STOP'
# Ctrl+C を無視した作業 process が、親の強制終了後に孤児として残らないための親の生存確認の間隔。
PARENT_CHECK_SEC = 1.0
STATE_NAME, COMPUTE_NAME = 'b3-state', 'b3-compute'   # status の exit_codes / unreaped / forced_termination の名前
# b3_async_status.json の finished が null のときの理由 (名前つきで残す)。
MISSING_INTERRUPTED = 'worker_interrupted'        # Ctrl+C (KeyboardInterrupt) が作業 process にも届いた
MISSING_FATAL = 'worker_fatal'                    # 作業 process が別の例外で終了した
MISSING_FORCED = 'forced_termination'             # 終了待ちが期限を超え強制終了した
MISSING_NO_DONE = 'exited_without_done'           # done も fatal も返さず終了した


def graceful_stop_enabled() -> bool:
    return os.environ.get(GRACEFUL_STOP_ENV, str(int(B3_WORKER_GRACEFUL_STOP_DEFAULT))) == '1'


def _protect_from_interrupt() -> None:
    """Ctrl+C はコンソールを共有する全 process に届く。作業 process が KeyboardInterrupt で先に死ぬと、
    親が close() で終了の合図 (None) を送っても done を返せず、b3_async_status.json の finished が null になる。
    ON のときだけ SIGINT を無視し、止まるのは親の合図か親の消滅のときだけにする。OFF は従来どおり。"""
    if graceful_stop_enabled():
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _parent_gone() -> bool:
    """自分を起動した process が終了したか (親の無い単体実行では False)。"""
    parent = mp.parent_process()
    return parent is not None and not parent.is_alive()


def missing_reason(fatal: list[str], forced: list[str]) -> str:
    """finished が null になった理由。KeyboardInterrupt は Ctrl+C が作業 process にも届いた印。"""
    if any(text.startswith('KeyboardInterrupt') for text in fatal):
        return MISSING_INTERRUPTED
    if forced:
        return MISSING_FORCED
    return MISSING_FATAL if fatal else MISSING_NO_DONE


def snapshot(rows: RealtimeRows) -> bytes:
    """特徴が参照する状態を固定。可変台帳本体と先読み口は配送しない。"""
    state = copy(rows.state)
    state.lean, state.lean_rows = None, []
    state.sl = _ledger_snapshot(state)
    return pickle.dumps((state, rows.trigger), protocol=pickle.HIGHEST_PROTOCOL)


def _ledger_snapshot(state: Any) -> Any:
    """上限補完が読む当該連鎖の生成量だけを固定し、台帳本体は配送しない。"""
    from src.b3_timeaxis import _episode_generated

    if state.sl is None:
        return None
    if getattr(state.sl, 'core', None) is None:
        return True
    chains = []
    for index, side in enumerate(state.sides):
        if side.episode is not None and side.episode.active:
            step = SimpleNamespace(t_formula=side.episode.first_trigger,
                                   generated=_episode_generated(state.sl, side))
            chains.append(SimpleNamespace(side=index, steps=(step,)))
    core = SimpleNamespace(chains=chains, st=(SimpleNamespace(chain=None), SimpleNamespace(chain=None)))
    return SimpleNamespace(core=core)


def calculate(request: dict, rows: RealtimeRows, model: Any,
              measure: bool = True) -> dict:
    """共有可変状態を持たず、同期版と同じ特徴・単行推論を実行する。"""
    result = {k: v for k, v in request.items() if k != 'snapshot'}
    result['compute_wait_ms'] = (perf_counter()-request['requested_at'])*MS
    result.update(error=None, raw_probability=None, row=None, contents=None)
    telemetry = os.environ.get('PUYO_B3_TIME_GATE') == '1'
    if telemetry:
        from src.b3_search import _search_cached
        before_cache = _search_cached.cache_info()
    with profile(measure) as costs:
        try:
            with span('feature_only'):
                rows.state, rows.trigger = pickle.loads(request['snapshot'])
                row = rows.build()
            with span('inference'):
                probability = None if request.get('_features_only', False) else model.predict_row(row)
                if not request.get('_features_only', False) and (not math.isfinite(probability) or not 0 <= probability <= 1):
                    raise ValueError('B3の予測が有限な確率ではありません')
            result.update(row=row, raw_probability=probability, contents=contents_from_row(row))
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
    result['timings'] = costs
    if telemetry:
        cache = _search_cached.cache_info()
        result['search_audit'] = dict(cache_hits=cache.hits-before_cache.hits,
            cache_misses=cache.misses-before_cache.misses, cache_size=cache.currsize,
            completion_candidates=[len(s.episode.prediction.options) if s.episode is not None
                and s.episode.prediction is not None else 0 for s in rows.state.sides])
    result['completed_at'] = perf_counter()
    return result


def _compute(connection: Any, model_path: Path, table: TimingTable,
             video: str, measure: bool) -> None:
    """spawn先でモデルとキャッシュを専有する。"""
    from src.phase_j.live_power import apply_power_policy
    apply_power_policy()
    _protect_from_interrupt()
    try:
        from threadpoolctl import threadpool_limits
        from src.b3_ensemble import ensemble_enabled, load_ensemble_model
        from src.b3_train_model import load_model
        # 既定OFFは従来どおり load_model (1 本)。b3_ensemble ON (環境変数で届く) のときだけマニフェストから平均の入れ物を作る。
        model = load_ensemble_model(model_path) if ensemble_enabled() else load_model(model_path)
        rows = RealtimeRows(table, video, measure_timing=measure)
        connection.send({'ready': True})
        with threadpool_limits(limits=1):
            while (request := connection.recv()) is not None:
                connection.send(calculate(request, rows, model, measure))
    except BaseException as exc:
        connection.send({'fatal': f'{type(exc).__name__}: {exc}'})
    finally:
        connection.close()


class _StateLoop:
    def __init__(self, table: TimingTable, video: str, drop: bool,
                 measure: bool, audit: Any) -> None:
        self.rows = RealtimeRows(table, video, measure_timing=measure)
        self.epoch: Any = None
        self.pending: deque[dict] = deque()
        self.drop, self.measure, self.audit = drop, measure, audit
        self.counts = dict(inputs=0, triggers=0, discarded=0, computed=0)

    def accept(self, packet: bytes) -> None:
        frame, events, queues, epoch, emit, metadata = pickle.loads(packet)
        if self.drop and (epoch != self.epoch or frame['game_idx'] != self.rows.state.game_idx):
            self.counts['discarded'] += len(self.pending)
            self.pending.clear()
        if epoch != self.epoch:
            self.rows = RealtimeRows(self.rows.table, self.rows.video_id,
                                     measure_timing=self.measure)
            self.epoch = epoch
        wait = (perf_counter()-metadata['submitted_at'])*MS
        with profile(self.measure) as costs:
            with span('state'):
                ready = self.rows.advance(frame, events, queues, emit)
                frozen = snapshot(self.rows) if ready else None
        if self.drop and self.rows.state.started:
            self.counts['discarded'] += len(self.pending)
            self.pending.clear()
        self.counts['inputs'] += 1
        if self.audit is not None:
            self.audit.write(json.dumps(dict(frame=frame['frame'], t_sec=frame['t_sec'],
                epoch=epoch, game=frame['game_idx'], trigger=self.rows.trigger if ready else 0,
                state_wait_ms=wait, **costs))+'\n')
        if not ready:
            return
        self.counts['triggers'] += 1
        if self.drop and self.pending:
            self.counts['discarded'] += len(self.pending)
            self.pending.clear()
        self.pending.append(dict(snapshot=frozen, epoch=epoch,
            game=frame['game_idx'], frame=frame['frame'], t_sec=frame['t_sec'],
            trigger=self.rows.trigger, state_timings=costs, state_wait_ms=wait,
            requested_at=perf_counter(), **metadata))


def _pump(inputs: Any, outputs: Any, connection: Any,
          state: _StateLoop, include_rows: bool) -> None:
    busy = ended = False
    watch_parent, next_check = graceful_stop_enabled(), perf_counter()+PARENT_CHECK_SEC
    while not ended or busy or state.pending:
        if watch_parent and perf_counter() >= next_check:
            if _parent_gone():
                raise RuntimeError('親processが終了しました')
            next_check = perf_counter()+PARENT_CHECK_SEC
        if connection.poll():
            result = connection.recv()
            if 'fatal' in result:
                raise RuntimeError(result['fatal'])
            busy = False
            state.counts['computed'] += 1
            requested_row = result.pop('_include_row', False)
            result.pop('_features_only', None)
            if not include_rows and not requested_row:
                row = result.pop('row')
                result['mode'] = None if row is None else row['mode_exchange']
            outputs.put(result)
        if not ended:
            try:
                packet = inputs.get(timeout=POLL_SEC)
                if packet is None:
                    ended = True
                else:
                    state.accept(packet)
            except Empty:
                pass
        if not busy and state.pending:
            connection.send(state.pending.popleft())
            busy = True
        if ended and busy:
            connection.poll(POLL_SEC)
    outputs.put({'done': state.counts})


def _state_worker(inputs: Any, outputs: Any, connection: Any, table: TimingTable,
                  video: str, drop: bool, measure: bool, output: Path | None,
                  include_rows: bool) -> None:
    from src.phase_j.live_power import apply_power_policy
    apply_power_policy()
    audit = None
    _protect_from_interrupt()
    try:
        if not connection.poll(START_TIMEOUT_SEC):
            raise TimeoutError('B3計算processの起動待ちが期限を超えました')
        ready = connection.recv()
        outputs.put(ready)
        if 'fatal' in ready:
            return
        if output is not None:
            from src.phase_j.live_obs_audit import open_audit
            audit = open_audit(output/'b3_state_timing.jsonl', buffering=-1)
        state = _StateLoop(table, video, drop, measure, audit)
        _pump(inputs, outputs, connection, state, include_rows)
    except BaseException as exc:
        outputs.put({'fatal': f'{type(exc).__name__}: {exc}'})
    finally:
        if audit is not None:
            audit.close()
        try:
            connection.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()


class AsyncB3:
    """評価スレッドは無制限FIFOへの投入と到着済み結果の読取だけを行う。"""

    def __init__(self, model: Path, table: TimingTable, video: str = 'live',
                 drop: bool = True, measure: bool = True,
                 output: Path | None = None, include_rows: bool = False) -> None:
        ctx = mp.get_context('spawn')
        self.inputs, self.outputs = ctx.Queue(), ctx.Queue()
        connection, child = ctx.Pipe()
        self.compute = ctx.Process(target=_compute, args=(child, model, table, video, measure), name=COMPUTE_NAME)
        self.process = ctx.Process(target=_state_worker, args=(self.inputs, self.outputs,
            connection, table, video, drop, measure, output, include_rows), name=STATE_NAME)
        self.finished = False
        self.closed = False
        self.output, self.submitted = output, 0
        self.tail: list[dict] = []
        self.fatals: list[str] = []    # 作業 process の終了理由 (status へ残す)
        self.summary: dict | None = None
        try:
            self.compute.start()
            self.process.start()
            connection.close()
            child.close()
            ready = self.outputs.get(timeout=START_TIMEOUT_SEC)
            if 'fatal' in ready:
                raise RuntimeError(ready['fatal'])
        except BaseException:
            self.close()
            raise

    def submit(self, frame: dict, events: tuple = (), queues: tuple = (),
               epoch: Any = None, emit: bool = True, captured_at: float | None = None,
               features_only: bool = False) -> None:
        if self.finished or self.closed:
            raise RuntimeError('終了済みB3へ入力できません')
        metadata = dict(submitted_at=perf_counter(), captured_at=captured_at)
        if features_only:
            metadata.update(_features_only=True, _include_row=True)
        self.inputs.put_nowait(pickle.dumps((frame, events, queues, epoch, emit, metadata),
                                           protocol=pickle.HIGHEST_PROTOCOL))
        self.submitted += 1

    def poll(self) -> list[dict]:
        results = []
        for _ in range(POLL_LIMIT):
            try:
                results.append(self.outputs.get_nowait())
            except Empty:
                break
        if not results and (not self.process.is_alive() or not self.compute.is_alive()) and not self.finished:
            results.append({'fatal': 'B3状態processが終了しました'})
        self.fatals.extend(str(result['fatal']) for result in results if 'fatal' in result)
        return results

    def finish(self) -> None:
        """検証時の末尾drain。通常更新中は待たない。"""
        if not self.finished:
            self.inputs.put_nowait(None)
            self.finished = True

    def close(self) -> None:
        if self.closed:
            return
        self.finish()
        timeout = AUDIT_CLOSE_TIMEOUT_SEC if os.environ.get('PUYO_OBS_FAST_AUDIT') == '1' else CLOSE_TIMEOUT_SEC
        deadline = perf_counter()+timeout
        while self.process.pid is not None and self.process.is_alive() and perf_counter() < deadline:
            self._drain_tail()
            self.process.join(POLL_SEC)
        self._drain_tail()
        forced = []
        for worker in (self.process, self.compute):
            if worker.pid is None:
                continue
            worker.join(max(POLL_SEC, deadline-perf_counter()))
            if worker.is_alive():
                forced.append(worker.name)
                worker.terminate()
                worker.join(CLOSE_TIMEOUT_SEC)
        for queue in (self.inputs, self.outputs):
            queue.cancel_join_thread()
            queue.close()
        self.closed = True
        if self.output is not None:
            (self.output/'b3_async_status.json').write_text(json.dumps(
                self.stop_status(forced), ensure_ascii=False, indent=2), encoding='utf-8')

    def stop_status(self, forced: list[str]) -> dict:
        """停止の結果。既存の鍵 (submitted / finished / forced_termination) はそのまま、回収の事実を足す:
        exit_codes は作業 process の終了コード (None = 回収できていない)、unreaped はその名前。
        finished が null のときは理由を名前つきで残す (missing_reason)。"""
        workers = (self.process, self.compute)
        status = dict(submitted=self.submitted, finished=self.summary, forced_termination=forced,
                      exit_codes={worker.name: worker.exitcode for worker in workers},
                      unreaped=[worker.name for worker in workers if worker.pid is not None and worker.exitcode is None],
                      graceful_stop=graceful_stop_enabled())
        if self.summary is None:
            status.update(finished_missing_reason=missing_reason(self.fatals, forced), worker_fatal=list(self.fatals))
        return status

    def _drain_tail(self) -> None:
        for result in self.poll():
            if 'done' in result:
                self.summary = result['done']
            else:
                self.tail.append(result)
