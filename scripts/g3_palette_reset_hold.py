"""同stepの実resetだけpalette vetoを保留する。原盤面と正常vetoは変更しない。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Callable
from scripts import g3_current_admission as G
from scripts.g3_agent_review import encoded, save

KEY = 'g3_palette_reset_hold'
SOURCE_SHA = '33ab8251d5b930558cc9b1a4a8b95b1e663daa4feb70e29dc438aebd8d5c05fc'
REASON = 'held_same_step_reset'
ORIGINAL_ERROR = 'palette_not_stable_ctx'
SCOPE_KEYS = ('source_id', 'run_id', 'pipe_object_id', 'side', 'frame_idx', 'time_sec')


def require(value: bool, reason: str) -> None:
    """局所の拒否理由を原ガード例外のcauseとして残す。"""
    if not value:
        raise ValueError('g3_palette_reset:' + reason)


def reset_scope(caller: Any, rec: Any, board: Any, frame: Any, region: Any) -> dict:
    """原active stepの開始scopeと、同じ入力時計の現scopeを比較する。"""
    journal, values = rec.journal, caller.f_locals
    require(caller.f_code in journal.codes, 'caller')
    pipe, side, ctx = values['self'], values['side'], values['ctx']
    require(pipe is journal.tracker._pipeline and side in ('1P', '2P'), 'pipe_side')
    sm = getattr(pipe, '_sm_' + side.lower())
    live = sm.context
    require(values['sm'] is sm and ctx is not live and ctx.state.name == 'STABLE', 'context_not_replaced')
    require(live.frame_idx == 0 and live.confirmed_board is None
            and live.state.name in ('MENU', 'STABLE'), 'not_initial_reset_context')
    require(board is ctx.confirmed_board and frame is values['frame_bgr']
            and region is values['region_for_validate'], 'argument_identity')
    clock = values['frame_idx'], values['time_sec']
    require(clock == (rec.history.frame, rec.history.time_sec) and ctx.frame_idx == clock[0], 'clock')
    item = journal.active
    require(item is not None and item['pipe'] is pipe, 'active_scope')
    before = item['scope']
    after = journal.scope(pipe, side, *clock)
    require(all(before[k] == after[k] for k in SCOPE_KEYS), 'scope_changed')
    previous, current = before['generation']['reset_epoch'], after['generation']['reset_epoch']
    require(type(previous) is int and type(current) is int and current > previous >= 0, 'reset_epoch')
    software = journal.epoch(pipe, side)
    require(type(item['epoch']) is int and type(software) is int and software >= item['epoch'] >= 0, 'software_epoch')
    return {k: after[k] for k in SCOPE_KEYS} | dict(reason=REASON, reset_epoch_before=previous,
        reset_epoch_after=current, software_epoch_before=item['epoch'], software_epoch_after=software,
        old_state=ctx.state.name, old_frame=ctx.frame_idx, live_state=live.state.name,
        live_frame=live.frame_idx, stage='veto_hold_observed', quality_gate_clear=False,
        physical_palette_certified=False, veto_applied=False)


class Hold:
    """別G3原票だけを所有し、旧Recorderの集計へ別種の行を混入しない。"""

    def __init__(self, module: Any, output: Path) -> None:
        self.module, self.output = module, output
        self.original_bound, self.original_apply = module.bound_cnn, module.apply_veto
        self.stream = (output / 'G3_PALETTE_RESET_HOLD.jsonl').open('xb')
        self.pending: tuple | None = None
        self.rows = self.applied = 0
        self.error: str | None = None

    def bound(self, caller: Any, rec: Any, board: Any, frame: Any, region: Any) -> tuple:
        """指定された一種類の原例外だけ、厳格な追加検査の後に保留する。"""
        require(self.pending is None and self.error is None, 'unfinished_or_failed')
        try:
            return self.original_bound(caller, rec, board, frame, region)
        except BaseException as original:
            if not isinstance(original, ValueError) or str(original) != ORIGINAL_ERROR:
                self.error = repr(original)
                raise
            try:
                row = reset_scope(caller, rec, board, frame, region)
            except BaseException as cause:
                self.error = repr(cause)
                raise original from cause
            try:
                self.stream.write(encoded(row) + b'\n')
                self.stream.flush()
            except BaseException as failure:
                self.error = repr(failure)
                raise
            self.rows += 1
            self.pending = (id(board), id(frame), id(region))
            return None, {k: row[k] for k in ('frame_idx', 'time_sec', 'side', 'source_id', 'run_id')}

    def apply(self, board: Any, result: Any, cnn: Any, frame: Any, region: Any) -> tuple:
        """同callの保留だけ専用理由へ変換し、旧rec.save/verifyを再利用する。"""
        require(self.error is None, 'apply_after_failure')
        if self.pending is None:
            try:
                return self.original_apply(board, result, cnn, frame, region)
            except BaseException as failure:
                self.error = repr(failure)
                raise
        try:
            require(cnn is None and self.pending == (id(board), id(frame), id(region)), 'apply_identity')
            cells = [dict(row=r, col=c, original=old, replacement=new, cnn_color=None,
                          reason=REASON, vetoed=False)
                     for r, c, old, new in self.module.candidates(board, result)]
            self.applied += 1
            return result, cells
        except BaseException as failure:
            self.error = repr(failure)
            raise
        finally:
            self.pending = None

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        """未消費・保存障害・参照復元を残し、元例外を置換しない。"""
        failure: BaseException | None = None
        restored = self.module.bound_cnn is self.original_bound and self.module.apply_veto is self.original_apply
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except BaseException as error:
            failure = error
        try:
            self.stream.close()
        except BaseException as error:
            failure = failure or error
        complete = (restored and self.pending is None and self.rows == self.applied
                    and self.error is None and failure is None and self.stream.closed)
        try:
            save(self.output / 'G3_PALETTE_RESET_STATUS.json', dict(closed=self.stream.closed,
                rows=self.rows, applied=self.applied, pending=self.pending is not None,
                references_restored=restored, complete=complete, error=self.error,
                original_error=None if body is None else repr(body), save_error=None if failure is None else repr(failure),
                quality_gate_clear=False))
        except BaseException as error:
            failure = failure or error
        self.pending = None
        if body is None:
            if failure is not None:
                raise failure
            require(complete, 'incomplete_or_reference_changed')
        return False


def install(stack: Any, state: dict, replace: Callable) -> Hold:
    """実palette Recorderの私有moduleを、既検収採録gateの後に束縛する。"""
    require(KEY not in state and G.KEY in state and state[G.KEY].current.rows == 0, 'install_order')
    rec = state['palette_evidence_veto']
    module = sys.modules[type(rec).__module__]
    require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == SOURCE_SHA, 'source_changed')
    require(all(fn.__globals__ is vars(module) and Path(fn.__code__.co_filename).resolve() == Path(module.__file__).resolve()
                for fn in (module.bound_cnn, module.apply_veto, module.wrapper)), 'original_function_owner')
    require(rec.journal is state['atomic_journal_observer'] and rec.calls == 0, 'recorder_owner_or_late')
    value = state[KEY] = Hold(module, Path(state['output']))
    stack.push(value.close)
    replace(stack, module, 'bound_cnn', value.bound)
    replace(stack, module, 'apply_veto', value.apply)
    return value
