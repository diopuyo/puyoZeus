"""初期化前の原O/J追加を同call参照で保持。配置や会計を発行しない。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from types import CodeType, FrameType, SimpleNamespace
from typing import Any

SIDES = ('1P','2P')
SLOT_COUNT = 1


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


class Witness:
    def __init__(self, provider: Any, controller: Any) -> None:
        self.provider, self.controller = provider,controller
        self.slots: dict[str,Any] = {}
        self.adoptions: list[Any] = []
        self.unwitnessed_appends: list[Any] = []
        self.error: str | None = None
        self.closed = False

    def caller(self, row: Any, caller: Any) -> None:
        method = type(self.provider.journal).wrap_enqueue
        code = next(c for c in method.__code__.co_consts if isinstance(c,CodeType) and c.co_name=='enqueue')
        require(type(caller) is FrameType and caller.f_code is code
            and caller.f_globals is method.__globals__, 'adoption_original_enqueue_caller')
        values = caller.f_locals
        require(values['self'] is self.provider.journal, 'adoption_journal_caller')
        require((values['side'],values['frame'],values['clock'])==(row['side'],row['frame_idx'],row['time_sec'])
            and values['token']==row['token'] and values['added'] is row['added_occurrence_tokens'],
                'adoption_enqueue_caller_values')

    def capture(self, row: Any, caller: Any) -> None:
        if row.get('kind')!='enqueue' or row.get('status')!='returned' or row['side'] in self.controller.history:
            return
        side, provider = row['side'],self.provider
        require(side in SIDES and not self.closed and self.error is None, 'adoption_witness_state')
        tokens = row.get('fifo_occurrence_tokens',[])
        old = self.slots.get(side)
        if old is not None and old['token'] not in tokens:
            self.slots.pop(side)
        added = row.get('added_occurrence_tokens',[])
        if not added:
            return
        self.caller(row,caller)
        if not self.creation_in_scope(row):
            return
        live = provider.link.latest.get(side)
        require(live is not None and live['frame']==row['frame_idx'], 'adoption_emit_missing')
        pipe = live['pipe']
        owner = provider.owner(pipe,side,live['epoch'])
        require(caller.f_locals['pipe'] is pipe and caller.f_locals['saved'] is owner,
                'adoption_enqueue_owner')
        scope = provider.journal.scope(pipe,side,row['frame_idx'],row['time_sec'])
        provider.check_enqueue(row,scope,live['epoch'],owner)
        self.current_creation(row,scope,owner,pipe,live)
        require(provider.link.attached and provider.link.error is None and not provider.journal.errors,
                'adoption_upstream_failed')
        require(live['committed'] and live['reason']=='occurrence_committed'
            and live['baseline_frame']==row['frame_idx'] and live['pending'] is None
            and live['armed_at'] is None, 'adoption_not_native_directional_commit')
        if len(owner['refs'])!=SLOT_COUNT or row['before']['pending_tsumo']:
            self.slots.pop(side,None)
            return
        require(owner['tokens']==added and len(added)==SLOT_COUNT, 'adoption_creation_token')
        require(live['queue'] is owner['queue'] and len(live['refs'])==SLOT_COUNT
            and live['refs'][0] is owner['refs'][0], 'adoption_creation_reference')
        event = live['event']
        require(type(event) is provider._parts.O.MotionCandidate and event.available_frame<=row['frame_idx'],
                'adoption_creation_candidate')
        self.slots[side] = dict(token=added[0],pair=owner['refs'][0],queue=owner['queue'],pipe=pipe,
            scope=deepcopy(scope),epoch=live['epoch'],segment=live['segment'],frame=row['frame_idx'],
            clock=row['time_sec'],event=deepcopy(vars(event)),journal=deepcopy(row),used=False)

    def creation_in_scope(self, row: Any) -> bool:
        adapter = self.provider.link.adapter
        require(adapter.enabled is True, 'adoption_directional_adapter_disabled')
        first,last = adapter.native.FIRST_FRAME,adapter.native.LAST_FRAME
        require(type(first) is int and type(last) is int and first<=last, 'adoption_native_scope')
        if first<=row['frame_idx']<=last:
            return True
        self.slots.pop(row['side'],None)
        self.unwitnessed_appends.append(dict(frame=row['frame_idx'],side=row['side'],
            tokens=list(row['added_occurrence_tokens']),native_first=first,native_last=last,
            reason='outside_original_directional_scope',adoption_permission=False))
        return False

    def current_creation(self, row: Any, scope: Any, owner: Any, pipe: Any, live: Any) -> None:
        side = row['side']
        sm = getattr(pipe,'_sm_'+side.lower())
        view = SimpleNamespace(frame=row['frame_idx'],clock=row['time_sec'],
            scope=(scope['source_id'],scope['run_id'],live['epoch'],id(pipe),id(sm),
                scope['generation']['reset_epoch'],side),
            queue=owner['queue'],refs=tuple(owner['refs']))
        require(self.provider.link.current(view) is live, 'adoption_creation_current_identity')

    def same_cycle(self, live: Any, saved: Any, view: Any) -> None:
        require(live['baseline_frame']==saved['frame'] and live['pending'] is None,
                'adoption_new_cycle_started')
        event = live['event']
        require(type(event) is self.provider._parts.O.MotionCandidate
            and vars(event)==saved['event'], 'adoption_cycle_event_changed')
        immediate = view.frame==saved['frame']
        require(live['committed'] is immediate and live['reason']==
            ('occurrence_committed' if immediate else 'await_motion'), 'adoption_cycle_reason')
        armed = live['armed_at']
        require(armed is None or (type(armed) is int and saved['frame']<armed<=view.frame),
                'adoption_rearm_clock')

    def prove(self, view: Any) -> dict[str,Any]:
        provider,side = self.provider,view.scope[-1]
        require(not self.closed and self.error is None and side not in self.controller.history,
                'adoption_already_owned')
        require(len(view.refs)==len(view.tokens)==SLOT_COUNT, 'adoption_single_head_required')
        saved = self.slots.get(side)
        require(saved is not None and not saved['used'] and saved['token']==view.tokens[0],
                'adoption_unwitnessed_head')
        live = provider.link.current(view)
        self.same_cycle(live,saved,view)
        require(saved['queue'] is view.queue and saved['pair'] is view.refs[0]
            and saved['pipe'] is live['pipe'], 'adoption_head_reference_changed')
        scope = provider.journal.scope(live['pipe'],side,view.frame,view.clock)
        require(all(saved['scope'][k]==scope[k] for k in ('source_id','run_id','side','pipe_object_id'))
            and saved['scope']['generation']['reset_epoch']==scope['generation']['reset_epoch']
            and saved['epoch']==view.scope[2] and saved['segment']==live['segment'], 'adoption_scope_changed')
        owner = provider.owner(live['pipe'],side,view.scope[2])
        provider.check_enqueue(provider.enqueues.get(side),scope,view.scope[2],owner)
        require(saved['frame']<=view.frame and saved['clock']<=view.clock and not owner['discarded_tokens'],
                'adoption_clock_or_discard')
        accepted,dnext = provider.link.basis(view)
        require(provider._parts.T.valid_pair(view.refs[0]) and view.quiet, 'adoption_head_not_quiet')
        require(sorted(view.refs[0])!=sorted(accepted), 'adoption_head_successor_color_ambiguous')
        return dict(kind='observed_existing_unsettled_slot/v1',token=saved['token'],
            pair=list(saved['pair']),enqueue_frame=saved['frame'],enqueue_time=saved['clock'],
            adoption_frame=view.frame,adoption_time=view.clock,journal_call_token=saved['journal']['token'],
            segment_id=saved['segment'],candidate=saved['event'],next_pair=accepted,dnext_pair=dnext,
            waiting_rearmed_at=live['armed_at'],same_adapter_cycle_only=True,
            source_scope=saved['scope'],action_since_is_adoption_not_backdated=True,
            placement_permission=False,accounting_permission=False,current_permission=False,
            prior_legacy_debt='UNKNOWN',physical_certified=False,quality_gate_clear=False)

    def baseline(self, original: Any, sm: Any, view: Any, raw: Any, side: str) -> Any:
        if not view.refs and not view.tokens:
            return original(sm,view,raw,side)
        parts,control = self.provider._parts,self.controller
        if not self.provider.baseline_selected(view):
            return None
        require(raw is not None and sm.context.state.value=='stable'
            and parts.T.board_key(sm.context.confirmed_board)==raw, 'history_baseline_not_current')
        proof = self.prove(view)
        baseline = dict(kind='live_history_baseline',frame=view.frame,time_sec=view.clock,
            grid=raw,live_scope=view.scope,prior_legacy_debt='UNKNOWN',existing_slot_observation=proof)
        binding = parts.H.establish(control.inventory,view,raw,baseline)
        parts.H.start(control.inventory,binding,view.tokens[0],view.frame,view.clock)
        control.history[side] = binding
        self.slots[side]['used'] = True
        self.adoptions.append(proof)
        self.provider.journal.history.emit(dict(kind='history_baseline_existing_slot',side=side,
            proof=proof,baseline_grid=raw,owner_state=asdict(binding.owner.state),quality_gate_clear=False))
        return binding

    def close(self) -> None:
        self.closed = True
        self.slots.clear()
