"""実返却PBと別に、唯一world条件下の分布を持つcurrentを生成する。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from typing import Any
import path_support as P
import prefix_witness as W
import prefix_provenance as V


def encoded(value: Any) -> str:
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def anchor(state: Any) -> str:
    return encoded(None if state.current is None else asdict(state.current))


@dataclass(frozen=True)
class ConditionalCurrent:
    scope: Any
    frame: int
    clock: float
    action: int
    sm_grid: Any
    inferred_grid: Any
    hidden: Any
    integer_anchor: str
    evidence_json: str
    provisional: bool = field(default=True,init=False)
    physical_certified: bool = field(default=False,init=False)
    integer_current_permission: bool = field(default=False,init=False)
    accounting_permission: bool = field(default=False,init=False)
    production_permission: bool = field(default=False,init=False)


def compatible(state: Any, binding: Any) -> bool:
    value = getattr(binding,'hidden_current',None)
    return (type(value) is ConditionalCurrent and value.scope==binding.scope
        and state.action>=value.action and value.sm_grid==binding.current
        and value.integer_anchor==anchor(state) and intact(value))


def intact(value: ConditionalCurrent) -> bool:
    proof = json.loads(value.evidence_json)
    fields = dict(scope=value.scope,frame=value.frame,clock=value.clock,action=value.action,
        sm=value.sm_grid,inferred_final=value.inferred_grid,integer_anchor=value.integer_anchor)
    return (all(encoded(proof.get(k))==encoded(v) for k,v in fields.items())
        and encoded(proof['conditional_PB'][0])==encoded(value.hidden)
        and proof['conditional_not_recognition_confidence'] is True
        and proof['original_PB_unchanged'] is True)


def cells(board: Any) -> tuple[Any,...]:
    return tuple(tuple(tuple(sorted(board.cell(r,c).probs.items())) for c in range(P.COLS))
        for r in range(P.ROWS))


def probability(pb_type: Any, original: Any, visible: Any, inferred: Any) -> tuple[Any,Any]:
    P.require(type(original) is pb_type,'conditional_original_PB_type')
    before = cells(original)
    for r in range(P.HIDDEN_ROWS,P.ROWS):
        for c in range(P.COLS):
            P.require(before[r][c]==((visible[r][c],1.0),),'conditional_visible_PB')
    value = deepcopy(original)
    for c in range(P.COLS): value.set_certain(0,c,inferred[0][c])
    P.require(cells(original)==before,'conditional_original_PB_changed')
    return value,before


def fresh(control: Any, call: Any, result: Any) -> bool:
    binding,view = call['binding'],call['view']
    if view.refs or binding.next_token is not None or view.quiet is not True: return False
    live = control.provider.link.current(view)
    if not control._parts.T.valid_pair(view.next_pair): return False
    if not (live['accepted']==view.next_pair==result.next_pair): return False
    if getattr(binding,'hidden_current',None) is not None:
        P.require(compatible(binding.owner.state,binding),'conditional_previous_correspondence')
    return True


def returned_board(control: Any, call: Any, result: Any) -> Any:
    """原返却・内部SM・対応anchorの差を保存し、資格判定は緩めない。"""
    values,binding = call['frame'].f_locals,call['binding']
    key = control._parts.T.board_key
    actual = key(result.confirmed_board)
    internal = key(values['ctx'].confirmed_board)
    if not actual==internal==binding.current:
        control.hidden_current_events.append(dict(kind='conditional_return_mismatch',
            frame=call['view'].frame,returned=actual,internal=internal,anchor=binding.current,
            published=key(values.get('published_confirmed')),
            correction=key(values.get('correction_board')),
            original_PB=cells(result.prob_board) if result.prob_board is not None else None))
    P.require(actual==internal==binding.current,'conditional_SM_return')
    return actual


def observation_matches(control: Any, call: Any, current: Any, raw: Any) -> bool:
    """新しいrawが旧確定盤面と異なるframeには候補を作らない。改変票は別に拒否。"""
    expected = call['binding'].grid
    matched = current[1:]==raw[1:]==expected[1:] and P.compatible(raw,expected)
    if not matched:
        control.hidden_current_events.append(dict(kind='conditional_candidate_hold',
            frame=call['view'].frame,reason='fresh_raw_not_same_confirmed',raw=raw,
            current=current,inferred=expected,current_permission=False))
    return matched


def make(control: Any, call: Any, result: Any, provisional: Any) -> Any:
    binding,view = call['binding'],call['view']
    state,values = binding.owner.state,call['frame'].f_locals
    if not getattr(binding,'hidden_tail_consumed',False): return None
    if ('hidden_transition' not in call and getattr(binding,'hidden_current',None) is None
        and getattr(binding,'hidden_anchor',None) is None): return None
    if result is None or result.state.value!='stable' or values['ctx'].state.value!='stable': return None
    if values['signals'].effect_gate_window_active is not False: return None
    if not fresh(control,call,result): return None
    captured = W.observed(control,binding,values['signals'],view)
    if captured is None: return None
    raw,raw_proof = captured
    current = returned_board(control,call,result)
    P.require(result.side==view.scope[-1] and values['ctx'].frame_idx==view.frame,'conditional_clock')
    if not observation_matches(control,call,current,raw): return None
    P.require(state.counter==control.inventory.S.color_counts(binding.grid)
        and not state.debts and not state.origins,'conditional_inventory')
    prefix = V.validate(control,binding,binding.hidden_prefix_votes)
    inferred,original = probability(provisional.ProbabilisticBoard,result.prob_board,current,binding.grid)
    key = provisional.FrameKey(view.scope[0],view.scope[1],view.frame,view.clock)
    side = provisional.SideInput(key,result.side,'STABLE',result.confirmed_board,inferred,key,result.side,
        'conditional_unique_legal_world_after_original_J;not_original_fallback;not_physical_certification')
    provisional.validate_side(side,result.side)
    _,hidden,visible_sha = provisional.prepare(side)
    proof = dict(kind='conditional_hidden_current/v1',frame=view.frame,clock=view.clock,scope=view.scope,
        original_PB=original,conditional_PB=cells(inferred),raw=raw,raw_capture=raw_proof,
        sm=current,inferred_final=binding.grid,integer_anchor=anchor(state),visible_sha256=visible_sha,
        action=state.action,counter=state.counter,prefix_source=prefix,
        prefix_consumed_frame=binding.hidden_prefix_votes.last[0],
        tail_consumed_frame=binding.hidden_tail_votes.last[0],
        natural_transition=call.get('hidden_transition'),same_origin_evidence_not_independent=True,
        original_PB_unchanged=True,conditional_not_recognition_confidence=True,physical_certified=False)
    value = ConditionalCurrent(view.scope,view.frame,view.clock,state.action,current,binding.grid,
        hidden,anchor(state),encoded(proof))
    return value,side
