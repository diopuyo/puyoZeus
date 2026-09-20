"""旧scopeの正規証拠を保持する部品。reset成功や新基点を認可しない。"""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from typing import Any

BINDING_FIELDS = ('scope', 'grid', 'current', 'next_token', 'next_started',
    'consumed_tokens', 'phase', 'hidden_anchor', 'hidden_current', 'candidate',
    'clear_grid', 'clear_first', 'clear_last', 'clear_count')
HISTORY_FIELDS = ('hidden_history_rows', 'hidden_current_events', 'hidden_lifetime_rows')
OUTPUT_FIELD = 'hidden_current_outputs'
CHECKPOINT_FIELD = 'conditional_next_hands'
RUNTIME_TYPES = frozenset(('TailVotes', 'Votes', 'Held', 'Candidate', 'View', 'NextHand'))
OPAQUE_FIELDS = frozenset(('binding', 'queue', 'pipe', 'sm', 'caller', 'invocation',
    'owner', 'policy', 'provider', 'control'))
EXTENSION_PREFIXES = ('conditional_', 'hidden_', 'private_suffix_', 'firing_')


def encode(value: Any) -> Any:
    """正規証拠だけを型付き保存する。不明な実行オブジェクトは黙って省略しない。"""
    kind = type(value)
    if value is None or kind in (str, int, float, bool):
        return [kind.__name__, value]
    if is_dataclass(value):
        return [kind.__module__+'.'+kind.__qualname__, encode(asdict(value))]
    if kind is dict:
        pairs = [[encode(k), encode(v)] for k,v in value.items()]
        return ['dict', sorted(pairs, key=lambda p:json.dumps(p[0],sort_keys=True))]
    if kind in (tuple, list, set, frozenset):
        values = [encode(v) for v in value]
        if kind in (set, frozenset):
            values.sort(key=lambda v:json.dumps(v,sort_keys=True))
        return [kind.__name__, values]
    raise TypeError('conditional_archive_unsupported:'+kind.__module__+'.'+kind.__qualname__)


def digest(value: Any) -> str:
    raw = json.dumps(encode(value), ensure_ascii=False, allow_nan=False, separators=(',',':'))
    return sha256(raw.encode('utf-8')).hexdigest()


def payload(binding: Any, runtime: Any) -> dict[str, Any]:
    names = set(BINDING_FIELDS) | {n for n in vars(binding) if n.startswith(EXTENSION_PREFIXES)}
    accounting = getattr(binding.policy, 'accounting', binding.policy)
    return dict(owner=asdict(binding.owner.state), policy_proofs=accounting.proofs,
        attributes={n:runtime.project(getattr(binding,n)) for n in sorted(names) if hasattr(binding,n)})


class RuntimeEvidence:
    """既存票の純値だけを射影。binding/native queue/frameは再帰コピーしない。"""
    def __init__(self, control: Any) -> None:
        self.control, self.references, self.source_digests = control, {}, {}
        self.sealed = False

    def reference(self, value: Any) -> dict[str, Any]:
        key = id(value)
        if self.sealed:
            assert key in self.references and self.references[key] is value, 'archive_runtime_reference'
        else:
            self.references[key] = value
        return dict(runtime_reference=key, type=type(value).__module__+'.'+type(value).__qualname__)

    def record(self, value: Any) -> dict[str, Any]:
        result = self.reference(value)
        values = {}
        for name, item in vars(value).items():
            if name in OPAQUE_FIELDS and item is not None:
                values[name] = self.reference(item)
            elif name in ('head', 'pair') and item is not None:
                values[name] = self.reference(item) | dict(value=self.project(item))
            elif name in ('refs', 'queue_refs'):
                values[name] = tuple(self.reference(p) | dict(value=self.project(p)) for p in item)
            else:
                values[name] = self.project(item)
        if type(value).__name__ == 'Held':
            actual = self.control.inventory.digest(value.proof)
            assert actual == value.proof_digest, 'archive_held_source_digest'
            if self.sealed:
                assert self.source_digests[str(id(value))] == actual, 'archive_held_source_changed'
            else:
                self.source_digests[str(id(value))] = actual
        return result | dict(fields=values)

    def project(self, value: Any) -> Any:
        kind = type(value)
        if kind is dict:
            return {k:self.project(v) for k,v in value.items()}
        if kind in (tuple, list, set, frozenset):
            return kind(self.project(v) for v in value)
        if is_dataclass(value) and kind.__name__ in RUNTIME_TYPES:
            return self.record(value)
        # 正規S/ConditionalCurrent/Supportなどの純値だけを元strict encoderで検査。
        encode(value)
        return value


class Archive:
    """値と実参照を分けて保持。原native queue/reset対象の内部を不変と偽らない。"""
    def __init__(self, control: Any, binding: Any) -> None:
        self.control, self.binding, self.owner = control, binding, binding.owner
        self.state = binding.owner.state
        self.attributes = dict(vars(binding))
        self.runtime = RuntimeEvidence(control)
        initial = payload(binding, self.runtime)
        self.saved, self.sha = encode(initial), digest(initial)
        self.history = {name:tuple(getattr(control,name,())) for name in HISTORY_FIELDS}
        self.history_sha = {name:digest(list(rows)) for name,rows in self.history.items()}
        self.outputs = tuple(getattr(control,OUTPUT_FIELD,()))
        self.output_sha = digest([pair[0] for pair in self.outputs])
        self.checkpoints = dict(getattr(control,CHECKPOINT_FIELD,{}))
        self.checkpoint_sha = digest(self.runtime.project(self.checkpoints))
        self.runtime.sealed = True

    def verify(self) -> None:
        assert self.binding.owner is self.owner and self.owner.state is self.state, 'archive_owner_reference'
        actual = vars(self.binding)
        assert set(actual) == set(self.attributes), 'archive_binding_fields'
        assert all(actual[name] is value for name,value in self.attributes.items()), 'archive_binding_reference'
        assert digest(payload(self.binding, self.runtime)) == self.sha, 'archive_canonical_evidence'
        for name, saved in self.history.items():
            current = getattr(self.control,name,())
            assert len(current) >= len(saved) and all(current[i] is row for i,row in enumerate(saved)), 'archive_history_prefix'
            assert digest(list(current[:len(saved)])) == self.history_sha[name], 'archive_history_changed'
        outputs = getattr(self.control,OUTPUT_FIELD,())
        assert len(outputs) >= len(self.outputs) and all(outputs[i] is row for i,row in enumerate(self.outputs)), 'archive_output_prefix'
        assert digest([pair[0] for pair in outputs[:len(self.outputs)]]) == self.output_sha, 'archive_current_changed'
        checkpoints = getattr(self.control,CHECKPOINT_FIELD,{})
        assert all(key in checkpoints and checkpoints[key] is val for key,val in self.checkpoints.items()), 'archive_checkpoint_reference'
        assert digest(self.runtime.project({k:checkpoints[k] for k in self.checkpoints})) == self.checkpoint_sha, 'archive_checkpoint_changed'

    def receipt(self) -> dict[str, Any]:
        self.verify()
        return dict(kind='conditional_reset_archive/v1', old_scope=self.binding.scope,
            canonical_sha256=self.sha, canonical=self.saved, old_binding_identity=id(self.binding),
            old_owner_identity=id(self.owner), retained_history_counts={n:len(v) for n,v in self.history.items()},
            retained_output_count=len(self.outputs), retained_checkpoint_count=len(self.checkpoints),
            runtime_reference_count=len(self.runtime.references), runtime_source_digests=dict(self.runtime.source_digests),
            runtime_projection_is_not_caller_authorization=True, side_input_values_reinterpreted=False,
            native_queue_and_caller_frame_deep_preservation=False, reset_success_verified=False,
            new_scope_permission=False, physical_certified=False, quality_gate_clear=False)
