"""過去pendingを許可せず、同時刻の独立fresh現在だけを既存consumerへ渡す。"""
from __future__ import annotations
import ast
import contextlib
import hashlib
import json
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT.parent / 'g2_current_consumer_release_2026-09-09_v1/release.py'
BINDING = ROOT.parent / 'g2_provisional_context_capture_2026-09-09_v1/binding.py'
FIXED = {RELEASE: '083693a073b5765b52ebb779a8b5522bcd39ea0ffe4aaac6d92c827b2903301e',
    BINDING: '401c1e1a5c5f4965c20750a9c47ecd210deab083f958f3ef4b29bf5c3db1aa2d'}
OWN = ('adapter.py', 'test_adapter.py', 'run_cpu.py', 'PLAN.md')
PENDING = 'pending_candidate_not_verified'
KEY, STATUS = 'pending_fresh_current', 'PENDING_FRESH_CURRENT_STATUS.json'
REQUIRED = frozenset((STATUS,))


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def guards() -> dict[str, str]:
    require(all(sha(p) == h for p, h in FIXED.items()), 'pending_fixed_source_changed')
    return {str(p): h for p, h in FIXED.items()} | {str(ROOT / n): sha(ROOT / n) for n in OWN}


def code(path: Path, name: str) -> CodeType:
    return next(v for v in compile(path.read_bytes(), str(path), 'exec').co_consts
        if isinstance(v, CodeType) and v.co_name == name)


def identity(module: Any, path: Path, names: tuple[str, ...]) -> None:
    require(type(module) is ModuleType and Path(module.__file__).resolve() == path, 'pending_module_identity')
    for name in names:
        fn = getattr(module, name)
        require(type(fn) is FunctionType and fn.__code__ == code(path, name)
            and fn.__globals__ is vars(module), 'pending_function_identity:' + name)


def candidate(binding: Any) -> Any:
    """原candidateの全検査を維持し、既知HOLD理由の分類だけをprivate化。"""
    identity(binding, BINDING, ('_candidate', '_published'))
    node = next(n for n in ast.parse(BINDING.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == '_published')
    expected = ast.dump(ast.parse('after.get("board_none_reason") == HELD_REASON', mode='eval').body,
        include_attributes=False)
    matches = [n for n in ast.walk(node) if isinstance(n, ast.Compare)
        and ast.dump(n, include_attributes=False) == expected]
    require(len(matches) == 1, 'pending_single_reason_site')
    target = matches[0]
    target.ops = [ast.In()]
    target.comparators = [ast.Tuple(elts=[ast.Name(id='HELD_REASON', ctx=ast.Load()),
        ast.Constant(value=PENDING)], ctx=ast.Load())]
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    changed = next(v for v in compile(tree, str(BINDING), 'exec').co_consts if isinstance(v, CodeType))
    private = dict(vars(binding))
    private['_published'] = FunctionType(changed, private, '_published')
    return FunctionType(binding._candidate.__code__, private, '_candidate')


def ticket(release: Any, checked_candidate: Any, row: Any, registration: Any, side: str) -> Any:
    b = release.B
    b.require(side in b.SIDES, 'release_side')
    b._identity(row, registration)
    b._update(row)
    b._generation(row)
    b._unknown_ledger(row)
    saved = row['sides'][side]
    b.require(saved['final']['confirmed'] is None and saved['final']['board_none_reason']
        in (b.HELD_REASON, PENDING), 'release_not_discard_hold', hold=True)
    checked_candidate(row, side)
    before = saved['before_hold']
    release.pointmass_grid(before['probability'], before['confirmed'])
    return release.ReleaseTicket(side, row['frame_idx'], b.digest(row), b.encoded(row), b.encoded(registration))


def install(stack: contextlib.ExitStack, release: Any, state: dict[str, Any], *, enabled: bool = False) -> None:
    if not enabled:
        return
    require(type(enabled) is bool and KEY not in state, 'pending_install_reentry')
    before = guards()
    identity(release, RELEASE, ('ticket', 'restore_side', 'release_result', 'pointmass_grid'))
    checked, original = candidate(release.B), release.ticket
    sink = {'guards': before, 'rows': [], 'errors': [], 'closed': False,
        'quality_gate_clear': False, 'accounting_permission': False}
    def wrapped(row: Any, registration: Any, side: str) -> Any:
        pending = row['sides'][side]['final']['board_none_reason'] == PENDING
        try:
            result = ticket(release, checked, row, registration, side)
        except release.B.ContextHold as error:
            if pending:
                sink['rows'].append({'frame': row['frame_idx'], 'side': side, 'accepted': False, 'reason': str(error)})
            raise
        except BaseException as error:
            sink['errors'].append(repr(error))
            raise
        if pending:
            sink['rows'].append({'frame': row['frame_idx'], 'side': side, 'accepted': True,
                'context_digest': result.context_digest, 'grid_sha256': row['sides'][side]['before_hold']['confirmed']['sha256']})
        return result
    def close() -> None:
        sink['closed'] = True
        with (state['output'] / STATUS).open('x') as stream:
            json.dump(sink, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    stack.callback(close)
    stack.callback(setattr, release, 'ticket', original)
    release.ticket = wrapped
    state[KEY] = sink


def finish(state: dict[str, Any]) -> None:
    require(state[KEY]['closed'] and not state[KEY]['errors'] and state[KEY]['guards'] == guards(), 'pending_unclosed_or_changed')
    verify(state['output'])


def verify(output: Path) -> None:
    value = json.loads((output / STATUS).read_text())
    require(value['closed'] is True and value['errors'] == [] and value['guards'] == guards(), 'pending_saved_source_or_closure')
    require(value['quality_gate_clear'] is False and value['accounting_permission'] is False, 'pending_saved_permission')
    require(type(value['rows']) is list, 'pending_saved_rows')
    for row in value['rows']:
        require(type(row['frame']) is int and row['frame'] >= 0 and row['side'] in ('1P', '2P')
            and type(row['accepted']) is bool, 'pending_saved_scope')
        require(set(row) == ({'frame', 'side', 'accepted', 'context_digest', 'grid_sha256'}
            if row['accepted'] else {'frame', 'side', 'accepted', 'reason'}), 'pending_saved_row_schema')
