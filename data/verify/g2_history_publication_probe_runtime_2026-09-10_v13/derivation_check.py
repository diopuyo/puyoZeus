"""旧v11から許可した接続差分だけを再生成して全AST比較する。"""
from __future__ import annotations
import ast
from typing import Any
import common as K

PRIOR = K.VERIFY/'g2_history_publication_probe_runtime_2026-09-10_v11'
LIVE_CHANGES = (
    ('import session as S', 'import session as S\nimport repeated_connection as REPEAT'),
    ('def constructor_guard(stack: Any, collector: Any, state: Any, base: Any) -> None:',
     'def constructor_guard(stack: Any, collector: Any, state: Any, base: Any, env: Any = None) -> None:'),
    ("    base.patch(stack, cls, 'load_default', classmethod(load))",
     "    base.patch(stack, cls, 'load_default', classmethod(load))\n    K.require(env is not None, 'repeated_factory_environment_required')\n    REPEAT.install(stack, collector, state, env)"),
    ('constructor_guard(stack, collector, state, base)', 'constructor_guard(stack, collector, state, base, env)'),
    ('        summary = Q.finish(env, state, rec, sink, handles, binding)',
     "        repeated = REPEAT.verify(state)\n        summary = Q.finish(env, state, rec, sink, handles, binding)\n        summary['repeated_firing'] = repeated"),
    ('    finally:\n        stream.close()',
     "    finally:\n        stream.close()\n        if 'repeated_firing_constructor' in state:\n            K.write(output/'REPEATED_FIRING.json', base.json_value(state['repeated_firing_constructor']))"),
)
SESSION_CHANGES = (
    ('import assembly_publication_probe as A',
     'import assembly_publication_probe as A\nimport repeated_connection as REPEAT'),
    ("        with installed() as env:\n            yield env",
     "        with ExitStack() as dependencies:\n            repeated = REPEAT.load(dependencies)\n            with installed() as env:\n                env['repeated_dependencies'] = repeated\n                yield env"),
)

LIVE_CHANGES += (("    finally:\n        stream.close()",
    "    finally:\n        stream.close()\n        K.write(output/'SCOPE_STOP_STATUS.json', base.json_value(REPEAT.scope_status(state)))"),)
LIVE_CHANGES += (("            original_refs_restored=refs(collector) == original, state_keys=sorted(state),",
    "            original_refs_restored=refs(collector) == original, state_keys=sorted(state),\n"
    "            pipeline=base.json_value(rec.pipeline_receipt), model_loads=base.json_value(rec.model_loads),"),)


def expected(file: str) -> ast.Module:
    source = (PRIOR/file).read_text()
    changes = LIVE_CHANGES if file == 'live_cli.py' else SESSION_CHANGES
    if file == 'closure.py':
        changes = (('import goals as G', 'import goals as G\nimport finalizer_connection as FINAL'),
            ("goal = G.evaluate(rows, control.legal, state['output'])",
             "goal = FINAL.evaluate(G, rows, control.legal, state['output'], state)"))
    for old, new in changes:
        assert source.count(old) == 1, 'derivation_anchor:'+old
        source = source.replace(old, new)
    return ast.parse(source)


def verify_function(file: str, name: str) -> None:
    actual = ast.parse((K.ROOT/file).read_text())
    select = lambda tree: next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    assert ast.dump(select(actual)) == ast.dump(select(expected(file)))


def verify_common() -> None:
    source = (PRIOR/'common.py').read_text()
    block = "REPEATED_MANIFEST = ROOT/'REPEATED_DEPENDENCIES.json'\n"
    block += "REPEATED_SHA = '"+K.REPEATED_SHA+"'\n"
    block += "if hashlib.sha256(REPEATED_MANIFEST.read_bytes()).hexdigest() != REPEATED_SHA:\n"
    block += "    raise RuntimeError('repeated_dependency_manifest_changed')\n"
    block += "FIXED[REPEATED_MANIFEST] = REPEATED_SHA\n"
    block += "FIXED.update({Path(path): digest for path, digest in json.loads(REPEATED_MANIFEST.read_bytes())['guards'].items()})\n"
    block += "OWN += ('repeated_connection.py', 'REPEATED_DEPENDENCIES.json', 'derivation_check.py', 'test_scope_connection.py',\n"
    block += "    'finalizer_connection.py', 'test_finalizer_connection.py', 'saved_finalizer_cpu.py')\n\n\n"
    assert source.count('def require(value: bool, reason: str) -> None:') == 1
    expected_source = source.replace('def require(value: bool, reason: str) -> None:', block+'def require(value: bool, reason: str) -> None:')
    assert ast.dump(ast.parse((K.ROOT/'common.py').read_text())) == ast.dump(ast.parse(expected_source))
