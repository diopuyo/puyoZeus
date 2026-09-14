"""既存creator CPU fixtureの実戻り値に、最終選択型/consumerの検査だけ挿入する。"""
import ast
import inspect
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent


def check(session: Any, parts: Any, receipts: list) -> None:
    first = session.state['probabilistic_tracking_mode']
    assert type(first) is parts.mode.Mode
    assert any(cls.__module__ == 'first_terminal_candidate' for cls in type(first).__mro__)
    assert parts.mode.E.Capture.__module__ == 'first_terminal_candidate'
    assert session._prefix_mode_type.__module__ == 'second_inactive_candidate'
    prefix = sys.modules['_g2_prefix_final']
    assert Path(prefix.verify.__code__.co_filename).resolve() == ROOT/'terminal_selection.py'
    assert Path(parts.arrival_saved.verify.__code__.co_filename).resolve() == (
        ROOT.parent/'g2_prefix_lane_integration_2026-09-13_v1/prefix_selection.py')
    receipts.append(dict(first_type_selected=True, first_capture_selected=True,
        second_type_selected=True, actual_prefix_saved_selected=True, original_prefix_wrapper_preserved=True))


def execute(probe: Any, adapter: Any, output: Path) -> dict:
    tree = ast.parse(inspect.getsource(probe.execute))
    calls, receipts = 0, []
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if not isinstance(body, list):
            continue
        for index, child in enumerate(tuple(body)):
            if isinstance(child, ast.Assign) and ast.unparse(child).startswith('session = create('):
                body.insert(index+1, ast.parse('terminal_check(session, parts, terminal_receipts)').body[0])
                calls += 1
    assert calls == 1
    namespace = dict(vars(probe), terminal_check=check, terminal_receipts=receipts)
    exec(compile(ast.fix_missing_locations(tree), __file__, 'exec'), namespace)
    result = namespace['execute'](adapter, output)
    assert len(receipts) == 1
    return result | receipts[0]
