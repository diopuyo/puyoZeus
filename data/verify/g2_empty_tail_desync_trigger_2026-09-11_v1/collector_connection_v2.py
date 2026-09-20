"""既存FunctionType複製を維持し、原drive内のcollector取得式だけを置換。"""
from __future__ import annotations
import ast
import inspect
import json
from typing import Any
import collector_connection as C


def saved(state: Any, loop: Any) -> None:
    stream = state['collector_metadata_sink'].rows
    value = dict(actual_original_downstream_loop=True, artificial_pixels_and_initial_history=True,
        metadata_rows=stream.count, actual_append_count=stream.append_count,
        configuration=loop.configuration, quality_gate_clear=False, actual_full_video=False)
    with (state['output']/'CPU_COLLECTOR_CONNECTION.json').open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,default=str)


def connect(stack: Any, state: Any) -> Any:
    loop = C.connect(stack,state)
    stack.callback(saved,state,loop)
    return loop


def drive(original: Any, namespace: Any, extend: Any) -> Any:
    tree = ast.parse(inspect.getsource(original))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and ast.unparse(node.iter)=='I.FRAMES':
            node.iter = ast.parse('I.PREFIX_FRAMES',mode='eval').body
            changed += 1
        if isinstance(node, ast.Assign) and ast.unparse(node)=='collector = m.loop.collector_loop()':
            node.value = ast.parse("__import__('collector_connection_v2').connect(stack,state)",mode='eval').body
            changed += 1
        if isinstance(node, ast.With):
            for index,item in enumerate(list(node.body)):
                if isinstance(item, ast.Assign) and ast.unparse(item).startswith('result = verify('):
                    node.body.insert(index+1,ast.parse('empty_continue(locals(),result)').body[0])
                    changed += 1
    assert changed==3,'complete_collector_original_drive_anchors'
    values = dict(namespace,empty_continue=extend)
    exec(compile(ast.fix_missing_locations(tree),__file__,'exec'),values)
    assert not values['drive'].__code__.co_freevars
    return values['drive']
