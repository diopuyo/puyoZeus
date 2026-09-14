"""原4更新collector対照で実namespaceへの接続/復元を確認。resetは人工設定もしない。"""
from __future__ import annotations
import ast
from pathlib import Path
import sys
from types import SimpleNamespace as N
import torch
from inflight_loader import CONNECTION as C

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'g2_empty_tail_reset_live_adapter_2026-09-11_v1/attach_cpu.py'


def connect(stack: object,factory: object,pipe: object,state: dict) -> None:
    C.install(stack,N(pending=None,pipe=pipe,journal=factory.provider.journal),state)


if __name__=='__main__':
    tree=ast.parse(SOURCE.read_bytes())
    imports,sites=0,0
    for node in ast.walk(tree):
        if isinstance(node,ast.Import) and len(node.names)==1 and node.names[0].name=='adapter_v2':
            node.names[0].name='runtime_adapter'
            imports+=1
        if isinstance(node,ast.FunctionDef) and node.name=='drive':
            node.body[1:1]=ast.parse("assert 'conditional_runtime_factory' not in state\nstate['conditional_runtime_factory']=factory").body
        if isinstance(node,ast.With):
            for index,item in enumerate(node.body):
                if ast.unparse(item).startswith('hook.attach('):
                    node.body.insert(index+1,ast.parse('CONNECT(stack,factory,pipe,state)').body[0])
                    sites+=1
                    break
    assert (imports,sites)==(1,1) and not torch.cuda.is_initialized()
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE),'exec'),
         dict(__name__='__main__',__file__=str(SOURCE),CONNECT=connect))
