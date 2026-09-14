"""既存current/publication/consumerを合成。原assembly名を占有しない。"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterator
import common as K
import adoption_connection as E
import ast_connection as AST
import initial_hand_connection as FIRST
import missing_connection as GAP


@contextmanager
def session(stack: Any, a: Any, c: Any, modules: Any) -> Iterator[Any]:
    K.guards()
    AST.configure(stack, a, c)
    current = K.load('_probe_current_assembly', K.CURRENT / 'assembly_current.py', stack)
    publication = K.load('_probe_publication_assembly', K.PUBLICATION / 'assembly_publication.py', stack)
    consumer = K.load('_probe_postcommit_consumer', K.CONSUMER / 'adapter.py', stack)
    E.configure(stack,c,current)
    GAP.configure(stack, c, current)
    loaded = current.load(stack)
    FIRST.configure(stack, c, loaded['controller_v3'])
    original = publication.install
    def install(inner: Any, collector: Any, state: Any) -> Any:
        receiver = original(inner, collector, state)
        consumer.install(inner, state)
        return receiver
    c.G.patch(stack, publication, 'install', install)
    with publication.session(a, current, c, modules, loaded) as (runtime, addon, factory):
        extra = runtime.extra
        c.G.patch(stack, runtime, 'extra', lambda args: extra(args) | current.guards() | publication.guards() | K.guards())
        yield runtime, addon, factory
