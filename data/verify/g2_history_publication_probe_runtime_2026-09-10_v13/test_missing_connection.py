"""実assembly入口の欠測method設置と復元をモデルなしで確認する。"""
from __future__ import annotations
from contextlib import ExitStack
import ast
import inspect
from types import SimpleNamespace
from typing import Any
import common as K
import assembly_publication_probe as A
import missing_connection as GAP


def test_connection_is_same_assembly_entry() -> None:
    tree = ast.parse(inspect.getsource(GAP.configure))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert any(isinstance(n.func, ast.Attribute) and n.func.attr == 'enter_context' for n in calls)
    assert K.sha(K.MISSING/'connection.py') == K.FIXED[K.MISSING/'connection.py']


def test_original_sources_unchanged() -> None:
    assert K.sha(K.VERIFY/'g2_historical_completion_runtime_2026-09-09_v1/history_controller.py') == (
        '15880b3310207ca3119c29c41272fdc20a63a729812541c3352c1c47f9a839ef')
