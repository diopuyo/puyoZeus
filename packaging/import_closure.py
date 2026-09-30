"""エントリ module からの内部 import 閉包 (関数内 import も含む静的解析)。

配布物に入れる .py を「実際に import され得るもの」だけに絞るために使う。
静的解析なので実行時に到達しない分岐も含む (過大側)。実行時に開かれる資産ファイルは
別途 sys.addaudithook で実測する (packaging/audit_opens.py)。
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

INTERNAL_TOPS = ('src', 'scripts')
STDLIB = frozenset(sys.stdlib_module_names)


def resolve_module(root: Path, module: str) -> Path | None:
    """`a.b.c` を root 配下の a/b/c.py または a/b/c/__init__.py に解決する。"""
    base = root / module.replace('.', '/')
    for candidate in (base.with_suffix('.py'), base / '__init__.py'):
        if candidate.exists():
            return candidate
    return None


def imported_names(tree: ast.AST, package: str) -> list[str]:
    """AST 内のすべての import 文 (関数内含む) が指し得る絶対 module 名を返す。"""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split('.')
                parts = parts[:len(parts) - (node.level - 1)]
                base = '.'.join(parts + ([node.module] if node.module else []))
            else:
                base = node.module or ''
            names += [base] + [f'{base}.{alias.name}' for alias in node.names]
    return names


def internal_closure(root: Path, entry: str) -> tuple[dict[str, Path], dict[str, set[str]]]:
    """(内部 module→ファイル, 外部パッケージ→それを import する内部 module 集合) を返す。"""
    seen: dict[str, Path] = {}
    external: dict[str, set[str]] = {}
    todo = [entry]
    while todo:
        module = todo.pop()
        path = resolve_module(root, module)
        if module in seen or path is None:
            continue
        seen[module] = path
        # 子 module を import すると親パッケージの __init__ も実行される (例: src.phase_j の
        # from .scheduler import ...)。親も閉包に含めないと同梱漏れになる。
        parent = module.rpartition('.')[0]
        if parent:
            todo.append(parent)
        package = module if path.name == '__init__.py' else parent
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for name in imported_names(tree, package):
            top = name.split('.')[0]
            if top in INTERNAL_TOPS:
                todo.append(name)
            elif top and top not in STDLIB:
                external.setdefault(top, set()).add(module)
    return seen, external


def package_init_files(root: Path, modules: dict[str, Path]) -> set[Path]:
    """閉包 module の親パッケージ __init__.py (namespace 扱いでも存在すれば同梱)。"""
    inits: set[Path] = set()
    for module in modules:
        parts = module.split('.')
        for depth in range(1, len(parts)):
            init = root.joinpath(*parts[:depth]) / '__init__.py'
            if init.exists():
                inits.add(init)
    return inits
