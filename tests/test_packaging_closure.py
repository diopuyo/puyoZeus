"""packaging/import_closure.py (同梱 .py の import 閉包) の試験。

実例: 親パッケージ __init__ 内の `from .scheduler import ...` を辿らず同梱漏れし、配布版が
ModuleNotFoundError で起動しなかった。回帰を防ぐ。
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packaging'))

from import_closure import internal_closure, package_init_files  # noqa: E402


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def test_parent_package_init_imports_are_followed(tmp_path: Path) -> None:
    write(tmp_path, 'src/__init__.py', '')
    write(tmp_path, 'src/pkg/__init__.py', 'from .sibling import x\n')
    write(tmp_path, 'src/pkg/sibling.py', 'x = 1\n')
    write(tmp_path, 'src/pkg/entry.py', 'import numpy\n')
    modules, external = internal_closure(tmp_path, 'src.pkg.entry')
    assert {'src.pkg.entry', 'src.pkg', 'src.pkg.sibling', 'src'} <= set(modules)
    assert external == {'numpy': {'src.pkg.entry'}}


def test_function_level_and_relative_imports(tmp_path: Path) -> None:
    write(tmp_path, 'src/__init__.py', '')
    write(tmp_path, 'scripts/main.py', 'def f():\n    from src.lazy import y\n')
    write(tmp_path, 'src/lazy.py', 'from . import other\nimport cv2\n')
    write(tmp_path, 'src/other.py', '')
    modules, external = internal_closure(tmp_path, 'scripts.main')
    assert {'scripts.main', 'src.lazy', 'src.other'} <= set(modules) and 'cv2' in external


def test_stdlib_and_unresolvable_are_excluded(tmp_path: Path) -> None:
    write(tmp_path, 'src/__init__.py', '')
    write(tmp_path, 'src/a.py', 'import json, os\nfrom src.missing import z\n')
    modules, external = internal_closure(tmp_path, 'src.a')
    assert 'src.missing' not in modules and external == {}
    assert package_init_files(tmp_path, modules) == {tmp_path / 'src/__init__.py'}


def test_live_b3_runtime_modules_are_in_bundle_closure() -> None:
    """遅延importとjoblibの型定義も、最終組立前に同梱対象であることを確認する。"""
    root = Path(__file__).resolve().parents[1]
    modules, _ = internal_closure(root, 'scripts.run_live_pipeline_20260928')
    assert {'src.phase_j.live_b3', 'src.b3_realtime', 'src.b3_rows', 'src.b3_features',
            'src.b3_search', 'src.b3_train_model', 'src.b3_predict', 'src.b3_observation',
            'src.b3_live_queues', 'scripts.b2_recollect_cache', 'src.record_redesign_observer'} <= set(modules)
