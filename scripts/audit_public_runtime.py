"""公開準備: import 閉包と画像読込箇所を実行せず収集する。"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packaging'))
from import_closure import internal_closure

ENTRIES = ('src.phase_j.distribution_app', 'src.phase_j.launcher',
           'scripts.run_live_pipeline_20260928')
IMAGE_READERS = frozenset({'imread', 'open'})


def image_calls(path: Path) -> list[dict]:
    """関数名と行番号付きで画像読込候補を列挙する。"""
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    result: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        target = ast.unparse(node.func)
        if node.func.attr not in IMAGE_READERS or not target.startswith(('cv2.', 'Image.')):
            continue
        scope, parent = [], node
        while parent in parents:
            parent = parents[parent]
            if isinstance(parent, (ast.FunctionDef, ast.ClassDef)):
                scope.insert(0, parent.name)
        result.append(dict(file=path.relative_to(ROOT).as_posix(), line=node.lineno,
                           function='.'.join(scope), call=ast.get_source_segment(source, node)))
    return result


def audit() -> dict:
    """動的 import は別途実測が必要であることを結果に明記する。"""
    closure: set[Path] = set()
    for entry in ENTRIES:
        modules, _ = internal_closure(ROOT, entry)
        closure.update(modules.values())
    readers = []
    for base in ('src', 'scripts'):
        for path in sorted((ROOT / base).rglob('*.py')):
            for call in image_calls(path):
                call['in_static_closure'] = path in closure
                readers.append(call)
    files = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size)
             for p in sorted(closure)]
    return dict(entries=ENTRIES, static_only=True, files=files, image_readers=readers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = audit()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(modules=len(report['files']), readers=len(report['image_readers']),
                          runtime_readers=[r for r in report['image_readers'] if r['in_static_closure']]),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
