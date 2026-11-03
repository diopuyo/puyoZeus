"""研究版の固定コミットから、未収録のama依存だけを取り込む。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [(Path(_public_dev_text('<work>/wt_ama_plain')), '0004dddd'),
           (Path(_public_dev_text('<work>/wt_ama_eval')), '4d7fd72f')]
SEEDS = ['ama_plain', 'ama_plain_reference', 'ama_plain_probability', 'ama_plain_v71_display',
         'ama_evaluation_v7_projection', 'ama_evaluation_v7_tape', 'ama_tsumo_seed_v5']


def read(name: str) -> tuple[bytes, str]:
    for repo, commit in SOURCES:
        result = subprocess.run(['git', '-C', str(repo), 'show', f'{commit}:{name}'], capture_output=True)
        if result.returncode == 0:
            return result.stdout, f'{repo}@{commit}'
    raise FileNotFoundError(name)


def dependencies(source: str) -> set[str]:
    result = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith('src.'):
                result.add(node.module)
            elif node.module == 'src':
                result.update('src.'+alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names if alias.name.startswith('src.'))
    return result


def main() -> None:
    pending, visited, receipt = ['src.'+name for name in SEEDS], set(), {}
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        name = module.replace('.', '/')+'.py'
        path = ROOT/name
        if path.exists():
            continue
        data, origin = read(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        receipt[name] = dict(origin=origin, sha256=hashlib.sha256(data).hexdigest())
        pending.extend(dependencies(data.decode('utf-8')))
    out = ROOT/'docs/verification/live_obs_20261020'
    out.mkdir(parents=True, exist_ok=True)
    (out/'ama_imports.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
