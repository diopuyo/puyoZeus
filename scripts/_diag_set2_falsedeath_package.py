"""診断原票を圧縮し、再現用のハッシュと納品対象一覧を保存する。"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import sys
from pathlib import Path

from scripts._diag_set2_falsedeath_batch import ROOT, OUT, records

SOURCE_FILES = ('production_config.py', 'exchange_event_tracker.py', 'exchange_event_landing.py',
                'exchange_death_inputs.py', 'exchange_single_death_proof.py',
                'exchange_post_counter_bound.py', 'exchange_pending_ledger.py',
                'recognition_pipeline.py', 'board_state_machine.py', 'placement_inferrer.py')


def digest(path: Path) -> str:
    """入力を変更せずSHA256を計算する。"""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_scripts() -> list[dict]:
    """構文と関数長のプロジェクト規約を確認する。"""
    results = []
    for path in sorted((ROOT / 'scripts').glob('_diag_set2_falsedeath_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        lengths = {n.name: n.end_lineno - n.lineno + 1 for n in functions}
        assert max(lengths.values(), default=0) <= 50, (path, lengths)
        assert all(n.returns is not None for n in functions), path
        results.append(dict(file=path.name, sha256=digest(path), function_lines=lengths))
    return results


def replay_files(mode: str) -> tuple[list[Path], list[dict]]:
    """完走した再生だけを納品し、イベントは可逆圧縮する。"""
    selected, hashes = [], []
    for directory in sorted((OUT / mode).iterdir()):
        if not (directory / 'status.json').exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix == '.gz':
                continue
            destination = path
            if path.name == 'events.jsonl':
                destination = path.with_suffix('.jsonl.gz')
                destination.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
                assert gzip.decompress(destination.read_bytes()) == path.read_bytes()
            selected.append(destination)
            hashes.append(dict(file=str(path.relative_to(OUT)), sha256=digest(path), bytes=path.stat().st_size))
    return selected, hashes


def main() -> None:
    """完全な監査が終わった成果物のみを納品一覧へ載せる。"""
    checks = check_scripts()
    for mode in ('attributed', 'corroborated'):
        assert all((OUT / mode / n / 'status.json').exists() for n in records()), mode
    selected = [p for p in OUT.iterdir() if p.suffix in ('.json', '.jpg', '.md')]
    hashes = []
    for mode in ('baseline', 'attributed', 'combined', 'corroborated'):
        files, rows = replay_files(mode)
        selected.extend(files)
        hashes.extend(rows)
    selected.extend((OUT / 'recognition').glob('*_identity.json'))
    selected.extend((OUT / 'recognition').glob('*_writes.jsonl'))
    manifest = dict(python=sys.version, executable=sys.executable, scripts=checks,
                    source_sha256={n: digest(ROOT / 'src' / n) for n in SOURCE_FILES},
                    records={n: dict(path=str(p), sha256=digest(p)) for n, p in records().items()},
                    replay_outputs=hashes)
    path = OUT / 'MANIFEST.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    selected.append(path)
    selected.extend((ROOT / 'scripts').glob('_diag_set2_falsedeath_*.py'))
    names = sorted({str(p.relative_to(ROOT)).replace('\\', '/') for p in selected})
    (OUT / 'DELIVERY_FILES.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
    print(json.dumps(dict(files=len(names), script_checks=len(checks))))


if __name__ == '__main__':
    if '--check-only' in sys.argv:
        print(json.dumps(check_scripts(), ensure_ascii=False))
    else:
        main()
