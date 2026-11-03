"""診断範囲・再生一致・画像除外を検査して圧縮原票を保存する。"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
from pathlib import Path

from scripts._diag_mia7_inspect import BASE, OUT, ROOT, START, END, guard, save

EXPECTED_UPDATES, FUNCTION_LIMIT = 781, 50
COMPRESS_NAMES = ('trace.jsonl', 'raw_window.json')
IMAGE_SUFFIXES = {'.jpg', '.png', '.mp4', '.jpeg', '.webp'}


def event_values(path: Path) -> list[dict]:
    """圧縮の有無を除いて同じイベント値を読む。"""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [v for line in stream for v in json.loads(line)['values'] if v['t_sec'] <= END]


def scripts_check() -> list[str]:
    """許可された診断スクリプトの構文と関数長を検査する。"""
    result = []
    for path in sorted((ROOT / 'scripts').glob('_diag_mia7_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno-node.lineno+1 <= FUNCTION_LIMIT, (path.name, node.name)
                assert node.returns is not None, (path.name, node.name)
        result.append(str(path.relative_to(ROOT)))
    return result


def compress() -> int:
    """この診断で作った二つの原票だけを可逆圧縮する。"""
    count = 0
    for name in COMPRESS_NAMES:
        path = OUT / name
        if path.exists():
            raw = path.read_bytes()
            packed = gzip.compress(raw, mtime=0)
            assert gzip.decompress(packed) == raw
            path.with_suffix(path.suffix + '.gz').write_bytes(packed)
            path.unlink()
        if name == 'raw_window.json':
            count = len(json.loads(gzip.decompress(path.with_suffix('.json.gz').read_bytes())))
    return count


def main() -> None:
    """出力範囲と実行資源を受領票に記録する。"""
    available = guard()
    sources = scripts_check()
    old = event_values(BASE / 'v2e/evidence/full/mia8KCjr52g/events.jsonl.gz')
    new = event_values(OUT / 'replay/events.jsonl')
    assert old == new
    count = compress()
    assert count == EXPECTED_UPDATES
    assert not [p for p in OUT.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES]
    boards = json.loads((OUT / 'stable_boards.json').read_text())
    save('RECEIPT.json', dict(scope='診断のみ・合否外', scripts=sources,
        raw_updates=count, window=[START, END], event_values=len(old), event_values_exact_equal=True,
        stable_board_changes=len(boards), nice=10, available_bytes=available,
        syntax_and_function_size='確認済み', no_game_images_in_artifacts=True))
    files = [dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('FILES.json', 'package.log')]
    save('FILES.json', files)
    print(json.dumps(dict(files=len(files), updates=count, boards=len(boards), event_values=len(old))))


if __name__ == '__main__':
    main()
