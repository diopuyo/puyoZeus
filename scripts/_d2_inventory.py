"""D2の原票・全CLI引数・資産ハッシュ・コミット由来を保存する。"""
from __future__ import annotations

import gzip
import argparse
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any
from itertools import zip_longest

from src.production_config import advantage_overlay_flags, recognition_load_default_kwargs

OUT = Path('logs/d2')
SOURCES = ('q_7gc4TgFig', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi', 'review')


def read(path: Path) -> Any:
    """UTF-8の原票を読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path: Path) -> str:
    """入力の内容を固定する。"""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name: str, value: Any) -> None:
    """診断結果を指定先に保存する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def flags(args: list[str]) -> dict[str, Any]:
    """値付き・値なしを区別し、全CLI項目を比較する。"""
    result: dict[str, Any] = {}
    for index, token in enumerate(args):
        if not token.startswith('--'):
            continue
        result[token] = (args[index+1] if index+1 < len(args)
                         and not args[index+1].startswith('--') else True)
    return result


def origins(source: str) -> Path:
    """E16のsource原票まで逆に辿った実収集ディレクトリを返す。"""
    if source in SOURCES[:3]:
        return Path('logs/e8/renders')/source/'on'
    return Path('logs/review_zenchi_g41_43_e14' if source == 'review'
                else 'logs/review_zenchi_part3/on_e10c')


def header(path: Path) -> dict:
    """巨大原票のヘッダーだけを取得する。"""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.loads(next(stream))


def source_inventory(source: str) -> dict:
    """出力先を含めて全引数と原票由来を省略せず保存する。"""
    base = origins(source)
    old_status = read(base/'status.json')
    new_status = read(Path(f'logs/e34b/records/{source}.jsonl.json'))
    old, new = flags(old_status['command']), flags(new_status['command'])
    entries = {key: dict(old=old.get(key), new=new.get(key), equal=old.get(key) == new.get(key))
               for key in sorted(old.keys() | new.keys())}
    paths = [base/'inputs.jsonl.gz']
    paths += [Path(f'logs/{stage}/records/{source}.jsonl.gz')
              for stage in ('e16', 'e26', 'e31', 'e34b')]
    return dict(old_status=old_status, new_status=new_status, all_cli=entries,
                records={str(p): dict(sha256=digest(p), header=header(p)) for p in paths},
                provenance=[read(Path(f'logs/e16/records/{source}.json')),
                            read(Path(f'logs/e26/records/{source}.json')),
                            read(Path(f'logs/e31/records/{source}.jsonl.json'))])


def assets() -> dict:
    """E3時点の資産ハッシュと現物を照合する。"""
    saved = read(Path('logs/e3/assets.json'))
    return {path: dict(saved=sha, current=digest(Path(path)), equal=sha == digest(Path(path)))
            for path, sha in saved.items()}


def updates(path: Path) -> Any:
    """盤面と状態だけを逐次取り出し、大きな原票を保持しない。"""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] != 'update':
                continue
            args = row['args']['tuple']
            sides = args[0]['namespace']
            yield args[3], [sides[k]['namespace'] for k in ('p1', 'p2')]


def board_comparison(left: Path, right: Path) -> dict:
    """原収集からの盤面保存と旧新不一致数を全行で確認する。"""
    count, boards, states, first, start = 0, 0, 0, None, None
    for old, new in zip_longest(updates(left), updates(right)):
        assert old is not None and new is not None and old[0] == new[0]
        if start is None:
            start = old[0]
        count += 1
        changed = any(a['confirmed_board'] != b['confirmed_board'] for a, b in zip(old[1], new[1]))
        boards += changed
        states += any(a['state'] != b['state'] for a, b in zip(old[1], new[1]))
        if changed and first is None:
            first = old[0]
    return dict(rows=count, boards=boards, states=states, first=first, start=start,
                elapsed_first=None if first is None else first-start)


def main() -> None:
    """全5記録と本番単一情報源を監査する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    revision = parser.parse_args().revision
    save('inventory.json', dict(revision=revision,
        production_cli=flags(shlex.split(advantage_overlay_flags())),
        production_recognition=recognition_load_default_kwargs(), assets=assets(),
        sources={source: source_inventory(source) for source in SOURCES}))
    comparisons = {}
    for source in SOURCES:
        old = Path(f'logs/e31/records/{source}.jsonl.gz')
        comparisons[source] = dict(
            original_to_e31=board_comparison(origins(source)/'inputs.jsonl.gz', old),
            e31_to_e34b=board_comparison(old, Path(f'logs/e34b/records/{source}.jsonl.gz')))
    save('original_board_audit.json', comparisons)


if __name__ == '__main__':
    main()
