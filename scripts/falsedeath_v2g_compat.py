"""E35‴の既存短区間を最終コードで再生し、表示・イベントの完全一致を確認する。"""
from __future__ import annotations

import argparse
import json
import sys

from scripts.falsedeath_replay_20261001 import OUT, main as replay_main
from scripts.falsedeath_v2_ladder_20261002 import STOP
from scripts.falsedeath_v2d_receipt import digest

DEST = OUT/'v2g'


def main() -> None:
    """条件を変えずmode=v2fを再生し、出力先だけ分離する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=tuple(STOP))
    source = parser.parse_args().source
    directory = DEST/'compat_v2f'/source
    sys.argv = ['replay', 'v2f', source, '--out', str(directory), '--end-sec', str(STOP[source])]
    replay_main()
    reference = OUT/'v2f/short'/source
    files = {}
    for name in ('display.npz', 'events.jsonl'):
        assert (directory/name).read_bytes() == (reference/name).read_bytes(), (source, name)
        files[name] = dict(sha256=digest(directory/name), comparison='byte_identical')
    result = dict(source=source, files=files)
    with (directory/'COMPATIBILITY.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
