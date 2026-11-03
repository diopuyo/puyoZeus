"""段2で記録した採点器ハッシュと一致する原文だけを固定保存する。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from scripts.gfe_phase_stage2b_train_20261002 import OUT
from scripts.gfe_phase_stage2_score_20261002 import PREFIRE
from scripts.d5_runtime import GIT
from scripts.gfe_phase_stage2b_score_20261002 import USED_SCORERS


def matching_source(name: str, expected: str) -> tuple[bytes, str]:
    """現ファイルかGit履歴から、バイトハッシュが一致した版のみを読む。"""
    path = f'scripts/{name}'
    current = (PREFIRE/path).read_bytes()
    if hashlib.sha256(current).hexdigest() == expected:
        return current, '現ファイル（段2ハッシュ一致）'
    commits = subprocess.check_output([GIT, 'log', '--all', '--format=%H', '--', path]).decode().splitlines()
    for commit in commits:
        content = subprocess.check_output([GIT, 'show', f'{commit}:{path}'])
        normalized = content.replace(b'\r\n', b'\n')
        for value in (content, normalized, normalized.replace(b'\n', b'\r\n')):
            if hashlib.sha256(value).hexdigest() == expected:
                return value, commit
    raise ValueError(f'段2の採点器原文が見つからない: {name}')


def main() -> None:
    """他worktreeへ書き込まず、検証済みの原文と由来を残す。"""
    previous = json.loads(Path('logs/gfe_phase_stage2_20261002/METRICS.json').read_text())
    destination = OUT/'scorer_reference/scripts'
    destination.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, expected in previous['scorer_sha256'].items():
        if name not in USED_SCORERS:
            provenance[name] = dict(used=False, previous_sha256=expected,
                current_sha256=hashlib.sha256((PREFIRE/'scripts'/name).read_bytes()).hexdigest(),
                note='段2の採点器が読み込まない参考ファイル。今回も実行・関数抽出しない。')
            continue
        content, origin = matching_source(name, expected)
        path = destination/name
        if path.exists():
            assert path.read_bytes() == content
        else:
            path.write_bytes(content)
        provenance[name] = dict(used=True, sha256=expected, origin=origin)
    (OUT/'SCORER_PROVENANCE.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(provenance, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
