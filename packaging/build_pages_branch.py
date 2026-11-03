"""site/ の4ファイルだけで親のないPages用の枝を作る。既存枝の更新とpushはしない。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

SITE_FILES = ('index.html', 'site.css', 'mascot_expressions_preview.png', '.nojekyll')
PUBLIC_NAME = 'diopuyo'
PUBLIC_EMAIL = 'diopuyo2@gmail.com'


def create_branch(repo: Path, source_ref: str = 'HEAD', branch: str = 'gh-pages') -> str:
    """元のコミット履歴やdocsを含めず、新しいrefだけを原子的に作る。"""
    git = ['git', '-C', str(repo)]
    subprocess.run(git + ['check-ref-format', '--branch', branch], check=True, capture_output=True)
    ref = 'refs/heads/' + branch
    if subprocess.run(git + ['show-ref', '--verify', '--quiet', ref]).returncode == 0:
        raise ValueError('既存の枝は更新しません: ' + branch)
    source = subprocess.check_output(git + ['rev-parse', '--verify', source_ref + '^{commit}']).decode().strip()
    entries = []
    for name in SITE_FILES:
        oid = subprocess.check_output(git + ['rev-parse', '--verify', source + ':site/' + name]).decode().strip()
        entries.append(f'100644 blob {oid}\t{name}\n')
    tree = subprocess.check_output(git + ['mktree'], input=''.join(entries).encode()).decode().strip()
    environment = dict(os.environ)
    environment.update(GIT_AUTHOR_NAME=PUBLIC_NAME, GIT_COMMITTER_NAME=PUBLIC_NAME,
                       GIT_AUTHOR_EMAIL=PUBLIC_EMAIL, GIT_COMMITTER_EMAIL=PUBLIC_EMAIL)
    commit = subprocess.check_output(git + ['-c', 'commit.gpgsign=false', 'commit-tree', tree,
        '-m', '配布サイトの下書きを独立した木として作成'], env=environment).decode().strip()
    subprocess.run(git + ['update-ref', ref, commit, '0' * 40], check=True)
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-ref', default='HEAD')
    parser.add_argument('--branch', default='gh-pages')
    args = parser.parse_args()
    print(create_branch(args.repo, args.source_ref, args.branch))


if __name__ == '__main__':
    main()
