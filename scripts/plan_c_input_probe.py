"""案Cの入力粒度を読み取り専用で確認する。"""
from pathlib import Path

import numpy as np

from scripts.prefire_hazard_samples_20260930 import LEAN


def main() -> None:
    """保存項目と最初の観測間隔を出す。"""
    paths = sorted(LEAN.glob('*.npz'))
    print('videos', len(paths))
    with np.load(paths[0]) as data:
        print(paths[0], {k: data[k].shape for k in data.files})
        for key in ('t_sec', 'side', 'game_idx', 'frame'):
            if key in data.files:
                print(key, data[key][:20])
    roots = [Path('/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data'),
             Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs')]
    for root in roots:
        print('root', root)
        print([p.name for p in root.iterdir() if p.is_dir()])


if __name__ == '__main__':
    main()
