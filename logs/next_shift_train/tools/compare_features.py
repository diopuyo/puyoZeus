"""台の確認: 再計算した特徴と exev logs/e15/features の一致を数える。"""
import json, sys
from pathlib import Path
import numpy as np
ref = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/e15/features')
new = Path(sys.argv[1])
out = {}
for vdir in sorted(new.glob('video_*')):
    games = sorted(vdir.glob('game_*.npz'))
    stat = dict(games=len(games), rows=0, s1_equal=0, s3_equal=0, info_equal=0)
    for g in games:
        a, b = np.load(g), np.load(ref / vdir.name / g.name)
        stat['rows'] += len(a['s1'])
        stat['s1_equal'] += int(np.array_equal(a['s1'], b['s1'], equal_nan=True))
        stat['s3_equal'] += int(np.array_equal(a['s3'], b['s3'], equal_nan=True))
        stat['info_equal'] += int(json.loads(g.with_suffix('.json').read_text()) == json.loads((ref / vdir.name / g.with_suffix('.json').name).read_text()))
        if len(a['s1']) == len(b['s1']) and len(a['s1']):
            stat.setdefault('s1_cells_diff', 0); stat['s1_cells_diff'] += int((~np.isclose(a['s1'], b['s1'], equal_nan=True)).sum())
            stat.setdefault('s3_cells_diff', 0); stat['s3_cells_diff'] += int((~np.isclose(a['s3'], b['s3'], equal_nan=True)).sum())
    out[vdir.name] = stat
print(json.dumps(out, indent=1))
