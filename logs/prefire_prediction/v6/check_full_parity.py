"""補正0の再生と保存済み本番の全列差を記録する。"""
import json
from pathlib import Path

import numpy as np


def main() -> None:
    """差のある時刻と由来を残し、微差を予測効果と混同しない。"""
    out = Path('logs/prefire_prediction/v6')
    baseline = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/set2/replay/prod')
    result = {}
    for index in range(7):
        part = f's{index}'
        with np.load(baseline/part/'display.npz') as left, np.load(out/'replay'/part/'display.npz') as right:
            rows = {}
            for column in ('display_p1', 'display_adv', 'source'):
                changed = np.flatnonzero(left[column] != right[column])
                rows[column] = dict(count=len(changed), examples=[dict(
                    t=float(left['t_sec'][i]), before=str(left[column][i]), after=str(right[column][i]),
                    before_source=str(left['source'][i]), after_source=str(right['source'][i]))
                    for i in changed[:20]])
            result[part] = rows
    (out/'FULL_PARITY.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print({key:{column:row['count'] for column,row in value.items()} for key,value in result.items()})


if __name__ == '__main__':
    main()
