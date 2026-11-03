"""全通知で公開の有無・保持・表示端点もLinux原票と照合する。"""
from __future__ import annotations
from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
from verify_live_v49f_windows import differences, stable, SUMMARY
from src.prefire_v49d import display

OLD = Path('D:/puyo_analyzer/packaging/build_20261019a')
OUT = Path('D:/puyo_analyzer/packaging/build_20261019a_release')


def main() -> None:
    coefficient = json.loads((ROOT/'docs/verification/prefire_v48/model.json').read_text())['c']
    receipts = []
    for video, game in ((80, 7), (51, 4)):
        left = OLD/f'replay_{video}_{game}_resume/v49f_updates.jsonl'
        right = OUT/f'parity_{video}_{game}/v49f_updates.jsonl'
        latest = [None, None]
        updated, held, endpoints, max_display = 0, 0, 0, 0.
        with left.open() as a, right.open() as b:
            for index, pair in enumerate(zip_longest(a, b)):
                assert None not in pair, '通知数不一致'
                rows = [json.loads(line) for line in pair]
                for key in ('id', 'game', 't', 'origins'):
                    assert stable(rows[0][key]) == stable(rows[1][key]), (index, key)
                for key in ('base', 'value'):
                    assert not differences(stable(rows[0][key]), stable(rows[1][key])), (index, key)
                if rows[0]['value'] is not None:
                    latest = [r['value'] for r in rows]
                    updated += 1
                else:
                    held += 1
                for b3 in (0., .1, .5, .9, 1.):
                    values = [display(b3, b3, r, coefficient,
                        r.get('pre_delta_full', 0) if r else 0, enabled=True) for r in latest]
                    delta = abs(values[0]-values[1])
                    max_display = max(max_display, delta)
                    assert delta <= 1e-9
                    if any(v in (0., 1.) for v in values):
                        assert values[0] == values[1]
                        endpoints += 1
        receipts.append(dict(video=video, game=game, notifications=index+1, updates=updated,
            no_new_publication=held, endpoint_cases=endpoints, max_display_difference=max_display))
    (OUT/'publication_parity.json').write_text(json.dumps(dict(games=receipts, **SUMMARY), indent=2))
    print(json.dumps(receipts), flush=True)


if __name__ == '__main__':
    main()
