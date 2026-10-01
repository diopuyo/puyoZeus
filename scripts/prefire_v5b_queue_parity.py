"""欠測改善で正解率の母数が変わるため、修正前後の共通既知枠を別に照合する。"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

from scripts import prefire_stable_queue_accuracy_20261001 as reference
from src.prefire_stable_queue import SideQueue
from src.prefire_v5b_queue import SideQueueV5B

OUT = Path('logs/prefire_prediction/v5b/final/queue_common.json')


def compare(items: list[tuple]) -> dict:
    """正解は将来の設置差分。予測器へはその時点の観測しか渡さない。"""
    old, new = SideQueue(), SideQueueV5B()
    boards, rows = [], []
    for time, board, reading in items:
        if not boards or board != boards[-1]:
            boards.append(board)
        old.observe(time, board, reading)
        new.observe(time, board, reading)
        rows.append((len(boards)-1, old.known(), new.known()))
    truth = [reference.placed(a, b) for a, b in zip(boards, boards[1:])]
    counts = {name: Counter() for name in reference.SLOTS}
    for index, before, after in rows:
        for slot, name in enumerate(reference.SLOTS):
            target = index+slot
            if target >= len(truth) or truth[target] is None:
                continue
            left, right = (tuple(sorted(k[2*slot:2*slot+2])) for k in (before, after))
            if 0 not in left and 0 not in right:
                counts[name].update(common=1, before_hit=int(left == truth[target]),
                                    after_hit=int(right == truth[target]), changed=int(left != right))
            elif 0 in left and 0 not in right:
                counts[name].update(newly_known=1, newly_known_hit=int(right == truth[target]))
    return counts


def main() -> None:
    """5記録の共通既知枠と追加既知枠を別集計する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    path = parser.parse_args().out
    if path.exists():
        raise FileExistsError('既存の読み監査を上書きしない')
    result = {}
    for source in reference.BASELINE_DIRS:
        result[source] = [compare(items) for items in reference.frames(source)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
