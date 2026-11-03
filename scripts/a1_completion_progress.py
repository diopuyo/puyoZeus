"""再生中の進捗と、完了した既存記録の互換性だけを読み取る。"""
from __future__ import annotations

import json
from scripts.a1_completion_replay import OUT
from scripts.falsedeath_collect_20261001 import SOURCES


def main() -> None:
    """未完了の原票は開かず、確定した成果物だけを照合する。"""
    completed = {mode: sorted(p.parent.name for p in (OUT/'replay'/mode).glob('*/status.json'))
                 for mode in ('original', 'prod', 'a1')}
    compatibility = {}
    for name in SOURCES:
        if name not in completed['original'] or name not in completed['prod']:
            continue
        folders = [OUT/'replay'/mode/name for mode in ('original', 'prod')]
        compatibility[name] = {file: (folders[0]/file).read_bytes() == (folders[1]/file).read_bytes()
                               for file in ('display.npz', 'events.jsonl')}
    value = dict(completed=completed, compatibility=compatibility)
    if all('review' in completed[mode] for mode in ('prod', 'a1')):
        from scripts import falsedeath_score_20261001 as score
        score.OUT = OUT
        # 第14試合の未完成原票は使わず、scene関数だけを読む。
        from scripts import eval_set2_regression_20261001 as regression
        value['scene'] = {}
        for mode in ('prod', 'a1'):
            regression.directory = lambda source: OUT/'replay'/mode/source
            value['scene'][mode] = regression.scene()
    (OUT/'PROGRESS.json').write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
