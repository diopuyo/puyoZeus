"""A2単独の保存済み18再生を基準に、併用の追加差を同じ採点器で求める。"""
from __future__ import annotations
import json
from pathlib import Path
from scripts.a1_completion_replay import ROOT
from scripts import a1_completion_score as scoring

A2 = Path('/mnt/d/puyo_analyzer/wt_a2/logs/a2_safety/replay/a2')


def main() -> None:
    """基準と候補は読み取り参照し、比較結果だけをこのworktreeに保存する。"""
    out = ROOT/'logs/a1b/a2_reference'
    (out/'replay').mkdir(parents=True, exist_ok=True)
    for mode, target in (('prod', A2), ('a1', ROOT/'logs/a1b/combo/replay/a1')):
        link = out/'replay'/mode
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    scoring.OUT = out
    result = scoring.score()
    result['baseline'] = str(A2)
    result['adoption'] = '6d3c4dc'
    (out/'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(result['gates']), flush=True)


if __name__ == '__main__':
    main()
