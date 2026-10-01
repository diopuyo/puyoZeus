"""WINパネル目視との照合と学習非重複監査。採点値は読まない。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.eval_set2_prepare_20261001 import DATA, OUT, observations

EXEV = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer')
VIDEO_ID = 'c0BQoMJwwQU'
E19_MODEL_SHA256 = '5a239cc95b2e35dc40eac35852b3e26143cdfea5f24704da62b189ae4762c762'
# panels_0/1.jpgを目視転記。各候補試合の開始+5秒の左右WIN数。
PANELS = (
    (0,0),(0,1),(1,1),(2,1),(2,2),(3,2),(4,2),(4,3),(4,4),(4,5),
    (4,6),(5,6),(5,7),(5,8),(5,9),(6,9),(6,10),(7,10),(7,11),(8,11),
    (8,12),(9,12),(9,13),(9,14),(10,14),(11,14),(11,15),(12,15),(12,16),(12,17),
    (13,17),(14,17),(14,18),(15,18),(16,18),(17,18),(18,18),(19,18),(19,19),(20,19),
    (21,19),(22,19),(22,20),(22,21),(23,21),(24,21),(24,22),(25,22),(25,23),(25,24),
    (26,24),(26,25),(26,26),(26,27),(26,28),(27,28),(27,29),(27,30),
)
CHUNKS = ((3626.,4293.2),(4293.2,4924.033),(4924.033,5482.066),
          (5482.066,6066.6),(6066.6,6664.166),(6664.166,7033.6))
PREREG_START = 3412.2


def digest(path: Path) -> str:
    """固定した原票とモデルのSHA256を記録する。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_audit() -> dict:
    """E19の実学習148動画をティア表のYouTube IDで照合する。"""
    inputs_path = EXEV/'logs/e19/train/INPUTS.json'
    inputs = json.loads(inputs_path.read_text())
    tiers_path = DATA/'video_tier_index_2026-08-07.tsv'
    with tiers_path.open() as stream:
        tiers = {r['video_name']: r for r in csv.DictReader(stream, delimiter='\t')}
    names = ['video_'+name for name in inputs['videos']]
    ids = [tiers[name]['video_id'] for name in names]
    assert len(names) == len(set(names)) == 148 and VIDEO_ID not in ids
    with (EXEV/'logs/e19/train/rows.csv').open() as stream:
        used = sorted({r['video_id'] for r in csv.DictReader(stream)})
    assert set(used) <= set(names) and len(used) == 148
    with (EXEV/'logs/e15/rows.csv').open() as stream:
        exchange_used = sorted({r['video_id'] for r in csv.DictReader(stream)})
    assert VIDEO_ID not in [tiers[name]['video_id'] for name in exchange_used]
    model = Path('models/landing_counter_prob_v1/model.json')
    assert json.loads((EXEV/model).read_text()) == json.loads(model.read_text())
    assert digest(EXEV/model) == E19_MODEL_SHA256
    return dict(input_videos=len(names), training_videos=len(used), exchange_videos=len(exchange_used),
                overlap_ids=[], names=names, youtube_ids=ids, inputs_sha256=digest(inputs_path),
                tier_sha256=digest(tiers_path), model_sha256=digest(EXEV/'models/landing_counter_prob_v1/model.json'))


def main() -> None:
    """観測と全WIN増分の一致を確認し、次の30先57試合を保存する。"""
    rows = observations()
    assert len(rows)+1 == len(PANELS)
    for index, row in enumerate(rows):
        delta = tuple(b-a for a,b in zip(PANELS[index], PANELS[index+1]))
        assert delta in ((1,0),(0,1))
        assert row['winner'] == ('1P' if delta == (1,0) else '2P'), row
        row.update(game=index+1, panel_before=PANELS[index], panel_after=PANELS[index+1],
                   part=next(f's{i+1}' for i,(start,end) in enumerate(CHUNKS) if start <= row['start'] < end))
    old = json.loads(Path('logs/eval_set/labels.json').read_text())['games']
    assert all(a['end'] <= b['start'] or b['end'] <= a['start'] for a in old for b in rows)
    result = dict(games=rows, checks=dict(panel_match=len(rows), panel_total=len(rows),
                  set1_overlap_games=0, unlabelled_games=0), training=training_audit(),
                  terminal_evidence='tail_0.jpg: 7022〜7028秒 WIN 27–30',
                  boundary_note='次の30先のみ。主判定は3412秒以降の全公式試合を使う')
    (OUT/'labels_next30.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    first = dict(game=1, winner='2P', start=PREREG_START, end=CHUNKS[0][0], part='s0',
                 panel_before=[28,29], panel_after=[28,30],
                 evidence='gap_0.jpg: 3475秒28–29、3485秒28–30')
    all_games = [first] + [dict(row, game=row['game']+1, next30_game=row['game']) for row in rows]
    assert all(a['end'] <= b['start'] or b['end'] <= a['start'] for a in old for b in all_games)
    primary = dict(result, games=all_games, checks=dict(panel_match=len(all_games),
                   panel_total=len(all_games), set1_overlap_games=0, unlabelled_games=0),
                   boundary_note='事前登録の3412秒以降の公式試合すべてを優先。前セット最終戦1+次セット57。'
                   '窓はセット1と同じく前試合終了=次試合開始。セット間メニューもs0の窓に含む。')
    (OUT/'labels.json').write_text(json.dumps(primary, ensure_ascii=False, indent=1))
    print(json.dumps(dict(games=len(rows), checks=result['checks'], training_videos=148)))


if __name__ == '__main__':
    main()
