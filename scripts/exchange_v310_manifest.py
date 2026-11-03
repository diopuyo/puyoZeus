"""第17〜20試合を採点前に固定し、旧未見を開発参考へ降格する。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from scripts.exchange_v35_manifest import read, VIDEOS, V34
from scripts.exchange_review_v32_data import BASE, source, fire_rows
from scripts.exchange_review_io import save, sha256

OUT = BASE/'v3_10'
EVIDENCE = Path('docs/verification/exch_review_v310')
GAMES = (17, 18, 19, 20)


def main() -> None:
    if (OUT/'manifest.json').exists():
        raise FileExistsError('採点前固定票は上書きしない')
    selected = []
    for video in VIDEOS:
        targets = [r for r in fire_rows(source(video)) if r['game_idx'] in GAMES
                   and len(r['exchange_input']['colors']) == 4]
        selected.append(dict(video=video, games=list(GAMES), targets=targets,
            input_sha256=sha256(source(video)/'inputs.jsonl.gz')))
    result = dict(version='v3.10', frozen_utc=datetime.now(timezone.utc).isoformat(),
        selected=selected, policy_sha256=sha256(V34/'policy.json'),
        rule='動画32/34/51/61/92の第17〜20試合の全初段通知。採点不能は理由と件数を残す。',
        old_unseen='v3.9の97件は原因調査専用の参考。改善の主張には使わない。',
        limitation='動画自体は既知。未見試合の検証。')
    save(OUT/'manifest.json', result)
    compact = [dict(e, targets=[dict(event_id=f['event_id'], game=f['game_idx'],
        t=f['t_sec'], side=f['exchange_input']['attacker']) for f in e['targets']]) for e in selected]
    save(EVIDENCE/'manifest.json', dict(result, selected=compact, sha256=sha256(OUT/'manifest.json')))
    print([(e['video'], len(e['targets'])) for e in selected], flush=True)


if __name__ == '__main__':
    main()
