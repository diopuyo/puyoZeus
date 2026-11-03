"""途中結果の少量表示。原票の大きい配列は標準出力へ流さない。"""
from __future__ import annotations

from collections import Counter
import json

from scripts._diag_dv2d_loss_inventory import OUT, BASE, read, resource


def firsts() -> None:
    """55件について、本番時刻以前の最終判定だけを表示する。"""
    for r in read(OUT/'EVENTS.json'):
        if r['group'] not in ('set1', 'set2'):
            continue
        i = int(r['side'] == '2P')
        a = r['at_first'] or {}
        one = (a.get('single_death_proof') or [None, None])[i] or {}
        post = next((v for v in a.get('post_counter_bound', []) if v['side'] == r['side']), {})
        history = a.get('history_integrity', {})
        print(r['group'], r['game'], r['record'], r['local_game'], round(r['delay_sec'], 3),
              'D1='+str(one.get('reason')), 'E='+str(post.get('reason')),
              'sub='+str([p['reason'] for p in post.get('proofs', [])]),
              'H='+str(history.get('blocked')), 'complete='+str(history.get('history_complete')))


def history() -> None:
    """開始未観測の減少幅を同一母数で比べる。"""
    rows = read(OUT/'OBSERVATIONS.json')
    totals = Counter()
    for name, value in rows.items():
        games = {g['game']:g for g in value['games']}
        counts = Counter()
        for original in value['history']:
            g = games[original['game']]
            counts['n'] += 1
            if original['history_complete']:
                continue
            counts['incomplete'] += 1
            for key in ('boundary_sec', 'zero_sec', 'zero_stable_sec', 'boundary_empty_sec'):
                counts[key] += g[key] is not None
        totals.update(counts)
        print('H1', name, dict(counts))
    print('H1 TOTAL', dict(totals))


def main() -> None:
    """証拠の概要と入力差分の種類を確認する。"""
    print('RESOURCE', resource())
    firsts()
    history()
    diff = read(OUT/'INPUT_DIFF_SUMMARY.json')
    for name, value in diff.items():
        print('D4', name, value['fields'])


if __name__ == '__main__':
    main()
