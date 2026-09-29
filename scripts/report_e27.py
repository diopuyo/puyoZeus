"""E27検収表・全新規死亡・組合せ使用数を一つの証跡へ集約する。"""
from __future__ import annotations

import json
from pathlib import Path
from scripts.run_e3_exchange_eval_20260926 import save_json, SOURCES

OUT = Path('logs/e27')


def hidden_usage() -> dict:
    """最大打ち返しで受け量がゼロになり探索を省略した使用も、原票から数える。"""
    from scripts.run_e17_ablation_20260928 import ALL_SOURCES
    results, combined = {}, dict(exchanges=0, evaluations=0)
    for source in ALL_SOURCES:
        suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
        cases, evaluations = {}, 0
        for line in (OUT/'on'/suffix/'events.jsonl').read_text().splitlines():
            event = json.loads(line)
            for row in event['values']:
                for idx, score in enumerate(row.get('hidden_death_scores', ())):
                    if score is None:
                        continue
                    key = (event['game_idx'], event['exchange_id'], idx)
                    cases.setdefault(key, dict(game=key[0], exchange=key[1], side=idx+1,
                        first_sec=row['t_sec'], maximum_score=score,
                        death_incoming=row['death_incoming'][idx]))
                    evaluations += 1
        results[source] = dict(exchanges=len(cases), evaluations=evaluations, rows=list(cases.values()))
        if source in SOURCES:
            combined['exchanges'] += len(cases)
            combined['evaluations'] += evaluations
    value = dict(three_videos=combined, sources=results)
    save_json(OUT/'HIDDEN_USAGE.json', value)
    return value


def accounting() -> dict:
    """fc過大計上場面で、確率用と死亡用の受け量が分離された原票を残す。"""
    results = {}
    for name in ('e26', 'e27'):
        path = Path('logs')/name/'on/renders/fcXG83vInDY/on/events.jsonl'
        rows = []
        for line in path.read_text().splitlines():
            event = json.loads(line)
            for row in event['values']:
                if 490.8 <= row['t_sec'] <= 492. and 'incoming' in row:
                    rows.append({k: row[k] for k in ('t_sec', 'source', 'incoming', 'death_incoming',
                        'pending_ledger', 'dead_sides', 'base_p1', 'gfe_p1') if k in row})
        results[name] = rows
    save_json(OUT/'FC_ACCOUNTING.json', results)
    return results


def main() -> None:
    """未確認や前倒しを新規正解へ混ぜず、動画は合格時だけ対象にする。"""
    metrics = json.loads((OUT/'on/METRICS.json').read_text())
    deaths = json.loads((OUT/'NEW_DEATH_CASES.json').read_text())
    hidden = json.loads((OUT/'HIDDEN_COUNTS.json').read_text())
    hidden['death_input_usage'] = hidden_usage()
    accounting()
    q, z, d, scene = (metrics[k] for k in ('q', 'zenchi', 'deaths', 'scene'))
    rows = [('q log loss', f"{q['log_loss']:.9f}（{q['frames']}フレーム・{q['matches']}試合）", '≤.513079', 'q'),
        ('zenchi一致', f"{z['hits']}/{z['frames']}＝{100*z['agreement']:.4f}%", '≥82.62%', 'zenchi'),
        ('誤発火', f"{d['false']}/{d['total']}、未判定{d['unlabelled']}", '≤1件、≤1/28', 'deaths'),
        ('2P≤5%初到達', f"{scene['first_sec']}秒（1場面）", '≤2766.0秒', 'scene')]
    lines = ['# E27検収結果', '', '|項目|実測・母数|事前登録基準|判定|', '|---|---|---|---|']
    lines += [f"|{label}|{actual}|{gate}|{'合格' if metrics['gates'][key] else '不合格'}|"
              for label, actual, gate, key in rows]
    lines += ['', '## 新規・前倒し死亡（3動画、E22比）', '',
        '|動画|試合|交換|側|種別|初時刻|実死亡|誤り|', '|---|---:|---:|---|---|---:|---|---|']
    lines += [f"|{r['source']}|{r['game_idx']}|{r['exchange_id']}|{r['side']}|{r['change']}|"
              f"{r['first_sec']:.3f}|{r['actually_died']}|{r['error']}|" for r in deaths['cases']]
    lines += ['', '## 隠し段組合せ', '', json.dumps(hidden, ensure_ascii=False, indent=2), '',
        '合格。指定レビュー動画の生成対象。' if metrics['candidate'] else
        '不合格。採用候補はE22を維持。指定レビュー動画は生成しない。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save_json(OUT/'SUMMARY.json', dict(metrics=metrics, deaths=deaths, hidden=hidden,
        video_generated=False, video_eligible=metrics['candidate']))


if __name__ == '__main__':
    main()
