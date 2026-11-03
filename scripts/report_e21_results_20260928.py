"""固定再生と148原票の監査を、観測できた母数ごとに報告する。"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from src.exchange_event_record import read_records
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json
from scripts.run_e21_death_candidate_20260928 import OUT
from scripts.aggregate_e3_exchange_eval_20260926 import outcomes, FPS


def winner_label(source: str, row: dict) -> str | None:
    """死亡文字が未観測の区間も、既存評価と同じ勝敗原票を照合する。"""
    if source not in SOURCES:
        return None
    windows, _ = outcomes(source)
    window = next((w for w in windows if w['start']/FPS <= row['start_sec'] < w['end']/FPS), None)
    if window is not None and window['winner'] is not None:
        return window['winner']
    labels = json.loads(Path('logs/e10b/panel_outcomes.json').read_text())
    label = next((r for r in labels if r['source'] == source and r['game_idx'] == row['game']), None)
    return label['winner'] if label is not None else None


def replay_audit(source: str) -> dict:
    """死亡確定が観測できる再生では、死亡側・生還側と候補を照合する。"""
    directory = OUT/'on'/source if source in ('review', 'zenchi') else OUT/'on/renders'/source/'on'
    rows = json.loads((directory/'death_candidates.json').read_text())
    deaths: dict[int, set] = {}
    for row in read_records(Path('logs/e16/records')/f'{source}.jsonl.gz'):
        if row['kind'] == 'update':
            result, _, _, _, game, *_ = row['args']
            deaths.setdefault(game, set()).update(getattr(result, 'confirmed_dead_sides', ()))
    for row in rows:
        observed = deaths.get(row['game'], set())
        winner = winner_label(source, row)
        row['candidate_false'] = ((row['side'] not in observed) if observed else
                                  (row['side'] == winner) if winner is not None else None)
        held = {r['trigger_sec'] for r in row['held']}
        accepted = {r['trigger_sec'] for r in row['accepted']}
        row['held_new_fires'] = len(held)
        row['correctly_held'] = len(held-accepted) if row['outcome'] == 'confirmed_death' else 0
    counts = dict(candidates=len(rows), confirmed_deaths=sum(map(len, deaths.values())),
        false_candidates=sum(r['candidate_false'] is True for r in rows),
        unknown_candidates=sum(r['candidate_false'] is None for r in rows),
        cleared_candidates=sum(r['outcome'] == 'cleared' for r in rows),
        held_new_fires=sum(r['held_new_fires'] for r in rows),
        correctly_held=sum(r['correctly_held'] for r in rows))
    return dict(source=source, counts=counts, rows=rows,
                definition='候補誤りは同試合生還（死亡確定信号、欠測時は既存勝敗原票）。解除は別列。正しく保留は未受理のまま当該候補が死亡確定した通知。')


def main() -> None:
    """集計不能な値を0に置換せず、そのまま監査未完了として記録する。"""
    replay = [replay_audit(s) for s in (*SOURCES, 'zenchi', 'review')]
    total = Counter()
    for row in replay[:len(SOURCES)]:
        total.update(row['counts'])
    corpus = json.loads((OUT/'corpus/SUMMARY.json').read_text())
    metrics = json.loads((OUT/'on/METRICS.json').read_text())
    save_json(OUT/'AUDIT.json', dict(replays=replay, replay_three=dict(total), corpus=corpus))
    save_json(OUT/'METRICS.json', dict(fixed_replay=metrics, corpus_audit_complete=False,
        accepted=False, limitation=corpus['limitation']))
    print(json.dumps(dict(metrics=metrics, replay_three=dict(total), corpus=corpus), ensure_ascii=False))


if __name__ == '__main__':
    main()
