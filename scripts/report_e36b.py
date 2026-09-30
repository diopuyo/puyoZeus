"""E36bをE36と同じ門(logs/e36b/PREREGISTRATION.md)で採点し、第14試合の再発有無を併記する。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts import report_e35 as e35
from scripts import report_e36 as base
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e36b')
GAME, SIDE, SOURCE = 14, '1P', 'q_7gc4TgFig'


def game14_false_rows(events: list[dict]) -> list[float]:
    """q第14試合(勝者1P)で1Pの死亡を断定した時刻を返す。空なら誤確定なし。"""
    return [v['t_sec'] for e in events if e['game_idx'] == GAME for v in e['values']
            if v['source'] == 'unavoidable_death' and SIDE in v['dead_sides']]


def report() -> dict:
    """E36の採点器を出力先だけ差し替えて実行し、第14試合の行を追記する。"""
    base.OUT = OUT
    summary = base.report()
    e35.OUT = OUT
    summary['game14_false_times'] = game14_false_rows(e35.events(SOURCE))
    save_json(OUT/'SUMMARY.json', summary)
    print(json.dumps(dict(game14_false_times=summary['game14_false_times']), ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    report()
