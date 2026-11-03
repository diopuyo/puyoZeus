"""qの従来指標とwarmupの固定勝敗照合を、合否条件を変えずに追記する。"""
from __future__ import annotations

import json

from scripts.falsedeath_replay_20261001 import OUT
from scripts import aggregate_e3_exchange_eval_20260926 as metrics
from scripts.report_e7_exchange_20260927 import load_display
from scripts.eval_set2_regression_20261001 import fixed_labels
from scripts.falsedeath_score_20261001 import SETS

DEST = OUT/'v2d'


def warmup_truth(rows: list[dict], terminals: dict, fixed: list[dict]) -> list[dict]:
    """既存ラベルを優先し、ラベルが無い場合だけ同じ試合の観測死亡を使う。"""
    labels = {name:json.loads(path.read_text())['games'] for name, (_, path) in SETS.items()}
    result = []
    for row in rows:
        source, game, side, stamp = (row[k] for k in ('record','game','side','first_sec'))
        part = next((name for name,(parts,_) in SETS.items() if source in parts), None)
        winners = {g['winner'] for g in labels[part] if g['start'] <= stamp < g['end']} if part else set()
        winners |= {r['winner'] for r in fixed if r['source']==source and r['game_idx']==game and r['winner']}
        observed = {r['side'] for r in terminals[source] if r['game']==game}
        known = len(winners) == 1 or (not winners and len(observed) == 1)
        false = side in winners if winners else side not in observed
        result.append(dict(**row, winners=sorted(winners), observed_dead_sides=sorted(observed),
                           verdict=('誤確定' if false else '正しい確定') if known else '未確定'))
    return result


def main() -> None:
    """全体採点完了後に同じ分母のq指標とwarmup別表を保存する。"""
    scored = json.loads((DEST/'RESULT_v2d.json').read_text())
    windows, provenance = metrics.outcomes('q_7gc4TgFig')
    q = {mode:metrics.m3_scores(load_display(OUT/'replay'/mode/'q_7gc4TgFig/display.npz'), windows)
         for mode in ('prod','v2d')}
    terminals = json.loads((OUT/'TERMINALS.json').read_text())
    fixed = fixed_labels()
    warmup = {mode:warmup_truth(rows,terminals['prod' if mode=='prod' else 'all'],fixed)
              for mode,rows in scored['warmup'].items()}
    with (DEST/'SUPPLEMENT.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(q=q, q_label_provenance=provenance, warmup=warmup), stream, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
