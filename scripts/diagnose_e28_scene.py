"""E27を変更せず、指定場面の判定入力と不採用条件を記録する。"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import csv
import json
from pathlib import Path
from typing import Any
import numpy as np

from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e27 import OPTIONS
from scripts.run_e3_exchange_eval_20260926 import save_json
from src.board import COLOR_UNKNOWN
from src.chain import ChainSimulator
from src.exchange_hidden_row_death import enumerate_hidden
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import score_to_ojama

OUT = Path('logs/e28')
START, END = 2755., 2771.
CONTEXT_START, CONTEXT_END = 2750., 2773.
SIMULATOR = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)


def encoded(value: Any) -> str:
    """複数着弾の各枝を、CSV一セル内でも復元可能に保存する。"""
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def entry_for(engine: Any, chain: Any) -> dict:
    """参照のみ行い、候補の採否・台帳・探索キャッシュを変更しない。"""
    return next((e for e in engine.entries.values() if e['chain'] is chain), {})


def sample_status(side: Any, entry: dict, hidden: bool = False) -> str:
    """先に弾かれる条件を表示し、複数原因の原票は別列に残す。"""
    if entry.get('active'):
        return '採用中'
    if entry.get('pending'):
        return '次段得点一致待ち'
    board = getattr(side, 'midchain_board', None)
    if side.state.name != 'GRAVITY_SETTLE':
        return 'GRAVITY_SETTLE以外'
    if board is None:
        return '途中盤面なし'
    grid = board._grid
    if np.any(grid[1:] == COLOR_UNKNOWN):
        return '可視段UNKNOWN'
    count = int(np.count_nonzero(grid[0] == COLOR_UNKNOWN))
    if hidden and not 1 <= count <= 3:
        return '隠し段UNKNOWNが1〜3個以外'
    if not hidden and count:
        return '隠し段UNKNOWN'
    if np.any((grid[:-1] != 0) & (grid[1:] == 0)):
        return '浮遊セルあり'
    if entry.get('formula') is None:
        return '段数・得点式なし'
    if entry.get('samples', 0) < 2:
        return '同一落ち切り盤面が1観測のみ（必要2）'
    return '残り連鎖なし、または同段で処理済み'


def chain_info(chain: Any, elapsed: float) -> dict | None:
    """予測得点と観測済み得点を混ぜず、換算レートの基準も残す。"""
    if chain is None:
        return None
    value = asdict(chain)
    value.update(provisional_score=chain.provisional_score, rate_elapsed=elapsed,
        provisional_ojama=int(score_to_ojama(chain.provisional_score, elapsed_sec=elapsed).ojama_count))
    return value


class Trace:
    """観測後の参照だけでCSVを作り、元のE27出力との同一性を検証する。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.raw: list[dict] = []
        self.candidates: list[dict] = []

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        result, _, _, stamp, game, *_ = inputs
        if not CONTEXT_START <= stamp <= CONTEXT_END:
            return
        projection, tracker = overlay._landing_projection, overlay.tracker
        chains = [tracker.latest_chain(label) for label in ('1P', '2P')]
        mid = entry_for(overlay._midchain, chains[1])
        hidden = entry_for(overlay._hidden_death, chains[1])
        infos = [chain_info(c, projection.safety.elapsed.get((i, c.chain_id), tracker._score_elapsed))
                 if c else None for i, c in enumerate(chains)]
        last = projection.last or {}
        raw = dict(t_sec=stamp, game=game, source=tracker.source, p1=tracker.probability,
            exchange_open=tracker.current is not None,
            chains=infos, counts=projection.counts.copy(), last=last,
            pending=projection.safety.ledger.pending,
            totals=[dict(side=k[0]+1, chain_id=k[1], amount=v[0], verified=v[1])
                    for k, v in projection.safety.ledger.totals.items()],
            chunks=projection.safety.ledger.chunks,
            midchain_status=sample_status(result.p2, mid),
            hidden_status=sample_status(result.p2, hidden, True),
            formula=hidden.get('formula'), hidden_samples=hidden.get('samples', 0),
            safety=projection.safety.signature(projection, tracker, stamp))
        # 生きた辞書への参照を残さず、その時点の値を固定する。
        raw = json.loads(encoded(raw))
        self.raw.append(raw)
        self.rows.append(self.flatten(raw, result))
        self.inspect_single_sample(raw, result.p2, hidden, overlay._hidden_death.palettes[1])

    @staticmethod
    def flatten(raw: dict, result: Any) -> dict:
        """未計算の最大火力は空欄とし、0個と明確に区別する。"""
        value, c1, c2 = raw['last'], *raw['chains']
        multi = value.get('multi_landing', [{}, {}])[1]
        board = getattr(result.p2, 'midchain_board', None)
        def second(key: str) -> Any:
            return value.get(key, [None, None])[1]
        return dict(t_sec=raw['t_sec'], game=raw['game'], p2_state=result.p2.state.name,
            source=raw['source'], p2_probability=None if raw['p1'] is None else 1-raw['p1'],
            landing_evaluation_sec=value.get('t_sec'),
            reevaluated=value.get('t_sec') == raw['t_sec'],
            p2_observed_chain_count=raw['counts'][1],
            p2_formula_score=c2['formula_total'] if c2 else None,
            p2_midchain=raw['midchain_status'], p2_hidden=raw['hidden_status'],
            p2_hidden_samples=raw['hidden_samples'],
            p2_hidden_unknown=None if board is None else int(np.count_nonzero(board._grid[0] == COLOR_UNKNOWN)),
            p2_visible_unknown=None if board is None else int(np.count_nonzero(board._grid[1:] == COLOR_UNKNOWN)),
            p2_floating=None if board is None else bool(np.any((board._grid[:-1] != 0) & (board._grid[1:] == 0))),
            p2_predicted_score=c2['predicted_final_score'] if c2 else None,
            p2_predicted_chain_count=c2['predicted_chain_count'] if c2 else None,
            p2_completion_certain=second('completion_certain'),
            p2_hidden_maximum_score=second('hidden_death_scores'),
            p2_maximum_extra_counter_ojama=second('optimistic_send'),
            p2_near_future_ojama=second('near_future_send'),
            p2_resolving_credit_ojama=second('resolving_send'),
            p1_formula_score=c1['formula_total'] if c1 else None,
            p1_provisional_send_ojama=c1['provisional_ojama'] if c1 else None,
            p2_provisional_counter_ojama=c2['provisional_ojama'] if c2 else None,
            p2_net_incoming_probability_ojama=second('incoming'),
            p2_net_incoming_death_ojama=second('death_incoming'),
            p2_pending_now_ojama=raw['pending'][1],
            p2_required_cancel_ojama=second('required_cancel') if second('near_future_send') is not None else None,
            p2_hands=second('hands'), p2_safety=raw['safety'][1],
            p2_death='2P' in value.get('dead_sides', []),
            p2_multi_reason=multi.get('reason'), p2_multi_nodes=multi.get('nodes'),
            p2_multi_rounds=encoded(multi.get('rounds')), p2_multi_detail=encoded(multi),
            p2_final_non_death_condition=multi.get('reason') if raw['exchange_open']
                else '交換終了・死亡保持なしのため着弾評価を再実行せず通常評価')

    def inspect_single_sample(self, raw: dict, side: Any, entry: dict, colors: tuple) -> None:
        """診断専用: 2観測条件を外した一枚の上限。判定へは一切戻さない。"""
        if sample_status(side, entry, True) != '同一落ち切り盤面が1観測のみ（必要2）':
            return
        options, trials = enumerate_hidden(side.midchain_board, colors, *entry['formula'], SIMULATOR)
        self.candidates.append(dict(t_sec=raw['t_sec'], formula=entry['formula'],
            colors=colors, board=side.midchain_board._grid.tolist(), trials=trials,
            options=options, adoption='診断専用・未採用'))


def main() -> None:
    """元記録を再生し、すべての観測と再評価を区別して保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-review', action='store_true')
    live = parser.parse_args().live_review
    out = OUT/'live' if live else OUT
    baseline = Path('logs/review_zenchi_g41_43_e27') if live else Path('logs/e27/on/review')
    record = baseline/'inputs.jsonl.gz' if live else Path('logs/e26/records/review.jsonl.gz')
    dest = out/'replay'
    dest.mkdir(parents=True, exist_ok=True)
    trace = Trace()
    result = replay(record, dest,
        Path('models/exchange_event_v3'), live_count=True, observer=trace, **OPTIONS)
    result['equivalence'] = compare(baseline, dest)
    save_json(out/'REPLAY_VERIFICATION.json', result)
    save_json(out/'SCENE_RAW.json', trace.raw)
    save_json(out/'SINGLE_SAMPLE_DIAGNOSTIC.json', trace.candidates)
    for name, rows in (('SCENE.csv', [r for r in trace.rows if START <= r['t_sec'] <= END]),
                       ('SCENE_CONTEXT.csv', trace.rows)):
        with (out/name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(encoded(result), flush=True)


if __name__ == '__main__':
    main()
