"""診断6件の保存原票に対してD1〜D4の境界修正を局所再現する。"""
from __future__ import annotations

import json
import ast
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

from src.exchange_event_tracker import ExchangeChainRecord, ExchangeEventTracker, ExchangeRecord
from src.exchange_single_death_proof import resolve_negative_only
from src.exchange_post_counter_bound import PostCounterDeathBound
from scripts._diag_set2_falsedeath_batch import ROOT

SOURCE = ROOT / 'logs/eval_set/set2_falsedeath'
OUT = ROOT / 'logs/eval_set/falsedeath_fix'


def write_fixture(part: str, stamp: float, side: str, line: int) -> dict:
    """外側呼出し先頭のbefore欠測には、直前呼出しの保存盤面を使う。"""
    previous = None
    with (SOURCE / 'recognition' / f'{part}_writes.jsonl').open() as stream:
        for raw in stream:
            row = json.loads(raw)
            if row['side'] != side:
                continue
            row['before'] = row['before'] if row['before'] is not None else previous
            previous = row['after']
            if (abs(row['t_sec']-stamp) < .001 and row['previous_line'] == line
                    and row['before'] is not None and row['before'][1][2] == 0
                    and row['after'][1][2] != 0):
                return row
    raise ValueError('診断対象の書込みが見つからない')


def falling_case(part: str, stamp: float, side: str, line: int) -> dict:
    """診断時の書込み前盤面を固定し、同じ色候補の昇格をOFF/ONで照合する。"""
    from src.board import Board
    from src.board_state_machine import StateContext, DetectorSignals, _apply_stable_recovery_gate
    from src.placement_inferrer import apply_persistent_landing_color_guard
    row = write_fixture(part, stamp, side, line)
    before, observed = Board.from_list(row['before']), Board.from_list(row['after'])
    values = []
    for enabled in (False, True):
        if part == 's5':
            ctx = StateContext(confirmed_board=before.copy())
            signals = DetectorSignals(time_sec=stamp, is_match_active=True,
                                      cnn_board=observed, hsv_board=observed)
            _apply_stable_recovery_gate(ctx, signals, 1, falling_puyo_integrity=enabled)
            result = ctx.confirmed_board
        else:
            region = NS(cell_sample_rect=lambda r, c: (0, 0, 4, 4))
            result, _ = apply_persistent_landing_color_guard(before, [(1, 2)], observed,
                NS(classify=lambda p: observed.get(1, 2)), np.zeros((4, 4, 3), dtype=np.uint8),
                region, falling_puyo_integrity=enabled)
        values.append(int(result.get(1, 2)))
    assert values == [row['after'][1][2], 0], values
    return dict(stamp=stamp, side=side, old_line=line, values=values,
                fixture='実保存の書込み前盤面と色候補。CNN/HSV一致を再現する局所入力')


def score_case(trace: dict) -> dict:
    """第41試合の実保存連鎖を復元し、式と表示の一致で訂正する。"""
    raw = next(c for c in trace['chains'].values() if c['side'] == '1P' and c['chain_id'] == 6)
    allowed = {f.name for f in fields(ExchangeChainRecord)}
    chain = ExchangeChainRecord(**{k: v for k, v in raw.items() if k in allowed})
    before = chain.provisional_score
    tracker = ExchangeEventTracker(None, score_attribution=True)
    tracker.current = ExchangeRecord(1, 1, chain.trigger_sec, chains=[chain])
    tracker.observe_score('1P', trace['t_sec'], chain.display_score, chain.score_before, chain.formula_total)
    assert before == 880 and chain.provisional_score == 39700
    return dict(before=before, after=chain.provisional_score, reason=chain.score_ready_reason,
                ledger=ledger_case(trace, chain))


def ledger_case(trace: dict, chain: ExchangeChainRecord) -> dict:
    """第41試合の送付・落下履歴をそのまま再構築し、同じ連鎖の得点だけ訂正する。"""
    from src.exchange_pending_ledger import PendingLedger
    from src.scoring import score_to_ojama
    raw, ledger = trace['ledger'], PendingLedger()
    ledger.game = raw['game']
    ledger.events = [(kind, tuple(key), amount) for kind, key, amount in raw['events']]
    ledger.totals = {ast.literal_eval(k): tuple(v) for k, v in raw['totals'].items()}
    ledger.dropped = tuple(raw['dropped'])
    ledger._rebuild()
    before = ledger.pending[0]
    key = (0, chain.chain_id)
    amount = int(score_to_ojama(chain.provisional_score, elapsed_sec=trace['elapsed'][str(key)]).ojama_count)
    ledger.observe(ledger.game, ledger.totals | {key: (amount, True)}, ledger.dropped)
    after = ledger.pending[0]
    assert before == 282 and after == 0, (before, after)
    return dict(incoming_before=before, incoming_after=after, corrected_send=amount)


def main() -> None:
    """単発・E35・会計の実測入力を参照し、試合番号は検査選択だけに用いる。"""
    rows = []
    for case in json.loads((SOURCE / 'cases.json').read_text()):
        game, stamp = case['game'], case['first_sec']
        trace = json.loads((SOURCE / 'baseline' / case['label']['part'] / f'trace_{stamp:.6f}.json').read_text())
        idx = int(case['side'] == '2P')
        if game in (15, 45, 58):
            reason = trace['guards'][idx]['reason']
            assert reason == 'color_score_mismatch'
            old = resolve_negative_only(dict(dead=False, reason=reason))
            new = resolve_negative_only(dict(dead=False, reason=reason), True)
            assert old['dead'] and not new['dead']
            result = dict(flag='D1', before=old, after=new)
        elif game in (13, 43):
            from src.exchange_landing_safety import ColorScoreGuard
            guard = ColorScoreGuard()
            guard.reason, guard.since = trace['guards'][idx]['reason'], trace['guards'][idx]['since']
            projection = NS(safety=NS(blocker=lambda *args: guard.blocker(stamp)))
            reason = PostCounterDeathBound(input_integrity=True).input_blocker(projection, NS(), idx, stamp)
            assert reason == 'board_unsettled'
            result = dict(flag='D2', before='unavoidable_death', after=reason)
        else:
            result = dict(flag='D3', **score_case(trace))
        rows.append(dict(game=game, stamp=stamp, **result))
    OUT.mkdir(parents=True, exist_ok=True)
    d4 = [falling_case('s5', 6308.6, '1P', 7581), falling_case('s6', 7002.516667, '2P', 8523)]
    (OUT / 'LOCAL.json').write_text(json.dumps(dict(cases=rows, d4=d4), ensure_ascii=False, indent=1))
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    main()
