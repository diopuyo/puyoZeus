"""全消しクレジットの保存則・因果性・旧動作の境界を検査する。"""
from __future__ import annotations
from dataclasses import replace
import pytest
from src.b3_score_ledger import ScoreLedgerV2, SideObs, FrameObs, FormulaStepObs

EMPTY = [[0]*6 for _ in range(13)]
START = [row.copy() for row in EMPTY]
START[-1][:4] = [1]*4

def physical(board: list[int], hidden: bool) -> list:
    return [(board, [0]*78, [], [], 40)]

def info(trigger: float, score: int = 40) -> dict:
    return dict(trigger_sec=trigger, before_board=START, chain_count=1, total_score=score)

def feed(core: ScoreLedgerV2, t: float, score: int, trigger: float | None = None,
         formula: int | None = None, game: int = 1, state: str = 'CHAIN') -> object:
    side = SideObs(score=score, state=state, chain_info=info(trigger) if trigger is not None else None,
                   formula_step=FormulaStepObs(t, formula) if formula is not None else None)
    # 採用窓内の2観測を与える。実原票と同様、単発OCRでは採用させない。
    core.update(FrameObs(t, round(t*30), game, True, (side, SideObs(0, state='STABLE'))))
    return core.update(FrameObs(t+.01, round(t*30)+1, game, True, (replace(side, formula_step=None), SideObs(0, state='STABLE'))))

def core() -> ScoreLedgerV2:
    result = ScoreLedgerV2(defer_all_clear=True, all_clear_native=physical)
    feed(result, 0, 0, state='STABLE')
    feed(result, .04, 0, state='STABLE')
    return result

@pytest.mark.parametrize('matched', [False, True])
def test_bonus_waits_then_releases_once(matched: bool) -> None:
    ledger = core()
    feed(ledger, 1, 0, 1, 40 if matched else None)
    feed(ledger, 1.1, 2140, 1)
    feed(ledger, 1.14, 2140, 1)
    assert ledger.st[0].generated_total == 0
    assert ledger.st[0].acc == 40
    assert ledger.st[0].cum_all_clear == 2100
    feed(ledger, 2, 2140, 2, state='STABLE')  # 終端の偽trigger
    assert ledger.all_clear[0].pending == [1]
    feed(ledger, 3, 2140, 3, 40 if matched else None)
    feed(ledger, 3.1, 2180, 3)
    feed(ledger, 3.14, 2180, 3)
    assert ledger.st[0].generated_total == 31
    feed(ledger, 3.18, 2180, 3)
    assert ledger.st[0].generated_total == 31

def test_fixed_thirty_after_margin_and_reset() -> None:
    ledger = core()
    feed(ledger, 1, 0, 1)
    feed(ledger, 1.1, 2140, 1)
    feed(ledger, 1.14, 2140, 1)
    ledger.rate_base = 52
    feed(ledger, 3, 2140, 3)
    feed(ledger, 3.1, 2180, 3)
    feed(ledger, 3.14, 2180, 3)
    assert ledger.st[0].generated_total == 31
    feed(ledger, 4, 0, game=2, state='STABLE')
    assert not ledger.all_clear[0].pending and ledger.st[0].generated_total == 0

def test_large_score_without_evidence_is_not_bonus() -> None:
    ledger = core()
    feed(ledger, 1, 2140)
    feed(ledger, 1.04, 2140)
    assert ledger.st[0].generated_total == 30
    assert not ledger.all_clear[0].pending

def test_consecutive_all_clears() -> None:
    ledger = core()
    for trigger, baseline in [(1, 0), (3, 2140)]:
        feed(ledger, trigger, baseline, trigger)
        feed(ledger, trigger+.1, baseline+2140, trigger)
        feed(ledger, trigger+.14, baseline+2140, trigger)
    assert ledger.st[0].generated_total == 31
    assert ledger.all_clear[0].pending == [3]
    assert ledger.st[0].adopted == 4280

def test_future_evidence_rejected() -> None:
    with pytest.raises(ValueError, match='未来'):
        feed(core(), 1, 0, 2)

def test_bonus_only_notification_does_not_open_pseudo_chain() -> None:
    ledger = core()
    feed(ledger, 1, 0, 1)
    feed(ledger, 1.1, 40, 1)
    sequence = ledger.st[0].chain_seq
    feed(ledger, 1.3, 2140, 1)
    assert ledger.st[0].chain_seq == sequence
    assert ledger.st[0].generated_total == 0
    assert ledger.all_clear[0].pending == [1]

def test_unused_credit_is_discarded_on_reset() -> None:
    ledger = core()
    feed(ledger, 1, 0, 1)
    feed(ledger, 1.1, 2140, 1)
    feed(ledger, 2, 0, game=2, state='STABLE')
    assert ledger.st[0].generated_total == 0 and not ledger.all_clear[0].pending

def test_bonus_offsets_existing_pending_before_sending() -> None:
    from src.b3_score_ledger import _Batch
    ledger = core()
    feed(ledger, 1, 0, 1)
    feed(ledger, 1.1, 2140, 1)
    ledger.batches[0].append(_Batch(n=40, src_side=1, chain_id=1, gen_t=2))
    feed(ledger, 3, 2140, 3)
    feed(ledger, 3.1, 2180, 3)
    assert ledger.st[0].offset_total == 31
    assert ledger.st[0].sent_total == 0
    assert ledger._pending(0) == 9

def test_ordinary_2100_points_are_not_deferred() -> None:
    ledger = core()
    feed(ledger, 1, 0, formula=2100)
    feed(ledger, 1.1, 2100)
    assert ledger.st[0].generated_total == 30
    assert not ledger.all_clear[0].pending

def test_formula_residual_without_clear_evidence_stays_uncertain() -> None:
    ledger = core()
    feed(ledger, 1, 0, formula=40)
    feed(ledger, 1.1, 2140)
    assert ledger.st[0].generated_total == 0
    assert ledger.all_clear[0].uncertain_points == 2100
    assert not ledger.all_clear[0].pending


@pytest.mark.parametrize('recognized,uncertain,allowed', [(True, 0, True), (False, 0, False), (True, 2100, False)])
def test_death_guard_only_releases_reconciled_credit(recognized: bool, uncertain: int, allowed: bool) -> None:
    from types import SimpleNamespace as NS
    from src.prefire_facts_v49d import FactsTracker
    from src.prefire_proof_contract import ATTACK_ALL_CLEAR_AMBIGUOUS
    tracker = FactsTracker(NS())
    record = NS(side=0, chain_id=1, t_start=1., t_end=2.)
    credit = NS(recognized_chain_ids={1} if recognized else set(), uncertain_points=uncertain)
    ledger = NS(defer_all_clear=True, all_clear=[credit, credit], chains=[record],
        batches=[[], [NS(src_side=0, chain_id=1, n=15, confirmed=True)]],
        st=[NS(chain=None), NS(chain=None)])
    tracker.tainted.add((0, 1))
    tracker.verdicts[0, 1.] = dict(start=1., end=2., total=15, reflected=15, finished=True)
    assert tracker.attack_reason(ledger, 1) == (None if allowed else ATTACK_ALL_CLEAR_AMBIGUOUS)
