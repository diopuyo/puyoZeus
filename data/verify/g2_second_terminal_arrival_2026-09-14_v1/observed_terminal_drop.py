"""初回30着弾が全候補で終端になる限定回復。量の推定根拠は別途必須。"""
from __future__ import annotations
from dataclasses import replace
from typing import Any

DROP_CAP, MAX_ARRIVALS, MAX_PATHS, OJAMA = 30, 2, 20000, 9
VERSION = 'observed_terminal_cap_drop/v1'


class Domain:
    def __init__(self, engine: Any, split: Any, arrivals: tuple, target: tuple) -> None:
        self.engine, self.split, self.arrivals, self.target = engine, split, arrivals, target
        self.simulator = engine.B.ChainSimulator(exclude_hidden_row_from_pop=True)
        self.matches: set[tuple] = set()
        self.visited, self.drops, self.nonterminal = 0, 0, 0

    def walk(self, grid: tuple, prefix: int) -> None:
        engine, simulator = self.engine, self.simulator
        self.visited += 1
        engine.B.require(self.visited <= MAX_PATHS, 'terminal_drop_path_budget')
        board = engine.B.Board.from_dict({'grid': grid})
        if board.is_dead():
            return
        dropped = simulator.drop_ojama(board, DROP_CAP, seed=0)
        self.drops += 1
        self.nonterminal += int(not dropped.is_dead())
        result = engine.B.grid(dropped)
        if result[engine.B.HIDDEN_ROWS:] == self.target[engine.B.HIDDEN_ROWS:]:
            self.matches.add((prefix, result))
        if prefix == len(self.arrivals):
            return
        for candidate in self.split.enumerate_candidates(engine.H, grid, self.arrivals[prefix].pair):
            placed = self.split.apply_candidate(engine.H, grid, candidate)
            if not simulator.find_erasable_groups(engine.B.Board.from_dict({'grid': placed})):
                self.walk(placed, prefix + 1)


def candidate(parts: Any, commit: Any, current: Any, ledger: Any,
              observed: Any, evidence: dict) -> tuple[Any, dict]:
    """一意な手数・隠しworldだけ返す。原台帳/FIFO/Registryは更新しない。"""
    engine, check = parts.mode.C.T, parts.mode.B.require
    parts.mode.L.check(ledger)
    engine.B.validate(current)
    check(current.scope == ledger.scope and current.deadline == ledger.deadline
          and current.frame < ledger.clock, 'terminal_drop_scope_clock')
    check(ledger.applied and current.tokens[-len(ledger.applied):] == ledger.applied,
          'terminal_drop_applied_base')
    arrivals = ledger.arrivals[len(ledger.applied):]
    check(0 < len(arrivals) <= MAX_ARRIVALS and all(current.frame < a.frame <= ledger.clock
          for a in arrivals), 'terminal_drop_arrival_window')
    check(evidence['kind'] == 'observed_warning_lower_bound/v1'
          and tuple(evidence['scope']) == ledger.scope and evidence['cutoff'] == ledger.clock
          and evidence['conditional_drop_amount'] == DROP_CAP
          and evidence['future_landing_guaranteed'] is False, 'terminal_drop_amount_condition')
    target = engine.B.grid(observed)
    check(any(OJAMA in row for row in target[engine.B.HIDDEN_ROWS:]), 'terminal_drop_no_observed_garbage')
    domain = Domain(engine, commit.P.S, arrivals, target)
    for world in current.worlds:
        board = engine.B.Board.from_dict({'grid': world.grid})
        check(not any(OJAMA in row for row in world.grid) and not board.is_dead()
              and not domain.simulator.find_erasable_groups(board), 'terminal_drop_initial_out_of_scope')
        domain.walk(world.grid, 0)
    check(domain.drops > 0 and domain.nonterminal == 0, 'terminal_drop_multiple_drop_not_excluded')
    check(len(domain.matches) == 1, 'terminal_drop_ambiguous_or_zero_support')
    prefix, grid = next(iter(domain.matches))
    check(prefix == len(arrivals), 'terminal_drop_unplaced_arrival_remaining')
    following = replace(current, frame=ledger.clock,
        tokens=current.tokens + tuple(a.token for a in arrivals), worlds=(engine.B.World(grid, 1.0),))
    engine.B.validate(following)
    receipt = dict(kind=VERSION, amount_condition=evidence, paths=domain.visited,
        first_drop_candidates=domain.drops, first_drop_nonterminal=domain.nonterminal,
        all_placements_before_terminal_drop=True, conditional_hidden_worlds=1,
        probability_calibrated=False, original_fifo_changed=False, quality_gate_clear=False)
    return commit.P.Family(prefix, commit.P.SETTLED, following), receipt
