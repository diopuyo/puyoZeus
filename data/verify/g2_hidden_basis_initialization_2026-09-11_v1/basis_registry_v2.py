"""原の基準登録/保存順序を維持し、実隠し分布と初期priorの由来を保存する。"""
from __future__ import annotations
from dataclasses import asdict
import json
from types import FunctionType
from typing import Any
import basis_registry_connection as OLD

B, C, S = OLD.B, OLD.C, OLD.S


class Connection(OLD.Connection):
    def publish(self, item: Any) -> None:
        candidate = self.observer.gate.candidate
        if candidate is None or self.binding is not None:
            return
        self.qualify(candidate, item)
        B.require(getattr(candidate, 'hidden_prior_policy', None)
                  == 'native_distribution_or_uniform_for_raw_unknown/v1', 'hidden_prior_policy')
        B.require(candidate.hidden_prior_calibrated is False, 'hidden_prior_not_calibrated')
        board = B.Board.from_dict({'grid': candidate.raw_grid})
        probability = B.ProbabilisticBoard.from_board(board)
        for col, distribution in enumerate(candidate.hidden_probability):
            probability.cell(0, col).probs = dict(distribution)
        value, mass, removed = C.establish_conditioned(candidate.scope, candidate.frame,
                                                       self.deadline, board, probability)
        packet = dict(kind='actual_settled_probabilistic_basis', source_call_token=item['token'],
            basis_deadline=self.observer.gate.deadline, tracking_deadline=self.deadline,
            retained_mass=mass, gravity_removed_worlds=removed, state=S.encode(value),
            initial_candidate=asdict(candidate), fresh_prior_not_temporal_posterior=True,
            integer_current_published=False, legacy_collector_append=False, quality_gate_clear=False)
        encoded = json.dumps(packet, ensure_ascii=False, allow_nan=False)
        self.binding = self.registry.bind(self.recovery.factory, value, item['token'])
        self.stream.write(encoded + '\n')
        self.stream.flush()
        self.rows += 1


def install(*args: Any, **kwargs: Any) -> Any:
    return FunctionType(OLD.install.__code__, dict(vars(OLD), Connection=Connection))(*args, **kwargs)
