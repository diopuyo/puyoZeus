"""H1/H1′: 開始を観測できた試合だけ、計算による死亡確定を公開する。"""
from __future__ import annotations

import numpy as np

from src.board_state_machine import BoardState
from src.indicators_v2 import SEC_PER_HAND

PAIR_CELLS = 2
# 最初の1組を置く時間内のゼロ読みだけを開始証拠にする。採点では選ばない。
BOUNDARY_ZERO_MAX_SEC = SEC_PER_HAND


class MatchHistoryIntegrity:
    """元H1を保ち、明示ON時だけ正式境界と得点の開始証拠を追加する。"""

    def __init__(self, *, retain_audit: bool = False, boundary_score_start: bool = False,
                 strict_integrity: bool = False, menu_boundary_start: bool = False) -> None:
        self.retain_audit = retain_audit
        self.game: int | None = None
        self.games: dict[int, dict] = {}
        self.audit: list[dict] = []
        self.empty_seen = [False, False]
        self.decided = False
        self.boundary_observed = False
        self.boundary_score_start = boundary_score_start
        self.strict_integrity = strict_integrity
        self.menu_boundary_start = menu_boundary_start
        self.menu_only = [True, True]
        self.nonzero_scores: list[dict | None] = [None, None]
        self.start_scores: list[int | None] = [None, None]

    def _observe_boundary_scores(self, sides: tuple, game: int, stamp: float) -> dict:
        """収集器の正式境界でだけ進むgame番号を使い、前試合からの得点低下を確認する。"""
        if not self.boundary_score_start:
            return {}
        scores = [getattr(side, 'score', None) for side in sides]
        if game != self.game:
            continuous = self.game is not None and game == self.game + 1
            self.menu_only = [True, True]
            self.start_scores = [None, None]
            evidence = dict(boundary_sec=stamp if continuous else None,
                previous_game=self.game, previous_nonzero_scores=self.nonzero_scores,
                zero_observed_secs=[None, None], accepted=False)
            self.nonzero_scores = [None, None]
        else:
            evidence = self.games[game]['boundary_score_evidence']
        for idx, score in enumerate(scores):
            if self.start_scores[idx] is None and score is not None:
                self.start_scores[idx] = score
                if score == 0:
                    evidence['zero_observed_secs'][idx] = stamp
            if score is not None and score > 0:
                self.nonzero_scores[idx] = dict(score=score, t_sec=stamp)
        evidence['accepted'] = (evidence['boundary_sec'] is not None
            and all(value is not None for value in evidence['previous_nonzero_scores'])
            and self.start_scores == [0, 0])
        if self.strict_integrity:
            self._restrict_start(sides, evidence)
        return evidence

    def _restrict_start(self, sides: tuple, evidence: dict) -> None:
        """受理時の両側STABLE盤面と、境界から最初のゼロ読みの期限を検査する。"""
        counts = [int(np.count_nonzero(s.confirmed_board._grid))
            if s.state == BoardState.STABLE and s.confirmed_board is not None else None
            for s in sides]
        boundary = evidence['boundary_sec']
        timely = boundary is not None and all(t is not None and
            boundary <= t <= boundary+BOUNDARY_ZERO_MAX_SEC for t in evidence['zero_observed_secs'])
        boards_ok = all(n is not None and n <= PAIR_CELLS for n in counts)
        if self.menu_boundary_start:
            self.menu_only = [old and side.state == BoardState.MENU
                              for old, side in zip(self.menu_only, sides)]
            eligible = [n is not None and n <= PAIR_CELLS or menu
                        for n, menu in zip(counts, self.menu_only)]
            boards_ok = all(eligible)
            evidence.update(menu_only=list(self.menu_only), start_side_eligible=eligible)
        evidence.update(stable_counts=counts, zero_max_sec=BOUNDARY_ZERO_MAX_SEC,
                        timely=timely, boards_ok=boards_ok,
                        accepted=evidence['accepted'] and timely and boards_ok)

    def observe(self, sides: tuple, game: int, stamp: float) -> None:
        """途中開始後の全消しを開始と誤認せず、次試合で観測資格を取り直す。"""
        boundary_evidence = self._observe_boundary_scores(sides, game, stamp)
        if game != self.game:
            self.boundary_observed = self.game is not None
            if not self.retain_audit:
                self.games.clear()
                self.audit.clear()
            self.game, self.empty_seen, self.decided = game, [False, False], False
            self.games[game] = dict(game=game, first_observation_sec=stamp,
                last_observation_sec=stamp, match_start_observed_sec=None,
                history_complete=False, first_occupied_observation_sec=None,
                boundary_observed=self.boundary_observed, empty_observed_secs=[None, None],
                boundary_residual_frames=0)
        row = self.games[game]
        row['last_observation_sec'] = stamp
        if self.boundary_score_start:
            row['boundary_score_evidence'] = boundary_evidence
            if boundary_evidence['accepted'] and not row['history_complete']:
                row.update(history_complete=True, match_start_observed_sec=stamp)
                self.decided = True
        if self.decided:
            return
        counts = [int(np.count_nonzero(s.confirmed_board._grid))
            if s.state == BoardState.STABLE and s.confirmed_board is not None else None
            for s in sides]
        for idx, count in enumerate(counts):
            if count == 0 and getattr(sides[idx], 'score', None) == 0:
                if not self.empty_seen[idx]:
                    row['empty_observed_secs'][idx] = stamp
                self.empty_seen[idx] = True
        occupied = [count for count in counts if count is not None and count > 0]
        if not occupied:
            return
        # 連続観測した境界の得点ゼロ時だけ、初期化直前の前試合残像を保留する。
        if (self.boundary_observed and not all(self.empty_seen)
                and all(getattr(side, 'score', None) == 0 for side in sides)):
            row['boundary_residual_frames'] += 1
            return
        self.decided = True
        complete = all(self.empty_seen) and all(count == PAIR_CELLS for count in occupied)
        row.update(first_occupied_observation_sec=stamp, history_complete=complete,
                   match_start_observed_sec=stamp if complete else None)

    def evidence(self) -> dict:
        """証明に使用した完全性と開始観測時刻を複製する。"""
        row = self.games[self.game]
        keys = ('match_start_observed_sec', 'history_complete')
        result = {key: row[key] for key in keys}
        if self.boundary_score_start:
            from copy import deepcopy
            result['boundary_score_evidence'] = deepcopy(row['boundary_score_evidence'])
        return result

    def apply(self, value: dict, probability: float, proof_rows: list[dict]) -> None:
        """探索結果は監査に残し、履歴欠測時は確率層へ戻して死亡保持を防ぐ。"""
        evidence = self.evidence()
        blocked = not evidence['history_complete'] and value['source'] == 'unavoidable_death'
        detail = dict(**evidence, blocked=blocked,
                      blocked_dead_sides=list(value['dead_sides']) if blocked else [])
        value['history_integrity'] = detail
        for row in proof_rows:
            row['history_integrity'] = dict(evidence,
                blocked=not evidence['history_complete'] and row.get('dead', False))
        self.audit.append(dict(game=self.game, t_sec=value['t_sec'], **detail))
        if blocked:
            self._revoke_proofs(value, evidence)
            value.update(source='S3_landing', p1=probability, dead_sides=[])

    @staticmethod
    def _revoke_proofs(value: dict, evidence: dict) -> None:
        """公開上の死亡を取り消し、計算自体の証拠は別項目へ残す。"""
        for key in ('single_death_proof', 'post_counter_bound'):
            # キャッシュや封印済みrecordの辞書を遡って変更しない。
            if key not in value:
                continue
            rows = []
            for original in value[key]:
                if original is None:
                    rows.append(None)
                    continue
                row = dict(original)
                row['computed_dead'] = row.get('dead', False)
                row['dead'] = False
                row['history_integrity'] = dict(evidence, blocked=row['computed_dead'])
                rows.append(row)
            value[key] = rows

    def summary(self) -> dict:
        """試合観測範囲と全抑止判定の原票を返す。"""
        return dict(games=list(self.games.values()), rows=self.audit)
