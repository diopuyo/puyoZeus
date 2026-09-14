"""新基準の未観測隠し段を、原SMの点決めと区別した未較正priorにする。"""
from __future__ import annotations
from dataclasses import dataclass, field, fields, replace
import gate_v4 as OLD

G = OLD.V1
V1 = G  # 原Observerが参照する契約型を維持する。
PRIOR_POLICY = 'native_distribution_or_uniform_for_raw_unknown/v1'
UNOBSERVED_PRIOR = tuple((color, 1.0 / len(G.PROB_COLORS)) for color in G.PROB_COLORS)


@dataclass(frozen=True)
class BasisCandidate(G.BasisCandidate):
    hidden_prior_policy: str = PRIOR_POLICY
    hidden_prior_calibrated: bool = field(default=False, init=False)
    newly_unobserved_columns: tuple[int, ...] = ()
    actual_hidden_probability: tuple[G.Dist, ...] = ()
    actual_sm_hidden_row: tuple[int, ...] = ()


def initial_hidden(raw: G.Grid, sm: G.Grid, hidden: tuple[G.Dist, ...]
                   ) -> tuple[tuple[G.Dist, ...], tuple[int, ...], str | None]:
    """実分布とSMの整合を検査後、rawUNKNOWNの点分布だけを初期化する。"""
    output, changed = [], []
    for col, distribution in enumerate(hidden):
        sm_color, raw_color = sm[0][col], raw[0][col]
        if sm_color == G.COLOR_UNKNOWN:
            if len(distribution) < 2:
                return (), (), 'native_unknown_without_distribution'
        elif distribution != ((sm_color, 1.0),):
            return (), (), 'native_hidden_probability_mismatch'
        if raw_color != G.COLOR_UNKNOWN and raw_color != sm_color:
            return (), (), 'known_hidden_channel_mismatch'
        if raw_color == G.COLOR_UNKNOWN and len(distribution) == 1:
            output.append(UNOBSERVED_PRIOR)
            changed.append(col)
        else:
            output.append(distribution)
    return tuple(output), tuple(changed), None


class SettledBasisGate(OLD.SettledBasisGate):
    def _eligible(self, obs: G.CallObservation) -> str | None:
        if not obs.landing_grace_expired:
            return 'landing_grace_pending'
        if not obs.match_active or obs.effect_gate_window_active is not False:
            return 'inactive_or_window'
        if obs.origin_present:
            return 'origin_present'
        raw, cnn = G.check_grid(obs.raw_grid, 'raw'), G.check_grid(obs.cnn_grid, 'cnn')
        sm = G.check_grid(obs.sm_confirmed_grid, 'sm')
        returned = G.check_grid(obs.returned_grid, 'returned')
        if raw != cnn or raw[G.HIDDEN_ROWS:] != sm[G.HIDDEN_ROWS:]:
            return 'raw_SM_mismatch'
        if sm != returned:
            return 'returned_mismatch'
        visible = G.check_distributions(obs.visible_probability,
            (G.BOARD_ROWS - G.HIDDEN_ROWS) * G.BOARD_COLS, 'visible')
        hidden = G.check_distributions(obs.hidden_probability, G.BOARD_COLS, 'hidden')
        derived, _, reason = initial_hidden(raw, sm, hidden)
        if reason is not None:
            return reason
        assert self.entry_hidden is not None
        return G.visible_mismatch(raw, visible) or G.hidden_mismatch(raw, derived, self.entry_hidden)

    def _issue(self, obs: G.CallObservation) -> BasisCandidate:
        assert obs.raw_grid is not None and obs.sm_confirmed_grid is not None
        assert obs.hidden_probability is not None
        derived, changed, reason = initial_hidden(obs.raw_grid, obs.sm_confirmed_grid,
                                                  obs.hidden_probability)
        G.require(reason is None, 'hidden_initialization_changed')
        # 原観測オブジェクト/SM/PBを変えない。新候補に由来と実分布を併記する。
        base = super()._issue(replace(obs, hidden_probability=derived))
        values = {item.name: getattr(base, item.name) for item in fields(base) if item.init}
        return BasisCandidate(**values, newly_unobserved_columns=changed,
                              actual_hidden_probability=obs.hidden_probability,
                              actual_sm_hidden_row=obs.sm_confirmed_grid[0])
