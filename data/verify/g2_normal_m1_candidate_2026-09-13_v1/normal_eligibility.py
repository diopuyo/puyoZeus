"""原資格共通部を保持し、正常Native側の未反映だけを明示検査する。"""
from typing import Any
import original_eligibility as OLD
import second_physical as P

F, SIDES = OLD.F, OLD.SIDES


def reasons(session: Any, frame: int) -> tuple[str, ...]:
    flags, witness = session.evaluation_flags, session.witness
    row = session.state['provisional_context_observer'].rows[-1]
    F.require(not flags.closed and flags.error is None and flags.journal is witness.journal, 'flags_lifetime')
    F.require(row['capture_status'] == 'CAPTURED' and not row['failures']
              and not any(row['upstream_failures'].values()), 'context_failure')
    steps = witness.pair(frame)
    F.require([s['side'] for s in steps] == list(SIDES) and set(flags.latest) == set(SIDES), 'both_sides')
    found = []
    for side, step in zip(SIDES, steps):
        found.extend(OLD.side_flags(flags.latest[side], step, frame))
        holds = set(row['sides'][side]['hold_reasons'])
        F.require(holds <= OLD.NORMAL_SIDE_HOLDS, 'unexpected_context_hold')
        probability = row['sides'][side]['before_hold']['probability']
        F.require(probability['present'] is False or probability['type_valid'] is True
                  and not probability['errors'], 'invalid_present_probability')
        found.extend(side + ':' + h for h in sorted(holds - {'next_missing'}))
    F.require(set(row['hold_reasons']) <= OLD.NORMAL_COMMON_HOLDS, 'unexpected_common_hold')
    found.extend(row['hold_reasons'])
    session.owner.require_live()
    for side, mode in zip(SIDES, session.current_modes):
        if mode is None:
            found.append(side + ':basis_missing')
            continue
        F.require(type(mode) is P.Mode and mode.side == side and mode.error is None, 'normal_mode_type')
        F.require(mode.connection.registry is session.owner.registry and mode.native is not None,
                  'normal_mode_registry')
        if mode.closed or mode.retired_receipt is not None:
            found.append(side + ':basis_missing')
        elif mode.native.pending:
            found.append(side + ':native_pending')
    return tuple(dict.fromkeys(found))
