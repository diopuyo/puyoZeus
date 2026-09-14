"""同runの因果prefixだけから開始資格を判定する候補。盤面評価は別のSTABLEゲート。"""
from __future__ import annotations
import hashlib
import json
from typing import Any

FPS, STRIDE, SIDES = 60, 2, 2
DEBOUNCE_SEC = 5.0  # 凍結collect_boards_lean.GAME_BOUNDARY_DEBOUNCE_SEC。
VISUAL_PERSIST_SEC = 0.5  # 原RecognitionPipelineの30 frame/60fps。
SCORE_RESET_THRESHOLD = 500  # 原frozen OjamaAccountingTrackerの同名値。producerで実値照合必須。
FRAME_EPSILON = 1e-7
ACTIVITY = ('generated', 'offset_uncapped', 'dropped_uncapped', 'clamp_loss', 'finalized')


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('start_qualification:' + reason)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, allow_nan=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def validate_trace(rows: list[dict], first: int, last: int) -> None:
    require(bool(rows) and [r['frame'] for r in rows] == list(range(first, last + STRIDE, STRIDE)), 'coverage')
    previous = None
    for row in rows:
        require(type(row['game']) is int and row['game'] >= 0 and type(row['active']) is bool, 'state_types')
        for key in ('states', 'scores', 'gates', 'unsettled'):
            require(type(row[key]) is list and len(row[key]) == SIDES, 'side_shape')
        require(all(type(v) is bool for key in ('gates', 'unsettled') for v in row[key]), 'bool_types')
        require(all(type(v) is int or v is None for v in row['scores']), 'score_types')
        require(all(v in ('menu', 'stable', 'chain', 'gravity_settle', 'ojama_fall', 'tsumo_fall')
                    for v in row['states']), 'board_state')
        require(set(row['activity']) == set(ACTIVITY), 'activity_keys')
        counters = [row[k] for k in ('resets', 'pending', 'capped', 'leftover')] + list(row['activity'].values())
        require(all(type(v) is list and len(v) == SIDES and all(type(n) is int and n >= 0 for n in v)
                    for v in counters), 'counter_types')
        if previous is not None:
            require(row['game'] >= previous['game'], 'game_regressed')
            require(all(a >= b for a, b in zip(row['resets'], previous['resets'])), 'reset_regressed')
            require(all(row['activity'][k][s] >= previous['activity'][k][s]
                        for k in ACTIVITY for s in range(SIDES)), 'activity_regressed')
        previous = row


def menu_exit_followup(rows: list[dict], entry: int | None, index: int, side: int) -> bool:
    """観測済みMENU入口に続く最初の離脱と原score下落だけを同じ境界へ束ねる。"""
    if entry is None or entry <= 0 or index <= entry:
        return False
    if rows[index]['states'][side] == 'menu':
        return False
    if any(row['states'][side] != 'menu' for row in rows[entry:index]):
        return False
    before, current = rows[entry - 1]['scores'][side], rows[index]['scores'][side]
    return (type(before) is int and type(current) is int
            and before - current >= SCORE_RESET_THRESHOLD)


def reset_evidence(rows: list[dict], after: int) -> tuple[list[int | None], list[list[int]]]:
    found: list[int | None] = [None] * SIDES
    follows: list[list[int]] = [[] for _ in range(SIDES)]
    for index in range(1, len(rows)):
        old, new = rows[index - 1], rows[index]
        if new['frame'] <= after:
            continue
        for side in range(SIDES):
            delta = new['resets'][side] - old['resets'][side]
            if not delta:
                continue
            if delta == 1 and old['states'][side] != 'menu' and new['states'][side] == 'menu':
                found[side], follows[side] = index, []
            elif delta == 1 and menu_exit_followup(rows, found[side], index, side):
                follows[side].append(new['frame'])
            else:
                found[side], follows[side] = None, []
    return found, follows


def resets_before(rows: list[dict], after: int) -> list[int | None]:
    return reset_evidence(rows, after)[0]


def accounting_reason(rows: list[dict], after: int) -> str | None:
    resets = resets_before(rows, after)
    if any(index is None for index in resets):
        return reset_reason(rows, after)
    begin = min(index for index in resets if index is not None)
    baseline, last = rows[begin - 1], rows[-1]
    if any(row['activity'] != baseline['activity'] for row in rows[begin:]):
        return 'activity_since_reset'
    if any(last[k][s] != 0 for k in ('pending', 'capped', 'leftover') for s in range(SIDES)):
        return 'nonzero_accounting_start'
    if any(last['unsettled']) or not last['active']:
        return 'start_unsettled_or_inactive'
    if 'menu' in last['states']:
        return 'start_board_menu'
    return None


def reset_reason(rows: list[dict], after: int) -> str:
    for old, new in zip(rows, rows[1:]):
        if new['frame'] <= after:
            continue
        for side in range(SIDES):
            if new['resets'][side] == old['resets'][side]:
                continue
            if old['states'][side] == new['states'][side] == 'menu':
                return 'menu_reset_edge_mismatch'
            if new['states'][side] != 'menu':
                return 'nonmenu_reset_in_window'
    return 'observed_both_menu_resets_missing'


def visual_reason(prefix: list[dict], rise: dict) -> str | None:
    raw_frame = round(rise['raw_value'] * FPS)
    require(abs(rise['raw_value'] * FPS - raw_frame) < FRAME_EPSILON, 'visual_occurrence_frame')
    if prefix[-1]['frame'] - raw_frame < VISUAL_PERSIST_SEC * FPS:
        return 'visual_persistence_missing'
    positions = {row['frame']: index for index, row in enumerate(prefix)}
    if raw_frame not in positions or positions[raw_frame] == 0:
        return 'visual_transition_unobserved'
    index = positions[raw_frame]
    if prefix[index - 1]['active'] or not all(row['active'] for row in prefix[index:]):
        return 'visual_signal_discontinuous'
    return None


def decide(value: dict, *, debounce_sec: float) -> dict:
    """資格後の行は開始判定に使わず、次gameには資格を引き継がない。"""
    rows, events = value['start_counter_trace'], value['boundary_events']
    validate_trace(rows, value['first_frame'], value['last_frame'])
    require(type(debounce_sec) in (int, float) and debounce_sec == DEBOUNCE_SEC, 'debounce')
    rises = [e for e in events if e['list'] == 'visual_rise_times']
    if not rises:
        return dict(eligible=False, reason='visual_start_missing')
    rise = rises[-1]
    frame = rise['available_frame']
    prefix = [r for r in rows if r['frame'] <= frame]
    require(prefix and prefix[-1]['frame'] == frame, 'visual_available_frame')
    require(rise['available_sec'] == frame / FPS and rows[0]['frame'] / FPS <= rise['raw_value'] <= frame / FPS,
            'visual_clock')
    if prefix[-1]['gates'] != [True, True]:
        return dict(eligible=False, reason='visual_gate_disabled')
    visual = visual_reason(prefix, rise)
    if visual is not None:
        return dict(eligible=False, reason=visual)
    advances = [e for e in events if e['list'] == 'advance_times' and e['available_frame'] <= frame]
    if not advances or abs(advances[-1]['raw_value'] - rise['raw_value']) >= debounce_sec:
        return dict(eligible=False, reason='visual_game_binding_missing')
    require(prefix[-1]['game'] == len(advances), 'visual_game_ordinal')
    if rows[-1]['game'] != prefix[-1]['game']:
        return dict(eligible=False, reason='start_game_expired')
    after = rises[-2]['available_frame'] if len(rises) > 1 else rows[0]['frame'] - STRIDE
    reason = accounting_reason(prefix, after)
    evidence = dict(identity=value['identity'], game=prefix[-1]['game'], qualification_frame=frame,
                    visual=rise, advance=advances[-1], scores=prefix[-1]['scores'],
                    reset_frames=[None if i is None else prefix[i]['frame'] for i in resets_before(prefix, after)],
                    menu_exit_reset_frames=reset_evidence(prefix, after)[1],
                    score_reset_threshold=SCORE_RESET_THRESHOLD, trace_sha256=digest(prefix))
    return dict(eligible=reason is None, reason=reason, evidence=evidence,
                evidence_sha256=digest(evidence), method='observed_menu_reset_visual_start/v1')
