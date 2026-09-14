"""実保存v12(35158..35174/1P)の対照と、修復済み入力の人工正常対照を組み立てる。

実保存票は書き換えない。PAIR_TRACE.json の返却盤面と REPRODUCED.json の隠しPBを
そのまま読み、保存されていないchannelは「gateに最も有利な仮定」で埋める。
この仮定は判定を通す方向にしか働かないので、それでも拒否されることが結論を強める。
"""
from __future__ import annotations
from typing import Any
import json
import os
import settled_basis_gate as G

VERIFY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(VERIFY, 'g2_hidden_probability_carry_2026-09-11_v1', 'PAIR_TRACE.json')
REPRODUCED = os.path.join(VERIFY, 'g2_hidden_probability_carry_2026-09-11_v1', 'REPRODUCED.json')
SCOPE: tuple[Any, ...] = ('video_38', 'pipe_1', 3, 0, 0, 1, '1P')
RESET_FRAME = 35158  # software_resetが2→3になる直前の最後の観測frame。
DEADLINE = RESET_FRAME + 14  # 据置の14frame期限。緩めない。
# 保存されていないchannelの埋め方。gateを通す方向の仮定であることを明示する。
UNSAVED_ASSUMPTIONS = ('raw=CNN=SM=返却盤面', '可視PB=返却盤面のpointmass',
                       '隠しPBは35172更新の前後2枚のみ保存（中間frameは前側を流用）')


def load() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with open(TRACE, encoding='utf-8') as handle: trace = json.load(handle)
    with open(REPRODUCED, encoding='utf-8') as handle: reproduced = json.load(handle)
    return [row for row in trace['rows'] if row['kind'] == 'step'], reproduced


def as_grid(grid: list[list[int]]) -> G.Grid:
    return tuple(tuple(int(c) for c in row) for row in grid)


def as_hidden(saved: list[Any]) -> tuple[G.Dist, ...]:
    return tuple(tuple((int(c), float(p)) for c, p in cell) for cell in saved)


def visible_pointmass(grid: G.Grid) -> tuple[G.Dist, ...]:
    return tuple(((grid[r][c], 1.0),) for r in range(G.HIDDEN_ROWS, G.BOARD_ROWS)
                 for c in range(G.BOARD_COLS))


def observation(row: dict[str, Any], hidden: tuple[G.Dist, ...] | None) -> G.CallObservation:
    """保存stepを1件のCallObservationへ写す。保存外の資格は最も有利に仮定する。"""
    returned, frame = row['returned'], int(row['frame_idx'])
    present = returned is not None and returned.get('confirmed') is not None
    grid = as_grid(returned['confirmed']['grid']) if present else None
    return G.CallObservation(
        call_token='saved_v12_step_' + str(frame), frame=frame, scope=SCOPE,
        epoch=SCOPE[2], generation=SCOPE[5],
        state=(returned['state'].lower() if present else 'menu'), exception=row['exception'],
        match_active=True, effect_gate_window_active=False, origin_present=False,
        landing_grace_expired=True, observation_stage=G.ACTUAL_CALL_STAGE, board_present=present,
        raw_grid=grid, cnn_grid=grid, sm_confirmed_grid=grid, returned_grid=grid,
        visible_probability=visible_pointmass(grid) if present else None,
        hidden_probability=hidden if present else None)


def saved_observations() -> list[G.CallObservation]:
    """実保存の並び。35172以降だけ隠しPBが全pointmass空へ潰れている。"""
    steps, reproduced = load()
    prior, after = as_hidden(reproduced['prior_hidden']), as_hidden(reproduced['returned_hidden'])
    result = []
    for row in steps:
        frame = int(row['frame_idx'])
        if frame <= RESET_FRAME: continue
        result.append(observation(row, after if frame >= reproduced['frame'] else prior))
    return result


def repaired_observations() -> list[G.CallObservation]:
    """人工正常対照。35172/35174のraw隠しとPB隠しを35170時点の保持値へ戻しただけ。

    この修復は実保存票の書き換えではなく、親側quarantine.pyの責務を仮置きした入力。
    現実のv12がこの入力になっていたという主張ではない。
    """
    steps, reproduced = load()
    prior = as_hidden(reproduced['prior_hidden'])
    result = []
    for row in steps:
        frame = int(row['frame_idx'])
        if frame <= RESET_FRAME: continue
        returned = row['returned']
        if returned is not None and returned.get('confirmed') is not None and frame >= reproduced['frame']:
            grid = [list(line) for line in returned['confirmed']['grid']]
            grid[0] = [G.COLOR_UNKNOWN if len(prior[c]) >= 2 else grid[0][c] for c in range(G.BOARD_COLS)]
            row = dict(row, returned=dict(returned, confirmed=dict(returned['confirmed'], grid=grid)))
        result.append(observation(row, prior))
    return result


def run(observations: list[G.CallObservation]) -> dict[str, Any]:
    gate = G.SettledBasisGate(SCOPE, RESET_FRAME, DEADLINE)
    for obs in observations: gate.observe(obs)
    candidate = gate.candidate
    return dict(gate_state=gate.state, deadline=DEADLINE, rows=gate.rows,
                issued=candidate is not None,
                candidate=None if candidate is None else dict(
                    frame=candidate.frame, source_call_token=candidate.source_call_token,
                    source_observation_stage=candidate.source_observation_stage,
                    tsumo_fall_frames=list(candidate.tsumo_fall_frames),
                    quality_gate_clear=candidate.quality_gate_clear,
                    physical_certified=candidate.physical_certified,
                    integer_current_permission=candidate.integer_current_permission,
                    accounting_permission=candidate.accounting_permission,
                    production_permission=candidate.production_permission,
                    current_permission=candidate.current_permission,
                    display_update_permission=candidate.display_update_permission,
                    both_sides_stable_confirmed=candidate.both_sides_stable_confirmed,
                    consumed_next_token=candidate.consumed_next_token))


def report() -> dict[str, Any]:
    return dict(source=dict(trace=os.path.basename(TRACE), reproduced=os.path.basename(REPRODUCED)),
                scope=list(SCOPE), reset_frame=RESET_FRAME, deadline=DEADLINE,
                unsaved_channel_assumptions=list(UNSAVED_ASSUMPTIONS),
                saved_v12_actual=run(saved_observations()),
                artificial_repaired_control=run(repaired_observations()),
                saved_record_modified=False, physical_color_truth=False,
                quality_gate_clear=False, production_adoption=False)


if __name__ == '__main__':
    print(json.dumps(report(), ensure_ascii=False, indent=1, allow_nan=False))
