"""理論枝の値を 70 列へ変換する (dl1k 研究専用・既定 OFF)。

Y 専用列は受け側の物理 p 列、Z 専用列は攻め側の物理 p 列だけ有効。他側は NOT_APPLICABLE。
一致度・状態は同じ値を両側に置く。証明は v4.9g の確定した物理証明 (death_sides) だけ。
仮想枝の死亡は hyp_dead_fraction に分け、proof_dead へ昇格しない。
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from src.phase_j._live_ex_frozen.src.dl1k_columns import (SIDES, Y_STEMS, Z_STEMS, AGREE_STEMS, MASK_OK, MASK_NOT_APPLICABLE,
                              MASK_TRUNCATED, MASK_UNPREDICTABLE, blank, normalize)

GENERATE_KIND = 0
TAPES = 2


def agreement(values: Sequence[float]) -> float:
    """両テープの得点の min/max。両方 0 なら 1。"""
    high = max(values)
    return min(values)/high if high > 0 else 1.


def sign(value: float) -> int:
    """0 も独立の符号として扱う。"""
    return (value > 0)-(value < 0)


def source_amount(row: dict, side: int, source: str) -> float:
    """枝の世界に積まれた該当 source の GENERATE 量 (観測済み Y は未反映の残り)。"""
    return float(sum(e['amount'] for e in row['events']
                     if e['kind'] == GENERATE_KIND and e['side'] == side and e['source'] == source))


def plan_values(row: dict, label: str, side: int, observed_scores: Sequence[int] | None) -> list[float]:
    """[得点, 生成, 送出, 枝終了時の自側宛 pending, 手数] の生値。"""
    plan = row[label]
    observed = label == 'Y' and row['factual_reply_used']
    if observed and observed_scores is None:
        raise ValueError('観測済み Y の得点証跡が無い')
    score = sum(observed_scores) if observed else sum(plan['scores'])
    source = f'C{side}' if observed else label
    sent = sum(x['sent_surplus'] for x in row['y_surplus' if label == 'Y' else 'z_surplus'])
    moves = 0 if observed else plan['moves']
    return [float(score), source_amount(row, side, source), float(sent),
            float(row['accounting']['pending'][side]), float(moves)]


def role_values(extra: dict, rows: list[dict], label: str, side: int,
                observed_scores: Sequence[int] | None) -> list[list[float]]:
    """Y または Z の 5 語幹 (生値・正規化) を物理側の列へ置く。"""
    values = [plan_values(r, label, side, observed_scores) for r in rows]
    stems = Y_STEMS if label == 'Y' else Z_STEMS
    physical, other = SIDES[side], SIDES[1-side]
    for stem, items in zip(stems, zip(*values), strict=True):
        mean = sum(items)/TAPES
        extra[f'{stem}_{physical}'] = mean
        extra[f'{stem}_nf_{physical}'] = normalize(mean)
    extra[f'mk_ex_{label.lower()}_{physical}'] = MASK_OK
    extra[f'mk_ex_{label.lower()}_{other}'] = MASK_NOT_APPLICABLE
    return values


def pair_values(extra: dict, rows: list[dict], y: list, z: list, receiver: int) -> None:
    """一致度・発火割合・向きの一致・仮想死亡割合を両側へ同じ値で置く。"""
    attacker = 1-receiver
    differences = [r['accounting']['pending'][attacker]-r['accounting']['pending'][receiver] for r in rows]
    shared = dict(ex_y_power_agree=agreement([v[0] for v in y]), ex_z_power_agree=agreement([v[0] for v in z]),
                  ex_direction_agree=float(sign(differences[0]) == sign(differences[1])),
                  ex_y_fired_fraction=sum(r['Y']['fire_sec'] is not None for r in rows)/TAPES,
                  ex_z_fired_fraction=sum(r['Z']['fire_sec'] is not None for r in rows)/TAPES)
    for index, physical in enumerate(SIDES):
        for stem in AGREE_STEMS:
            extra[f'{stem}_{physical}'] = shared[stem]
        extra[f'ex_hyp_dead_fraction_{physical}'] = sum(
            bool(r['accounting']['projected_dead'][index]) for r in rows)/TAPES
        extra[f'mk_ex_pair_{physical}'] = MASK_OK


def status_values(extra: dict, output: dict) -> None:
    """探索の完了・打切り・完成テープ割合。既知の打切りは有効な 0/1 として残す。"""
    for physical in SIDES:
        extra[f'ex_search_complete_{physical}'] = float(bool(output.get('complete')))
        extra[f'ex_search_cutoff_{physical}'] = float(bool(output.get('cutoff')))
        extra[f'ex_completed_tapes_nf_{physical}'] = output.get('completed_tapes', 0)/TAPES


def proof_values(extra: dict, proof: dict | None) -> None:
    """v4.9g の確定死亡証明だけ。証明が計算できなかった更新は欠測 (0 件合格にしない)。"""
    for index, physical in enumerate(SIDES):
        if proof is None or not proof.get('ok'):
            extra[f'ex_proof_dead_{physical}'] = math.nan
            extra[f'mk_ex_proof_{physical}'] = MASK_UNPREDICTABLE
            continue
        extra[f'ex_proof_dead_{physical}'] = float(index in proof['death_sides'])
        extra[f'mk_ex_proof_{physical}'] = MASK_OK


def failure_reason(output: dict | None) -> int:
    """選べる組が無いときの欠測理由。打切りなら TRUNCATED、それ以外は不成立。"""
    return MASK_TRUNCATED if output and output.get('cutoff') else MASK_UNPREDICTABLE


def branch_values(output: dict | None, proof: dict | None,
                  observed_scores: Sequence[int] | None = None) -> dict[str, Any]:
    """探索出力と証明から 70 列 (available_at は結合時に付ける) を作る。"""
    extra = blank(failure_reason(output))
    proof_values(extra, proof)
    if output is None:
        return extra
    status_values(extra, output)
    # 受け側・攻め側が分かっていれば、役割の無い側は組が無くても NOT_APPLICABLE。
    extra[f"mk_ex_y_{SIDES[output['attacker']]}"] = MASK_NOT_APPLICABLE
    extra[f"mk_ex_z_{SIDES[output['receiver']]}"] = MASK_NOT_APPLICABLE
    selection = output.get('selection')
    if not selection:
        return extra
    rows, receiver = selection['branches'], output['receiver']
    y = role_values(extra, rows, 'Y', receiver, observed_scores)
    z = role_values(extra, rows, 'Z', output['attacker'], None)
    pair_values(extra, rows, y, z, receiver)
    return extra
