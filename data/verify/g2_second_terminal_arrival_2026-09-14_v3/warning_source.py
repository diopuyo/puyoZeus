"""原Native認証済み同call画像の予告履歴。推定条件と将来保証を分離する。"""
from __future__ import annotations
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any
from src.scoring import is_pure_chain_score_delta

STRIDE, FPS, OJAMA, DROP_CAP = 2, 60, 9, 30
VERSION = 'observed_warning_lower_bound/v2'
# 元score_ocr.FORMULA_LEFT_MINと同じ、最小消去4個×10点。学習閾値ではない。
MIN_ERASURE_SCORE = 40


def original_local(binding: Any, item: dict) -> dict:
    """同じ元step/pipe/時計を要求。保存画像や別runの同frameを受け付けない。"""
    mode, check = binding.mode, binding.L.require
    recovery = mode.connection.recovery
    frame = item['scope']['frame_idx']
    check(type(mode.native) is binding.N.Recorder and mode.native.last_frame == frame
          and item['token'] in mode.native.seen_calls, 'warning_native_unverified')
    check(item['frame'].f_code in recovery.journal.codes and item['pipe'] is recovery.pipe,
          'warning_original_step')
    local = item['frame'].f_locals
    check(local['self'] is recovery.pipe and local['side'] == binding.ledger.scope[-1]
          and local['frame_idx'] == frame and local['time_sec'] == frame / FPS,
          'warning_original_clock_side')
    check(mode.connection.binding.scope == binding.ledger.scope
          and frame == binding.ledger.clock, 'warning_ledger_scope')
    return local


def flags(local: dict, item: dict, engine: Any) -> dict:
    score = local.get('score_d_for_self')
    known = type(score) in (int, float) and math.isfinite(score)
    chain = local.get('own_chain_active')
    active = local.get('is_active')
    confirmed = local['sm'].context.confirmed_board
    return dict(context_known=known and type(chain) is bool and type(active) is bool,
        active=active, own_chain_active=chain, own_score_delta=score if known else None,
        active_origin=any(event.get('active_origin') is not None for event in item['events']),
        chain_event=local.get('chain_event') is not None,
        confirmed_has_ojama=None if confirmed is None else
            any(OJAMA in row for row in engine.B.grid(confirmed)[engine.B.HIDDEN_ROWS:]))


def score_evidence(check: Any, window: tuple[dict, ...]) -> dict:
    """元連鎖flagと組み合わせ、小さい非連鎖得点を相殺へ読み替えない。"""
    deltas = tuple(row['own_score_delta'] for row in window)
    check(all(type(value) in (int, float) and math.isfinite(value)
              and value >= 0 and value == int(value) for value in deltas), 'warning_score_unknown')
    # 非10倍数だけでも40点+落下1点を見逃すため、区間累積にも最小消去量の上限を置く。
    check(sum(deltas) < MIN_ERASURE_SCORE
          and not any(is_pure_chain_score_delta(int(value)) for value in deltas),
          'warning_possible_erasure_score')
    return dict(kind='small_non_chain_score_with_original_no_chain_flags/v1',
        total_delta=sum(deltas), positive_calls=[row['call_token'] for row in window if row['own_score_delta'] > 0],
        source_rule='src.scoring.is_pure_chain_score_delta',
        minimum_erasure_score=MIN_ERASURE_SCORE, positive_score_alone_is_cancellation=False)


def select(check: Any, rows: tuple[dict, ...], scope: tuple, current_frame: int,
           frame: int) -> dict:
    """連続陽性と相殺/発火不在を条件に、観測後30仮説への根拠だけ返す。"""
    window = tuple(row for row in rows if current_frame < row['frame'] <= frame)
    check(tuple(row['frame'] for row in window) == tuple(range(current_frame + STRIDE, frame + STRIDE, STRIDE)),
          'warning_missing_coverage')
    check(bool(window) and all(tuple(row['scope']) == scope for row in window), 'warning_scope')
    check(all(row['context_known'] and row['active'] is True
              and row['own_chain_active'] is False
              and row['active_origin'] is False and row['chain_event'] is False
              and row['confirmed_has_ojama'] is not None for row in window), 'warning_cancellation_or_unknown')
    score = score_evidence(check, window)
    positive = [row for row in window if row['observation']['status'] == 'POSITIVE']
    check(bool(positive), 'warning_no_positive')
    first = positive[0]['frame']
    landed = [row['frame'] for row in window if row['confirmed_has_ojama']]
    check(bool(landed) and first < min(landed), 'warning_not_before_confirmed_drop')
    tail = [row for row in window if row['frame'] >= first]
    check(all(row['observation']['status'] == 'POSITIVE'
              and row['observation']['conditional_lower_bound'] == DROP_CAP
              and tuple(row['observation']['scope']) == scope
              and row['observation']['frame'] == row['frame']
              and row['observation']['call_token'] == row['call_token']
              and bool(row['observation']['image_sha256']) for row in tail), 'warning_positive_interrupted')
    return dict(kind=VERSION, scope=scope, cutoff=frame, first_positive=first, score_evidence=score,
        first_confirmed_ojama=min(landed), last_positive=tail[-1]['frame'],
        source_calls=[row['call_token'] for row in tail], conditional_drop_amount=DROP_CAP,
        assumption='同側の予告陽性が継続し、原stepで自発火/相殺を観測しない条件付き初回着弾',
        calibrated=False, future_landing_guaranteed=False, quality_gate_clear=False)


class History:
    def __init__(self, binding: Any, sampler: Any, path: Path) -> None:
        self.binding, self.sampler = binding, sampler
        self.rows: list[dict] = []
        self.stream = path.open('x', encoding='utf-8')
        self.closed = False

    def observe(self, item: dict) -> None:
        check = self.binding.L.require
        check(not self.closed, 'warning_closed')
        local = original_local(self.binding, item)
        frame, scope = item['scope']['frame_idx'], self.binding.ledger.scope
        check(not self.rows or self.rows[-1]['frame'] + STRIDE == frame, 'warning_history_gap')
        observation = self.sampler.sample(scope, frame, item['token'], local.get('frame_bgr'))
        row = dict(scope=scope, frame=frame, call_token=item['token'], observation=asdict(observation),
                   **flags(local, item, self.binding.mode.prefix.engine))
        self.stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.stream.flush()
        self.rows.append(row)

    def close(self) -> None:
        if not self.closed:
            self.stream.close()
            self.closed = True
