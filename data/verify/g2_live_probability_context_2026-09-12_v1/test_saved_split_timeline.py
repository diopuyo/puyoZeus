"""既存v67原票の保存先切替を確認。終了検査・実動画品質の合格には使わない。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SAVED = ROOT / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v67_core_activation'
STRIDE = 2


def rows(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (SAVED / name).read_text().splitlines()]


def test_original_timeline_switch_has_no_missing_or_duplicate_frame() -> None:
    history = rows('directional_history.jsonl')
    tracking = rows('PROBABILISTIC_TRACKING.jsonl')
    status = json.loads((SAVED / 'PROBABILISTIC_TRACKING_STATUS.json').read_bytes())
    outer = json.loads((SAVED / 'POSTCOMMIT_CONSUMER_ROWS.json').read_bytes())
    prior = [row['scope']['frame_idx'] if 'decision' in row else row['frame'] for row in history]
    later = [row['scope']['frame_idx'] for row in tracking]
    activation = status['activation']['frame']
    assert prior[-1] == activation and later[0] == activation + STRIDE
    assert prior + later == list(range(prior[0], outer[-1]['frame_idx'] + STRIDE, STRIDE))
    assert len(set(prior + later)) == len(prior + later)
    assert len(tracking) == status['rows']
    assert history[-1]['journal_token'] == status['activation']['source_call_token']


def test_probability_continuation_keeps_original_scope_and_next_call() -> None:
    history = rows('directional_history.jsonl')
    tracking = rows('PROBABILISTIC_TRACKING.jsonl')
    source, run, epoch, pipe, _, _, side = history[-1]['scope']
    tokens = [history[-1]['journal_token'], *[row['journal_token'] for row in tracking]]
    numbers = [int(token.removeprefix('step:')) for token in tokens]
    assert numbers == list(range(numbers[0], numbers[-1] + STRIDE, STRIDE))
    for row in tracking:
        scope = row['scope']
        assert (scope['source_id'], scope['run_id'], scope['pipe_object_id'], scope['side']) == (
            source, run, pipe, side)
        assert scope['generation']['reset_epoch'] == epoch
        assert row['integer_current_published'] is False and row['quality_gate_clear'] is False
