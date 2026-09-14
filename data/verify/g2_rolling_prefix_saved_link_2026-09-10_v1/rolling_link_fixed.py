"""元保存validatorの純粋な値検査だけを固定ロードする。実ランタイムは作らない。"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'g2_repeated_finalizer_compatibility_2026-09-10_v1/finalizer_evidence.py'
ORIGINAL_SHA = 'bc056e07a43a72c280846aae86449c01dc6ed2eea716c91d4d61e0ae592dfff3'
PREFIX = 'hidden_two_hand_prefix_history/v1'
PLACEMENT = 'placement'
SUPPORT_KEYS = {'prefix', 'final', 'raw', 'pairs', 'physical_certified',
                'probability_assigned', 'accounting_permission', 'current_permission'}
GENERATION_ROOT = ROOT.parent / 'g2_private_suffix_finalizer_stage1_2026-09-10_v1'
GENERATION_PINS = {
    'private_fixed.py': '042ceba922b8076a9adb09a30843e8852fcb53b4dedbf8310baa96b46e3e4a73',
    'private_generation.py': 'eb5dfc5875bc94c9c6102079419ac1cae14897c77e1239c5deaab4148c826f9f'}


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError('rolling_saved_link:' + reason)


def libraries() -> Any:
    require(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == ORIGINAL_SHA, 'validator_source')
    spec = importlib.util.spec_from_file_location('_rolling_saved_link_original_evidence', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generation_library() -> Any:
    """初回Noneの原整合検査をそのまま使い、原import名を必ず戻す。"""
    original = sys.modules.get('private_fixed')
    try:
        for name, pin in GENERATION_PINS.items():
            path = GENERATION_ROOT / name
            require(hashlib.sha256(path.read_bytes()).hexdigest() == pin, 'generation_source')
            spec = importlib.util.spec_from_file_location('_rolling_saved_' + name[:-3], path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if name == 'private_fixed.py':
                sys.modules['private_fixed'] = module
        return module
    finally:
        if original is None:
            sys.modules.pop('private_fixed', None)
        else:
            sys.modules['private_fixed'] = original


def state(row: dict[str, Any]) -> dict[str, Any]:
    return row['decision']['history_state']


def prefix(row: dict[str, Any]) -> bool:
    return type(row.get('prepared')) is dict and row['prepared'].get('kind') == PREFIX


def point(e: Any, value: Any) -> tuple[int, float]:
    require(type(value) in (list, tuple) and len(value) == 2, 'point_shape')
    frame, clock = value
    require(type(frame) is int and frame >= 0 and type(clock) is float
            and clock == frame / e.FPS, 'point_clock')
    return frame, clock


def event(e: Any, row: dict[str, Any]) -> dict[str, Any]:
    proof, scope = row['prepared'], row['scope']
    require(type(proof) is dict and row['decision']['history_consumed'] is True, 'placement_row')
    values = [v for v in state(row)['history'] if v['kind'] == PLACEMENT
              and v['available_at']['frame'] == scope['frame_idx']]
    entry = e.singleton(values, 'rolling_placement_event')
    e.same(entry['event_id'], e.digest(proof), 'rolling_policy_event_digest')
    require(e.clock(entry['available_at'], 1) == scope['frame_idx'], 'placement_event_clock')
    e.same(entry['available_at']['time_sec'], scope['time_sec'], 'rolling_event_time')
    require(type(entry['action']) is int and entry['action'] >= 0, 'event_action')
    return entry
