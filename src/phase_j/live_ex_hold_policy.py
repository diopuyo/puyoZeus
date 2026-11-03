"""研究用。通信の欠測だけを、同じ試合・側の直前の行で期限つき保持する。"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any

from .live_ex_fresh import EX_FRESH_SEC_DEFAULT, validate_ex_fresh_sec

EX_HOLD_MAX_SEC = 0.5
EX_ORIGINAL_MAX_AGE_SEC = EX_FRESH_SEC_DEFAULT
SIDES = ('p1', 'p2')
RAW_STEMS = tuple(f'ex_{role}_{name}' for role in ('y', 'z')
                  for name in ('score', 'generated', 'sent', 'pending'))
RAW_STEMS = RAW_STEMS[:4] + ('ex_hY',) + RAW_STEMS[4:] + ('ex_hZ',)
STATE_STEMS = ('ex_y_power_agree', 'ex_z_power_agree', 'ex_direction_agree',
               'ex_y_fired_fraction', 'ex_z_fired_fraction', 'ex_search_complete',
               'ex_search_cutoff', 'ex_completed_tapes_nf', 'ex_proof_dead', 'ex_hyp_dead_fraction')
LEARNING = tuple(f'{s}_nf_{p}' for s in RAW_STEMS for p in SIDES) + tuple(
    f'{s}_{p}' for s in STATE_STEMS for p in SIDES)
MASKS = tuple(f'mk_ex_{s}_{p}' for s in ('y', 'z', 'pair', 'proof') for p in SIDES)
EX_COLUMNS = tuple(f'{s}_{p}' for s in RAW_STEMS for p in SIDES) + LEARNING + MASKS + ('av_ex_p1', 'av_ex_p2')
TRANSPORT_MISSING = frozenset(('not_yet', 'stale', 'not_completed', 'key_mismatch'))

@dataclass
class Snapshot:
    values: dict[str, Any]
    attached_t: float
    input_t: float
    row_key: str
    cache_key: str | None

@dataclass
class HoldPolicy:
    max_sec: float = EX_HOLD_MAX_SEC
    previous: dict[str, Snapshot] = field(default_factory=dict)
    cache: dict[tuple[str, str], Snapshot] = field(default_factory=dict)
    group: Any = None
    last_t: float = -math.inf
    extra_columns: tuple[str, ...] = ()
    fresh_sec: float = EX_ORIGINAL_MAX_AGE_SEC

    def __post_init__(self) -> None:
        self.fresh_sec = validate_ex_fresh_sec(self.fresh_sec)

    def clear(self, side: str) -> None:
        self.previous.pop(side, None)
        self.cache = {key: val for key, val in self.cache.items() if key[0] != side}

    def start_row(self, row: dict) -> None:
        if row['group'] != self.group:
            self.previous.clear()
            self.cache.clear()
            self.group, self.last_t = row['group'], -math.inf
        if row['t'] < self.last_t:
            raise ValueError('同じ試合で時刻が逆転')
        self.last_t = row['t']
        for side in SIDES:
            if row['reset'].get(side, False) or not row['fight']:
                self.clear(side)

    def remember(self, row: dict, side: str) -> None:
        values = {k: row['extra'].get(k, math.nan) for k in (*EX_COLUMNS, *self.extra_columns) if k.endswith('_'+side)}
        current_key = row.get('current_cache_keys', {}).get(side)
        source_key = row.get('source_cache_keys', {}).get(side)
        snap = Snapshot(values, row['t'], row['input_t'], row['row_key'], source_key)
        self.previous[side] = snap
        if source_key and source_key == current_key:
            self.cache[(side, source_key)] = snap

    def select(self, row: dict, side: str) -> tuple[Snapshot | None, str]:
        key = row.get('current_cache_keys', {}).get(side)
        snap = self.cache.get((side, key)) if key else None
        if self.usable(snap, row['t']):
            return snap, 'cache'
        snap = self.previous.get(side)
        if self.usable(snap, row['t']):
            return snap, 'held'
        return None, 'missing'

    def usable(self, snap: Snapshot | None, t: float) -> bool:
        if snap is None or not isinstance(snap.input_t, (int, float)):
            return False
        return (0 <= t-snap.attached_t <= self.max_sec
                and 0 <= t-snap.input_t <= self.fresh_sec+self.max_sec)

    def validate_original(self, row: dict) -> None:
        if row['status'] != 'ok':
            return
        if row['input_t'] is None or not 0 <= row['t']-row['input_t'] <= self.fresh_sec:
            raise ValueError('元の採用行が設定された入力の鮮度以内ではない')
        for side in SIDES:
            available = row['extra'][f'av_ex_{side}']
            if not math.isfinite(available) or available > row['t']:
                raise ValueError('元の採用行の公開時刻が未来または不明')

    def apply(self, row: dict) -> dict:
        self.validate_original(row)
        self.start_row(row)
        result = dict(row['extra'])
        if row['status'] != 'ok':
            result = {k: (8 if not row['fight'] else 3) if k in MASKS else math.nan for k in (*EX_COLUMNS, *self.extra_columns)}
        for side in SIDES:
            audit = dict(mode='original', held_sec=0., source_row_key=row['row_key'], source_input_t=row['input_t'])
            if row['fight'] and row['status'] == 'ok':
                self.remember(row, side)
            elif row['fight'] and row['status'] in TRANSPORT_MISSING:
                snap, mode = self.select(row, side)
                audit['mode'] = mode
                if snap is not None:
                    result.update(snap.values)
                    audit.update(held_sec=row['t']-snap.attached_t, source_row_key=snap.row_key,
                                 source_input_t=snap.input_t)
            else:
                self.clear(side)
            for name, value in audit.items():
                result[f'ex_hold_{name}_{side}'] = value
        return result
