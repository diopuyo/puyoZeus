"""既存式リプレイに候補幅だけを添える。未来の終了観測は読まない。"""
from __future__ import annotations
from pathlib import Path
from scripts import prefire_v45_origins as previous
from scripts.replay_exchange_v310 import sent_values
from scripts.exchange_review_io import save
from src.prefire_confidence import candidate_summary

BASE_SNAPSHOT = previous.snapshot


def snapshot(state: dict, inp: dict, policy: dict, elapsed: float, initial: bool) -> dict:
    output = BASE_SNAPSHOT(state, inp, policy, elapsed, initial)
    values = (state['initial']['variants']['board_candidates'] if initial else
              sent_values(state['pool'], inp, state['formulas'], state['reflected'], policy))
    conflict = state['initial']['conflicts']['board_candidates'] if initial else not bool(values)
    output['confidence'] = candidate_summary(values, len(state['formulas']),conflict)
    return output


def collect(video: int, destination: Path, native: object, policy: dict) -> list[dict]:
    result = []
    previous.snapshot = snapshot
    try:
        for inp, stream in previous.sources(video):
            result.extend(previous.replay(inp, stream, policy, native))
    finally:
        previous.snapshot = BASE_SNAPSHOT
    result.sort(key=lambda row: row['input_sec'])
    save(destination/f'origins_{video}.json', result)
    return result
