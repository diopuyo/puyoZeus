"""保存評価を実公開器へ再投入し、番号以外の公開値をバイト照合する。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any
from unittest.mock import patch

from scripts import replay_terminal_hold_20261012 as replay
from scripts.visualize_advantage_overlay import (
    _detect_score_reset, accept_formal_boundary, update_score_reset_latch)
from src.phase_j.live_match_number import PublicMatchNumber


def boundaries(rows: list[dict], inputs: list[dict]) -> list[dict]:
    """保存理由を優先。旧eは保存得点と入力世代から復元し、不明分は省略しない。"""
    epochs = {}
    for row in inputs:
        if 'epoch' in row:
            epochs.setdefault(row['epoch'], row)
    changes = sorted((r['at'], e, r.get('device_id')) for e, r in epochs.items())
    previous, latch, last, game, epoch, device = (None, None), False, None, 0, 0, None
    seen, result = set(), []
    last_input = max(r['at'] for r in inputs)
    for row in rows:
        if row['frame'] in seen:
            continue
        seen.add(row['frame'])
        current = next(((e, d) for t, e, d in reversed(changes)
                        if t <= row['captured_at']), (0, None))
        forced = current[0] != epoch and current[0] > 0
        recovery = device is not None and device == current[1]
        epoch, device = current
        reset = _detect_score_reset(row['score1'], row['score2'], *previous)
        scored = accept_formal_boundary(reset, latch, row['t_sec'], last)
        if scored:
            last = row['t_sec']
        latch = update_score_reset_latch(latch, reset, row['score1'], row['score2'])
        previous = row['score1'], row['score2']
        if row['game'] == game:
            continue
        game = row['game']
        reason = row.get('terminal_boundary_reason')
        known = reason is not None or row['captured_at'] <= last_input
        if reason is not None:
            scored, forced = reason == 'scored', reason == 'input_recovery'
            recovery = forced
        result.append(dict(game=game, t_sec=row['t_sec'], captured_at=row['captured_at'],
            forced=forced if known else False, scored=scored if known else False,
            recovery=recovery if known else False, known=known, reason=reason))
    return result


def replay_public(events: list[tuple], numbers: PublicMatchNumber | None) -> list[dict]:
    original = replay.LivePublisher

    def factory(*args: Any, **kwargs: Any) -> Any:
        owner = original(*args, **kwargs)
        owner.public_matches = numbers
        # l配布と同じ既存設定。新フラグはこの保持規則を変更しない。
        owner.terminal.input_recovery = True
        return owner

    with patch.object(replay, 'LivePublisher', factory):
        return replay.replay(events, True)


def without_number(records: list[dict]) -> bytes:
    return pickle.dumps([{k: v for k, v in r.items() if k != 'game'} for r in records])


def run(source: Path, out: Path) -> dict:
    spool = source/'spool'
    rows = replay.load_rows(spool/'evaluations.pickle')
    inputs = replay.load_rows(spool/'input_events.pickle')
    publications = replay.load_rows(spool/'publications.pickle')
    original = pickle.dumps(rows)
    limits = boundaries(rows, inputs)
    numbers = PublicMatchNumber()
    for bound in limits:
        numbers.boundary(bound['game'], bound['forced'], bound['scored'], bound['recovery'])
    events = sorted([(r['evaluated_at'], 'evaluation', r) for r in rows]
                    + replay.input_deliveries(inputs, publications), key=lambda e: e[0])
    before, after = replay_public(events, None), replay_public(events, numbers)
    assert without_number(before) == without_number(after), '公開確率・状態が変化した'
    assert pickle.dumps(rows) == original, '保存評価行が変化した'
    first, available = {}, {}
    for row in after:
        first.setdefault(row['latest_game'], row)
        if row['p1'] is not None:
            available.setdefault(row['latest_game'], row)
    scored = [b for b in limits if b['scored']]
    extremes = [b['game'] for b in scored if first.get(b['game'], {}).get('p1') in (0., 1.)]
    result = dict(source=str(source), evaluation_rows=len(rows), publications=len(after),
        evaluation_sha256=hashlib.sha256(original).hexdigest(),
        publication_without_number_sha256=hashlib.sha256(without_number(after)).hexdigest(),
        evaluation_bytes_equal=True, publication_bytes_equal_except_number=True,
        internal_increments=max(r['game'] for r in rows),
        public_increments=numbers.number(max(r['game'] for r in rows)),
        removed_increments=len(numbers.recoveries), scored_boundaries=len(scored),
        unknown_boundaries=sum(not b['known'] for b in limits),
        scored_first_publication_extremes=extremes, boundaries=limits,
        scored_first_available_extremes=[b['game'] for b in scored
            if available.get(b['game'], {}).get('p1') in (0., 1.)],
        scope='保存評価・入力通知の公開再投入。画像再認識・モデル再推論は含まない。')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'boundaries'}, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.source, args.out)


if __name__ == '__main__':
    main()
