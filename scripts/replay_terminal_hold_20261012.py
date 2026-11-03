"""保存評価を公開側だけへ再投入し、決着秒数と次試合への持越しを照合する。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
from typing import Any
from unittest.mock import patch

from src.phase_j.live_freshness import evaluation_stale
from src.phase_j.live_publish import LivePublisher, PUBLISH_PERIOD_SEC
from src.phase_j.live_terminal_hold import TERMINAL_DISPLAY_HOLD_SEC

ASSETS = {key: 'replay' for key in ('app_build_id', 'recognition_model_hash',
    'recognition_config_hash', 'prediction_model_hash', 'calibration_hash')}
FIRST_SECONDS = 2.0
# 最終行で確定した場合も、保持期限の解除まで積分する。
END_MARGIN_SEC = TERMINAL_DISPLAY_HOLD_SEC + PUBLISH_PERIOD_SEC


def load_rows(path: Path) -> list[dict[str, Any]]:
    """読取専用。ファイル終端以外の欠損はエラーにする。"""
    rows = []
    with path.open('rb') as stream:
        while True:
            position = stream.tell()
            if not stream.read(1):
                return rows
            stream.seek(position)
            rows.append(pickle.load(stream))


def input_deliveries(inputs: list[dict], publications: list[dict]) -> list[tuple]:
    """既存公開時計で入力保留のIPC遅延を復元する（fix2再投入台と同じ規則）。"""
    delivered = {r['evaluated_at']: r['published_at'] for r in publications if r['hold']}
    events, following, epoch = [], None, None
    for row in reversed(inputs):
        if row.get('epoch') != epoch or row['phase'] == 'ready':
            following = None
        epoch = row.get('epoch')
        stamp = delivered.get(row['at'], max(row['at'], following or row['at']))
        events.append((stamp, 'input', row))
        following = stamp
    return list(reversed(events))


def record(owner: LivePublisher, stamp: float, records: list[dict]) -> None:
    """ネットワークを起動せず、実writer・SnapshotHub・DTO検証を通す。"""
    row = owner.latest
    owner._publish(row, 0, stamp if row.get('hold') else None)
    data = owner.state.latest()
    lane, display = data['evaluations']['practical'], data['display']
    records.append(dict(at=stamp, game=int(data['identity']['match_id']),
        frame=data['timing']['source_available_frame'], p1=lane['p1_win_probability'],
        source=lane.get('source'), status=display['status'], phase=display.get('match_phase'),
        message=display.get('message'), held=owner.terminal_visible,
        latest_game=row['game'], deadline=owner.terminal.deadline if owner.terminal else None))


def replay(events: list[tuple], enabled: bool) -> list[dict]:
    """完了駆動と0.5秒鮮度監視を仮想時計で実行。決着期限は専用タイマーで進める。"""
    owner = LivePublisher(ASSETS, '127.0.0.1', 0, completion_driven=True,
                          freshness_gate=True, terminal_display_hold=enabled)
    records, previous, expired = [], None, None
    clock, tick = [events[0][0]], events[0][0]
    with patch('src.phase_j.live_publish.time.perf_counter', side_effect=lambda: clock[0]):
        for stamp, kind, original in events + [(events[-1][0]+END_MARGIN_SEC, 'end', {})]:
            while True:
                deadline = owner.terminal.deadline if owner.terminal_visible else None
                wake = min(tick, deadline) if deadline is not None else tick
                if wake >= stamp:
                    break
                clock[0] = wake
                due = deadline is not None and wake >= deadline
                if owner.latest is not None and (due or (
                        owner.latest is not expired and evaluation_stale(owner.latest, wake))):
                    record(owner, wake, records)
                    expired = owner.latest
                if wake == tick:
                    tick += PUBLISH_PERIOD_SEC
            clock[0] = stamp
            if kind == 'input':
                owner.input_pending(original)
            elif kind == 'evaluation':
                owner.offer(original)
            elif kind == 'end' and owner.latest is not None:
                record(owner, stamp, records)
            if owner.ready.is_set() and owner.latest is not None and owner.latest is not previous:
                owner.ready.clear()
                record(owner, stamp, records)
                previous, expired = owner.latest, None
    return records


def durations(records: list[dict]) -> dict[int, float]:
    """公開から次の公開までを積分し、死亡source以外の100/0は別物として除く。"""
    seconds: dict[int, float] = defaultdict(float)
    for row, following in zip(records, records[1:]):
        if row['source'] == 'confirmed_death' and row['p1'] in (0., 1.):
            seconds[row['game']] += following['at']-row['at']
    return seconds


def safety(records: list[dict], starts: dict[int, float], first_seconds: float = FIRST_SECONDS) -> dict:
    """先頭公開と指定秒数を別々に数え、値の持越しを試合番号でも照合する。"""
    first, early, mismatch = {}, [], []
    for row in records:
        if row['at'] < starts.get(row['game'], float('inf')):
            continue
        first.setdefault(row['game'], row)
        if row['p1'] in (0., 1.) and row['at'] < starts[row['game']]+first_seconds:
            early.append(row)
        if row['held'] and row['game'] != row['latest_game']:
            mismatch.append(row)
    # 初期ゲームを除き、「次試合」の最初の公開を対象にする。
    games = sorted(first)[1:]
    first_bad = [first[game] for game in games if first[game]['p1'] in (0., 1.)]
    early = [row for row in early if row['game'] in games]
    return dict(next_games=len(games), first_publication_extremes=len(first_bad),
        first_seconds=first_seconds, early_extremes=len(early),
        early_extreme_games=sorted({row['game'] for row in early}),
        held_wrong_game=len(mismatch), first_bad=first_bad, early_bad=early)


def per_game(evaluations: list[dict], before: list[dict], after: list[dict]) -> list[dict]:
    """認識側の試合番号をそのまま使用する。実ゲームと強制境界を勝手に統合しない。"""
    old, new = durations(before), durations(after)
    grouped: dict[int, list] = defaultdict(list)
    for row in evaluations:
        grouped[row['game']].append(row)
    return [dict(game=game, evaluations=len(rows),
        confirmed_rows=sum(row['source'] == 'confirmed_death' for row in rows),
        before_sec=old[game], after_sec=new[game]) for game, rows in sorted(grouped.items())]


def replay_baseline(events: list[tuple], baseline: Path | None) -> list[dict]:
    """旧保持実装の指定時はON同士で比較し、未指定なら従来のOFFを使う。"""
    if baseline is None:
        return replay(events, False)
    spec = importlib.util.spec_from_file_location('terminal_hold_baseline', baseline)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with patch('src.phase_j.live_publish.TerminalDisplayHold', module.TerminalDisplayHold):
        return replay(events, True)


def run(directory: Path, output: Path, baseline: Path | None = None,
        first_seconds: float = FIRST_SECONDS) -> dict:
    spool = directory/'spool'
    evaluations = load_rows(spool/'evaluations.pickle')
    inputs = load_rows(spool/'input_events.pickle')
    publications = load_rows(spool/'publications.pickle')
    original = hashlib.sha256(pickle.dumps(evaluations)).hexdigest()
    events = [(row['evaluated_at'], 'evaluation', row) for row in evaluations]
    events += input_deliveries(inputs, publications)
    events.sort(key=lambda event: event[0])
    before, after = replay_baseline(events, baseline), replay(events, True)
    digest = hashlib.sha256(pickle.dumps(evaluations)).hexdigest()
    assert original == digest, '評価行が変更された'
    starts: dict[int, float] = {}
    for row in evaluations:
        starts.setdefault(row['game'], row['evaluated_at'])
    with output.with_suffix('.jsonl').open('w', encoding='utf-8') as stream:
        for variant, records in (('before', before), ('after', after)):
            for row in records:
                stream.write(json.dumps(dict(row, variant=variant), ensure_ascii=False)+'\n')
    games = per_game(evaluations, before, after)
    result = dict(directory=str(directory), evaluations=len(evaluations), input_events=len(inputs),
        games=len(games), confirmed_games=sum(row['confirmed_rows'] > 0 for row in games),
        before_publications=len(before), after_publications=len(after), per_game=games,
        before_safety=safety(before, starts, first_seconds), after_safety=safety(after, starts, first_seconds),
        baseline=str(baseline) if baseline else 'OFF',
        evaluations_sha256_before=original, evaluations_sha256_after=digest,
        evaluation_rows_unchanged=True,
        scope='保存評価・入力通知の公開再投入。画像再認識・評価再計算なし。境界/較正は元記録のまま。')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-terminal-hold', type=Path)
    parser.add_argument('--first-seconds', type=float, default=FIRST_SECONDS)
    args = parser.parse_args()
    result = run(args.directory, args.output, args.baseline_terminal_hold, args.first_seconds)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('per_game', 'before_safety', 'after_safety')}, ensure_ascii=False))
    print('次試合の先頭100/0:', result['after_safety']['first_publication_extremes'],
          '/', result['after_safety']['next_games'])


if __name__ == '__main__':
    main()
