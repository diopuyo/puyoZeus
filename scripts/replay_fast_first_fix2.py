"""保存した評価・入力通知を、修正後の境界生成と公開経路へ逐次再投入する。

画像・HSV生サンプル・較正保留中の未生成評価は再構成しない。
較正は記録された再収集状態をDeviceSessionへ入力する契約再生である。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.replay_forced_boundary_20261011 import load_pickle_rows, run_mini_generate
from src.phase_j.live_calibration import ColorWarmup
from src.phase_j.live_device_session import DeviceSession
from src.phase_j.live_freshness import evaluation_stale
from src.phase_j.live_publish import LivePublisher, PUBLISH_PERIOD_SEC
from src.phase_j.live_boundary import RECOVERY_BOUNDARY_WINDOW_SEC

ASSETS = {key: 'replay' for key in ('app_build_id', 'recognition_model_hash',
    'recognition_config_hash', 'prediction_model_hash', 'calibration_hash')}
PROBABILITY_KEYS = ('probability', 'raw_probability', 'advantage', 'source', 'display_layers')


def recorded_updates(directory: Path) -> list[dict]:
    """重い評価は再実行せず、保存された全状態通知の得点と時刻を読む。"""
    rows = []
    with gzip.open(directory / 'output/inputs.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            data = json.loads(line)
            if data.get('kind') != 'update':
                continue
            args = data['args']['tuple']
            result = args[0]['namespace']
            rows.append(dict(t_sec=args[3], game=args[4],
                score1=result['p1']['namespace']['score'], score2=result['p2']['namespace']['score']))
    return rows


def boundary_replay(directory: Path, evaluations: list[dict], inputs: list[dict]) -> tuple[dict, dict]:
    """入力世代変更を、直後の認識通知で一回だけ消費する。"""
    updates = recorded_updates(directory)
    recognition = load_pickle_rows(directory / 'output/spool/recognition_rows.pickle')
    by_time = {row['t_sec']: row for row in recognition}
    epochs, previous = [], None
    for event in inputs:
        epoch = event.get('epoch')
        if epoch is not None and epoch != previous:
            if epoch > 0:
                epochs.append(event['at'])
            previous = epoch
    events, used, missing = [], 0, 0
    for row in updates:
        notice = by_time.get(row['t_sec'])
        if notice is None:
            missing += 1
            continue
        changed = False
        while used < len(epochs) and epochs[used] <= notice['recognized_at']:
            changed, used = True, used+1
        events.append((row['t_sec'], (row['score1'], row['score2']), changed))
    before = run_mini_generate(events, True)
    after = run_mini_generate(events, True, coalesce=True)
    legacy = run_mini_generate(events, False)
    times = [event[0] for event in events]
    mapping = {stamp: game for stamp, (_, game, _) in zip(times, after)}
    report = dict(state_notifications=len(updates), matched=len(events), missing=missing,
        input_epochs=len(epochs), consumed_epochs=used,
        recorded_last_game=max(row['game'] for row in updates),
        isolation_before=before[-1][1], coalesced_after=after[-1][1], legacy=legacy[-1][1],
        baseline_mismatch_isolated=sum(r['game'] != trace[1] for r, trace in zip(updates, before)),
        baseline_mismatch_legacy=sum(r['game'] != trace[1] for r, trace in zip(updates, legacy)),
        close_pairs_before=close_pairs(times, before), close_pairs_after=close_pairs(times, after),
        reset_body_executions_before=sum(a[2] != b[2] for a, b in zip(before, before[1:])),
        reset_body_executions_after=sum(a[2] != b[2] for a, b in zip(after, after[1:])))
    return report, mapping


def close_pairs(times: list[float], trace: list[tuple]) -> int:
    accepted, previous = [], 0
    for stamp, (_, game, _) in zip(times, trace):
        if game != previous:
            accepted.append(stamp)
        previous = game
    return sum(b-a < RECOVERY_BOUNDARY_WINDOW_SEC for a, b in zip(accepted, accepted[1:]))


def delivered_inputs(directory: Path, inputs: list[dict]) -> list[tuple]:
    """認識側の発生時計を到着時計と誤認しない。公開spoolで保留の配送遅延を復元する。"""
    publications = load_pickle_rows(directory / 'output/spool/publications.pickle')
    delivered = {row['evaluated_at']: row['published_at'] for row in publications if row['hold']}
    events, following, epoch = [], None, None
    for event in reversed(inputs):
        if event.get('epoch') != epoch or event['phase'] == 'ready':
            following = None
        epoch = event.get('epoch')
        stamp = delivered.get(event['at'])
        if stamp is None:
            # 同じ保留内でlatest-winsにより省略された中間状態は、次の公開状態と一緒に配送する。
            stamp = max(event['at'], following or event['at'])
        events.append((stamp, 'input', event))
        following = stamp
    return list(reversed(events))


class CalibrationReplay:
    """同一世代でready後に起きた再収集だけを、実DeviceSessionへ通す。"""

    def __init__(self, enabled: bool) -> None:
        self.enabled, self.epoch = enabled, None
        self.seen_ready = False
        self.recollecting = False
        self.rechecks = self.changed_events = 0

    def event(self, event: dict) -> dict:
        epoch = event.get('epoch')
        if epoch != self.epoch:
            self.epoch, self.seen_ready, self.recollecting = epoch, False, False
        if event['phase'] == 'ready':
            self.seen_ready, self.recollecting = True, False
        if event['phase'] == 'calibrating' and self.seen_ready:
            if not self.recollecting:
                self.rechecks += 1
            self.recollecting = True
        if not self.enabled or not self.recollecting:
            return event
        if event['phase'] not in ('calibrating', 'no_puyo_screen'):
            return event
        output = self.recheck_status(event)
        self.changed_events += int(output['phase'] != event['phase'])
        return output

    def recheck_status(self, event: dict) -> dict:
        """保存通知で表された照合失敗を注入し、実装が返す公開状態を取得する。"""
        session = DeviceSession.__new__(DeviceSession)
        warmup = ColorWarmup.__new__(ColorWarmup)
        warmup.device = SimpleNamespace(fast_first_judgment=True,
                                        calibration_path=Path(event['device_id']+'.json'))
        warmup.ready, warmup.progress, warmup.cached = True, 100, True
        warmup.verification_only, warmup.mode = True, 'verification'
        session.warmup, session._needs_recheck = warmup, True
        session._recheck = SimpleNamespace(observe=Mock(), changed_profile=True, ready=False)
        session.pipe = Mock()
        assert not session.recheck_calibration(None, None)
        return dict(event, **warmup.status())


def publish_events(directory: Path, evaluations: list[dict], inputs: list[dict],
                   mapping: dict, enabled: bool) -> dict:
    """公開writerを仮想単調時計で駆動し、HTTPと同じlatestを読む。"""
    fast = json.loads((directory / 'puyo_live.json').read_text(encoding='utf-8-sig')).get(
        'fast_first_judgment', False)
    calibration = CalibrationReplay(enabled and fast)
    publisher = LivePublisher(ASSETS, '127.0.0.1', 0, completion_driven=True,
                              freshness_gate=enabled)
    events = [(r['evaluated_at'], 'evaluation', r) for r in evaluations]
    events += delivered_inputs(directory, inputs)
    events.sort(key=lambda event: event[0])
    records, tick, previous, expired, hashes = [], events[0][0], None, None, []
    for stamp, kind, original in events:
        while tick < stamp:
            current = publisher.latest
            if enabled and current is not expired and evaluation_stale(current, tick):
                record_publish(publisher, current, tick, records)
                expired = current
            tick += PUBLISH_PERIOD_SEC
        with patch('src.phase_j.live_publish.time.perf_counter', return_value=stamp):
            if kind == 'input':
                publisher.input_pending(calibration.event(original))
            else:
                row = dict(original)
                if enabled:
                    row['game'] = mapping.get(row['t_sec'], row['game'])
                hashes.append(pickle.dumps({key: row.get(key) for key in PROBABILITY_KEYS}))
                publisher.offer(row)
            current = publisher.latest
            if publisher.ready.is_set() and current is not None and current is not previous:
                publisher.ready.clear()
                record_publish(publisher, current, stamp, records)
                previous, expired = current, None
    return dict(evaluations=len(evaluations), input_events=len(inputs), publications=len(records),
        rechecks=calibration.rechecks, adjusted_input_events=calibration.changed_events,
        evaluation_hash=hashlib.sha256(b''.join(hashes)).hexdigest(),
        summary=publication_summary(records), records=records)


def record_publish(publisher: LivePublisher, row: dict, stamp: float, records: list) -> None:
    with patch('src.phase_j.live_publish.time.perf_counter', return_value=stamp):
        publisher._publish(row, 0, stamp if row.get('hold') else None)
    current = publisher.state.latest()
    display = current['display']
    value = current['evaluations']['practical']
    records.append(dict(at=stamp, frame=row['frame'], game=row['game'],
        phase=display.get('match_phase'), input=display.get('input_status'), status=display['status'],
        reason=display.get('primary_hold_reason'), p1=value['p1_win_probability'],
        message=display.get('message'), age=stamp-row['evaluated_at']))


def publication_summary(records: list[dict]) -> dict:
    drops, games, live_games, active, calibrating = 0, set(), set(), False, False
    for record in records:
        if record['status'] in ('live', 'physical_prediction') and record['phase'] == 'playing':
            live_games.add(record['game'])
            active = True
        if record['input'] == 'calibrating' and not calibrating and active:
            drops += 1
            games.add(record['game'])
        calibrating = record['input'] == 'calibrating'
        if record['phase'] == 'between_matches' or record['input'] in ('verifying', 'no_puyo_screen'):
            active = False
    return dict(calibration_drops=drops, games_with_drop=len(games), live_games=len(live_games),
        freshness_holds=sum(r['message'] == '評価更新待ち（更新が遅れています）' for r in records))


def longest_stale_interval(before: list[dict], after: list[dict]) -> dict:
    """評価値を出した公開から次の状態変更までの最長区間を抽出する。"""
    candidates = [(b['at']-a['at'], a, b) for a, b in zip(before, before[1:])
                  if a['status'] == 'live' and a['p1'] not in (None, 0., 1.)]
    duration, start, end = max(candidates, key=lambda item: item[0])
    holds = [r for r in after if start['at'] <= r['at'] < end['at']
             and r['message'] == '評価更新待ち（更新が遅れています）']
    return dict(duration_sec=duration, frame=start['frame'], probability=start['p1'],
        start_at=start['at'], end_at=end['at'], holds=len(holds),
        hold_after_sec=holds[0]['at']-start['at'] if holds else None,
        hold_duration_sec=end['at']-holds[0]['at'] if holds else 0.)


def run(directory: Path, output: Path | None = None) -> dict:
    spool = directory / 'output/spool'
    evaluations = load_pickle_rows(spool / 'evaluations.pickle')
    inputs = load_pickle_rows(spool / 'input_events.pickle')
    original = pickle.dumps(evaluations)
    boundaries, mapping = boundary_replay(directory, evaluations, inputs)
    before = publish_events(directory, evaluations, inputs, mapping, False)
    after = publish_events(directory, evaluations, inputs, mapping, True)
    longest = longest_stale_interval(before['records'], after['records'])
    assert before['evaluation_hash'] == after['evaluation_hash']
    assert pickle.dumps(evaluations) == original
    if output is not None:
        path = output.with_name(output.stem+'_'+directory.name+'_snapshots.jsonl')
        with path.open('w', encoding='utf-8') as stream:
            for variant, report in (('before', before), ('after', after)):
                for record in report['records']:
                    stream.write(json.dumps(dict(record, variant=variant), ensure_ascii=False)+'\n')
    before.pop('records')
    after.pop('records')
    return dict(run=directory.name, boundary=boundaries, before=before, after=after,
        longest_live_gap=longest, evaluation_rows_unchanged=True,
        scope='保存状態通知・評価・入力通知の契約再生。画像/HSV生サンプル、保留中の未生成評価は対象外。')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = [run(directory, args.output) for directory in args.directories]
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
