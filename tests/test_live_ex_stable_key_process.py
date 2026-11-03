"""資産を明示した実 spawn 試験。実行記録は指定ジョブ内だけに書く。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import struct
import time
import unittest
from typing import Any

from src.phase_j.live_ex_async import AsyncEx
from src.phase_j.live_ex_capture import CaptureEx

PACKET_COUNT = 1381
TIMEOUT_SEC = 120.
POLL_SEC = .02
FORMULA_INTERVAL = 30
FORMULA_SCORE = 40
WARMUP_COUNT = 300
FEED_PAUSE_SEC = .002
POLICY_SHA = '0f034db5e3fd960fba5640d8484e0d57cc242c3710601b58481a4c7590924b58'
SEED_SHA = '5aec04e78d2e3128b4706978969c669b5cb27ee79f7e22ea9f7d40f219d4f851'


def packets(path: Path, with_forecast: bool) -> list[dict]:
    """保存観測は保持し、別ケースでだけ制御した式と盤を追加する。"""
    rows = copy.deepcopy(json.loads(path.read_text(encoding='utf-8'))[:PACKET_COUNT])
    board = [[0]*6 for _ in range(13)]
    board[-1][:4] = [1]*4
    for index, packet in enumerate(rows):
        for side in packet['raw']:
            if with_forecast:
                side['raw_cnn_board'] = board
                side['formula'] = dict(valid=index % FORMULA_INTERVAL == 1, product=FORMULA_SCORE)
    capture = CaptureEx()
    return [capture.capture(None, SimpleNamespace(media_sec=p['t']), dict(game_idx=p['game'], _v49f=p))
            for p in rows]


def settings(config: dict, stable: bool, first: bool) -> dict:
    for name, digest in [('policy', POLICY_SHA), ('seed_table', SEED_SHA)]:
        assert hashlib.sha256(Path(config[name]).read_bytes()).hexdigest() == digest, name
    return dict(policy=config['policy'], seed_table=config['seed_table'], native_dir=config['native_dir'],
                work_limit=450, tape_next=True, ex_speed=True, retr=True, xc=True,
                retr_h_forecast=True, stable_key=stable, ex_fire_first=first)


def feed(owner: AsyncEx, rows: list[dict], events: list[dict]) -> None:
    """起動を待ってから観測を流し、2本目にも計算を割り当てる機会を作る。"""
    for index, packet in enumerate(rows):
        owner.submit(packet, 0, time.perf_counter())
        if index + 1 == WARMUP_COUNT:
            deadline = time.perf_counter()+TIMEOUT_SEC
            while time.perf_counter() < deadline:
                events.extend(owner.poll())
                if owner.error or any(e['kind'] == 'result' and e['value'].get('serial') == WARMUP_COUNT
                                      for e in events):
                    break
                time.sleep(POLL_SEC)
        elif index >= WARMUP_COUNT:
            events.extend(owner.poll())
            if owner.error:
                break
            time.sleep(FEED_PAUSE_SEC)


def run_case(config: dict, stable: bool, first: bool, forecast: bool) -> dict:
    rows = packets(Path(config['packets']), forecast)
    owner = AsyncEx(settings(config, stable, first))
    events = []
    try:
        feed(owner, rows, events)
        deadline = time.perf_counter()+TIMEOUT_SEC
        while time.perf_counter() < deadline:
            events.extend(owner.poll())
            if owner.error or any(e['kind'] == 'result' and e['value'].get('serial') == len(rows) for e in events):
                break
            time.sleep(POLL_SEC)
    finally:
        owner.close()
        events = owner.take_events()
    keys = {e['serial']: e for e in events if e['kind'] == 'key'}
    results = [e['value'] for e in events if e['kind'] == 'result']
    mismatches = [v.get('serial') for v in results if 'serial' in v and
                  (v['key'] != keys[v['serial']]['key'] or v['fight'] != keys[v['serial']]['fight'])]
    errors = [e for e in events if e['kind'] == 'error']
    report = dict(stable=stable, first=first, forecast=forecast, packets=len(rows), keys=len(keys),
                  results=len(results), fight_results=sum(v.get('fight', False) for v in results),
                  workers=sorted({v.get('compute_worker', 0) for v in results}), mismatches=mismatches,
                  errors=errors, owner_error=owner.error, alive=owner.process.is_alive(),
                  last_result=any(v.get('serial') == len(rows) for v in results),
                  counts=[e['counts'] for e in events if e['kind'] == 'counts'])
    destination = Path(config['out'])/f'process_s{int(stable)}_f{int(first)}_forecast{int(forecast)}.json'
    destination.write_text(json.dumps(dict(report=report, events=events), default=str), encoding='utf-8')
    assert not errors and not owner.error and not mismatches, report
    assert len(keys) == len(rows) and report['last_result'] and not report['alive'], report
    assert report['fight_results'] > 0, report
    assert all(not value.get('failure') for value in results), report
    if first:
        assert {v.get('compute_worker') for v in results if v.get('fight')} == {0, 1}, report
    if stable and forecast:
        forecast_id = round(rows[1]['t'], 6)
        assert any(forecast_id in k['key'][2] for k in keys.values()), report
    return report


def compare_values(config: dict, first: bool, forecast: bool) -> dict:
    """共通の要求 serial の値を比較し、到着時刻の列だけ除く。"""
    groups = []
    for stable in (False, True):
        path = Path(config['out'])/f'process_s{int(stable)}_f{int(first)}_forecast{int(forecast)}.json'
        events = json.loads(path.read_text(encoding='utf-8'))['events']
        groups.append({e['value']['serial']: e['value'] for e in events
                       if e['kind'] == 'result' and e['value'].get('fight')})
    common = sorted(groups[0].keys() & groups[1].keys())
    assert common, '比較できる撃ち合い要求がない'
    compared, differences = 0, []
    for serial in common:
        old, new = (g[serial]['extra'] for g in groups)
        assert old.keys() == new.keys()
        for name in old:
            if name.startswith('av_'):
                continue
            compared += 1
            if struct.pack('!d', float(old[name])) != struct.pack('!d', float(new[name])):
                differences.append((serial, name))
    report = dict(first=first, forecast=forecast, requests=len(common), values=compared, differences=differences)
    assert not differences, report
    return report


@unittest.skipUnless(os.environ.get('PUYO_STABLE_KEY_PROCESS_CONFIG'), '実プロセス試験用の資産指定が必要')
class StableKeyProcessTests(unittest.TestCase):
    def test_all_flag_combinations(self) -> None:
        config = json.loads(Path(os.environ['PUYO_STABLE_KEY_PROCESS_CONFIG']).read_text(encoding='utf-8-sig'))
        reports = []
        for stable in (False, True):
            for first in (False, True):
                for forecast in (False, True):
                    with self.subTest(stable=stable, first=first, forecast=forecast):
                        reports.append(run_case(config, stable, first, forecast))
        (Path(config['out'])/'process_summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
        comparisons = [compare_values(config, first, forecast) for first in (False, True) for forecast in (False, True)]
        (Path(config['out'])/'values_summary.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
