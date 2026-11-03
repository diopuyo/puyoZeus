"""zenchiの再生入力を同期・非同期へ流し、公開DTOと保存評価行を完全照合する。"""
from __future__ import annotations

import argparse
from io import StringIO
import json
from pathlib import Path
from time import perf_counter, sleep
from types import SimpleNamespace

import pandas as pd

from src.b3_async import AsyncB3
from src.b3_live_display import B3Display
from src.b3_realtime import RealtimeRows
from src.b3_realtime_replay import replay_inputs
from src.b3_rows import _inside_match
from src.b3_source import resolve_attempt
from src.b3_timing import TimingTable
from src.b3_train_model import load_model
from src.phase_j.live_b3 import LiveB3
from src.phase_j.live_publish import initial_snapshot, result_snapshot
from src.phase_j.validator import validate_snapshot

VIDEO = 'b3_zenchi_s1'
MAX_FRAME = 36000
MAX_SECONDS = 600
DRAIN_TIMEOUT = 1800
POLL_SECONDS = .01
DIRECT = ('sl_pending', 'sl_generated', 'sl_offset', 'chain_active', 'chain_n_steps',
          'chain_pred_l', 'chain_rem_stages', 'chain_rem_sec', 'k1_ojama',
          'cells_color', 'cells_ojama', 'mk_sl', 'mk_chain', 'mk_k')
REVERSED = ('ta_H_med', 'ta_q_med', 'ta_m_sl_med', 'ta_bound', 'mk_ta')
ASSETS = {key: 'replay' for key in ('app_build_id', 'recognition_model_hash',
          'recognition_config_hash', 'prediction_model_hash', 'calibration_hash')}


def compare(contents: dict, expected: dict) -> None:
    """公開変換を期待値生成に再利用せず、列値・欠測・側を検査する。"""
    assert contents['frame'] == expected['frame_idx']
    assert contents['t_sec'] == expected['t_sec']
    assert contents['model'] == ('exchange' if expected['mode_exchange'] else 'static')
    assert contents['sl_rate'] == nullable(expected['sl_rate'])
    for side, attacker in (('p1', 'p2'), ('p2', 'p1')):
        actual = contents[side]
        for key in DIRECT + REVERSED:
            column = key+'_'+(attacker if key in REVERSED else side)
            assert actual[key] == nullable(expected[column]), (contents['frame'], column, actual[key], expected[column])
        sent = expected['sl_generated_'+side] - expected['sl_offset_'+side]
        assert actual['sl_sent'] == nullable(sent)
        assert actual['unpredictable'] == any(expected[column] == 6 for column in
            ('mk_chain_'+side, 'mk_k_'+side, 'mk_ta_'+attacker))


def nullable(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def live_wrapper(rows: RealtimeRows, model: object) -> LiveB3:
    live = LiveB3.__new__(LiveB3)
    live.rows, live.model, live.display = rows, model, B3Display()
    live.generation, live.result_frame = 1, -1
    live.audit = StringIO()
    return live


def publish(live: LiveB3, reference: dict, initial: object) -> dict:
    """製品のdecorate→DTO経路を通し、NaNも契約違反も許さない。"""
    stamp, frame = reference['t_sec'], int(reference['frame_idx'])
    live.display.update(stamp)
    row = dict(frame=frame, t_sec=stamp, game=int(reference['game_idx']),
        captured_at=stamp, recognized_at=stamp, evaluated_at=stamp,
        queue_depth=0, capture_gap=False, digest='sha256:replay')
    live.decorate(row)
    dto = result_snapshot(initial, row, stamp, frame, None)
    payload = dto.to_mapping()
    # スキーマは各公開値に適用。意味制約はモデルに依存しない。
    report = validate_snapshot(dto)
    assert report.is_valid, report
    json.dumps(payload, allow_nan=False)
    compare(payload['evaluations']['b3_contents'], reference)
    live.audit.seek(0)
    live.audit.truncate()
    return payload


def accept_results(engine: AsyncB3, live: LiveB3, references: dict,
                   initial: object, seen: set[int]) -> bool:
    for result in engine.poll():
        if 'done' in result:
            return True
        assert 'fatal' not in result and result['error'] is None, result
        expected = references[result['frame']]
        live.display.boundary(int(expected['game_idx']))
        live._accept_result(result, dict(game_idx=int(expected['game_idx']),
            frame=result['frame'], t_sec=result['t_sec']))
        publish(live, expected, initial)
        assert result['frame'] not in seen
        seen.add(result['frame'])
    return False


def run(args: argparse.Namespace) -> dict:
    reference = pd.read_parquet(args.rows / (VIDEO+'.parquet'))
    reference = reference[(reference.orientation == 0) & (reference.frame_idx <= MAX_FRAME)
                          & (reference.t_sec <= MAX_SECONDS)]
    references = {int(r['frame_idx']): r for r in reference.to_dict('records')}
    manifest = json.loads((args.rows / (VIDEO+'.manifest.json')).read_text())
    table = TimingTable.from_json(args.table.read_text())
    rows = RealtimeRows(table, VIDEO, measure_timing=False)
    sync = live_wrapper(rows, load_model(args.model))
    asynchronous = live_wrapper(rows, None)
    initial, seen_sync, seen_async = initial_snapshot(ASSETS), set(), set()
    engine = AsyncB3(args.model, table, VIDEO, drop=False, measure=False)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        with (args.out/'snapshots.jsonl').open('w', encoding='utf-8') as output:
            for frame, events, queues in replay_inputs(resolve_attempt(VIDEO, args.root).record, MAX_FRAME):
                emit = _inside_match(frame['t_sec'], manifest['winner_mapping'].get(str(frame['game_idx']), {}))
                engine.submit(frame, events, queues, 1, emit)
                row = rows.update(frame, events, queues, emit)
                if row is not None:
                    sync.display.boundary(frame['game_idx'])
                    sync._predict(row, SimpleNamespace(frame=frame['frame'], t_sec=frame['t_sec'], captured_at=0.))
                    payload = publish(sync, references[frame['frame']], initial)
                    output.write(json.dumps(payload, ensure_ascii=False)+'\n')
                    seen_sync.add(frame['frame'])
                accept_results(engine, asynchronous, references, initial, seen_async)
        engine.finish()
        deadline = perf_counter()+DRAIN_TIMEOUT
        while not accept_results(engine, asynchronous, references, initial, seen_async):
            if perf_counter() > deadline:
                raise TimeoutError('非同期再生の終了待ち')
            sleep(POLL_SECONDS)
    finally:
        engine.close()
    assert seen_sync == seen_async == set(references)
    return dict(video=VIDEO, seconds=MAX_SECONDS, reference='eval_rows_v7',
        sync_rows=len(seen_sync), async_rows=len(seen_async), missing=0, extra=0,
        differences=0, schema_valid=True, reference_model=str(args.model))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'rows', 'model', 'table', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    (args.out/'parity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
