"""A28保存履歴を元consumeと同じ凍結物理/私有台帳へ渡す。動画・モデル生成なし。"""
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
ASYNC = ROOT.parent / 'g2_async_projected_evaluation_2026-09-13_v1'
RUN = ROOT.parent / 'video38_early_origin_candidate_v28'
AVAILABLE = 35160
sys.path.insert(0, str(ASYNC))
import probe_cold_prediction_dependency_v3 as C


def replay() -> dict:
    from src.board import Board
    from journal_origin_capture_candidate import OriginCapture
    import settled_notice
    physics = sys.modules['_g2_probability_scope_frozen_physics']
    ledger = sys.modules['_async_live_prediction_ledger']
    simulator = physics.ChainSimulator(exclude_hidden_row_from_pop=True)
    capture = OriginCapture(None, None, None, ledger, simulator, Board.from_dict,
                            settled_notice.is_settled, io.StringIO())
    path = RUN / 'EARLY_ORIGIN_HISTORY.jsonl'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    count, result = 0, {}
    try:
        with path.open() as stream:
            for line in stream:
                for item in json.loads(line)['rows']:
                    row = item['row']
                    try:
                        capture.consume(row, available_frame=AVAILABLE)
                    except BaseException as error:
                        events = []
                        for event in row['events']:
                            raw = event.get('active_origin')
                            if raw is None or raw['before_board'] is None: continue
                            simulation = simulator.simulate(Board.from_dict(raw['before_board']))
                            events.append(dict(stage=event['stage'], raw=raw,
                                simulated_chain_count=simulation.chain_count, steps=len(simulation.steps)))
                        result = dict(error=repr(error), frame=row['frame_idx'], side=row['side'],
                            source_row=row, events=events, preceding_rows=count)
                        return result | dict(source_sha256=digest, source_unchanged=
                            hashlib.sha256(path.read_bytes()).hexdigest() == digest)
                    count += 1
        holds = [json.loads(line) for line in capture.stream.getvalue().splitlines()]
        from dataclasses import asdict
        snapshots = {side: asdict(capture.ledger.snapshot(handle)) for side, handle in capture.handles.items()}
        return dict(no_failure=True, rows=count, source_sha256=digest, holds=holds, snapshots=snapshots,
                    source_unchanged=hashlib.sha256(path.read_bytes()).hexdigest() == digest)
    finally:
        capture.close()


def main() -> None:
    started, initial, paths = time.perf_counter(), Path.cwd(), list(sys.path)
    sys.path.insert(0, str(C.P.RUNTIME))
    import owned_adapter as A
    result: dict[str, Any] = {}
    try:
        with ExitStack() as stack:
            selected = A.configured(stack)
            adapter = A.configured.__globals__['A'].A.A.V4
            owner = sys.modules[adapter.OWNED_ALIAS]
            with selected.__globals__['S'].configured() as env:
                base = env['runtime'].M.base
                os.chdir(base.SNAPSHOT)
                base.load_collector()
                with ExitStack() as inner:
                    C.exercise(inner, owner.bootstrap())
                    for name in ('g2_belief_live_publication_2026-09-11_v1',
                                 'g2_second_origin_postrun_2026-09-13_v1'):
                        sys.path.insert(0, str(ROOT.parent / name))
                    result = replay()
        result.update(seconds=time.perf_counter() - started, video_updates=0,
            original_consume_used=True, frozen_simulator_used=True, model_created=False,
            configured_context_exited=True, quality_gate_clear=False)
    finally:
        os.chdir(initial)
        sys.path[:] = paths
    with (ROOT / 'REPLAY_CANDIDATE_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result.get(key) for key in
        ('error', 'frame', 'side', 'preceding_rows', 'seconds', 'no_failure')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
