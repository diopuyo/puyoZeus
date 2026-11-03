"""同じ保存観測で本線と試験OFFの台帳・B3公開DTO・設定をバイト比較する片側ランナー。"""
from __future__ import annotations
import argparse
from dataclasses import asdict, is_dataclass
import gzip
from io import BytesIO
import json
from pathlib import Path
import sys

OUTPUT_BUFFER_BYTES = 1024*1024


def serial(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--packets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    from src.b3_score_ledger import ScoreLedger
    from src.phase_j.live_b3 import LiveB3
    from src.b3_live_display import B3Display
    from src.phase_j import live_publish
    from src.phase_j.launcher import parse_user_config, build_pipeline_config
    live_publish.utc_now = lambda: '2026-10-19T00:00:00+00:00'
    assets = {k: 'test' for k in ('app_build_id', 'recognition_model_hash', 'recognition_config_hash',
                                 'prediction_model_hash', 'calibration_hash')}
    initial = live_publish.initial_snapshot(assets)
    payload = initial.to_mapping()
    payload['identity']['session_id'] = 'fixed-off-verification'
    initial = live_publish.OverlaySnapshot.from_mapping(payload)
    owner = LiveB3.__new__(LiveB3)
    owner.display, owner.result_frame = B3Display(), -1
    ledger, count = ScoreLedger(), 0
    with args.out.open('x', encoding='utf-8', newline='\n', buffering=OUTPUT_BUFFER_BYTES) as output:
        config = build_pipeline_config(parse_user_config(dict(device_name='OBS Virtual Camera'), Path('D:/fixed')))
        output.write(json.dumps(config, sort_keys=True)+'\n')
        with gzip.open(BytesIO(args.packets.read_bytes()), 'rt', encoding='utf-8') as stream:
            for line in stream:
                packet = json.loads(line)
                frame = frame_from_packet(packet)
                snap = ledger.update(frame)
                owner.display.boundary(packet['game'])
                # 同一入力の公開勝率列を固定し、端点・保持・切替を両実装で踏む。
                probability = (count % 101)/100
                if count % 5 == 0:
                    owner.display.prediction(probability, int(count % 2), packet['t'])
                owner.display.update(packet['t'], active=frame['match_active'])
                row = dict(frame=count, t_sec=packet['t'], game=packet['game'], captured_at=1.,
                    recognized_at=1.1, evaluated_at=1.2, queue_depth=0, capture_gap=False, digest='fixed')
                owner.decorate(row)
                dto = live_publish.result_snapshot(initial, row, 1.5, count+1, None).to_mapping()
                output.write(json.dumps(dict(ledger=snap, row=row, dto=dto), default=serial,
                                        sort_keys=True, separators=(',', ':'))+'\n')
                count += 1
    print('OFF records', count, flush=True)


def frame_from_packet(packet: dict) -> dict:
    sides = []
    for index, source in enumerate(packet['raw']):
        board = source.get('confirmed_board')
        sides.append(dict(side=('1P', '2P')[index], state=source['state']['state'], score=source.get('score'),
            confirmed_board=board['board']['array'] if board else None,
            next_slide_motion=source.get('next_slide_motion', False)))
    return dict(t_sec=packet['t'], frame=packet['id'], game_idx=packet['game'],
                match_active=not all(s['state'] == 'MENU' for s in sides), sides=sides)


if __name__ == '__main__':
    main()
