"""発火根拠・予測値・独立確定盤面と原映像を対応付ける。"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from src.exchange_event_record import read_records
from src.scoring import calculate_chain_score
from src.chain import ChainSimulator
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from scripts.render_e20_audit_frames_20260928 import SOURCE
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e21')
TIMES = (2633.65, 2633.7833333, 2633.8166667, 2633.95, 2634.35, 2634.75)


def facts() -> None:
    """記録されたイベント値と本番シミュレーション結果を両方保存する。"""
    saved, first = None, None
    for row in read_records(Path('logs/e16/records/review.jsonl.gz')):
        if row['kind'] != 'update':
            continue
        args = row['args']
        side, stamp = args[0].p2, args[3]
        if stamp < 2633.8 and side.state.name == 'STABLE':
            saved = (stamp, side.confirmed_board)
        if 2633.8 <= stamp <= 2634.75 and side.chain_event and first is None:
            event = side.chain_event
            sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED).simulate(event.before_board)
            first = dict(observed_sec=stamp, trigger_sec=event.trigger_sec, mechanism=event.mechanism,
                predicted_chain_count=sim.chain_count, predicted_final_score=calculate_chain_score(sim).total_score,
                previous_confirmed_sec=saved[0], death_cell=int(saved[1]._grid[1, 2]),
                confirmed_board=saved[1]._grid.tolist(), before_board=event.before_board._grid.tolist(),
                formula_totals=args[5], displayed_scores=args[6], formula_visible=args[7])
    save_json(OUT/'FACTS.json', first)
    with Path('logs/review_zenchi_g41_43_e19/review_data.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if 2633.8 <= float(r['t_sec']) <= 2634.8]
    save_json(OUT/'E19_SCENE.json', [{k: r[k] for k in ('t_sec', 'source', 'p1_selected', 'p1_display', '2P_state')} for r in rows])


def frames() -> None:
    """指定時刻の原映像を、死亡演出と盤面消去の区別に使う。"""
    cap, tiles = cv2.VideoCapture(str(SOURCE)), []
    for stamp in TIMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, stamp*1000)
        ok, frame = cap.read()
        assert ok
        frame = cv2.resize(frame, (960, 540))
        tile = cv2.copyMakeBorder(frame[70:515, 555:920], 25, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(tile, f'{stamp:.4f}', (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        tiles.append(tile)
    cv2.imwrite(str(OUT/'scene_original.jpg'), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    cap.release()


if __name__ == '__main__':
    facts()
    frames()
