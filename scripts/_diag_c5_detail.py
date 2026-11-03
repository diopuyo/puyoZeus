"""式と予告の実映像を拡大抽出し、台帳の変化を保存する。"""
from __future__ import annotations

import json
import cv2
import numpy as np

from scripts._diag_c5_extract import VIDEO
from scripts._diag_c5_inspect import BASE, OUT, guard, save
from src.exchange_event_record import read_records

TIMES = [1708.3, 1710., 1712., 1714., 1716., 1718., 1720., 1722., 1724.,
         1725., 1726., 1726.5, 1727., 1727.433333, 1727.7, 1728.2, 1728.5,
         1729.5, 1730.8, 1731.7, 1731.9, 1732.2, 1733.7, 1734., 1734.3]


def frames() -> None:
    """上端予告を高解像度で並べる。"""
    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS)
    strips = []
    for stamp in TIMES:
        guard()
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(stamp*fps))
        ok, frame = capture.read()
        assert ok
        cv2.imwrite(str(OUT / f'detail_{stamp:.3f}.jpg'), frame)
        strip = np.concatenate([frame[95:195, 250:695], frame[95:195, 1230:1675]], axis=1)
        strip = np.pad(strip, ((0,0),(150,0),(0,0)))
        cv2.putText(strip, f'{stamp:.3f}', (5,50), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        strips.append(strip)
    capture.release()
    for start in range(0, len(strips), 9):
        cv2.imwrite(str(OUT / f'forecast_{start//9}.jpg'), np.concatenate(strips[start:start+9]))


def changes() -> None:
    """連鎖式・基準得点・台帳を変化点だけにする。"""
    previous = None
    rows = []
    for row in read_records(BASE / 'collect/records/c5.jsonl.gz'):
        if row['kind'] != 'update' or row['args'][3] > 1740:
            continue
        sides, snapshot, final, stamp, game, formula, displayed, visible = row['args']
        value = dict(formula=formula, final=vars(final), snapshot=vars(snapshot))
        if value != previous:
            rows.append(dict(t=stamp, displayed=displayed, **value))
            previous = value
    save('accounting_changes.json', rows)
    trace = OUT / 'first_proof_inputs.json'
    if trace.exists():
        data = json.loads(trace.read_text())
        for c in data['chains']:
            print('CHAIN', {k:v for k,v in c.items() if 'board' not in k and 'proof' not in k})
        print('LEDGER', data['safety']['ledger'])
        print('ELAPSED', data['safety']['elapsed'])
        print('CONTEXT', data['context'])


def main() -> None:
    """診断情報だけを新規先へ出力する。"""
    guard()
    frames()
    changes()


if __name__ == '__main__':
    main()
