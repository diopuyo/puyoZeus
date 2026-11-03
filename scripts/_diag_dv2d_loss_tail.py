"""観測死亡の直前十一秒を一秒刻みで保存する。"""
from __future__ import annotations

import cv2
import numpy as np

from scripts._diag_dv2d_loss_inventory import OUT, read, save, resource
from scripts.falsedeath_collect_20261001 import original_command

TAIL_SECONDS = 11
WIDTH, HEIGHT = 480, 270
COLS = 4


def capture(row: dict) -> dict:
    """死亡演出と検出時刻の差を目視確認できる連続静止画を作る。"""
    command = original_command(row['record'])
    video = command[command.index('--video')+1]
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), video
    frames, stamps = [], []
    for offset in range(-TAIL_SECONDS, 1):
        stamp = row['candidate_observed_death_sec']+offset
        cap.set(cv2.CAP_PROP_POS_MSEC, stamp*1000)
        ok, frame = cap.read()
        assert ok, stamp
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        cv2.rectangle(frame, (0, 0), (WIDTH, 24), (0, 0, 0), -1)
        cv2.putText(frame, f'{stamp:.3f}s ({offset:+d}s)', (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        frames.append(frame)
        stamps.append(stamp)
    cap.release()
    contact = np.vstack([np.hstack(frames[i:i+COLS]) for i in range(0, len(frames), COLS)])
    name = f"{row['group']}_{row['game']}_tail.jpg"
    assert not (OUT/'frames'/name).exists()
    cv2.imwrite(str(OUT/'frames'/name), contact)
    return dict(group=row['group'], game=row['game'], stamps=stamps, contact=name)


def main() -> None:
    """上位八件に限定し、既存動画には書き込まない。"""
    save('RESOURCE_tail.json', resource())
    rows = sorted(read(OUT/'EVENTS.json'), key=lambda r:r['delay_sec'], reverse=True)[:8]
    save('TAIL_FRAMES.json', [capture(row) for row in rows])


if __name__ == '__main__':
    main()
