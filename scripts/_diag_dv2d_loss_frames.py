"""遅れ上位八件の実フレームを読み取り、診断先だけへ保存する。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts._diag_dv2d_loss_inventory import OUT, read, save, resource
from scripts.falsedeath_collect_20261001 import original_command

TOP_N = 8
WIDTH, HEIGHT = 960, 540
FRACTIONS = (0., 1/3, 2/3, 1.)


def capture(row: dict) -> dict:
    """本番確定から観測死亡までを四時点で比較する。"""
    command = original_command(row['record'])
    video = Path(command[command.index('--video')+1])
    dest = OUT/'frames'
    dest.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return dict(video=str(video), available=False)
    images, times = [], []
    for index, fraction in enumerate(FRACTIONS):
        stamp = row['first_sec']+row['delay_sec']*fraction
        cap.set(cv2.CAP_PROP_POS_MSEC, stamp*1000)
        ok, frame = cap.read()
        assert ok, (video, stamp)
        filename = f"{row['group']}_{row['game']}_{index}_{stamp:.3f}.jpg"
        assert not (dest/filename).exists()
        cv2.imwrite(str(dest/filename), frame)
        resized = cv2.resize(frame, (WIDTH, HEIGHT))
        cv2.rectangle(resized, (0, 0), (WIDTH, 36), (0, 0, 0), -1)
        cv2.putText(resized, f"{row['group']} game {row['game']}  t={stamp:.3f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)
        images.append(resized)
        times.append(dict(requested_sec=stamp, decoded_next_frame=cap.get(cv2.CAP_PROP_POS_FRAMES), file=filename))
    cap.release()
    contact = f"{row['group']}_{row['game']}_contact.jpg"
    cv2.imwrite(str(dest/contact), np.vstack((np.hstack(images[:2]), np.hstack(images[2:]))))
    return dict(video=str(video), available=True, times=times, contact=contact)


def main() -> None:
    """同じ動画の重複報告を除いた八件だけを取り出す。"""
    save('RESOURCE_frames.json', resource())
    rows = sorted(read(OUT/'EVENTS.json'), key=lambda r: r['delay_sec'], reverse=True)[:TOP_N]
    result = [dict(group=r['group'], game=r['game'], **capture(r)) for r in rows]
    save('FRAMES.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
