"""B20: ObservedDeathDetector.update の単体費用を実フレームで内訳計測 (1スレッド設定)。"""
import time
from src.phase_j.live_cpu import configure_environment
configure_environment(1, 10)
import cv2
import numpy as np
from src.exchange_event_terminal import ObservedDeathDetector, _shrink, _peak, SEARCH_P1, SEARCH_P2
from scripts.measure_realtime_breakdown_20260928 import DEFAULT_VIDEO

cv2.setNumThreads(1)
cap = cv2.VideoCapture(str(DEFAULT_VIDEO)); fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(5900*fps))
frames = []
for _ in range(400):
    ok, im = cap.read(); cap.grab()
    frames.append(im)


def timeit(fn) -> float:
    s = time.perf_counter()
    for f in frames:
        fn(f)
    return (time.perf_counter()-s)/len(frames)*1000


off, on = ObservedDeathDetector(fast=False), ObservedDeathDetector(fast=True)
rois = (SEARCH_P1, SEARCH_P2)
for _ in range(2):
    print('off update ms %.2f' % timeit(off.update))
    print('on  update ms %.2f (full_matches %s/%d)' % (timeit(on.update), on.full_matches, on.frames))
print('cvt ROI x2   ms %.2f' % timeit(lambda f: [cv2.cvtColor(f[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY) for x, y, w, h in rois]))
grays = [[cv2.cvtColor(f[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY) for x, y, w, h in rois] for f in frames]
s = time.perf_counter()
for g in grays:
    [_shrink(r) for r in g]
print('shrink x2    ms %.2f' % ((time.perf_counter()-s)/len(grays)*1000))
smalls = [[_shrink(r) for r in g] for g in grays]
s = time.perf_counter()
for g in smalls:
    [_peak(r, on.small_template) for r in g]
print('coarse match x2 ms %.2f' % ((time.perf_counter()-s)/len(smalls)*1000))
