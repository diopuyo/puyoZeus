"""E22とE27の実表示EMAを同じ式で比較し、指定場面の初到達を記録する。"""
from __future__ import annotations

import csv
from pathlib import Path
from scripts.measure_e23_display_20260928 import displayed, first, START, END
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e27')
DEADLINE_SEC = 2766.


def main() -> None:
    """平滑化前の確率で前倒しを主張せず、描画後の到達時刻を保存する。"""
    before = displayed(Path('logs/e22/on/review/display.npz'))
    after = displayed(OUT/'on/review/display.npz')
    stamp = first(*after)
    result = dict(first_sec=stamp, baseline_first_sec=first(*before), deadline_sec=DEADLINE_SEC,
        scene_gate=stamp is not None and stamp <= DEADLINE_SEC, scenes=1)
    save_json(OUT/'SCENE_DISPLAY.json', result)
    with (OUT/'SCENE_DISPLAY.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t_sec', 'E22_p1_display', 'E27_p1_display'])
        writer.writerows((float(t), float(a), float(b)) for t, a, b in zip(before[0], before[1], after[1])
                         if START <= t <= END+1)
    print(result, flush=True)


if __name__ == '__main__':
    main()
