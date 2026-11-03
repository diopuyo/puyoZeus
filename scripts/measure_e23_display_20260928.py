"""採用確率とEMA後の実表示を区別して、指定場面の初到達時刻を測る。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.visualize_advantage_overlay import EMA_ALPHA

OUT = Path('logs/e23')
START, END, THRESHOLD = 2755., 2771., .95


def displayed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """記録表示行の採用値へ、既存EMAと死亡確定のバイパスを再適用する。"""
    data = np.load(path)
    probability, values = .5, []
    for selected, source in zip(data['display_p1'], data['source']):
        if source == 'confirmed_death':
            values.append(float(selected))
        else:
            probability = EMA_ALPHA*selected+(1-EMA_ALPHA)*probability
            values.append(float(probability))
    return data['t_sec'], np.asarray(values)


def first(times: np.ndarray, values: np.ndarray, end: float | None = None) -> float | None:
    """指定開始以降に表示1P勝率95%以上になる最初の時刻を返す。"""
    mask = (times >= START) & (values >= THRESHOLD)
    if end is not None:
        mask &= times <= end
    return float(times[mask][0]) if mask.any() else None


def main() -> None:
    """E22実描画CSVとの窓内一致を検証し、E23の実表示を同じ式で測る。"""
    before = displayed(Path('logs/e22/on/review/display.npz'))
    after = displayed(OUT/'on/review/display.npz')
    with Path('logs/review_zenchi_g41_43_e22/review_data.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    actual = {float(r['t_sec']): float(r['p1_display']) for r in rows}
    confirmed = next(float(r['t_sec']) for r in rows if float(r['t_sec']) >= START
                     and r['source'] == 'confirmed_death')
    errors = [abs(p-actual[t]) for t, p in zip(*before) if START <= t <= END]
    assert errors and max(errors) <= 1e-12
    result = dict(baseline_first_sec=first(*before), first_sec=first(*after, END),
        first_after_window_sec=first(*after), baseline_csv_max_error=max(errors),
        baseline_confirmed_death_sec=confirmed,
        scene_gate=first(*after, END) is not None and first(*after, END) < first(*before))
    save_json(OUT/'SCENE_DISPLAY.json', result)
    with (OUT/'SCENE_DISPLAY.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['t_sec', 'E22_p1_display', 'E23_p1_display'])
        writer.writerows((float(t), float(a), float(b)) for t, a, b in zip(before[0], before[1], after[1])
                        if START <= t <= END+1)
    print(result)


if __name__ == '__main__':
    main()
