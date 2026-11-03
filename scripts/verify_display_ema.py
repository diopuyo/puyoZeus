"""同じ保存判定列を表示だけに流し、EMA係数の効果を送信周期から分離する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.b3_live_display import B3Display
from src.exchange_display_smoothing import EMA_ALPHA, FAST_EMA_ALPHA

PERCENT_SCALE = 100.0


def replay(rows: list[dict], fast: bool) -> tuple[list[float | None], dict]:
    """物理信号は保存列に無いため両条件で固定し、同じ境界・由来・生値を与える。"""
    display = B3Display(fast_ema=fast)
    raw_values: list[float | None] = []
    gaps: list[float] = []
    death_count = 0
    for row in rows:
        display.boundary(row['game'])
        raw = row['raw_probability']
        if raw is not None:
            display.prediction(raw, int(row['source'] == 'exchange'), row['t_sec'])
        shown = display.update(row['t_sec'], row['source'], raw,
                               active=row.get('match_active', True) and raw is not None)
        raw_values.append(display.raw)
        assert display.raw == raw
        if raw is not None:
            gaps.append(abs(shown[1]-raw)*PERCENT_SCALE)
        if row['source'] == 'confirmed_death' and raw is not None:
            assert shown[1] == raw
            death_count += 1
    return raw_values, dict(alpha=FAST_EMA_ALPHA if fast else EMA_ALPHA,
        n=len(gaps), mean_gap_pp=float(np.mean(gaps)), p95_gap_pp=float(np.percentile(gaps, 95)),
        confirmed_death_immediate=death_count)


def verify(path: Path) -> dict:
    started = perf_counter()
    rows = json.loads(path.read_text(encoding='utf-8'))
    before_raw, before = replay(rows, False)
    after_raw, after = replay(rows, True)
    equal = sum(a == b for a, b in zip(before_raw, after_raw, strict=True))
    assert equal == len(rows)
    return dict(input=str(path), basis='保存raw列・同一時刻/境界/由来・物理信号固定、送信変更なし',
                rows=len(rows), judgment_equal=equal, before=before, after=after,
                elapsed_sec=perf_counter()-started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    result = verify(options.input)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
