"""開始直前の同GPU照会を三時点で保存する。CPU熱安全性は保証しない。"""
from pathlib import Path
import json
import time
import resource_guard as G

ROOT = Path(__file__).resolve().parent


def main() -> None:
    rows = []
    for index in range(3):
        if index: time.sleep(G.R.INTERVAL_SEC)
        row = dict(time=time.time(), **G.gpu())
        rows.append(row)
        print(json.dumps(row), flush=True)
    passed = all(r['hw_thermal'] == r['sw_thermal'] == 'Not Active' for r in rows)
    with (ROOT / 'THERMAL_PREFLIGHT_v1.json').open('x') as stream:
        json.dump(dict(rows=rows, thermal_preflight_pass=passed, cpu_package_temperature_verified=False), stream)
    if not passed: raise SystemExit(1)


if __name__ == '__main__':
    main()
