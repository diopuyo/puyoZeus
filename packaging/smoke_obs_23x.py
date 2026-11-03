"""23x の起動の煙試験 (OBS なし)。21c の煙試験 smoke_live_obs_20261020.py を同じ起動口で短く流す。

配布の起動口 (obs_runtime_bootstrap.py --video = launcher --dev-allow-video) が動画入力で起動・終了し、
3 枠がそろい、終了検査に通ることを確かめる。--x-on は puyo_live.json の X の 3 項目を true にする。
追加の検査: OFF は X の記録が出ず v4.9f の状態票が従来の 4 欄のまま、ページに暫定 X の script が無い。
ON は直接公開の状態票 (x_fresh 等) と段3 の記録が出て、ページに暫定 X の script が入る。
PC1 は M4 走行中なので、空きメモリ 8GiB を割ったらこの試験だけを止める (既定の 6.5GiB より厳しく)。
"""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_live_obs_20261020 as smoke

MIN_FREE_GIB = 8.0
X_FLAGS = ('live_x_direct_publish', 'live_x_prefire_shadow', 'live_x_prefire')
X_SCRIPT = 'overlay_x_prefire.js'
OFF_STATUS_KEYS = frozenset(('submitted', 'received', 'failure', 'pump_alive'))
X_STATUS_KEYS = ('direct_publish', 'x_fresh', 'x_fresh_fires', 'x_wakes')
PREFIRE_FILES = ('live_x_prefire.jsonl', 'live_x_prefire_summary.json')
original_config, original_observe = smoke.config, smoke.observe
x_on = '--x-on' in sys.argv
if x_on:
    sys.argv.remove('--x-on')


def config(args) -> Path:
    path = original_config(args)
    values = json.loads(path.read_text(encoding='utf-8'))
    assert not any(values.get(k) for k in X_FLAGS), '配布設定に X のフラグが入っている'
    if x_on:
        values.update({k: True for k in X_FLAGS})
        path.write_text(json.dumps(values, indent=2), encoding='utf-8')
    return path


def observe(child, args, log_errors=None) -> dict:
    """既存の観測に、暫定 X の script 取得 (ON のみ配信される) を足す。"""
    counts = original_observe(child, args, log_errors)
    page = args.out/'served_overlay.html'
    counts['page_has_x_script'] = page.exists() and X_SCRIPT in page.read_text(encoding='utf-8')
    return counts


def x_checks(out: Path) -> dict:
    root = out/'output'
    status_path = root/'v49f_display_status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    receipt = json.loads((out/'receipt.json').read_text(encoding='utf-8'))
    files = {name: (root/name).exists() for name in PREFIRE_FILES}
    if x_on:
        passed = (status.get('direct_publish') is True and all(k in status for k in X_STATUS_KEYS)
                  and all(files.values()) and receipt.get('page_has_x_script') is True)
    else:
        passed = (set(status) == OFF_STATUS_KEYS and not any(files.values())
                  and receipt.get('page_has_x_script') is False)
    return dict(x_on=x_on, v49f_status=status, prefire_files=files,
                page_has_x_script=receipt.get('page_has_x_script'), passed=passed)


def main() -> None:
    smoke.memory_guard = partial(smoke.memory_guard, minimum_free_gib=MIN_FREE_GIB)
    smoke.config = config
    smoke.observe = observe
    smoke.run_cli()
    out = smoke.arguments().out
    result = x_checks(out)
    (out/'x_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result['passed']:
        raise SystemExit('X の検査 不合格')


if __name__ == '__main__':
    main()
