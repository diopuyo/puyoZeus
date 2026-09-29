"""段2合格後だけ、R1の5記録を最大3並列で収集する。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from scripts.run_e17_ablation_20260928 import ALL_SOURCES
from scripts import compare_e34c_records as compare

OUT = Path('logs/r1')
WORKERS = 3


def verify_source(source: str) -> dict:
    """一記録の全入力と再生用特徴が一致した場合だけ、そのONを許可する。"""
    compare.OUT = OUT/'off_capture'
    check = json.loads((compare.OUT/'checks'/f'{source}.full.json').read_text())
    payload = compare.completed_source(source)
    assert check['passed'], f'{source}: OFFの全行一致が不成立'
    assert all(payload['records'][kind]['passed'] for kind in compare.INPUT_KINDS), source
    assert payload['feature_cache']['passed'], f'{source}: OFFの特徴量が不一致'
    return dict(check=check, payload=payload)


def verify_off() -> dict:
    """全記録の検収結果を一つにまとめる。"""
    checked = {s: verify_source(s) for s in ALL_SOURCES}
    checks = {s: value['check'] for s, value in checked.items()}
    payloads = {s: value['payload'] for s, value in checked.items()}
    return dict(passed=True, sources=checks, payloads=payloads)


def launch(task: tuple[str, str]) -> dict:
    """既存原票を上書きせず、完了記録だけを再開時に省略する。"""
    source, mode = task
    root = OUT if mode == 'on' else OUT/'off_capture'
    root.mkdir(parents=True, exist_ok=True)
    with (root/f'{source}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = root/('completed' if mode == 'on' else 'checks')/(
            f'{source}.json' if mode == 'on' else f'{source}.full.json')
        if not (root/'records'/f'{source}.jsonl.json').exists() or not marker.exists():
            with (OUT/f'collect_{mode}_{source}.log').open('a') as stream:
                subprocess.run([sys.executable, '-B', '-m', 'scripts.collect_r1', '--source', source,
                                '--mode', mode], stdout=stream, stderr=subprocess.STDOUT, check=True)
    return dict(source=source, mode=mode, completed=True)


def paired(source: str) -> dict:
    """記録ごとのOFF一致を必須にし、待機中の枠を他記録のONへ回す。"""
    launch((source, 'off'))
    verify_source(source)
    return launch((source, 'on'))


def main() -> None:
    """各ONの前に対応OFFを検収し、両条件全5記録が揃うまで完了にしない。"""
    assert json.loads((OUT/'stage2/SUMMARY.json').read_text())['passed']
    # 長いzenchiを先に投入し、末尾に長区間一つだけが残る待ち時間を減らす。
    order = ('zenchi', *(source for source in ALL_SOURCES if source != 'zenchi'))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        result = list(pool.map(paired, order))
    checks = verify_off()
    (OUT/'OFF_INPUT_CHECK.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    for mode in ('off', 'on'):
        (OUT/f'{mode.upper()}_COLLECT_COMPLETE.json').write_text(json.dumps(
            [dict(row, mode=mode) for row in result]))


if __name__ == '__main__':
    main()
