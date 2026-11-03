"""旧ワーカーを維持し、候補改版時だけ新経路を0から再開する監督。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

AUTHORIZATION = 'user-authorized-start-before-r2-2026-10-08'
POLL_SECONDS = 5
CPU_LAUNCH_LOCK = Path('/tmp/puyo_image_free_cpu_launch.lock')
GUARD_COUNTS: dict[str, int] = {}


def guard_event(root: Path, kind: str, error: Exception | None = None) -> None:
    """監督の差し替え件数と例外を、計算出力と別に記録する。"""
    import traceback
    GUARD_COUNTS[kind] = GUARD_COUNTS.get(kind, 0)+1
    record = dict(pid=os.getpid(), at=time.time(), counts=dict(GUARD_COUNTS), event=kind,
                  error=repr(error) if error else None,
                  traceback=traceback.format_exc() if error else None)
    try:
        with (root/'supervisor_guard_events.jsonl').open('a') as file:
            file.write(json.dumps(record, ensure_ascii=False)+'\n')
        (root/'supervisor_guard.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    except Exception as logging_error:
        print('監督例外の記録失敗:', repr(logging_error), record, file=sys.stderr, flush=True)


def install_progress_guard(root: Path) -> None:
    """固定ソースを変更せず、DrvFs上の一時的な進捗欠落だけ読み飛ばす。"""
    from scripts.image_free_resources import ResourceBudget
    original = ResourceBudget.observe
    def observe(self: Any, children: dict, output: Path) -> None:
        for name, child in children.items():
            try:
                original(self, {name: child}, output)
            except (FileNotFoundError, json.JSONDecodeError) as error:
                guard_event(root, 'progress_read_skipped', error)
    ResourceBudget.observe = observe
    if ResourceBudget.observe is not observe:
        raise RuntimeError('進捗読取ガードの差し替え失敗')
    guard_event(root, 'observe_replacements')


def guarded_tick(root: Path, action: Any) -> bool:
    """一時的な監督例外でfinallyへ抜けず、既存ワーカーを維持して次巡へ進む。"""
    try:
        action()
        return True
    except Exception as error:
        guard_event(root, 'tick_exceptions', error)
        return False


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(source: Path) -> dict:
    return {str(p.relative_to(source)): sha(p) for base in ('src', 'scripts')
            for p in (source/base).rglob('*.py')}


def verify_initial(config: dict, source: Path, gate: dict) -> None:
    """r2完了を偽装せず、今回の先行起動指示と検証済み本体を照合する。"""
    from scripts.image_free_stream_merge import validate
    root = Path(config['output'])
    if gate.get('authorization') != AUTHORIZATION:
        raise ValueError('先行起動の明示指示が必要です')
    if source_manifest(source) != gate['source_manifest']:
        raise ValueError('固定ソースSHA不一致')
    for name, digest in gate['model_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError('候補モデルSHA不一致: '+name)
    for key, digest in gate['input_sha256'].items():
        if sha(Path(config[key])) != digest:
            raise ValueError('入力SHA不一致: '+key)
    for name in ('route_proof', 'route_plan'):
        if sha(root/(name+'.json')) != gate[name+'_sha256']:
            raise ValueError('証拠/計画SHA不一致: '+name)
    proof, plan = read(root/'route_proof.json'), read(root/'route_plan.json')
    if (not proof['passed'] or proof['source_manifest'] != gate['source_manifest']
            or proof['model_sha256'] != gate['model_sha256']):
        raise ValueError('同居版との一致検証が不正です')
    validate(plan)
    keys = ('id', 'video_id', 'route', 'start_frame', 'end_frame', 'owned_start', 'owned_end', 'seed_sha256')
    assigned = [j for j in plan['jobs'] if j['host'] == config['host']]
    if [[j.get(k) for k in keys] for j in assigned] != [[j.get(k) for k in keys] for j in config['videos']]:
        raise ValueError('担当経路が計画と違います')


def revision_routes(record: dict, initial: dict) -> dict:
    """候補世代の完全な資産照合を済ませるまで現ワーカーを止めない。"""
    config = read(Path(record['config']))
    if record.get('r2_exit') != 0 or not record.get('commit'):
        raise ValueError('r2完了票が必要です')
    if source_manifest(Path(record['source'])) != record['source_manifest']:
        raise ValueError('改版ソースSHA不一致')
    for name, digest in record['model_sha256'].items():
        if sha(Path(config['output'])/name) != digest:
            raise ValueError('改版モデルSHA不一致')
    expected = {j['id']: j for j in initial['videos'] if j['route'] == 'new'}
    if {j['id'] for j in config['videos']} != set(expected):
        raise ValueError('新経路の担当が変わっています')
    for job in config['videos']:
        prior = expected[job['id']]
        if (job['route'] != 'new' or job['start_frame'] != 0 or job.get('seed')
                or job['end_frame'] != prior['end_frame'] or job['path'] != prior['path']):
            raise ValueError('候補改版は同じ動画を0から再計算します')
    if Path(config['output']).resolve() == Path(initial['output']).resolve():
        raise ValueError('改版は別の出力先が必要です')
    return {j['id']: dict(job=j, config=record['config'], source=record['source'],
                        output=config['output']) for j in config['videos']}


def replace_new(routes: dict, replacements: dict, children: dict, codes: dict, stop: Any) -> None:
    """旧プロセスと旧チェックポイントには触れず、新だけを置換する。"""
    for name, replacement in replacements.items():
        if routes[name]['job']['route'] != 'new':
            raise ValueError('旧経路の置換は禁止です')
        if name in children:
            stop(children.pop(name))
        codes.pop(name, None)
        routes[name] = replacement


def reserved_cpus(proc: Path = Path('/proc')) -> set[int]:
    """起動直後の広いCPUマスクでも、argvの割当予定CPUを予約とみなす。"""
    result = set()
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path/'cmdline').read_bytes().split(b'\0')
            if not all(flag in args for flag in (b'--worker', b'--config')):
                continue
            affinity = os.sched_getaffinity(int(path.name))
            if len(affinity) == 1:
                result.update(affinity)
            elif b'--cpu' in args:
                result.add(int(args[args.index(b'--cpu')+1]))
        except (OSError, ValueError, IndexError):
            continue
    return result


def launch_route(route: dict) -> subprocess.Popen | None:
    import fcntl
    with CPU_LAUNCH_LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return launch_reserved(route)


def launch_reserved(route: dict) -> subprocess.Popen | None:
    from scripts.image_free_parallel import THREAD_ENV, worker_cpus
    from scripts.image_free_resources import unoccupied_cpus
    job, source = route['job'], Path(route['source'])
    directory = Path(route['output'])/job['id']
    directory.mkdir(parents=True, exist_ok=True)
    reserved = reserved_cpus()
    candidates = [c for c in unoccupied_cpus(worker_cpus()) if c not in reserved]
    if not candidates:
        return None
    cpu = candidates[0]
    entry = route.get('worker_entry') or str(source/'scripts/image_free_parallel.py')
    command = ['taskset', '-c', str(cpu), sys.executable, entry,
               '--config', route['config'], '--source', str(source), '--worker', job['id'],
               '--parent', str(os.getpid()), '--cpu', str(cpu)]
    with (directory/'parallel.log').open('a') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, **THREAD_ENV, PYTHONPATH=str(source)))
    # 重いimportやライブラリのスレッド生成より前に、親側から固定する。
    try:
        os.sched_setaffinity(child.pid, {cpu})
    except OSError:
        child.terminate()
        child.wait()
        raise
    return child


def adopt_revision(root: Path, initial: dict, state: dict) -> None:
    from scripts.image_free_parallel import stop_worker, save_json
    publication = root/'candidate_revision.json'
    if not publication.exists():
        return
    record = read(publication)
    if record['revision_id'] == state['revision_id']:
        return
    replacements = revision_routes(record, initial)
    replace_new(state['routes'], replacements, state['children'], state['codes'], stop_worker)
    state['revision_id'] = record['revision_id']
    save_json(root/'candidate_adoption.json', dict(revision_id=state['revision_id'],
        new_restarted_from_zero=list(replacements), old_preserved=True, at=time.time()))


def completed(route: dict) -> bool:
    """完了CPを空ループで再保存すると末尾チャンクを壊すため再起動しない。"""
    directory = Path(route['output'])/route['job']['id']
    try:
        progress, checkpoint = read(directory/'progress.json'), read(directory/'checkpoint.json')
    except FileNotFoundError:
        return False
    end = route['job']['end_frame']
    return bool(progress.get('complete') and progress['frames'] == end and checkpoint['frame'] == end)


def tick(root: Path, initial: dict, state: dict, monitor: Any, budget: Any, limit: int) -> None:
    from scripts.image_free_parallel import resource_control, worker_cpus, save_json
    try:
        adopt_revision(root, initial, state)
    except (ValueError, OSError, KeyError) as error:
        save_json(root/'revision_error.json', dict(error=str(error), at=time.time(), old_preserved=True))
    children, codes, routes = state['children'], state['codes'], state['routes']
    hold = read(root/'validation_capacity.json') if (root/'validation_capacity.json').exists() else {}
    held = set(hold.get('hold_routes', [])) if hold.get('until', 0) > time.time() else set()
    if held:
        limit = min(limit, int(hold['max_workers']))
    seed = root/'rss_seed.json'
    if not budget.qualified and seed.exists():
        measured = read(seed)
        if measured.get('source_manifest') == state.get('source_manifest'):
            budget.peak_rss = max(budget.peak_rss, int(measured['measured_peak_rss']))
            budget.qualified = True
    paused = monitor.poll()
    for name, child in list(children.items()):
        budget.observe({name: child}, Path(routes[name]['output']))
        if child.poll() is not None:
            codes[name] = child.returncode
            children.pop(name)
    resource_control(monitor, children, paused)
    maximum = min(limit, budget.limit(monitor, len(children), len(worker_cpus())))
    for name, route in routes.items():
        if name not in children and name not in codes and completed(route):
            codes[name] = 0
    pending = [name for name in routes if name not in children and name not in codes and name not in held]
    if pending and len(children) < maximum and budget.can_start(monitor):
        name = pending[0]
        child = launch_route(routes[name])
        if child is not None:
            children[name] = child
    save_json(root/'supervisor.json', dict(supervisor_pid=os.getpid(), worker_pids={k: p.pid for k, p in children.items()},
        exit_codes=codes, effective_workers=maximum, max_workers=limit, measured_peak_rss=budget.peak_rss,
        paused=paused, reason=monitor.reason, windows_free_bytes=monitor.windows_free,
        wsl_free_bytes=monitor.wsl_free, revision_id=state['revision_id'], started_at=state['started_at'],
        routes={k: dict(route=v['job']['route'], source=v['source'], output=v['output']) for k, v in routes.items()},
        status='running' if children or pending else 'routes_finished_revision_watch', updated_at=time.time()))


def main() -> None:
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--execute', choices=[AUTHORIZATION], required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from scripts.image_free_parallel import Monitor, stop_worker
    from scripts.image_free_resources import ResourceBudget
    config, gate = read(args.config), read(args.gate)
    verify_initial(config, args.source, gate)
    root = Path(config['output'])
    plan = read(root/'route_plan.json')
    limit = plan['worker_limits'][config['host']]
    routes = {j['id']: dict(job=j, config=str(args.config.resolve()), source=str(args.source.resolve()),
                           output=str(root), worker_entry=config.get('audit_worker')) for j in config['videos']}
    started = read(root/'supervisor.json').get('started_at', time.time()) if (root/'supervisor.json').exists() else time.time()
    state = dict(routes=routes, children={}, codes={}, revision_id='fix2-78e33b5e', started_at=started,
                 source_manifest=gate['source_manifest'])
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        install_progress_guard(root)
        monitor, budget = Monitor(), ResourceBudget()
        try:
            while True:
                guarded_tick(root, lambda: tick(root, config, state, monitor, budget, limit))
                time.sleep(POLL_SECONDS)
        finally:
            for child in state['children'].values():
                stop_worker(child)


if __name__ == '__main__':
    main()
