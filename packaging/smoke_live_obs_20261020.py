"""新規コピーで3枠を検収。正式速度も起動元のCPU集合を維持する。"""
from __future__ import annotations
import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import traceback
from contextlib import closing
from threading import Thread
from urllib.request import urlopen

VIDEO = Path('D:/puyo_analyzer/packaging/build_20261020a_work/smoke_game_600_720.avi')
MIN_C_FREE_GIB = 10
DEFAULT_STARTUP_TIMEOUT = 240
LOG_CHUNK_BYTES = 64*1024
HEAVY_EXTERNAL_CPU_FRACTION = .2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def memory_guard(minimum_free_gib: float = 6.5) -> None:
    if shutil.disk_usage('C:/').free/1024**3 < MIN_C_FREE_GIB:
        raise RuntimeError('C:の空きが10GiB未満のため、このsmoke/測定だけを停止')
    from src.phase_j.live_b3_time_gate import MemoryStatus
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise RuntimeError('空きメモリを確認できないため開始しない')
    if memory.available/1024**3 < minimum_free_gib:
        raise RuntimeError('空きメモリ6GiBの余裕を確保するため、このsmoke/測定だけを停止')


def co_load(out: Path, label: str) -> None:
    value = subprocess.check_output(['powershell','-NoProfile','-Command',
        'Get-Process | Select-Object Id,ProcessName,CPU,WorkingSet64,PriorityClass | ConvertTo-Json -Compress'],text=True)
    (out/f'co_load_{label}.json').write_text(value,encoding='utf-8')
    clock = subprocess.check_output(['powershell', '-NoProfile', '-Command',
        'Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation | '
        'Select-Object Name,ProcessorFrequency,PercentProcessorPerformance,PercentofMaximumFrequency,PercentProcessorUtility | '
        'ConvertTo-Json -Compress'], text=True)
    (out/f'cpu_clock_{label}.json').write_text(clock, encoding='utf-8')


def guard(benchmark: bool, out: Path, wait_started_at: str | None = None) -> None:
    if not benchmark:
        return
    command = ['powershell', '-NoProfile', '-File',
        str(Path(__file__).with_name('wait_live_obs_idle.ps1')),
        '-Output', str(out/'idle_checks.jsonl'), '-ExcludePid', str(os.getpid())]
    if wait_started_at:
        command += ['-WaitStartedAt', wait_started_at]
    subprocess.run(command, check=True)
    last = json.loads((out/'idle_checks.jsonl').read_text(encoding='utf-8-sig').splitlines()[-1])
    busy = last['busy'] > 0
    if busy and not last['timed_out']:
        raise RuntimeError('CPU監視原票がbusy。速度測定を開始しない。')
    if busy:
        # 8時間後の参考測定も根本対策のCPU集合へは入らない。
        return


def config(args: argparse.Namespace) -> Path:
    assert args.video.is_file(), args.video
    values = json.loads((args.bundle/'puyo_live.json').read_text(encoding='utf-8'))
    values.update(source='video',video_path=str(args.video),start_sec=args.start_sec,end_sec=args.start_sec+args.duration,
        port=args.port,output_dir=str(args.out/'output'),
        b3_model=str(args.bundle/'app/models/b3/model.joblib'),
        b3_timing_table=str(args.bundle/'app/models/b3/timing_table.json'))
    if args.ama_off:
        values['live_ama_plain'] = False
    path = args.out/'smoke_config.json'
    path.write_text(json.dumps(values,indent=2),encoding='utf-8')
    return path


def external_processes() -> list[dict]:
    from src.phase_j.live_process_stats import process_rows, details
    rows, owned = process_rows(), {os.getpid()}
    while (found := owned | {r['pid'] for r in rows if r['parent'] in owned}) != owned:
        owned = found
    return [details(r) for r in rows if r['pid'] not in owned and
            r['name'].lower() in ('python.exe', 'pythonw.exe', 'obs64.exe', 'vmmemwsl', 'ama_reference.exe')]


def check_quiet(previous: dict | None, current: dict) -> None:
    """他の計算が再開した場合、この測定だけを停止して混在値の採用を防ぐ。"""
    if previous is None:
        return
    elapsed = current['monotonic']-previous['monotonic']
    old = {r['pid']: r for r in previous['processes']}
    busy = [r['pid'] for r in current['processes'] if r['name'].lower() != 'obs64.exe'
        and r['pid'] in old and r.get('cpu_sec', 0)-old[r['pid']].get('cpu_sec', 0)
        > elapsed*HEAVY_EXTERNAL_CPU_FRACTION]
    if busy:
        raise RuntimeError(f'外部計算が再開したため正式測定を停止: {busy}')


def observe(child: subprocess.Popen, args: argparse.Namespace, log_errors: list | None = None) -> dict:
    from src.phase_j.live_obs_audit import AsyncAudit, BUFFER_BYTES
    began, counts, served = time.monotonic(), dict(polls=0,b3=0,v49f=0,ama=0,all_three=0), False
    resource_at = float('-inf')
    previous_resource = None
    # 測定器自身のSMB書込み待ちを、HTTP表示の標本間隔へ混入させない。
    with closing(AsyncAudit((args.out/'polls.jsonl').open(
            'x', encoding='utf-8', buffering=BUFFER_BYTES))) as stream, closing(
            AsyncAudit((args.out/'external_load.jsonl').open(
                'x', encoding='utf-8', buffering=BUFFER_BYTES))) as resources:
        while child.poll() is None and time.monotonic()-began < args.duration+args.startup_timeout:
            memory_guard()
            if log_errors:
                raise RuntimeError('実行ログの保存に失敗') from log_errors[0]
            if time.monotonic()-resource_at >= 5:
                resource_at = time.monotonic()
                resource = dict(monotonic=resource_at, processes=external_processes())
                resources.write(json.dumps(resource)+'\n')
                if args.quiet_throughout:
                    check_quiet(previous_resource, resource)
                previous_resource = resource
            try:
                with urlopen(f'http://127.0.0.1:{args.port}/latest',timeout=1) as response:
                    value = json.load(response)
            except (OSError,ValueError):
                time.sleep(.1)
                continue
            # 保存の失敗をHTTPの一時エラーとして握りつぶさない。
            stream.write(json.dumps(dict(at=time.perf_counter(),snapshot=value),ensure_ascii=False)+'\n')
            counts['polls'] += 1
            valid = valid_evaluations(value)
            for key, present in zip(('b3', 'v49f', 'ama'), valid):
                counts[key] += present
            counts['all_three'] += all(valid)
            if not served:
                for name in ('','overlay_v49f.js','overlay_ama.js'):
                    with urlopen(f'http://127.0.0.1:{args.port}/'+name) as response:
                        (args.out/('served_'+(name or 'overlay.html'))).write_bytes(response.read())
                served = True
            time.sleep(.1)
    return counts


def valid_evaluations(value: dict) -> tuple[bool, ...]:
    """枠自体がnullの初期DTOも、未評価として数える。"""
    evaluation = value.get('evaluations') or {}
    fields = (('practical', 'p1_win_probability'), ('v49f', 'p1'), ('ama_plain', 'p1'))
    return tuple((evaluation.get(name) or {}).get(field) is not None for name, field in fields)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--duration',type=int,default=60)
    parser.add_argument('--video',type=Path,default=VIDEO)
    parser.add_argument('--start-sec',type=float,default=0.)
    parser.add_argument('--ffmpeg-dll-dir',type=Path)
    parser.add_argument('--port',type=int,default=18820)
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--wait-started-at')
    parser.add_argument('--ama-off', action='store_true')
    parser.add_argument('--cpu-isolation', action='store_true')
    parser.add_argument('--fast-audit', action='store_true')
    parser.add_argument('--ama-idle', action='store_true')
    parser.add_argument('--exact-seed', action='store_true')
    parser.add_argument('--stdlib-cache', action='store_true')
    parser.add_argument('--prepare-exact', action='store_true')
    parser.add_argument('--deep-timing', action='store_true')
    parser.add_argument('--quiet-throughout', action='store_true')
    parser.add_argument('--verified-runtime', type=Path)
    parser.add_argument('--source-overlay', type=Path,
                        help='既存本体を変更せず指定ソースを読む開発用比較。配布検収とは別扱い')
    parser.add_argument('--source-plan', type=Path)
    parser.add_argument('--startup-timeout', type=int, default=DEFAULT_STARTUP_TIMEOUT)
    return parser.parse_args()


def source_overlay(args: argparse.Namespace, command: list[str], environment: dict) -> list[str]:
    """較正・pycも既存資産へ書かず、使用したソースのハッシュを原票へ残す。"""
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PUYO_CALIBRATION_DIR'] = str(args.out/'device_calibration')
    if not args.source_overlay:
        return command
    plan = json.loads(args.source_plan.read_text(encoding='utf-8'))
    names = [row['name'] for row in plan['changed']]
    sources = {name: sha256((args.source_overlay/name).read_bytes()).hexdigest() for name in names}
    environment.update(PUYO_OBS_SOURCE_ROOT=str(args.source_overlay),
                       PUYO_OBS_SOURCE_MODULES=json.dumps(names),
                       PUYO_OBS_ASSET_APP=str(args.bundle/'app'))
    (args.out/'source_overlay.json').write_text(json.dumps(
        dict(source_root=str(args.source_overlay), files=sources, distribution_smoke=False),
        indent=2), encoding='utf-8')
    script = Path(__file__).with_name('run_live_obs_overlay.py')
    return [command[0], '-B', str(script), *command[command.index('--config'):command.index('--require-manifest')],
            '--skip-manifest', '--dev-allow-video']


def copy_log(child: subprocess.Popen, log: object, errors: list) -> None:
    try:
        with child.stdout:
            while value := child.stdout.read1(LOG_CHUNK_BYTES):
                log.write(value)
    except Exception as error:
        errors.append(error)


def run_child(args: argparse.Namespace, command: list[str], environment: dict) -> tuple[dict, int]:
    """標準出力の細かなwriteもOS pipeで受け、終了時まで欠落なく保存する。"""
    from src.phase_j.live_obs_audit import AsyncAudit, BUFFER_BYTES
    with closing(AsyncAudit((args.out/'smoke.log').open('xb', buffering=BUFFER_BYTES))) as log:
        child = subprocess.Popen(command, cwd=args.bundle/'app', stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
        errors: list[Exception] = []
        reader = Thread(target=copy_log, args=(child, log, errors), daemon=True)
        reader.start()
        try:
            result = observe(child, args, errors)
        except Exception as error:
            result = dict(error=f'{type(error).__name__}: {error}', b3=0, v49f=0, ama=0)
        finally:
            if child.poll() is None:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
            child.wait(timeout=30)
            reader.join(timeout=30)
        if reader.is_alive() or errors:
            raise RuntimeError('実行ログの全件保存を確認できない')
    return result, child.returncode


def shutdown_checks(args: argparse.Namespace) -> dict:
    """親のexit=0だけで、監査writerの強制終了を見逃さない。"""
    root, statuses = args.out/'output', {}
    for name in ('b3_async_status.json', 'v49f_b3_v6/b3_async_status.json'):
        path = root/name
        value = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        statuses[name] = dict(value, passed=bool(value) and not value.get('forced_termination')
            and (value.get('finished') or {}).get('inputs') == value.get('submitted'))
    if args.fast_audit:
        path = root/'v49f_display_status.json'
        value = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        statuses[path.name] = dict(value, passed=bool(value) and not value.get('failure')
            and not value.get('pump_alive') and value.get('submitted') == value.get('received'))
        for name in ('v49f_async_status.json', *(() if args.ama_off else ('ama_async_status.json',))):
            path = root/name
            value = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            statuses[name] = dict(value, passed=bool(value) and value.get('exit_code') == 0
                and not value.get('forced_termination') and not value.get('pump_alive'))
    return dict(passed=all(row['passed'] for row in statuses.values()), statuses=statuses)


def launch_command(args: argparse.Namespace) -> list[str]:
    """実配布コードと、全実行系照合を必須にした既存runtimeを選ぶ。"""
    command = [str(args.bundle/'python/python.exe'),'-B','-m','src.phase_j.launcher',
        '--config',str(config(args)),'--require-manifest','--dev-allow-video']
    if args.verified_runtime:
        if args.source_overlay:
            raise ValueError('配布実体の検収とソース重ね合わせは併用しない')
        command = [str(args.verified_runtime/'python.exe'), '-B', '-u',
            str(args.bundle/'obs_runtime_bootstrap.py'), '--bundle', str(args.bundle),
            '--runtime', str(args.verified_runtime), '--config', str(args.out/'smoke_config.json'),
            '--receipt', str(args.out/'runtime_receipt.json'), '--video']
    return command


def main(args: argparse.Namespace | None = None) -> None:
    args = args or arguments()
    memory_guard()
    args.out.mkdir(exist_ok=False)
    guard(args.benchmark, args.out, args.wait_started_at)
    co_load(args.out,'start')
    command = launch_command(args)
    environment = dict(os.environ,PYTHONIOENCODING='utf-8')
    environment['PUYO_OBS_CPU_ISOLATION'] = str(int(args.cpu_isolation))
    environment['PUYO_OBS_FAST_AUDIT'] = str(int(args.fast_audit))
    environment['PUYO_OBS_AMA_IDLE'] = str(int(args.ama_idle))
    environment['PUYO_OBS_EXACT_SEED'] = str(int(args.exact_seed))
    environment['PUYO_OBS_STDLIB_CACHE'] = str(int(args.stdlib_cache))
    environment['PUYO_OBS_PREPARE_EXACT'] = str(int(args.prepare_exact))
    environment['PUYO_OBS_DEEP_TIMING'] = str(int(args.deep_timing))
    if args.benchmark:
        environment.pop('PUYO_LIVE_INPUT_AUDIT', None)  # 速度は配布設定のログだけで測る。
    if args.ffmpeg_dll_dir:
        environment['OPENCV_FFMPEG_DLL_DIR'] = str(args.ffmpeg_dll_dir)
    command = source_overlay(args, command, environment)
    result, exit_code = run_child(args, command, environment)
    result.update(exit_code=exit_code,benchmark=args.benchmark,command=command,
        app_manifest_sha256=sha256((args.bundle/'app/MANIFEST.json').read_bytes()).hexdigest(),
        video=str(args.video),start_sec=args.start_sec,decoder_dll_dir=str(args.ffmpeg_dll_dir),
        ama_off=args.ama_off, cpu_isolation=args.cpu_isolation, fast_audit=args.fast_audit,
        ama_idle=args.ama_idle, exact_seed=args.exact_seed, stdlib_cache=args.stdlib_cache,
        prepare_exact=args.prepare_exact, deep_timing=args.deep_timing,
        verified_runtime=str(args.verified_runtime),
        source_overlay=str(args.source_overlay))
    failure = args.out/'output/ama_failure.json'
    result['ama_worker_fatal'] = json.loads(failure.read_text(encoding='utf-8'))['fatal'] if failure.exists() else None
    errors = args.out/'output/evaluation_errors.jsonl'
    result['evaluation_errors'] = (
        len(errors.read_text(encoding='utf-8').splitlines()) if errors.exists() else 0)
    result['shutdown'] = shutdown_checks(args)
    co_load(args.out,'end')
    (args.out/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result,flush=True)
    assert result['shutdown']['passed'] and not result['evaluation_errors'] and not result['ama_worker_fatal'] and exit_code == 0 and (result.get('all_three', 0)>0 or
                                    (args.ama_off and result['b3'] and result['v49f']))


def run_cli() -> None:
    """標準出力の保存自体が失敗しても、例外票は新規ファイルに残す。"""
    options = arguments()
    try:
        main(options)
    except BaseException as error:
        options.out.mkdir(parents=True, exist_ok=True)
        (options.out/'fatal_receipt.json').write_text(json.dumps(dict(
            exception=type(error).__name__, message=str(error), stack=traceback.format_exc()),
            ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    run_cli()
