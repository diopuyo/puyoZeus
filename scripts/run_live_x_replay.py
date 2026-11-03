"""録画を配信経路へ流し直す唯一の起動口 (F=全コマ / L=配信と同じ時間制約)。合言葉の門つき。

plan: 何も作らず計画を表示する (状態確認はこれだけで行う)。
run:  --token と環境変数の確認語が両方そろったときだけ、出力先を作って順に走らせる。
      そろわなければ何も作らずに落ちる (誤起動防止。工程共通 §8)。
条件: L = 21c と同じ puyo_live.json・realtime=True (コマ落ち・壁時計門あり)。PC1 で M4 の担当分が
      終わった後か一時停止中に回す (親の裁定 §8-6)。F = realtime=False で全コマ (入力時刻へ置いて採点)。
腕:   live_x_direct_publish の OFF/ON を ABBA 順 (既定 OFF,ON,ON,OFF) で回す。
採点: 各腕の output を scripts/measure_live_x_timing.py score (L は --placement wall、F は input) と
      scripts/measure_live_x_effects.py misfire / compare で数える。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from typing import Any, Mapping

WORKTREE = Path(__file__).resolve().parents[1]
BUNDLE = Path('D:/puyo_analyzer/packaging/build_20261021c_stage/PuyoLive')
RUNTIME = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive/python'))
VIDEO = Path('D:/puyo_analyzer/obs_vs_video_20261022/pc2_inputs/recording.mp4')
VIDEO_SHA256 = '27EFFFDC262545CE0DED4277C3B5CFB40B915739D6CDDDB5A91A723BEF03BDCA'.lower()  # TRANSFER_READY.json
BASE_COMMIT = 'a1e2321d'        # 21c 配布 app と同じソース (改行以外) の起点
SAVED = Path('D:/PuyoLive_OBS_20261021c_USER_ONLY/output/20261021T205226_258349')
DLL = Path(_public_dev_text('<work>/media_tmp/b2_cv_20261020/cv2'))
CHILD = Path(__file__).with_name('live_x_replay_child.py')
RUN_TOKEN = 'RUN-LIVE-X-REPLAY'
CONFIRM_ENV = 'LIVE_X_REPLAY_CONFIRM'
CONFIRM_WORD = 'PC1-M4-QUIET'
OFFSET_SEC = 715.394343291      # 記録秒 = 録画秒 + OFFSET (alignment_summary.json の採用値)
CROP = (1636, 920)              # OBS シーン倍率 0.852 の左上 (OBS_VS_VIDEO §6)
PORT = 18871
PUYOLIVE_PORT = 8765
MIN_FREE_GIB = 8.5              # 21c 起動時の空き予約 (start_obs.ps1) と同じ
GIB = 1024**3
ARMS = ('OFF', 'ON', 'ON', 'OFF')
MIN_SOURCE_PATCHED_PROCESSES = 2   # 評価 process + 認識の子 process
CONDITIONS = dict(L=True, F=False)   # 条件 → realtime
OBS_ENV = dict(PUYO_OBS_CPU_ISOLATION='0', PUYO_OBS_AMA_IDLE='0', PUYO_OBS_FAST_AUDIT='1',
               PUYO_OBS_EXACT_SEED='1', PUYO_OBS_PREPARE_EXACT='1', PUYO_OBS_DEEP_TIMING='1',
               PUYO_OBS_STDLIB_CACHE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')


def authorize(token: str | None, environ: Mapping[str, str]) -> None:
    """合言葉と確認語の両方が無ければ、何も作らずに落ちる。"""
    if token != RUN_TOKEN or environ.get(CONFIRM_ENV) != CONFIRM_WORD:
        raise SystemExit(f'起動しない: --token {RUN_TOKEN} と 環境変数 {CONFIRM_ENV}={CONFIRM_WORD} の両方が必要'
                         ' (PC1 の M4 が止まっていることを人が確かめてから)')


def git_lines(*args: str) -> list[str]:
    output = subprocess.run(['git', '-C', str(WORKTREE), *args], check=True, capture_output=True, text=True)
    return [line for line in output.stdout.splitlines() if line]


# 配布 app が読む scripts のうち、差し替えてよいもの。run_live_pipeline_20260928.py は ROOT を __file__ から決めて models などを
# 探すので、worktree から読ませると壊れる (models が worktree に無い)。差し替えず、変更もしない (記録は src から書く)。
OVERLAY_SCRIPTS = frozenset({'scripts/collect_boards_lean.py'})
FDR_RECOGNITION_KEYS = ('enable_media_tick_counters',)   # ON の腕だけが認識設定に足す鍵 (コマ落ちに強い状態判定)


def overlay_modules() -> list[str]:
    """起点 (21c と同じソース) から変えた・足した src の .py と、OVERLAY_SCRIPTS のうち変えたものだけを差し替える。"""
    names = git_lines('diff', '--name-only', BASE_COMMIT, '--', 'src', 'scripts')
    names += git_lines('ls-files', '--others', '--exclude-standard', '--', 'src')
    return sorted({name for name in names if name.endswith('.py') and (WORKTREE/name).exists()
                   and (name.startswith('src/') or name in OVERLAY_SCRIPTS)})


def same_text(left: Path, right: Path) -> bool:
    """改行コードだけの違い (配布は CRLF) は同じとみなす。"""
    return left.read_bytes().replace(b'\r\n', b'\n') == right.read_bytes().replace(b'\r\n', b'\n')


def bundle_drift(modules: list[str], app: Path = BUNDLE/'app') -> list[str]:
    """差し替えない src が配布 app と違えば、起点が 21c と一致していない (走らせない)。"""
    drift = []
    for path in sorted([*(app/'src').rglob('*.py'), *(app/'scripts').rglob('*.py')]):
        relative = path.relative_to(app).as_posix()
        source = WORKTREE/relative
        if relative not in modules and source.exists() and not same_text(path, source):
            drift.append(relative)
    return drift


def user_config(arm: str, condition: str, start: float, end: float | None, out: Path,
                saved: Path = SAVED) -> dict:
    """21c の puyo_live.json に、入力・区間・出力先・段1 フラグだけを上書きする。"""
    values = json.loads((saved/'puyo_live.json').read_text(encoding='utf-8-sig'))
    values.update(source='video', video_path=str(VIDEO), start_sec=max(0., start-OFFSET_SEC),
                  port=PORT, output_dir=str(out/'output'), live_x_direct_publish=arm == 'ON',
                  b3_model=str(BUNDLE/'app/models/b3/model.joblib'),
                  b3_timing_table=str(BUNDLE/'app/models/b3/timing_table.json'))
    values.pop('device_name', None)
    values.pop('device_index', None)
    values.pop('dshow_color_correction', None)   # 601→709 は子の入力差し替えで掛ける
    if end is not None:
        values['end_sec'] = end-OFFSET_SEC
    return values


def child_plan(condition: str, out: Path, modules: list[str], saved: Path = SAVED) -> dict:
    reference = json.loads((saved/'recognition_config.json').read_text(encoding='utf-8'))
    return dict(app=str(BUNDLE/'app'), source_root=str(WORKTREE), source_modules=modules,
                crop=list(CROP), offset_sec=OFFSET_SEC, color_correction='601to709',
                realtime=CONDITIONS[condition], patch_log=str(out/'patches.jsonl'),
                recognition_overrides=dict(enable_placement_signal_reconcile=
                                           reference['enable_placement_signal_reconcile']))


def plan(args: argparse.Namespace) -> list[dict]:
    """副作用なしで各腕の出力先・設定・差し替え対象・コマンドを組む。"""
    modules = overlay_modules()
    drift = bundle_drift(modules)
    if drift:
        raise SystemExit(f'配布 app と起点 {BASE_COMMIT} のソースが違う: {drift[:10]}')
    runs = []
    for order, arm in enumerate(args.arms, 1):
        out = args.root/f'{args.condition}{order}_{arm}'
        runs.append(dict(name=out.name, arm=arm, out=str(out),
                         config=user_config(arm, args.condition, args.start, args.end, out),
                         child=child_plan(args.condition, out, modules),
                         command=[str(RUNTIME/'python.exe'), '-B', '-u', str(CHILD), '--config',
                                  str(out/'puyo_live.json'), '--skip-manifest', '--dev-allow-video']))
    return runs


def blocked() -> str | None:
    """配信本番 (PuyoLive) が動いている・空きメモリ不足なら走らせない。"""
    with socket.socket() as client:
        client.settimeout(.1)
        if client.connect_ex(('127.0.0.1', PUYOLIVE_PORT)) == 0:
            return 'PuyoLive が待受中 (OBS 試験中の可能性)'
    import ctypes
    from src.phase_j.live_b3_time_gate import MemoryStatus
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)) and memory.available/GIB < MIN_FREE_GIB:
        return f'空きメモリが {MIN_FREE_GIB}GiB 未満'
    return None


def digest(path: Path) -> str:
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def run_one(run: dict) -> dict:
    """1 腕を走らせ、受領票 (終了コード・差し替え件数・認識設定の一致) を残す。"""
    out = Path(run['out'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'temp').mkdir()
    (out/'puyo_live.json').write_text(json.dumps(run['config'], ensure_ascii=False, indent=2), encoding='utf-8')
    environment = dict(os.environ, **OBS_ENV, TEMP=str(out/'temp'), TMP=str(out/'temp'),
                       OPENCV_FFMPEG_DLL_DIR=str(DLL), PUYO_CALIBRATION_DIR=str(out/'calibration'),
                       LIVE_X_REPLAY_CHILD=json.dumps(run['child']))
    with (out/'run.log').open('xb') as log:
        code = subprocess.run(run['command'], cwd=BUNDLE/'app', env=environment, stdout=log,
                              stderr=subprocess.STDOUT, check=False).returncode
    patches = [json.loads(line) for line in (out/'patches.jsonl').read_text(encoding='utf-8').splitlines()] \
        if (out/'patches.jsonl').exists() else []
    recognition = out/'output/recognition_config.json'
    reference = json.loads((SAVED/'recognition_config.json').read_text(encoding='utf-8'))
    observed = json.loads(recognition.read_text(encoding='utf-8')) if recognition.exists() else None
    # コマ落ちに強い状態判定の鍵 (ON の腕だけ。23y の ALL) は別に記録し、残りを 21c と比べる。OFF の腕には鍵が無い。
    extra = {key: observed.pop(key) for key in FDR_RECOGNITION_KEYS if observed and key in observed}
    # 差し替えを入れた process 数。入力の差し替えは評価 process と認識の子 process の両方 (>=2) が要る。
    receipt = dict(name=run['name'], arm=run['arm'], exit_code=code, overlay_modules=run['child']['source_modules'],
                   patches={name: len({p['pid'] for p in patches if p['patch'] == name})
                            for name in ('video_source', 'recognition_config', 'launcher')},
                   recognition_matches_21c=observed is not None and observed == reference,
                   recognition_extra_keys=extra)
    (out/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


def execute(args: argparse.Namespace) -> None:
    authorize(args.token, os.environ)
    if (reason := blocked()) is not None:
        raise SystemExit(f'起動しない: {reason}')
    if digest(VIDEO) != VIDEO_SHA256:
        raise SystemExit('録画の SHA256 が転送票と違う')
    runs = plan(args)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root/f'plan_{args.condition}.json').write_text(json.dumps(runs, ensure_ascii=False, indent=2),
                                                        encoding='utf-8')
    for run in runs:
        receipt = run_one(run)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        if (receipt['exit_code'] or receipt['patches']['video_source'] < MIN_SOURCE_PATCHED_PROCESSES
                or not receipt['recognition_matches_21c']):
            raise SystemExit(f"{run['name']} が不合格: 原票を残して止める")


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--condition', choices=tuple(CONDITIONS), default='L')
    parser.add_argument('--root', type=Path, default=Path('D:/puyo_analyzer/live_x_stage01_20261022/replay'))
    parser.add_argument('--arms', type=lambda s: tuple(s.split(',')), default=ARMS)
    parser.add_argument('--start', type=float, default=OFFSET_SEC, help='記録秒。試合開始を含めること (原則2b)')
    parser.add_argument('--end', type=float, help='記録秒。省略時は録画の終わりまで')
    parser.add_argument('--token')
    args = parser.parse_args(argv)
    if any(arm not in ('OFF', 'ON') for arm in args.arms):
        parser.error('--arms は OFF/ON のカンマ区切り')
    return args


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    if args.command == 'plan':
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))
        return
    sys.path.insert(0, str(WORKTREE))
    execute(args)


if __name__ == '__main__':
    main()
