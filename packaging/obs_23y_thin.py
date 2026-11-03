"""間引きの台 (L1_OFF・half・third・obs21c) で、23y の ALL の最終構成を 30Hz レビューと比べる起動口 (重い段 (c))。

FDR 担当の台 (claude/live-thin-fdr-a-20261023、2b44dd32) と物差しをそのまま使う。台は使い捨ての複写 (<work>/wt_thin_y) に、
(1) preset all (X + コマ落ちに強い状態判定 + 予告 CNN の試作 2 つ)、(2) 配布 app が読む run_live_pipeline_20260928.py の差し替えだけを
足して使う。差し替える src は --source-root で 23y の木を指す。各パターンで OFF と ON(ALL) を別々に流す (T1_OFF・T1_ON)。
  run <パターン> <OFF|ON> … 1 本流す (空きメモリ 10GiB 未満なら起動しない)
  compare <パターン>      … scripts.compare_live_thin_replay で 30Hz レビューと突き合わせる (読み取りのみ)
  summary                … 4 パターンの結果を FDR 担当の受領票 (OFF・足す前の版・R15 だけ) と並べて判定する
判定 (FDR 担当の規則): ALL が「足す前の版 (R35 まで)」より平均 0.05pt を超えて悪くならない。測定器の健全性として、
自分の OFF が FDR 担当の OFF を平均 0.1pt 以内で再現すること (再現しなければ台が壊れている疑い)。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

Y_TREE = Path(__file__).resolve().parents[1]
THIN_TREE = Path(_public_dev_text('<work>/wt_thin_y'))
RUNTIME = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive/python/python.exe'))
# 比較 (読み取りのみ) は台の複写の scripts を import するので、cwd を sys.path に足すシステムの python で回す。
# 配布の実行系は ._pth で 19a の app の scripts を先に見てしまい、複写の scripts を読めない。
COMPARE_PYTHON = shutil.which('python') or 'python'
WORK = Path('D:/puyo_analyzer/packaging/build_20261023_obs_y/work/thin')
RECEIPT = Path('D:/puyo_analyzer/frame_drop_a_20261023/RECEIPT_R06_R10_R15.json')
PATTERNS = ('L1_OFF', 'half', 'third', 'obs21c')
ARMS = ('OFF', 'ON')
TOKEN, CONFIRM_ENV, CONFIRM_WORD = 'RUN-LIVE-X-REPLAY', 'LIVE_X_REPLAY_CONFIRM', 'PC1-M4-QUIET'
MIN_FREE_GIB = 10.
BELOW_NORMAL = 0x00004000
WORSE_LIMIT_PT = .05          # FDR 担当の規則: 足す前の版より平均がこれを超えて悪くなる箇所は外す
INSTRUMENT_TOLERANCE_PT = .1  # 自分の OFF が FDR 担当の OFF を再現する幅
PASS_LINE = dict(mean_pt=2., p95_pt=10., over_10pt_ratio=.05)   # 設計の合格線 (間引きの台・30Hz レビューとの差)
REFERENCES = ('OFF', 'before(R35まで)', '+R15 だけ')
COMPARE_FILE = 'compare_fdr.json'


def free_gib() -> float:
    import ctypes
    from ctypes import wintypes

    class Status(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD), ('total', ctypes.c_ulonglong),
                    ('avail', ctypes.c_ulonglong), ('p1', ctypes.c_ulonglong), ('p2', ctypes.c_ulonglong),
                    ('v1', ctypes.c_ulonglong), ('v2', ctypes.c_ulonglong), ('e', ctypes.c_ulonglong)]
    status = Status()
    status.length = ctypes.sizeof(status)
    ctypes.WinDLL('kernel32').GlobalMemoryStatusEx(ctypes.byref(status))
    return status.avail/2**30


def run_command(pattern: str, arm: str) -> list[str]:
    return [str(RUNTIME), '-B', 'scripts/run_live_thin_replay.py', 'run', '--pattern', pattern, '--arms', arm, '--on', 'all',
            '--source-root', str(Y_TREE), '--root', str(WORK/pattern), '--token', TOKEN]


def run_one(pattern: str, arm: str) -> int:
    free = free_gib()
    if free < MIN_FREE_GIB:
        raise SystemExit(f'空きメモリ {free:.1f}GiB < {MIN_FREE_GIB}GiB: 起動しない')
    WORK.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **{CONFIRM_ENV: CONFIRM_WORD}, PYTHONIOENCODING='utf-8')
    started = time.time()
    with (WORK/f'{pattern}_{arm}.log').open('wb') as log:
        done = subprocess.run(run_command(pattern, arm), cwd=THIN_TREE, env=env, stdout=log, stderr=subprocess.STDOUT,
                              creationflags=BELOW_NORMAL)
    (WORK/f'{pattern}_{arm}.exit.json').write_text(json.dumps(dict(
        exit_code=done.returncode, seconds=round(time.time()-started, 1), free_gib_at_start=round(free, 2))), encoding='utf-8')
    return done.returncode


def compare_command(pattern: str) -> list[str]:
    """FDR 担当の突き合わせ台 (compare_fdr_a.py thin): OFF と ON の腕を、30Hz レビューと同じコマで試合ごと・全体で比べる。"""
    return [COMPARE_PYTHON, '-B', '-m', 'scripts.compare_fdr_a', 'thin', '--off', str(WORK/pattern/'T1_OFF'),
            '--on', str(WORK/pattern/'T1_ON'), '--out', str(WORK/pattern/COMPARE_FILE)]


def compare_one(pattern: str) -> int:
    env = dict(os.environ, PYTHONPATH=str(THIN_TREE), PYTHONIOENCODING='utf-8')   # 配布の実行系は cwd を sys.path に足さない
    done = subprocess.run(compare_command(pattern), cwd=THIN_TREE, env=env, capture_output=True, text=True, encoding='utf-8')
    (WORK/f'{pattern}_compare.log').write_text(done.stdout+done.stderr, encoding='utf-8')
    return done.returncode


def range_stats(compare: dict, arm: str) -> dict:
    """30Hz レビューとの差 (全試合の範囲。FDR 担当の台 compare_fdr_a.py thin の groups.range = 受領票の range と同じ母数)。"""
    stats = compare['groups']['range']['vs_review'][arm]
    return dict(n=stats['n'], mean=round(stats['mean_pt'], 3), p95=round(stats['p95_pt'], 2),
                over_10pt=round(stats['over_10pt']['ratio'], 4), identical=stats.get('identical'))


def judge(pattern: str, mine: dict, theirs: dict) -> dict:
    """1 パターンの判定。mine = {OFF, ALL} (自分の run)、theirs = FDR 担当の受領票の同じパターンの値 {基準名: range}。"""
    off, all_ = mine['OFF'], mine['ALL']
    reference_off = theirs.get('OFF')
    before = theirs.get('before(R35まで)')
    result = dict(pattern=pattern, mine=mine, theirs=theirs, n_match=off['n'] == all_['n'] == (reference_off or {}).get('n'))
    result['instrument_reproduces_off'] = None if reference_off is None else abs(off['mean']-reference_off['mean']) <= INSTRUMENT_TOLERANCE_PT
    result['delta_vs_my_off_pt'] = round(all_['mean']-off['mean'], 3)
    result['delta_vs_before_pt'] = None if before is None else round(all_['mean']-before['mean'], 3)
    result['not_worse_than_before'] = None if before is None else all_['mean']-before['mean'] <= WORSE_LIMIT_PT
    result['better_than_off'] = all_['mean'] < off['mean']
    result['pass_line'] = dict(mean=all_['mean'] <= PASS_LINE['mean_pt'], p95=all_['p95'] <= PASS_LINE['p95_pt'],
                               over_10pt=all_['over_10pt'] <= PASS_LINE['over_10pt_ratio'])
    return result


def summary() -> dict:
    receipt = json.loads(RECEIPT.read_text(encoding='utf-8'))['stage2b']
    rows = []
    for pattern in PATTERNS:
        compare = json.loads((WORK/pattern/COMPARE_FILE).read_text(encoding='utf-8'))
        mine = dict(OFF=range_stats(compare, 'OFF'), ALL=range_stats(compare, 'ON'))
        theirs = {name: receipt[name][pattern]['range'] for name in REFERENCES if pattern in receipt.get(name, {})}
        rows.append(judge(pattern, mine, {k: dict(n=v['n'], mean=v['mean'], p95=v['p95'], over_10pt=v['over_10pt'])
                                          for k, v in theirs.items()}))
    result = dict(rows=rows, worse_limit_pt=WORSE_LIMIT_PT, instrument_tolerance_pt=INSTRUMENT_TOLERANCE_PT,
                  passed=all(r['instrument_reproduces_off'] is not False and r['not_worse_than_before'] is not False
                             and r['better_than_off'] for r in rows))
    (WORK/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('step', choices=('run', 'compare', 'summary', 'plan'))
    parser.add_argument('pattern', nargs='?', choices=PATTERNS)
    parser.add_argument('arm', nargs='?', choices=ARMS)
    args = parser.parse_args()
    if args.step == 'plan':
        print(json.dumps({f'{p}_{a}': run_command(p, a) for p in PATTERNS for a in ARMS}, indent=1))
    elif args.step == 'run':
        sys.exit(run_one(args.pattern, args.arm))
    elif args.step == 'compare':
        sys.exit(compare_one(args.pattern))
    else:
        result = summary()
        print(json.dumps(result, ensure_ascii=False, indent=1))
        sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
