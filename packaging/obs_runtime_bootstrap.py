"""検収済み実行系で配布本体を読む。ソースの重ね合わせやC:へのコピーはしない。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy
import sys
import traceback


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--camera', action='store_true')
    parser.add_argument('--video', action='store_true')
    return parser.parse_args()


def prepare(bundle: Path, runtime: Path) -> dict:
    """旧C: appを探索先から外し、製品コードは必ず今回の配布appへ向ける。"""
    if Path(sys.executable).resolve().parent != runtime.resolve():
        raise ValueError('照合対象と実際に起動したPythonが一致しない')
    if 'src' in sys.modules:
        raise ValueError('配布appを選ぶ前に別のsrcが読み込まれた')
    sys.path.insert(0, str(bundle))
    from obs_runtime_check import validate
    result = validate(bundle, runtime)
    old_app = (runtime.parent/'app').resolve()
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != old_app]
    sys.path.insert(0, str(bundle/'app'))
    result.update(app=str(bundle/'app'), dont_write_bytecode=sys.dont_write_bytecode)
    if not sys.dont_write_bytecode:
        raise RuntimeError('-Bを付けて起動してください')
    return result


def main(args: argparse.Namespace) -> None:
    value = prepare(args.bundle, args.runtime)
    args.receipt.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.camera:
        script = args.bundle/'camera_indices.py'
        sys.argv = [str(script)]
        runpy.run_path(str(script), run_name='__main__')
    elif args.entry:
        script = args.bundle/'start_obs_trial.py'
        sys.argv = [str(script), str(args.entry/'puyo_live.json'), str(args.bundle)]
        runpy.run_path(str(script), run_name='__main__')
    else:
        if args.config is None:
            raise ValueError('設定ファイルが必要')
        sys.argv = ['launcher', '--config', str(args.config), '--require-manifest']
        if args.video:
            sys.argv.append('--dev-allow-video')
        runpy.run_module('src.phase_j.launcher', run_name='__main__')


def run_cli() -> None:
    """launcherの正常なSystemExit(0)は失敗票を生成しない。"""
    options = arguments()
    try:
        main(options)
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code in (None, 0):
            raise
        options.receipt.with_name('runtime_failure.json').write_text(json.dumps(dict(
            exception=type(error).__name__, message=str(error), stack=traceback.format_exc()),
            ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    run_cli()
