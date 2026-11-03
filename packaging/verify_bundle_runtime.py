"""配布ランチャーから描画CLIまでの実効本番引数とテロップ資産を照合する。"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
from unittest.mock import patch


class CapturedOptions(Exception):
    """実パーサーで照合後、動画処理を開始せずに終了する。"""


def effective_options(app: Path, output: Path, user_config: Path | None = None) -> dict:
    """本番フラグの現在値を実パーサーの型・destで比較する。"""
    from src import exchange_event_cli, production_config
    from src.phase_j import launcher
    from scripts import run_live_pipeline_20260928 as live
    from scripts import visualize_advantage_overlay as render
    data = (json.loads(user_config.read_text(encoding='utf-8-sig')) if user_config else
            dict(source='dshow', device_name='OBS Virtual Camera'))
    user = launcher.parse_user_config(data, user_config.parent if user_config else output)
    config = output / 'pipeline_config.json'
    config.write_text(json.dumps(launcher.build_pipeline_config(user)), encoding='utf-8')
    with patch.object(sys, 'argv', ['live', *launcher.build_pipeline_argv(user, config)]):
        options = live.parse_args()
        options.no_render = True
        command = live.build_command(options)
    original = exchange_event_cli.parse_exchange_event_args
    result: dict = dict(command=command, flags={}, live_options=vars(options))
    if user.b3_judgment:
        result['b3'] = check_b3_defaults(user, options, production_config)

    def capture(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> None:
        actual = original(parser, argv)
        flags = shlex.split(production_config.exchange_event_flags())
        flags += shlex.split(production_config.placement_reconcile_flags())
        expected = parser.parse_args(flags)
        for flag in flags:
            if not flag.startswith('--'):
                continue
            dest = parser._option_string_actions[flag].dest
            value = getattr(actual, dest)
            assert value == getattr(expected, dest), (flag, value, getattr(expected, dest))
            result['flags'][flag] = value
        result['effective_options'] = vars(actual)
        raise CapturedOptions

    with patch.object(exchange_event_cli, 'parse_exchange_event_args', capture):
        with patch.object(sys, 'argv', ['render', *command]):
            try:
                render.main()
            except CapturedOptions:
                pass
    assert result['flags'], '実効引数が取得できませんでした'
    return result


def check_b3_defaults(user: object, options: object, production: object) -> dict:
    """配布設定から実CLIへB3と採用済みの表示既定値が届くことを確認する。"""
    expected = dict(b3_judgment=True,
                    display_completion_driven=production.DISPLAY_COMPLETION_DRIVEN_DEFAULT,
                    display_fast_ema=production.DISPLAY_FAST_EMA_DEFAULT)
    for name, value in expected.items():
        assert getattr(user, name) == getattr(options, name) == value, name
    for name in ('b3_model', 'b3_timing_table'):
        assert getattr(user, name) == getattr(options, name), name
    return dict(**expected, model_sha256=hashlib.sha256(user.b3_model.read_bytes()).hexdigest(),
                timing_sha256=hashlib.sha256(user.b3_timing_table.read_bytes()).hexdigest())


def telop_check(app: Path) -> dict:
    """実テンプレートを検索領域へ配置し、読み込みだけでなく陽性検出まで確認する。"""
    import cv2
    import numpy as np
    from src.phase_j.launcher import check_telop_templates
    from src.telop_detector import DEFAULT_TEMPLATE_DIR, SEARCH_X, SEARCH_Y, TelopDetector
    directory = app / DEFAULT_TEMPLATE_DIR
    count = check_telop_templates(app)
    assert count > 0
    detector = TelopDetector.load_default(directory, fast=True)
    positives = []
    for path in sorted(directory.glob('telop_*.png')):
        template = cv2.imread(str(path))
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        height, width = template.shape[:2]
        frame[SEARCH_Y:SEARCH_Y+height, SEARCH_X:SEARCH_X+width] = template
        detected = detector.detect(frame)
        assert detected.is_visible, path
        positives.append(dict(file=path.name, score=detected.score))
    return dict(count=count, synthetic_positives=positives)


def distribution_check(app: Path) -> dict:
    """配布固有の制約と、統合元の本番設定・追加モデルの同一性を確認する。"""
    from src.phase_j.launcher import LauncherConfigError, parse_user_config
    root = Path(__file__).resolve().parents[1]
    paths = ('src/production_config.py', 'models/exchange_event_v1_r1_hgb/G_fe.joblib',
             'models/exchange_event_v1_r1_hgb/manifest.json')
    checks = {name: hashlib.sha256((app / name).read_bytes()).hexdigest() ==
              hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
    site = app.parent / 'python/Lib/site-packages'
    checks['ffmpeg_absent'] = not list(site.glob('cv2/opencv_videoio_ffmpeg*.dll'))
    checks['mit_license'] = (app.parent / 'LICENSE').read_bytes() == (root / 'LICENSE').read_bytes()
    try:
        parse_user_config(dict(source='video', video_path='test.mp4'), app)
        checks['video_rejected'] = False
    except LauncherConfigError:
        checks['video_rejected'] = True
    assert all(checks.values()), checks
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--user-config', type=Path)
    args = parser.parse_args()
    app, output = args.app.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(app))
    os.chdir(app)
    with redirect_stdout(io.StringIO()):
        result = effective_options(app, output, args.user_config)
        result['telop'] = telop_check(app)
        result['distribution'] = distribution_check(app)
    (output / 'runtime.json').write_text(json.dumps(result, default=str, ensure_ascii=False, indent=2),
                                       encoding='utf-8')
    print(f"照合成功: 本番引数 {len(result['flags'])} 項目、テロップ {result['telop']['count']} 枚")


if __name__ == '__main__':
    main()
