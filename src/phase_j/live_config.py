"""ライブ実行設定。CLI明示値をJSONより優先し、旧引数も維持する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PATH_KEYS = frozenset({'video', 'status', 'output', 'input_config'})
LIVE_DURATION_SEC = 24*60*60


def apply_config(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    defaults = json.loads(Path('config/live_defaults.json').read_text(encoding='utf-8'))
    # 比較用threadモードは認識と評価を同一processで実行するためniceを変えない。
    if '--worker-mode=thread' in argv or any(a == '--worker-mode' and argv[i+1:i+2] == ['thread']
                                            for i, a in enumerate(argv)):
        defaults['evaluation_nice'] = 0
    parser.set_defaults(**defaults)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--config', type=Path)
    config, _ = probe.parse_known_args(argv)
    if config.config:
        data = json.loads(config.config.read_text(encoding='utf-8'))
        allowed = {action.dest for action in parser._actions}
        unknown = set(data)-allowed-{'name', 'index', 'verification_only', 'color_correction',
                                  'relaxed_verify', 'verify_fail_streak'}
        if unknown:
            parser.error(f'未知の設定項目: {sorted(unknown)}')
        defaults = {key: Path(value) if key in PATH_KEYS else value
                    for key, value in data.items() if key in allowed}
        parser.set_defaults(**defaults)
    options = parser.parse_args(argv)
    if options.source not in (None, 'video', 'dshow') or options.cnn_device not in ('auto', 'cpu'):
        parser.error('sourceまたはcnn_deviceが不正です')
    if options.source == 'video' and options.input_config:
        parser.error('videoとinput_configは同時指定できません')
    if options.source == 'dshow':
        options.input_config = options.input_config or options.config
        if options.input_config is None:
            parser.error('dshowには機器name/indexを持つ--configが必要です')
        options.realtime = True
        options.duration_sec = options.duration_sec or LIVE_DURATION_SEC
    if options.source is None:
        options.source = 'dshow' if options.input_config else 'video'
    explicit_source = any(arg == '--source' or arg.startswith('--source=') for arg in argv)
    options.lifecycle = bool(options.config or explicit_source) and not options.compare
    if type(options.mc_rollouts) is not int or options.mc_rollouts < 1:
        parser.error('mc_rolloutsは正の整数が必要です')
    if options.lifecycle and options.worker_mode == 'thread':
        parser.error('統合ライブ入力はprocessモード専用です')
    if options.cnn_device == 'cpu':
        # torchを読み込むより前に設定し、認識spawn先へ継承する。
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    return options


def input_identity(config: str | None, video: str) -> Any:
    from .live_device import DeviceConfig
    if config:
        data = json.loads(Path(config).read_text(encoding='utf-8'))
        return DeviceConfig(data.get('name', Path(video).stem), data.get('index', 0), True,
                            relaxed_verify=data.get('relaxed_verify', False))
    return DeviceConfig(Path(video).stem, 0, True)
