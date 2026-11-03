"""24hの起動設定。運転中の制御ファイルは読まず、子へ固定値を渡す。"""
from __future__ import annotations

import argparse
import math
import os
import sysconfig
from pathlib import Path
from typing import Any

from src.production_config import LIVE_24H_DEFAULT
from .live_power import configure_power, power_enabled
from .live_chain_split_config import configure_split, split_enabled, validate_split
from .live_ex_fresh import EX_FRESH_SEC_DEFAULT, validate_ex_fresh_sec
from .live_retr_v2_config import retr_v2_settings, retr_v2_values, configure_retr_v2

ENV_ENABLED = 'PUYO_LIVE_24H'
ENV_AMA = 'PUYO_LIVE_24H_AMA'
ENV_RETR = 'PUYO_LIVE_24H_RETR'
ENV_XC = 'PUYO_LIVE_24H_XC'
ENV_RETR_SECOND = 'PUYO_LIVE_24H_RETR_SECOND'
ENV_XC_SAME_TIME = 'PUYO_LIVE_24H_XC_SAME_TIME'
ENV_TAPE_NEXT = 'PUYO_LIVE_24H_TAPE_NEXT'   # live_ex_capture.ENV_TAPE_NEXT と同じ名前
ENV_EX_FRESH_SEC = 'PUYO_LIVE_24H_EX_FRESH_SEC'
ENV_MEDIA_FRESH = 'PUYO_LIVE_24H_MEDIA_FRESH'
ENV_ROW_TIME = 'PUYO_LIVE_24H_ROW_TIME'
ENV_EARLY_SUBMIT = 'PUYO_LIVE_24H_EARLY_SUBMIT'
ENV_EX_SPEED = 'PUYO_LIVE_24H_EX_SPEED'
ENV_EX_FIRE_FIRST = 'PUYO_LIVE_24H_EX_FIRE_FIRST'
ENV_STABLE_KEY = 'PUYO_LIVE_24H_STABLE_KEY'
ENV_RETR_H_FORECAST = 'PUYO_LIVE_24H_RETR_H_FORECAST'
ENV_DELAY = 'PUYO_LIVE_24H_DELAY_MS'
ENV_MODEL = 'PUYO_LIVE_24H_MODEL'
LOGICAL_DELAY_MS_DEFAULT = 200.0  # PC2のL200Hで確定。
WORK_LIMIT = 450
CONFIG_KEYS = ('live_disable_power_throttling', 'live_chain_split_place', 'live_24h', 'live_24h_ama', 'live_24h_retr', 'live_24h_xc', 'live_24h_retr_second', 'live_24h_retr_move050', 'live_24h_retr_deadline_extend', 'live_24h_xc_same_time', 'live_24h_tape_next', 'live_24h_delay_ms',
               'live_24h_model', 'live_24h_media_fresh', 'live_24h_row_time', 'live_24h_retr_h_forecast',
               'live_24h_early_submit', 'live_24h_ex_speed', 'live_24h_ex_fire_first', 'live_24h_stable_key', 'live_24h_ex_fresh_sec')


def enabled() -> bool:
    return os.environ.get(ENV_ENABLED, str(int(LIVE_24H_DEFAULT))) == '1'


def ama_enabled() -> bool:
    return enabled() and os.environ.get(ENV_AMA, '1') == '1'


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """配布の既定JSONを変えず、開発CLIと入力JSONに設定口を設ける。"""
    if 'live_24h' in {action.dest for action in parser._actions}:
        return
    parser.add_argument('--live-disable-power-throttling', action=argparse.BooleanOptionalAction, default=power_enabled())
    parser.add_argument('--live-24h', action=argparse.BooleanOptionalAction, default=LIVE_24H_DEFAULT)
    parser.add_argument('--live-24h-ama', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--live-24h-delay-ms', type=float, default=LOGICAL_DELAY_MS_DEFAULT)
    parser.add_argument('--live-24h-model', type=Path)
    parser.add_argument('--live-24h-retr', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--live-24h-xc', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--live-chain-split-place', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--live-24h-retr-second', action=argparse.BooleanOptionalAction, default=False)
    from .live_retr_v2_config import add_arguments as add_retr_v2_arguments
    add_retr_v2_arguments(parser)
    parser.add_argument('--live-24h-xc-same-time', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--live-24h-tape-next', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--live-24h-media-fresh', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--live-24h-ex-fresh-sec', type=float, default=EX_FRESH_SEC_DEFAULT)
    parser.add_argument('--live-24h-row-time', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--live-24h-early-submit', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--live-24h-ex-speed', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--live-24h-ex-fire-first', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--live-24h-stable-key', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--live-24h-retr-h-forecast', action=argparse.BooleanOptionalAction, default=False)


def configure(options: Any, parser: Any) -> None:
    """省電力の設定は24hの有効無効と独立に子へ渡す。"""
    try:
        configure_power(getattr(options, 'live_disable_power_throttling', power_enabled()))
    except ValueError as exc:
        parser.error(str(exc))
    _configure_24h(options, parser)


def _configure_24h(options: Any, parser: Any) -> None:
    """旧欄を起動前に止める。OFFは既存設定もEX資産も参照しない。"""
    active = getattr(options, 'live_24h', LIVE_24H_DEFAULT)
    if type(active) is not bool:
        parser.error('live_24hはboolです')
    retr = getattr(options, 'live_24h_retr', False)
    if type(retr) is not bool:
        parser.error('live_24h_retrはboolです')
    try:
        xc = validate_xc(getattr(options, 'live_24h_xc', False), retr)
        same_time = validate_xc_same_time(getattr(options, 'live_24h_xc_same_time', False), xc)
    except ValueError as exc:
        parser.error(str(exc))
    tape_next = getattr(options, 'live_24h_tape_next', False)
    if type(tape_next) is not bool:
        parser.error('live_24h_tape_nextはboolです')
    os.environ[ENV_XC] = str(int(active and xc))
    os.environ[ENV_XC_SAME_TIME] = str(int(active and same_time))
    configure_split(getattr(options, 'live_chain_split_place', False), active, parser)
    os.environ[ENV_TAPE_NEXT] = str(int(active and tape_next))
    media_fresh = getattr(options, 'live_24h_media_fresh', False)
    if type(media_fresh) is not bool:
        parser.error('live_24h_media_freshはboolです')
    os.environ[ENV_MEDIA_FRESH] = str(int(active and media_fresh))
    configure_ex_options(options, parser, active, retr)
    os.environ[ENV_RETR] = str(int(active and retr))
    os.environ[ENV_ENABLED] = str(int(active))
    if not active:
        return
    use_ama = getattr(options, 'live_24h_ama', True)
    delay = getattr(options, 'live_24h_delay_ms', LOGICAL_DELAY_MS_DEFAULT)
    if type(use_ama) is not bool:
        parser.error('live_24h_amaはboolです')
    if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
        parser.error('live_24h_delay_msは0以上の有限なmsです')
    required = ('b3_judgment', 'b3_async', 'b3_m3', 'b3_c2')
    if not all(getattr(options, name, False) is True for name in required):
        parser.error('live_24hにはb3_judgment・b3_async・b3_m3・b3_c2が必要です')
    for name in ('live_v49f', 'live_ama_plain', 'live_winrate_switch', 'live_x_prefire'):
        setattr(options, name, False)
        os.environ['PUYO_' + name.upper()] = '0'
    os.environ[ENV_AMA], os.environ[ENV_DELAY] = str(int(use_ama)), str(delay)
    if use_ama:
        path = getattr(options, 'live_24h_model', None)
        if path is None or not Path(path).is_file():
            parser.error('live_24h_modelにC2+EX ref3マニフェストが必要です')
        options.live_24h_model = Path(path).resolve()
        os.environ[ENV_MODEL] = str(options.live_24h_model)
        os.environ['PUYO_EX_WORK_BUDGET'] = '1'
        os.environ['PUYO_EX_WORK_LIMIT'] = str(WORK_LIMIT)


def ex_settings() -> dict:
    from .live_ex_runtime import settings_from_env
    assets = Path(__file__).resolve().parents[2]/'models/live_ex'
    defaults = dict(policy=str(assets/'policy_locked.json'), seed_table=str(assets/'seed_table.npz'),
                    native_dir=sysconfig.get_path('purelib'))
    return dict(settings_from_env(defaults), **({'chain_split_place': True} if split_enabled() else {}), logical_delay_ms=float(os.environ.get(
        ENV_DELAY, LOGICAL_DELAY_MS_DEFAULT)), work_limit=WORK_LIMIT,
        retr=ama_enabled() and os.environ.get(ENV_RETR) == '1',
        xc=ama_enabled() and validate_xc(os.environ.get(ENV_XC) == '1', os.environ.get(ENV_RETR) == '1'),
        **retr_v2_settings(ama_enabled()),
        retr_second=ama_enabled() and validate_retr_second(
            os.environ.get(ENV_RETR_SECOND) == '1', os.environ.get(ENV_RETR) == '1'),
        xc_same_time=ama_enabled() and validate_xc_same_time(
            os.environ.get(ENV_XC_SAME_TIME) == '1', os.environ.get(ENV_XC) == '1'),
        tape_next=ama_enabled() and os.environ.get(ENV_TAPE_NEXT) == '1',
        media_fresh=ama_enabled() and os.environ.get(ENV_MEDIA_FRESH) == '1',
        ex_fresh_sec=validate_ex_fresh_sec(float(os.environ.get(ENV_EX_FRESH_SEC, EX_FRESH_SEC_DEFAULT))),
        row_time=ama_enabled() and os.environ.get(ENV_ROW_TIME) == '1',
        early_submit=ama_enabled() and os.environ.get(ENV_EARLY_SUBMIT) == '1',
        ex_speed=ama_enabled() and os.environ.get(ENV_EX_SPEED) == '1',
        ex_fire_first=ama_enabled() and os.environ.get(ENV_EX_FIRE_FIRST) == '1',
        stable_key=ama_enabled() and os.environ.get(ENV_STABLE_KEY) == '1',
        retr_h_forecast=ama_enabled() and validate_retr_h_forecast(
            os.environ.get(ENV_RETR_H_FORECAST) == '1', os.environ.get(ENV_RETR) == '1'))


def launcher_values(data: dict, base: Path) -> dict:
    """利用者JSONの設定口。検証時はprocessの環境を書き換えない。"""
    active = data.get('live_24h', LIVE_24H_DEFAULT)
    if type(active) is not bool:
        raise ValueError('live_24hはboolです')
    if not active:
        return {}
    retr = data.get('live_24h_retr', False)
    if type(retr) is not bool:
        raise ValueError('live_24h_retrはboolです')
    xc = validate_xc(data.get('live_24h_xc', False), retr)
    h_forecast = validate_retr_h_forecast(data.get('live_24h_retr_h_forecast', False), retr)
    tape_next = data.get('live_24h_tape_next', False)
    if type(tape_next) is not bool:
        raise ValueError('live_24h_tape_nextはboolです')
    media_fresh = data.get('live_24h_media_fresh', False)
    if type(media_fresh) is not bool:
        raise ValueError('live_24h_media_freshはboolです')
    ama = data.get('live_24h_ama', True)
    delay = data.get('live_24h_delay_ms', LOGICAL_DELAY_MS_DEFAULT)
    if type(ama) is not bool:
        raise ValueError('live_24h_amaはboolです')
    if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
        raise ValueError('live_24h_delay_msは0以上の有限なmsです')
    if not all(data.get(name) is True for name in ('b3_judgment', 'b3_async', 'b3_m3', 'b3_c2')):
        raise ValueError('live_24hにはb3_judgment・b3_async・b3_m3・b3_c2が必要です')
    result = dict(live_24h=True, live_24h_xc=xc, live_24h_tape_next=tape_next, live_24h_retr=retr, live_24h_ama=ama, live_24h_delay_ms=float(delay),
                  live_chain_split_place=validate_split(data.get('live_chain_split_place', False)),
                  live_24h_media_fresh=media_fresh,
                  **retr_v2_values(data, retr),
                  live_24h_retr_second=validate_retr_second(data.get('live_24h_retr_second', False), retr),
                  live_24h_xc_same_time=validate_xc_same_time(data.get('live_24h_xc_same_time', False), xc),
                  live_24h_ex_fresh_sec=validate_ex_fresh_sec(data.get('live_24h_ex_fresh_sec', EX_FRESH_SEC_DEFAULT)),
                  live_24h_row_time=validate_row_time(data.get('live_24h_row_time', False)),
                  live_24h_early_submit=validate_early_submit(data.get('live_24h_early_submit', False)),
                  live_24h_ex_speed=validate_ex_speed(data.get('live_24h_ex_speed', False)),
                  live_24h_ex_fire_first=validate_ex_fire_first(data.get('live_24h_ex_fire_first', False)),
                  live_24h_stable_key=validate_stable_key(data.get('live_24h_stable_key', False)),
                  live_24h_retr_h_forecast=h_forecast,
                  live_v49f=False, live_ama_plain=False, live_winrate_switch=False, live_x_prefire=False)
    if ama:
        raw = data.get('live_24h_model')
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError('live_24h_modelにC2+EX ref3マニフェストが必要です')
        path = Path(raw)
        path = (path if path.is_absolute() else base / path).resolve()
        if not path.is_file():
            raise ValueError(f'live_24h_modelがありません: {path}')
        result['live_24h_model'] = path
    return result


def pipeline_values(user: Any) -> dict:
    """OFFではキー自体を増やさず、既存の生成JSONを維持する。"""
    if not user.live_24h:
        return {}
    result = dict(live_24h=True, live_24h_retr=getattr(user, 'live_24h_retr', False), live_24h_ama=user.live_24h_ama,
                  live_24h_delay_ms=user.live_24h_delay_ms,
                  **retr_v2_values(vars(user), getattr(user, 'live_24h_retr', False)),
                  live_24h_retr_second=validate_retr_second(getattr(user, 'live_24h_retr_second', False),
                                                          getattr(user, 'live_24h_retr', False)),
                  live_24h_xc_same_time=validate_xc_same_time(getattr(user, 'live_24h_xc_same_time', False),
                                                          getattr(user, 'live_24h_xc', False)),
                  live_24h_xc=validate_xc(getattr(user, 'live_24h_xc', False),
                                          getattr(user, 'live_24h_retr', False)),
                  live_24h_tape_next=getattr(user, 'live_24h_tape_next', False),
                  live_24h_media_fresh=getattr(user, 'live_24h_media_fresh', False),
                  live_24h_ex_fresh_sec=validate_ex_fresh_sec(getattr(user, 'live_24h_ex_fresh_sec', EX_FRESH_SEC_DEFAULT)),
                  live_24h_row_time=validate_row_time(getattr(user, 'live_24h_row_time', False)),
                  live_24h_early_submit=validate_early_submit(getattr(user, 'live_24h_early_submit', False)),
                  live_24h_ex_speed=validate_ex_speed(getattr(user, 'live_24h_ex_speed', False)),
                  live_24h_ex_fire_first=validate_ex_fire_first(getattr(user, 'live_24h_ex_fire_first', False)),
                  live_24h_stable_key=validate_stable_key(getattr(user, 'live_24h_stable_key', False)),
                  live_24h_retr_h_forecast=validate_retr_h_forecast(getattr(user, 'live_24h_retr_h_forecast', False),
                                                                    getattr(user, 'live_24h_retr', False)))
    if validate_split(getattr(user, 'live_chain_split_place', False)):
        result['live_chain_split_place'] = True
    if user.live_24h_ama:
        result['live_24h_model'] = str(user.live_24h_model)
    return result


def validate_retr_second(value: bool, retr: bool) -> bool:
    if type(value) is not bool:
        raise ValueError('live_24h_retr_secondはboolです')
    if value and not retr:
        raise ValueError('live_24h_retr_secondにはlive_24h_retrが必要です')
    return value


def configure_retr_second(options: Any, parser: Any, active: bool, retr: bool) -> None:
    try:
        value = validate_retr_second(getattr(options, 'live_24h_retr_second', False), retr)
    except ValueError as exc:
        parser.error(str(exc))
    os.environ[ENV_RETR_SECOND] = str(int(active and value))


def validate_xc_same_time(value: bool, xc: bool = True) -> bool:
    if type(value) is not bool:
        raise ValueError('live_24h_xc_same_timeはboolです')
    if value and not xc:
        raise ValueError('live_24h_xc_same_timeにはlive_24h_xcが必要です')
    return value


def validate_xc(xc: bool, retr: bool) -> bool:
    """xc は連鎖の予測を持つ retR と一緒にだけ有効にする。"""
    if type(xc) is not bool:
        raise ValueError('live_24h_xcはboolです')
    if xc and not retr:
        raise ValueError('live_24h_xcにはlive_24h_retrが必要です')
    return xc


def validate_retr_h_forecast(value: bool, retr: bool) -> bool:
    """依頼 F3: retR の持ち時間 h を共通の連鎖の予測から作る (EX の子に PUYO_RETR_H_FORECAST)。retR と一緒にだけ有効。"""
    if type(value) is not bool:
        raise ValueError('live_24h_retr_h_forecastはboolです')
    if value and not retr:
        raise ValueError('live_24h_retr_h_forecastにはlive_24h_retrが必要です')
    return value


def validate_row_time(value: bool) -> bool:
    if type(value) is not bool:
        raise ValueError('live_24h_row_timeはboolです')
    return value


def configure_row_time(options: Any, parser: Any, active: bool) -> None:
    try:
        value = validate_row_time(getattr(options, 'live_24h_row_time', False))
    except ValueError as exc:
        parser.error(str(exc))
    os.environ[ENV_ROW_TIME] = str(int(active and value))


def validate_early_submit(value: bool) -> bool:
    if type(value) is not bool:
        raise ValueError('live_24h_early_submitはboolです')
    return value


def configure_early_submit(options: Any, parser: Any, active: bool) -> None:
    try:
        value = validate_early_submit(getattr(options, 'live_24h_early_submit', False))
    except ValueError as exc:
        parser.error(str(exc))
    os.environ[ENV_EARLY_SUBMIT] = str(int(active and value))


def configure_retr_h_forecast(options: Any, parser: Any, active: bool, retr: bool) -> None:
    try:
        value = validate_retr_h_forecast(getattr(options, 'live_24h_retr_h_forecast', False), retr)
    except ValueError as exc:
        parser.error(str(exc))
    os.environ[ENV_RETR_H_FORECAST] = str(int(active and value))


def validate_ex_speed(value: bool) -> bool:
    if type(value) is not bool:
        raise ValueError('live_24h_ex_speedはboolです')
    return value


def configure_ex_speed(options: Any, parser: Any, active: bool) -> None:
    try:
        value = validate_ex_speed(getattr(options, 'live_24h_ex_speed', False))
    except ValueError as exc:
        parser.error(str(exc))
    os.environ[ENV_EX_SPEED] = str(int(active and value))


def configure_ex_options(options: Any, parser: Any, active: bool, retr: bool) -> None:
    """EXの起動設定を同じ順序で検証し、子へ渡す。"""
    configure_retr_second(options, parser, active, retr)
    configure_retr_v2(options, parser, active, retr)
    configure_ex_fresh_sec(options, parser, active)
    configure_row_time(options, parser, active)
    configure_early_submit(options, parser, active)
    configure_ex_speed(options, parser, active)
    configure_ex_fire_first(options, parser, active)
    configure_stable_key(options, parser, active)
    configure_retr_h_forecast(options, parser, active, retr)


def configure_ex_fresh_sec(options: Any, parser: Any, active: bool) -> None:
    try:
        value = validate_ex_fresh_sec(getattr(options, 'live_24h_ex_fresh_sec', EX_FRESH_SEC_DEFAULT))
    except ValueError as exc:
        parser.error(str(exc))
    os.environ[ENV_EX_FRESH_SEC] = str(value if active else EX_FRESH_SEC_DEFAULT)


def validate_ex_fire_first(value: bool) -> bool:
    if type(value) is not bool:
        raise ValueError('live_24h_ex_fire_firstはboolです')
    return value


def configure_ex_fire_first(options: Any, parser: Any, active: bool) -> None:
    try:
        value = validate_ex_fire_first(getattr(options, 'live_24h_ex_fire_first', False))
    except ValueError as exc:
        parser.error(str(exc))
    os.environ[ENV_EX_FIRE_FIRST] = str(int(active and value))


def validate_stable_key(value: bool) -> bool:
    if type(value) is not bool:
        raise ValueError('live_24h_stable_keyはboolです')
    return value


def configure_stable_key(options: Any, parser: Any, active: bool) -> None:
    try:
        value = validate_stable_key(getattr(options, 'live_24h_stable_key', False))
    except ValueError as exc:
        parser.error(str(exc))
    os.environ[ENV_STABLE_KEY] = str(int(active and value))
