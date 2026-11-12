"""配布試験だけで有効にする認識高速化の設定とspawn配送。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
from src.production_config import LIVE_WINRATE_SWITCH_DEFAULT
from src.production_config import (LIVE_COLOR_BUFFER_DEFAULT, LIVE_FRAME_DROP_ROBUST_SITES,
                                   LIVE_LOCKDOWN_SCORE_ZERO_THEN_MOVE_DEFAULT, LIVE_MATCH_END_PAIR_DEFAULT,
                                   LIVE_ONNX_PARALLEL_DEFAULT)

from src.production_config import (B3_SCORE_MATMUL_DEFAULT, B3_WARNING_PARALLEL_DEFAULT,
    B3_WARNING_WORKERS, B3_WARNING_MAX_WORKERS, LIVE_DISABLE_PLACEMENT_RECONCILE_DEFAULT,
    B3_LEDGER_MODE_DEFAULT, B3_LEDGER_LANDING_DEFAULT, B3_SKIP_ICON_READING_DEFAULT,
    B3_SHARE_SCORE_READ_DEFAULT, LIVE_ASYNC_SCREEN_VERIFY_DEFAULT, LIVE_STAGE_TIMING_DEFAULT,
    TERMINAL_HOLD_INPUT_RECOVERY_DEFAULT, LIVE_LATEST_EVALUATION_DEFAULT,
    LIVE_REFERENCE_TIMES_ONLY_DEFAULT, LIVE_BORDER_GRACE_DEFAULT, LIVE_NEAR_FUTURE_CACHE_DEFAULT,
    LIVE_EXACT_CONNECTIVITY_DEFAULT, LIVE_EXACT_NATIVE_DEFAULT, LIVE_PUBLIC_MATCH_NUMBER_DEFAULT,
    LIVE_DELTA_CELLS_DEFAULT, LIVE_CNN_DEDUP_DEFAULT, LIVE_ROW1_GUARD_DEFAULT,
    LIVE_SCORE_ROI_ALIGNMENT_DEFAULT, LIVE_X_DIRECT_PUBLISH_DEFAULT,
    LIVE_X_PREFIRE_SHADOW_DEFAULT, LIVE_X_PREFIRE_DEFAULT, LIVE_FRAME_DROP_ROBUST_DEFAULT,
    B3_WARNING_BATCH_DEFAULT, B3_WARNING_PREFETCH_DEFAULT, LIVE_VERIFY_REASON_LOG_DEFAULT,
    B3_WARNING_ASYNC_DEFAULT,
    LIVE_NATIVE_HSV_MAIN_DEFAULT,
    LIVE_VERIFY_ASYNC_DEFAULT, LIVE_MATCH_END_FEED_ON_HOLD_DEFAULT,
    MATCH_END_YATTA_TEXT_TEMPLATE_DEFAULT, LIVE_CAPTURE_RECONNECT_DEFAULT,
    B3_WORKER_GRACEFUL_STOP_DEFAULT, LIVE_BOARD_VECTORIZED_DEFAULT, LIVE_VERIFY_SOFT_HOLD_DEFAULT,
    LIVE_RECOG_SPEED_PY_DEFAULT, LIVE_RECOG_SPEED_PY_PAIR_LAZY_DEFAULT,
    LIVE_NATIVE_READ_BOARD_DEFAULT, LIVE_VERIFY_ENTRY_FAST_RETRY_DEFAULT,
    LIVE_CHAIN_FORECAST_RECHECK_DEFAULT, LIVE_CHAIN_END_FORMULA_DEFAULT)

SPEED_DEFAULTS = dict(live_verify_read_fast=False,
                     live_disable_placement_reconcile=LIVE_DISABLE_PLACEMENT_RECONCILE_DEFAULT,
                     live_winrate_switch=LIVE_WINRATE_SWITCH_DEFAULT,
                     live_color_buffer=LIVE_COLOR_BUFFER_DEFAULT,
                     live_onnx_parallel=LIVE_ONNX_PARALLEL_DEFAULT,
                     live_public_match_number=LIVE_PUBLIC_MATCH_NUMBER_DEFAULT,
                     live_delta_cells=LIVE_DELTA_CELLS_DEFAULT,
                     live_cnn_dedup=LIVE_CNN_DEDUP_DEFAULT,
                     live_row1_guard=LIVE_ROW1_GUARD_DEFAULT,
                     align_score_roi=LIVE_SCORE_ROI_ALIGNMENT_DEFAULT,
                     live_exact_native=LIVE_EXACT_NATIVE_DEFAULT,
                     live_exact_connectivity=LIVE_EXACT_CONNECTIVITY_DEFAULT,
                     live_near_future_cache=LIVE_NEAR_FUTURE_CACHE_DEFAULT,
                     live_reference_times_only=LIVE_REFERENCE_TIMES_ONLY_DEFAULT,
                     live_border_grace=LIVE_BORDER_GRACE_DEFAULT,
                     terminal_hold_input_recovery=TERMINAL_HOLD_INPUT_RECOVERY_DEFAULT,
                     live_latest_evaluation=LIVE_LATEST_EVALUATION_DEFAULT,
                     live_stage_timing=LIVE_STAGE_TIMING_DEFAULT,
                     live_async_screen_verify=LIVE_ASYNC_SCREEN_VERIFY_DEFAULT,
                     b3_share_score_read=B3_SHARE_SCORE_READ_DEFAULT,
                     b3_score_matmul=B3_SCORE_MATMUL_DEFAULT,
                     b3_warning_parallel=B3_WARNING_PARALLEL_DEFAULT,
                     b3_ledger_mode=B3_LEDGER_MODE_DEFAULT,
                     b3_ledger_landing=B3_LEDGER_LANDING_DEFAULT,
                     b3_skip_icon_reading=B3_SKIP_ICON_READING_DEFAULT,
                     # 段1 直接公開。spawn 子へ PUYO_LIVE_X_DIRECT_PUBLISH で届く (live_x_direct.py)。
                     live_x_direct_publish=LIVE_X_DIRECT_PUBLISH_DEFAULT,
                     # 段3 撃つ直前の暫定 X (live_v49f_prefire.py)。live_v49f=true が前提。
                     live_x_prefire_shadow=LIVE_X_PREFIRE_SHADOW_DEFAULT,
                     live_x_prefire=LIVE_X_PREFIRE_DEFAULT,
                     # コマ落ちに強い数え方 (DESIGN frame_drop_robust_20261023)。spawn 子へ
                     # PUYO_LIVE_FRAME_DROP_ROBUST で届き、認識設定の enable_media_tick_counters になる。
                     live_frame_drop_robust=LIVE_FRAME_DROP_ROBUST_DEFAULT,
                     live_lockdown_score_zero_then_move_release=LIVE_LOCKDOWN_SCORE_ZERO_THEN_MOVE_DEFAULT,
                     # 予告 CNN の 1 回 forward 化と、盤面認識の前からの先読み (試作、既定OFF)。
                     b3_warning_batch=B3_WARNING_BATCH_DEFAULT,
                     b3_warning_prefetch=B3_WARNING_PREFETCH_DEFAULT,
                     live_match_end_pair=LIVE_MATCH_END_PAIR_DEFAULT,
                     live_verify_reason_log=LIVE_VERIFY_REASON_LOG_DEFAULT,
                     b3_warning_async=B3_WARNING_ASYNC_DEFAULT,
                     live_native_hsv_main=LIVE_NATIVE_HSV_MAIN_DEFAULT,
                     live_verify_async=LIVE_VERIFY_ASYNC_DEFAULT,
                     live_match_end_feed_on_hold=LIVE_MATCH_END_FEED_ON_HOLD_DEFAULT,
                     match_end_yatta_text_template=MATCH_END_YATTA_TEXT_TEMPLATE_DEFAULT,
                     # 起動と停止のガード (試作、既定OFF)。spawn 子へ PUYO_LIVE_CAPTURE_RECONNECT /
                     # PUYO_B3_WORKER_GRACEFUL_STOP で届く (live_capture_reconnect.py / b3_async.py)。
                     live_capture_reconnect=LIVE_CAPTURE_RECONNECT_DEFAULT,
                     b3_worker_graceful_stop=B3_WORKER_GRACEFUL_STOP_DEFAULT,
                     live_board_vectorized=LIVE_BOARD_VECTORIZED_DEFAULT,
                     live_verify_soft_hold=LIVE_VERIFY_SOFT_HOLD_DEFAULT,
                     live_verify_entry_fast_retry=LIVE_VERIFY_ENTRY_FAST_RETRY_DEFAULT,
                     live_chain_forecast_recheck=LIVE_CHAIN_FORECAST_RECHECK_DEFAULT,
                     live_chain_end_formula=LIVE_CHAIN_END_FORMULA_DEFAULT,
                     # 認識ループの Python 側の出力不変の高速化 (src/recog_speed_py.py、既定OFF)。
                     # spawn 子へ PUYO_LIVE_RECOG_SPEED_PY で届く。実効値の記録 (TRIAL_FLAG_KEYS) には足さない
                     # (24d の組立・検収が TRIAL_FLAG_KEYS の全項目で腕を組むため)。
                     live_recog_speed_py=LIVE_RECOG_SPEED_PY_DEFAULT,
                     # 同上のうち壁時計に依存する終了の組の crop だけを別の旗にしたもの (既定OFF)。
                     live_recog_speed_py_pair_lazy=LIVE_RECOG_SPEED_PY_PAIR_LAZY_DEFAULT,
                     live_native_read_board=LIVE_NATIVE_READ_BOARD_DEFAULT)
from src.production_config import LIVE_C_ITEMS_DEFAULTS
SPEED_DEFAULTS.update(LIVE_C_ITEMS_DEFAULTS)
SPEED_DEFAULTS['live_death_check_retr'] = False

WORKERS_KEY = 'b3_warning_workers'
PREFIRE_KEYS = ('live_x_prefire_shadow', 'live_x_prefire')
FRAME_DROP_ROBUST_ENV = 'PUYO_LIVE_FRAME_DROP_ROBUST'
SCORE_ZERO_THEN_MOVE_KEY = 'live_lockdown_score_zero_then_move_release'
SCORE_ZERO_THEN_MOVE_ENV = 'PUYO_' + SCORE_ZERO_THEN_MOVE_KEY.upper()
# OBS 試験版 (23y) で ON/OFF を切り替える試作フラグ。実行時に output へ実効値を残す (fail-silent の防止)。
X_TRIAL_KEYS = ('live_x_direct_publish', 'live_x_prefire_shadow', 'live_x_prefire')
SPEED_TRIAL_KEYS = ('live_frame_drop_robust', 'b3_warning_batch', 'b3_warning_prefetch', SCORE_ZERO_THEN_MOVE_KEY)
MATCH_END_PAIR_KEY = 'live_match_end_pair'
VERIFY_REASON_KEY = 'live_verify_reason_log'
VERIFY_ASYNC_KEY = 'live_verify_async'
NATIVE_HSV_MAIN_KEY = 'live_native_hsv_main'
TRIAL_FLAG_KEYS = X_TRIAL_KEYS+SPEED_TRIAL_KEYS+(MATCH_END_PAIR_KEY, VERIFY_REASON_KEY, 'b3_warning_async', NATIVE_HSV_MAIN_KEY, VERIFY_ASYNC_KEY)
# 23y の OFF/X/ALL の分類は既存6項目を維持し、R37 と組判定は独立して記録する。
ARM_TRIAL_KEYS = tuple(key for key in TRIAL_FLAG_KEYS if key not in (SCORE_ZERO_THEN_MOVE_KEY, MATCH_END_PAIR_KEY, VERIFY_REASON_KEY, 'b3_warning_async', NATIVE_HSV_MAIN_KEY, VERIFY_ASYNC_KEY))
TRIAL_FLAGS_FILE = 'live_trial_flags.json'
TRIAL_FLAG_KEYS += ('live_match_end_feed_on_hold', 'match_end_yatta_text_template')
# 起動と停止のガード (2026-10-24)。24c の受領票が実効値 (flags/env) を照合できるよう、記録の鍵へ足す。
CAPTURE_RECONNECT_KEY = 'live_capture_reconnect'
WORKER_GRACEFUL_STOP_KEY = 'b3_worker_graceful_stop'
TRIAL_FLAG_KEYS += (CAPTURE_RECONNECT_KEY, WORKER_GRACEFUL_STOP_KEY)
# 盤面読取の一括経路 (2026-10-24)。末尾に追加し、既存キーの順序と腕 (OFF/X/ALL) の分類は変えない。
BOARD_VECTORIZED_KEY = 'live_board_vectorized'
TRIAL_FLAG_KEYS += (BOARD_VECTORIZED_KEY,)
# 画面確認の失敗で状態を捨てない短い保留 (2026-10-24、既定 OFF)。実効値の記録の末尾
VERIFY_SOFT_HOLD_KEY = 'live_verify_soft_hold'
TRIAL_FLAG_KEYS += (VERIFY_SOFT_HOLD_KEY,)
# read_board の 1st pass の Rust 経路 (2026-10-25、既定 OFF)。設定・環境変数・認識設定には載るが、
# 試作フラグの記録 (TRIAL_FLAG_KEYS) には足さない: 24a〜24d の受領票・試験が TRIAL_FLAG_KEYS の完全一致を前提にしているため。
# 実効値は認識子が runtime (live_trial_flags.json) へ追記する (live_native_read_board_runtime.py)。
# 採用して腕へ入れる組み立てのとき、受領票の一覧と一緒に TRIAL_FLAG_KEYS へ足す。
NATIVE_READ_BOARD_KEY = 'live_native_read_board'
TRIAL_FLAGS_SCHEMA = 1
ARM_OFF, ARM_X, ARM_ALL, ARM_CUSTOM = 'OFF', 'X', 'ALL', 'CUSTOM'
# コマ落ちに強い状態判定の箇所別 OFF (既定は何も外さない = 全部の箇所が live_frame_drop_robust のまま働く)。
# 箇所の名前は設計 frame_drop_robust_20261023 の R 番号 (R02 は R03 を含む)。実装は live_fdr_rules.py。
FRAME_DROP_RULES = ('R01', 'R02', 'R04', 'R06', 'R10', 'R15', 'B07', 'R35')
FRAME_DROP_RULES_OFF_KEY = 'live_frame_drop_rules_off'
FRAME_DROP_RULES_OFF_ENV = 'PUYO_LIVE_FRAME_DROP_RULES_OFF'


def validate(data: dict[str, Any]) -> dict[str, Any]:
    """JSON経由でもboolとworker数の範囲を厳密に検証する。"""
    values = {key: data.get(key, default) for key, default in SPEED_DEFAULTS.items()}
    # R37 の null は未指定と同じ既定値へ戻す。他のフラグの厳密検証は維持する。
    if values[SCORE_ZERO_THEN_MOVE_KEY] is None:
        values[SCORE_ZERO_THEN_MOVE_KEY] = SPEED_DEFAULTS[SCORE_ZERO_THEN_MOVE_KEY]
    for key, value in values.items():
        if type(value) is not bool:
            raise ValueError(f'{key} はboolが必要です')
    workers = data.get(WORKERS_KEY, B3_WARNING_WORKERS)
    if type(workers) is not int or not 1 <= workers <= B3_WARNING_MAX_WORKERS:
        raise ValueError(f'{WORKERS_KEY} は1〜{B3_WARNING_MAX_WORKERS}の整数です')
    return dict(values, b3_warning_workers=workers)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for key, default in SPEED_DEFAULTS.items():
        parser.add_argument('--'+key.replace('_', '-'), action=argparse.BooleanOptionalAction,
                            default=default, help=('起動時の式12コマで得点・式の枠を固定較正（既定ON）'
                                                   if key == 'align_score_roi' else None))
    parser.add_argument('--b3-warning-workers', type=int, default=B3_WARNING_WORKERS)
    parser.add_argument('--'+FRAME_DROP_RULES_OFF_KEY.replace('_', '-'), default='')


def configure(options: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """動画のR1bを維持し、子processへ検証済みの実効値を渡す。"""
    try:
        values = validate(vars(options))
        if values['b3_warning_parallel'] and options.cpu_threads != 1:
            raise ValueError('b3_warning_parallel にはcpu_threads=1が必要です')
        if values['live_latest_evaluation']:
            if not (getattr(options, 'b3_judgment', False)
                    and getattr(options, 'split_evaluation', False)
                    and getattr(options, 'worker_mode', '') == 'process'
                    and not getattr(options, 'compare', False)):
                raise ValueError('live_latest_evaluationにはB3・process・split-evaluationが必要です')
            # 状態通知は全件配送し、凍結済みの計算要求だけを最新一件にする既存経路。
            options.b3_async = True
        if any(values[key] for key in PREFIRE_KEYS) and not getattr(options, 'live_v49f', False):
            raise ValueError('live_x_prefire / live_x_prefire_shadow には live_v49f=true が必要です')
        if values['b3_warning_prefetch'] and not (values['b3_warning_batch'] or values['b3_warning_async']):
            raise ValueError('b3_warning_prefetch には b3_warning_batch=true が必要です')
        rules_off = rules_off_of(vars(options), values)
    except ValueError as error:
        parser.error(str(error))
    os.environ[FRAME_DROP_RULES_OFF_ENV] = ','.join(rules_off)   # 起動ごとに上書き (空 = 何も外さない)
    values['live_disable_placement_reconcile'] &= options.source == 'dshow'
    for key, value in values.items():
        os.environ['PUYO_'+key.upper()] = str(int(value))
    write_trial_flags_if_ready(options)
    from .live_native_hsv import configure_record_path
    configure_record_path(getattr(options, 'output', None))
    from .live_screen_errors import configure_output as configure_screen_output
    configure_screen_output(getattr(options, 'output', None))
    from .live_capture_reconnect import configure_output as configure_reconnect_output
    configure_reconnect_output(values['live_capture_reconnect'], getattr(options, 'output', None))
    from .live_verify_reasons import OUTPUT_ENV
    os.environ.pop(OUTPUT_ENV, None)
    if values['live_verify_reason_log'] and getattr(options, 'output', None) is not None:
        os.environ[OUTPUT_ENV] = str(Path(options.output).resolve())


def parse_rules_off(value: Any) -> tuple[str, ...]:
    """箇所別 OFF の指定 (None・空・'R06,R10'・['R06', 'R10']) を FRAME_DROP_RULES の順の tuple にする。

    知らない名前・文字列以外は ValueError (黙って無視しない)。
    """
    if value is None or value == '':
        return ()
    items = value.split(',') if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)) or not all(isinstance(item, str) for item in items):
        raise ValueError(f'{FRAME_DROP_RULES_OFF_KEY} は箇所名の配列です (例: ["R06"])')
    names = {item.strip() for item in items if item.strip()}
    unknown = sorted(names-set(FRAME_DROP_RULES))
    if unknown:
        raise ValueError(f'{FRAME_DROP_RULES_OFF_KEY} に知らない箇所: {unknown} (候補 {list(FRAME_DROP_RULES)})')
    return tuple(rule for rule in FRAME_DROP_RULES if rule in names)


def rules_off_of(data: dict[str, Any], values: dict[str, Any]) -> tuple[str, ...]:
    """設定 data の箇所別 OFF。live_frame_drop_robust が OFF のまま指定されたら ValueError (効かない指定を通さない)。"""
    rules = parse_rules_off(data.get(FRAME_DROP_RULES_OFF_KEY))
    if rules and not values['live_frame_drop_robust']:
        raise ValueError(f'{FRAME_DROP_RULES_OFF_KEY} には live_frame_drop_robust=true が必要です')
    return rules


def rules_off_from_env() -> tuple[str, ...]:
    """子 process が継いだ環境変数から、外す箇所を読む (未設定・空は ())。"""
    return parse_rules_off(os.environ.get(FRAME_DROP_RULES_OFF_ENV))


def recognition_config(config: dict[str, Any], live_input: bool) -> dict[str, Any]:
    """実効認識設定を保存前に更新する。通常動画では既存値を変更しない。"""
    if live_input and os.environ.get('PUYO_LIVE_DISABLE_PLACEMENT_RECONCILE') == '1':
        config = dict(config, enable_placement_signal_reconcile=False)
    if os.environ.get('PUYO_W68_INFERENCE_PENDING') == '1':
        config = dict(config, enable_w68_inference_pending=True)
    if os.environ.get('PUYO_W68_ROLLBACK') == '1':
        config = dict(config, enable_w68_rollback=True)
    if os.environ.get('PUYO_LIVE_ROW1_GUARD') == '1':
        config = dict(config, enable_row1_col2_guard=True)
    if os.environ.get('PUYO_ALIGN_SCORE_ROI') == '1':
        config = dict(config, enable_score_roi_alignment=True)
    if frame_drop_robust_enabled():
        # OFF では鍵を足さない (recognition_config.json は従来と同じ)。
        config = dict(config, enable_media_tick_counters=True)
    if os.environ.get(SCORE_ZERO_THEN_MOVE_ENV) == '1':
        config = dict(config, enable_lockdown_score_zero_then_move_release=True)
    if os.environ.get('PUYO_LIVE_MATCH_END_PAIR') == '1':
        config = dict(config, enable_match_end_pair=True)
    if os.environ.get('PUYO_'+NATIVE_HSV_MAIN_KEY.upper()) == '1':
        config = dict(config, enable_native_hsv_classifier=True)
    if os.environ.get('PUYO_'+BOARD_VECTORIZED_KEY.upper()) == '1':
        config = dict(config, enable_live_board_vectorized=True)
    if os.environ.get('PUYO_'+NATIVE_READ_BOARD_KEY.upper()) == '1':
        config = dict(config, enable_live_native_read_board=True)
    return config


def frame_drop_robust_enabled() -> bool:
    """配信のコマ落ちに強い数え方が ON か (spawn 子でも環境変数で受け取る)。"""
    return os.environ.get(FRAME_DROP_ROBUST_ENV) == '1'


def observation_options() -> dict[str, Any]:
    """既定OFFではコンストラクタへの追加引数も無く、旧経路を維持する。"""
    values: dict[str, Any] = {}
    if os.environ.get('PUYO_B3_SKIP_ICON_READING') == '1':
        values['skip_icon_reading'] = True
    if os.environ.get('PUYO_B3_SCORE_MATMUL') == '1':
        values['enable_score_matmul'] = True
    if os.environ.get('PUYO_B3_WARNING_PARALLEL') == '1':
        values.update(enable_warning_parallel=True,
                      warning_workers=int(os.environ.get('PUYO_B3_WARNING_WORKERS', B3_WARNING_WORKERS)))
    if os.environ.get('PUYO_B3_WARNING_BATCH') == '1':
        values['enable_warning_batch'] = True
        if os.environ.get('PUYO_B3_WARNING_PREFETCH') == '1':
            values['warning_prefetch'] = True
    if os.environ.get('PUYO_B3_WARNING_ASYNC') == '1':
        values['enable_warning_async'] = True
    return values


def classify_arm(flags: dict[str, bool]) -> str:
    """23y の既存6項目から腕を返す。R37 の ON/OFF は flags・env で別に記録する。"""
    enabled = {key for key in ARM_TRIAL_KEYS if flags.get(key)}
    if not enabled:
        return ARM_OFF
    if enabled == set(X_TRIAL_KEYS):
        return ARM_X
    return ARM_ALL if enabled == set(ARM_TRIAL_KEYS) else ARM_CUSTOM


INSTANCE_GUARD_RECORD_KEY = 'instance_guard'


def instance_guard_record() -> dict[str, Any]:
    """二重起動の検出の実効値。要求値 (設定) でなく、この process が実際に取った鍵と bind の方式を残す。
    起動口 (launcher) を通らない起動では held が空になる (検出が働いていない印)。"""
    from .live_instance_guard import allow_second_instance_from_env, exclusive_bind_default, held_names
    return dict(allow_second_instance=allow_second_instance_from_env(), exclusive_bind=exclusive_bind_default(),
                held=held_names())


def trial_flags_record(options: argparse.Namespace) -> dict[str, Any]:
    """試作フラグの実効値の記録。flags は検証済みの設定、env は子 process が継ぐ環境変数 (食い違えば配線の誤り)。"""
    flags = {key: bool(getattr(options, key, SPEED_DEFAULTS[key])) for key in TRIAL_FLAG_KEYS}
    env = {'PUYO_'+key.upper(): os.environ.get('PUYO_'+key.upper()) for key in TRIAL_FLAG_KEYS}
    record = dict(schema=TRIAL_FLAGS_SCHEMA, arm=classify_arm(flags), flags=flags, env=env,
                frame_drop_rules_off=list(rules_off_from_env()),
                frame_drop_robust_sites=sorted(LIVE_FRAME_DROP_ROBUST_SITES),
                runtime={key: dict(active=False, reason=(
                    'recognition_not_started' if flags[key] else 'disabled'))
                    for key in (NATIVE_HSV_MAIN_KEY, BOARD_VECTORIZED_KEY)})
    if flags['b3_warning_async']:
        record['warning_execution'] = ('disabled' if getattr(options, 'b3_skip_icon_reading', False)
                                       else 'async_latest_batched_without_prefetch')
    record[INSTANCE_GUARD_RECORD_KEY] = instance_guard_record()   # 既存の鍵の順序は変えず、末尾に足すだけ
    if getattr(options, 'live_verify_entry_fast_retry', False):
        from .live_verify_entry import ENTRY_VERIFY_PERIOD_SEC
        record['entry_verification'] = dict(fast_retry=True, period_sec=ENTRY_VERIFY_PERIOD_SEC)
    chain_keys = ('live_chain_forecast_recheck', 'live_chain_end_formula')
    if any(getattr(options, key, False) for key in chain_keys):
        record['chain_continuity'] = {key: dict(enabled=bool(getattr(options, key, False)),
            env=os.environ.get('PUYO_' + key.upper())) for key in chain_keys}
    return record


def write_trial_flags_if_ready(options: argparse.Namespace) -> bool:
    """output が既にあるとき (ランチャが作った後) だけ記録を書く。無ければ何もしない (単体の呼び出しで勝手に作らない)。

    配布の起動口 (ランチャ) は設定の写しを書くために output を先に作るので、OBS の走行・流し直しの台はこの経路で記録が残る。
    scripts/run_live_pipeline_20260928.py は配布 app が実行時に読む (台が差し替えられない) ので、記録はここ (src) から書く。
    """
    output = getattr(options, 'output', None)
    if not isinstance(output, Path) or not output.is_dir():
        return False
    write_trial_flags(options)
    return True


def write_trial_flags(options: argparse.Namespace) -> None:
    """output/live_trial_flags.json へ試作フラグの実効値を書く。OFF でも書く (無い = 未記録と区別するため)。"""
    import json
    path = options.output/TRIAL_FLAGS_FILE
    path.write_text(json.dumps(trial_flags_record(options), ensure_ascii=False, indent=2), encoding='utf-8')
