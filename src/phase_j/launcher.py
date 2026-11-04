"""配布版リアルタイム・オーバーレイの単一起動口。

    python -m src.phase_j.launcher --config puyo_live.json

利用者用の小さな設定ファイル (機器名・ポート・出力先) を読み、既存パイプライン
`scripts/run_live_pipeline_20260928` の CLI/--config へ写像して起動する。
パイプライン本体・既存 CLI は変更しない (追加のみ)。手順:
  1. 設定を検証 (不正は日本語で原因と直し方を出して終了コード 2)
  2. 配布物の完全性を MANIFEST.json で照合 (不一致は終了コード 3)
  3. 作業ディレクトリをアプリ根へ移し、既存パイプラインを呼ぶ
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from .launcher_manifest import MANIFEST_FILENAME, ManifestError, verify_manifest
from .launcher_runtime import EXIT_RUNTIME, missing_runtime_dlls, runtime_guidance
from .overlay_config import validate_overlay
from .release_config import release_user_config, release_environment
from src.production_config import DISPLAY_COMPLETION_DRIVEN_DEFAULT, DISPLAY_FAST_EMA_DEFAULT
from src.production_config import FAST_FIRST_JUDGMENT_DEFAULT
from src.production_config import LIVE_COLOR_BUFFER_DEFAULT, LIVE_ONNX_PARALLEL_DEFAULT
from src.production_config import RECOVERY_BOUNDARY_COALESCING_DEFAULT
from src.production_config import EVALUATION_FRESHNESS_GATE_DEFAULT
from src.production_config import TERMINAL_DISPLAY_HOLD_DEFAULT
from src.production_config import B3_ENSEMBLE_DEFAULT
from src.production_config import LIVE_B3_FIRE_BOARD_DEFAULT
from .live_ex_fresh import EX_FRESH_SEC_DEFAULT
from .live_24h_config import CONFIG_KEYS, LIVE_24H_DEFAULT, LOGICAL_DELAY_MS_DEFAULT
from .live_power import power_pipeline_values
from .live_speed import (FRAME_DROP_RULES_OFF_KEY, SPEED_DEFAULTS, WORKERS_KEY, rules_off_of,
                         validate as validate_speed)
from src.production_config import LIVE_DELTA_CELLS_DEFAULT, LIVE_CNN_DEDUP_DEFAULT, LIVE_ROW1_GUARD_DEFAULT
from src.production_config import (LIVE_DISABLE_PLACEMENT_RECONCILE_DEFAULT,
    B3_SCORE_MATMUL_DEFAULT, B3_WARNING_PARALLEL_DEFAULT, B3_WARNING_WORKERS,
    B3_LEDGER_MODE_DEFAULT, B3_LEDGER_LANDING_DEFAULT, B3_SKIP_ICON_READING_DEFAULT,
    B3_SHARE_SCORE_READ_DEFAULT, LIVE_ASYNC_SCREEN_VERIFY_DEFAULT, LIVE_STAGE_TIMING_DEFAULT,
    TERMINAL_HOLD_INPUT_RECOVERY_DEFAULT, LIVE_LATEST_EVALUATION_DEFAULT,
    LIVE_REFERENCE_TIMES_ONLY_DEFAULT, LIVE_BORDER_GRACE_DEFAULT, LIVE_NEAR_FUTURE_CACHE_DEFAULT,
    LIVE_EXACT_CONNECTIVITY_DEFAULT, LIVE_EXACT_NATIVE_DEFAULT)
from src.production_config import LIVE_X_DIRECT_PUBLISH_DEFAULT, ALLOW_SECOND_INSTANCE_DEFAULT

APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USER_CONFIG = Path('puyo_live.json')
PIPELINE_MODULE = 'scripts.run_live_pipeline_20260928'
PIPELINE_CONFIG_NAME = 'pipeline_config.json'

SOURCE_DSHOW, SOURCE_VIDEO = 'dshow', 'video'
DEFAULT_PORT = 8765
MIN_PORT, MAX_PORT = 1024, 65535
DEFAULT_HOST = '127.0.0.1'
DEFAULT_MC_ROLLOUTS = 30
MAX_PROBE_INDEX = 8  # --list-devices が調べる機器 index の上限
DEFAULT_OUTPUT_DIR = 'output'
# 既存 live_dshow.example.json と同じ配布既定 (CPU のみ・評価は低優先度)
# fast_terminal: 死亡検出の低解像度予備判定。491,928 フレームで決定が完全一致 (rt logs/live_b20/RESULT.md、2026-09-30)。
# fast_telop: 中央テロップ検出の縮小予備判定。491,928 フレーム (983,856 テンプレート×フレーム) で決定差 0。
#   ただし実データに可視テロップが 0 件のため、陽性側は合成 (216 対で誤スキップ 0) の証拠のみ (packaging/perf/RESULT.md)。
# performance_cores: 認識を Windows の性能コアへ寄せる (ハイブリッド CPU のみ作用。決定には無関係)。
# async_notice_queue は 54 秒の配信空白を起こしたため配布でも OFF のまま (キーを出さない = 既定 False)。
PIPELINE_FIXED = dict(verification_only=True, coalesce_features=True, cpu_threads=1,
                      evaluation_nice=10, cnn_device='cpu', fast_terminal=True,
                      performance_cores=True, fast_telop=True)
# 画面確認の緩和 (2026-09-30 verifier_diag): 連鎖中の掛け算式を許容し外枠の上辺を免除 (対戦中合格 44%→87%)、
# 継続確認は連続 3 回不合格で初めて hold (1 回で履歴・色較正を全リセットしない)。詳細は packaging/verifier_diag/FIX_RESULT.md。
VERIFY_FAIL_STREAK = 3
PIPELINE_FIXED['relaxed_verify'] = True
# score_zero を ScoreOcr の「8 桁全て 0」で判定し、スコアゼロの画像テンプレート (ゲーム画面の切り出し) を配布しない
# (user 決定 2026-09-30)。テンプレート方式との不一致は 6 窓 30,540 フレームのうち 866 フレーム (25 区間) が全て「テンプレートの取りこぼし」 (逆方向 0)、
# 認識パイプラインの確定盤面は 116 秒窓 409,188 セルで差 0 (packaging/ab_ui_2026-09-30)。
PIPELINE_FIXED['score_zero_from_ocr'] = True
PIPELINE_FIXED['verify_fail_streak'] = VERIFY_FAIL_STREAK
VIDEO_WARMUP_SEC = 1.0

EXIT_OK, EXIT_CONFIG, EXIT_MANIFEST = 0, 2, 3
# EXIT_RUNTIME = 4 (launcher_runtime)。二重起動・port の bind 失敗は live_instance_guard.EXIT_ALREADY_RUNNING = 5。
CALIBRATION_APP, CALIBRATION_LOCALAPPDATA = 'app', 'localappdata'  # 機器別較正の保存先 (既定は app)
# live_device.CALIBRATION_DIR_ENV と同値 (live_device は cv2 を import するためここでは複製し、試験で一致を固定)
CALIBRATION_DIR_ENV = 'PUYO_CALIBRATION_DIR'
# cnn_onnx.BACKEND_ENV / BACKEND_ONNX / INDEX_NAME と同値 (cnn_onnx は numpy を import するためここでは複製し、試験で一致を固定)
CNN_BACKEND_ENV, CNN_BACKEND_ONNX = 'PUYO_CNN_BACKEND', 'onnx'
ONNX_INDEX = Path('models') / 'onnx' / 'index.json'
# DirectShow の色行列補正 (off / auto / 601to709)。dshow_color.COLOR_CORRECTION_MODES と同値
# (dshow_color は cv2 を import するためここでは複製し、試験で一致を固定)。既定 off = 従来と同一。
COLOR_CORRECTION_MODES = ('off', 'auto', '601to709')
DEFAULT_COLOR_CORRECTION = 'off'
LOCALAPPDATA_SUBDIR =Path('PuyoLive') / 'device_calibration'
USER_KEYS = frozenset({'source', 'device_name', 'device_index', 'video_path', 'port', 'host',
                       'output_dir', 'mc_rollouts', 'start_sec', 'end_sec', 'calibration_location',
                       'dshow_color_correction', 'b3_judgment', 'b3_async', 'live_b3_fire_board', 'b3_model', 'b3_timing_table', 'live_v49f',
                       'live_ama_plain', 'b3_m3', 'b3_time_gate', 'live_score_cadence', 'live_static_exact',
                       'b3_c2', 'b3_ensemble', 'display_completion_driven', 'display_fast_ema', 'overlay', 'fast_first_judgment',
                       'recovery_boundary_coalescing', 'evaluation_freshness_gate', 'terminal_display_hold',
                       'allow_second_instance', *CONFIG_KEYS, *SPEED_DEFAULTS, WORKERS_KEY, FRAME_DROP_RULES_OFF_KEY})


class LauncherConfigError(ValueError):
    """利用者設定の誤り。メッセージはそのまま画面に出す。"""


@dataclass(frozen=True)
class UserConfig:
    source: str
    device_name: str | None
    device_index: int
    video_path: Path | None
    port: int
    host: str
    output_dir: Path
    mc_rollouts: int
    start_sec: float | None
    end_sec: float | None
    calibration_location: str = CALIBRATION_APP
    dshow_color_correction: str = DEFAULT_COLOR_CORRECTION
    b3_judgment: bool = False
    b3_model: Path | None = None
    b3_timing_table: Path | None = None
    b3_async: bool = False
    live_v49f: bool = False
    live_ama_plain: bool = False
    b3_m3: bool = False
    b3_time_gate: bool = False
    live_score_cadence: bool = False
    live_static_exact: bool = False
    overlay: dict[str, Any] | None = None
    display_completion_driven: bool = DISPLAY_COMPLETION_DRIVEN_DEFAULT
    display_fast_ema: bool = DISPLAY_FAST_EMA_DEFAULT
    fast_first_judgment: bool = FAST_FIRST_JUDGMENT_DEFAULT
    recovery_boundary_coalescing: bool = RECOVERY_BOUNDARY_COALESCING_DEFAULT
    evaluation_freshness_gate: bool = EVALUATION_FRESHNESS_GATE_DEFAULT
    terminal_display_hold: bool = TERMINAL_DISPLAY_HOLD_DEFAULT
    live_disable_placement_reconcile: bool = LIVE_DISABLE_PLACEMENT_RECONCILE_DEFAULT
    b3_score_matmul: bool = B3_SCORE_MATMUL_DEFAULT
    b3_share_score_read: bool = B3_SHARE_SCORE_READ_DEFAULT
    live_async_screen_verify: bool = LIVE_ASYNC_SCREEN_VERIFY_DEFAULT
    live_stage_timing: bool = LIVE_STAGE_TIMING_DEFAULT
    terminal_hold_input_recovery: bool = TERMINAL_HOLD_INPUT_RECOVERY_DEFAULT
    live_latest_evaluation: bool = LIVE_LATEST_EVALUATION_DEFAULT
    live_reference_times_only: bool = LIVE_REFERENCE_TIMES_ONLY_DEFAULT
    live_border_grace: bool = LIVE_BORDER_GRACE_DEFAULT
    b3_warning_parallel: bool = B3_WARNING_PARALLEL_DEFAULT
    b3_warning_workers: int = B3_WARNING_WORKERS
    b3_ledger_mode: bool = B3_LEDGER_MODE_DEFAULT
    b3_ledger_landing: bool = B3_LEDGER_LANDING_DEFAULT
    b3_skip_icon_reading: bool = B3_SKIP_ICON_READING_DEFAULT
    live_near_future_cache: bool = LIVE_NEAR_FUTURE_CACHE_DEFAULT
    live_color_buffer: bool = LIVE_COLOR_BUFFER_DEFAULT
    live_onnx_parallel: bool = LIVE_ONNX_PARALLEL_DEFAULT
    live_exact_connectivity: bool = LIVE_EXACT_CONNECTIVITY_DEFAULT
    live_exact_native: bool = LIVE_EXACT_NATIVE_DEFAULT
    live_public_match_number: bool = SPEED_DEFAULTS['live_public_match_number']
    live_delta_cells: bool = LIVE_DELTA_CELLS_DEFAULT
    live_cnn_dedup: bool = LIVE_CNN_DEDUP_DEFAULT
    live_row1_guard: bool = LIVE_ROW1_GUARD_DEFAULT
    align_score_roi: bool = SPEED_DEFAULTS['align_score_roi']
    live_x_direct_publish: bool = LIVE_X_DIRECT_PUBLISH_DEFAULT
    live_x_prefire_shadow: bool = SPEED_DEFAULTS['live_x_prefire_shadow']
    live_x_prefire: bool = SPEED_DEFAULTS['live_x_prefire']
    live_frame_drop_robust: bool = SPEED_DEFAULTS['live_frame_drop_robust']
    live_lockdown_score_zero_then_move_release: bool = SPEED_DEFAULTS['live_lockdown_score_zero_then_move_release']
    live_match_end_pair: bool = SPEED_DEFAULTS['live_match_end_pair']
    b3_warning_batch: bool = SPEED_DEFAULTS['b3_warning_batch']
    b3_warning_prefetch: bool = SPEED_DEFAULTS['b3_warning_prefetch']
    b3_warning_async: bool = SPEED_DEFAULTS['b3_warning_async']
    live_verify_reason_log: bool = SPEED_DEFAULTS['live_verify_reason_log']
    live_native_hsv_main: bool = SPEED_DEFAULTS['live_native_hsv_main']
    live_verify_async: bool = SPEED_DEFAULTS['live_verify_async']
    live_frame_drop_rules_off: tuple[str, ...] = ()   # コマ落ちに強い状態判定で外す箇所 (既定は空 = 全部 ON)
    live_match_end_feed_on_hold: bool = SPEED_DEFAULTS['live_match_end_feed_on_hold']
    match_end_yatta_text_template: bool = SPEED_DEFAULTS['match_end_yatta_text_template']
    live_capture_reconnect: bool = SPEED_DEFAULTS['live_capture_reconnect']
    b3_worker_graceful_stop: bool = SPEED_DEFAULTS['b3_worker_graceful_stop']
    allow_second_instance: bool = ALLOW_SECOND_INSTANCE_DEFAULT   # 二重起動の検出で止めない (既定 False = 止める)
    live_board_vectorized: bool = SPEED_DEFAULTS['live_board_vectorized']
    live_verify_soft_hold: bool = SPEED_DEFAULTS['live_verify_soft_hold']
    live_verify_entry_fast_retry: bool = SPEED_DEFAULTS['live_verify_entry_fast_retry']
    live_chain_forecast_recheck: bool = SPEED_DEFAULTS['live_chain_forecast_recheck']
    live_chain_end_formula: bool = SPEED_DEFAULTS['live_chain_end_formula']
    live_color_roi: bool = SPEED_DEFAULTS['live_color_roi']
    live_capture_thread: bool = SPEED_DEFAULTS['live_capture_thread']
    live_boundary_half_second: bool = SPEED_DEFAULTS['live_boundary_half_second']
    live_hsv_lazy: bool = SPEED_DEFAULTS['live_hsv_lazy']
    live_recog_speed_py: bool = SPEED_DEFAULTS['live_recog_speed_py']   # 認識ループの Python 側の出力不変の高速化 (既定 OFF)
    live_recog_speed_py_pair_lazy: bool = SPEED_DEFAULTS['live_recog_speed_py_pair_lazy']   # 同上の壁時計依存の 1 項目 (既定 OFF)
    live_native_read_board: bool = SPEED_DEFAULTS['live_native_read_board']
    live_verify_read_fast: bool = SPEED_DEFAULTS['live_verify_read_fast']
    live_death_check_retr: bool = SPEED_DEFAULTS['live_death_check_retr']
    live_winrate_switch: bool = SPEED_DEFAULTS['live_winrate_switch']
    b3_c2: bool = False   # 勝率モデルを C2 にして予告アイコンを読まない試験構成 (既定 OFF。前提は b3_m3・b3_skip_icon_reading)
    b3_ensemble: bool = B3_ENSEMBLE_DEFAULT   # C2 の seed 違いを複数並べて確率を平均する試験構成 (既定 OFF。前提は b3_c2。b3_model はマニフェスト JSON)
    live_24h: bool = LIVE_24H_DEFAULT
    live_disable_power_throttling: bool = False
    live_24h_ama: bool = True
    live_24h_retr: bool = False
    live_24h_delay_ms: float = LOGICAL_DELAY_MS_DEFAULT
    live_24h_model: Path | None = None
    live_b3_fire_board: bool = LIVE_B3_FIRE_BOARD_DEFAULT
    live_24h_xc: bool = False
    live_24h_tape_next: bool = False   # EX のツモのテープに毎コマの NEXT を送る (既定 OFF、10-29)
    live_24h_retr_h_forecast: bool = False   # retR の持ち時間 h を共通の連鎖の予測から (依頼 F3、既定 OFF、10-29)
    live_24h_media_fresh: bool = False   # EX の鮮度と保持を媒体時刻で判断する。
    live_24h_row_time: bool = False   # 再評価もB3行の入力時刻の鍵と解除で選ぶ。
    live_24h_ex_speed: bool = False  # retR・xcを状態の写しで探索と並列に計算する。
    live_24h_stable_key: bool = False  # 共通の連鎖予測の番号で鍵を作る
    live_chain_split_place: bool = False  # 置きの後の式を別の連鎖として数える
    live_24h_retr_second: bool = False  # 撃った側のセカンド8列
    live_24h_retr_move050: bool = False  # 1手の時間に上位50%の平均を使う
    live_24h_retr_deadline_extend: bool = False  # 継続中の連鎖の終了を延長する
    live_24h_xc_same_time: bool = False  # 行の時刻の観測から xc だけを選ぶ
    live_24h_ex_fire_first: bool = False  # 新しい撃ち合いを空き計算子へ渡す
    live_24h_early_submit: bool = False  # 評価通知の受理直後にEXを要求する。
    live_24h_ex_fresh_sec: float = EX_FRESH_SEC_DEFAULT  # 元のEX入力からの鮮度（秒）。


def _speed_config(data: dict[str, Any]) -> dict[str, Any]:
    """配布試験の高速化設定を既存の日本語エラーへ変換する。"""
    try:
        values = validate_speed(data)
        values[FRAME_DROP_RULES_OFF_KEY] = rules_off_of(data, values)
        return values
    except ValueError as error:
        raise LauncherConfigError(str(error)) from error


def _instance_config(data: dict[str, Any]) -> bool:
    """二重起動を許可するか。boolだけ受け付ける (文字列 "false" が真になる事故を防ぐ)。"""
    value = data.get('allow_second_instance', ALLOW_SECOND_INSTANCE_DEFAULT)
    if type(value) is not bool:
        raise LauncherConfigError('allow_second_instance は true / false です')
    return value


def _overlay_config(data: dict[str, Any]) -> dict[str, Any]:
    """表示設定のエラーも既存ランチャの日本語エラー経路へ流す。"""
    try:
        return validate_overlay(data.get('overlay', {}))
    except ValueError as error:
        raise LauncherConfigError(str(error)) from error


def _c2_config(data: dict[str, Any], asynchronous: bool, m3: bool) -> dict[str, Any]:
    """b3_c2 (前提: b3_async・b3_m3) と b3_ensemble (前提: b3_c2) の型と前提を検査する。どちらも既定 OFF。"""
    c2 = data.get('b3_c2', False)
    if type(c2) is not bool or (c2 and not (asynchronous and m3)):
        raise LauncherConfigError('b3_c2はboolで、trueにはb3_async=trueとb3_m3=trueが必要です')
    ensemble = data.get('b3_ensemble', B3_ENSEMBLE_DEFAULT)
    if type(ensemble) is not bool or (ensemble and not c2):
        raise LauncherConfigError('b3_ensembleはboolで、trueにはb3_c2=trueが必要です')
    return dict(b3_c2=c2, b3_ensemble=ensemble)


def _b3_config(data: dict[str, Any], base: Path) -> dict[str, Any]:
    """B3は既定OFF。指定ファイルは起動前に検査し、旧判定へ黙って戻さない。"""
    enabled = data.get('b3_judgment', False)
    if type(enabled) is not bool:
        raise LauncherConfigError('b3_judgment は true / false です')
    values: dict[str, Any] = dict(b3_judgment=enabled)
    fast_first = data.get('fast_first_judgment', FAST_FIRST_JUDGMENT_DEFAULT)
    if type(fast_first) is not bool:
        raise LauncherConfigError('fast_first_judgment はboolが必要です')
    values['fast_first_judgment'] = fast_first
    for key, default in (('recovery_boundary_coalescing', RECOVERY_BOUNDARY_COALESCING_DEFAULT),
                         ('evaluation_freshness_gate', EVALUATION_FRESHNESS_GATE_DEFAULT),
                         ('terminal_display_hold', TERMINAL_DISPLAY_HOLD_DEFAULT)):
        value = data.get(key, default)
        if type(value) is not bool:
            raise LauncherConfigError(f'{key} はboolが必要です')
        values[key] = value
    asynchronous = data.get('b3_async', False)
    if type(asynchronous) is not bool or (asynchronous and not enabled):
        raise LauncherConfigError('b3_async はboolで、trueにはb3_judgment=trueが必要です')
    values['b3_async'] = asynchronous
    from .live_fire_board_config import user_value
    values['live_b3_fire_board'] = user_value(data, LauncherConfigError)
    trial = data.get('live_v49f', False)
    if type(trial) is not bool or (trial and not asynchronous):
        raise LauncherConfigError('live_v49fはboolで、trueにはb3_async=trueが必要です')
    values['live_v49f'] = trial
    for key in ('live_ama_plain', 'b3_m3', 'b3_time_gate', 'live_score_cadence', 'live_static_exact'):
        value = data.get(key, False)
        if type(value) is not bool or (value and not asynchronous):
            raise LauncherConfigError(f'{key}はboolで、trueにはb3_async=trueが必要です')
        values[key] = value
    values.update(_c2_config(data, asynchronous, values['b3_m3']))
    for key, default in (('display_completion_driven', DISPLAY_COMPLETION_DRIVEN_DEFAULT),
                         ('display_fast_ema', DISPLAY_FAST_EMA_DEFAULT)):
        value = data.get(key, default)
        if type(value) is not bool:
            raise LauncherConfigError(f'{key} はboolが必要です')
        values[key] = value and enabled
    for key, default in (('b3_model', 'model.joblib'), ('b3_timing_table', 'timing_table.json')):
        raw = data.get(key)
        if raw is not None and (not isinstance(raw, str) or not raw.strip()):
            raise LauncherConfigError(f'{key} は空でないパスです')
        path = _resolve(base, raw) if raw else APP_ROOT / 'models' / 'b3' / default
        if enabled and not path.is_file():
            raise LauncherConfigError(f'{key} がありません: {path}')
        values[key] = path if enabled or raw else None
    return values


def _b3_and_speed_config(data: dict[str, Any], base: Path) -> dict[str, Any]:
    """B3 と高速化の設定を検証して合わせる。b3_c2 は予告アイコンを読まない構成 (b3_skip_icon_reading=true) が前提。"""
    values = dict(**_b3_config(data, base), **_speed_config(data))   # 従来の ** 展開どおり、キーの重複は TypeError
    if values['b3_c2'] and not values['b3_skip_icon_reading']:
        raise LauncherConfigError('b3_c2は予告アイコンを読まない構成です。b3_skip_icon_readingをtrueにしてください')
    from .live_24h_config import launcher_values
    try:
        from .live_power import power_enabled, validate_power
        values['live_disable_power_throttling'] = validate_power(data.get('live_disable_power_throttling', power_enabled()))
        values.update(launcher_values(data, base))
    except ValueError as error:
        raise LauncherConfigError(str(error)) from error
    return values


def _int(data: dict[str, Any], key: str, default: int, low: int, high: int | None = None) -> int:
    value = data.get(key, default)
    if type(value) is not int or value < low or (high is not None and value > high):
        span = f'{low}以上' if high is None else f'{low}〜{high}'
        raise LauncherConfigError(f'"{key}" は{span}の整数で指定してください (現在: {value!r})')
    return value


def _optional_sec(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise LauncherConfigError(f'"{key}" は0以上の秒数で指定してください (現在: {value!r})')
    return float(value)


def _calibration_location(data: dict[str, Any]) -> str:
    value = data.get('calibration_location', CALIBRATION_APP)
    if value not in (CALIBRATION_APP, CALIBRATION_LOCALAPPDATA):
        raise LauncherConfigError(f'"calibration_location" は "{CALIBRATION_APP}" (既定: アプリ内) か '
                                  f'"{CALIBRATION_LOCALAPPDATA}" (%LOCALAPPDATA%) です (現在: {value!r})')
    return value


def _color_correction(data: dict[str, Any]) -> str:
    value = data.get('dshow_color_correction', DEFAULT_COLOR_CORRECTION)
    if value not in COLOR_CORRECTION_MODES:
        raise LauncherConfigError(f'"dshow_color_correction" は {list(COLOR_CORRECTION_MODES)} のいずれかです '
                                  f'(既定: "{DEFAULT_COLOR_CORRECTION}"、現在: {value!r})')
    return value


def calibration_dir(user: UserConfig, environ: dict[str, str] | None = None) -> Path | None:
    """機器別較正の保存先。app (既定) は None = 既存の config/device_calibration のまま。"""
    if user.calibration_location == CALIBRATION_APP:
        return None
    base = (os.environ if environ is None else environ).get('LOCALAPPDATA')
    if not base:
        raise LauncherConfigError('calibration_location が "localappdata" ですが、環境変数 LOCALAPPDATA が'
                                  '設定されていません (Windows 以外では "app" を使ってください)')
    return Path(base) / LOCALAPPDATA_SUBDIR


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


VIDEO_REJECTED_MESSAGE = ('配布版は OBS 仮想カメラ等の入力 (DirectShow) のみ対応です。動画ファイル入力 (source="video") は'
                          '使えません。"source" を "dshow" にして device_name / device_index を指定してください')


def parse_user_config(data: Any, base_dir: Path, allow_video: bool = False) -> UserConfig:
    """渡された辞書だけを検証する低水準API。相対パスは設定ファイルの場所が基準。
    配布既定の補完はload_user_configで行い、開発試験の直接呼出しは従来値を保つ。
    動画入力 (source=video) は配布版では受け付けない (FFmpeg を同梱しないため)。allow_video は開発用。"""
    if not isinstance(data, dict):
        raise LauncherConfigError('設定ファイルの最上位は { ... } のオブジェクトにしてください')
    unknown = sorted(set(data) - USER_KEYS)
    if unknown:
        raise LauncherConfigError(f'未知の設定項目があります: {unknown} (使える項目: {sorted(USER_KEYS)})')
    source = data.get('source', SOURCE_DSHOW)
    if source not in (SOURCE_DSHOW, SOURCE_VIDEO):
        raise LauncherConfigError(f'"source" は "{SOURCE_DSHOW}" か "{SOURCE_VIDEO}" です (現在: {source!r})')
    if source == SOURCE_VIDEO and not allow_video:
        raise LauncherConfigError(VIDEO_REJECTED_MESSAGE)
    name, video = data.get('device_name'), data.get('video_path')
    if source == SOURCE_DSHOW and (not isinstance(name, str) or not name.strip()):
        raise LauncherConfigError('"device_name" (例: "OBS Virtual Camera") を指定してください。'
                                  '機器名だけで自動選択はしません')
    if source == SOURCE_VIDEO and (not isinstance(video, str) or not video.strip()):
        raise LauncherConfigError('source が "video" のときは "video_path" が必要です')
    host = data.get('host', DEFAULT_HOST)
    if not isinstance(host, str) or not host.strip():
        raise LauncherConfigError(f'"host" は空でない文字列にしてください (現在: {host!r})')
    return UserConfig(source=source, device_name=name.strip() if isinstance(name, str) else None,
        device_index=_int(data, 'device_index', 0, 0), video_path=_resolve(base_dir, video) if video else None,
        port=_int(data, 'port', DEFAULT_PORT, MIN_PORT, MAX_PORT), host=host.strip(),
        output_dir=_resolve(base_dir, str(data.get('output_dir', DEFAULT_OUTPUT_DIR))),
        mc_rollouts=_int(data, 'mc_rollouts', DEFAULT_MC_ROLLOUTS, 1),
        start_sec=_optional_sec(data, 'start_sec'), end_sec=_optional_sec(data, 'end_sec'),
        calibration_location=_calibration_location(data),
        dshow_color_correction=_color_correction(data), overlay=_overlay_config(data),
        allow_second_instance=_instance_config(data), **_b3_and_speed_config(data, base_dir))


def load_user_config(path: Path, allow_video: bool = False) -> UserConfig:
    """配布既定を補って読む。明示値を優先し、不正・資産不足は起動前に知らせる。"""
    if not path.is_file():
        raise LauncherConfigError(f'設定ファイルがありません: {path}\n'
                                  'tools/puyo_live.example.json をコピーして puyo_live.json を作ってください')
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError as error:
        raise LauncherConfigError(f'設定ファイルの JSON が不正です: {path} '
                                  f'({error.lineno}行{error.colno}列: {error.msg})') from error
    if isinstance(data, dict):
        data = release_user_config(data, APP_ROOT)
    return parse_user_config(data, path.resolve().parent, allow_video)


def build_pipeline_config(user: UserConfig) -> dict[str, Any]:
    """既存 apply_config が受け取る --config JSON (live_dshow/live_video.example.json と同形)。"""
    config: dict[str, Any] = dict(source=user.source, name=user.device_name or user.video_path.stem,
        index=user.device_index, mc_rollouts=user.mc_rollouts, host=user.host, port=user.port,
        **PIPELINE_FIXED)
    if user.source == SOURCE_DSHOW and user.dshow_color_correction != DEFAULT_COLOR_CORRECTION:
        config['color_correction'] = user.dshow_color_correction  # 既定 off のときはキー自体を出さない
    if user.source == SOURCE_VIDEO:
        config.update(realtime=True, warmup_sec=VIDEO_WARMUP_SEC)
        for key in ('start_sec', 'end_sec'):
            if getattr(user, key) is not None:
                config[key] = getattr(user, key)
    if user.b3_judgment:
        config.update(b3_judgment=True, b3_model=str(user.b3_model),
                      b3_timing_table=str(user.b3_timing_table))
        if user.b3_async:
            config['b3_async'] = True
        if user.live_b3_fire_board:
            config['live_b3_fire_board'] = True
        if user.live_v49f:
            config['live_v49f'] = True
        for key in ('live_ama_plain', 'b3_m3', 'b3_time_gate', 'live_score_cadence', 'live_static_exact'):
            if getattr(user, key):
                config[key] = True
        if user.b3_c2:   # OFF のときはキー自体を出さない (従来の pipeline_config.json と同じ)
            config['b3_c2'] = True
        if user.b3_ensemble:   # 同じく OFF のときはキー自体を出さない
            config['b3_ensemble'] = True
        config['display_completion_driven'] = user.display_completion_driven
        config['display_fast_ema'] = user.display_fast_ema
    if user.overlay:
        config['overlay'] = dict(user.overlay)
    if user.fast_first_judgment:
        config['fast_first_judgment'] = True
    if user.recovery_boundary_coalescing:
        config['recovery_boundary_coalescing'] = True
    if user.evaluation_freshness_gate:
        config['evaluation_freshness_gate'] = True
    if user.terminal_display_hold:
        config['terminal_display_hold'] = True
    for key in SPEED_DEFAULTS:
        # 既定ONの明示OFFも配送する。省略すると子CLIの既定ONへ戻ってしまう。
        if getattr(user, key) or SPEED_DEFAULTS[key]:
            config[key] = getattr(user, key)
    if user.b3_warning_parallel:
        config[WORKERS_KEY] = user.b3_warning_workers
    if user.live_frame_drop_rules_off:   # 既定の空は書かない (従来の pipeline_config.json と同じ)
        config[FRAME_DROP_RULES_OFF_KEY] = list(user.live_frame_drop_rules_off)
    from .live_24h_config import pipeline_values
    config.update(pipeline_values(user), **power_pipeline_values(user))
    return config


def build_pipeline_argv(user: UserConfig, pipeline_config: Path) -> list[str]:
    """既存パイプラインの CLI 引数列 (プログラム名は含まない)。"""
    argv = ['--config', str(pipeline_config), '--output', str(user.output_dir)]
    if user.source == SOURCE_VIDEO:
        argv += ['--video', str(user.video_path)]
    return argv


def site_packages_of(root: Path) -> Path | None:
    """配布版 (root の隣に python/Lib/site-packages) のときだけパスを返す。"""
    path = root.parent / 'python' / 'Lib' / 'site-packages'
    return path if path.is_dir() else None


def check_bundle(root: Path, require: bool) -> str | None:
    """MANIFEST 照合。問題があれば表示文字列、無ければ None。無い場合 require なら失敗扱い。"""
    manifest = root / MANIFEST_FILENAME
    if not manifest.is_file():
        return f'{MANIFEST_FILENAME} がありません (配布物が不完全です)' if require else None
    try:
        report = verify_manifest(root, manifest, site_packages_of(root))
    except ManifestError as error:
        return str(error)
    return None if report.ok else report.summary()


def probe_devices(max_index: int = MAX_PROBE_INDEX, factory: Callable[..., Any] | None = None) -> list[dict[str, Any]]:
    """DirectShow の機器 index を順に開き、開けたか・1 フレーム読めたか・解像度を返す。
    機器は device_index で選ぶ (device_name は較正保存用の名札。live_device.py:135)。名前列挙は OpenCV に無いため
    index ごとの実測を利用者に見せ、OBS 仮想カメラ等の index を自分で確かめられるようにする。"""
    if factory is None:
        import cv2
        factory = cv2.VideoCapture
        args: tuple = (cv2.CAP_DSHOW,)
    else:
        args = ()
    rows: list[dict[str, Any]] = []
    for index in range(max_index + 1):
        capture = factory(index, *args)
        row: dict[str, Any] = dict(index=index, opened=bool(capture.isOpened()), frame=False, size=None)
        if row['opened']:
            ok, image = capture.read()
            row.update(frame=bool(ok), size=(image.shape[1], image.shape[0]) if ok else None)
        capture.release()
        rows.append(row)
    return rows


def format_devices(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        if not row['opened']:
            lines.append(f"index {row['index']}: 開けません (無い / 他のアプリが使用中)")
        elif row['frame']:
            lines.append(f"index {row['index']}: 映像あり {row['size'][0]}x{row['size'][1]}")
        else:
            lines.append(f"index {row['index']}: 開けたが映像を読めません")
    return '\n'.join(lines)


def parse_launcher_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='puyoZeus Live — ぷよぷよeスポーツ対戦分析')
    parser.add_argument('--config', type=Path, default=DEFAULT_USER_CONFIG, help='利用者設定 (JSON)')
    parser.add_argument('--require-manifest', action='store_true', help='MANIFEST 無しを失敗にする (配布版)')
    parser.add_argument('--skip-manifest', action='store_true', help='完全性照合を省く (開発用)')
    parser.add_argument('--check-only', action='store_true', help='設定と完全性だけ確認して終了')
    parser.add_argument('--dev-allow-video', action='store_true', help='開発用: source=video を許可 (配布版では動画デコーダが無く動かない)')
    parser.add_argument('--list-devices', action='store_true', help='機器 index ごとの映像有無を表示して終了')
    return parser.parse_args(argv)


def prepare(options: argparse.Namespace) -> tuple[UserConfig, list[str]]:
    """設定検証→完全性照合→パイプライン設定の書き出し。失敗は SystemExit(終了コード)。"""
    try:
        user = load_user_config(options.config, options.dev_allow_video)
    except LauncherConfigError as error:
        print(f'[設定エラー] {error}', file=sys.stderr)
        raise SystemExit(EXIT_CONFIG) from error
    if not getattr(options, 'check_only', False):
        apply_instance_guard(user)   # 起動の最初 (出力先・設定の書き出しより前) に二重起動を止める
    missing = missing_runtime_dlls(Path(sys.executable).parent)
    if missing:
        print(f'[実行環境エラー] {runtime_guidance(missing)}', file=sys.stderr)
        raise SystemExit(EXIT_RUNTIME)
    check_telop_templates(APP_ROOT)
    problem = None if options.skip_manifest else check_bundle(APP_ROOT, options.require_manifest)
    if problem:
        print(f'[配布物エラー] {problem}', file=sys.stderr)
        raise SystemExit(EXIT_MANIFEST)
    user.output_dir.mkdir(parents=True, exist_ok=True)
    apply_calibration_dir(user)
    apply_cnn_backend()
    os.environ.update(release_environment(dict(os.environ)))
    pipeline_config = user.output_dir / PIPELINE_CONFIG_NAME
    pipeline_config.write_text(json.dumps(build_pipeline_config(user), ensure_ascii=False, indent=2),
                               encoding='utf-8')
    return user, build_pipeline_argv(user, pipeline_config)


def apply_instance_guard(user: UserConfig) -> None:
    """同じ port・同じカメラの PuyoLive がすでに動いていれば、案内を出して SystemExit(EXIT_ALREADY_RUNNING)。
    許可設定 (allow_second_instance) は、評価 process (spawn 子) の排他 bind へ環境変数で渡す。"""
    from .live_instance_guard import ALLOW_SECOND_INSTANCE_ENV, guard_instances, instance_keys
    os.environ[ALLOW_SECOND_INSTANCE_ENV] = str(int(user.allow_second_instance))
    guard_instances(instance_keys(user.source, user.host, user.port, user.device_index),
                    user.host, user.port, user.allow_second_instance)


def check_telop_templates(app_root: Path = APP_ROOT) -> int:
    """起動時に実際に画像を読み、検出の黙った無効化を警告する。"""
    from src.telop_detector import DEFAULT_TEMPLATE_DIR, TelopDetector
    detector = TelopDetector.load_default(app_root / DEFAULT_TEMPLATE_DIR)
    count = detector.template_count
    if count:
        print(f'[テロップ検出] 有効: テンプレート {count} 枚', flush=True)
    else:
        print('[警告] telop_*.png が無い、または読み込めません。'
              '中央テロップ検出が無効です。配布物を復元してください。', file=sys.stderr, flush=True)
    return count


def apply_calibration_dir(user: UserConfig) -> None:
    """設定が localappdata のときだけ環境変数で live_device へ渡す (spawn 子へも継承される)。"""
    try:
        target = calibration_dir(user)
    except LauncherConfigError as error:
        print(f'[設定エラー] {error}', file=sys.stderr)
        raise SystemExit(EXIT_CONFIG) from error
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        os.environ[CALIBRATION_DIR_ENV] = str(target)


def apply_cnn_backend(app_root: Path = APP_ROOT) -> str | None:
    """ONNX の CNN が同梱された配布物では ONNX Runtime 版を既定にする (spawn 子へも環境変数で継承)。

    torch 版と ONNX 版は全セル argmax 0 不一致 (2,073,168 セル) かつパイプライン出力 0 差 (docs/PHASE_J_ONNX_PREREGISTRATION)。
    利用者が環境変数を明示済みならそれを尊重する。同梱が無い配布物では何も設定しない (torch 版のまま)。
    ONNX を要求して使えない場合は cnn_onnx が例外で止まる (黙って torch へ戻さない)。"""
    if CNN_BACKEND_ENV not in os.environ and (app_root / ONNX_INDEX).is_file():
        os.environ[CNN_BACKEND_ENV] = CNN_BACKEND_ONNX
    return os.environ.get(CNN_BACKEND_ENV)


def overlay_url(user: UserConfig) -> str:
    return f'http://{user.host}:{user.port}/'


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_launcher_args(argv)
    if options.list_devices:  # 設定ファイル不要 (index を調べて設定へ書くための道具)
        print(format_devices(probe_devices()))
        return EXIT_OK
    user, pipeline_argv = prepare(options)
    print(f'[起動] OBS ブラウザソース URL: {overlay_url(user)}  (入力: {user.source})', flush=True)
    if options.check_only:
        return EXIT_OK
    os.chdir(APP_ROOT)  # 既存パイプラインは config/ models/ data/ を相対パスで読む
    sys.path.insert(0, str(APP_ROOT))
    sys.argv = [PIPELINE_MODULE, *pipeline_argv]
    from scripts.run_live_pipeline_20260928 import main as pipeline_main
    pipeline_main()
    return EXIT_OK


if __name__ == '__main__':
    raise SystemExit(main())
