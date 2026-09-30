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
PIPELINE_FIXED = dict(verification_only=True, coalesce_features=True, cpu_threads=1,
                      evaluation_nice=10, cnn_device='cpu')
VIDEO_WARMUP_SEC = 1.0

EXIT_OK, EXIT_CONFIG, EXIT_MANIFEST = 0, 2, 3
CALIBRATION_APP, CALIBRATION_LOCALAPPDATA = 'app', 'localappdata'  # 機器別較正の保存先 (既定は app)
# live_device.CALIBRATION_DIR_ENV と同値 (live_device は cv2 を import するためここでは複製し、試験で一致を固定)
CALIBRATION_DIR_ENV = 'PUYO_CALIBRATION_DIR'
LOCALAPPDATA_SUBDIR =Path('PuyoLive') / 'device_calibration'
USER_KEYS = frozenset({'source', 'device_name', 'device_index', 'video_path', 'port', 'host',
                       'output_dir', 'mc_rollouts', 'start_sec', 'end_sec', 'calibration_location'})


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


def parse_user_config(data: Any, base_dir: Path) -> UserConfig:
    """辞書を検証して UserConfig にする。相対パスは設定ファイルの場所が基準。"""
    if not isinstance(data, dict):
        raise LauncherConfigError('設定ファイルの最上位は { ... } のオブジェクトにしてください')
    unknown = sorted(set(data) - USER_KEYS)
    if unknown:
        raise LauncherConfigError(f'未知の設定項目があります: {unknown} (使える項目: {sorted(USER_KEYS)})')
    source = data.get('source', SOURCE_DSHOW)
    if source not in (SOURCE_DSHOW, SOURCE_VIDEO):
        raise LauncherConfigError(f'"source" は "{SOURCE_DSHOW}" か "{SOURCE_VIDEO}" です (現在: {source!r})')
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
        calibration_location=_calibration_location(data))


def load_user_config(path: Path) -> UserConfig:
    """設定ファイルを読む。無い・壊れている場合は直し方つきで LauncherConfigError。"""
    if not path.is_file():
        raise LauncherConfigError(f'設定ファイルがありません: {path}\n'
                                  'puyo_live.example.json をコピーして puyo_live.json を作ってください')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise LauncherConfigError(f'設定ファイルの JSON が不正です: {path} '
                                  f'({error.lineno}行{error.colno}列: {error.msg})') from error
    return parse_user_config(data, path.resolve().parent)


def build_pipeline_config(user: UserConfig) -> dict[str, Any]:
    """既存 apply_config が受け取る --config JSON (live_dshow/live_video.example.json と同形)。"""
    config: dict[str, Any] = dict(source=user.source, name=user.device_name or user.video_path.stem,
        index=user.device_index, mc_rollouts=user.mc_rollouts, host=user.host, port=user.port,
        **PIPELINE_FIXED)
    if user.source == SOURCE_VIDEO:
        config.update(realtime=True, warmup_sec=VIDEO_WARMUP_SEC)
        for key in ('start_sec', 'end_sec'):
            if getattr(user, key) is not None:
                config[key] = getattr(user, key)
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
    parser = argparse.ArgumentParser(description='ぷよぷよ有利不利オーバーレイ ランチャー')
    parser.add_argument('--config', type=Path, default=DEFAULT_USER_CONFIG, help='利用者設定 (JSON)')
    parser.add_argument('--require-manifest', action='store_true', help='MANIFEST 無しを失敗にする (配布版)')
    parser.add_argument('--skip-manifest', action='store_true', help='完全性照合を省く (開発用)')
    parser.add_argument('--check-only', action='store_true', help='設定と完全性だけ確認して終了')
    parser.add_argument('--list-devices', action='store_true', help='機器 index ごとの映像有無を表示して終了')
    return parser.parse_args(argv)


def prepare(options: argparse.Namespace) -> tuple[UserConfig, list[str]]:
    """設定検証→完全性照合→パイプライン設定の書き出し。失敗は SystemExit(終了コード)。"""
    try:
        user = load_user_config(options.config)
    except LauncherConfigError as error:
        print(f'[設定エラー] {error}', file=sys.stderr)
        raise SystemExit(EXIT_CONFIG) from error
    missing = missing_runtime_dlls(Path(sys.executable).parent)
    if missing:
        print(f'[実行環境エラー] {runtime_guidance(missing)}', file=sys.stderr)
        raise SystemExit(EXIT_RUNTIME)
    problem = None if options.skip_manifest else check_bundle(APP_ROOT, options.require_manifest)
    if problem:
        print(f'[配布物エラー] {problem}', file=sys.stderr)
        raise SystemExit(EXIT_MANIFEST)
    user.output_dir.mkdir(parents=True, exist_ok=True)
    apply_calibration_dir(user)
    pipeline_config = user.output_dir / PIPELINE_CONFIG_NAME
    pipeline_config.write_text(json.dumps(build_pipeline_config(user), ensure_ascii=False, indent=2),
                               encoding='utf-8')
    return user, build_pipeline_argv(user, pipeline_config)


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
