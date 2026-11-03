"""実配布物を隔離先へ導入し、保存・削除とzipの登録解除を検証する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import winreg
import zipfile

REGISTRY_KEY = r'Software\Microsoft\Windows\CurrentVersion\Uninstall\PuyoLive-EarlyAccess-Distribution_is1'
TIMEOUT = 1800
ZIP_WORKERS = 8
POLL_SECONDS = 0.25
RUNNING_PROBE_SECONDS = 120
DATA = ('output/record.json', 'logs/runtime.log', 'distribution.json', 'puyo_live.json',
        'install.log', 'app/config/device_calibration/device.json')
TEMPORARY = ('distribution.json.tmp', 'logs/instance.lock', 'logs/session.running',
             'python/Lib/site-packages/__pycache__/uninstall_probe.pyc')
SOURCE = Path(__file__).resolve().parents[1]


def run(command: list[str], **options: object) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=TIMEOUT, **options)


def registered() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REGISTRY_KEY):
            return True
    except FileNotFoundError:
        return False


def populate(root: Path) -> list[Path]:
    """実リンクを作り、データと一時ファイルを置く。"""
    command = ('from pathlib import Path; from src.phase_j.distribution_windows import shortcuts, '
               'shortcut_name, shortcut_directories; import json; '
               f'r=Path({str(root)!r}); '
               'shortcuts(r,dict(desktop_shortcut=True,start_menu_shortcut=True,autostart=True)); '
               'print(json.dumps([str(p/shortcut_name(r)) for p in shortcut_directories().values()]))')
    result = run([str(root / 'python/python.exe'), '-B', '-c', command], cwd=root)
    links = [Path(p) for p in json.loads(result.stdout)]
    assert all(p.exists() for p in links), links
    for name in DATA + TEMPORARY:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}' if name.endswith('.json') else 'probe', encoding='utf-8')
    return links


def inspect(root: Path, links: list[Path], delete: bool) -> dict:
    """従来のcleanup.jsonと同じ項目に、保存データと一時ファイルを追加する。"""
    assert all((root / name).exists() != delete for name in DATA)
    assert all(not (root / name).exists() for name in TEMPORARY)
    assert all(not p.exists() for p in links)
    script = ('@(Get-CimInstance Win32_Process | Where-Object { '
              '$_.ExecutablePath -and $_.ExecutablePath.StartsWith($env:PUYO_TRIAL) }).Count')
    count = int(run(['powershell', '-NoProfile', '-Command', script],
                    env={**os.environ, 'PUYO_TRIAL': str(root)}).stdout.strip())
    assert count == 0
    return dict(uninstall_registration_exists=registered(), trial_process_count=count,
                trial_folder_exists=root.exists(), shortcuts_remaining={
                    name: path.exists() for name, path in zip(('desktop', 'start_menu', 'startup'), links)},
                data_preserved=not delete, temporary_files_remaining=False,
                remaining_files=[p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()])


def running_refused(root: Path, command: list[str], links: list[Path]) -> None:
    """実ロックを持った配布プロセスがある間は登録もデータも消さない。"""
    code = ('from pathlib import Path; from src.phase_j.distribution_app import acquire_instance; '
            f'lock=acquire_instance(Path({str(root)!r})); '
            f'assert lock; print("ready",flush=True); import time; time.sleep({RUNNING_PROBE_SECONDS})')
    child = subprocess.Popen([str(root / 'python/python.exe'), '-B', '-c', code],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        assert child.stdout.readline().strip() == 'ready'
        result = subprocess.run(command, timeout=TIMEOUT, capture_output=True)
        assert result.returncode != 0, result
        assert registered() and all(p.exists() for p in links)
        assert all((root / name).exists() for name in DATA)
    finally:
        child.terminate()
        child.wait(timeout=TIMEOUT)


def installed(output: Path, installer: Path, delete: bool) -> dict:
    root = output / 'installed_trial_uninstaller'
    language = 'en' if delete else 'ja'
    run([str(installer), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
         '/SP-', f'/LANG={language}', f'/DIR={root}', f'/LOG={output / (language + "-install.log")}'])
    assert registered() and (root / 'unins000.exe').exists()
    group = Path(os.environ['APPDATA']) / 'Microsoft/Windows/Start Menu/Programs/puyoZeus Live'
    icon = group / ('Uninstall puyoZeus Live.lnk' if delete else 'puyoZeus Live をアンインストール.lnk')
    assert icon.exists(), icon
    links = populate(root)
    command = [str(root / 'unins000.exe'), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
               f'/LOG={output / (language + "-uninstall.log")}']
    if delete:
        command.append('/DELETEUSERDATA=1')
    else:
        running_refused(root, command, links)
    run(command)
    deadline = time.monotonic() + TIMEOUT
    while (root / 'unins000.exe').exists() and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
    result = inspect(root, links, delete)
    assert not registered() and not icon.exists()
    assert result['remaining_files'] == [] if delete else set(result['remaining_files']) == set(DATA)
    if delete:
        assert not root.exists(), result
    result['installer_temporary_folder_exists'] = verify_installer_temp(output / (language + '-uninstall.log'))
    return result


def verify_installer_temp(log: Path) -> bool:
    """Inno自身の一時実行ファイルも、終了後に消えることを確認する。"""
    marker = 'puyoZeus temporary cleanup: '
    lines = log.read_text(encoding='utf-8-sig').splitlines()
    targets = [Path(line.split(marker, 1)[1]) for line in lines if marker in line]
    assert len(targets) == 1, '一時フォルダの後片付けが起動されていません'
    deadline = time.monotonic() + TIMEOUT
    while targets[0].exists() and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
    assert not targets[0].exists(), targets[0]
    return False


def extract_portable(archive: Path, parent: Path) -> None:
    """展開だけを並列化し、独立した試験ディレクトリへの書き込み待ちを減らす。"""
    with zipfile.ZipFile(archive) as source:
        assert source.testzip() is None
        names = [item.filename for item in source.infolist() if not item.is_dir()]
    for name in names:
        (parent / name).parent.mkdir(parents=True, exist_ok=True)
    def extract(names: list[str]) -> None:
        with zipfile.ZipFile(archive) as source:
            for name in names:
                source.extract(name, parent)
    with ThreadPoolExecutor(max_workers=ZIP_WORKERS) as executor:
        list(executor.map(extract, [names[i::ZIP_WORKERS] for i in range(ZIP_WORKERS)]))


def portable(output: Path, archive: Path) -> dict:
    parent = output / 'zip_trial_uninstaller'
    assert not parent.exists()
    extract_portable(archive, parent)
    root = parent / 'PuyoLive'
    results = {}
    for delete, name, answer in ((False, 'アンインストール.bat', '\n\n'), (True, 'Uninstall.bat', 'y\n\n')):
        links = populate(root)
        result = run(['cmd.exe', '/d', '/c', str(root / name)], input=answer, cwd=root)
        assert 'Shortcuts and startup removed' in result.stdout, result.stdout
        report = inspect(root, links, delete)
        report['remaining_file_count'] = len(report.pop('remaining_files'))
        results['delete' if delete else 'keep'] = report
    # zip版の案内に従うフォルダ削除も、検証専用の直下パスに限定する。
    assert parent.resolve().parent == output.resolve() and parent.name == 'zip_trial_uninstaller'
    shutil.rmtree(parent)
    results['folder_removed_after_guidance'] = not parent.exists()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output == Path('D:/puyo_analyzer/b3_data/codex_tasks/uninstaller_release').resolve()
    assert not registered(), '既存インストールがあるため検証を中止します'
    assert not (output / 'installed_trial_uninstaller').exists()
    ledger = json.loads((output / 'releases.json').read_text(encoding='utf-8'))
    files = [output / item['name'] for item in ledger['files']]
    for path, item in zip(files, ledger['files']):
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
    installer = next(p for p in files if p.suffix == '.exe')
    archive = next(p for p in files if p.suffix == '.zip')
    result = dict(installer_keep=installed(output, installer, False),
                  installer_delete=installed(output, installer, True), zip=portable(output, archive))
    (output / 'cleanup.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('実インストール・zipとも保存/削除と残留物検証に合格', flush=True)


if __name__ == '__main__':
    main()
