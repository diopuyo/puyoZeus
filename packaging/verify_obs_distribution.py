"""隔離した実インストールでGUI起動・図形入力・削除を試験する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen
import winreg

from playwright.sync_api import sync_playwright

OUTPUT = Path('D:/puyo_analyzer/b3_data/codex_tasks/dist_app_release')
ROOT = OUTPUT / 'installed_trial_obs'
SOURCE = Path(__file__).resolve().parent
FRAMES = Path('D:/puyo_analyzer/local_frames/obs_integration')
PORT = 18785
URL = f'http://127.0.0.1:{PORT}'
REGISTRY = r'Software\Microsoft\Windows\CurrentVersion\Uninstall\PuyoLive-EarlyAccess-Distribution_is1'
TIMEOUT = 180
INSTALL_TIMEOUT = 1800
POLL_SECONDS = .2


def registered() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REGISTRY):
            return True
    except FileNotFoundError:
        return False


def wait_ready() -> dict:
    deadline = time.monotonic()+TIMEOUT
    while time.monotonic() < deadline:
        try:
            with urlopen(URL+'/api/state',timeout=2) as response:
                return json.load(response)
        except OSError:
            time.sleep(POLL_SECONDS)
    raise TimeoutError('試験サーバーの起動待ちが時間切れ')


def run(command: list[str]) -> None:
    subprocess.run(command,check=True,timeout=INSTALL_TIMEOUT,creationflags=subprocess.CREATE_NO_WINDOW)


def gui_trial() -> dict:
    settings = dict(completed=False,check_updates=False,start_menu_shortcut=False,obs_password='trial-local-secret')
    (ROOT/'distribution.json').write_text(json.dumps(settings),encoding='utf-8-sig')
    command = [str(ROOT/'python/pythonw.exe'),'-m','src.phase_j.distribution_app','--root',str(ROOT),
               '--port',str(PORT),'--no-browser','--setup']
    child = subprocess.Popen(command,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        state = wait_ready()
        assert 'trial-local-secret' not in json.dumps(state)
        run(command)
        assert child.poll() is None
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='msedge',headless=True)
            page = browser.new_page(viewport=dict(width=1440,height=1000))
            page.goto(URL+'/settings')
            page.locator('#obs-check').scroll_into_view_if_needed()
            page.screenshot(path=str(FRAMES/'installed-settings.png'))
            browser.close()
        return dict(gui_pythonw=True,bom=True,duplicate_blocked=True,password_not_exposed=True)
    finally:
        request = Request(URL+'/api/stop',b'{}',{'Content-Type':'application/json'})
        try:
            urlopen(request,timeout=10).close()
            child.wait(timeout=TIMEOUT)
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=TIMEOUT)


def synthetic_trial() -> dict:
    script = SOURCE/'smoke_synthetic_input.py'
    command = [str(ROOT/'python/python.exe'),str(script),'--root',str(ROOT),'--port',str(PORT)]
    with (OUTPUT/'obs-synthetic.log').open('w',encoding='utf-8') as log:
        child = subprocess.Popen(command,cwd=ROOT/'app',stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            wait_ready()
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel='msedge',headless=True)
                page = browser.new_page(viewport=dict(width=1920,height=1080))
                page.goto(URL+'/?language=ja')
                page.wait_for_function("document.querySelector('#state').textContent==='試合待ち'",timeout=TIMEOUT*1000)
                page.screenshot(path=str(FRAMES/'installed-synthetic-waiting.png'))
                page.wait_for_function("document.querySelector('#state').textContent.includes('映り込み')",timeout=TIMEOUT*1000)
                page.screenshot(path=str(FRAMES/'installed-feedback-warning.png'))
                browser.close()
            assert child.wait(timeout=TIMEOUT) == 0
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=TIMEOUT)
    return json.loads((OUTPUT/'synthetic-input-smoke.json').read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-installed',action='store_true')
    options = parser.parse_args()
    assert ROOT.resolve().parent == OUTPUT.resolve() and ROOT.name == 'installed_trial_obs'
    FRAMES.mkdir(parents=True,exist_ok=True)
    if options.resume_installed:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,REGISTRY) as key:
            assert Path(winreg.QueryValueEx(key,'InstallLocation')[0]).resolve() == ROOT.resolve()
        assert 'Installation process succeeded.' in (OUTPUT/'obs-install.log').read_text(encoding='utf-8-sig')
    else:
        assert not ROOT.exists() and not registered(), '既存導入先・登録を保護するため中止'
        run([str(OUTPUT/'puyoZeus-Live-0.6.0-preview.10-setup.exe'),'/VERYSILENT','/SUPPRESSMSGBOXES',
             '/NORESTART','/SP-','/LANG=ja',f'/DIR={ROOT}',f'/LOG={OUTPUT / "obs-install.log"}'])
    result = {}
    try:
        assert registered()
        result['gui'] = gui_trial()
        result['synthetic'] = synthetic_trial()
    finally:
        try:
            run([str(ROOT/'unins000.exe'),'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART',
                 '/DELETEUSERDATA=1',f'/LOG={OUTPUT / "obs-uninstall.log"}'])
        except OSError as error:
            result['uninstall_error'] = dict(winerror=error.winerror, message=str(error))
            (OUTPUT/'obs-install-smoke.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            raise
        deadline = time.monotonic()+TIMEOUT
        while ROOT.exists() and time.monotonic()<deadline:
            time.sleep(POLL_SECONDS)
        result['uninstalled'] = not ROOT.exists() and not registered()
        (OUTPUT/'obs-install-smoke.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        assert result['uninstalled']
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
