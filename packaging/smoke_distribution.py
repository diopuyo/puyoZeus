"""試験専用のインストール先で設定から実入力の試合待ちまで検証する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

from playwright.sync_api import Browser, sync_playwright

WAIT_SECONDS = 180
POLL_SECONDS = 0.05
TRIAL_PARENT = Path('D:/puyo_analyzer/b3_data/codex_tasks/dist_app_release')
OUTPUT = Path('D:/puyo_analyzer/local_frames/dist_app')


def get(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def wait_ready(url: str) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            get(url + '/api/state')
            return
        except OSError:
            time.sleep(POLL_SECONDS)
    raise TimeoutError('初期設定サーバーが起動しません')


def configure(browser: Browser, url: str) -> dict:
    page = browser.new_page(viewport=dict(width=1440, height=1080))
    page.goto(url + '/settings')
    page.locator('[name=language]').select_option('ja')
    for key in ('desktop-shortcut', 'start-menu-shortcut', 'autostart', 'check-updates'):
        page.locator('#' + key).uncheck()
    page.locator('[name=mascot]').check()
    page.screenshot(path=str(OUTPUT / 'initial_settings_ja.png'), full_page=True)
    page.locator('#save-app-settings').click()
    page.wait_for_function("document.getElementById('app-save-status').textContent.includes('保存しました')")
    page.locator('#toggle-runtime').click()
    page.goto(url + '/?language=ja')
    page.locator('#state').wait_for(state='visible')
    deadline = time.monotonic() + WAIT_SECONDS
    previous = None
    while time.monotonic() < deadline:
        status = page.locator('#state').inner_text()
        if status != previous:
            print('表示状態:', status, flush=True)
            previous = status
        if status == '試合待ち':
            page.screenshot(path=str(OUTPUT / 'installed_waiting_ja.png'))
            result = dict(selected_input=get(url + '/api/state')['diagnostics']['input'],
                          state=status, latest=get(url + '/latest'))
            page.close()
            return result
        time.sleep(POLL_SECONDS)
    page.screenshot(path=str(OUTPUT / 'installed_input_checking_ja.png'))
    (TRIAL_PARENT / 'installed-smoke-failure.json').write_text(
        json.dumps(get(url + '/latest'), ensure_ascii=False, indent=2), encoding='utf-8')
    raise TimeoutError('試合待ちまで到達しません: ' + str(previous))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18775)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.parent != TRIAL_PARENT.resolve() or not root.name.startswith('installed_trial'):
        raise ValueError('試験専用インストール先以外では実行しません')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    url = f'http://127.0.0.1:{args.port}'
    with (root / 'smoke-launch.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([str(root / 'python/python.exe'), '-m', 'src.phase_j.distribution_app',
            '--root', str(root), '--port', str(args.port), '--no-browser', '--setup'],
            cwd=root, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            wait_ready(url)
            with sync_playwright() as runner:
                browser = runner.chromium.launch(channel='msedge', headless=True)
                result = configure(browser, url)
                browser.close()
            (TRIAL_PARENT / 'installed-smoke.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print('実インストール先の初期設定→OBS入力テスト→試合待ち: OK', flush=True)
        finally:
            try:
                request = Request(url + '/api/stop', b'{}', {'Content-Type': 'application/json'})
                urlopen(request, timeout=10).close()
                child.wait(timeout=15)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=10)


if __name__ == '__main__':
    main()
