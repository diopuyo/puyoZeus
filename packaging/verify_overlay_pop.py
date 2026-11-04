"""配布物の埋め込みPythonとHTMLを隔離起動し、表示更新を検証する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import zipfile

from playwright.sync_api import sync_playwright

OUTPUT = Path('D:/puyo_analyzer/local_frames/overlay_pop')
ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.6.0-preview.11'
ASSETS = ('overlay.html', 'overlay_options.js', 'overlay_position.js', 'overlay_layout.js',
    'overlay_mascot.js', 'overlay_themes.css', 'overlay_settings.html', 'overlay_settings.js',
    'overlay_i18n.js')
SERVER_CODE = '''import json
import src.phase_j.live_publish as module
server=module.LiveOverlayServer(module.LiveStreamState(),'127.0.0.1',0)
server.start()
print(json.dumps(dict(port=server.address()[1],module=module.__file__)),flush=True)
try: input()
finally: server.stop()
'''


def verify_archive(release: Path) -> None:
    """組立元とzipの両方が作業ツリーの最終HTMLと同一であることを確認する。"""
    with zipfile.ZipFile(release / f'puyoZeus-Live-{VERSION}.zip') as archive:
        for name in ASSETS:
            source = (ROOT / 'src/phase_j' / name).read_bytes()
            assert archive.read('PuyoLive/app/src/phase_j/'+name) == source, name
            assert (release / 'build/PuyoLive/app/src/phase_j' / name).read_bytes() == source, name
    manifest = json.loads((release / 'releases.json').read_text(encoding='utf-8'))
    assert manifest['version'] == VERSION
    for entry in manifest['files']:
        with (release / entry['name']).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']


def browser_check(port: int) -> None:
    """OBSやカメラを操作せず、配布HTTPの全組合せを確認する。"""
    with sync_playwright() as runner:
        browser = runner.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport=dict(width=1920, height=1080))
        errors: list[str] = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        origin = f'http://127.0.0.1:{port}'
        for theme in ('normal', 'esports', 'pop'):
            for layout in ('card', 'ribbon', 'minimal', 'l'):
                page.goto(f'{origin}/?preview=sample&theme={theme}&layout={layout}&x=40&y=400')
                assert page.locator('#stage').get_attribute('data-theme') == theme
                page.wait_for_function("document.querySelector('#mascot img').naturalWidth>0")
                if layout != 'l':
                    page.wait_for_function("document.getElementById('panel').offsetLeft===40 && document.getElementById('panel').offsetTop===400")
                    box = page.locator('#panel').bounding_box()
                    assert box['x'] == 40 and box['y'] == 400
        page.goto(origin+'/settings')
        assert page.locator('[name=theme] option').count() == 3
        assert page.locator('[name=layout] option').count() == 4
        assert page.request.get(origin+'/overlay_position.js').status == 200
        page.screenshot(path=str(OUTPUT / 'bundled_settings.png'))
        assert not errors, errors
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    verify_archive(args.release)
    bundle = args.release / 'build/PuyoLive'
    process = subprocess.Popen([str(bundle / 'python/python.exe'), '-c', SERVER_CODE],
        cwd=bundle / 'app', stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        metadata = json.loads(process.stdout.readline())
        assert Path(metadata['module']).resolve().is_relative_to(bundle.resolve())
        browser_check(metadata['port'])
    finally:
        process.communicate('\n', timeout=30)
    assert process.returncode == 0
    report = dict(version=VERSION, combinations=12, archive_matches=True,
        embedded_python=True, settings_choices=[3, 4], javascript_errors=0)
    (OUTPUT / 'bundle_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
