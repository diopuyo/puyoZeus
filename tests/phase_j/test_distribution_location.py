"""同期フォルダの境界と、停止しない注意表示を検証する。"""
from pathlib import Path
import json
import subprocess

import pytest
from playwright.sync_api import Browser, expect

from src.phase_j.distribution_location import is_onedrive_root
from tests.phase_j.test_distribution_app import distribution
from tests.phase_j.test_overlay_browser import browser


@pytest.mark.parametrize('root,environment,expected', [
    ('C:/Users/a/OneDrive/Desktop/PuyoLive', {}, True),
    ('C:/Users/a/OneDrive - Company/Documents/PuyoLive', {}, True),
    ('C:/Users/a/onedrive/PuyoLive', {}, True),
    ('D:/Sync/PuyoLive', {'OneDrive': 'D:/Sync'}, True),
    ('D:/Sync', {'OneDriveConsumer': 'd:/sync/'}, True),
    ('D:/Sync/PuyoLive', {'OneDriveCommercial': 'D:\\Sync'}, True),
    ('C:/puyoZeus', {'OneDrive': 'D:/Sync'}, False),
    ('D:/SyncBackup/PuyoLive', {'OneDrive': 'D:/Sync'}, False),
    ('C:/Users/a/OneDriveBackup/PuyoLive', {}, False),
    ('C:/Users/a/MyOneDrive/PuyoLive', {}, False),
    ('C:/Users/a/OneDrive-old/PuyoLive', {}, False),
    ('C:/puyoZeus', {'OneDrive': ''}, False),
])
def test_onedrive_boundary(root: str, environment: dict[str, str], expected: bool) -> None:
    assert is_onedrive_root(Path(root), environment) is expected


@pytest.mark.parametrize('language', ['ja', 'en'])
def test_onedrive_caution(browser: Browser, distribution: tuple, language: str) -> None:
    runtime, url = distribution
    runtime.settings.update(language=language, onboarding_seen=True)
    page = browser.new_page()
    page.goto(url + '/settings')
    page.wait_for_function('Boolean(window.lastDistributionState)')
    page.route('**/api/state', lambda route: route.abort())
    for onedrive in (True, False):
        state = dict(running=True, settings={}, diagnostics=dict(onedrive=onedrive,
                     calibration='ready', match_phase='playing'))
        page.evaluate('(state)=>{window.lastDistributionState=state;renderAppNotice();}', state)
        expect(page.locator('#runtime-state')).to_have_text('Analyzing' if language == 'en' else '解析中')
        if onedrive:
            expect(page.locator('#location-notice')).to_contain_text('Caution:' if language == 'en' else '注意:')
            expect(page.locator('#location-notice')).to_contain_text('C:\\puyoZeus')
            expect(page.locator('#location-notice')).to_be_visible()
        else:
            expect(page.locator('#location-notice')).to_be_hidden()
    page.close()


def test_runtime_location_diagnostic(distribution: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, _ = distribution
    monkeypatch.setenv('OneDrive', str(runtime.root))
    assert runtime.diagnostics()['onedrive'] is True
    monkeypatch.setenv('OneDrive', str(runtime.root / 'separate'))
    assert runtime.diagnostics()['onedrive'] is False


def test_release_documents(tmp_path: Path) -> None:
    from build_release import arrange_release_documents, write_sendlogs
    for name in ('README.md', 'README_distribution.md', 'SendLogs.ps1', 'UninstallCleanup.ps1',
                 'puyo_live.example.json', 'mascot_expressions_preview.png'):
        (tmp_path / name).write_text('old', encoding='utf-8')
    write_sendlogs(tmp_path)
    arrange_release_documents(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ['README_ja.txt']
    assert sorted(p.name for p in (tmp_path / 'tools').iterdir()) == [
        'SendLogs.ps1', 'mascot_expressions_preview.png', 'puyo_live.example.json']
    assert 'Extract' in (tmp_path / 'README_ja.txt').read_text('utf-8')


def test_sendlogs_in_tools(tmp_path: Path) -> None:
    from build_release import write_sendlogs
    write_sendlogs(tmp_path)
    (tmp_path / 'app').mkdir()
    (tmp_path / 'app/RELEASE.json').write_text('{"app":"test-version"}', encoding='utf-8')
    (tmp_path / 'distribution.json').write_text(json.dumps(dict(safe='from-root',
        relay_url='https://test.invalid/api/reports')), encoding='utf-8')
    (tmp_path / 'install.log').write_text('test-root-log', encoding='utf-8')
    (tmp_path / 'SendLogs.ps1').write_text("throw 'old script must not run'", encoding='ascii')
    wrapper = tmp_path / 'invoke.ps1'
    wrapper.write_text('function global:Read-Host { return "SEND" }\n'
                       '& "$PSScriptRoot/tools/SendLogs.ps1"\n', encoding='ascii')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                             '-File', str(wrapper)], capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    files = list((tmp_path / 'logs/reports').glob('pending-*.json'))
    assert len(files) == 1
    report = json.loads(files[0].read_bytes())
    assert report['settings'] == {'safe': 'from-root'}
    assert 'test-version' in files[0].read_text('utf-8')
    assert 'test-root-log' in files[0].read_text('utf-8')
