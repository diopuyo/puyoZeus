"""手動報告の実際の PS 5.1 出力と旧 pending の秘密を検証する。"""
import json
from pathlib import Path
import re
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.phase_j.distribution_logs import ReportStore, redact

ROOT = Path(__file__).resolve().parents[2]
SECRET_KEYS = ['password', 'passwd', 'obs_password', 'SECRET', 'access_token',
               'api_key', 'api-key', 'apiKey']


@pytest.mark.parametrize('key', SECRET_KEYS)
def test_redact_secret_keys(key: str) -> None:
    value = {key: {'nested': 'synthetic-sensitive'}, 'safe': [{key: 'sensitive'}]}
    result = redact(value, ('synthetic-user',))
    assert result == {key: '[redacted]', 'safe': [{key: '[redacted]'}]}
    assert value[key] == {'nested': 'synthetic-sensitive'}


def test_legacy_pending_preview_and_send(tmp_path: Path) -> None:
    store = ReportStore(tmp_path)
    old = {'report_id': '12345678-1234-1234-1234-123456789abc',
           'settings': {'obs_password': 'synthetic-secret', 'api_key': 'synthetic-key'}}
    (tmp_path / 'pending-old.json').write_text(json.dumps(old), encoding='utf-8-sig')
    preview = store.preview({'report_id': 'new'}, 'https://synthetic.test/api/reports')
    assert preview['payload']['reports'][0]['settings'] == {
        'obs_password': '[redacted]', 'api_key': '[redacted]'}
    response = MagicMock()
    response.status = 200
    response.read.return_value = b'{"accepted":true,"inserted":2,"duplicates":0}'
    opener = MagicMock()
    opener.open.return_value.__enter__.return_value = response
    with patch('src.phase_j.distribution_logs.build_opener', return_value=opener):
        assert store.send(preview['token'], True)
    sent = opener.open.call_args.args[0].data.decode('utf-8')
    assert 'synthetic-secret' not in sent and 'synthetic-key' not in sent
    assert json.loads(sent) == preview['payload']


@pytest.mark.parametrize('existing', [False, True])
def test_powershell_output(tmp_path: Path, existing: bool) -> None:
    script = ROOT / 'packaging/SendLogs.ps1'
    script.read_bytes().decode('ascii')
    (tmp_path / 'SendLogs.ps1').write_bytes(script.read_bytes())
    (tmp_path / 'app').mkdir()
    reports = tmp_path / 'logs/reports'
    reports.mkdir(parents=True)
    if existing:
        (reports / 'installation_id.txt').write_text('ABCDEF01-2345-6789-ABCD-EF0123456789')
    config = {'obs_password': 'synthetic-secret', 'relay_url': 'https://synthetic.invalid/api/reports',
              'releases_url': 'https://synthetic.invalid/releases.json', 'safe': True}
    (tmp_path / 'distribution.json').write_text(json.dumps(config), encoding='utf-8')
    wrapper = tmp_path / 'invoke.ps1'
    wrapper.write_text("function global:Read-Host { return 'SEND' }\n& \"$PSScriptRoot/SendLogs.ps1\"\n", encoding='ascii')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                             '-File', str(wrapper)], capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    files = list(reports.glob('pending-*.json'))
    assert len(files) == 1
    raw = files[0].read_bytes()
    assert not raw.startswith(b'\xef\xbb\xbf')
    report = json.loads(raw)
    assert re.fullmatch('[a-f0-9]{32}', report['report_id'])
    assert re.fullmatch('[a-f0-9]{32}', report['installation_id'])
    assert isinstance(report['created_at'], (int, float))
    assert report['settings'] == {'safe': True}
    assert b'synthetic-secret' not in raw
    if existing:
        assert report['installation_id'] == 'abcdef0123456789abcdef0123456789'


@pytest.mark.parametrize('enabled', [False, True])
def test_packaged_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    import build_release
    from src import production_config
    from src.phase_j import distribution_config
    monkeypatch.setattr(production_config, 'DISTRIBUTION_SITE_FEEDBACK', enabled)
    monkeypatch.setattr(distribution_config, 'DISTRIBUTION_SITE_URL', 'https://synthetic.test')
    build_release.write_sendlogs(tmp_path)
    text = (tmp_path / 'tools/SendLogs.ps1').read_bytes().decode('ascii')
    expected = 'https://synthetic.test/api/reports' if enabled else 'https://puyolive-reports.example.invalid/api/reports'
    assert "[string]$RelayUrl = '" + expected + "'" in text
    assert 'Source: "{#SourceDir}\\tools\\SendLogs.ps1"; Flags: dontcopy' in (ROOT / 'packaging/PuyoLive.iss').read_text(encoding='utf-8')


def test_pending_paths_and_invalid_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local_appdata'))
    script = ROOT / 'packaging/SendLogs.ps1'
    (tmp_path / 'SendLogs.ps1').write_bytes(script.read_bytes())
    (tmp_path / 'app').mkdir()
    reports = tmp_path / 'logs/reports'
    reports.mkdir(parents=True)
    report_id = 'x/../../../escaped-report'
    good = dict(schema_version=1, report_id=report_id, settings={})
    for name, value in [('escape', good), ('array', [good]), ('id', dict(report_id=3)),
                        ('settings', dict(schema_version=1, report_id='bad', settings=[]))]:
        (reports / f'pending-{name}.json').write_text(json.dumps(value), encoding='utf-8')
    wrapper = tmp_path / 'invoke.ps1'
    wrapper.write_text("function global:Read-Host { return 'SEND' }\n& \"$PSScriptRoot/SendLogs.ps1\"\n", encoding='ascii')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                             '-File', str(wrapper)], capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    target = reports / ('pending-' + hashlib.sha256(report_id.encode()).hexdigest() + '.json')
    assert json.loads(target.read_text(encoding='utf-8'))['report_id'] == report_id
    assert len(list(reports.glob('invalid-*.json'))) == 3
    assert not (tmp_path / 'escaped-report.json').exists()
    assert all(p.resolve().is_relative_to(reports.resolve()) for p in reports.glob('pending-*.json'))
